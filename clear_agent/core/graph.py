"""StateGraph - 声明式状态图

ClearAgent 的核心执行抽象，把 1.x 的硬编码 while 循环替换为可组合、
可恢复、可中断的图执行模型。

核心概念：
- State：用户定义的 TypedDict / pydantic / dataclass，字段级 reducer 合并
- Node：纯函数 (state) -> partial_state，支持同步/异步
- Edge：静态边 add_edge / 条件边 add_conditional_edges
- START / END：内置常量节点
- Reducer：字段合并策略（replace/add_messages/merge_dict/自定义）


"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from .checkpoint import BaseCheckpointer, Checkpoint, _uuid7
from .exceptions import ClearAgentException
from .interrupt import (
    GraphInterrupt,
    GraphPaused,
    _RunContext,
    _set_run_ctx,
    _reset_run_ctx,
    _get_run_ctx as _current_run_ctx_get,
)

# ==================== 常量 ====================

START = "__start__"
END = "__end__"

S = TypeVar("S")  # State type variable

NodeReturn = Union[Mapping[str, Any], None]
SyncNodeFn = Callable[[Any], NodeReturn]
AsyncNodeFn = Callable[[Any], Awaitable[NodeReturn]]
NodeFn = Union[SyncNodeFn, AsyncNodeFn]

RouterReturn = Union[str, List[str]]
SyncRouterFn = Callable[[Any], RouterReturn]
AsyncRouterFn = Callable[[Any], Awaitable[RouterReturn]]
RouterFn = Union[SyncRouterFn, AsyncRouterFn]


# ==================== 异常 ====================


class GraphError(ClearAgentException):
    """图相关错误的基类"""


class GraphRecursionError(GraphError):
    """超过 max_steps 或 recursion_limit"""


class GraphCompileError(GraphError):
    """编译期错误（不存在的节点、未连通等）"""


# ==================== Reducers ====================


def replace(_old: Any, new: Any) -> Any:
    """默认 reducer：覆盖"""
    return new


def add_messages(old: Optional[List[Any]], new: Any) -> List[Any]:
    """消息列表追加 + 按 id 去重

    new 可以是单个 Message、Message 列表，或 None。
    去重规则：如果 new 中的 message 有 id 字段且与 old 中重复，覆盖 old 对应项。
    """
    if old is None:
        old = []
    if new is None:
        return old
    if not isinstance(new, list):
        new = [new]

    # 构建 id -> index 索引（仅对有 id 的）
    out = list(old)
    id_to_idx: Dict[Any, int] = {}
    for i, m in enumerate(out):
        mid = _msg_id(m)
        if mid is not None:
            id_to_idx[mid] = i

    for m in new:
        mid = _msg_id(m)
        if mid is not None and mid in id_to_idx:
            out[id_to_idx[mid]] = m
        else:
            out.append(m)
            if mid is not None:
                id_to_idx[mid] = len(out) - 1
    return out


def _msg_id(m: Any) -> Optional[Any]:
    """获取 message 的 id 字段（兼容 dict / dataclass / pydantic）"""
    if isinstance(m, dict):
        return m.get("id")
    return getattr(m, "id", None)


def merge_dict(old: Optional[Dict], new: Optional[Dict]) -> Dict:
    """字典浅合并 reducer"""
    out = dict(old or {})
    if new:
        out.update(new)
    return out


def append_list(old: Optional[List], new: Any) -> List:
    """列表无脑追加 reducer（不去重）"""
    out = list(old or [])
    if new is None:
        return out
    if isinstance(new, list):
        out.extend(new)
    else:
        out.append(new)
    return out


# ==================== State Schema 解析 ====================


def _extract_reducers(schema: Type) -> Dict[str, Callable]:
    """从 TypedDict / dataclass / pydantic 中提取字段的 reducer

    通过 typing.Annotated[T, reducer_fn] 标注：
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            count: int  # 默认 replace

    未标注 reducer 的字段使用 replace。
    """
    reducers: Dict[str, Callable] = {}
    if schema is None or schema is dict:
        return reducers

    try:
        hints = get_type_hints(schema, include_extras=True)
    except Exception:
        return reducers

    for field_name, hint in hints.items():
        # 检查 typing.Annotated 元数据
        meta = getattr(hint, "__metadata__", None)
        if meta:
            for m in meta:
                if callable(m):
                    reducers[field_name] = m
                    break
    return reducers


# ==================== 图定义 ====================


@dataclass
class _Edge:
    source: str
    target: str


@dataclass
class _ConditionalEdge:
    source: str
    router: RouterFn
    mapping: Optional[Dict[str, Union[str, List[str]]]]


@dataclass
class RunConfig:
    """单次执行的配置

    Attributes:
        thread_id: 会话 ID（resume 时用）；不传则随机
        checkpoint_id: resume 时指定从哪个 checkpoint 开始；None 取最新
        max_steps: 单次执行最多经过多少节点（防死循环）
        recursion_limit: 同一节点最多重入次数
        on_error: "raise" | "record_and_continue"
        callbacks: 节点级钩子（暂未启用，按需接 Callbacks 协议）
    """

    thread_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    max_steps: int = 50
    recursion_limit: int = 25
    on_error: str = "raise"
    callbacks: Optional[List[Callable]] = None

    def with_thread_id(self) -> str:
        if self.thread_id is None:
            self.thread_id = f"t-{_uuid7()}"
        return self.thread_id


@dataclass
class StreamEvent:
    """图执行流式事件（graph 内部专用，与 streaming.StreamEvent 兼容）

    type: "node_start" | "node_finish" | "edge" | "checkpoint" | "error" | "end"
    """

    type: str
    node: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ==================== StateGraph ====================


class StateGraph(Generic[S]):
    """声明式状态图

    使用方式：
        class MyState(TypedDict):
            messages: Annotated[list, add_messages]
            count: int

        g = StateGraph(MyState)
        g.add_node("greet", greet_fn)
        g.add_edge(START, "greet")
        g.add_conditional_edges("greet", router, {"more": "greet", "done": END})
        compiled = g.compile()
    """

    def __init__(self, state_schema: Optional[Type[S]] = None) -> None:
        self.state_schema = state_schema
        self._nodes: Dict[str, NodeFn] = {}
        self._edges: List[_Edge] = []
        self._conditional_edges: List[_ConditionalEdge] = []
        self._reducers: Dict[str, Callable] = _extract_reducers(state_schema)

    # ---------- 构建 API ----------

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph[S]":
        if name in (START, END):
            raise GraphCompileError(f"节点名 {name} 是保留字")
        if name in self._nodes:
            raise GraphCompileError(f"节点 {name} 已存在")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph[S]":
        self._edges.append(_Edge(source=source, target=target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        mapping: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> "StateGraph[S]":
        self._conditional_edges.append(
            _ConditionalEdge(source=source, router=router, mapping=mapping)
        )
        return self

    def set_reducer(self, field_name: str, reducer: Callable) -> "StateGraph[S]":
        """显式设置字段 reducer（覆盖 Annotated 元数据）"""
        self._reducers[field_name] = reducer
        return self

    # ---------- 编译 ----------

    def compile(
        self, checkpointer: Optional[BaseCheckpointer] = None
    ) -> "CompiledGraph[S]":
        self._validate()
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=list(self._edges),
            conditional_edges=list(self._conditional_edges),
            reducers=dict(self._reducers),
            checkpointer=checkpointer,
            state_schema=self.state_schema,
        )

    def _validate(self) -> None:
        """编译期校验：所有引用的节点都已定义、START 必须有出边"""
        all_node_names = set(self._nodes.keys()) | {START, END}

        # 校验 add_edge 的两端
        for e in self._edges:
            if e.source not in all_node_names:
                raise GraphCompileError(f"add_edge 引用了未知节点: {e.source}")
            if e.target not in all_node_names:
                raise GraphCompileError(f"add_edge 引用了未知节点: {e.target}")

        # 校验 conditional edges 的 source
        for ce in self._conditional_edges:
            if ce.source not in all_node_names:
                raise GraphCompileError(
                    f"add_conditional_edges 引用了未知节点: {ce.source}"
                )
            if ce.mapping:
                for v in ce.mapping.values():
                    targets = v if isinstance(v, list) else [v]
                    for t in targets:
                        if t not in all_node_names:
                            raise GraphCompileError(
                                f"conditional mapping 引用了未知节点: {t}"
                            )

        # START 必须至少有一条出边
        has_start_edge = any(e.source == START for e in self._edges) or any(
            ce.source == START for ce in self._conditional_edges
        )
        if not has_start_edge:
            raise GraphCompileError("START 必须至少连接一个节点")


# ==================== CompiledGraph ====================


class CompiledGraph(Generic[S]):
    """编译后的可执行图

    职责：
    - 维护 nodes/edges/reducers 的不可变副本
    - 提供 invoke / ainvoke / stream / astream / resume 接口
    - 在每个节点后调 checkpointer（若提供）
    - 触发流式事件
    """

    def __init__(
        self,
        nodes: Dict[str, NodeFn],
        edges: List[_Edge],
        conditional_edges: List[_ConditionalEdge],
        reducers: Dict[str, Callable],
        checkpointer: Optional[BaseCheckpointer],
        state_schema: Optional[Type[S]],
    ) -> None:
        self._nodes = nodes
        self._reducers = reducers
        self.checkpointer = checkpointer
        self.state_schema = state_schema

        # 索引：source -> 静态后继 / 条件路由
        self._static_next: Dict[str, str] = {e.source: e.target for e in edges}
        self._cond_next: Dict[str, _ConditionalEdge] = {
            ce.source: ce for ce in conditional_edges
        }

    # ---------- 同步接口 ----------

    def invoke(
        self, input: Mapping[str, Any], config: Optional[RunConfig] = None
    ) -> Dict[str, Any]:
        """同步执行图，返回最终 state"""
        cfg = config or RunConfig()
        cfg.with_thread_id()
        state = self._normalize_input(input)
        return self._with_run_ctx_sync(
            cfg.thread_id,
            None,
            False,
            lambda: self._run_loop_sync(state, start_node=START, config=cfg),
        )

    def stream(
        self, input: Mapping[str, Any], config: Optional[RunConfig] = None
    ) -> Iterator[StreamEvent]:
        """同步流式执行，yield StreamEvent"""
        cfg = config or RunConfig()
        cfg.with_thread_id()
        state = self._normalize_input(input)
        ctx = _RunContext(thread_id=cfg.thread_id)
        token = _set_run_ctx(ctx)
        try:
            yield from self._stream_loop_sync(state, start_node=START, config=cfg)
        finally:
            _reset_run_ctx(token)

    # ---------- 异步接口 ----------

    async def ainvoke(
        self, input: Mapping[str, Any], config: Optional[RunConfig] = None
    ) -> Dict[str, Any]:
        cfg = config or RunConfig()
        cfg.with_thread_id()
        state = self._normalize_input(input)
        ctx = _RunContext(thread_id=cfg.thread_id)
        token = _set_run_ctx(ctx)
        try:
            return await self._run_loop_async(state, start_node=START, config=cfg)
        finally:
            _reset_run_ctx(token)

    async def astream(
        self, input: Mapping[str, Any], config: Optional[RunConfig] = None
    ) -> AsyncIterator[StreamEvent]:
        cfg = config or RunConfig()
        cfg.with_thread_id()
        state = self._normalize_input(input)
        ctx = _RunContext(thread_id=cfg.thread_id)
        token = _set_run_ctx(ctx)
        try:
            async for ev in self._stream_loop_async(state, start_node=START, config=cfg):
                yield ev
        finally:
            _reset_run_ctx(token)

    # ---------- 内部 helper ----------

    def _with_run_ctx_sync(
        self,
        thread_id: Optional[str],
        resume_value: Any,
        has_resume_value: bool,
        fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """同步路径下设置 RunContext 跑 fn"""
        ctx = _RunContext(
            thread_id=thread_id,
            live_value=resume_value,
            has_live_value=has_resume_value,
        )
        token = _set_run_ctx(ctx)
        try:
            return fn()
        finally:
            _reset_run_ctx(token)

    # ---------- Resume / Checkpoint 操作 ----------

    def resume(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        state_patch: Optional[Mapping[str, Any]] = None,
        value: Any = None,
    ) -> Dict[str, Any]:
        """从 checkpoint 恢复执行（同步）

        Args:
            thread_id: 会话 ID
            checkpoint_id: 指定从哪个 checkpoint 续跑；None 取最新
            state_patch: 续跑前对 state 的字段级修改（按 reducer 合并）
            value: 中断（HITL）回执；当被恢复的 checkpoint
                ``metadata.source == "interrupt"`` 时，节点重入时
                ``interrupt()`` 调用会返回此 value 而非再次抛出

        Returns:
            最终 state
        """
        if self.checkpointer is None:
            raise GraphError("resume 需要 checkpointer")
        ckpt = self.checkpointer.get_tuple(thread_id, checkpoint_id)
        if ckpt is None:
            raise GraphError(f"thread {thread_id} 无可恢复 checkpoint")
        state = dict(ckpt.state)
        if state_patch:
            state = self._merge(state, dict(state_patch))
        next_node = ckpt.next_nodes[0] if ckpt.next_nodes else END

        is_interrupt_resume = ckpt.metadata.get("source") == "interrupt"
        history = (
            list(ckpt.metadata.get("resume_values") or [])
            if is_interrupt_resume
            else []
        )

        cfg = RunConfig(thread_id=thread_id)
        ctx = _RunContext(
            thread_id=thread_id,
            resume_values=history,
            live_value=value if is_interrupt_resume else None,
            has_live_value=is_interrupt_resume,
        )
        token = _set_run_ctx(ctx)
        try:
            return self._run_loop_sync(state, start_node=next_node, config=cfg)
        finally:
            _reset_run_ctx(token)

    async def aresume(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        state_patch: Optional[Mapping[str, Any]] = None,
        value: Any = None,
    ) -> Dict[str, Any]:
        """从 checkpoint 恢复执行（异步）"""
        if self.checkpointer is None:
            raise GraphError("aresume 需要 checkpointer")
        ckpt = await self.checkpointer.aget_tuple(thread_id, checkpoint_id)
        if ckpt is None:
            raise GraphError(f"thread {thread_id} 无可恢复 checkpoint")
        state = dict(ckpt.state)
        if state_patch:
            state = self._merge(state, dict(state_patch))
        next_node = ckpt.next_nodes[0] if ckpt.next_nodes else END

        is_interrupt_resume = ckpt.metadata.get("source") == "interrupt"
        history = (
            list(ckpt.metadata.get("resume_values") or [])
            if is_interrupt_resume
            else []
        )
        cfg = RunConfig(thread_id=thread_id)

        ctx = _RunContext(
            thread_id=thread_id,
            resume_values=history,
            live_value=value if is_interrupt_resume else None,
            has_live_value=is_interrupt_resume,
        )
        token = _set_run_ctx(ctx)
        try:
            return await self._run_loop_async(state, start_node=next_node, config=cfg)
        finally:
            _reset_run_ctx(token)

    def list_checkpoints(self, thread_id: str, limit: int = 50) -> List[Checkpoint]:
        if self.checkpointer is None:
            return []
        return self.checkpointer.list(thread_id, limit=limit)

    def get_state(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if self.checkpointer is None:
            return None
        ckpt = self.checkpointer.get_tuple(thread_id, checkpoint_id)
        return dict(ckpt.state) if ckpt else None

    # ---------- Mermaid 可视化 ----------

    def draw_mermaid(self) -> str:
        """生成 mermaid flowchart 字符串"""
        lines = ["flowchart TD"]
        # 节点
        lines.append(f"    {START}([START])")
        lines.append(f"    {END}([END])")
        for name in self._nodes:
            lines.append(f"    {name}[{name}]")
        # 静态边
        for src, tgt in self._static_next.items():
            lines.append(f"    {src} --> {tgt}")
        # 条件边（用虚线）
        for src, ce in self._cond_next.items():
            if ce.mapping:
                for label, tgt in ce.mapping.items():
                    targets = tgt if isinstance(tgt, list) else [tgt]
                    for t in targets:
                        lines.append(f'    {src} -.{label}.-> {t}')
            else:
                lines.append(f'    {src} -.?.-> ???')
        return "\n".join(lines)

    # ==================== 内部执行 ====================

    def _normalize_input(self, input: Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(input, dict):
            return dict(input)
        # pydantic / dataclass 兼容
        if hasattr(input, "model_dump"):
            return input.model_dump()
        if hasattr(input, "__dict__"):
            return dict(input.__dict__)
        raise GraphError(f"无法将 input 转为 dict: {type(input)}")

    def _merge(self, state: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
        """按 reducer 合并 partial state 到 state"""
        out = dict(state)
        for k, v in update.items():
            reducer = self._reducers.get(k, replace)
            out[k] = reducer(out.get(k), v)
        return out

    def _route(self, state: Dict[str, Any], current: str) -> str:
        """决定 current 节点的下一节点（同步路径）

        条件边优先于静态边（如果同时存在，条件边胜出）。
        条件边 router 可能是 async，调用方需检查并走 async 路径。
        """
        if current in self._cond_next:
            ce = self._cond_next[current]
            # 这里强制走同步；async router 在 _route_async 处理
            if inspect.iscoroutinefunction(ce.router):
                raise GraphError(
                    f"节点 {current} 的 router 是 async，请使用 ainvoke/astream"
                )
            decision = ce.router(state)
            return self._resolve_decision(decision, ce.mapping)
        if current in self._static_next:
            return self._static_next[current]
        return END

    async def _route_async(self, state: Dict[str, Any], current: str) -> str:
        if current in self._cond_next:
            ce = self._cond_next[current]
            if inspect.iscoroutinefunction(ce.router):
                decision = await ce.router(state)
            else:
                decision = ce.router(state)
            return self._resolve_decision(decision, ce.mapping)
        if current in self._static_next:
            return self._static_next[current]
        return END

    def _resolve_decision(
        self,
        decision: RouterReturn,
        mapping: Optional[Dict[str, Union[str, List[str]]]],
    ) -> str:
        """根据 router 决策与 mapping 解析为目标节点"""
        if isinstance(decision, list):
            # 并行多分支：本期选第一个，告警 todo
            # （实现真正的并行执行；本期先支持单分支返回）
            decision = decision[0] if decision else END
        if mapping:
            target = mapping.get(decision)
            if target is None:
                raise GraphError(
                    f"router 返回 '{decision}' 但 mapping 中无对应项"
                )
            if isinstance(target, list):
                target = target[0] if target else END
            return target
        return decision

    def _call_node_sync(self, name: str, state: Dict[str, Any]) -> NodeReturn:
        fn = self._nodes[name]
        if inspect.iscoroutinefunction(fn):
            raise GraphError(f"节点 {name} 是 async，请使用 ainvoke/astream")
        # 进入新节点前重置 interrupt 计数器
        ctx = _current_run_ctx_get()
        if ctx is not None:
            ctx.reset_counter()
        return fn(state)

    async def _call_node_async(self, name: str, state: Dict[str, Any]) -> NodeReturn:
        fn = self._nodes[name]
        ctx = _current_run_ctx_get()
        if ctx is not None:
            ctx.reset_counter()
        if inspect.iscoroutinefunction(fn):
            return await fn(state)
        # 同步 fn 在 ainvoke 路径下放线程池
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, state)

    def _write_checkpoint(
        self,
        state: Dict[str, Any],
        next_node: str,
        thread_id: str,
        parent_id: Optional[str],
        node_just_done: str,
        source: str = "loop",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Checkpoint]:
        if self.checkpointer is None:
            return None
        metadata: Dict[str, Any] = {"source": source, "node": node_just_done}
        if extra_metadata:
            metadata.update(extra_metadata)
        ckpt = Checkpoint(
            id=_uuid7(),
            thread_id=thread_id,
            parent_id=parent_id,
            state=state,
            next_nodes=[next_node],
            metadata=metadata,
        )
        self.checkpointer.put(ckpt)
        return ckpt

    # ---------- 主循环（同步） ----------

    def _run_loop_sync(
        self,
        state: Dict[str, Any],
        start_node: str,
        config: RunConfig,
    ) -> Dict[str, Any]:
        current = start_node
        steps = 0
        node_visit_count: Dict[str, int] = {}
        last_ckpt_id: Optional[str] = None

        # 处理起点：START 直接路由到第一个真节点
        if current == START:
            current = self._route(state, START)

        while current != END:
            if steps >= config.max_steps:
                raise GraphRecursionError(
                    f"超过 max_steps={config.max_steps}（当前节点 {current}）"
                )
            node_visit_count[current] = node_visit_count.get(current, 0) + 1
            if node_visit_count[current] > config.recursion_limit:
                raise GraphRecursionError(
                    f"节点 {current} 重入超过 recursion_limit={config.recursion_limit}"
                )

            try:
                update = self._call_node_sync(current, state)
            except GraphInterrupt as gi:
                # HITL 中断：写 source=interrupt ckpt，抛 GraphPaused
                ckpt = self._write_checkpoint(
                    state,
                    next_node=current,
                    thread_id=config.thread_id or "",
                    parent_id=last_ckpt_id,
                    node_just_done=current,
                    source="interrupt",
                    extra_metadata={"payload": gi.payload, "resume_values": list((_current_run_ctx_get() or _RunContext()).resume_values)},
                )
                ckpt_id = ckpt.id if ckpt else ""
                raise GraphPaused(
                    thread_id=config.thread_id or "",
                    checkpoint_id=ckpt_id,
                    payload=gi.payload,
                ) from None
            except Exception as e:
                if config.on_error == "raise":
                    raise
                # record_and_continue：记录到 state，跳到 END
                state = self._merge(state, {"__error__": str(e), "__error_node__": current})
                self._write_checkpoint(
                    state, END, config.thread_id or "", last_ckpt_id, current, source="error"
                )
                return state

            if update:
                state = self._merge(state, update)

            next_node = self._route(state, current)

            ckpt = self._write_checkpoint(
                state, next_node, config.thread_id or "", last_ckpt_id, current
            )
            if ckpt:
                last_ckpt_id = ckpt.id

            steps += 1
            current = next_node

        return state

    # ---------- 主循环（异步） ----------

    async def _run_loop_async(
        self,
        state: Dict[str, Any],
        start_node: str,
        config: RunConfig,
    ) -> Dict[str, Any]:
        current = start_node
        steps = 0
        node_visit_count: Dict[str, int] = {}
        last_ckpt_id: Optional[str] = None

        if current == START:
            current = await self._route_async(state, START)

        while current != END:
            if steps >= config.max_steps:
                raise GraphRecursionError(
                    f"超过 max_steps={config.max_steps}（当前节点 {current}）"
                )
            node_visit_count[current] = node_visit_count.get(current, 0) + 1
            if node_visit_count[current] > config.recursion_limit:
                raise GraphRecursionError(
                    f"节点 {current} 重入超过 recursion_limit={config.recursion_limit}"
                )

            try:
                update = await self._call_node_async(current, state)
            except GraphInterrupt as gi:
                ckpt = self._write_checkpoint(
                    state,
                    next_node=current,
                    thread_id=config.thread_id or "",
                    parent_id=last_ckpt_id,
                    node_just_done=current,
                    source="interrupt",
                    extra_metadata={"payload": gi.payload, "resume_values": list((_current_run_ctx_get() or _RunContext()).resume_values)},
                )
                ckpt_id = ckpt.id if ckpt else ""
                raise GraphPaused(
                    thread_id=config.thread_id or "",
                    checkpoint_id=ckpt_id,
                    payload=gi.payload,
                ) from None
            except Exception as e:
                if config.on_error == "raise":
                    raise
                state = self._merge(state, {"__error__": str(e), "__error_node__": current})
                self._write_checkpoint(
                    state, END, config.thread_id or "", last_ckpt_id, current, source="error"
                )
                return state

            if update:
                state = self._merge(state, update)

            next_node = await self._route_async(state, current)

            ckpt = self._write_checkpoint(
                state, next_node, config.thread_id or "", last_ckpt_id, current
            )
            if ckpt:
                last_ckpt_id = ckpt.id

            steps += 1
            current = next_node

        return state

    # ---------- 流式（同步） ----------

    def _stream_loop_sync(
        self,
        state: Dict[str, Any],
        start_node: str,
        config: RunConfig,
    ) -> Iterator[StreamEvent]:
        current = start_node
        steps = 0
        node_visit_count: Dict[str, int] = {}
        last_ckpt_id: Optional[str] = None

        if current == START:
            current = self._route(state, START)
            yield StreamEvent(type="edge", node=START, data={"next": current})

        while current != END:
            if steps >= config.max_steps:
                yield StreamEvent(
                    type="error", node=current, data={"reason": "max_steps_exceeded"}
                )
                raise GraphRecursionError(
                    f"超过 max_steps={config.max_steps}（当前节点 {current}）"
                )
            node_visit_count[current] = node_visit_count.get(current, 0) + 1
            if node_visit_count[current] > config.recursion_limit:
                raise GraphRecursionError(
                    f"节点 {current} 重入超过 recursion_limit={config.recursion_limit}"
                )

            yield StreamEvent(type="node_start", node=current, state=dict(state))

            try:
                update = self._call_node_sync(current, state)
            except GraphInterrupt as gi:
                ckpt = self._write_checkpoint(
                    state,
                    next_node=current,
                    thread_id=config.thread_id or "",
                    parent_id=last_ckpt_id,
                    node_just_done=current,
                    source="interrupt",
                    extra_metadata={"payload": gi.payload, "resume_values": list((_current_run_ctx_get() or _RunContext()).resume_values)},
                )
                yield StreamEvent(
                    type="interrupt",
                    node=current,
                    state=dict(state),
                    data={
                        "payload": gi.payload,
                        "checkpoint_id": ckpt.id if ckpt else None,
                        "thread_id": config.thread_id,
                    },
                )
                raise GraphPaused(
                    thread_id=config.thread_id or "",
                    checkpoint_id=ckpt.id if ckpt else "",
                    payload=gi.payload,
                ) from None
            except Exception as e:
                yield StreamEvent(
                    type="error", node=current, data={"error": str(e), "type": type(e).__name__}
                )
                if config.on_error == "raise":
                    raise
                state = self._merge(
                    state, {"__error__": str(e), "__error_node__": current}
                )
                yield StreamEvent(type="end", state=dict(state))
                return

            if update:
                state = self._merge(state, update)

            yield StreamEvent(type="node_finish", node=current, state=dict(state))

            next_node = self._route(state, current)
            yield StreamEvent(type="edge", node=current, data={"next": next_node})

            ckpt = self._write_checkpoint(
                state, next_node, config.thread_id or "", last_ckpt_id, current
            )
            if ckpt:
                last_ckpt_id = ckpt.id
                yield StreamEvent(
                    type="checkpoint",
                    node=current,
                    data={"checkpoint_id": ckpt.id, "thread_id": config.thread_id},
                )

            steps += 1
            current = next_node

        yield StreamEvent(type="end", state=dict(state))

    # ---------- 流式（异步） ----------

    async def _stream_loop_async(
        self,
        state: Dict[str, Any],
        start_node: str,
        config: RunConfig,
    ) -> AsyncIterator[StreamEvent]:
        current = start_node
        steps = 0
        node_visit_count: Dict[str, int] = {}
        last_ckpt_id: Optional[str] = None

        if current == START:
            current = await self._route_async(state, START)
            yield StreamEvent(type="edge", node=START, data={"next": current})

        while current != END:
            if steps >= config.max_steps:
                yield StreamEvent(
                    type="error", node=current, data={"reason": "max_steps_exceeded"}
                )
                raise GraphRecursionError(
                    f"超过 max_steps={config.max_steps}（当前节点 {current}）"
                )
            node_visit_count[current] = node_visit_count.get(current, 0) + 1
            if node_visit_count[current] > config.recursion_limit:
                raise GraphRecursionError(
                    f"节点 {current} 重入超过 recursion_limit={config.recursion_limit}"
                )

            yield StreamEvent(type="node_start", node=current, state=dict(state))

            try:
                update = await self._call_node_async(current, state)
            except GraphInterrupt as gi:
                ckpt = self._write_checkpoint(
                    state,
                    next_node=current,
                    thread_id=config.thread_id or "",
                    parent_id=last_ckpt_id,
                    node_just_done=current,
                    source="interrupt",
                    extra_metadata={"payload": gi.payload, "resume_values": list((_current_run_ctx_get() or _RunContext()).resume_values)},
                )
                yield StreamEvent(
                    type="interrupt",
                    node=current,
                    state=dict(state),
                    data={
                        "payload": gi.payload,
                        "checkpoint_id": ckpt.id if ckpt else None,
                        "thread_id": config.thread_id,
                    },
                )
                raise GraphPaused(
                    thread_id=config.thread_id or "",
                    checkpoint_id=ckpt.id if ckpt else "",
                    payload=gi.payload,
                ) from None
            except Exception as e:
                yield StreamEvent(
                    type="error", node=current, data={"error": str(e), "type": type(e).__name__}
                )
                if config.on_error == "raise":
                    raise
                state = self._merge(
                    state, {"__error__": str(e), "__error_node__": current}
                )
                yield StreamEvent(type="end", state=dict(state))
                return

            if update:
                state = self._merge(state, update)

            yield StreamEvent(type="node_finish", node=current, state=dict(state))

            next_node = await self._route_async(state, current)
            yield StreamEvent(type="edge", node=current, data={"next": next_node})

            ckpt = self._write_checkpoint(
                state, next_node, config.thread_id or "", last_ckpt_id, current
            )
            if ckpt:
                last_ckpt_id = ckpt.id
                yield StreamEvent(
                    type="checkpoint",
                    node=current,
                    data={"checkpoint_id": ckpt.id, "thread_id": config.thread_id},
                )

            steps += 1
            current = next_node

        yield StreamEvent(type="end", state=dict(state))


# ==================== 公开导出 ====================

__all__ = [
    # 常量
    "START",
    "END",
    # 核心类
    "StateGraph",
    "CompiledGraph",
    "RunConfig",
    "StreamEvent",
    # Reducers
    "replace",
    "add_messages",
    "merge_dict",
    "append_list",
    # 异常
    "GraphError",
    "GraphRecursionError",
    "GraphCompileError",
]
