# 01 · StateGraph 架构设计

> **阶段**：2.0-α / W1
> **目标文件**：`clear_agent/core/graph.py`、`clear_agent/core/state.py`
> **关联文档**：02-checkpoint-and-resume.md（per-step 持久化）、03-hitl-guide.md（interrupt 语义）

---

## 1. 设计目标

把 ClearAgent 1.0 的「`while current_step < max_steps` 单层循环」抽象成**声明式状态图**，让流程可组合、可恢复、可中断。

**核心要求**：
- 节点 / 边 / 条件路由声明式 API（参考 LangGraph 但不抄运行时）
- State 是 `TypedDict` 或 pydantic，字段级 reducer
- 同步 + 异步双轨支持（`invoke` / `ainvoke` / `stream`）
- 与现有 4 个 Agent 范式 100% 向后兼容（旧 API 内部走 graph）
- **不引入 BSP/Pregel 复杂度**：单进程同步执行即可，super-step 概念用「节点完成」近似

**非目标**（明确不做）：
- 分布式/集群图执行 → 不做
- 节点级 GPU 编排 → 不做
- Pregel 严格 BSP 语义 → 不做（够用即可）

---

## 2. 核心数据模型

### 2.1 State Schema

```python
from typing import TypedDict, Annotated
from clear_agent.core.graph import add_messages, replace

class ReActState(TypedDict):
    # add_messages: 列表追加 reducer（多次写入会 append）
    messages: Annotated[list[Message], add_messages]
    # replace: 默认覆盖
    current_step: Annotated[int, replace]
    total_tokens: Annotated[int, replace]
    # 自定义 reducer
    tool_calls_history: Annotated[list[dict], lambda old, new: old + new]
    # 中断/恢复用
    pending_interrupt: Annotated[Optional[dict], replace]
```

**Reducer 规则**：
- 字段未声明 reducer → `replace`（覆盖）
- `add_messages`：自动按 `id` 去重 + 追加（与 `Message.id` 字段配合）
- `merge_dict`：字典浅合并
- 自定义：`Annotated[T, callable]`，签名 `(old: T, new: T) -> T`

### 2.2 节点（Node）

```python
NodeFn = Callable[[State], State | dict | Awaitable[State | dict]]
```

- 节点是**纯函数**：接收当前 State，返回部分/完整 State 更新
- 支持同步与异步两种签名
- 返回 `dict` 时按字段并入 State（按 reducer 合并）
- 返回 `None` 等价于 `{}`（不修改 State）
- 节点内可调用 `interrupt(payload)` 触发 HITL（详见 03 文档）

**示例**：
```python
def llm_node(state: ReActState) -> dict:
    response = llm.invoke_with_tools(state["messages"], tools)
    return {
        "messages": [response.to_message()],
        "total_tokens": state["total_tokens"] + response.usage["total_tokens"],
    }
```

### 2.3 边（Edge）

```python
class StateGraph(Generic[S]):
    def add_edge(self, source: str, target: str) -> Self
    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[S], str | list[str]],
        mapping: dict[str, str] | None = None,
    ) -> Self
```

- **静态边**：`add_edge("a", "b")` —— a 完成后无条件去 b
- **条件边**：`router(state)` 返回字符串（节点名）或列表（并行多个节点）；可选 `mapping` 做 router 输出 → 节点名映射
- **特殊节点**：`START`、`END`（常量字符串）

**条件边示例**：
```python
def should_continue(state: ReActState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"

g.add_conditional_edges("llm", should_continue, {"tools": "tool_executor", "end": END})
```

### 2.4 编译产物（CompiledGraph）

```python
class CompiledGraph(Generic[S]):
    def invoke(self, input: dict, config: RunConfig | None = None) -> S: ...
    async def ainvoke(self, input: dict, config: RunConfig | None = None) -> S: ...
    def stream(self, input: dict, config: RunConfig | None = None) -> Iterator[StreamEvent]: ...
    async def astream(self, input: dict, config: RunConfig | None = None) -> AsyncIterator[StreamEvent]: ...
    def resume(self, thread_id: str, value: Any | None = None) -> S: ...
    def get_state(self, thread_id: str, checkpoint_id: str | None = None) -> StateSnapshot: ...
```

```python
@dataclass
class RunConfig:
    thread_id: str | None = None         # checkpoint 隔离
    checkpoint_id: str | None = None     # 从特定 checkpoint resume
    max_steps: int = 50                  # 防死循环
    recursion_limit: int = 25            # 单节点最大重入
    callbacks: list[Callable] | None = None
```

---

## 3. 执行模型

### 3.1 单步语义

```
1. 选择当前活跃节点（START 时即第一个）
2. 调用 node_fn(state) → partial_update
3. 按 reducer 合并 partial_update 到 state
4. checkpointer.put(thread_id, snapshot)   # 见 02
5. 决定下一节点：
   - 静态边：直接路由
   - 条件边：调用 router(state) 决定
6. 触发 lifecycle 事件 STEP_START / STEP_FINISH（复用 lifecycle.EventType）
7. 重复直到 END 或达 max_steps
```

### 3.2 并行（最小可用）

- 条件路由 router 返回 `list[str]` 时并行执行多个分支
- 用 `asyncio.gather` 等待所有分支完成后回到下一个汇聚节点
- **限制**：本期不实现 LangGraph 的 super-step BSP；并行节点对 State 的写入按声明顺序合并（后写覆盖前写，除非 reducer 是 append/merge）
- 限流：`max_concurrent_nodes`（沿用 `Config.max_concurrent_tools` 字段语义）

### 3.3 错误处理

- 节点抛异常 → 捕获后写入 checkpoint（含异常 traceback）
- `RunConfig.on_error` 三选一：`raise`（默认） / `record_and_continue` / `route_to_node`
- 复用 `lifecycle.EventType.AGENT_ERROR`

---

## 4. 与现有 Agent 的对接

### 4.1 旧入口完全保留

```python
# 1.x 用法继续工作
agent = ReActAgent(name="x", llm=llm, tool_registry=registry)
result = agent.run("hello")
```

内部委托：

```python
class ReActAgent(Agent):
    def run(self, input_text: str, **kwargs) -> str:
        graph = build_react_graph(self.llm, self.tool_registry, self.config)
        result = graph.invoke({"messages": [Message(input_text, "user")]})
        return result["messages"][-1].content
```

### 4.2 新入口（推荐）

```python
from clear_agent.graph import build_react_graph

graph = build_react_graph(llm, tool_registry, config)
result = graph.invoke(
    {"messages": [Message("hello", "user")]},
    config=RunConfig(thread_id="user-42"),
)
```

### 4.3 4 个范式各自的 `build_*_graph`

| 函数 | 节点结构 | 文件 |
|---|---|---|
| `build_simple_graph()` | START → llm → tools? → END（条件回环） | `agents/simple_agent.py` |
| `build_react_graph()` | START → llm → router(tools/end) → tool_executor → llm | `agents/react_agent.py` |
| `build_reflection_graph()` | START → generate → reflect → revise → END | `agents/reflection_agent.py` |
| `build_plan_solve_graph()` | START → plan → execute_step (loop) → finalize → END | `agents/plan_solve_agent.py` |

---

## 5. 与现有资产的复用

| 资产 | 用法 |
|---|---|
| `lifecycle.EventType` | 扩展 `NODE_START / NODE_FINISH / EDGE_TRAVERSED`，不重新定义 |
| `streaming.StreamEvent` | graph stream 直接生产 StreamEvent，复用 `to_sse()` |
| `pyproject.toml:networkx` | DAG 结构校验 + `graph.draw_mermaid()` 输出 |
| `Config` | 新增 `graph_max_steps`、`graph_recursion_limit`、`graph_max_concurrent_nodes` |
| `TraceLogger` | 每个节点完成后 `log_event("node_finish", {...})` |
| `Message` | 作为 `messages` 字段的元素类型，配合 `add_messages` reducer |

---

## 6. API 速览（最终对外签名）

```python
from clear_agent.core.graph import StateGraph, START, END, RunConfig
from clear_agent.core.graph import add_messages, replace, merge_dict

# 1. 定义 State
class MyState(TypedDict):
    messages: Annotated[list[Message], add_messages]
    counter: int

# 2. 构建图
g = StateGraph(MyState)
g.add_node("greet", greet_fn)
g.add_node("count", count_fn)
g.add_edge(START, "greet")
g.add_edge("greet", "count")
g.add_conditional_edges("count", router, {"more": "greet", "done": END})

# 3. 编译
compiled = g.compile(checkpointer=InMemoryCheckpointer())

# 4. 执行
result = compiled.invoke({"messages": [], "counter": 0})

# 5. 流式
for event in compiled.stream({"messages": [], "counter": 0}):
    print(event)

# 6. 可视化
print(compiled.draw_mermaid())
```

---

## 7. 测试清单（W1 出口）

`tests/test_graph_basics.py`：

| # | 测试 | 通过标准 |
|---|---|---|
| 1 | 线性图执行顺序 | START → A → B → END，节点函数按序触发 |
| 2 | 条件分支 | router 返回不同值时走不同分支 |
| 3 | 循环终止 | router 死循环 + max_steps=10 → 抛 `GraphRecursionError` |
| 4 | reducer 合并 | `add_messages` 追加去重；自定义 reducer 调用正确 |
| 5 | 同步 / 异步等价 | `invoke` 与 `ainvoke` 同输入返回相同 state |
| 6 | 流式事件顺序 | `stream` 产出的事件类型序列符合 NODE_START → NODE_FINISH |
| 7 | mermaid 输出 | `draw_mermaid()` 生成有效 mermaid 语法 |
| 8 | 错误传播 | 节点抛异常时 RunConfig.on_error="raise" 抛出 |

---

## 8. 不在 W1 做的事（推迟）

| 项 | 推迟到 |
|---|---|
| Subgraph 嵌套（`graph_a.add_node("sub", graph_b)`） | 2.0-β |
| 时间旅行 UI（基于 checkpoint 树的状态浏览器） | 2.0-RC |
| 节点级 retry / fallback（`with_retry`） | 2.0-β |
| BSP 严格并行（多 super-step） | 不做 |

---

## 9. 待决问题（开 W1 前需要 review 决议）

1. **State 类型用 TypedDict 还是 pydantic BaseModel？**
   - 推荐 TypedDict（与 LangGraph 习惯一致、零序列化开销）
   - 但 pydantic 在 checkpoint 序列化时更友好
   - **建议**：两种都支持，TypedDict 优先

2. **`add_messages` 是否做去重？**
   - 推荐做（按 `Message.id` 字段，1.x 已有该字段）
   - 否则多节点写消息会出现重复

3. **同步节点 vs 异步节点能否混用？**
   - 推荐能（异步图遇到同步节点时 `run_in_executor` 包装；同步图遇到异步节点抛错并提示用户用 `ainvoke`）

4. **`max_steps` 默认值**
   - 1.x 是 5（ReAct）/ 15（subagent）
   - graph 默认建议 50（足够大但不至于死循环消耗到爆）

请在每个待决项标注你的选择，否则我按上述「建议」执行。
