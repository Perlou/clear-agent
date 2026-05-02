# 03 · Human-in-the-Loop 设计

> **阶段**：2.0-α / W3
> **目标文件**：`clear_agent/core/interrupt.py`、`clear_agent/hitl/patterns.py`
> **关联文档**：02（中断时写 checkpoint）、01（节点内调 `interrupt()`）

---

## 1. 设计目标

让 agent 流程**可暂停、可审批、可编辑、可续跑**。
覆盖三类典型场景：
- **审批**：高风险工具调用前需人工 approve（发邮件、删数据、转账）
- **编辑**：LLM 草稿不满意，人工改了再继续
- **工具校验**：tool 参数模型不一定靠谱，关键参数让用户复核

**核心要求**：
- API 极简：节点内一行 `interrupt(payload)` 即可
- 状态 100% 持久化（依赖 02 checkpointer）
- 进程退出后能恢复中断 thread
- 支持流式：HITL 暂停事件实时推送给前端

**非目标**：
- 内置审批 UI → 不做（用户用自己的前端 + 调 `compiled.resume()`）
- 多人审批工作流 → 不做（一个 thread 一次只接受一个 resume value）

---

## 2. 核心 API

### 2.1 `interrupt()` 函数

```python
from clear_agent.core.interrupt import interrupt

def risky_tool_node(state: ReActState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]

    # 暂停执行，等待外部 resume
    user_decision = interrupt({
        "type": "approval_required",
        "tool_name": tool_call.name,
        "args": tool_call.arguments,
        "message": f"Approve calling {tool_call.name}?",
    })
    # ↓ 仅当 resume(value=...) 后才执行到这里

    if not user_decision.get("approved"):
        return {"messages": [Message(role="tool", content="User rejected the action")]}

    result = execute_tool(tool_call)
    return {"messages": [Message(role="tool", content=result)]}
```

### 2.2 `resume()` 接口

```python
# 1. 主流程触发中断
events = list(compiled.stream({"messages": [...]}, config=RunConfig(thread_id="t42")))
# 最后一个事件是 INTERRUPT，包含 payload

# 2. 用户决策（前端 / CLI / 邮件回调 ...）
decision = {"approved": True}

# 3. 注入决策续跑
final_state = compiled.resume(thread_id="t42", value=decision)
```

### 2.3 在流式事件中观察中断

```python
async for event in compiled.astream(input, config=RunConfig(thread_id="t42")):
    if event.type == StreamEventType.INTERRUPT:
        payload = event.data["payload"]
        decision = await ask_user(payload)
        # 注入回执并续跑
        async for ev2 in compiled.aresume_stream(thread_id="t42", value=decision):
            yield ev2
        break
    yield event
```

---

## 3. 内部机制

### 3.1 `interrupt()` 实现

```python
class GraphInterrupt(BaseException):
    """非 Exception 子类（避免被 try/except Exception 误吞）"""
    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__()

def interrupt(payload: dict) -> Any:
    # 取当前线程的 ContextVar
    ctx = _current_run_ctx.get()
    if ctx is None:
        raise RuntimeError("interrupt() can only be called inside a graph node")

    # 如果 ctx.resume_value 已注入（resume 路径），直接返回
    if ctx.has_resume_value:
        val = ctx.resume_value
        ctx.has_resume_value = False
        return val

    # 否则抛 GraphInterrupt，由 CompiledGraph 捕获
    raise GraphInterrupt(payload)
```

### 3.2 CompiledGraph 中的捕获

```python
def _run_node(self, name: str, state: S):
    try:
        return self._nodes[name](state)
    except GraphInterrupt as gi:
        # 写中断 checkpoint
        ckpt = Checkpoint(
            id=str(uuid7()),
            thread_id=self._ctx.thread_id,
            state=state,
            next_nodes=[name],   # 重要：resume 时回到此节点
            metadata={
                "source": "interrupt",
                "payload": gi.payload,
                "node": name,
            },
        )
        self.checkpointer.put(ckpt)
        # 抛事件给调用方
        self._emit(StreamEventType.INTERRUPT, payload=gi.payload, checkpoint_id=ckpt.id)
        raise GraphPaused(thread_id=self._ctx.thread_id, checkpoint_id=ckpt.id)
```

### 3.3 Resume 路径

```python
def resume(self, thread_id: str, value: Any | None = None):
    ckpt = self.checkpointer.get_tuple(thread_id)
    if ckpt.metadata.get("source") != "interrupt":
        raise ValueError("No interrupt to resume")

    # 在 ContextVar 注入 resume_value
    ctx = RunCtx(thread_id=thread_id, resume_value=value, has_resume_value=True)
    _current_run_ctx.set(ctx)

    # 从中断节点继续（next_nodes[0]）
    return self._execute_loop(state=ckpt.state, start_node=ckpt.next_nodes[0])
```

---

## 4. 三种内置中断模式

### 4.1 Approval（最常用）

```python
from clear_agent.hitl import approval

def send_email_node(state):
    decision = approval(
        prompt=f"Send email to {state['recipient']}?",
        options=["approve", "reject", "edit"],
    )
    if decision == "reject":
        return {"messages": [Message("tool", "user rejected")]}
    if decision == "edit":
        new_content = approval.edit_field("content", default=state["draft"])
        state["draft"] = new_content
    send(state)
```

### 4.2 Edit（让用户改 state）

```python
from clear_agent.hitl import edit_state

def review_plan_node(state):
    edited = edit_state(
        fields=["plan"],
        prompt="Review the plan and edit if needed",
    )
    return {"plan": edited["plan"]}
```

### 4.3 ToolValidation（工具参数复核）

```python
from clear_agent.hitl import validate_tool_args

def critical_tool_node(state):
    tool_call = state["messages"][-1].tool_calls[0]
    validated_args = validate_tool_args(
        tool_name=tool_call.name,
        proposed_args=tool_call.arguments,
        sensitive_fields=["amount", "recipient_account"],
    )
    return {"tool_result": execute(tool_call.name, validated_args)}
```

三个 helper 内部都是 `interrupt(payload)` + 标准化 payload schema 的薄包装。

---

## 5. Payload Schema（前端契约）

中断 payload 是 JSON，固定字段 + 自定义扩展：

```typescript
{
  "type": "approval" | "edit" | "tool_validation" | "custom",
  "thread_id": "t42",
  "checkpoint_id": "ckpt_01H...",
  "node": "send_email_node",
  "prompt": "Send email to alice@example.com?",   // 可选
  "options": ["approve", "reject"],                 // type=approval 时必填
  "fields": ["plan", "tone"],                       // type=edit 时必填
  "current_state_snapshot": { ... },                // 可选，给前端做 diff
  "custom": { ... }                                 // 用户自定义
}
```

---

## 6. 与 streaming.py 的集成

`StreamEventType` 新增枚举值（**复用现有枚举类，不新建**）：

```python
class StreamEventType(Enum):
    # ... 已有值 ...
    INTERRUPT = "interrupt"      # 新增
    RESUMED = "resumed"          # 新增（resume 后续跑首事件）
```

`StreamEvent.to_sse()` 自动支持新枚举值，前端按 `event: interrupt` 过滤即可。

---

## 7. 与 lifecycle.py 的集成

```python
class EventType(Enum):
    # ... 已有值 ...
    AGENT_INTERRUPTED = "agent_interrupted"
    AGENT_RESUMED = "agent_resumed"
```

用户在 `arun(on_interrupt=..., on_resumed=...)` 中注册回调即可。

---

## 8. 测试清单（W3 出口）

`tests/test_hitl_interrupt.py`：

| # | 测试 | 通过标准 |
|---|---|---|
| 1 | 节点内 `interrupt(payload)` | 抛 `GraphPaused`，checkpointer 有 `source=interrupt` 的 ckpt |
| 2 | resume(value=X) 续跑 | 节点内 `interrupt()` 返回 X，下游正常执行 |
| 3 | 进程重启后 resume | kill 后重启进程，`compiled.resume(thread_id)` 仍能恢复 |
| 4 | 流式中断事件 | `stream` 在中断点产出 `INTERRUPT` 事件后自然结束 |
| 5 | Approval 三选项 | "approve" / "reject" / "edit" 三种走不同分支 |
| 6 | Edit 修改 state | 用户改了 state，下游节点看到改后的值 |
| 7 | ToolValidation 改参 | 用户改了工具参数，工具执行的是改后的参数 |
| 8 | 嵌套 interrupt | 同一节点连续两次 `interrupt()` 不串扰（用 has_resume_value 标志位） |

---

## 9. 与 LangGraph 的差异

| 维度 | LangGraph | ClearAgent 2.0 |
|---|---|---|
| 中断 API | `interrupt(payload)` 同名 | 同名（保持迁移友好） |
| Resume API | `graph.invoke(Command(resume=value), config)` | `compiled.resume(thread_id, value=...)`（更直白） |
| 多中断 | 同节点多次 interrupt 自动配对 resume value | 同左 |
| 时间旅行 + 编辑 | `update_state(values)` 写新分支 | `compiled.resume(thread_id, checkpoint_id, state_patch)` |
| 内置模式 | 需自己写 | **提供 `approval` / `edit_state` / `validate_tool_args` 三个 helper** |

---

## 10. 待决问题

1. **interrupt 后是否立即返回 `GraphPaused` 异常给调用方，还是让 stream() 静默结束？**
   - 推荐 stream 静默结束（最后一个事件是 `INTERRUPT`），invoke 抛 `GraphPaused`
   - 用户调 invoke 时希望知道暂停了，调 stream 时已经看到事件就够

2. **resume 时如果已经过期（>24h）怎么办？**
   - 建议提供 `RunConfig.interrupt_ttl_seconds`（默认 86400）
   - 过期后 resume 抛 `InterruptExpiredError`

3. **是否支持「拒绝中断」（即 resume 表示放弃整个 thread）？**
   - 建议提供 `compiled.abort(thread_id)`，写一个 `metadata.source=user_abort` 的 ckpt 并标记 thread 终结

请确认。
