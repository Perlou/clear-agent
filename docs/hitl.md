# Human-in-the-Loop

让 graph 执行到一半暂停，把决策权交给人类（或外部系统），决策后续跑。

## 核心 API

```python
from clear_agent.core.interrupt import interrupt, GraphPaused, GraphInterrupt
```

- `interrupt(payload: dict)` —— 在节点函数内部调用；首次执行抛 `GraphInterrupt` 让 graph 暂停并写 checkpoint
- `GraphPaused` —— graph 抛给外部调用方的信号，含 `thread_id` / `checkpoint_id` / `payload`
- `compiled.resume(thread_id, value=...)` —— 把外部决策注入回执并续跑；节点函数重入时 `interrupt()` 直接返回该 value

## 最小例子：审批工作流

```python
from clear_agent.core.graph import StateGraph, START, END, RunConfig
from clear_agent.core.interrupt import interrupt, GraphPaused
from clear_agent.core.checkpoint import SqliteCheckpointer

def draft_node(state):
    return {"draft": "Hi! Want to grab coffee tomorrow?"}

def approval_node(state):
    decision = interrupt({
        "type": "approval",
        "message": "Send this email?",
        "draft": state["draft"],
    })
    if not decision.get("approved"):
        return {"messages": [{"role": "system", "content": "User rejected"}]}
    return {"messages": [{"role": "system", "content": "Email sent"}]}

g = StateGraph(dict)
g.add_node("draft", draft_node)
g.add_node("approval", approval_node)
g.add_edge(START, "draft")
g.add_edge("draft", "approval")
g.add_edge("approval", END)

compiled = g.compile(checkpointer=SqliteCheckpointer("memory/runs.db"))

try:
    compiled.invoke({}, config=RunConfig(thread_id="email-1"))
except GraphPaused as p:
    # 把 payload 展示给前端 / Slack / 邮件等待用户决策
    print(p.payload)
    # → {"type": "approval", "message": "Send this email?", "draft": "Hi! ..."}

    # 用户点了 ✅ 之后：
    final = compiled.resume("email-1", value={"approved": True})
    print(final["messages"])  # → "Email sent"
```

## 三种内置 HITL 模式

```python
from clear_agent.hitl.patterns import Approval, EditState, ToolValidation
```

### Approval —— 三选项审批

```python
def risky_action(state):
    result = Approval.request(
        message="Confirm deletion?",
        options=["approve", "reject", "edit"],
    )
    if result == "approve":
        ...
```

### EditState —— 让用户编辑 state 字段

```python
def review_node(state):
    new_state = EditState.request(
        fields={"draft": state["draft"]},
        message="Please edit the draft if needed",
    )
    return {"draft": new_state["draft"]}
```

### ToolValidation —— 工具调用前人审

```python
def safe_tool_node(state):
    args = ToolValidation.request(
        tool_name="send_email",
        proposed_args={"to": "boss@x.com", "subject": "..."},
    )
    # args 可能被用户修改过，也可能用户拒绝（抛异常）
    return tool.run(args)
```

## 同节点多次中断

一个节点可以多次调 `interrupt`，resume 时按顺序回放历史值：

```python
def multi_approval(state):
    a = interrupt({"step": 1, "msg": "First decision?"})
    b = interrupt({"step": 2, "msg": "Second decision?"})
    return {"result": (a, b)}

# 第一次跑 → GraphPaused (step 1)
compiled.invoke({}, config=RunConfig(thread_id="t1"))
# resume → 节点重入；interrupt 第一次返回 a，第二次再抛 GraphPaused (step 2)
compiled.resume("t1", value="yes_to_first")
# 再 resume → 节点重入；interrupt 第一次返回 a (回放)，第二次返回 "yes_to_second"
compiled.resume("t1", value="yes_to_second")
```

## 跨进程 / 跨服务

`SqliteCheckpointer` / `JsonFileCheckpointer` 把 checkpoint 写在共享存储上：
- Web 后端：第一个请求触发 `GraphPaused` → 把 `thread_id` 返回前端
- 用户在前端点确认 → 第二个请求带 `thread_id` 调 `compiled.resume(...)`
- 不同进程通过文件 / SQLite 共享 checkpoint

## 注意事项

- `interrupt()` 必须在 graph 节点函数内部调用；否则抛 `RuntimeError`
- 节点函数应是**幂等的**（resume 时会从头重新执行该节点，不是从 `interrupt()` 处继续）
- `payload` 必须是 JSON 可序列化（dict / list / 基本类型）
- `GraphInterrupt` 继承 `BaseException` 而非 `Exception`，避免被节点的 `try/except Exception` 误吞
- `Config.hitl_interrupt_ttl_seconds` 可设置 ckpt 过期时间（默认 86400 秒，即 24 小时；设为 `0` 表示不过期）；超时后 `resume` / `aresume` 抛 `InterruptExpiredError`

## 排错

| 现象 | 原因 |
|---|---|
| `RuntimeError: interrupt() 只能在 graph 节点函数内调用` | 你在裸函数 / 模块顶层调了 `interrupt`，必须在 `graph.invoke` 触发的节点里 |
| resume 后节点的早期副作用又跑了一次 | 节点会从头重入，把副作用包在 `if state.get("phase_done")` 之类的条件里 |
| `GraphInterrupt` 被吞了 | 节点里的 `try/except` 捕到了；改成 `except Exception` 而非裸 `except` |
