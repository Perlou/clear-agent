# 1.x → 2.x 迁移指南

> 设计 spec 详见 [`project_docs/06-migration-1.x-to-2.x.md`](../project_docs/06-migration-1.x-to-2.x.md)。

## TL;DR

**不必动你现有的代码**。所有 1.x 的 `Agent.run()` / `arun()` 入口、`PlanAndSolveAgent` 别名、所有顶层导出全部保留，行为不变。

升级 = `pip install -e .` 后**多了**这些能力：
- `agent.as_graph(checkpointer=...)` 拿 graph 实例
- `llm.with_structured_output(Schema)` 一行结构化输出
- `clear_agent.eval` eval-harness
- `clear_agent.retrieval` 嵌入 + SQLite 文档存储

## 三步迁移路径

### 路径 A：完全不改

```python
# 1.x 代码原样可跑
agent = ReActAgent(name="x", llm=llm, tool_registry=registry)
result = agent.run("...")
```

### 路径 B：加 checkpoint + resume

```python
from clear_agent import SqliteCheckpointer
from clear_agent.core.graph import RunConfig

agent = ReActAgent(name="x", llm=llm, tool_registry=registry)
graph = agent.as_graph(checkpointer=SqliteCheckpointer("memory/runs.db"))

result = graph.invoke(
    {"messages": [{"role": "user", "content": "..."}], "max_steps": 5},
    config=RunConfig(thread_id="thread-1"),
)
# 进程崩了？再跑：
graph.resume("thread-1")
```

### 路径 C：自定义 graph

直接用 `StateGraph` 写完全自定义的多节点流程：

```python
from clear_agent.core.graph import StateGraph, START, END
g = StateGraph(MyState)
g.add_node("planner", ...)
g.add_node("executor", ...)
g.add_conditional_edges("planner", router, {...})
compiled = g.compile(checkpointer=ck)
```

## API 对照

| 1.x | 2.x 等价 / 增强 |
|---|---|
| `agent.run("...")` | 不变 |
| `agent.arun("...")` | 不变 |
| `agent._history` | 不变（property） |
| `PlanAndSolveAgent` | 不变（`PlanSolveAgent` 别名） |
| 手动每 N 步 `session_store.save()` | `agent.as_graph(checkpointer=...)` 自动每节点写 ckpt |
| 工程退化 / 重启需要重跑 | `graph.resume(thread_id)` |
| 写 prompt 让 LLM 输出 JSON 自己解析 | `llm.with_structured_output(Schema).invoke(...)` |
| 自己写循环跑数据集对比 | `clear_agent.eval.run_eval(...)` |

## 配置变化

`Config` 字段**只新增不修改**，1.x 配置仍然有效：

新增（2.0-α）：
- `structured_output_max_retries: int = 2`
- `graph_recursion_limit: int = 25`
- `hitl_interrupt_ttl_seconds: int = 0`（0 = 不过期）
- `checkpoint_backend: str = "memory"`（memory/json/sqlite）
- `eval_default_parallel: int = 4`
- `embed_*` 一族（移植自 AntonAgents）

## Breaking changes（无）

`2.0.0a1` **没有** breaking change。如果你发现你的代码因升级而坏了，请提 Issue。

唯一行为变化：
- 默认 `trace_enabled=True` → 现在 trace 默认输出到 `memory/traces/`，已经在 `.gitignore` 里。

## 常见迁移问答

**Q: 我自定义了 `Agent` 子类覆写了 `_run_impl`，会受影响吗？**
A: 不会。`_run_impl` 仍然是同步入口的核心；`as_graph()` 是平行能力。

**Q: 我能把现有 `ReActAgent` 切成 graph 跑而不改测试吗？**
A: 可以。`ReActAgent.run()` 默认仍走 1.x 单层循环；想用 graph 改成 `agent.as_graph().invoke({...})`。

**Q: 我能在 graph 里继续用 1.x 的 ToolRegistry / TaskTool 吗？**
A: 可以。所有工具系统不变，graph 节点直接调用即可。

**Q: Checkpoint 文件怎么删？**
A: `JsonFileCheckpointer` 直接删目录；`SqliteCheckpointer` 删 .db 文件。或调 `ck.delete(thread_id)`。

**Q: 想跨进程 resume 怎么办？**
A: 用 `JsonFileCheckpointer("memory/sessions/")` 或 `SqliteCheckpointer("memory/runs.db")`，进程间共享文件即可。
