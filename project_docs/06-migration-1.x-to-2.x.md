# 06 · 1.x → 2.x 迁移指南

> **阶段**：2.0-α 发版前
> **读者**：现有 ClearAgent 1.0 用户
> **核心承诺**：**旧 API 100% 向后兼容，所有 1.x 代码无需修改即可运行**。本文档只是介绍**新能力如何渐进采用**。

---

## 0. TL;DR

```python
# 你的 1.x 代码无需任何修改
from clear_agent import ReActAgent, ClearAgentLLM, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())
agent = ReActAgent(name="x", llm=llm, tool_registry=registry)
result = agent.run("hello")    # ✅ 仍然工作
```

升级到 2.x 后，`agent.run()` 内部已切换为 graph 执行；行为等价但获得了 checkpoint、HITL 等基础能力，无感升级。

---

## 1. 升级路径速览

| 你是这种用户 | 是否需要改代码 | 推荐升级动作 |
|---|---|---|
| 只用 `agent.run()` 跑 ReAct/Reflection/Plan | ❌ 不需要 | 装新版即可 |
| 自定义工具（继承 `Tool`） | ❌ 不需要 | 装新版即可；想用 pydantic schema 推导可改 W3 风格 |
| 用了 `TaskTool` 派子代理 | ❌ 不需要 | 装新版即可 |
| 自己用 `SkillLoader` | ❌ 不需要 | 装新版即可 |
| 用 `SessionStore.save / load` | ⚠️ 兼容但建议迁移 | 改用 `compiled.resume(thread_id)`（见 §3） |
| 自己改了 `_run_impl` 内部 | ⚠️ **需要改写** | 改成 graph 节点（见 §4） |
| 想用 HITL / 时间旅行 / 结构化输出 / RAG | ✅ 必须用新 API | 见 §5-§7 |

---

## 2. 包结构变化

```diff
clear_agent/
  agents/
    react_agent.py            # 仍存在；ReActAgent 类内部委托 build_react_graph
    factory.py                # 仍存在
+   __init__.py 新增：build_react_graph / build_reflection_graph / ...
  context/
    builder.py                # 仍存在；MemoryTool 移除注释将更新（接通 retrieval）
  core/
    agent.py                  # 加 as_graph() 方法
+   graph.py                  # 新增：StateGraph / CompiledGraph
+   checkpoint.py             # 新增：BaseCheckpointer + 3 种实现
+   interrupt.py              # 新增：interrupt() / GraphInterrupt
+   structured.py             # 新增：StructuredLLM（被 llm.with_structured_output 调用）
    session_store.py          # 仍存在；改为 JsonFileCheckpointer 适配层
  tools/
    base.py                   # 增加 pydantic schema 推导支持（旧手写仍工作）
+ hitl/                       # 新增：approval / edit_state / validate_tool_args
+ retrieval/                  # 新增：embeddings / document_store（移植自 AntonAgents）
+ eval/                       # 新增：dataset / evaluator / runner
```

---

## 3. SessionStore 用户的迁移

### 3.1 旧代码继续工作

```python
# 1.x
agent = ReActAgent(...)
agent.save_session("my-session")            # ✅ 2.x 仍可用
data = SessionStore().load("memory/sessions/my-session.json")  # ✅
```

### 3.2 推荐迁移到 Checkpointer

```python
# 2.x 推荐写法
from clear_agent.core.graph import RunConfig
from clear_agent.core.checkpoint import JsonFileCheckpointer

graph = build_react_graph(llm, registry)
compiled = graph.compile(checkpointer=JsonFileCheckpointer())

# 跑一次（自动 per-step 落盘）
result = compiled.invoke(
    {"messages": [Message("hello", "user")]},
    config=RunConfig(thread_id="user-42"),
)

# 崩溃后续跑
result = compiled.resume(thread_id="user-42")

# 时间旅行
ckpts = compiled.list_checkpoints("user-42")
result = compiled.resume(thread_id="user-42", checkpoint_id=ckpts[3].id)
```

**好处**：自动 per-node 落盘（不再只是 end-of-run），任意 step 可恢复。

---

## 4. 自定义 `_run_impl` 用户的迁移

如果你 1.x 时继承了 `ReActAgent` 并重写了 `_run_impl`，2.x 强烈建议改写为 graph 节点：

### 4.1 旧代码

```python
class MyAgent(ReActAgent):
    def _run_impl(self, input_text, session_start_time, **kwargs):
        # 自定义循环逻辑
        for step in range(self.max_steps):
            response = self.llm.invoke_with_tools(...)
            # 自己处理 tool_calls...
```

### 4.2 新写法（graph）

```python
from clear_agent.core.graph import StateGraph, START, END
from clear_agent.core.message import Message
from typing import TypedDict, Annotated
from clear_agent.core.graph import add_messages

class MyState(TypedDict):
    messages: Annotated[list[Message], add_messages]

def llm_node(state):
    response = llm.invoke_with_tools(state["messages"], tools)
    return {"messages": [response.to_message()]}

def tool_node(state):
    last = state["messages"][-1]
    results = [execute_tool(tc) for tc in last.tool_calls]
    return {"messages": [Message("tool", r) for r in results]}

def router(state):
    return "tools" if state["messages"][-1].tool_calls else END

g = StateGraph(MyState)
g.add_node("llm", llm_node)
g.add_node("tools", tool_node)
g.add_edge(START, "llm")
g.add_conditional_edges("llm", router, {"tools": "tools", END: END})
g.add_edge("tools", "llm")
compiled = g.compile()
```

**好处**：自动获得 checkpoint、HITL、时间旅行能力；没人需要重新发明 `while` 循环。

---

## 5. 用结构化输出（推荐）

```python
from pydantic import BaseModel

class Decision(BaseModel):
    action: Literal["search", "calculate", "finish"]
    reason: str

llm = ClearAgentLLM()
structured = llm.with_structured_output(Decision)

decision: Decision = structured.invoke([{"role": "user", "content": "What's 2+2?"}])
print(decision.action)  # "calculate"
```

详见 [`04-structured-output.md`](04-structured-output.md)。

> ⚠️ **2.0-α 仅 OpenAI 兼容接口端到端验证；Anthropic / Gemini 路径在 2.0-β 完成。**

---

## 6. 用 HITL（人工审批）

```python
from clear_agent.core.interrupt import interrupt

def send_email_node(state):
    decision = interrupt({
        "type": "approval",
        "message": f"Send email to {state['recipient']}?",
        "draft": state["draft"],
    })
    if not decision.get("approved"):
        return {"messages": [Message("tool", "user rejected")]}
    send(state)

# 主流程
events = list(compiled.stream(input, config=RunConfig(thread_id="t1")))
# 最后事件: INTERRUPT, payload 给前端

# 用户确认后
compiled.resume(thread_id="t1", value={"approved": True})
```

详见 [`03-hitl-guide.md`](03-hitl-guide.md)。

---

## 7. 用 RAG（2.0-β 起完整可用）

```python
# 2.0-α：仅 Embedding + DocumentStore
from clear_agent.retrieval import OpenAIEmbeddings, SQLiteDocumentStore

emb = OpenAIEmbeddings()
store = SQLiteDocumentStore("memory/docs.db")
store.add_documents(docs, embeddings=emb)
results = store.similarity_search("query", k=5)

# 2.0-β：完整 RAG pipeline + Memory
from clear_agent.retrieval import RAGPipeline
from clear_agent.memory import WorkingMemory, SemanticMemory

rag = RAGPipeline(embedder=emb, vectorstore=qdrant)
mem = SemanticMemory(...)  # 移植自 AntonAgents
```

详见 [`07-anton-agents-port.md`](07-anton-agents-port.md)。

---

## 8. 已弃用 / 已移除

无 —— 2.0-α 不删除任何 1.x API。

未来版本计划：
- 2.0-RC：`auto_save_enabled` / `auto_save_interval` 字段标记 deprecated（被 checkpointer 取代）
- 2.1：`SessionStore` class 标记 deprecated（仍保留至 3.0）

---

## 9. 升级清单（建议）

```diff
# pyproject.toml / requirements.txt
- clear-agent==1.0.0
+ clear-agent>=2.0.0a1,<2.1.0

# 可选：装 RAG / MCP / 评估扩展
+ clear-agent[memory]>=2.0.0a1     # Qdrant/SQLite vectorstore
+ clear-agent[mcp]>=2.0.0a1         # Model Context Protocol
+ clear-agent[eval]>=2.0.0a1        # 评估依赖
```

跑一次现有测试，确保 1.x 代码不破。然后按 §3-§7 逐项采用新能力。

---

## 10. 常见问题

**Q: 升级后 trace 文件位置变了吗？**
A: 没变（`memory/traces/`）；checkpoint 是新目录 `memory/checkpoints/`。

**Q: 我自己实现的 `Tool.run()` 还能用吗？**
A: 完全能用。`ToolResponse` 协议未变；`to_openai_schema()` 增加 pydantic 推导但旧手写路径保留。

**Q: 4 种 Agent 类还存在吗？**
A: 全部存在；`ReActAgent / ReflectionAgent / PlanSolveAgent / SimpleAgent` 类完全保留，只是内部走 graph。

**Q: 我能继续用 `factory.create_agent("react", ...)` 吗？**
A: 能，未变。

**Q: 子代理（TaskTool）行为有变化吗？**
A: 行为 100% 等价；2.0-RC 会基于 graph 提供更原生的 multi-agent handoff，到时再选择是否切换。

**Q: 我必须改用 pydantic State 吗？**
A: 不必。新 graph API 才需要 State；旧 API 完全无感。

**Q: 性能有变化吗？**
A: graph 引擎单进程同步执行，开销近乎为零（每节点一次 dict 合并 + 一次 checkpoint 写入）。开 InMemoryCheckpointer 时 checkpoint 是字典 set，可以忽略。
