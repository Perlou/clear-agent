# Multi-agent

基于 StateGraph 原生构建多 agent 系统，三种范式开箱即用。

## 三种范式

| 范式 | 说明 | 适用 |
|---|---|---|
| **Supervisor** | 一个中心节点决策路由到哪个 worker，worker 完成回到 supervisor | 工作流明确，要求中心控制 |
| **Swarm** | agents 之间直接 handoff，无中心节点 | 各 agent 自主，技能边界清晰 |
| **TaskTool 子代理** | 父 agent 通过工具派发隔离的子任务 | 父子关系明确，子任务一次性 |

## Supervisor 模式

```python
from clear_agent.multiagent import build_supervisor_graph, HANDOFF_END

def supervisor(state):
    """决策下一步路由到哪个 worker"""
    n = state.get("handoff_count", 0)
    plan = ["researcher", "writer", "reviewer", HANDOFF_END]
    return {"active_agent": plan[n] if n < len(plan) else HANDOFF_END}

def researcher(state):
    return {"messages": [{"role": "assistant", "content": "已收集资料"}]}

def writer(state):
    return {"messages": [{"role": "assistant", "content": "已起草报告"}]}

def reviewer(state):
    return {"messages": [{"role": "assistant", "content": "已校对"}]}

graph = build_supervisor_graph(
    supervisor=supervisor,
    workers={"researcher": researcher, "writer": writer, "reviewer": reviewer},
    max_handoffs=10,
)
result = graph.invoke({"messages": []})
```

**行为**：
- supervisor 决策 `active_agent` → 路由到对应 worker
- worker 完成后**自动清空** `active_agent` → 控制权回到 supervisor 重新决策
- 路由到 `HANDOFF_END` 或不存在的 worker → 终止
- `max_handoffs` 强制终止防死循环

## Swarm 模式

agents 之间通过 `active_agent` 字段直接 handoff：

```python
from clear_agent.multiagent import build_swarm_graph, HANDOFF_END

def planner(state):
    return {
        "messages": [{"role": "assistant", "content": "I'll plan first"}],
        "active_agent": "executor",   # handoff 给 executor
    }

def executor(state):
    return {
        "messages": [{"role": "assistant", "content": "Executed"}],
        "active_agent": HANDOFF_END,  # 终止
    }

graph = build_swarm_graph(
    agents={"planner": planner, "executor": executor},
    default_active="planner",
    max_handoffs=10,
)
graph.invoke({"messages": []})
```

**行为**：
- agent 返回**自己的名字** = 继续工作（不计 handoff）
- agent 返回**其他名字** = 移交（handoff_count +1）
- 返回 `HANDOFF_END` 或不存在的名字 → 终止
- 也支持 `state["active_agent"]` 显式指定入口 agent（覆盖 `default_active`）

## Handoff 原语

让 LLM 自己决定移交：

```python
from clear_agent.multiagent import (
    make_handoff_tools,
    parse_handoff_from_tool_calls,
    HANDOFF_END,
)

def llm_supervisor(state):
    """让 LLM 决定路由"""
    handoff_tools = make_handoff_tools(["researcher", "writer", HANDOFF_END])
    response = llm.invoke_with_tools(
        messages=state["messages"],
        tools=handoff_tools,
        tool_choice="required",
    )

    handoff = parse_handoff_from_tool_calls(response.tool_calls)
    if handoff:
        return {
            "active_agent": handoff.target,
            "messages": [{"role": "assistant", "content": handoff.message}],
        }
    return {"active_agent": HANDOFF_END}
```

`make_handoff_tools(targets, descriptions)` 自动构建 OpenAI function-calling schema：

```json
[
    {
        "type": "function",
        "function": {
            "name": "transfer_to_researcher",
            "description": "Transfer control to agent 'researcher'.",
            "parameters": {...}
        }
    },
    ...
]
```

## TaskTool 子代理（替代方案）

适合"父调一个工具就出一个子结果"的简单场景：

```python
from clear_agent.tools.builtin.task_tool import TaskTool

registry.register_tool(TaskTool())

# 父 agent 调 task 工具：
# task(prompt="research X", agent_type="react", tools=["calculator"])
# 自动起一个新 ReActAgent 完成任务，返回 summary
```

详见 [`subagent-guide.md`](subagent-guide.md)。

## Checkpointer 集成

三种范式都支持 checkpointer：

```python
from clear_agent import SqliteCheckpointer

graph = build_supervisor_graph(
    supervisor, workers, checkpointer=SqliteCheckpointer("memory/multi.db")
)
```

每次 worker / agent 节点跑完都会写一次快照，进程崩了照样 resume。

## 完整 Agent 集成

把 `ReActAgent.run()` 包成 worker：

```python
from clear_agent import ReActAgent, ClearAgentLLM

researcher_agent = ReActAgent(name="r", llm=ClearAgentLLM(), tool_registry=research_tools)
writer_agent = ReActAgent(name="w", llm=ClearAgentLLM(), tool_registry=write_tools)

def make_worker(agent):
    def _worker(state):
        # 把最后一条 user 消息作为输入
        user_msg = next(m for m in reversed(state["messages"]) if m["role"] == "user")
        result = agent.run(user_msg["content"])
        return {"messages": [{"role": "assistant", "content": result}]}
    return _worker

graph = build_supervisor_graph(
    supervisor,
    workers={
        "researcher": make_worker(researcher_agent),
        "writer": make_worker(writer_agent),
    },
)
```

## 选哪种？

- **流程明确、有"老板"决策** → Supervisor
- **平等协作、各管一段** → Swarm
- **父 agent 临时派一个子任务** → TaskTool

三种可以混用：supervisor 内部 worker 也可以是另一个 swarm 的 graph。
