# StateGraph 架构


> 本文给最短上手路径。

## 1. 心智模型

```
StateSchema (TypedDict)
  └─ 字段级 reducer 决定多次写入如何合并
       ├─ add_messages   ：追加消息（去重）
       ├─ append_list    ：列表追加
       ├─ merge_dict     ：字典浅合并
       └─ <自定义 fn>    ：你的累计逻辑

Node = 函数 (state) -> dict
Edge = 节点 → 节点
ConditionalEdge = router(state) -> 下一节点名
```

每跑完一个节点，graph 把节点返回的 dict 通过 reducer 合进当前 state，写一份 checkpoint，然后路由到下一节点。

## 2. 最小例子

```python
from typing import Annotated, TypedDict
from clear_agent.core.graph import StateGraph, START, END, add_messages

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    counter: int

def step1(state):
    return {"messages": [{"role": "system", "content": "hi"}], "counter": 1}

def step2(state):
    return {"counter": state.get("counter", 0) + 1}

g = StateGraph(State)
g.add_node("a", step1)
g.add_node("b", step2)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)

compiled = g.compile()
result = compiled.invoke({})
print(result)  # {"messages": [...], "counter": 2}
```

## 3. 条件边

```python
def router(state):
    return "loop" if state["counter"] < 3 else "done"

g.add_conditional_edges("step", router, {"loop": "step", "done": END})
```

## 4. Checkpointer + thread

```python
from clear_agent.core.checkpoint import SqliteCheckpointer
from clear_agent.core.graph import RunConfig

ck = SqliteCheckpointer("memory/runs.db")
compiled = g.compile(checkpointer=ck)

# 跑：每节点自动写 checkpoint
compiled.invoke({}, config=RunConfig(thread_id="t1"))

# 跑到一半被 kill / 程序崩 → 任意时刻：
compiled.resume("t1")  # 从最后 checkpoint 续跑
```

`thread_id` 是会话级别的标识，同一 `thread_id` 的多次 invoke / resume 共享 checkpoint 链。

## 5. Time travel（重放某个历史快照）

```python
ckpts = ck.list("t1")        # 倒序：最新在前
target = ckpts[3]            # 选第 4 个历史快照
compiled.resume("t1", checkpoint_id=target.id)
```

## 6. 内置 Agent 都能转 graph

```python
agent = ReActAgent(name="x", llm=llm)
graph = agent.as_graph(checkpointer=ck)   # 等价行为，多了 ckpt + HITL 能力
```

`SimpleAgent / ReflectionAgent / PlanSolveAgent` 都暴露 `as_graph()`。也可直接：

```python
from clear_agent.agents import (
    build_react_graph,
    build_simple_graph,
    build_reflection_graph,
    build_plan_solve_graph,
)
```

## 7. 调试：可视化

```python
print(compiled.draw_mermaid())
# graph TD
#   START --> a
#   a --> b
#   b --> END
```

## 8. 异步

`compiled.invoke()` 同步入口；`await compiled.ainvoke(...)` 异步入口（节点函数可以是 async）。
混用规则：sync 节点在 sync invoke 中正常；async 节点在 sync invoke 中会抛错，改用 `ainvoke`。

## 9. 常见坑

- **`reducer` 漏写** → 多次写入同一字段会被覆盖。给 list 字段加 `Annotated[list, add_messages]` 之类。
- **回路无终止** → `compiled.invoke` 默认 `recursion_limit=25`；超出抛错。在 router 里给终止条件，或调高 limit。
- **节点抛异常** → 默认中断 graph。需要"记录并继续"行为时，自己在节点里 try/except 把错误写入 state。
- **`thread_id` 漏传** → checkpoint 不会写入。同 graph 不同 thread 互不影响。
