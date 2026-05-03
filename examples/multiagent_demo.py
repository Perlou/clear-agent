"""Multi-agent 演示 —— Supervisor + Swarm 两种范式

跑这个文件可看到：
1. **Supervisor 模式**：中心 supervisor 决策 → 路由到 worker → 完成回到 supervisor
2. **Swarm 模式**：agents 之间直接 handoff，无中心节点

不依赖外部 LLM；用纯 Python 节点演示路由行为。

运行：
    python examples/multiagent_demo.py
"""

from __future__ import annotations

from typing import Any, Dict

from clear_agent.multiagent import (
    HANDOFF_END,
    build_supervisor_graph,
    build_swarm_graph,
)


# ==================================================================
# Part 1: Supervisor 模式 —— 中心化研究→写作→审稿流水线
# ==================================================================


def demo_supervisor() -> None:
    print("=" * 60)
    print("Part 1: Supervisor 模式（研究 → 写作 → 审稿）")
    print("=" * 60)

    plan = ["researcher", "writer", "reviewer", HANDOFF_END]

    def supervisor(state: Dict[str, Any]) -> Dict[str, Any]:
        n = state.get("handoff_count", 0)
        next_agent = plan[n] if n < len(plan) else HANDOFF_END
        print(f"  [supervisor] 第 {n + 1} 步 → {next_agent}")
        return {"active_agent": next_agent}

    def researcher(state: Dict[str, Any]) -> Dict[str, Any]:
        print("    🔍 [researcher] 收集资料中...")
        return {
            "messages": [{"role": "assistant", "content": "找到 5 篇相关论文"}],
            "research": "5 papers on the topic",
        }

    def writer(state: Dict[str, Any]) -> Dict[str, Any]:
        print("    ✍️  [writer] 起草报告中...")
        return {
            "messages": [
                {"role": "assistant", "content": f"基于「{state.get('research')}」起草完成"}
            ],
            "draft": "Report draft v1",
        }

    def reviewer(state: Dict[str, Any]) -> Dict[str, Any]:
        print("    ✅ [reviewer] 校对中...")
        return {
            "messages": [
                {"role": "assistant", "content": f"校对完成: {state.get('draft')}"}
            ],
            "final": "Report final",
        }

    graph = build_supervisor_graph(
        supervisor=supervisor,
        workers={"researcher": researcher, "writer": writer, "reviewer": reviewer},
        max_handoffs=10,
    )
    result = graph.invoke({"messages": []})

    print(f"\n  最终: handoff_count={result.get('handoff_count')}")
    print(f"  final = {result.get('final')}")


# ==================================================================
# Part 2: Swarm 模式 —— 去中心化 planner ↔ executor handoff
# ==================================================================


def demo_swarm() -> None:
    print()
    print("=" * 60)
    print("Part 2: Swarm 模式（planner ↔ executor 自主 handoff）")
    print("=" * 60)

    def planner(state: Dict[str, Any]) -> Dict[str, Any]:
        steps = state.get("steps_done", 0)
        print(f"  📋 [planner] 第 {steps + 1} 轮规划")
        if steps >= 2:
            print("     → 任务完成，handoff_END")
            return {"active_agent": HANDOFF_END}
        return {
            "messages": [{"role": "assistant", "content": f"plan-{steps + 1}"}],
            "active_agent": "executor",
            "steps_done": steps + 1,
        }

    def executor(state: Dict[str, Any]) -> Dict[str, Any]:
        steps = state.get("steps_done", 0)
        print(f"  ⚙️  [executor] 执行第 {steps} 轮，结果回交 planner")
        return {
            "messages": [{"role": "assistant", "content": f"executed step {steps}"}],
            "active_agent": "planner",
        }

    graph = build_swarm_graph(
        agents={"planner": planner, "executor": executor},
        default_active="planner",
        max_handoffs=10,
    )
    result = graph.invoke({"messages": [], "steps_done": 0})

    print(f"\n  最终: handoff_count={result.get('handoff_count')}")
    msgs = [m["content"] for m in result.get("messages", []) if m.get("role") == "assistant"]
    print(f"  消息序列: {msgs}")


# ==================================================================
# Part 3: 子代理（TaskTool）—— 父 agent 派子任务
# ==================================================================


def demo_subagent_overview() -> None:
    print()
    print("=" * 60)
    print("Part 3: 子代理（TaskTool）—— 概念演示")
    print("=" * 60)
    print(
        "  典型用法：\n"
        "    父 agent 调 task 工具：\n"
        "       task(prompt='research X', agent_type='react', tools=['search'])\n"
        "    → 自动起新 ReActAgent 完成任务，返回 summary\n\n"
        "  详见 docs/subagent-guide.md\n"
    )


def main() -> None:
    demo_supervisor()
    demo_swarm()
    demo_subagent_overview()
    print("✅ Multi-agent demo 跑通")


if __name__ == "__main__":
    main()
