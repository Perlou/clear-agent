# ClearAgent 2.0 设计文档总览

> **状态**：设计阶段（pre-implementation）
> **目标版本**：`clear-agent==2.0.0a1` → α / β / RC / GA
> **关联计划**：`/Users/perlou/.claude/plans/lucky-rolling-nebula.md`

## 文档索引

| 编号 | 文档 | 阶段 | 一句话说明 |
|---|---|---|---|
| 01 | [graph-architecture.md](01-graph-architecture.md) | 2.0-α / W1 | StateGraph + 节点/边/路由的核心数据模型 |
| 02 | [checkpoint-and-resume.md](02-checkpoint-and-resume.md) | 2.0-α / W2 | Checkpointer 协议 + 三种实现 + resume 语义 |
| 03 | [hitl-guide.md](03-hitl-guide.md) | 2.0-α / W3 | `interrupt()` / `resume(value=...)` + 内置中断模式 |
| 04 | [structured-output.md](04-structured-output.md) | 2.0-α / W4 | `with_structured_output(Pydantic)` API 与 provider 兼容矩阵 |
| 05 | [eval-harness.md](05-eval-harness.md) | 2.0-α / W4 | Dataset / Evaluator / Runner MVP |
| 06 | [migration-1.x-to-2.x.md](06-migration-1.x-to-2.x.md) | 发版前 | 用户迁移指南（5 个常见路径） |
| 07 | [anton-agents-port.md](07-anton-agents-port.md) | 2.0-α/β | 从 AntonAgents 移植 memory + RAG 的 SOP |

## 设计原则（贯穿所有文档）

1. **轻量定位优先** — 不引入 `langchain-core` / `langgraph`；新依赖必须走 `[optional]`
2. **向后兼容** — 旧 `agent.run("...")` 入口永久保留，内部走 graph
3. **可复用资产** — 优先复用 `Config / ToolResponse / TraceLogger / SkillLoader / lifecycle.EventType`，不新建对应物
4. **AntonAgents 移植不直接 copy** — 必须改 namespace、并 Config、换异常类、补测试
5. **API 命名风格** — 沿用现有 `subagent_*` / `trace_*` / `skills_*` 命名

## 阅读顺序建议

- **想理解整体架构**：00 → 01 → 02 → 03（核心运行模型）
- **想看 API 怎么用**：04 → 05（对外接口）
- **想做迁移**：06
- **想协调与 AntonAgents 的代码复用**：07

## 关键约束（已与用户确认）

- ✅ 4 周交付 2.0-α（StateGraph + Checkpointer + HITL + 结构化输出 + eval-harness MVP）
- ✅ 包含 RAG（移植 AntonAgents），包含 eval-harness
- ❌ 永久砍掉 LangServe-style 部署辅助
- ❌ 永久不引入 RL 模块（改为 TraceLogger 导出 SFT/DPO 数据）
- ⚠️ AntonAgents 的 `MemoryManager` 是 0 字节空文件，移植时需要重写
- ⚠️ Anthropic / Gemini 适配器仓库内未实现，2.0-α 结构化输出只覆盖 OpenAI 兼容

## Review 流程

每篇文档读完后，请在 PR / 评论中标注：
- ✅ 同意（可进入实施）
- ⚠️ 需要修改（指出具体段落 + 修改建议）
- ❌ 反对（说明原因，重新讨论）

逐篇 ✅ 后才开 W1 编码。
