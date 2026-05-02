# ClearAgent

> 🤖 生产级多智能体框架 —— 基于 OpenAI 原生 API，集成上下文工程、子代理、Skills 知识外化等 16 项核心能力。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## ✨ 核心特性

- **4 种 Agent 范式**：`SimpleAgent` · `ReActAgent` · `ReflectionAgent` · `PlanSolveAgent`
- **统一 LLM 接口**：自动适配 OpenAI 兼容（DeepSeek/Qwen/Kimi/Ollama 等）、Anthropic、Gemini，支持同步 / 异步 / 流式 / Function Calling
- **上下文工程**：GSSC 流水线、历史压缩、工具输出截断、Token 增量计数
- **工具响应协议**：`ToolResponse` 统一返回，配套熔断器、权限过滤、文件编辑乐观锁
- **子代理机制**：`TaskTool` 派发隔离子任务，工具权限可精确裁剪
- **Skills 知识外化**：按需加载 `SKILL.md`，预期节省 ~85% Token
- **工程化套件**：会话持久化、TodoWrite 进度、DevLog 决策记录、TraceLogger（JSONL+HTML）、异步生命周期钩子、SSE 流式输出

## 🚀 快速开始

```bash
# 安装
pip install -e .            # 或 uv sync

# 配置环境变量
cp .env.example .env        # 填入 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
```

```python
from clear_agent import ClearAgentLLM, ReActAgent, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()                              # 从 .env 自动加载
registry = ToolRegistry(); registry.register_tool(CalculatorTool())
agent = ReActAgent(name="demo", llm=llm, tool_registry=registry)

print(agent.run("计算 (123 + 456) * 2"))
```

异步示例见 [`examples/async_agent_demo.py`](examples/async_agent_demo.py)。

## 📦 项目结构

```
clear_agent/
├── core/             # Agent 基类、LLM、Config、SessionStore、生命周期
├── agents/           # 4 种范式 + 子代理工厂
├── context/          # ContextBuilder（GSSC）/ History / Truncator / TokenCounter
├── tools/            # 工具系统 + 内置工具（file/todowrite/devlog/task/skill 等）
├── observability/    # TraceLogger
└── skills/           # SkillLoader
skills/               # 18 个内置 Skill 包（pdf、docx、ASR、TTS、web-search…）
docs/                 # 16 篇专项指南
tests/                # pytest 测试套件
```

## 📚 文档

完整指南位于 [`docs/`](docs/)：工具响应协议、上下文工程、子代理、Skills、TodoWrite、DevLog、流式 SSE、异步生命周期、可观测性等。

新贡献者请先阅读 [`CLAUDE.md`](CLAUDE.md) 了解架构与开发约定。

## 🛠️ 开发

```bash
pytest                           # 测试
black clear_agent tests          # 格式化
isort clear_agent tests
mypy clear_agent                 # 类型检查
```

## 📄 License

[CC BY-NC-SA 4.0](LICENSE) —— 允许学习/研究/分享，**禁止商业使用**。商用请联系作者 `perloukevin@gmail.com`。
