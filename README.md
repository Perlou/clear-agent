# ClearAgent

> 轻量、可组合的多智能体框架 —— 基于 OpenAI 原生 API，单包提供图编排、检查点、HITL、RAG、Memory、Multi-agent、MCP、结构化输出、Eval-harness 等能力。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-clear--agent-brightgreen.svg)](https://pypi.org/project/clear-agent/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## ✨ 核心能力

**编排与执行**
- **StateGraph** 声明式图构建（节点 / 边 / 条件边 / 字段级 reducer）
- **Checkpointer** 自动每节点快照（Memory / JsonFile / Sqlite），kill 进程也能 resume + 时间旅行
- **Human-in-the-Loop** `interrupt()` + `resume(value=...)` 暂停-续跑，支持同节点多次中断顺序回放
- **Multi-agent** 三种范式：supervisor（中心化）/ swarm（去中心化）/ handoff 原语
- **LCEL-lite** Runnable + `|` 管道组合（自研，零 langchain 依赖）

**LLM 与工具**
- **多 provider** 自动适配：OpenAI 兼容（DeepSeek/Qwen/Kimi/Ollama 等）、Anthropic、Gemini
- **同步 / 真异步 / 流式** 全套接口（`AsyncOpenAI` / `AsyncAnthropic` 真异步，不走线程池假异步）
- **结构化输出** `llm.with_structured_output(MyPydanticModel)` 一行打通三种 method
- **工具系统** `ToolResponse` 协议 + 熔断器 + 权限过滤 + Pydantic 自动 Tool schema 推导
- **工具并行** `run_tools_parallel` / `arun_tools_parallel` 多 tool_calls 并发
- **Resilience** Retry / Fallback / 负载均衡装饰器
- **Multimodal** vision / audio / file content parts 构造器
- **Prompt caching** Anthropic ephemeral helper

**数据与记忆**
- **完整 RAG Pipeline** 7 大职责（加载 / 分块 / 索引 / 检索 / 重排 / 合并 / 压缩）+ MQE / HyDE 查询扩展
- **多层 Memory** WorkingMemory（短期）+ SemanticMemory（长期 + 内存知识图谱）+ MemoryManager 协调
- **向量库** QdrantVectorStore + SQLiteDocumentStore
- **嵌入抽象** Local / DashScope / TFIDF + 工厂回退
- **文档加载** MarkItDown 50+ 格式（PDF/DOCX/XLSX/图像 OCR/音频转写/HTML/代码/配置）

**生态与工程化**
- **MCP 协议** Client（吃外部 MCP 工具）+ Server（暴露给 Cursor / Claude Desktop）
- **Skills 系统** 知识按需注入，~85% Token 节省
- **子代理机制** TaskTool 派发隔离子任务，工具权限可精确裁剪
- **Eval-harness** Dataset + 4 种 Evaluator（含 LLMAsJudge）+ 并发 Runner + Markdown 报告
- **Callbacks** 13 个 hooks（LLM/工具/节点/检索）+ 内置 Logging / Metrics handler
- **TraceLogger** JSONL + HTML 双格式 + SFT / DPO 训练数据导出

## 🚀 快速开始

```bash
# 最小安装
pip install clear-agent

# 按需扩展
pip install "clear-agent[retrieval-qdrant,rag]"   # 完整 RAG
pip install "clear-agent[memory]"                  # 多层记忆 + spaCy NER
pip install "clear-agent[anthropic,gemini]"        # 多 provider
pip install "clear-agent[mcp]"                     # MCP 协议
```

环境变量（参考 `.env.example`）：
```bash
LLM_MODEL_ID=gpt-4o
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

### 1 分钟示例：ReActAgent

```python
from clear_agent import ClearAgentLLM, ReActAgent, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()
registry = ToolRegistry(); registry.register_tool(CalculatorTool())
agent = ReActAgent(name="demo", llm=llm, tool_registry=registry)
print(agent.run("计算 (123 + 456) * 2"))
```

### StateGraph + Checkpoint

```python
from clear_agent import ReActAgent, SqliteCheckpointer
from clear_agent.core.graph import RunConfig

agent = ReActAgent(name="x", llm=ClearAgentLLM())
graph = agent.as_graph(checkpointer=SqliteCheckpointer("memory/runs.db"))

result = graph.invoke(
    {"messages": [{"role": "user", "content": "..."}], "max_steps": 5},
    config=RunConfig(thread_id="thread-1"),
)
# 进程崩了？任意时间：graph.resume("thread-1") 续跑
```

### Human-in-the-Loop

```python
from clear_agent.core.interrupt import interrupt, GraphPaused

def risky_node(state):
    decision = interrupt({"type": "approval", "message": "Send email?", "draft": state["draft"]})
    if not decision.get("approved"):
        return {"messages": [...]}
    ...

try:
    graph.invoke(state, config=RunConfig(thread_id="t1"))
except GraphPaused as p:
    # 把 p.payload 展示给用户，待决策后：
    graph.resume("t1", value={"approved": True})
```

### 结构化输出

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

structured = llm.with_structured_output(Person)
p = structured.invoke([{"role": "user", "content": "Alice 是 30 岁的老师"}])
print(p.name, p.age)  # Alice 30
```

### 完整 RAG

```python
from clear_agent.retrieval.rag import create_rag_pipeline

rag = create_rag_pipeline(qdrant_url="http://localhost:6333", rag_namespace="my_kb")
rag["add_documents"](["docs/a.pdf", "docs/b.md"])
hits = rag["search"]("如何配置 LLM？", top_k=5)
hits = rag["search_advanced"]("...", enable_mqe=True, enable_hyde=True)
```

### Multi-agent supervisor

```python
from clear_agent.multiagent import build_supervisor_graph, HANDOFF_END

def supervisor(state):
    n = state.get("handoff_count", 0)
    return {"active_agent": ["researcher", "writer", HANDOFF_END][n] if n < 3 else HANDOFF_END}

graph = build_supervisor_graph(supervisor, {"researcher": researcher_fn, "writer": writer_fn})
result = graph.invoke({"messages": []})
```

### Pydantic 自动 Tool schema

```python
from pydantic import BaseModel, Field
from clear_agent.tools.from_pydantic import pydantic_tool

class AddArgs(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")

@pydantic_tool(description="加法")
def add(args: AddArgs) -> int:
    return args.a + args.b

registry.register_tool(add)
```

### Resilience

```python
from clear_agent.core.resilience import retry, with_fallbacks

@retry(max_attempts=3, retry_on=(ConnectionError,), backoff=0.5)
def call_api():
    return llm.invoke(...)

safe_llm = with_fallbacks(primary_llm.invoke, [backup_llm_1.invoke, backup_llm_2.invoke])
response = safe_llm(messages)
```

更多示例见 [`examples/`](examples/) 目录。

## 📦 项目结构

```
clear_agent/
├── core/             # Agent 基类 / LLM / Config / 编排基础
│   ├── graph.py            # StateGraph + reducers
│   ├── checkpoint.py       # Memory/JsonFile/Sqlite
│   ├── interrupt.py        # interrupt() + GraphPaused
│   ├── structured.py       # with_structured_output
│   ├── runnable.py         # LCEL-lite Runnable + |
│   ├── callbacks.py        # 13 hooks 协议
│   ├── parallel.py         # 工具并行 helper
│   ├── resilience.py       # Retry / Fallback / 负载均衡
│   └── multimodal.py       # vision / audio + cache_control
├── agents/           # 4 种范式（Simple/ReAct/Reflection/PlanSolve）+ graph builders
├── multiagent/       # supervisor / swarm / handoff
├── mcp/              # MCP client / server / adapter
├── hitl/             # Human-in-the-Loop patterns
├── eval/             # Dataset / Evaluator / Runner
├── retrieval/        # 嵌入 + Qdrant + SQLite + RAG pipeline
│   ├── embeddings.py
│   ├── rag/                # document + pipeline (7 大职责)
│   └── storage/            # SQLite + Qdrant
├── memory/           # WorkingMemory + SemanticMemory + Manager
├── context/          # GSSC 流水线
├── tools/            # 工具系统 + Pydantic 自动推导 + 内置工具
├── observability/    # TraceLogger（JSONL + HTML + SFT/DPO 导出）
└── skills/           # SkillLoader

skills/               # 18 个内置 Skill 包（pdf/docx/xlsx/ASR/TTS/VLM/web-search…）
docs/                 # 用户指南
examples/             # 演示
tests/                # 740+ pytest 测试
```

## 📚 文档

- **快速开始**：[`docs/quickstart.md`](docs/quickstart.md)
- **核心架构**：[`docs/graph-architecture.md`](docs/graph-architecture.md) · [`docs/hitl.md`](docs/hitl.md)
- **数据与记忆**：[`docs/rag-guide.md`](docs/rag-guide.md) · [`docs/memory-guide.md`](docs/memory-guide.md)
- **工具与协议**：[`docs/tool-system.md`](docs/tool-system.md) · [`docs/structured-output.md`](docs/structured-output.md) · [`docs/mcp.md`](docs/mcp.md)
- **Multi-agent**：[`docs/multi-agent.md`](docs/multi-agent.md)
- **评估与可观测性**：[`docs/eval-harness.md`](docs/eval-harness.md) · [`docs/observability.md`](docs/observability.md)
- **进阶**：[`docs/context-engineering.md`](docs/context-engineering.md) · [`docs/skills.md`](docs/skills.md) · [`docs/async-streaming.md`](docs/async-streaming.md)

## 🛠️ 本地开发与调试

### 1. 克隆与环境

```bash
git clone https://github.com/Perlou/clear-agent.git
cd clear-agent

python3.10 -m venv .venv && source .venv/bin/activate     # 推荐 3.10/3.11/3.12
pip install --upgrade pip

# editable 安装 + 全套可选依赖（开发态推荐）
pip install -e ".[mcp,retrieval-qdrant,rag,memory,anthropic,gemini]"

# 仅装最小核心也行
pip install -e .

# 配置 .env
cp .env.example .env  # 填 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
```

> 📌 装好后任何位置都能 `from clear_agent import ...`，且改 `clear_agent/` 源码即时生效，无需重装。

### 2. 在 `examples/` 里调试新 demo

`examples/` 下的脚本（如 `examples/trip-planner/`）默认就用本仓库的源码：

```bash
# 单文件 demo
python examples/async_agent_demo.py

# 子项目 demo（trip-planner 等需要后端 / 前端）
cd examples/trip-planner/backend
pip install fastapi 'uvicorn[standard]' pydantic-settings python-dotenv uv
python run.py
```

如果想验证「外部用户 pip install clear-agent 后是否能跑」，把整个子目录复制出去用 `requirements.txt` 重装即可（trip-planner 已带）。

### 3. 测试

```bash
pytest                                # 全量
pytest tests/test_graph_basics.py -v  # 单文件
pytest -k structured -v               # 关键字过滤
pytest -m "not integration" -q        # 跳过需真 API 的集成测试
```

**首次跑测试常见坑：**

| 现象 | 原因 | 解决 |
|---|---|---|
| `Using SOCKS proxy, but socksio not installed` | 系统设了 `all_proxy=socks5://...` | `pip install 'httpx[socks]'` 或 `unset all_proxy http_proxy https_proxy` |
| `async def functions are not natively supported` | 缺 pytest-asyncio | `pip install pytest-asyncio` |
| 真实 LLM 测试 401 / 超时 | `.env` 没配 / endpoint 不通 | 先 `pytest -m "not integration"` 跑单测，再单独修 |

### 4. 代码风格 / 类型检查

```bash
black clear_agent tests && isort clear_agent tests   # 格式化
mypy clear_agent                                      # 严格类型
```

提交前可选挂 pre-commit：`pip install pre-commit && pre-commit install`。

### 5. 调试技巧

**最小复现环境变量：**

```bash
export PYTHONBREAKPOINT=ipdb.set_trace      # 让 breakpoint() 进 ipdb
export CLEAR_AGENT_LOG_LEVEL=DEBUG          # 打开 framework 内部日志
pytest tests/test_xxx.py::test_yyy -v -s    # -s 不吞 print/输入
```

**只跑挂掉的那一个：**

```bash
pytest --lf -x                              # last-failed + 第一个失败就停
pytest tests/test_xxx.py -k "name and not slow" --pdb   # 失败处自动进 pdb
```

**调试 LLM 真实调用 + 工具循环：**

```python
import logging; logging.basicConfig(level=logging.DEBUG)
from clear_agent.observability.trace_logger import TraceLogger
TraceLogger().enable()                      # 每一步落 JSONL + HTML 时间线
```

**追踪 MCP 子进程：** 给 `MCPClient.connect_stdio(...)` 传 `env={"DEBUG": "1", ...}`，子进程的 stderr 会原样透出到父进程。

## 📦 发布到 PyPI

`scripts/release.sh` 是一键发布脚本，按 **11 个 Phase** 顺序执行；任何一个 Phase 失败都会停下并给出可恢复的提示。

### 脚本流程

| Phase | 做什么 | 失败后怎么办 |
|---|---|---|
| 0  工具与环境检查 | 校验 `python` / `build` / `twine` / `git` | 装缺失工具 |
| 1  Git 工作区检查 | 确认 working tree 干净 | `git stash` 或 `git commit` |
| 2  版本号 | 校验 `pyproject.toml` 版号未在 PyPI 占用 | `--bump patch` 或 `--version X.Y.Z` |
| 3  必备文件 | 检查 README / LICENSE / MANIFEST / py.typed | 补齐对应文件 |
| 4  全量 pytest | 跑所有测试（可 `--skip-tests`） | 修代码或确认是环境问题再 `--skip-tests` |
| 5  清理 + 构建 | `rm -rf dist build` + `python -m build` | 看 build 日志 |
| 6  twine check | `twine check dist/*` 检查长描述/元数据 | 修 `pyproject.toml` |
| 7  包内容审查 | 列 wheel/sdist 内容确认没漏 / 没多 | 调 `MANIFEST.in` 或 `[tool.setuptools]` |
| 8  干净环境装机验证 | 临时 venv 装 wheel + 冒烟 import | `--skip-clean-install` 跳过 |
| 9  上传 | `twine upload`（pypi / testpypi） | 看凭证 / 网络 |
| 10 Git tag | `git tag vX.Y.Z` + 可选 `git push --tags` | `--skip-tag` 跳过 |

### 常用姿势

```bash
# 1) 干跑：只构建 + 校验，不上传（强烈建议每次发版前先跑一遍）
bash scripts/release.sh --dry-run

# 2) 先发 TestPyPI 验证（推荐流程）
bash scripts/release.sh --test

# 3) 正式发到 PyPI
bash scripts/release.sh                # 交互式
bash scripts/release.sh --yes          # CI 用，全部自动 yes

# 4) 自动 bump 版本号
bash scripts/release.sh --bump patch          # 2.0.0 → 2.0.1
bash scripts/release.sh --bump minor          # 2.0.0 → 2.1.0
bash scripts/release.sh --version 2.0.0rc1    # 显式指定

# 5) 紧急发版组合（不推荐）
bash scripts/release.sh --skip-tests          # 跳测试（环境问题暂时绕过）
bash scripts/release.sh --skip-clean-install  # 跳干净环境验证（提速）
bash scripts/release.sh --skip-tag            # 不打 git tag
```

完整参数详见 `scripts/release.sh` 头部注释；端到端 SOP 与版本策略见 [`docs/pypi-release.md`](docs/pypi-release.md)。

### 退出码

`0` 成功 ｜ `1` 通用失败 ｜ `2` 参数错误 ｜ `3` 环境/工具缺失 ｜ `4` 版本检查失败 ｜ `5` 测试失败 ｜ `6` 构建失败 ｜ `7` 上传失败

### 凭证配置（任选其一）

```bash
# 方式 A：环境变量（CI 推荐）
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...

# 方式 B：~/.pypirc（本地推荐）
cat > ~/.pypirc <<EOF
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
EOF
chmod 600 ~/.pypirc
```

### 发版前自查清单

- [ ] `pyproject.toml` 与 `clear_agent/version.py` 版本号一致
- [ ] `CHANGELOG` / commit history 已整理
- [ ] `pytest` 全绿（环境问题除外，见上文表格）
- [ ] PyPI 上目标版本号未被占用（脚本会自动检查）
- [ ] 本地 `--dry-run` 通过
- [ ] 先打 TestPyPI 验证 `pip install -i https://test.pypi.org/simple/ clear-agent==X.Y.Z`

## 📄 License

[CC BY-NC-SA 4.0](LICENSE) —— 允许学习/研究/分享，**禁止商业使用**。商用请联系作者 `perloukevin@gmail.com`。
