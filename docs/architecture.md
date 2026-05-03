# ClearAgent 核心技术架构全景

> 一份"上帝视角"的架构地图：让你 30 分钟内看清 clear-agent 的全部模块、它们的边界、彼此的依赖、以及关键设计决策。需要细节时再跳到对应专题文档。

---

## 1. 一句话定位

> **ClearAgent 是一个轻量、可组合的多智能体框架**，核心抽象 `StateGraph + Agent + LLM + Tool`，单包提供图编排、检查点、HITL、RAG、Memory、Multi-agent、MCP、结构化输出、Eval-harness 等能力。**零 langchain 依赖**，所有上层能力构建在自研图引擎之上。

| 对比维度 | LangChain | LangGraph | ClearAgent |
|---|---|---|---|
| 图引擎 | 无（LCEL） | LangGraph | 自研 StateGraph |
| Agent 范式 | LCEL 组合 | 节点函数 | 4 内建范式 + StateGraph |
| Checkpoint | 无 | LangGraph CP | Memory/JsonFile/Sqlite |
| MCP | 第三方 | 第三方 | 内建 client + server |
| Eval | LangSmith（外） | LangSmith（外） | 自带 Eval-harness |
| 工具 schema | 手写 | 手写 | Pydantic 自动推导 |
| 体积 | 重 | 中 | **单包，~12k LOC** |

---

## 2. 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ⓪ 用户空间 (Examples / Apps)                  │
│        examples/, skills/<name>/SKILL.md, your_app.py              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  ④ 上层能力     RAG · Memory · Multi-agent · Eval · HITL · MCP      │
│  retrieval/    memory/    multiagent/    eval/    hitl/    mcp/    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  ③ Agent 范式   SimpleAgent · ReActAgent · ReflectionAgent · PlanSolve│
│  agents/  ─→  每个 Agent 都暴露 .as_graph() 转 StateGraph             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  ② 编排原语    StateGraph · Checkpoint · Interrupt · Runnable        │
│                Callbacks · Resilience · Parallel · Streaming         │
│  core/graph.py  core/checkpoint.py  core/interrupt.py ...           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  ① 核心抽象     LLM (适配 OpenAI/Anthropic/Gemini)                    │
│                Tool (Pydantic 自动推导 + 熔断 + 过滤)                  │
│                Message · Config · Lifecycle                         │
│  core/llm.py  core/llm_adapters.py  tools/                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  ⓪ 横切关注     Observability (TraceLogger) · Context (GSSC) ·       │
│                 Skills (按需注入) · Session (持久化) · Errors         │
│  observability/  context/  skills/  core/session_store.py           │
└─────────────────────────────────────────────────────────────────────┘
```

**层间规则**：上层可以自由依赖下层，**严禁**反向依赖。同层之间通过明确的接口（如 `BaseLLMAdapter`、`Tool` 协议、`Checkpointer` 协议）协作。

---

## 3. 四个核心抽象

### 3.1 StateGraph —— 编排引擎

```
StateSchema (TypedDict)
  └─ 字段级 reducer 决定多次写入如何合并
       ├─ add_messages   追加消息（去重）
       ├─ append_list    列表追加
       ├─ merge_dict     字典浅合并
       └─ <自定义 fn>     你的累计逻辑

Node = 函数 (state) -> dict
Edge = 节点 → 节点
ConditionalEdge = router(state) -> 下一节点名
```

每跑完一个节点：返回 dict → 经 reducer 合进 state → 写 checkpoint → 路由到下一节点。

**核心 API**：`StateGraph` / `add_node` / `add_edge` / `add_conditional_edges` / `compile()` → `CompiledGraph` 提供 `invoke`/`ainvoke`/`stream`/`resume`/`get_state`/`draw_mermaid`。

### 3.2 LLM —— 多 Provider 统一接口

```
ClearAgentLLM
  └─ 通过 base_url 自动选 BaseLLMAdapter
       ├─ OpenAIAdapter      ← OpenAI 兼容（含 DeepSeek/Qwen/Kimi/Ollama）
       ├─ AnthropicAdapter   ← Claude
       └─ GeminiAdapter      ← Google
  
  统一接口：
    invoke / ainvoke              ← 普通对话
    invoke_with_tools / ainvoke_~ ← Function Calling
    stream_invoke / astream_~     ← 流式
    with_structured_output(P)     ← Pydantic 直出
    serialize_assistant_message() ← 多轮回写策略（处理 reasoning_content 等）
```

**关键设计**：thinking 模型（DeepSeek-V4 thinking / R1 / Anthropic extended thinking 等）的 `reasoning_content` 多轮回传策略由 adapter 自行决定，不写死模型名单。

### 3.3 Tool —— 工具协议与 Pydantic 桥

```
Tool 协议
  ├─ get_parameters() -> List[ToolParameter]
  └─ run(args: dict) -> ToolResponse(status, text, data, error_info)

@pydantic_tool(description="...")
def my_tool(args: MyArgsModel) -> int: ...
   └─ 自动从 BaseModel 推导 OpenAI function spec

ToolRegistry
  ├─ register_tool / list_tools / get_tool
  ├─ ToolFilter (ReadOnly/FullAccess/Custom)
  └─ CircuitBreaker (失败熔断)
```

**响应协议**：所有工具统一返回 `ToolResponse`，包含 `status` / `text`（给 LLM 看）/ `data`（结构化数据）/ `error_info`（错误码 + 消息 + 上下文）。

### 3.4 Checkpoint —— 状态快照

```
BaseCheckpointer 协议
  ├─ InMemoryCheckpointer    ← 内存（测试 / 临时）
  ├─ JsonFileCheckpointer    ← JSON 文件（兼容老 sessions/）
  └─ SqliteCheckpointer      ← SQLite（推荐生产）

每节点写一份快照：
  {thread_id, checkpoint_id, parent, state, next_nodes, metadata}

支持时间旅行：ck.list(thread_id) 倒序拿历史 + 任意点 resume。
```

---

## 4. 一次 `agent.run()` 的全链路

以 `ReActAgent` 为例，看从用户提问到最终答案的数据流：

```
用户调 agent.run("...")
  │
  ├─① Agent.__init__ 阶段（一次性）
  │    ├─ HistoryManager（对话历史 + 自动压缩）
  │    ├─ ObservationTruncator（长工具输出截断到磁盘）
  │    ├─ TokenCounter（tiktoken 缓存 + 增量）
  │    ├─ TraceLogger（JSONL + HTML 双写，session 滚动）
  │    ├─ SkillLoader（按需加载 skill 元数据）
  │    └─ 自动注册 TaskTool/TodoWrite/DevLog 等内建工具
  │
  ├─② 进入 ReAct 循环（_react_graph.py 编译的 StateGraph）
  │    
  │    ┌──→ llm_node ──┐
  │    │   • llm.invoke_with_tools(messages, tools=[...], "auto")
  │    │   • adapter.serialize_assistant_message(response)
  │    │     └─ 自动写回 reasoning_content（DeepSeek V4 thinking 等需要）
  │    │   • TraceLogger.log_event("model_output")
  │    │
  │    │  router_after_llm:
  │    │    ├─ 有 tool_calls 且未到 max_steps → tool_executor
  │    │    └─ 否则 → END
  │    │
  │    └── tool_executor ──┐
  │        • 串行/并行执行 tool_calls (run_tools_parallel)
  │        • Thought/Finish 走特殊路径
  │        • 工具输出过 ObservationTruncator → 长内容落盘
  │        • CircuitBreaker 检测失败熔断
  │        ↓
  │        回到 llm_node（上下文累加）
  │
  ├─③ 每节点结束自动：
  │    ├─ Reducer 合并 state（messages 追加 / counter 累加 / ...）
  │    ├─ Checkpointer.put(snapshot)（生产用 Sqlite）
  │    └─ 触发 Callbacks（13 hooks：on_llm_*, on_tool_*, on_node_*, ...）
  │
  └─④ 返回 final_answer
       └─ TraceLogger.finalize() 关 JSONL + 写 HTML 统计面板
       └─ SessionStore 持久化对话历史（默认 memory/sessions/）
```

**关键不变量**：每次 `agent.run()` 是一个独立 session，TraceLogger 自动滚动到新文件（v2.0 修复了单例复用 bug）。

---

## 5. 模块全景 + 关系

```
clear_agent/
├── core/                        ⬛ 编排引擎核心（无业务）
│   ├── graph.py                 StateGraph / CompiledGraph / reducers
│   ├── checkpoint.py            BaseCheckpointer + 3 个实现
│   ├── interrupt.py             interrupt() / GraphPaused / GraphInterrupt
│   ├── structured.py            with_structured_output（4 method）
│   ├── runnable.py              LCEL-lite（Runnable + | 管道）
│   ├── callbacks.py             13 hooks 协议 + LoggingHandler/MetricsHandler
│   ├── parallel.py              run_tools_parallel / arun_tools_parallel
│   ├── resilience.py            @retry / with_fallbacks / round_robin
│   ├── multimodal.py            text/image/audio/file parts + cache_control
│   ├── llm.py                   ClearAgentLLM 门面
│   ├── llm_adapters.py          OpenAI/Anthropic/Gemini 三个 adapter
│   ├── llm_response.py          LLMResponse / LLMToolResponse / ToolCall
│   ├── agent.py                 Agent 基类（持有 LLM + tools + ctx + trace + session）
│   ├── config.py                Config（pydantic-settings 集中管理）
│   ├── lifecycle.py             AgentEvent / EventType / LifecycleHook
│   ├── streaming.py             StreamEvent / StreamEventType
│   ├── message.py               Message dataclass + roles
│   ├── session_store.py         SessionStore（JSON 落盘对话历史）
│   └── exceptions.py            ClearAgentException 等
│
├── agents/                      ⬛ 4 内建 Agent 范式
│   ├── simple_agent.py          单轮 / 多轮 Function Calling
│   ├── react_agent.py           Thought/Action/Observation 循环（主力）
│   ├── reflection_agent.py      自我反思优化
│   ├── plan_solve_agent.py      规划-执行
│   ├── factory.py               按字符串 / 配置创建 agent
│   ├── _react_graph.py          ReAct 的 StateGraph 实现
│   ├── _simple_graph.py         同上
│   ├── _reflection_graph.py     同上
│   └── _plan_solve_graph.py     同上
│
├── tools/                       ⬛ 工具系统
│   ├── base.py                  Tool 基类 + ToolParameter + @tool_action
│   ├── response.py              ToolResponse 协议
│   ├── errors.py                ToolErrorCode 枚举
│   ├── circuit_breaker.py       熔断器
│   ├── tool_filter.py           ReadOnly/FullAccess/Custom 过滤器
│   ├── registry.py              ToolRegistry + global_registry
│   ├── from_pydantic.py         @pydantic_tool / tool_from_pydantic
│   └── builtin/
│       ├── calculator.py        CalculatorTool（AST 安全求值）
│       ├── file_tools.py        Read/Write/Edit/MultiEdit（乐观锁）
│       ├── todowrite_tool.py    TodoWrite（任务列表）
│       ├── devlog_tool.py       DevLog（开发日志）
│       ├── task_tool.py         TaskTool（子代理派发）
│       └── skill_tool.py        SkillTool（按需注入 skill 内容）
│
├── multiagent/                  🔵 多智能体协作
│   ├── handoff.py               Handoff 数据类 + make_handoff_tool(s)
│   ├── supervisor.py            build_supervisor_graph（中心化）
│   └── swarm.py                 build_swarm_graph（去中心化）
│
├── mcp/                         🔵 MCP 协议集成
│   ├── client.py                MCPClient（stdio + sse）
│   ├── server.py                MCPServer（暴露给 Cursor/Claude Desktop）
│   └── adapter.py               MCPToolAdapter（MCP tool → ClearAgent tool）
│
├── hitl/                        🔵 Human-in-the-Loop 模式
│   └── patterns.py              Approval / Edit / ToolValidation
│
├── retrieval/                   🔵 RAG + 嵌入 + 向量库
│   ├── embeddings.py            Local / DashScope / TFIDF
│   ├── rag/
│   │   ├── document.py          MarkItDown 50+ 格式
│   │   └── pipeline.py          7 大职责（加载/分块/索引/检索/重排/合并/压缩）
│   └── storage/
│       ├── document_store.py    SQLiteDocumentStore
│       └── qdrant_store.py      QdrantVectorStore
│
├── memory/                      🔵 多层 Memory
│   ├── base.py                  MemoryItem + BaseMemory（7 抽象接口）
│   ├── working.py               WorkingMemory（短期 + heapq 优先级 + TTL）
│   ├── semantic.py              SemanticMemory（向量 + 内存知识图谱）
│   └── manager.py               MemoryManager（路由 + 跨子系统聚合）
│
├── eval/                        🟢 评估框架
│   ├── dataset.py               Dataset.from_jsonl / from_list
│   ├── evaluator.py             ExactMatch/Contains/LLMAsJudge/Custom
│   └── runner.py                run_eval（并发 + Markdown 报告）
│
├── observability/               🟡 可观测性
│   ├── trace_logger.py          JSONL + HTML 双写 + session 自动滚动
│   └── trace_export.py          export_to_sft_jsonl / export_to_dpo_pairs
│
├── context/                     🟡 上下文工程
│   ├── builder.py               ContextBuilder（GSSC 流水线）
│   ├── history.py               HistoryManager（按轮次自动压缩）
│   ├── truncator.py             ObservationTruncator（长输出截断 + 落盘）
│   └── token_counter.py         TokenCounter（tiktoken 缓存 + 增量）
│
└── skills/                      🟡 知识外化
    └── loader.py                SkillLoader（启动加载元数据，按需注入正文）

skills/                          📦 18 个内建 Skill 包（pdf/docx/xlsx/ASR/TTS/...）
```

**模块依赖方向**（粗箭头 = 依赖）：

```
agents/ ───→ core/   (LLM + Graph + Agent 基类)
agents/ ───→ tools/  (注册 + 调用)

multiagent/ ───→ core/graph.py
hitl/ ───→ core/interrupt.py
mcp/ ───→ tools/registry.py

retrieval/ ──→ core/llm.py (RAG 内部用 LLM 做 MQE/HyDE/重排)
memory/ ───→ retrieval/embeddings.py + retrieval/storage/qdrant_store.py
eval/ ───→ core/structured.py (LLMAsJudge 用 Pydantic 直出)

observability/, context/, skills/  ⊥ (横切，不依赖业务)
```

---

## 6. 扩展点（用户自定义）

| 我想… | 扩展点 | 入口 |
|---|---|---|
| 加新工具 | `@pydantic_tool` 装饰器 / 继承 `Tool` | `tools/from_pydantic.py` |
| 加新 Agent 范式 | 继承 `Agent`，实现 `run()`；可选 `as_graph()` | `core/agent.py` |
| 接入新 LLM provider | 继承 `BaseLLMAdapter` | `core/llm_adapters.py` |
| 加新 reducer | 写 `(left, right) -> merged` 函数 | `core/graph.py` |
| 加新 Checkpointer 后端 | 实现 `BaseCheckpointer` 协议 | `core/checkpoint.py` |
| 加新 Skill 包 | `skills/<name>/SKILL.md` + 正文 | 自动发现 |
| 加新 Eval | 继承 `Evaluator` | `eval/evaluator.py` |
| 加可观测性 hook | 实现 13 个 callback 中任意 | `core/callbacks.py` |
| 接入新 MCP server | `MCPClient.connect_stdio/sse` | `mcp/client.py` |
| 加新 Memory 子系统 | 实现 `BaseMemory` | `memory/base.py` |
| 加 thinking 模型 reasoning 回写策略 | 子类化 adapter 覆盖 `_should_echo_reasoning` | `core/llm_adapters.py` |
| 加 structured output method | 在 `Method` Literal 加 + `_loop` 加分支 | `core/structured.py` |

---

## 7. 关键设计决策

| 决策 | 现状 | 理由 |
|---|---|---|
| **零 langchain 依赖** | 自研 `Runnable` + 自研 StateGraph | 体积小、无版本绑定、无第三方 breaking change 风险 |
| **Memory 不引 Neo4j** | `SemanticMemory` 知识图谱完全在内存 | 90% 场景 Qdrant 够用，避免运维负担 |
| **Skills 按需注入** | 启动只加载 `SKILL.md` 元数据，调用时才注入正文 | 20 个 skill 场景预期省 ~85% token |
| **TraceLogger 双格式** | JSONL（机器） + HTML（人类） | jq/grep 友好 + 可视化兼得 |
| **工具响应统一 `ToolResponse`** | 不返回 raw string，而是 `(status, text, data, error_info)` | 给 LLM 看 text，给程序用 data，错误码可枚举 |
| **Pydantic 自动 Tool schema** | `@pydantic_tool` 单装饰器 | 避免手写 OpenAI function spec 的错位 |
| **thinking 模型策略钩子** | `_should_echo_reasoning` 由 adapter 决定 | DeepSeek V4 必须回写，R1 禁止回写 —— 不能一刀切 |
| **真异步 + 流式** | `AsyncOpenAI` / `AsyncAnthropic` 真异步，不走线程池 | 避免线程池假异步的 GIL 瓶颈 |
| **结构化输出 4 method** | `function_calling` / `json_mode` / `json_schema` / `prompt_json` | thinking 模型不支持强制 tool_choice → `prompt_json` 兜底 |
| **乐观锁文件操作** | EditTool 检查 `mtime` | 多 agent 并发改文件不互相覆盖 |
| **集成测试隔离** | `@pytest.mark.integration` + `addopts = -m "not integration"` | 默认 pytest 不依赖网络/真实 LLM |

---

## 8. 持久化目录约定

| 类型 | 默认路径 | Config 字段 |
|---|---|---|
| Trace（JSONL+HTML） | `memory/traces/` | `trace_dir` |
| Session（对话历史） | `memory/sessions/` | `session_dir` |
| Todo | `memory/todos/` | `todowrite_persistence_dir` |
| DevLog | `memory/devlogs/` | `devlog_persistence_dir` |
| 工具完整输出 | `tool-output/` | `tool_output_dir` |
| Eval 报告 | `memory/eval/<run_id>/` | runner 入参 |
| Sqlite Checkpoint | 用户指定 | `SqliteCheckpointer(path=...)` |
| Qdrant RAG 集合 | `clear_agent_rag_vectors` | `create_rag_pipeline(collection_name=...)` |
| Qdrant Semantic 集合 | `clear_agent_semantic` | 自动创建 |

`memory/` 与 `tool-output/` 已加入 `.gitignore`。

---

## 9. 可选依赖矩阵

```
clear-agent                # 核心（runtime 8 个包）
  ├── [anthropic]           Claude support
  ├── [gemini]              Google Gemini support
  ├── [dashscope]           阿里百炼 embeddings
  ├── [retrieval]           scikit-learn (TFIDF)
  ├── [retrieval-qdrant]    Qdrant 向量库
  ├── [rag]                 sentence-transformers + transformers + torch + markitdown + langdetect (~2GB)
  ├── [memory]              scikit-learn + spacy（NER）
  ├── [mcp]                 MCP 协议
  ├── [all]                 上面除 dev 全装
  └── [dev]                 pytest/black/isort/mypy/ruff/build/twine
```

---

## 10. 常见陷阱与最佳实践

| 陷阱 | 怎么避免 |
|---|---|
| 在 FastAPI async startup 里调 sync `MCPClient.register_to` | 改用 `asyncio.to_thread` 或 sync handler 触发 lazy init |
| 多次 `agent.run()` 复用 agent 实例后 trace 文件被关 | 已修：v2.0 TraceLogger 自动滚动 session |
| `with_structured_output(method="function_calling")` 在 thinking 模型上 400 | 已修：自动降级 `tool_choice="auto"` + 注入 prompt |
| DeepSeek V4 thinking 多轮调用 400 `reasoning_content must be passed back` | 已修：adapter 自动回写 `reasoning_content` |
| EditTool 改重复字符串报 `INTERNAL_ERROR` 而非 `INVALID_PARAM` | 已修：`ToolResponse.error(context=...)` 而非 `data=` |
| Python 3.14 跑 calculator 挂在 `ast.Num` | 已修：改用 `ast.Constant` 主路径 + `hasattr` 兜底 |
| 工具输出过长爆 token | `ObservationTruncator` 自动截断 + 完整结果落盘 `tool-output/` |
| 不知道 ReAct 在干什么 | `TraceLogger().enable()` 看 JSONL/HTML 时间线 |
| RAG 检索质量差 | 开启 `enable_mqe=True, enable_hyde=True` 多查询扩展 |
| 多 agent supervisor 死循环 | `build_supervisor_graph(max_handoffs=N)` 限次数 |

---

## 11. 后续阅读路径

按角色推荐入口：

- **想快速写一个 demo** → [`quickstart.md`](quickstart.md) → [`tool-system.md`](tool-system.md)
- **想深入图编排** → [`graph-architecture.md`](graph-architecture.md) → [`hitl.md`](hitl.md)
- **想做严格输出** → [`structured-output.md`](structured-output.md)
- **想做 RAG 系统** → [`rag-guide.md`](rag-guide.md)
- **想做多智能体** → [`multi-agent.md`](multi-agent.md) → [`subagent-guide.md`](subagent-guide.md)
- **想接 Claude Desktop / Cursor** → [`mcp.md`](mcp.md)
- **想做 Eval** → [`eval-harness.md`](eval-harness.md)
- **想做长会话** → [`context-engineering.md`](context-engineering.md) → [`memory-guide.md`](memory-guide.md)
- **想观测性** → [`observability.md`](observability.md)
- **想异步/流式** → [`async-streaming.md`](async-streaming.md)
- **想发布到 PyPI** → [`pypi-release.md`](pypi-release.md)
- **想做 Skills 知识包** → [`skills.md`](skills.md)

---

## 12. 版本与许可

- **当前版本**：v2.0.0（含 thinking 模型协议、Python 3.14 兼容、TraceLogger session 滚动等修复）
- **License**：CC BY-NC-SA 4.0（学习/研究/分享，**禁止商用**）
- **Repo**：https://github.com/Perlou/clear-agent
- **PyPI**：https://pypi.org/project/clear-agent/

---

> 这份文档是 ClearAgent 的"地图"，不是"GPS"。看完之后请按"后续阅读路径"切到具体专题文档拿细节。
