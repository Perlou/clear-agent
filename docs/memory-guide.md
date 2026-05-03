# Memory 指南

ClearAgent 提供两层记忆 + 一层管理器：

| 层 | 类 | 用途 |
|---|---|---|
| 短期 | `WorkingMemory` | 会话级、容量+TTL+token 限制、TF-IDF 检索 |
| 长期 | `SemanticMemory` | 嵌入向量 + 内存知识图谱（Entity/Relation） |
| 协调 | `MemoryManager` | 注册多子系统、按 type 路由、跨子系统聚合检索 |

> **重要决策**：ClearAgent **不引入 Neo4j**（plan §07 §2.3）。SemanticMemory 的图谱完全在内存里（`self.entities` dict + `self.relations` list），重启即失。需要持久化图谱用 Qdrant payload 索引或自行接入图数据库。

## 1. 安装

```bash
pip install clear-agent[memory]              # sklearn (TF-IDF) + spacy (NER)
python -m spacy download zh_core_web_sm      # 中文 NER 模型（可选）
python -m spacy download en_core_web_sm      # 英文 NER 模型（可选）
pip install clear-agent[retrieval-qdrant]    # SemanticMemory 需要 Qdrant
```

## 2. WorkingMemory（短期会话级）

```python
from datetime import datetime
from clear_agent.memory import WorkingMemory, MemoryConfig, MemoryItem

cfg = MemoryConfig(
    working_memory_capacity=20,         # 最多 20 条
    working_memory_tokens=4000,         # token 上限
    working_memory_ttl_minutes=60,      # 60 分钟过期
)
wm = WorkingMemory(cfg)

# 添加
wm.add(MemoryItem(
    id="m1", content="用户问了如何调试", memory_type="working",
    user_id="alice", timestamp=datetime.now(), importance=0.7,
))

# 检索（TF-IDF + 关键词匹配 + 时间衰减 + 重要性加权）
hits = wm.retrieve("调试", limit=5, user_id="alice")

# 上下文摘要（按重要性 / 时间 倒序）
ctx = wm.get_context_summary(max_length=500)

# 遗忘
wm.forget(strategy="importance_based", threshold=0.2)
wm.forget(strategy="time_based", max_age_days=1)
wm.forget(strategy="capacity_based")

# 统计
print(wm.get_stats())   # count / capacity_usage / token_usage / avg_importance ...
```

## 3. SemanticMemory（长期向量+图谱）

```python
from clear_agent.memory import SemanticMemory, MemoryConfig, MemoryItem
from clear_agent.retrieval.embeddings import get_text_embedder
from clear_agent.retrieval.storage.qdrant_store import QdrantVectorStore

cfg = MemoryConfig()

# 全部组件可注入（便于测试 / 自定义 backend）
embedder = get_text_embedder()                      # 缺省全局单例
store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name="my_semantic",
    vector_size=embedder.dimension,
)

sm = SemanticMemory(
    config=cfg,
    embedding_model=embedder,
    vector_store=store,
    nlp=None,    # None 走 fallback；传入 spacy.load(...) 启用 NER
)

# 添加（自动嵌入 + 实体/关系提取 + Qdrant 写入）
item = MemoryItem(
    id="s1", content="Alice 在 OpenAI 工作", memory_type="semantic",
    user_id="default", timestamp=datetime.now(), importance=0.8,
)
sm.add(item)

# 混合检索（向量 + 内存图）
hits = sm.retrieve("Alice 在哪？", limit=5, user_id="default")
for h in hits:
    print(h.metadata["combined_score"], h.metadata["probability"], h.content)

# 实体查询
e = sm.get_entity("entity_xxx")
related = sm.get_related_entities("entity_xxx", max_hops=2)
matches = sm.search_entities("Alice", limit=10)

# 导出图谱
graph = sm.export_knowledge_graph()
# {"entities": {...}, "relations": [...], "graph_stats": {...}}
```

## 4. MemoryManager（多子系统协调）

```python
from clear_agent.memory import MemoryManager, WorkingMemory, SemanticMemory, MemoryConfig

mgr = MemoryManager()
mgr.register("working", WorkingMemory(MemoryConfig()))
mgr.register("semantic", SemanticMemory(
    config=MemoryConfig(),
    embedding_model=embedder,
    vector_store=store,
))

# 按 memory_type 自动路由
mgr.add(MemoryItem(id="w1", memory_type="working", ...))   # → WorkingMemory
mgr.add(MemoryItem(id="s1", memory_type="semantic", ...))  # → SemanticMemory

# 跨子系统检索（按 importance 合并去重）
hits = mgr.retrieve("query", limit=10)

# 限定子系统
hits = mgr.retrieve("query", memory_types=["semantic"])

# 显式指定 type 更新 / 删除
mgr.update("w1", memory_type="working", importance=0.9)
mgr.remove("s1", memory_type="semantic")

# 任一子系统含 id 即 True
mgr.has_memory("w1")

# 选择性清空
mgr.clear(memory_types=["working"])

# 聚合统计
print(mgr.get_stats())
# {
#   "total_count": 42,
#   "by_type": {"working": {...}, "semantic": {...}},
#   "registered_types": ["working", "semantic"]
# }
```

## 5. 在 graph 节点里用 Memory

```python
from clear_agent.core.graph import StateGraph, START, END
from clear_agent.memory import MemoryManager, WorkingMemory, MemoryConfig, MemoryItem

mgr = MemoryManager()
mgr.register("working", WorkingMemory(MemoryConfig()))

def memory_aware_node(state):
    # 1. 检索相关历史
    history_hits = mgr.retrieve(state["question"], memory_types=["working"], limit=5)
    history_text = "\n".join(h.content for h in history_hits)

    # 2. 调 LLM 生成
    response = llm.invoke([
        {"role": "system", "content": f"相关历史:\n{history_text}"},
        {"role": "user", "content": state["question"]},
    ])

    # 3. 把这一轮存到 working memory
    mgr.add(MemoryItem(
        id=state["thread_id"] + "-" + str(state["step"]),
        content=f"Q: {state['question']}\nA: {response.content}",
        memory_type="working",
        user_id=state.get("user_id", "default"),
        timestamp=datetime.now(),
        importance=0.6,
    ))

    return {"answer": response.content}
```

## 6. 自定义 Memory 子类

```python
from clear_agent.memory import BaseMemory, MemoryItem
from typing import Any, Dict, List

class RedisMemory(BaseMemory):
    """例：基于 Redis 的记忆实现"""

    def add(self, memory_item: MemoryItem) -> str:
        # 实现你的存储
        ...

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        ...

    # 实现剩下 5 个抽象方法...
```

注册到 manager 即可：
```python
mgr.register("redis", RedisMemory(config))
```

## 7. forget 策略对比

| 策略 | WorkingMemory | SemanticMemory |
|---|---|---|
| `importance_based` | 删除 importance < threshold；同时执行 TTL | 同 |
| `time_based` | 删除 timestamp < now-max_age_days；同时执行 TTL | 同 |
| `capacity_based` | 超过 capacity 时按优先级（重要性×时间衰减）删最低 | 超过 max_capacity 时按 importance 删最低 |

## 8. 常见坑

- **`spacy.load('zh_core_web_sm')` 报错** → 模型没下载：`python -m spacy download zh_core_web_sm`
- **`SemanticMemory.retrieve` 返回空** → 检查 Qdrant 集合是否存在；维度是否匹配；user_id 过滤是否过严
- **WorkingMemory 死循环** → `forget` 默认会先 TTL 过期；如果 timestamp 全是「未来」时间，TTL 不会触发
- **MemoryManager 重复注册** → `register` 抛 ValueError；先 `unregister` 再 `register`
- **跨进程共享** → WorkingMemory 是纯内存的，不持久化；SemanticMemory 走 Qdrant 持久化但内存图谱重启即失
