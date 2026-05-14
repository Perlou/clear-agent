# Clear Agent Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the highest-risk correctness and security issues found in the repository scan.

**Architecture:** Keep the public APIs stable while tightening validation at boundaries. Each fix gets a regression test first, then the smallest production change needed to preserve current behavior and block the bug.

**Tech Stack:** Python 3.10+, pytest, pydantic, OpenAI-compatible tool schemas, Anthropic/Gemini adapters, Qdrant optional storage.

---

### Task 1: Checkpoint Path Safety

**Files:**
- Modify: `tests/test_checkpoint_roundtrip.py`
- Modify: `clear_agent/core/checkpoint.py`

- [ ] **Step 1: Write failing tests**

Add tests that store a `Checkpoint` with `thread_id="../escape"` and `id="../evil"`. The expected behavior is that JSON checkpoint files remain under the configured `base_dir`, and `get_tuple()` with unsafe IDs cannot read files outside that base.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_checkpoint_roundtrip.py -q`
Expected before implementation: a failure showing escaped files can be created or read.

- [ ] **Step 3: Implement minimal fix**

Encode checkpoint path segments with URL-safe base64 or another reversible safe encoding. Route every JSON backend read/write/list path through that helper, preserving legacy session loading.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_checkpoint_roundtrip.py -q`
Expected after implementation: all checkpoint roundtrip tests pass.

### Task 2: File Tool Root Confinement

**Files:**
- Modify: `tests/test_file_tools.py`
- Modify: `clear_agent/tools/builtin/file_tools.py`

- [ ] **Step 1: Write failing tests**

Add tests for `ReadTool`, `WriteTool`, `EditTool`, and `MultiEditTool` using absolute paths and `../` paths outside `project_root`. The expected behavior is a failed `ToolResponse` and no outside-file modification.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_file_tools.py -q`
Expected before implementation: outside paths are allowed.

- [ ] **Step 3: Implement minimal fix**

Resolve every requested path against `project_root`, reject paths outside that root, and keep existing optimistic-lock behavior unchanged.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_file_tools.py -q`
Expected after implementation: all file tool tests pass.

### Task 3: Provider Tool Schema Compatibility

**Files:**
- Modify: `tests/test_anthropic_gemini_async.py`
- Modify: `clear_agent/core/llm_adapters.py`

- [ ] **Step 1: Write failing tests**

Add Anthropic adapter tests showing OpenAI-style tools are converted to Anthropic `name/input_schema` tools and OpenAI tool messages are converted to Anthropic `tool_result` content blocks.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_anthropic_gemini_async.py -q`
Expected before implementation: raw OpenAI schemas/messages are passed through.

- [ ] **Step 3: Implement minimal fix**

Add Anthropic conversion helpers and apply them in sync and async tool-call paths.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_anthropic_gemini_async.py -q`
Expected after implementation: provider conversion tests pass.

### Task 4: Embedding and Qdrant Dimension Consistency

**Files:**
- Modify: `tests/test_embeddings.py`
- Modify: `tests/test_qdrant_store.py`
- Modify: `clear_agent/retrieval/embeddings.py`
- Modify: `clear_agent/retrieval/rag/pipeline.py`
- Modify: `clear_agent/retrieval/storage/qdrant_store.py`
- Modify: `clear_agent/memory/semantic.py`

- [ ] **Step 1: Write failing tests**

Add tests for fallback factory kwargs filtering, Qdrant connection manager keys including `vector_size`, and RAG/SemanticMemory using a supplied embedder dimension.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py tests/test_qdrant_store.py tests/test_semantic_memory.py -q`
Expected before implementation: fallback/type or identity assertions fail.

- [ ] **Step 3: Implement minimal fix**

Filter kwargs by provider, use `embedder.dimension` when an embedder is passed, and include vector configuration in the Qdrant singleton key.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_embeddings.py tests/test_qdrant_store.py tests/test_semantic_memory.py -q`
Expected after implementation: all targeted tests pass.

### Task 5: Runtime Semantics Guardrails

**Files:**
- Modify: `tests/test_hitl_interrupt.py`
- Modify: `tests/test_graph_basics.py`
- Modify: `tests/test_simple_graph.py`
- Modify: `clear_agent/core/graph.py`
- Modify: `clear_agent/agents/_simple_graph.py`
- Modify: `clear_agent/core/llm.py`
- Modify: `clear_agent/context/truncator.py`

- [ ] **Step 1: Write failing tests**

Add tests for HITL TTL expiry, multi-edge/list-route rejection, SimpleGraph reasoning echo, async tool-call fallback kwargs, stream temperature defaulting, and truncator byte limits.

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_hitl_interrupt.py tests/test_graph_basics.py tests/test_simple_graph.py tests/test_context_engineering.py tests/test_callbacks_parallel.py -q`
Expected before implementation: the new behavioral tests fail.

- [ ] **Step 3: Implement minimal fix**

Enforce advertised TTL, raise graph compile errors for unsupported fan-out, use LLM assistant serialization in SimpleGraph, preserve async fallback kwargs, apply default stream temperature, and enforce byte trimming after line trimming.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_hitl_interrupt.py tests/test_graph_basics.py tests/test_simple_graph.py tests/test_context_engineering.py tests/test_callbacks_parallel.py -q`
Expected after implementation: targeted tests pass.

### Task 6: Final Regression

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run full tests**

Run: `.venv/bin/python -m pytest -q`
Expected: full non-integration suite passes.

- [ ] **Step 2: Run compile check**

Run: `.venv/bin/python -m compileall -q clear_agent`
Expected: command exits successfully.

- [ ] **Step 3: Run mypy snapshot**

Run: `.venv/bin/python -m mypy clear_agent`
Expected: existing type debt may remain, but no newly introduced obvious type errors in changed areas.

### Task 7: Qdrant Existing Collection Validation

**Files:**
- Modify: `tests/test_qdrant_store.py`
- Modify: `clear_agent/retrieval/storage/qdrant_store.py`

- [ ] **Step 1: Write failing test**

Add a test where the fake Qdrant client reports an existing collection with vector size `384`, then construct `QdrantVectorStore(vector_size=768)`. The expected behavior is a `RetrievalException` before upserts/searches can fail later.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qdrant_store.py::test_existing_collection_vector_size_mismatch_raises -q`
Expected before implementation: the constructor succeeds.

- [ ] **Step 3: Implement minimal fix**

When `_ensure_collection()` sees an existing collection, inspect vector params size and distance if available. Raise `RetrievalException` on a vector size mismatch.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_qdrant_store.py -q`
Expected after implementation: all Qdrant store tests pass.

### Task 8: SemanticMemory Remove Graph Cleanup

**Files:**
- Modify: `tests/test_semantic_memory.py`
- Modify: `clear_agent/memory/semantic.py`

- [ ] **Step 1: Write failing test**

Add a memory with unique entities and relations, remove it, and assert those now-orphaned entities and relations are removed from the in-memory graph.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_semantic_memory.py::test_remove_cleans_orphan_entities_and_relations -q`
Expected before implementation: entities/relations remain.

- [ ] **Step 3: Implement minimal fix**

After local memory removal, recompute the set of still referenced entity ids and drop orphan entities and relations.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_semantic_memory.py -q`
Expected after implementation: all SemanticMemory tests pass.

### Task 9: File Edit Atomic Writes

**Files:**
- Modify: `tests/test_file_tools.py`
- Modify: `clear_agent/tools/builtin/file_tools.py`

- [ ] **Step 1: Write failing tests**

Patch `os.replace` to fail during `EditTool` and `MultiEditTool` writes, then assert original file content remains intact.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_file_tools.py::TestEditTool::test_edit_write_failure_preserves_original tests/test_file_tools.py::TestMultiEditTool::test_multiedit_write_failure_preserves_original -q`
Expected before implementation: tests fail or reveal direct write behavior.

- [ ] **Step 3: Implement minimal fix**

Write edited content to a temporary sibling file and `os.replace()` it into place. Clean up temp files on failure.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_file_tools.py -q`
Expected after implementation: all file tool tests pass.

### Task 10: Documentation and Local Type Cleanup

**Files:**
- Modify: `docs/hitl.md`
- Modify: `docs/graph-architecture.md`
- Modify: touched implementation files where local type fixes are small and behavior-neutral.

- [ ] **Step 1: Sync docs**

Document that HITL interrupt TTL defaults to 86400 seconds and that current StateGraph does not support fan-out list routes.

- [ ] **Step 2: Run docs-free validation**

Run: `.venv/bin/python -m pytest tests/test_hitl_interrupt.py tests/test_graph_basics.py -q`
Expected: tests still pass.

- [ ] **Step 3: Local mypy cleanup**

Fix small type errors introduced or exposed in the touched modules only, without starting a whole-repo type migration.

- [ ] **Step 4: Final verification**

Run: `.venv/bin/python -m pytest -q`, `.venv/bin/python -m compileall -q clear_agent`, `git diff --check`, and `.venv/bin/python -m mypy clear_agent`.
