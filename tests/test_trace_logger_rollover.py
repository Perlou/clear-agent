"""TraceLogger 在 Agent 单例多次 run 时的会话滚动测试

历史 bug：
- ``Agent.__init__`` 创建一个 trace_logger 实例，open jsonl/html 文件
- ``agent.run()`` 结尾 finalize() 关闭文件
- 同一 agent 实例第二次 ``run()`` 写已关闭文件 → ``ValueError: I/O operation on closed file``

修复：finalize 后下次 log_event 自动滚动到新 session。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clear_agent.observability.trace_logger import TraceLogger


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestRollOverAfterFinalize:
    def test_log_event_after_finalize_starts_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)

            # session 1
            session1_id = logger.session_id
            session1_jsonl = logger.jsonl_path
            logger.log_event("session_start", {"agent_name": "X"})
            logger.finalize()

            # session 2 —— 这里曾经会抛 ValueError
            logger.log_event("session_start", {"agent_name": "X-second-run"})

            session2_id = logger.session_id
            session2_jsonl = logger.jsonl_path

            assert session2_id != session1_id, "应该生成新的 session_id"
            assert session2_jsonl != session1_jsonl, "应该写到新文件"
            assert session2_jsonl.exists()

            # session 1 的内容仍在
            events1 = _read_jsonl(session1_jsonl)
            assert any(e["payload"].get("agent_name") == "X" for e in events1)

            # session 2 的内容只有自己的事件
            events2 = _read_jsonl(session2_jsonl)
            assert len(events2) == 1
            assert events2[0]["payload"]["agent_name"] == "X-second-run"

    def test_double_finalize_is_safe(self):
        """重复 finalize 不应该崩 / 重复关文件"""
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)
            logger.log_event("e", {})
            logger.finalize()
            # 第二次应该 no-op，不抛
            logger.finalize()

    def test_three_sessions_rollover(self):
        """连续滚 3 个 session 都正常"""
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)
            session_ids = []
            for i in range(3):
                logger.log_event("session_start", {"run": i})
                logger.log_event("session_end", {"run": i})
                session_ids.append(logger.session_id)
                logger.finalize()
            assert len(set(session_ids)) == 3, "三次会话 id 全部不同"
            # 每个 session 的 JSONL 都存在
            for sid in session_ids:
                p = Path(tmp) / f"trace-{sid}.jsonl"
                assert p.exists()
                assert len(_read_jsonl(p)) == 2

    def test_html_file_also_rolls_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)
            html1 = logger.html_path
            logger.finalize()
            logger.log_event("e", {})
            html2 = logger.html_path
            assert html1 != html2
            assert html1.exists() and html2.exists()


class TestBackwardCompatibility:
    def test_single_session_unchanged(self):
        """单 session 用法行为不变"""
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)
            logger.log_event("session_start", {"agent_name": "Test"})
            logger.log_event("tool_call", {"tool_name": "Calc"}, step=1)
            logger.finalize()

            events = _read_jsonl(logger.jsonl_path)
            assert len(events) == 2
            assert events[0]["event"] == "session_start"
            assert events[1]["event"] == "tool_call"
            assert events[1]["step"] == 1

    def test_finalized_flag_initially_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TraceLogger(output_dir=tmp)
            assert logger._finalized is False
            logger.finalize()
            assert logger._finalized is True
