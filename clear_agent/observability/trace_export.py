"""TraceLogger 训练数据导出 —— SFT / DPO

按 plan §三 RC 阶段补：把 TraceLogger 写出的 JSONL trace 文件转换为
**训练数据**格式，让用户用独立的 ``trl`` / ``axolotl`` 训练脚本喂数据，
ClearAgent 框架本身不长 RL 模块的肉。

提供三种导出：

- ``export_to_sft_jsonl(trace_path, out_path)`` —— 把成功完成的 agent run
  转为 SFT 格式（``{"messages": [...]}`` 每行一个）
- ``export_to_dpo_pairs(pass_traces, fail_traces, out_path)`` —— 同 prompt
  下成功/失败的回答配对成 DPO 偏好数据
- ``read_trace_events(trace_path)`` —— 通用 JSONL 流式迭代器

设计原则：
- **零 RL 依赖** —— 仅用 stdlib（json + pathlib），用户拿到 jsonl 后自己接训练
- 容错：单条 JSONL 解析失败时跳过 + warning，不中断整批
- 与 ``TraceLogger`` 输出的 ``log_event`` 格式约定对齐
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


logger = logging.getLogger(__name__)


# ==================== 通用读取 ====================


def read_trace_events(trace_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """流式读取一个 trace JSONL 文件，逐行 yield 事件 dict

    解析失败的行会被 warning 跳过。
    """
    p = Path(trace_path)
    if not p.exists():
        raise FileNotFoundError(f"Trace 文件不存在：{p}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ {p}:{line_no} JSON 解析失败（跳过）: {e}")


# ==================== SFT 导出 ====================


def _extract_messages_from_trace(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从 trace 事件序列拼出 OpenAI ``messages`` 格式

    依次检查事件 ``event_type`` 字段：
    - ``"user_input"`` → user message
    - ``"llm_response"`` / ``"agent_response"`` → assistant message
    - ``"tool_call"`` / ``"tool_result"`` → 跳过（SFT 一般只要主对话）
    """
    messages: List[Dict[str, Any]] = []
    for ev in events:
        et = ev.get("event_type") or ev.get("type")
        if et == "user_input":
            content = (
                ev.get("data", {}).get("content")
                or ev.get("data", {}).get("input")
                or ev.get("input")
                or ""
            )
            messages.append({"role": "user", "content": str(content)})
        elif et in ("llm_response", "agent_response", "final_answer"):
            content = (
                ev.get("data", {}).get("content")
                or ev.get("data", {}).get("output")
                or ev.get("output")
                or ev.get("content")
                or ""
            )
            if content:
                messages.append({"role": "assistant", "content": str(content)})
    return messages


def export_to_sft_jsonl(
    trace_path: Union[str, Path],
    out_path: Union[str, Path],
    only_successful: bool = True,
    min_messages: int = 2,
) -> int:
    """把单个 trace JSONL 转为 SFT 训练 JSONL

    Args:
        trace_path: 输入 trace 文件
        out_path: 输出 JSONL 路径（每行一个 ``{"messages": [...]}`` 样本）
        only_successful: True 时跳过含 ``error`` / ``failed`` 事件的会话
        min_messages: 至少 N 条消息才写出（默认 2 = 一对 Q/A）

    Returns:
        实际写出的样本数
    """
    events = list(read_trace_events(trace_path))
    if not events:
        return 0

    # 简单成功判定：是否有 event_type=error 或 status=failed
    if only_successful:
        for ev in events:
            et = ev.get("event_type") or ev.get("type")
            status = (ev.get("data", {}) or {}).get("status") or ev.get("status")
            if et == "error" or status in ("failed", "error"):
                logger.info(f"[skip] trace 含错误事件，跳过：{trace_path}")
                return 0

    messages = _extract_messages_from_trace(events)
    if len(messages) < min_messages:
        return 0

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = {"messages": messages, "source": str(trace_path)}
    # 追加模式以支持批量
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return 1


def export_traces_to_sft_jsonl(
    trace_paths: List[Union[str, Path]],
    out_path: Union[str, Path],
    only_successful: bool = True,
    min_messages: int = 2,
    overwrite: bool = True,
) -> int:
    """批量导出多个 trace 文件为单个 SFT JSONL

    Returns:
        累计写出的样本数
    """
    out = Path(out_path)
    if overwrite and out.exists():
        out.unlink()
    n = 0
    for p in trace_paths:
        try:
            n += export_to_sft_jsonl(
                p, out, only_successful=only_successful, min_messages=min_messages
            )
        except Exception as e:
            logger.warning(f"⚠️ 导出 {p} 失败: {e}")
    return n


# ==================== DPO 导出 ====================


def _trace_to_prompt_and_response(
    events: List[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """从 events 抽出 (prompt, response) 对（用于 DPO 偏好对）"""
    messages = _extract_messages_from_trace(events)
    if len(messages) < 2:
        return None
    # 以最后一条 user 之前的为 prompt 上下文，最后一条 assistant 为 response
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            user_idx = i
            break
    if user_idx is None:
        return None
    asst_idx = None
    for j in range(user_idx + 1, len(messages)):
        if messages[j]["role"] == "assistant":
            asst_idx = j
            break
    if asst_idx is None:
        return None
    prompt_text = messages[user_idx]["content"]
    response_text = messages[asst_idx]["content"]
    return {"prompt": prompt_text, "response": response_text}


def export_to_dpo_pairs(
    pass_traces: List[Union[str, Path]],
    fail_traces: List[Union[str, Path]],
    out_path: Union[str, Path],
    overwrite: bool = True,
) -> int:
    """成对的 pass/fail traces → DPO 偏好对 JSONL

    每行一个 ``{"prompt": ..., "chosen": ..., "rejected": ...}`` 样本，
    按数组顺序 1:1 配对（短数组截断）。

    Returns:
        实际写出的 DPO 对数
    """
    out = Path(out_path)
    if overwrite and out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    pairs = zip(pass_traces, fail_traces)  # 短的一方决定长度
    with out.open("a", encoding="utf-8") as f:
        for pp, fp in pairs:
            try:
                pass_pr = _trace_to_prompt_and_response(list(read_trace_events(pp)))
                fail_pr = _trace_to_prompt_and_response(list(read_trace_events(fp)))
            except FileNotFoundError as e:
                logger.warning(f"⚠️ trace 文件缺失（跳过）: {e}")
                continue

            if not pass_pr or not fail_pr:
                continue

            # prompt 不一致时仍以 pass 的 prompt 为准（DPO 实践通用做法）
            sample = {
                "prompt": pass_pr["prompt"],
                "chosen": pass_pr["response"],
                "rejected": fail_pr["response"],
                "source_pass": str(pp),
                "source_fail": str(fp),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1
    return n


__all__ = [
    "read_trace_events",
    "export_to_sft_jsonl",
    "export_traces_to_sft_jsonl",
    "export_to_dpo_pairs",
]
