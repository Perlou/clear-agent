"""HITL 内置中断模式

提供 ``approval`` / ``edit_state`` / ``validate_tool_args`` 三个 helper，
都是 ``interrupt(payload)`` 的薄包装 + 标准化的 payload schema。


"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.interrupt import interrupt


# ==================== Approval ====================


def approval(
    prompt: str,
    options: Optional[List[str]] = None,
    *,
    default: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """请求人工审批

    在节点中调用：
        decision = approval("Send email to alice?", options=["approve", "reject"])
        if decision == "reject":
            return {...}

    Resume 端：
        compiled.resume(thread_id, value="approve")

    Args:
        prompt: 给用户的提示语
        options: 可选项列表（默认 ["approve", "reject"]）
        default: 默认选项（仅作为 hint，不会自动应用）
        metadata: 自定义扩展字段

    Returns:
        用户选择的字符串（来自 options 或自定义）
    """
    options = options or ["approve", "reject"]
    payload: Dict[str, Any] = {
        "type": "approval",
        "prompt": prompt,
        "options": options,
    }
    if default is not None:
        payload["default"] = default
    if metadata:
        payload["custom"] = metadata
    decision = interrupt(payload)

    # 兼容 dict 形式（{"choice": "approve"}）和直接字符串
    if isinstance(decision, dict) and "choice" in decision:
        return decision["choice"]
    return decision


# ==================== Edit State ====================


def edit_state(
    fields: List[str],
    *,
    prompt: str = "Review and edit fields if needed",
    current_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """请求人工编辑指定字段

    用法：
        edited = edit_state(["plan", "tone"], current_values={"plan": ..., "tone": ...})
        return {"plan": edited["plan"], "tone": edited["tone"]}

    Resume 端：
        compiled.resume(thread_id, value={"plan": "new plan...", "tone": "formal"})

    Args:
        fields: 允许编辑的字段名列表
        prompt: 给用户的提示语
        current_values: 当前值（供前端做 diff 展示）

    Returns:
        包含编辑后字段的字典；未编辑字段保留原值
    """
    payload: Dict[str, Any] = {
        "type": "edit",
        "prompt": prompt,
        "fields": list(fields),
    }
    if current_values is not None:
        payload["current_values"] = dict(current_values)

    edited = interrupt(payload)
    if not isinstance(edited, dict):
        raise ValueError(
            f"edit_state resume value 必须是 dict（拿到 {type(edited).__name__}）"
        )

    # 用 current_values 兜底缺失字段
    out: Dict[str, Any] = dict(current_values or {})
    for f in fields:
        if f in edited:
            out[f] = edited[f]
    return out


# ==================== Tool Validation ====================


def validate_tool_args(
    tool_name: str,
    proposed_args: Dict[str, Any],
    *,
    sensitive_fields: Optional[List[str]] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """请求人工复核工具参数（关键工具调用前的最后一道防线）

    用法：
        validated = validate_tool_args(
            "send_email",
            proposed_args={"to": "alice@example.com", "subject": "...", "body": "..."},
            sensitive_fields=["to", "subject"],
        )
        execute_tool("send_email", validated)

    Resume 端：
        # 接受原参数
        compiled.resume(thread_id, value={"approved": True})
        # 修改参数
        compiled.resume(thread_id, value={"approved": True, "args": {...修改后...}})
        # 拒绝
        compiled.resume(thread_id, value={"approved": False})

    Args:
        tool_name: 工具名（仅作展示）
        proposed_args: LLM 提议的工具参数
        sensitive_fields: 敏感字段名列表（前端高亮用）
        prompt: 自定义提示语

    Returns:
        校验后的工具参数（可能被用户修改）

    Raises:
        ValueError: 用户拒绝（approved=False）
    """
    payload: Dict[str, Any] = {
        "type": "tool_validation",
        "tool_name": tool_name,
        "proposed_args": dict(proposed_args),
    }
    if sensitive_fields:
        payload["sensitive_fields"] = list(sensitive_fields)
    if prompt:
        payload["prompt"] = prompt

    decision = interrupt(payload)
    if not isinstance(decision, dict):
        raise ValueError(
            f"validate_tool_args resume value 必须是 dict（拿到 {type(decision).__name__}）"
        )

    if not decision.get("approved", False):
        raise ValueError(
            f"用户拒绝了工具调用 {tool_name}: {decision.get('reason', '未提供原因')}"
        )

    # 用户改了参数？
    if "args" in decision and isinstance(decision["args"], dict):
        return decision["args"]
    return dict(proposed_args)


__all__ = [
    "approval",
    "edit_state",
    "validate_tool_args",
]
