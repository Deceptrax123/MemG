"""Normalize chat-completion-style turns for the memory extractor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

CHAT_ROLES = {"user", "assistant"}


def normalize_chat_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert role/content chat turns to the extractor's internal format."""
    if not turns:
        raise ValueError("turns must contain at least one chat message")

    base_time = datetime.now(timezone.utc)
    normalized = []
    for index, turn in enumerate(turns):
        role = turn.get("role")
        content = turn.get("content")
        if role not in CHAT_ROLES:
            raise ValueError(
                f"turns[{index}].role must be 'user' or 'assistant'"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"turns[{index}].content must be a non-empty string")

        timestamp = turn.get("timestamp")
        if timestamp is None:
            timestamp = (base_time + timedelta(microseconds=index)).isoformat()
        elif not isinstance(timestamp, str):
            raise ValueError(f"turns[{index}].timestamp must be an ISO 8601 string")
        else:
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"turns[{index}].timestamp must be a valid ISO 8601 string") from exc

        name = turn.get("name")
        speaker = name.strip() if isinstance(name, str) and name.strip() else role
        normalized.append(
            {"speaker": speaker, "date_time": timestamp, "text": content.strip()}
        )

    return normalized


__all__ = ["CHAT_ROLES", "normalize_chat_turns"]

