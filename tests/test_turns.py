from datetime import datetime

import pytest

from mcp.turns import normalize_chat_turns


def test_normalizes_chat_turns():
    turns = normalize_chat_turns([
        {"role": "user", "name": " Ada ", "content": " Hello ", "timestamp": "2026-01-02T03:04:05Z"},
        {"role": "assistant", "content": "Hi"},
    ])
    assert turns[0] == {"speaker": "Ada", "date_time": "2026-01-02T03:04:05Z", "text": "Hello"}
    assert turns[1]["speaker"] == "assistant"
    datetime.fromisoformat(turns[1]["date_time"])


@pytest.mark.parametrize("turns", [[], [{"role": "system", "content": "x"}], [{"role": "user", "content": " "}]])
def test_rejects_invalid_turns(turns):
    with pytest.raises(ValueError):
        normalize_chat_turns(turns)


def test_rejects_invalid_timestamp():
    with pytest.raises(ValueError, match="valid ISO 8601"):
        normalize_chat_turns([{"role": "user", "content": "x", "timestamp": "yesterday"}])
