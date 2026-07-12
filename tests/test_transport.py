import json

import pytest

from mcp.transport import handle_request, initialize_result
from mcp.tools import TOOL_SCHEMAS, content_result


def test_initialize_and_tool_listing():
    initialized = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, {})
    assert initialized["result"] == initialize_result()
    listed = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, {})
    assert {tool["name"] for tool in listed["result"]["tools"]} == set(TOOL_SCHEMAS)


def test_tool_call_and_json_content():
    response = handle_request(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "status", "arguments": {}}},
        {"status": lambda: {"loaded": False}},
    )
    assert json.loads(response["result"]["content"][0]["text"]) == {"loaded": False}


def test_rejects_non_json_rpc_request():
    with pytest.raises(ValueError, match="JSON-RPC 2.0"):
        handle_request({"id": 1, "method": "initialize"}, {})


def test_content_result_serializes_unknown_values():
    assert "object" in content_result(object())["content"][0]["text"]
