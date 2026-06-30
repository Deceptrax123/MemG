"""Minimal stdio JSON-RPC transport for MCP tool calls."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict

from .tools import bind_tool_methods, content_result, list_tools

SERVER_INFO = {"name": "geologic-memory", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def json_error(message: str, *, code: int = -32000, data: Any = None) -> Dict[str, Any]:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def initialize_result() -> Dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": SERVER_INFO,
    }


def handle_request(request: Dict[str, Any], methods) -> Dict[str, Any] | None:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        result = initialize_result()
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in methods:
            raise KeyError(f"Unknown tool: {name}")
        result = content_result(methods[name](**arguments))
    elif method == "notifications/initialized":
        return None
    else:
        raise KeyError(f"Unknown method: {method}")

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def run_stdio_server(service) -> None:
    methods = bind_tool_methods(service)
    request: Dict[str, Any] | None = None

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request, methods)
            if response is None:
                continue
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if request else None,
                "error": json_error(str(exc), data=traceback.format_exc()),
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
