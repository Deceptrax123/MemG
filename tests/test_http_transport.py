import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from mcp.token import mint_token
from mcp.transport import create_http_handler


class Service:
    def status(self):
        return {"loaded": False}

    def __getattr__(self, name):
        return lambda **kwargs: {}


SECRET = "a-secure-http-test-secret-that-is-longer-than-32-bytes"

def access_token(user_id="alice"):
    return mint_token(secret=SECRET, subject="test", user_id=user_id, issuer="issuer", audience="audience")


def running_server(*, origins=()):
    handler = create_http_handler(
        Service(), endpoint="/mcp", jwt_secret=SECRET, jwt_issuer="issuer", jwt_audience="audience", allowed_origins=origins
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def request(server, body, *, token=None, origin=None):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if origin is not None:
        headers["Origin"] = origin
    if body.get("method") != "initialize":
        headers["MCP-Protocol-Version"] = "2025-11-25"
    connection.request("POST", "/mcp", json.dumps(body), headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.headers, payload


def test_http_requires_bearer_token():
    server, thread = running_server()
    try:
        status, headers, payload = request(
            server, {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        assert status == 401
        assert headers["WWW-Authenticate"].startswith("Bearer")
    finally:
        server.shutdown()
        thread.join()


def test_http_initialize_and_status():
    server, thread = running_server()
    try:
        status, _, payload = request(
            server,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            token=access_token(),
        )
        assert status == 200
        assert json.loads(payload)["result"]["protocolVersion"] == "2025-11-25"

        status, _, payload = request(
            server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "status", "arguments": {}},
            },
            token=access_token(),
        )
        assert status == 200
        tool_result = json.loads(payload)["result"]["content"][0]["text"]
        assert json.loads(tool_result) == {"loaded": False}
    finally:
        server.shutdown()
        thread.join()


def test_http_rejects_unlisted_origin():
    server, thread = running_server(origins=("https://allowed.example",))
    try:
        status, _, _ = request(
            server,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            token=access_token(),
            origin="https://evil.example",
        )
        assert status == 403
    finally:
        server.shutdown()
        thread.join()


def test_http_accepts_notification_with_202():
    server, thread = running_server()
    try:
        status, _, payload = request(
            server,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            token=access_token(),
        )
        assert status == 202
        assert payload == b""
    finally:
        server.shutdown()
        thread.join()
