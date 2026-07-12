"""MCP stdio and authenticated stateless Streamable HTTP transports."""

from __future__ import annotations

import contextlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from threading import Lock
from typing import Any, Dict

import jwt

from .tools import bind_tool_methods, content_result, list_tools

SERVER_INFO = {"name": "geologic-memory", "version": "0.1.0"}
PROTOCOL_VERSION = "2025-11-25"
MAX_REQUEST_BYTES = 1_048_576
USER_STORAGE_TOOLS = {"load_session", "save_session", "load_user_memory", "save_user_memory"}
ACTIVE_USER_TOOLS = {
    "clear_memory", "ingest_turns", "retrieve_facts", "answer_question",
    "save_session", "save_user_memory",
}


def json_error(message: str, *, code: int = -32000, data: Any = None) -> Dict[str, Any]:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def initialize_result() -> Dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": SERVER_INFO,
    }


def handle_request(request: Dict[str, Any], methods) -> Dict[str, Any] | None:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        raise ValueError("Request must be a JSON-RPC 2.0 object")
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        result = initialize_result()
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in methods:
            raise KeyError(f"Unknown tool: {name}")
        with contextlib.redirect_stdout(sys.stderr):
            value = methods[name](**arguments)
        result = content_result(value)
    elif method == "notifications/initialized":
        return None
    else:
        raise KeyError(f"Unknown method: {method}")

    if "id" not in request:
        return None
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def request_error(request, exc: Exception) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request.get("id") if isinstance(request, dict) else None,
        "error": json_error(str(exc)),
    }


def run_stdio_server(service) -> None:
    methods = bind_tool_methods(service)
    request: Dict[str, Any] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request, methods)
            if response is None:
                continue
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None, "error": json_error("Parse error", code=-32700)}
        except Exception as exc:
            response = request_error(request, exc)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def decode_access_token(token, *, secret, issuer, audience):
    if not secret or len(secret.encode("utf-8")) < 32:
        raise ValueError("MCP_JWT_SECRET must contain at least 32 bytes")
    claims = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        issuer=issuer,
        audience=audience,
        options={"require": ["exp", "iat", "iss", "aud", "sub", "user_id", "scope"]},
    )
    scopes = set(str(claims["scope"]).split())
    if "mcp" not in scopes:
        raise jwt.InvalidTokenError("Required mcp scope is missing")
    return claims


def authorize_tool_request(request, claims, service):
    """Bind HTTP tool calls to the authenticated JWT tenant."""
    if request.get("method") != "tools/call":
        return
    params = request.get("params") or {}
    name = params.get("name")
    arguments = params.setdefault("arguments", {})
    user_id = claims["user_id"]
    scopes = set(str(claims["scope"]).split())

    if name == "configure_memory" and "admin" not in scopes:
        raise PermissionError("The admin scope is required for configure_memory")
    if name in USER_STORAGE_TOOLS:
        supplied = arguments.get("user_id")
        if supplied is not None and supplied != user_id:
            raise PermissionError("JWT user_id does not match requested user_id")
        arguments["user_id"] = user_id
    if name in ACTIVE_USER_TOOLS and service.active_user_id != user_id:
        raise PermissionError("Call load_user_memory before accessing this user's active memory")


def create_http_handler(
    service,
    *,
    endpoint: str,
    jwt_secret: str,
    jwt_issuer: str,
    jwt_audience: str,
    allowed_origins=(),
):
    """Create a stateless Streamable HTTP handler for one service instance."""
    methods = bind_tool_methods(service)
    service_lock = Lock()
    origins = frozenset(allowed_origins)

    class StreamableHTTPHandler(BaseHTTPRequestHandler):
        server_version = "GeoLogicMemoryMCP/0.1"

        def log_message(self, fmt, *args):
            print(f"[http] {self.address_string()} {fmt % args}", file=sys.stderr)

        def _send_json(self, status, value, extra_headers=None):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, status, message, headers=None):
            self._send_json(status, {"jsonrpc": "2.0", "id": None, "error": json_error(message)}, headers)

        def _authorize(self):
            origin = self.headers.get("Origin")
            if origin and origin not in origins:
                self._reject(HTTPStatus.FORBIDDEN, "Origin is not allowed")
                return None
            authorization = self.headers.get("Authorization", "")
            scheme, separator, token = authorization.partition(" ")
            if not separator or scheme.lower() != "bearer" or not token:
                self._reject(
                    HTTPStatus.UNAUTHORIZED,
                    "Missing or invalid bearer JWT",
                    {"WWW-Authenticate": 'Bearer realm="geologic-memory-mcp", error="invalid_token"'},
                )
                return None
            try:
                return decode_access_token(token, secret=jwt_secret, issuer=jwt_issuer, audience=jwt_audience)
            except (jwt.PyJWTError, ValueError):
                self._reject(
                    HTTPStatus.UNAUTHORIZED,
                    "Missing, expired, or invalid bearer JWT",
                    {"WWW-Authenticate": 'Bearer realm="geologic-memory-mcp", error="invalid_token"'},
                )
                return None

        def _valid_endpoint(self):
            if self.path != endpoint:
                self._reject(HTTPStatus.NOT_FOUND, "MCP endpoint not found")
                return False
            return True

        def do_POST(self):
            if not self._valid_endpoint():
                return
            claims = self._authorize()
            if claims is None:
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._reject(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._reject(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length < 1 or length > MAX_REQUEST_BYTES:
                self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request body size")
                return
            try:
                request = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reject(HTTPStatus.BAD_REQUEST, "Parse error")
                return
            if isinstance(request, list):
                self._reject(HTTPStatus.BAD_REQUEST, "JSON-RPC batches are not supported")
                return
            if request.get("method") != "initialize" and self.headers.get("MCP-Protocol-Version") != PROTOCOL_VERSION:
                self._reject(HTTPStatus.BAD_REQUEST, f"MCP-Protocol-Version must be {PROTOCOL_VERSION}")
                return
            try:
                with service_lock:
                    authorize_tool_request(request, claims, service)
                    response = handle_request(request, methods)
            except PermissionError as exc:
                self._reject(HTTPStatus.FORBIDDEN, str(exc))
                return
            except Exception as exc:
                response = request_error(request, exc)
            if response is None:
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send_json(HTTPStatus.OK, response)

        def do_GET(self):
            if not self._valid_endpoint() or self._authorize() is None:
                return
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_DELETE(self):
            self.do_GET()

    return StreamableHTTPHandler


def run_streamable_http_server(
    service,
    *,
    host: str,
    port: int,
    endpoint: str = "/mcp",
    jwt_secret: str,
    jwt_issuer: str,
    jwt_audience: str,
    allowed_origins=(),
) -> None:
    if not jwt_secret or len(jwt_secret.encode("utf-8")) < 32:
        raise ValueError("MCP_JWT_SECRET must contain at least 32 bytes")
    if not endpoint.startswith("/") or "?" in endpoint or "#" in endpoint:
        raise ValueError("HTTP endpoint must be an absolute path such as /mcp")
    handler = create_http_handler(
        service,
        endpoint=endpoint,
        jwt_secret=jwt_secret,
        jwt_issuer=jwt_issuer,
        jwt_audience=jwt_audience,
        allowed_origins=allowed_origins,
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"GeoLogic Memory MCP listening on http://{host}:{port}{endpoint}", file=sys.stderr)
    server.serve_forever()
