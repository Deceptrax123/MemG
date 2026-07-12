from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest

from mcp.token import mint_token
from mcp.transport import authorize_tool_request, decode_access_token

SECRET = "a-secure-test-secret-that-is-longer-than-32-bytes"


def claims_for(user="alice", admin=False):
    token = mint_token(
        secret=SECRET,
        subject="test-client",
        user_id=user,
        issuer="issuer",
        audience="audience",
        expires_minutes=5,
        admin=admin,
    )
    return decode_access_token(token, secret=SECRET, issuer="issuer", audience="audience")


def tool_request(name, arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


def test_jwt_has_validated_identity_and_scope():
    claims = claims_for()
    assert claims["sub"] == "test-client"
    assert claims["user_id"] == "alice"
    assert "mcp" in claims["scope"]


def test_expired_jwt_is_rejected():
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": "client", "user_id": "alice", "iss": "issuer", "aud": "audience",
         "iat": now - timedelta(minutes=2), "exp": now - timedelta(minutes=1), "scope": "mcp"},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, secret=SECRET, issuer="issuer", audience="audience")


def test_user_id_is_injected_and_cross_tenant_request_rejected():
    service = SimpleNamespace(active_user_id=None)
    request = tool_request("load_user_memory")
    authorize_tool_request(request, claims_for("alice"), service)
    assert request["params"]["arguments"]["user_id"] == "alice"

    with pytest.raises(PermissionError, match="does not match"):
        authorize_tool_request(
            tool_request("load_user_memory", {"user_id": "bob"}),
            claims_for("alice"),
            service,
        )


def test_active_memory_must_match_jwt_user():
    service = SimpleNamespace(active_user_id="bob")
    with pytest.raises(PermissionError, match="load_user_memory"):
        authorize_tool_request(tool_request("retrieve_facts", {"query": "x"}), claims_for("alice"), service)


def test_configure_requires_admin_scope():
    service = SimpleNamespace(active_user_id=None)
    with pytest.raises(PermissionError, match="admin"):
        authorize_tool_request(tool_request("configure_memory"), claims_for(), service)
    authorize_tool_request(tool_request("configure_memory"), claims_for(admin=True), service)
