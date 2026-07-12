"""Mint short-lived JWTs for trusted GeoLogic Memory clients."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import os

import jwt


def mint_token(*, secret, subject, user_id, issuer, audience, expires_minutes=60, admin=False):
    if not secret or len(secret.encode("utf-8")) < 32:
        raise ValueError("MCP_JWT_SECRET must contain at least 32 bytes")
    now = datetime.now(timezone.utc)
    scopes = ["mcp"]
    if admin:
        scopes.append("admin")
    return jwt.encode(
        {
            "sub": subject,
            "user_id": user_id,
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": now + timedelta(minutes=expires_minutes),
            "scope": " ".join(scopes),
        },
        secret,
        algorithm="HS256",
    )


def main():
    parser = argparse.ArgumentParser(description="Mint a GeoLogic Memory MCP JWT")
    parser.add_argument("--subject", required=True, help="Calling client or account identifier")
    parser.add_argument("--user-id", required=True, help="Memory tenant bound to this token")
    parser.add_argument("--expires-minutes", type=int, default=60)
    parser.add_argument("--issuer", default=os.getenv("MCP_JWT_ISSUER", "geologic-memory"))
    parser.add_argument("--audience", default=os.getenv("MCP_JWT_AUDIENCE", "geologic-memory-mcp"))
    parser.add_argument("--admin", action="store_true")
    args = parser.parse_args()
    if args.expires_minutes < 1:
        parser.error("--expires-minutes must be positive")
    try:
        token = mint_token(
            secret=os.getenv("MCP_JWT_SECRET"),
            subject=args.subject,
            user_id=args.user_id,
            issuer=args.issuer,
            audience=args.audience,
            expires_minutes=args.expires_minutes,
            admin=args.admin,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(token)


if __name__ == "__main__":
    main()
