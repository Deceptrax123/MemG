"""GeoLogic Memory MCP server entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.config import parse_args
from mcp.service import GeoLogicMemoryService
from mcp.transport import run_stdio_server, run_streamable_http_server


def main() -> None:
    args = parse_args()
    service = GeoLogicMemoryService(args)
    if args.transport == "stdio":
        run_stdio_server(service)
        return
    run_streamable_http_server(
        service,
        host=args.http_host,
        port=args.http_port,
        endpoint=args.http_path,
        jwt_secret=args.jwt_secret,
        jwt_issuer=args.jwt_issuer,
        jwt_audience=args.jwt_audience,
        allowed_origins=args.allowed_origin,
    )


if __name__ == "__main__":
    main()
