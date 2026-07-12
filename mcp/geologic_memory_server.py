
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.config import parse_args
from mcp.service import GeoLogicMemoryService
from mcp.transport import run_stdio_server


def main() -> None:
    service = GeoLogicMemoryService(parse_args())
    run_stdio_server(service)


if __name__ == "__main__":
    main()
