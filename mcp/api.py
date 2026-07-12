"""Convenience helpers for using GeoLogic Memory as a Python package."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .service import GeoLogicMemoryService


def runtime_options(**overrides: Any) -> SimpleNamespace:
    """Return service options with the same defaults as the MCP CLI."""
    defaults = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "gen_model_name": "gpt-4.1-mini",
        "weights_path": "",
        "cache_dir": str(Path.home() / ".cache" / "geologic-memory"),
        "encryption_key": os.getenv("GEOLOGIC_MEMORY_ENCRYPTION_KEY"),
        "embedding_model": "all-MiniLM-L6-v2",
        "device": "cuda",
        "gpu_id": 0,
        "window_size": 5,
        "overlap_size": 2,
        "hidden_dim": 64,
        "output_dim": 1024,
        "rank": 8,
        "gaussian_eps": 1e-6,
        "alpha_success": 1.0,
        "beta_consistency": 0.5,
        "gamma_inertia": 0.1,
        "bc_weight": 0.33,
        "entity_weight": 0.33,
        "topic_weight": 0.33,
        "base_threshold": 0.3,
        "top_k": 15,
    }
    unknown = sorted(set(overrides) - defaults.keys())
    if unknown:
        raise TypeError(f"Unknown runtime option(s): {', '.join(unknown)}")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def create_service(**options: Any) -> GeoLogicMemoryService:
    """Create a service from keyword options."""
    return GeoLogicMemoryService(runtime_options(**options))


__all__ = ["GeoLogicMemoryService", "create_service", "runtime_options"]

