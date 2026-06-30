"""Configuration and CLI parsing for the GeoLogic memory MCP server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODEL_CHOICES = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoLogic memory MCP server")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--gen-model-name", default="gpt-4.1-mini", choices=MODEL_CHOICES)
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--cache-dir", default=str(ROOT / "agent" / "memory_cache"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--overlap-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gaussian-eps", type=float, default=1e-6)
    parser.add_argument('--alpha_success', type=float, default=1)
    parser.add_argument('--beta_consistency', type=float, default=0.5)
    parser.add_argument('--gamma_inertia', type=float, default=0.1)
    parser.add_argument('--bc_weight', type=float, default=0.33)
    parser.add_argument('--entity_weight', type=float, default=0.33)
    parser.add_argument('--topic_weight', type=float, default=0.33)
    parser.add_argument('--base_threshold', type=float, default=0.3)
    parser.add_argument('--top_k', type=int, default=15)
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        # Memory/model runtime
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        hidden_dim=args.hidden_dim,
        model_name=args.embedding_model,
        device=args.device,
        path=args.weights_path,
        output_dim=args.output_dim,
        rank=args.rank,
        gaussian_eps=args.gaussian_eps,
        cuda=args.device == "cuda",
        gpu_id=args.gpu_id,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        # Existing agent knobs expected by GeoLogicMemorySystem
        alpha_success=args.alpha_success,
        beta_consistency=args.beta_consistency,
        gamma_inertia=args.gamma_inertia,
        bc_weight=args.bc_weight,
        entity_weight=args.entity_weight,
        topic_weight=args.topic_weight,
        base_threshold=args.base_threshold,
        # OpenAI-only generation config
        openai_api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY"),
        openai_base_url=args.openai_base_url or os.getenv("OPENAI_BASE_URL"),
        gen_model_name=args.gen_model_name
    )
