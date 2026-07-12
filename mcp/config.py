"""Configuration shared by the Python API and MCP command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "geologic-memory"
MODEL_CHOICES = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoLogic memory MCP server")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--gen-model-name", default="gpt-4.1-mini", choices=MODEL_CHOICES)
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--overlap-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gaussian-eps", type=float, default=1e-6)
    parser.add_argument("--alpha-success", "--alpha_success", dest="alpha_success", type=float, default=1)
    parser.add_argument("--beta-consistency", "--beta_consistency", dest="beta_consistency", type=float, default=0.5)
    parser.add_argument("--gamma-inertia", "--gamma_inertia", dest="gamma_inertia", type=float, default=0.1)
    parser.add_argument("--bc-weight", "--bc_weight", dest="bc_weight", type=float, default=0.33)
    parser.add_argument("--entity-weight", "--entity_weight", dest="entity_weight", type=float, default=0.33)
    parser.add_argument("--topic-weight", "--topic_weight", dest="topic_weight", type=float, default=0.33)
    parser.add_argument("--base-threshold", "--base_threshold", dest="base_threshold", type=float, default=0.3)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=15)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return create_parser().parse_args(argv)


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
        # Geometric policy knobs expected by GeoLogicMemorySystem
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
