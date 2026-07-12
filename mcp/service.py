"""Lazy-loading service facade used by both public APIs."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .config import build_runtime_config
from .turns import normalize_chat_turns
from geologic_memory.helpers import load_model, load_tokenizer
from geologic_memory.memory import GeoLogicMemorySystem
import torch
CONFIG_ALIASES = {"weights_path": "path", "embedding_model": "model_name"}


class GeoLogicMemoryService:
    def __init__(self, args):
        self.config = build_runtime_config(args)
        self.system = None
        self.tokenizer = None
        self.encoder = None
        self.active_user_id = None

    def _require_system(self):
        if self.system is not None:
            return self.system


        if not self.config.path:
            raise RuntimeError(
                "weights_path is required before memory tools can run. "
                "Pass --weights-path or call configure_memory."
            )

        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available; configure device='cpu'.")
            torch.cuda.set_device(self.config.gpu_id)

        weights_path = os.path.abspath(os.path.expanduser(self.config.path))
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {weights_path}")
        self.config.path = weights_path

        self.tokenizer = load_tokenizer(self.config)
        self.encoder = load_model(self.config, weights_path=self.config.path)
        self.system = GeoLogicMemorySystem(self.config, self.encoder, self.tokenizer)
        self.system.db.cache_dir = self.config.cache_dir
        os.makedirs(self.config.cache_dir, exist_ok=True)
        return self.system

    def configure_memory(self, **kwargs):
        unknown = sorted(set(kwargs) - ({"weights_path", "embedding_model"} | set(vars(self.config))))
        if unknown:
            raise TypeError(f"Unknown configuration option(s): {', '.join(unknown)}")
        for key, value in kwargs.items():
            if value is not None:
                setattr(self.config, CONFIG_ALIASES.get(key, key), value)
        if self.config.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.config.top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.config.cuda = self.config.device == "cuda"
        self.system = None
        self.tokenizer = None
        self.encoder = None
        return self.status()

    def status(self):
        return {
            "loaded": self.system is not None,
            "cache_dir": self.config.cache_dir,
            "weights_path": self.config.path,
            "embedding_model": self.config.model_name,
            "device": self.config.device,
            "openai_model": self.config.gen_model_name,
            "openai_api_key_configured": bool(self.config.openai_api_key),
            "active_user_id": self.active_user_id,
        }

    def clear_memory(self):
        system = self._require_system()
        system.db.clear()
        return {"cleared": True, "nodes": len(system.db.nodes_dict)}

    def load_session(self, session_id: str, user_id: Optional[str] = None):
        system = self._require_system()
        loaded = system.db.load_session(session_id, user_id=user_id)
        return {"loaded": loaded, "session_id": session_id, "user_id": user_id, "nodes": len(system.db.nodes_dict)}

    def save_session(self, session_id: str, user_id: Optional[str] = None):
        system = self._require_system()
        system.db.save_session(session_id, user_id=user_id)
        return {"saved": True, "session_id": session_id, "user_id": user_id, "nodes": len(system.db.nodes_dict)}

    def load_user_memory(self, user_id: str):
        system = self._require_system()
        system.db.clear()
        loaded = system.db.load_session("shared-memory", user_id=user_id)
        self.active_user_id = user_id
        return {"loaded": loaded, "session_id": "shared-memory", "user_id": user_id, "nodes": len(system.db.nodes_dict)}

    def save_user_memory(self, user_id: str):
        if self.active_user_id != user_id:
            raise RuntimeError("Load this user's memory before saving it")
        return self.save_session("shared-memory", user_id=user_id)

    def ingest_turns(
        self,
        turns: List[Dict[str, Any]],
        window_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        max_workers: int = 4,
    ):
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be at least 1")
        if overlap_size is not None and overlap_size < 0:
            raise ValueError("overlap_size cannot be negative")
        effective_window = window_size or self.config.window_size
        effective_overlap = overlap_size if overlap_size is not None else self.config.overlap_size
        if effective_overlap >= effective_window:
            raise ValueError("overlap_size must be smaller than window_size")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        normalized_turns = normalize_chat_turns(turns)
        system = self._require_system()
        before = len(system.db.nodes_dict)
        system.process_windows_parallel(
            normalized_turns,
            window_size=effective_window,
            overlap_size=effective_overlap,
            max_workers=max_workers,
        )
        after = len(system.db.nodes_dict)
        return {"ingested_nodes": after - before, "total_nodes": after}

    def retrieve_facts(self, query: str, top_k: Optional[int] = None):
        query = self._validate_text(query, "query")
        top_k = self._validate_top_k(top_k)
        system = self._require_system()
        facts = system.retrieve_and_revise(
            query,
            top_k=top_k,
            run_optimization=False,
        )
        return {"query": query, "facts": facts}

    def answer_question(self, question: str, top_k: Optional[int] = None):
        question = self._validate_text(question, "question")
        top_k = self._validate_top_k(top_k)
        system = self._require_system()
        answer, rationale = system.answer_question(question, top_k=top_k)
        return {"question": question, "answer": answer, "rationale": rationale}

    @staticmethod
    def _validate_text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    def _validate_top_k(self, value: Optional[int]) -> int:
        result = self.config.top_k if value is None else value
        if isinstance(result, bool) or not isinstance(result, int) or result < 1:
            raise ValueError("top_k must be a positive integer")
        return result
