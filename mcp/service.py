"""GeoLogic memory service operations exposed through MCP tools."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .config import build_runtime_config


class GeoLogicMemoryService:
    def __init__(self, args):
        self.config = build_runtime_config(args)
        self.system = None
        self.tokenizer = None
        self.encoder = None

    def _require_system(self):
        if self.system is not None:
            return self.system
        if not self.config.path:
            raise RuntimeError(
                "weights_path is required before memory tools can run. "
                "Pass --weights-path or call configure_memory."
            )

        from agent.helpers import load_model, load_tokenizer
        from agent.memory import GeoLogicMemorySystem

        if self.config.device == "cuda":
            import torch
            torch.cuda.set_device(self.config.gpu_id)

        self.tokenizer = load_tokenizer(self.config)
        self.encoder = load_model(self.config, weights_path=self.config.path)
        self.system = GeoLogicMemorySystem(self.config, self.encoder, self.tokenizer)
        self.system.db.cache_dir = self.config.cache_dir
        os.makedirs(self.config.cache_dir, exist_ok=True)
        return self.system

    def configure_memory(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None and hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config.cuda = self.config.device == "cuda"
        self.config.path = kwargs.get("weights_path") or getattr(self.config, "path", None)
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
        }

    def clear_memory(self):
        system = self._require_system()
        system.db.clear()
        return {"cleared": True, "nodes": len(system.db.nodes_dict)}

    def load_session(self, session_id: str):
        system = self._require_system()
        loaded = system.db.load_session(session_id)
        return {"loaded": loaded, "session_id": session_id, "nodes": len(system.db.nodes_dict)}

    def save_session(self, session_id: str):
        system = self._require_system()
        system.db.save_session(session_id)
        return {"saved": True, "session_id": session_id, "nodes": len(system.db.nodes_dict)}

    def ingest_turns(
        self,
        turns: List[Dict[str, Any]],
        window_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        max_workers: int = 4,
    ):
        system = self._require_system()
        before = len(system.db.nodes_dict)
        system.process_windows_parallel(
            turns,
            window_size=window_size or self.config.window_size,
            overlap_size=overlap_size if overlap_size is not None else self.config.overlap_size,
            max_workers=max_workers,
        )
        after = len(system.db.nodes_dict)
        return {"ingested_nodes": after - before, "total_nodes": after}

    def ingest_facts(self, facts: List[Dict[str, Any]]):
        system = self._require_system()
        from agent.schemas import AtomicFact

        before = len(system.db.nodes_dict)
        atomic_facts = [AtomicFact(**fact) for fact in facts]
        system.db.ingest_facts(atomic_facts)
        after = len(system.db.nodes_dict)
        return {"ingested_nodes": after - before, "total_nodes": after}

    def retrieve_facts(self, query: str, top_k: Optional[int] = None):
        system = self._require_system()
        facts = system.retrieve_and_revise(
            query,
            top_k=top_k or self.config.top_k,
            run_optimization=False,
        )
        return {"query": query, "facts": facts}

    def answer_question(self, question: str, top_k: Optional[int] = None):
        system = self._require_system()
        answer, rationale = system.answer_question(question, top_k=top_k or self.config.top_k)
        return {"question": question, "answer": answer, "rationale": rationale}
