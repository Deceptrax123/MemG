"""MCP tool schemas and service-method binding."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

from .config import MODEL_CHOICES

EMPTY_OBJECT_SCHEMA = {"type": "object", "properties": {}}

TURN_SCHEMA = {
    "type": "object",
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["user", "assistant"]},
        "content": {"type": "string", "minLength": 1},
        "timestamp": {
            "type": "string",
            "description": "Optional ISO 8601 timestamp; current UTC time is used when omitted.",
        },
        "name": {
            "type": "string",
            "description": "Optional participant name, such as a customer or tutor name.",
        },
    },
}

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "status": {
        "description": "Return server configuration and lazy-load status.",
        "inputSchema": EMPTY_OBJECT_SCHEMA,
    },
    "configure_memory": {
        "description": "Configure local memory runtime, OpenAI API key, and GPT model. Resets the lazy-loaded memory system.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "openai_api_key": {"type": "string"},
                "openai_base_url": {"type": "string"},
                "gen_model_name": {"type": "string", "enum": list(MODEL_CHOICES)},
                "weights_path": {"type": "string"},
                "cache_dir": {"type": "string"},
                "device": {"type": "string", "enum": ["cuda", "cpu"]},
                "gpu_id": {"type": "integer"},
                "top_k": {"type": "integer", "minimum": 1},
            },
        },
    },
    "clear_memory": {
        "description": "Clear the in-memory FAISS/BM25/node store.",
        "inputSchema": EMPTY_OBJECT_SCHEMA,
    },
    "load_session": {
        "description": "Load a local cached memory session by id.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}},
        },
    },
    "save_session": {
        "description": "Save the current memory session to local cache.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}},
        },
    },
    "ingest_turns": {
        "description": "Use an LLM to extract memories from user/assistant chat turns and ingest them into geometric memory.",
        "inputSchema": {
            "type": "object",
            "required": ["turns"],
            "properties": {
                "turns": {"type": "array", "items": TURN_SCHEMA},
                "window_size": {"type": "integer", "minimum": 1},
                "overlap_size": {"type": "integer", "minimum": 0},
                "max_workers": {"type": "integer", "minimum": 1},
            },
        },
    },
    "retrieve_facts": {
        "description": "Retrieve top memory facts for a query without final answer generation.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
        },
    },
    "answer_question": {
        "description": "Retrieve memory and answer a question using the configured OpenAI GPT model.",
        "inputSchema": {
            "type": "object",
            "required": ["question"],
            "properties": {
                "question": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
        },
    },
}


def bind_tool_methods(service) -> Dict[str, Callable[..., Any]]:
    return {
        "status": service.status,
        "configure_memory": service.configure_memory,
        "clear_memory": service.clear_memory,
        "load_session": service.load_session,
        "save_session": service.save_session,
        "ingest_turns": service.ingest_turns,
        "retrieve_facts": service.retrieve_facts,
        "answer_question": service.answer_question,
    }


def list_tools():
    return [{"name": name, **schema} for name, schema in TOOL_SCHEMAS.items()]


def content_result(value: Any) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, indent=2, default=str)}]}
