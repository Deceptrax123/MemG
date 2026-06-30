---
name: geologic-memory
description: Use when a coding agent needs to test, deploy, or integrate the GeoLogic long-term memory MCP server, ingest/retrieve/answer from local memory sessions, configure OpenAI GPT model credentials, or preserve memory_cache as local-only state.
---

# GeoLogic Memory

Use this skill when working with the GeoLogic memory system in this repo or connecting it to a coding agent through MCP.

## Server

Run the stdio MCP server from the repo root:

```bash
python mcp/geologic_memory_server.py --weights-path /path/to/checkpoint --openai-api-key "$OPENAI_API_KEY" --gen-model-name gpt-4.1-mini
```

Useful options:

```bash
--gen-model-name gpt-4.1-mini   # default
--gen-model-name gpt-4.1
--gen-model-name gpt-4.1-nano
--gen-model-name gpt-4o
--gen-model-name gpt-4o-mini
--gen-model-name o4-mini
--cache-dir agent/memory_cache
--device cuda
--device cpu
```

Prefer `OPENAI_API_KEY` from the environment for deployment. Use the `configure_memory` MCP tool only for local testing or ephemeral agent sessions.

## MCP Tools

Start with `status`. It does not load model weights.

Use `configure_memory` to set or override:

```json
{
  "openai_api_key": "sk-...",
  "gen_model_name": "gpt-4.1-mini",
  "weights_path": "/path/to/checkpoint",
  "cache_dir": "agent/memory_cache",
  "device": "cuda"
}
```

Use `ingest_facts` for deterministic tests because it avoids LLM extraction. Use `ingest_turns` when testing full dialogue-to-memory extraction with OpenAI.

Use `retrieve_facts` when debugging retrieval quality. Use `answer_question` only when final generation is needed.

Use `load_session` and `save_session` for local cache sessions. Keep cache paths local; never commit `memory_cache/` or `agent/memory_cache/`.

## Integration Pattern

1. Configure the MCP server in the agent client as a stdio command.
2. Call `status` to confirm model path, OpenAI model, cache dir, and API key presence.
3. Call `load_session` if a local cache exists, otherwise ingest facts/turns.
4. Debug with `retrieve_facts` before using `answer_question`.
5. Save named sessions only under the ignored cache directory.

## Local Code Notes

The agent is OpenAI-only for generation. The configurable fields are `openai_api_key`, `openai_base_url`, and `gen_model_name`. The default GPT model is `gpt-4.1-mini`.

Memory model weights are separate from OpenAI credentials. If `weights_path` is missing, only `status` and `configure_memory` should be expected to work.
