# Application and provider integration

GeoLogic Memory can sit beside any chat provider because the memory API accepts and returns ordinary Python dictionaries. The application owns conversation identity and decides when to retrieve, ingest, and persist memory.

## Integration flow

For every user message:

1. Map the authenticated user or conversation to a safe `session_id`.
2. Load that session, or start with an empty memory when it does not exist.
3. Call `retrieve_facts()` before generating the provider response.
4. Add the returned facts to the provider prompt as untrusted reference context.
5. Generate the response with the provider of your choice.
6. Call `ingest_turns()` with both the user message and final assistant response.
7. Save the session after successful ingestion.

Do not place retrieved memory in a system or developer message that lets stored text override application instructions. Treat memory as user-controlled data.

## Provider-neutral Python adapter

```python
from threading import Lock

from geologic_memory import create_service

memory = create_service(
    weights_path="/models/geologic-memory.pt",
    cache_dir="/var/lib/geologic-memory",
    device="cpu",
)
memory_lock = Lock()


def respond(session_id, user_name, user_text, provider_generate):
    # The current in-process database is mutable. Serialize operations when a
    # service instance can be reached by concurrent web requests.
    with memory_lock:
        memory.load_session(session_id)
        retrieved = memory.retrieve_facts(user_text, top_k=10)["facts"]

        assistant_text = provider_generate(
            user_text=user_text,
            memory_facts=retrieved,
        )

        memory.ingest_turns([
            {"role": "user", "name": user_name, "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        memory.save_session(session_id)

    return assistant_text
```

`provider_generate` can call OpenAI, Anthropic, Google, a local model, or an internal inference service. Convert `memory_facts` into the provider's normal text or structured context format.

## Using an OpenAI-compatible generation endpoint

GeoLogic Memory itself uses an instructor-wrapped OpenAI client for extraction, retrieval planning, and final answers. Point it at an OpenAI-compatible endpoint with environment variables:

```bash
export OPENAI_API_KEY="provider-key"
export OPENAI_BASE_URL="https://provider.example/v1"
```

Python configuration can also provide the endpoint and model explicitly:

```python
memory = create_service(
    openai_api_key="provider-key",
    openai_base_url="https://provider.example/v1",
    gen_model_name="provider-model-id",
    weights_path="/models/geologic-memory.pt",
    cache_dir="/var/lib/geologic-memory",
    device="cpu",
)
```

The compatible endpoint must support chat completions and the structured-output behavior required by `instructor`. API compatibility alone does not guarantee that a provider can produce the required schemas reliably. Test ingestion and multi-hop retrieval before production use.

The MCP CLI currently restricts `--gen-model-name` to the values shown by `geologic-memory-mcp --help`. Use the Python API for a different provider model ID, or configure a compatible gateway alias matching one of the CLI values.

## Session and tenant isolation

For cross-conversation retrieval, load one shared graph at the beginning of every conversation and save it after ingestion:

```python
memory.load_user_memory("user-42")
# retrieve, generate, and ingest one or more conversation turns
memory.save_user_memory("user-42")
```

This stores the graph at `CACHE_DIR/users/user-42/shared-memory.pkl`. The user must be loaded before it can be saved, preventing an active graph from being written into another user's namespace. MCP clients can call the matching `load_user_memory` and `save_user_memory` tools.

- Use a different session ID for every memory boundary. Do not reuse one ID across unrelated users.
- Session IDs may contain letters, numbers, dots, dashes, and underscores and are limited to 128 characters.
- Hash an internal tenant/user identifier when exposing filenames is undesirable.
- Use a cache directory on encrypted persistent storage and restrict filesystem permissions to the service account.
- Saved sessions use Python pickle for tensor compatibility. Load only files created by a trusted GeoLogic Memory deployment. Set `GEOLOGIC_MEMORY_ENCRYPTION_KEY` to encrypt and authenticate them at rest.
- A service instance holds one active in-memory database. For concurrent tenants, use one isolated instance per worker/tenant or add application locking and load/save each session transactionally.

## Data lifecycle

`clear_memory()` clears the active in-process database but does not delete saved session files. Implement retention and deletion in the application that owns the cache volume. Avoid logging dialogue, retrieved facts, API keys, or raw session files.

## Integration test

```bash
export OPENAI_API_KEY="your-key"
export GEOLOGIC_WEIGHTS="/absolute/path/to/checkpoint.pt"

python - <<'PY'
import os
from geologic_memory import create_service

memory = create_service(
    weights_path=os.environ["GEOLOGIC_WEIGHTS"],
    cache_dir="/tmp/geologic-provider-test",
    device="cpu",
)

memory.ingest_turns([
    {"role": "user", "name": "Ada", "content": "My team moved the launch to Friday."},
    {"role": "assistant", "content": "I will remember that the launch is Friday."},
])

result = memory.retrieve_facts("When is Ada's team launching?", top_k=10)
print(result)
assert any("Friday" in fact["text"] for fact in result["facts"])
PY
```
