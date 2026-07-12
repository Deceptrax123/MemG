# Python package guide

## Installation

Install from a checkout with `python -m pip install .`, or use `python -m pip install -e .` while developing.

The distribution name is `geologic-memory`, and its public Python import namespace is `geologic_memory`. The bundled `mcp` module provides the stdio server.

## Create a service

Use the convenience API to configure the service with keyword arguments:

```python
from geologic_memory import create_service

service = create_service(
    weights_path="/models/geologic-memory.pt",
    cache_dir="/tmp/geologic-memory",
    embedding_model="all-MiniLM-L6-v2",
    device="cpu",
    top_k=15,
)
```

## Lifecycle

The constructor stores configuration. The first memory operation loads the tokenizer, checkpoint, vector index, and sentence-transformer model.

```python
service.status()
service.configure_memory(device="cuda", gpu_id=0)
service.save_session("customer-42")
service.clear_memory()
service.load_session("customer-42")
```

`configure_memory()` resets the loaded model so the next operation uses the new configuration. It does not delete saved sessions.

Use `ingest_turns()` to send `role`/`content` user and assistant messages through the configured OpenAI model. The extractor resolves speakers and pronouns, normalizes time, creates atomic memories internally, and feeds them into the geometric store. `retrieve_facts()` returns evidence without generating a final response, while `answer_question()` performs both retrieval and generation.

All public service methods return dictionaries suitable for JSON serialization.

See [application and provider integration](integrations.md) for the retrieve-generate-ingest lifecycle, provider adapters, concurrency guidance, and tenant isolation. See the [deployment guide](deployment.md) for production runtime patterns.

