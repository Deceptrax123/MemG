# GeoLogic Memory

GeoLogic Memory is a geometric long-term memory engine for dialogue and facts. It ships as both a Python distribution and a stdio Model Context Protocol (MCP) server.

## Install the pip package

Python 3.10 or newer is required. From this repository, run:

```bash
python -m pip install .
```

For development, use `python -m pip install -e .`. The distribution installs the standalone `geologic_memory` runtime, the `mcp` server module, and a `geologic-memory-mcp` console command.

The runtime needs a trained GeoLogic checkpoint. The repository’s `agent/` and `finetune/` directories remain research references and are not included in the wheel. Dialogue extraction and generated answers also need `OPENAI_API_KEY`. Transformers may download model files on first use.

## Run the MCP server

```bash
geologic-memory-mcp \
  --weights-path /absolute/path/to/checkpoint.pt \
  --cache-dir /absolute/path/to/memory-cache \
  --device cpu
```

See the [MCP guide](docs/mcp.md) for client configuration, tools, and input formats.

## Use the Python package

```python
from geologic_memory import create_service

memory = create_service(
    weights_path="/absolute/path/to/checkpoint.pt",
    cache_dir="/tmp/geologic-memory",
    device="cpu",
)

memory.ingest_turns([
    {
        "role": "user",
        "name": "Ada",
        "content": "I moved to London last year.",
        "timestamp": "2024-01-01T10:00:00Z",
    },
    {
        "role": "assistant",
        "content": "How do you like living there?",
        "timestamp": "2024-01-01T10:01:00Z",
    },
])
print(memory.retrieve_facts("Where does Ada live?"))
```

See the [Python package guide](docs/python-package.md) for configuration and lifecycle details.

## Development and release checks

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m compileall -q geologic_memory mcp
python -m build
python -m twine check dist/*
```

The unit tests and metadata checks do not require a checkpoint, API key, GPU, or model download. End-to-end ingestion and retrieval additionally require the runtime assets described below.

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [MIT license](LICENSE)

## Documentation

- [MCP server guide](docs/mcp.md)
- [Python package guide](docs/python-package.md)
- [Application and provider integration](docs/integrations.md)
- [Authenticated Streamable HTTP](docs/http-mcp.md)
- [Gaussian geometry visualizations](docs/visualizations.md)
- [Deployment guide](docs/deployment.md)
- [Architecture and development](docs/documentation.md)

