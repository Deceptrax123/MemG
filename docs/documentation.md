# Architecture and development

## Layout

```text
geologic_memory/        Published geometric memory runtime
agent/                  Research reference implementation
mcp/                    Service facade, schemas, server, and stdio transport
docs/                   MCP, Python, and developer documentation
pyproject.toml          Build metadata, dependencies, and console command
```

The request path is:

1. `mcp.geologic_memory_server` parses command-line settings.
2. `mcp.service.GeoLogicMemoryService` manages lazy model loading.
3. `mcp.tools` declares MCP input schemas and binds service methods.
4. `mcp.transport` handles newline-delimited JSON-RPC over stdio.
5. `geologic_memory.memory.GeoLogicMemorySystem` handles extraction, storage, geometric retrieval, and answer generation.

## Runtime requirements

- A checkpoint compatible with `geologic_memory.model.GaussianEmbeddingTuner`.
- The configured Hugging Face tokenizer and embedding model, locally cached or downloadable.
- An OpenAI API key for dialogue extraction and generated answers.
- CUDA and a compatible PyTorch build when using `--device cuda`; otherwise select CPU.

Always provide a writable `--cache-dir` in deployed MCP configurations.

## Development checks

```bash
python -m compileall geologic_memory mcp
python -m build
```

After installing the wheel in a fresh virtual environment, smoke-test it with:

```bash
geologic-memory-mcp --help
python -c "import geologic_memory; print(geologic_memory.__version__)"
```

CLI help, MCP initialization, tool listing, and `status` do not require a checkpoint because the model is lazy-loaded. Ingestion and retrieval require model assets.

## Publishing checklist

Before publishing, confirm the license and project ownership metadata, update the version in `pyproject.toml`, `geologic_memory/__init__.py`, and `mcp/__init__.py`, build in a clean environment, and install the resulting wheel into a fresh environment for validation.

