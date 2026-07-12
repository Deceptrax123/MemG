# Deployment guide

GeoLogic Memory supports newline-delimited JSON-RPC over stdio and authenticated, stateless Streamable HTTP. Use stdio when the MCP host launches the process directly; use Streamable HTTP for remote clients. See [Authenticated Streamable HTTP](http-mcp.md) for token, TLS proxy, and header requirements.

## Production prerequisites

Provision:

- Python 3.10 or newer and the installed `geologic-memory` package;
- a trusted compatible checkpoint;
- enough RAM for the encoder, sentence-transformer model, FAISS index, and stored nodes;
- a writable persistent cache directory;
- outbound access to the configured generation provider and model registry during initial model download;
- CUDA-compatible drivers and PyTorch only when using `--device cuda`.

Keep the checkpoint and Hugging Face cache on local or attached storage to reduce cold-start time.

## Remote Streamable HTTP

For remote clients, start `geologic-memory-mcp --transport streamable-http` and follow [Authenticated Streamable HTTP](http-mcp.md). The deployment must set `MCP_JWT_SECRET`, `MCP_JWT_ISSUER`, `MCP_JWT_AUDIENCE`, and `GEOLOGIC_MEMORY_ENCRYPTION_KEY`, terminate TLS at a reverse proxy, and preserve the Authorization and MCP protocol headers.

## Pattern 1: MCP host and server on the same machine

This is the simplest supported deployment. Install the package into a dedicated virtual environment:

```bash
python -m venv /opt/geologic-memory/venv
/opt/geologic-memory/venv/bin/python -m pip install geologic-memory
install -d -m 700 /var/lib/geologic-memory
```

Place the checkpoint at `/opt/geologic-memory/models/checkpoint.pt`, then add this provider-neutral entry to the MCP host configuration:

```json
{
  "mcpServers": {
    "geologic-memory": {
      "command": "/opt/geologic-memory/venv/bin/geologic-memory-mcp",
      "args": [
        "--weights-path", "/opt/geologic-memory/models/checkpoint.pt",
        "--cache-dir", "/var/lib/geologic-memory",
        "--device", "cpu"
      ],
      "env": {
        "OPENAI_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

The JSON shape is commonly accepted by stdio MCP hosts, but the location and name of the configuration file are host-specific. Restart the host after changing its configuration, then call `status` before a model-loading tool.

## Pattern 2: Remote VM through SSH

A desktop MCP host can use SSH as the stdio bridge. Install GeoLogic Memory on the VM, store secrets on the VM, and create `/opt/geologic-memory/bin/run-mcp`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="$(cat /run/secrets/openai_api_key)"
exec /opt/geologic-memory/venv/bin/geologic-memory-mcp \
  --weights-path /opt/geologic-memory/models/checkpoint.pt \
  --cache-dir /var/lib/geologic-memory \
  --device cpu
```

Make it executable and test it on the VM:

```bash
chmod 700 /opt/geologic-memory/bin/run-mcp
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | /opt/geologic-memory/bin/run-mcp
```

Configure the local MCP host to launch SSH:

```json
{
  "mcpServers": {
    "geologic-memory": {
      "command": "ssh",
      "args": [
        "-T",
        "-o", "BatchMode=yes",
        "memory-host.example.com",
        "/opt/geologic-memory/bin/run-mcp"
      ]
    }
  }
}
```

Use key-based authentication with a restricted service account. Do not enable a pseudo-terminal: terminal formatting can corrupt JSON-RPC traffic. Network interruption ends the MCP process; the host must reconnect.

## Pattern 3: Container deployment

Run the MCP host and GeoLogic Memory container together, or let the host launch the container with interactive stdin. The image should:

1. install the built wheel;
2. run as a non-root user;
3. use `geologic-memory-mcp` as its entrypoint;
4. mount the checkpoint read-only;
5. mount the cache and model cache as persistent writable volumes;
6. receive provider credentials from the platform secret manager.

Example host command after building an image named `geologic-memory:0.1.0`:

```json
{
  "mcpServers": {
    "geologic-memory": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--mount", "type=bind,src=/srv/models,dst=/models,readonly",
        "--mount", "type=volume,src=geologic-memory-data,dst=/data",
        "--env", "OPENAI_API_KEY",
        "geologic-memory:0.1.0",
        "--weights-path", "/models/checkpoint.pt",
        "--cache-dir", "/data",
        "--device", "cpu"
      ]
    }
  }
}
```

The MCP host must preserve container stdin and stdout. Send application logs to stderr because stdout is reserved for JSON-RPC.

## VM, container, and Kubernetes providers

- **Persistent VM:** fully supported when the MCP host runs on the VM or connects through SSH. Back up the cache volume according to the application's retention policy.
- **Managed container platform:** suitable when the MCP host and server share a task/pod and communicate over stdio. A standalone container listening only on stdio is not a network service.
- **Kubernetes:** run GeoLogic Memory as a sidecar or child process of the MCP host, mount a PersistentVolumeClaim for sessions, and inject credentials through a Secret. Do not expose the stdio process with a Kubernetes Service.
- **Serverless functions:** generally a poor fit because model loading is expensive and local state is ephemeral. Use external persistent storage and a supported network transport before choosing this pattern.

For a remote URL, run the authenticated Streamable HTTP mode behind a TLS reverse proxy. The built-in JWT validator authenticates the calling service; use an OAuth-aware gateway when individual third-party users require delegated authorization, audience validation, and scopes.

## Operations

### Health and smoke checks

Initialization and `status` do not load the model:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
  | geologic-memory-mcp --device cpu
```

A production readiness check should additionally load a trusted session or run one retrieval so checkpoint and model failures surface before traffic is accepted.

### Scaling

Each process has its own mutable in-memory index. Do not round-robin one logical session across replicas unless the application explicitly saves and reloads it from shared storage. Avoid concurrent writes to the same pickle session file. Prefer session affinity or one owner process per session.

### Upgrades

1. Back up the cache volume.
2. Build and validate the new wheel.
3. Test checkpoint compatibility in a staging cache.
4. Stop the MCP host/process cleanly.
5. Install or deploy the new version.
6. run initialization, `status`, and an end-to-end retrieval check.
7. Roll back both code and cache data if session serialization is incompatible.

### Security

- Keep API keys in environment-backed platform secrets, not MCP JSON committed to source control.
- Restrict checkpoint, Hugging Face cache, and memory-cache permissions.
- Treat dialogue and saved sessions as sensitive data.
- Load checkpoints and pickle sessions only from trusted sources.
- Terminate TLS and authenticate users in an MCP HTTP transport or gateway if one is added later.
