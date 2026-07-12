# MCP server guide

Install the package, then configure an MCP client to launch the `geologic-memory-mcp` stdio executable.

## Client configuration

Use absolute paths for the executable, checkpoint, and cache when the client does not inherit your shell environment.

```json
{
  "mcpServers": {
    "geologic-memory": {
      "command": "geologic-memory-mcp",
      "args": [
        "--weights-path", "/absolute/path/to/checkpoint.pt",
        "--cache-dir", "/absolute/path/to/memory-cache",
        "--device", "cpu"
      ],
      "env": {
        "OPENAI_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

During development, the equivalent command is `python -m mcp.geologic_memory_server` from the repository root.

## Available tools

| Tool | Purpose |
| --- | --- |
| `status` | Return configuration and lazy-load status. |
| `configure_memory` | Change settings and reset the loaded model. |
| `ingest_turns` | Extract memories from user/assistant chat messages and store them. |
| `retrieve_facts` | Retrieve facts without final answer generation. |
| `answer_question` | Retrieve facts and generate an answer. |
| `save_session` | Persist memory under a session ID. |
| `load_session` | Restore a persisted session. |
| `clear_memory` | Clear the active memory store. |

The `status` tool does not load model assets. Other memory operations load the checkpoint lazily on their first call.

## Inputs

Each `ingest_turns` item uses the familiar chatbot `role` and `content` fields. `timestamp` and `name` are optional:

```json
{"turns":[{"role":"user","name":"Ada","content":"I moved to London.","timestamp":"2026-07-11T10:00:00Z"},{"role":"assistant","content":"How do you like it there?"}]}
```

Run `geologic-memory-mcp --help` for all server options.

For same-host, SSH, container, VM, and Kubernetes patterns, see the [deployment guide](deployment.md). For remote MCP clients, see [Authenticated Streamable HTTP](http-mcp.md). For application and model-provider adapters, see [application and provider integration](integrations.md). Prefer environment variables for `OPENAI_API_KEY` and `OPENAI_BASE_URL` so secrets do not appear in process listings.


## Hosted Streamable HTTP and authentication

Use the hosted transport for remote MCP clients:

```bash
export MCP_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export GEOLOGIC_MEMORY_ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export MCP_JWT_ISSUER="geologic-memory"
export MCP_JWT_AUDIENCE="geologic-memory-mcp"
export OPENAI_API_KEY="provider-key"

geologic-memory-mcp \
  --transport streamable-http \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --http-path /mcp \
  --weights-path /models/geologic-memory.pt \
  --cache-dir /var/lib/geologic-memory \
  --device cpu
```

Publish `/mcp` through an HTTPS reverse proxy. Preserve `Authorization`, `Content-Type`, `Origin`, and `MCP-Protocol-Version`; disable caching and Authorization-header logging. Keep the Python server private. See [JWT-authenticated Streamable HTTP](http-mcp.md) for token minting, exact claims, encryption migration, and key rotation.

The built-in authentication validates short-lived HS256 JWTs and binds `user_id` to storage. It is not a complete OAuth 2.1 authorization server. Put an OAuth/OIDC-aware gateway in front when clients require discovery, authorization code with PKCE, delegated user consent, or refresh tokens.

## Connect ChatGPT

ChatGPT connects to remote MCP servers, not directly to localhost. Deploy a stable public HTTPS URL such as:

```text
https://memory.example.com/mcp
```

For a private-network or developer-machine deployment, use a supported secure MCP tunnel instead of exposing the Python port publicly.

### Prerequisites

- Use a ChatGPT workspace and web experience that currently supports developer mode and custom MCP apps.
- Have a workspace admin enable developer mode and grant the appropriate role access.
- Deploy GeoLogic Memory behind an OAuth/OIDC gateway. The gateway should map each authenticated ChatGPT user to a stable internal `user_id`, then mint or inject the JWT expected by this server.
- Configure refresh tokens and `offline_access` when supported so the connection survives access-token expiry.

The current OpenAI workflow is documented in [Developer mode and MCP apps in ChatGPT](https://help.openai.com/en/articles/12584461-developer-mode-apps-and-full-mcp-connectors-in-chatgpt-beta).

### Create the app

1. In ChatGPT web, enable developer mode for the authorized account.
2. Open Workspace Settings or user Settings, then choose **Apps → Create**.
3. Enter `https://memory.example.com/mcp` as the remote MCP endpoint.
4. Select OAuth authentication and complete the authorization flow.
5. Choose **Scan Tools** and confirm that `load_user_memory`, `save_user_memory`, `ingest_turns`, `retrieve_facts`, and `answer_question` are discovered.
6. Create the draft app and enable it for the test account.
7. Open a new chat and select GeoLogic Memory from the apps/tools menu.
8. After changing tool schemas, refresh the app actions and review the diff before enabling them.

### Example: remember across chats

In the first chat, prompt:

```text
Use GeoLogic Memory. Load my shared memory, remember that I moved from
Boston to Portland in June 2026 because I joined Northstar Robotics,
ingest that as my user turn, and save my shared memory.
```

Expected tool order:

```text
load_user_memory → ingest_turns → save_user_memory
```

In a new chat, prompt:

```text
Use GeoLogic Memory. Load my shared memory and retrieve where I lived
before Portland, when I moved, and why. Answer only from retrieved memory.
```

Expected tool order:

```text
load_user_memory → retrieve_facts
```

The JWT identity must resolve to the same `user_id` in both chats. ChatGPT does not automatically archive every conversation into this database; persistence occurs only when it calls `ingest_turns` and `save_user_memory`.

### Troubleshooting

- **Tool scan gets 401:** verify OAuth gateway token exchange and the internal JWT issuer, audience, expiry, and `mcp` scope.
- **Tool call gets 403:** verify the gateway-mapped `user_id`; callers cannot override the JWT tenant.
- **Retrieval says no active user:** call `load_user_memory` before retrieval or ingestion.
- **Tools are missing after an update:** refresh the app actions or recreate the draft app when the workspace does not support updating a published app.
- **Local URL fails:** deploy remotely or use a secure MCP tunnel; ChatGPT does not launch the local stdio command.
