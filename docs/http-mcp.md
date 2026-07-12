# JWT-authenticated Streamable HTTP

GeoLogic Memory supports stateless MCP Streamable HTTP in addition to stdio. The HTTP transport accepts JSON-RPC requests with `POST /mcp`. This tool-only server does not need server-initiated messages, so `GET` and `DELETE` return `405 Method Not Allowed` and no SSE stream or MCP session ID is created.

## Generate server keys

JWT access tokens are signed, not encrypted. HTTPS protects tokens in transit. Generate a separate high-entropy JWT signing secret and Fernet database-encryption key:

```bash
export MCP_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export GEOLOGIC_MEMORY_ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export MCP_JWT_ISSUER="geologic-memory"
export MCP_JWT_AUDIENCE="geologic-memory-mcp"
```

Store both keys in the deployment secret manager. Losing the Fernet key makes encrypted sessions unrecoverable. Anyone holding the JWT secret can mint valid access tokens, so it must never be distributed to ordinary clients.

## Start the server

```bash
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

Bind to `127.0.0.1` behind a TLS reverse proxy. Bind to `0.0.0.0` only inside a protected container or private network.

If browser-originated clients are required, allow each exact origin explicitly:

```bash
geologic-memory-mcp \
  --transport streamable-http \
  --allowed-origin https://assistant.example.com \
  --allowed-origin https://admin.example.com \
  ...
```

Requests without an `Origin` header are accepted for non-browser MCP clients. A present but unlisted Origin receives HTTP 403.

## Mint a user token

The trusted deployment operator or identity gateway mints a short-lived JWT. The token binds the caller to one memory `user_id`:

```bash
export USER_TOKEN="$(geologic-memory-token \
  --subject assistant-client \
  --user-id user-42 \
  --expires-minutes 60)"
```

JWTs contain and validate:

- `sub`: calling client/account;
- `user_id`: only user-memory namespace the token can access;
- `iss` and `aud`: configured issuer and MCP audience;
- `iat` and `exp`: issue and expiration times;
- `scope`: must contain `mcp`.

An administrative token is required for the `configure_memory` tool:

```bash
geologic-memory-token \
  --subject deployment-admin \
  --user-id admin \
  --expires-minutes 15 \
  --admin
```

## Client requests

Every request supplies the JWT. After initialization, requests also carry the negotiated protocol version:

```bash
curl --fail-with-body http://127.0.0.1:8000/mcp \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'

curl --fail-with-body http://127.0.0.1:8000/mcp \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"load_user_memory","arguments":{}}}'
```

The server injects `user_id` from the JWT. A different explicit `user_id` receives HTTP 403. Retrieval, ingestion, clearing, and saving require that token's user memory to be loaded first.

Missing, expired, incorrectly signed, wrong-issuer, or wrong-audience tokens receive HTTP 401. The request body limit is 1 MiB.

## Encrypted session storage

When `GEOLOGIC_MEMORY_ENCRYPTION_KEY` is configured, saved user and conversation sessions are encrypted and authenticated with Fernet before being written. Files are written atomically with owner-only permissions.

Existing plaintext sessions cannot be opened while encryption is enabled. Before enforcing encryption:

1. back up the cache directory;
2. load each trusted plaintext session without an encryption key;
3. enable the Fernet key;
4. save each session into a new encrypted cache directory;
5. verify retrieval before deleting the plaintext backup.

Rotate the encryption key by decrypting with the old key and re-saving with the new key. Keep old keys available until migration and rollback windows close.

## OAuth boundary

The included JWT implementation is a local HS256 issuer/validator for trusted deployments. It is not a complete MCP OAuth 2.1 authorization server and does not publish protected-resource or authorization-server metadata. For public third-party clients, use an OAuth-aware gateway or identity provider that validates asymmetric JWTs, issuer, audience, expiry, and scopes before forwarding to this server.

The inbound JWT is never forwarded to the model provider. `OPENAI_API_KEY` remains a separate upstream credential.

## Reverse proxy requirements

The proxy must terminate HTTPS, preserve authentication and MCP headers, disable caching, enforce rate limits, and use timeouts long enough for initial model loading. Never log Authorization headers or memory payloads.
