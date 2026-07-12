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

Run `geologic-memory-mcp --help` for all server options. Prefer environment variables for `OPENAI_API_KEY` and `OPENAI_BASE_URL` so secrets do not appear in process listings.

