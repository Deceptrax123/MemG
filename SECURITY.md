# Security policy

Please report suspected vulnerabilities privately to the repository maintainers rather than opening a public issue. Include reproduction steps, affected versions, and potential impact. Maintainers will acknowledge the report and coordinate a fix and disclosure timeline.

## Security notes

- Treat memory content as sensitive user data, protect the cache directory, and configure `GEOLOGIC_MEMORY_ENCRYPTION_KEY` for authenticated encryption at rest.
- Never pass untrusted session files to `load_session`. Sessions use Python pickle for tensor compatibility, and pickle is unsafe for untrusted data.
- Keep JWT signing, Fernet encryption, and OpenAI credentials in a secret manager, not command-line arguments or committed configuration.
- Only load checkpoints from trusted sources. Checkpoints are loaded with PyTorch's restricted `weights_only=True` mode.
