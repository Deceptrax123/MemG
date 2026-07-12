# Security policy

Please report suspected vulnerabilities privately to the repository maintainers rather than opening a public issue. Include reproduction steps, affected versions, and potential impact. Maintainers will acknowledge the report and coordinate a fix and disclosure timeline.

## Security notes

- Treat memory content as sensitive user data and protect the configured cache directory accordingly.
- Never pass untrusted session files to `load_session`. Sessions use Python pickle for tensor compatibility, and pickle is unsafe for untrusted data.
- Supply OpenAI credentials through environment variables, not command-line arguments or committed configuration.
- Only load checkpoints from trusted sources. Checkpoints are loaded with PyTorch's restricted `weights_only=True` mode.
