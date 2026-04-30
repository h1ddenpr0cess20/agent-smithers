# Operations

## Start

```bash
agent-smithers --env-file .env
```

## Recommended Persistence

- Persist `store/` for Matrix device state and encrypted conversation history
- Keep `.env` outside version control
- Persist Docker `/data/store` if running in containers

## Encrypted Conversation History

Set `HISTORY_ENCRYPTION_KEY` in `.env` to persist conversations across restarts. Generate a key:

```bash
agent-smithers --generate-key
```

History is saved to `store/history.enc` (or wherever `MATRIX_STORE_PATH` points). The file is Fernet-encrypted — plaintext never touches disk. User locations set via `.location` are also persisted.

If the key is lost or changed, the bot starts with empty history (a warning is logged). Back up the key alongside your `.env`.

## Logs

Use:

```bash
AGENT_SMITHERS_LOG_LEVEL=DEBUG agent-smithers --env-file .env
```

## Common Issues

- No replies:
  Check room membership, Matrix credentials, and OpenAI connectivity.
- Model errors:
  Confirm the model exists in OpenAI and that your API key can access it.
- MCP failures:
  Validate `MCP_SERVERS` JSON and any remote auth headers.
- Image/code tool issues:
  Confirm hosted tools are enabled in `.env`.
- E2EE problems:
  Persist `store/` and verify the bot device if needed.
