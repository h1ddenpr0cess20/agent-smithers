# Operations

## Start

```bash
infinigpt-matrix --env-file .env
```

## Recommended Persistence

- Persist `store/` for Matrix device state
- Keep `.env` outside version control
- Persist Docker `/data/store` if running in containers

## Logs

Use:

```bash
INFINIGPT_LOG_LEVEL=DEBUG infinigpt-matrix --env-file .env
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
