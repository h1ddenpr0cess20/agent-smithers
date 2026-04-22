# CLI

The main command is:

```bash
agent-smithers --env-file .env
```

## Options

- `-e, --env-file PATH`
  Path to the env file. Default: `./.env`
- `-L, --log-level`
  One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `-E, --e2e`
  Force E2EE on
- `-N, --no-e2e`
  Force E2EE off
- `-m, --model`
  Override the default model
- `-s, --store-path`
  Override `MATRIX_STORE_PATH`
- `-S, --server-models`
  Force model refresh from the configured provider on startup
- `-v, --verbose`
  Enable verbose mode for new conversations
- `--generate-key`
  Generate a Fernet encryption key for `HISTORY_ENCRYPTION_KEY` and exit

## Examples

```bash
agent-smithers --env-file .env
agent-smithers --env-file .env --model gpt-5-mini
agent-smithers --generate-key
AGENT_SMITHERS_LOG_LEVEL=DEBUG agent-smithers --env-file .env
python -m agent_smithers --env-file .env
```
