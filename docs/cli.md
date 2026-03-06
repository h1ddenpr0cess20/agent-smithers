# CLI

The main command is:

```bash
infinigpt-matrix --env-file .env
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

## Examples

```bash
infinigpt-matrix --env-file .env
infinigpt-matrix --env-file .env --model gpt-5-mini
INFINIGPT_LOG_LEVEL=DEBUG infinigpt-matrix --env-file .env
python -m infinigpt --env-file .env
```
