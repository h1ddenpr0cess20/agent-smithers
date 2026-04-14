# Migration

## From the Upstream Layout

The project still looks like the upstream Matrix bot, but the runtime behavior changed materially.

## Main Changes

- Replace `config.json` with `.env`
- Replace `--config` with `--env-file`
- Consolidate to xAI and LM Studio providers
- Replace Chat Completions with Responses API
- Replace local custom tools with hosted tools
- Replace local `fastmcp` client execution with hosted MCP configuration

## Start Command

```bash
agent-smithers --env-file .env
```
