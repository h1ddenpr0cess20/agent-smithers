# Architecture

The bot is a small async application that wires a Matrix client to a provider Responses API through a command router and stateless handlers.

## Main Components

- `infinigpt/app.py`
  Runtime context, hosted tool config, MCP approval loop, artifact handling, and Matrix sync startup.
- `infinigpt/llm_client.py`
  Provider-aware Responses API client plus model discovery and file download helpers.
- `infinigpt/history.py`
  In-memory per-room, per-user message history.
- `infinigpt/matrix_client.py`
  Thin wrapper around `matrix-nio`.
- `infinigpt/handlers/*`
  Command handlers for `.ai`, `.model`, `.persona`, `.x`, and admin commands.
- `infinigpt/config.py`
  `.env` parsing and validation.

## Request Flow

1. Matrix event arrives.
2. Router resolves a handler.
3. Handler updates the per-user history.
4. `AppContext.generate_reply()` converts history into Responses API input.
5. Hosted tools and MCP definitions are attached when enabled.
6. Text output is sent back to Matrix.
7. Generated images are uploaded to Matrix if present.

## Model Discovery

On startup, the bot can fetch models from the configured provider's `/models` endpoint. Those models populate `.model` and `.mymodel`, with the configured list kept as fallback.

## Secrets

Secrets live in `.env`, not in committed JSON config.
