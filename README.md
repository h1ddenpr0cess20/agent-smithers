# Agent Smithers

Agent Smithers is an agentic AI assistant for the Matrix chat protocol using the Responses API format for OpenAI, xAI, or local models through LM Studio. It keeps separate conversation history for each user, lets people choose their own models and personas.  It also supports search and code execution tools, remote MCP servers, and image or video features such as Grok Imagine and Sora.  It even has GPT-4o available, if you miss that model.  

This is a fork of [inifinigpt-matrix](https://github.com/h1ddenpr0cess20/infinigpt-matrix)

## Table of Contents

- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
pip install .
cp .env.example .env
# Edit .env — set MATRIX_*, DEFAULT_MODEL, and at least one provider key
agent-smithers --env-file .env
```

Or without installing the package:

```bash
pip install -r requirements.txt
python -m agent_smithers --env-file .env
```

## Documentation

Full documentation lives in the [docs/](docs/) folder:

- [docs/getting-started.md](docs/getting-started.md) — prerequisites, install, first run
- [docs/configuration.md](docs/configuration.md) — all environment variables with types and defaults
- [docs/commands.md](docs/commands.md) — complete command reference
- [docs/tools-and-mcp.md](docs/tools-and-mcp.md) — hosted tools and remote MCP server setup
- [docs/lm-studio.md](docs/lm-studio.md) — local model setup via LM Studio
- [docs/operations.md](docs/operations.md) — E2E encryption, device verification, store persistence
- [docs/docker.md](docs/docker.md) — running in Docker
- [docs/architecture.md](docs/architecture.md) — internal design
- [docs/development.md](docs/development.md) — contributing and code style

## Commands

Send these in any room the bot has joined.

### User commands

| Command | What it does |
|---|---|
| `.ai <message>` | Chat with the bot using your conversation history |
| `BotName: <message>` | Same as `.ai` — address by name |
| `.x <user> <message>` | Send a message into another user's conversation context |
| `.persona <text>` | Set a personality using the configured prompt wrapper |
| `.custom <prompt>` | Replace your system prompt with arbitrary text |
| `.reset` | Clear your history and restore the default persona |
| `.stock` | Clear your history and run without any system prompt |
| `.mymodel [name]` | Show your current model, or set a per-user override |
| `.location [place]` | Show, set, or clear your location (added to system prompt) |
| `.help` | Show inline help (reads `help.md` if present) |

### Admin commands

Admins are set via `MATRIX_ADMINS` in `.env`.

| Command | What it does |
|---|---|
| `.model [name\|reset]` | Show available models, or change the active model globally |
| `.tools [on\|off\|toggle\|status]` | Enable or disable hosted tools and MCP at runtime |
| `.clear` | Reset history and defaults for all users |
| `.verbose [on\|off\|toggle]` | Control whether the brevity clause is included in new conversations |

## Conversation Persistence

Conversation history can be encrypted and saved to disk so it survives restarts. Generate a key and add it to `.env`:

```bash
agent-smithers --generate-key
# Paste the output into .env as HISTORY_ENCRYPTION_KEY
```

History is stored in `store/history.enc` alongside the Matrix device state. Without a key, history remains in-memory only (the default).

## Configuration

Copy `.env.example` to `.env`. Minimum required variables:

```env
MATRIX_SERVER=https://matrix.org
MATRIX_USERNAME=@bot:example.org
MATRIX_PASSWORD=secret
MATRIX_CHANNELS=!roomid:example.org
DEFAULT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

See [docs/configuration.md](docs/configuration.md) for all variables, including LM Studio, MCP servers, tool toggles, history size, and E2E encryption settings.

## E2E Encryption

The bot supports end-to-end encrypted Matrix rooms via `matrix-nio[e2e]`. This requires `libolm` to be available. Persist the `store/` directory (or whatever `MATRIX_STORE_PATH` points to) across restarts to retain device keys. See [docs/operations.md](docs/operations.md) for verification and troubleshooting.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution process and code style guide.

## License

AGPL-3.0 — see [LICENSE](LICENSE) for details.
