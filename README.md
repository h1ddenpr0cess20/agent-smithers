# Agent Smithers

Agent Smithers is a Matrix bot built around the Responses API. It keeps the upstream Matrix/history/command structure, and supports OpenAI or xAI with hosted tools plus remote MCP servers.

The name is a deliberate mashup of Agent Smith from *The Matrix* and Smithers, Mr. Burns' assistant from *The Simpsons*.

## Documentation

- [Overview](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Commands](docs/commands.md)
- [Tools & MCP](docs/tools-and-mcp.md)
- [Images Directory](docs/images.md)
- [Docker](docs/docker.md)
- [CLI Reference](docs/cli.md)
- [Operations & E2E](docs/operations.md)
- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Migration](docs/migration.md)
- [Legacy Map](docs/legacy-map.md)
- [Security](docs/security.md)
- [Not a Companion](docs/not-a-companion.md)
- [AI Output Disclaimer](docs/ai-output-disclaimer.md)

## Features

- Dynamic personalities with quick switching
- Per‑user history, isolated per room and user
- Collaborative mode to talk across histories
- OpenAI and xAI Responses API support with server-fetched model discovery
- Hosted tool support for web search, code interpreter, image generation (OpenAI), and `x_search` (xAI)
- Remote MCP server support through provider-hosted MCP tools
- Admin controls for model switching and global resets

## Installation

From source (installs CLI):

- `pip install .`
- Or use pipx: `pipx install .`

From source without installing the package:

- `pip install -r requirements.txt`
- Run with: `python -m agent_smithers --env-file .env`

After installation, use the `agent-smithers` command.

## Quick Start

1) Create a Matrix account for the bot and note the server URL, username, and password.
2) Copy `.env.example` to `.env`
3) Fill in your provider key and Matrix settings in `.env`
4) Run:

- Installed command: `agent-smithers --env-file .env`
- As module: `python -m agent_smithers --env-file .env`

## Usage

Common commands (see [Commands](docs/commands.md) for the full list):

| Command | Description | Example |
|---------|-------------|---------|
| `.ai <message>` or `BotName: <message>` | Chat with the AI | `.ai Hello there!` |
| `.x <user> <message>` | Continue another user's conversation | `.x Alice What did we discuss?` |
| `.persona <text>` | Change your personality | `.persona helpful librarian` |
| `.custom <prompt>` | Use a custom system prompt | `.custom You are a coding expert` |
| `.reset` / `.stock` | Clear history (default/stock prompt) | `.reset` |
| `.mymodel [name]` | Show/change personal model | `.mymodel gpt-4o-mini` |
| `.model [name|reset]` (admin) | Show/change model | `.model gpt-5-mini` |
| `.clear` (admin) | Reset globally for all users | `.clear` |
| `.help` | Show inline help | `.help` |

## Encryption Support

- Works in encrypted Matrix rooms using `matrix-nio[e2e]` with device verification.
- Requires `libolm` available to Python for E2E. If unavailable, you can run without E2E; see [Operations](docs/operations.md) and [Verification](docs/verification.md).
- Persist the `store/` directory to retain device keys and encryption state.

## Community & Policies

- Code of Conduct: [Code of Conduct](CODE_OF_CONDUCT.md)
- Contributing: [Contributing](CONTRIBUTING.md)
- Security Policy: [Security Policy](SECURITY.md)
- Security Guide: [Security Guide](docs/security.md)

## License

AGPL‑3.0 — see [License](LICENSE) for details.
