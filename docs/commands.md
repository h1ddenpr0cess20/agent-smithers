# Commands

Users interact with Agent Smithers using dot-commands or by mentioning the bot name followed by a colon. Each user has their own conversation history, isolated per room.

## User Commands

| Command | Description |
|---|---|
| `.ai <message>` | Chat with the bot using your conversation history in the current room. |
| `BotName: <message>` | Same as `.ai` — address the bot by its display name. |
| `.x <display_name or @user:server> <message>` | Send a message into another user's conversation context. The bot replies as if talking to that user. Useful for collaborative threads. |
| `.persona <text>` | Set a persona using the configured prompt wrapper (`BOT_PROMPT_PREFIX` + text + `BOT_PROMPT_SUFFIX`). The bot introduces itself immediately. |
| `.custom <prompt>` | Replace your system prompt entirely with arbitrary text. The bot introduces itself immediately. |
| `.mymodel [name]` | No args: show your current model and all available models. With a name: set a per-user, per-room model override. |
| `.reset` | Clear your history and restore the default persona. |
| `.stock` | Clear your history and run without any system prompt. |
| `.help` | Show help text. Reads `help.md` or `help.txt` from the working directory if present. |

## Admin Commands

Admin users are configured via `MATRIX_ADMINS` in `.env` (comma-separated Matrix user IDs or display names). Admins receive an additional section when they run `.help`.

| Command | Description |
|---|---|
| `.model [name\|reset]` | No args: show the current model and all available models by provider. With a model name: switch the global model. `reset` restores the configured default. |
| `.tools [on\|off\|toggle\|status]` | Enable or disable hosted tools and MCP access at runtime without restarting. |
| `.clear` | Reset conversation history for all users and restore default model and personality globally. |
| `.verbose [on\|off\|toggle]` | Control whether the brevity clause is appended to new conversations. Useful when you want longer, more detailed responses. |
| `.whitelist add\|remove\|list` | Manage the video generation whitelist. `add <user>` grants access, `remove <user>` revokes it, `list` shows current entries. Admins are always allowed. |

## Addressing the Bot

The bot responds to `.ai` and to `BotName:` where `BotName` is the bot's Matrix display name. No other message triggers a response unless it matches a registered command.

## Per-user Model Overrides

`.mymodel` sets a model override scoped to the current user and room. The override persists for the session. It does not affect other users. To see which models are available, run `.mymodel` with no arguments.
