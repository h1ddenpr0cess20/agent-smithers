# Agent Smithers — Help

Use dot-commands or address the bot by name followed by a colon.

## User Commands

| Command | Description | Example |
|---|---|---|
| `.ai <message>` or `BotName: <message>` | Chat with the bot using your own conversation history. | `.ai Hello there!` |
| `.x <display_name or @user:server> <message>` | Send a message into another user's conversation context. | `.x Alice What did we decide?` |
| `.persona <personality>` | Set an AI personality using the configured prompt wrapper. | `.persona grumpy historian` |
| `.custom <prompt>` | Replace your system prompt with custom text entirely. | `.custom You are a coding tutor.` |
| `.mymodel [name]` | Show your current model or set a per-user model override. | `.mymodel gpt-4o-mini` |
| `.reset` | Clear your history and restore the default persona. | `.reset` |
| `.stock` | Clear your history and run without any system prompt. | `.stock` |
| `.help` | Show this help message. | `.help` |

~~~

## Admin Commands

| Command | Description | Example |
|---|---|---|
| `.model [name or reset]` | No args: show current model and all available models. With a name: switch model globally. `reset` restores the default. | `.model gpt-4o` |
| `.tools [on\|off\|toggle\|status]` | Enable or disable hosted tools and MCP access at runtime. | `.tools toggle` |
| `.clear` | Reset history and defaults for all users globally. | `.clear` |
| `.verbose [on\|off\|toggle]` | Control whether the brevity clause is added to new conversations. | `.verbose off` |
| `.country [on\|off\|toggle\|status]` | Toggle search country filtering (e.g. US-only results) at runtime. | `.country off` |
| `.whitelist add\|remove\|list` | Manage the video generation whitelist. Admins are always allowed. | `.whitelist add @user:server` |
