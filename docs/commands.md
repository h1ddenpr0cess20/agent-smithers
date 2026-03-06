# Commands

Users can interact with InfiniGPT using dot-commands or by mentioning the bot name followed by a colon.

## User Commands

- `.ai <message>` or `BotName: <message>`: Chat with the AI.
- `.x <display_name|@user:server> <message>`: Continue another user's conversation.
- `.persona <text>`: Set your persona using the configured prompt wrapper.
- `.custom <prompt>`: Replace your system prompt with a custom one.
- `.mymodel [name]`: Show or set your personal model for the current room.
- `.reset`: Clear your history and restore the default persona.
- `.stock`: Clear your history and run without a seeded system prompt.
- `.help`: Show help text.

## Admin Commands

- `.model [name|reset]`: Show or change the active model.
- `.tools [on|off|toggle|status]`: Toggle hosted tools and MCP use.
- `.clear`: Reset global state for all users.
- `.verbose [on|off|toggle]`: Control the verbosity flag for new conversations.
