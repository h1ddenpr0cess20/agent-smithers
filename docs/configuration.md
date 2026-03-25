# Configuration

Agent Smithers now reads configuration from a `.env` file. By default it uses `./.env`, or you can pass a custom path with `--env-file`.

## Required Variables

- `DEFAULT_MODEL`
- `MATRIX_SERVER`
- `MATRIX_USERNAME`
- `MATRIX_PASSWORD`
- `MATRIX_CHANNELS`

## Common Variables

- `XAI_API_KEY`
  Required when you use any xAI model.
- `XAI_MODELS`
  Comma-separated xAI model list used as startup fallback before server refresh.
- `DEFAULT_MODEL`
  Default model name.
- `SERVER_MODELS`
  `true` or `false`. When enabled, the bot refreshes models from any configured provider on startup.
- `BOT_PERSONALITY`
  Default persona text.
- `BOT_PROMPT_PREFIX`
  Prefix used to build the default system prompt.
- `BOT_PROMPT_SUFFIX`
  Suffix used to build the default system prompt.
- `BOT_PROMPT_SUFFIX_EXTRA`
  Extra suffix text toggled by the `.verbose` command. Defaults to the brevity clause (`keep your responses relatively short.`). Set to empty to disable brevity by default.
- `RESPONSES_OPTIONS`
  JSON object merged into Responses API calls.
- `TOOLS_WEB_SEARCH`
  `true` or `false`.
- `TOOLS_WEB_SEARCH_COUNTRY`
  Optional ISO 3166-1 alpha-2 country code used for `web_search` location biasing, for example `US`.
- `TOOLS_X_SEARCH`
  `true` or `false`. Used by xAI models.
- `TOOLS_CODE_INTERPRETER`
  `true` or `false`.
- `TOOLS_IMAGE_GENERATION`
  `true` or `false`. Used by Grok Imagine local image tools when `XAI_API_KEY` is configured.
- `TOOLS_VIDEO_GENERATION`
  `true` or `false`. Used by the local `generate_video` tool for xAI Grok Imagine when `XAI_API_KEY` is configured.
- `VIDEO_WHITELIST`
  Comma-separated Matrix user IDs or display names allowed to generate video. When set, only these users (plus admins) can trigger video generation. Leave empty to allow everyone. Admins configured in `MATRIX_ADMINS` are always allowed regardless of the whitelist. Can also be managed at runtime with the `.whitelist` admin command.
- `MCP_SERVERS`
  JSON object defining remote MCP servers.
- `LLM_TIMEOUT`
  Request timeout in seconds.
- `HISTORY_SIZE`
  Per-thread message cap.
- `HISTORY_ENCRYPTION_KEY`
  Fernet encryption key for persisting conversation history to disk. Generate one with `agent-smithers --generate-key`. When set, history is saved to `<MATRIX_STORE_PATH>/history.enc` and restored on startup. When empty (default), history is in-memory only and lost on restart.
- `MARKDOWN`
  Enable Markdown-to-HTML rendering for Matrix messages.
- `MATRIX_ADMINS`
  Comma-separated Matrix admin IDs.
- `MATRIX_DEVICE_ID`
  Optional persisted Matrix device ID.
- `MATRIX_STORE_PATH`
  Local nio store directory.
- `MATRIX_E2E`
  `true` or `false`.

## Example

```env
XAI_API_KEY=xai-...
XAI_MODELS=grok-4
DEFAULT_MODEL=grok-4-1-fast-non-reasoning
SERVER_MODELS=true
BOT_PERSONALITY=an AI that can assume any personality, named Agent Smithers
BOT_PROMPT_PREFIX="assume the personality of "
BOT_PROMPT_SUFFIX=". roleplay and never break character."
RESPONSES_OPTIONS={}
TOOLS_WEB_SEARCH=true
TOOLS_WEB_SEARCH_COUNTRY=US
TOOLS_X_SEARCH=false
TOOLS_CODE_INTERPRETER=true
TOOLS_IMAGE_GENERATION=true
TOOLS_VIDEO_GENERATION=true
VIDEO_WHITELIST=@trusted:example.org,@creator:example.org
MCP_SERVERS={"deepwiki":{"server_url":"https://mcp.deepwiki.com/mcp","require_approval":"never"}}
LLM_TIMEOUT=180
MATRIX_SERVER=https://matrix.org
MATRIX_USERNAME=@bot:example.org
MATRIX_PASSWORD=secret
MATRIX_CHANNELS=!roomid:example.org,#ops:example.org
MATRIX_ADMINS=@admin:example.org
HISTORY_ENCRYPTION_KEY=
MATRIX_STORE_PATH=store
MATRIX_E2E=true
```

## Notes

- `MCP_SERVERS` must be valid JSON.
- `RESPONSES_OPTIONS` must be valid JSON.
- Set the keys and model lists for the providers you want available at the same time.
- `XAI_MODELS` is still useful as fallback even when `SERVER_MODELS=true`.
- `TOOLS_WEB_SEARCH_COUNTRY` is applied as an xAI search-policy instruction so `x_search` and `web_search` bias toward the configured country's sources.
- `TOOLS_WEB_SEARCH_COUNTRY` filtering is enabled by default when the variable is set, but can be toggled at runtime with the `.country` admin command.
- `TOOLS_VIDEO_GENERATION` applies to xAI chat models via the Grok Imagine backend.
- `VIDEO_WHITELIST` is enforced at tool execution time. The video tool definitions are still sent to the model so it can explain the restriction, but the actual API call is blocked for non-whitelisted users. Admins are always allowed.
- `HISTORY_ENCRYPTION_KEY` must be a valid Fernet key (base64-encoded 32 bytes). Use `agent-smithers --generate-key` to create one.
- When encrypted history is enabled, user locations set via `.location` are also persisted.
- Keep `.env` out of version control.
