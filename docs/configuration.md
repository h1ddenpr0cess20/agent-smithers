# Configuration

Agent Smithers now reads configuration from a `.env` file. By default it uses `./.env`, or you can pass a custom path with `--env-file`.

## Required Variables

- `DEFAULT_MODEL`
- `MATRIX_SERVER`
- `MATRIX_USERNAME`
- `MATRIX_PASSWORD`
- `MATRIX_CHANNELS`

## Common Variables

- `OPENAI_API_KEY`
  Required when you use any OpenAI model.
- `XAI_API_KEY`
  Required when you use any xAI model.
- `OPENAI_MODELS`
  Comma-separated OpenAI model list used as startup fallback before server refresh.
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
  Optional ISO 3166-1 alpha-2 country code used for OpenAI `web_search` location biasing, for example `US`.
- `TOOLS_X_SEARCH`
  `true` or `false`. Used by xAI models.
- `TOOLS_CODE_INTERPRETER`
  `true` or `false`.
- `TOOLS_IMAGE_GENERATION`
  `true` or `false`. Used by OpenAI hosted image generation and Grok Imagine local image tools when `XAI_API_KEY` is configured.
- `TOOLS_VIDEO_GENERATION`
  `true` or `false`. Used by the local `generate_video` tool for OpenAI Sora and xAI Grok Imagine when the corresponding API keys are configured.
- `VIDEO_WHITELIST`
  Comma-separated Matrix user IDs or display names allowed to generate video. When set, only these users (plus admins) can trigger video generation. Leave empty to allow everyone. Admins configured in `MATRIX_ADMINS` are always allowed regardless of the whitelist. Can also be managed at runtime with the `.whitelist` admin command.
- `MCP_SERVERS`
  JSON object defining remote MCP servers.
- `LLM_TIMEOUT`
  Request timeout in seconds.
- `HISTORY_SIZE`
  Per-thread message cap.
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
OPENAI_API_KEY=sk-...
XAI_API_KEY=
OPENAI_MODELS=gpt-5-mini
XAI_MODELS=grok-4
DEFAULT_MODEL=gpt-5-mini
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
MATRIX_STORE_PATH=store
MATRIX_E2E=true
```

## Notes

- `MCP_SERVERS` must be valid JSON.
- `RESPONSES_OPTIONS` must be valid JSON.
- Set the keys and model lists for the providers you want available at the same time.
- `OPENAI_MODELS` or `XAI_MODELS` is still useful as fallback even when `SERVER_MODELS=true`.
- `TOOLS_WEB_SEARCH_COUNTRY` is currently applied only to OpenAI `web_search`, because xAI's provider docs do not document a country filter for `web_search` or `x_search`.
- `TOOLS_WEB_SEARCH_COUNTRY` is sent as a structured OpenAI `web_search.user_location.country` value. For xAI search tools, which do not currently document a country parameter, the bot adds a search-policy instruction so `x_search` and `web_search` still bias toward US sources.
- `TOOLS_VIDEO_GENERATION` applies across OpenAI and xAI chat models; the tool backend is selected automatically or via the tool's `backend` argument.
- `VIDEO_WHITELIST` is enforced at tool execution time. The video tool definitions are still sent to the model so it can explain the restriction, but the actual API call is blocked for non-whitelisted users. Admins are always allowed.
- Keep `.env` out of version control.
