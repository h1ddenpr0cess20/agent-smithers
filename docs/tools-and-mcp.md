# Tools and MCP

Agent Smithers uses provider-hosted tools through the Responses API. It does not execute the old local custom tools or the previous `fastmcp` client path.

## Hosted Tools

These are controlled through `.env`:

- `TOOLS_WEB_SEARCH=true`
- `TOOLS_WEB_SEARCH_COUNTRY=US`
- `TOOLS_X_SEARCH=true`
- `TOOLS_CODE_INTERPRETER=true`
- `TOOLS_IMAGE_GENERATION=true`
- `TOOLS_VIDEO_GENERATION=true`

When enabled, the bot sends those tools in the Responses API request and the selected provider runs them on the hosted side.

Provider notes:

- OpenAI supports hosted `web_search`, `code_interpreter`, and `image_generation`.
- This project also exposes a local `generate_video` function tool for OpenAI chat models, backed by the Sora Video API.
- OpenAI `web_search` also supports `user_location`; this project maps `TOOLS_WEB_SEARCH_COUNTRY=US` to `{"user_location":{"type":"approximate","country":"US"}}`.
- xAI documents `web_search`, `x_search`, `code_interpreter`, and remote MCP for the Grok 4 family. This project only attaches those hosted tools to `grok-4*` models.
- Local media function tools are broader than hosted tools. When the relevant API keys are configured, this project exposes Grok Imagine image tools and a backend-selectable `generate_video` tool to both OpenAI and xAI chat models.
- xAI's current provider docs do not document a country filter for `web_search`, and `x_search` exposes X-specific filters rather than country selection.
- To keep behavior aligned, this project also applies `TOOLS_WEB_SEARCH_COUNTRY=US` as an xAI search-policy instruction whenever `web_search` or `x_search` is enabled.
- `TOOLS_X_SEARCH` is used only when the active model is from xAI.
- `TOOLS_IMAGE_GENERATION` enables OpenAI hosted image generation plus Grok Imagine image generation/editing local tools when `XAI_API_KEY` is configured.
- `TOOLS_VIDEO_GENERATION` enables a local `generate_video` tool with `backend` selection for OpenAI Sora and xAI Grok Imagine when the relevant provider keys are configured.

## MCP

Remote MCP servers are configured with `MCP_SERVERS` as a JSON object.

Example:

```env
MCP_SERVERS={"deepwiki":{"server_url":"https://mcp.deepwiki.com/mcp","require_approval":"never"}}
```

Supported fields depend on the hosted MCP tool shape used by the provider.

OpenAI pass-through fields:

- `server_url`
- `connector_id`
- `server_label`
- `server_description`
- `allowed_tools`
- `require_approval`
- `authorization`
- `headers`

xAI pass-through fields:

- `server_url`
- `server_label`
- `server_description`
- `allowed_tools` -> sent as `allowed_tool_names`
- `authorization`
- `headers` -> sent as `extra_headers`

There is also one bot-specific helper:

- `auto_approve`
  If set to `true`, the bot will auto-approve MCP approval requests for that server.

Example:

```env
MCP_SERVERS={"deepwiki":{"server_url":"https://mcp.deepwiki.com/mcp","auto_approve":true}}
```

## Video Whitelist

Video generation through Sora and Grok costs real money per call. To prevent surprise bills, you can restrict who is allowed to generate video using the `VIDEO_WHITELIST` environment variable or the `.whitelist` admin command.

When the whitelist is active, only listed users and admins can generate video. Everyone else gets a rejection message. Admins (configured via `MATRIX_ADMINS`) are always allowed regardless of the whitelist.

```env
VIDEO_WHITELIST=@trusted:example.org,@creator:example.org
```

At runtime, admins can manage the whitelist without restarting:

```text
.whitelist add @user:server
.whitelist remove @user:server
.whitelist list
```

Adding the first user via `.whitelist add` automatically enables enforcement. Pre-seeding via the env var also enables it on startup.

**Warning about smaller models:** Low-powered or local LLM models (e.g. small models served through LM Studio) may call video generation tools unpredictably or by accident. If you run such models alongside video-capable providers, either set `TOOLS_VIDEO_GENERATION=false` or use the whitelist to prevent unintended expensive API calls. The whitelist blocks the actual video API call even if the model emits a tool call, so it acts as a safety net regardless of model behavior.

## Image Output

If image or video generation returns media data, the bot writes temporary artifacts locally and uploads them to Matrix.

## Tool Toggle

Admins can disable all hosted tools and MCP access at runtime with:

```text
.tools off
```

## Country Filter Toggle

When `TOOLS_WEB_SEARCH_COUNTRY` is set, search results are biased to that country by default. Admins can toggle this at runtime:

```text
.country off    # disable country filtering
.country on     # re-enable it
.country toggle # flip the current state
.country        # show current status
```

When disabled, OpenAI's `user_location` is stripped from `web_search` tools and the xAI search-policy instruction is omitted.
