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

- OpenAI supports `web_search`, `code_interpreter`, and `image_generation`.
- OpenAI `web_search` also supports `user_location`; this project maps `TOOLS_WEB_SEARCH_COUNTRY=US` to `{"user_location":{"type":"approximate","country":"US"}}`.
- xAI documents `web_search`, `x_search`, `code_interpreter`, and remote MCP for the Grok 4 family. This project only attaches those hosted tools to `grok-4*` models.
- xAI function tools are broader than hosted xAI tools. This project exposes local `generate_image`, `edit_image`, and `generate_video` tools backed by Grok Imagine on `grok-4*`, `grok-3*`, and `grok-code-fast-1`.
- xAI's current provider docs do not document a country filter for `web_search`, and `x_search` exposes X-specific filters rather than country selection.
- To keep behavior aligned, this project also applies `TOOLS_WEB_SEARCH_COUNTRY=US` as an xAI search-policy instruction whenever `web_search` or `x_search` is enabled.
- `TOOLS_X_SEARCH` is used only when the active model is from xAI.
- `TOOLS_IMAGE_GENERATION` enables OpenAI hosted image generation and xAI Grok Imagine image generation/editing.
- `TOOLS_VIDEO_GENERATION` enables xAI Grok Imagine video generation/editing.

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

## Image Output

If image or video generation returns media data, the bot writes temporary artifacts locally and uploads them to Matrix.

## Tool Toggle

Admins can disable all hosted tools and MCP access at runtime with:

```text
.tools off
```
