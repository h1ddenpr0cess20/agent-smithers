# Tools and MCP

Agent Smithers uses provider-hosted tools through the Responses API. It does not execute the old local custom tools or the previous `fastmcp` client path.

## Hosted Tools

These are controlled through `.env`:

- `TOOLS_WEB_SEARCH=true`
- `TOOLS_X_SEARCH=true`
- `TOOLS_CODE_INTERPRETER=true`
- `TOOLS_IMAGE_GENERATION=true`

When enabled, the bot sends those tools in the Responses API request and the selected provider runs them on the hosted side.

Provider notes:

- OpenAI supports `web_search`, `code_interpreter`, and `image_generation`.
- xAI supports `web_search`, `x_search`, and `code_interpreter` through its OpenAI-compatible Responses API.
- `TOOLS_X_SEARCH` is used only when the active model is from xAI.
- `TOOLS_IMAGE_GENERATION` is used only when the active model is from OpenAI.

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

If image generation returns image data, the bot writes temporary artifacts locally and uploads them to Matrix.

## Tool Toggle

Admins can disable all hosted tools and MCP access at runtime with:

```text
.tools off
```
