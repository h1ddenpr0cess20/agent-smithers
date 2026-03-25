# Legacy Map

The current codebase keeps the upstream Matrix bot structure, but several major pieces changed:

- `config.json` -> `.env`
- provider routing -> xAI/LM Studio client
- Chat Completions loop -> Responses API
- local builtin tools -> provider-hosted tools
- `fastmcp` execution -> hosted remote MCP tool definitions

Most handler and Matrix runtime structure remains recognizable from upstream.
