# LM Studio Setup

Agent Smithers supports local models served by [LM Studio](https://lmstudio.ai) via its compatible local server.

## Prerequisites

- LM Studio installed and running with the local server enabled (default: `http://127.0.0.1:1234`)
- At least one model loaded in LM Studio

## Configuration

Add these to your `.env`:

```env
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODELS=your-model-name
DEFAULT_MODEL=your-model-name
```

`LMSTUDIO_API_KEY` is optional. LM Studio's local server does not require authentication by default, but you can set it if you have configured an API key in LM Studio.

`SERVER_MODELS=true` will cause the bot to fetch the current model list from LM Studio on startup. This is useful if you frequently swap models without restarting the bot.

## Docker

When running in Docker, keep `LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1` in your `.env` — the bot automatically rewrites it to `host.docker.internal` when it detects it is running inside a container. On Linux you must also pass `--add-host=host.docker.internal:host-gateway` (or the Compose equivalent) so the container can reach the host network. See [Docker](docker.md) for the full run command.

Make sure LM Studio's local server is bound to `0.0.0.0` (not just `127.0.0.1`) in LM Studio → Local Server settings.

## Notes

- LM Studio does not support hosted tools (web search, code interpreter, image generation, or MCP). Tool calling is disabled automatically for LMStudio models.
- The bot requires at least one non-empty `user` message in the conversation. A fallback user message is injected automatically if the history contains only a system prompt, which can happen on persona or custom prompt initialization.
- Reasoning model output tags (`<think>`, `<|begin_of_thought|>`) are stripped from responses before they are sent to the room.
- You can run LM Studio alongside xAI — set `DEFAULT_MODEL` to a model from whichever provider you want as the primary, and users can switch with `.mymodel`.
