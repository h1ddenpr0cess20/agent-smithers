# LM Studio Setup

Agent Smithers supports local models served by [LM Studio](https://lmstudio.ai) via its OpenAI-compatible local server.

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

## Notes

- LM Studio does not support hosted tools (web search, code interpreter, image generation, or MCP). Tool calling is disabled automatically for LMStudio models.
- The bot requires at least one non-empty `user` message in the conversation. A fallback user message is injected automatically if the history contains only a system prompt, which can happen on persona or custom prompt initialization.
- Reasoning model output tags (`<think>`, `<|begin_of_thought|>`) are stripped from responses before they are sent to the room.
- You can run LM Studio alongside OpenAI or xAI — set `DEFAULT_MODEL` to a model from whichever provider you want as the primary, and users can switch with `.mymodel`.
