# Development

## Local Setup

- Config lives in `.env`
- Main runtime is `infinigpt/app.py`
- OpenAI API integration is in `infinigpt/llm_client.py`

## Run

```bash
python -m infinigpt --env-file .env
```

## Tests

Focused local verification:

```bash
pytest -q tests/test_cli.py tests/test_config.py tests/test_llm_client.py tests/test_handlers_model_ai_help.py tests/test_handlers_x.py tests/test_tool_calling.py
python3 -m compileall infinigpt
```

## Guidelines

- Keep docs aligned with `.env` config, not JSON config
- Prefer OpenAI Responses API paths over legacy Chat Completions assumptions
- Do not reintroduce local provider or `fastmcp` plumbing unless explicitly intended
