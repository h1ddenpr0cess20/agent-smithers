"""Tests for handle_mymodel (cmd_mymodel.py).

Coverage strategy:
- No args: shows current model and available models
- Valid model arg: sets per-user model for room
- Invalid model arg: shows error with available models
- Per-user model already set: shows correct current model
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_mymodel import handle_mymodel


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx():
    return SimpleNamespace(
        model="gpt-4o",
        models={"openai": ["gpt-4o", "gpt-4o-mini"], "xai": ["grok-4"]},
        user_models={},
        matrix=FakeMatrix(),
        render=lambda s: None,
        log=lambda *a, **k: None,
    )


def test_mymodel_no_args_shows_current_and_available():
    ctx = _make_ctx()
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", ""))
    body = ctx.matrix.sent[-1][1]
    assert "gpt-4o" in body
    assert "gpt-4o-mini" in body
    assert "grok-4" in body
    assert "Your current model" in body
    assert "Available models" in body


def test_mymodel_no_args_shows_per_user_model_when_set():
    ctx = _make_ctx()
    ctx.user_models = {"!r": {"@u": "grok-4"}}
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", ""))
    body = ctx.matrix.sent[-1][1]
    assert "grok-4" in body


def test_mymodel_valid_model_sets_per_user():
    ctx = _make_ctx()
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", "grok-4"))
    assert ctx.user_models["!r"]["@u"] == "grok-4"
    body = ctx.matrix.sent[-1][1]
    assert "grok-4" in body
    assert "User" in body


def test_mymodel_invalid_model_shows_error():
    ctx = _make_ctx()
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", "nonexistent-model"))
    body = ctx.matrix.sent[-1][1]
    assert "not found" in body
    assert "gpt-4o" in body  # shows available models
    # Should NOT have set the model
    assert "!r" not in ctx.user_models or "@u" not in ctx.user_models.get("!r", {})


def test_mymodel_second_set_overwrites_first():
    ctx = _make_ctx()
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", "grok-4"))
    asyncio.run(handle_mymodel(ctx, "!r", "@u", "User", "gpt-4o-mini"))
    assert ctx.user_models["!r"]["@u"] == "gpt-4o-mini"
