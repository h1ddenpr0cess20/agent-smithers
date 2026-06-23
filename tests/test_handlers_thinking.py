"""Tests for handle_thinking (cmd_thinking.py).

Coverage strategy:
- status query (empty args) shows current state
- on/off set the flag explicitly
- toggle flips the current state
- invalid arg shows usage and leaves state unchanged
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_thinking import handle_thinking


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(thinking=False):
    return SimpleNamespace(
        thinking=thinking,
        matrix=FakeMatrix(),
        render=lambda s: None,
    )


def test_thinking_status_shows_off_by_default():
    ctx = _make_ctx(thinking=False)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", ""))
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_thinking_status_shows_on_when_set():
    ctx = _make_ctx(thinking=True)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", ""))
    assert "ON" in ctx.matrix.sent[-1][1]


def test_thinking_status_does_not_mutate_state():
    ctx = _make_ctx(thinking=True)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", ""))
    assert ctx.thinking is True


def test_thinking_on_sets_true():
    ctx = _make_ctx(thinking=False)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "on"))
    assert ctx.thinking is True
    assert "ON" in ctx.matrix.sent[-1][1]


def test_thinking_off_sets_false():
    ctx = _make_ctx(thinking=True)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "off"))
    assert ctx.thinking is False
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_thinking_toggle_from_off():
    ctx = _make_ctx(thinking=False)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "toggle"))
    assert ctx.thinking is True
    assert "ON" in ctx.matrix.sent[-1][1]


def test_thinking_toggle_from_on():
    ctx = _make_ctx(thinking=True)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "toggle"))
    assert ctx.thinking is False
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_thinking_arg_is_case_insensitive():
    ctx = _make_ctx(thinking=False)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "ON"))
    assert ctx.thinking is True


def test_thinking_invalid_arg_shows_usage_and_keeps_state():
    ctx = _make_ctx(thinking=False)
    asyncio.run(handle_thinking(ctx, "!r", "@u", "Admin", "badarg"))
    assert "Usage" in ctx.matrix.sent[-1][1]
    assert ctx.thinking is False
