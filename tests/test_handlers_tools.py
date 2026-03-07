"""Tests for handle_tools (cmd_tools.py).

Coverage strategy:
- status query (empty args, "status" arg)
- enable with "on"/"enable"/"enabled"
- disable with "off"/"disable"/"disabled"
- toggle with unrecognized arg
- verify sent message contains the correct state
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_tools import handle_tools


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(tools_enabled=False):
    return SimpleNamespace(
        tools_enabled=tools_enabled,
        matrix=FakeMatrix(),
        render=lambda s: None,
    )


def test_tools_status_shows_disabled():
    ctx = _make_ctx(tools_enabled=False)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", ""))
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_tools_status_shows_enabled():
    ctx = _make_ctx(tools_enabled=True)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "status"))
    assert "enabled" in ctx.matrix.sent[-1][1]


def test_tools_enable_on():
    ctx = _make_ctx(tools_enabled=False)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "on"))
    assert ctx.tools_enabled is True
    assert "enabled" in ctx.matrix.sent[-1][1]


def test_tools_enable_with_enable_keyword():
    ctx = _make_ctx(tools_enabled=False)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "enable"))
    assert ctx.tools_enabled is True


def test_tools_disable_off():
    ctx = _make_ctx(tools_enabled=True)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "off"))
    assert ctx.tools_enabled is False
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_tools_disable_with_disabled_keyword():
    ctx = _make_ctx(tools_enabled=True)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "disabled"))
    assert ctx.tools_enabled is False


def test_tools_toggle_from_enabled():
    ctx = _make_ctx(tools_enabled=True)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "toggle"))
    assert ctx.tools_enabled is False
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_tools_toggle_from_disabled():
    ctx = _make_ctx(tools_enabled=False)
    asyncio.run(handle_tools(ctx, "!r", "@u", "Admin", "anything"))
    assert ctx.tools_enabled is True
    assert "enabled" in ctx.matrix.sent[-1][1]
