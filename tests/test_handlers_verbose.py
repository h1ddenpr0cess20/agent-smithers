"""Tests for handle_verbose (cmd_verbose.py).

Coverage strategy:
- status query (empty args, "status")
- on/off/toggle with various aliases
- invalid arg shows usage
- set_verbose is called on history when available
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_verbose import handle_verbose
from agent_smithers.history import HistoryStore


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(verbose=False, with_history=True):
    ctx = SimpleNamespace(
        verbose=verbose,
        matrix=FakeMatrix(),
        render=lambda s: None,
    )
    if with_history:
        ctx.history = HistoryStore("you are ", ".", "helper", prompt_suffix_extra=" keep it brief")
    return ctx


def test_verbose_status_shows_off_by_default():
    ctx = _make_ctx(verbose=False)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", ""))
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_verbose_status_shows_on_when_set():
    ctx = _make_ctx(verbose=True)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "status"))
    assert "ON" in ctx.matrix.sent[-1][1]


def test_verbose_on_sets_verbose_true():
    ctx = _make_ctx(verbose=False)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "on"))
    assert ctx.verbose is True
    assert "ON" in ctx.matrix.sent[-1][1]


def test_verbose_enable_sets_verbose_true():
    ctx = _make_ctx(verbose=False)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "enable"))
    assert ctx.verbose is True


def test_verbose_off_sets_verbose_false():
    ctx = _make_ctx(verbose=True)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "off"))
    assert ctx.verbose is False
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_verbose_false_keyword():
    ctx = _make_ctx(verbose=True)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "false"))
    assert ctx.verbose is False


def test_verbose_toggle_from_off():
    ctx = _make_ctx(verbose=False)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "toggle"))
    assert ctx.verbose is True
    assert "ON" in ctx.matrix.sent[-1][1]


def test_verbose_toggle_from_on():
    ctx = _make_ctx(verbose=True)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "switch"))
    assert ctx.verbose is False
    assert "OFF" in ctx.matrix.sent[-1][1]


def test_verbose_invalid_arg_shows_usage():
    ctx = _make_ctx(verbose=False)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "badarg"))
    assert "Usage" in ctx.matrix.sent[-1][1]
    # verbose should remain unchanged
    assert ctx.verbose is False


def test_verbose_on_calls_set_verbose_on_history():
    ctx = _make_ctx(verbose=False, with_history=True)
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "on"))
    assert ctx.verbose is True
    # The history should now have _include_extra == False (verbose is on)
    assert ctx.history._include_extra is False


def test_verbose_off_calls_set_verbose_on_history():
    ctx = _make_ctx(verbose=True, with_history=True)
    ctx.history.set_verbose(True)  # simulate prior state
    asyncio.run(handle_verbose(ctx, "!r", "@u", "Admin", "off"))
    assert ctx.verbose is False
    assert ctx.history._include_extra is True
