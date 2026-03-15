import asyncio
from unittest.mock import AsyncMock

from agent_smithers.history import HistoryStore
from agent_smithers.handlers.cmd_location import handle_location


class Ctx:
    def __init__(self):
        self.history = HistoryStore("you are ", ".", "helper")
        self.matrix = AsyncMock()
        self.log = lambda *a, **kw: None

    def render(self, body):
        return None


def test_location_no_args_shows_no_location():
    ctx = Ctx()
    asyncio.run(handle_location(ctx, "!r", "@u", "Alice", ""))
    body = ctx.matrix.send_text.call_args[0][1]
    assert "no location set" in body


def test_location_set():
    ctx = Ctx()
    asyncio.run(handle_location(ctx, "!r", "@u", "Alice", "Tokyo, Japan"))
    body = ctx.matrix.send_text.call_args[0][1]
    assert "Tokyo, Japan" in body
    assert ctx.history.get_location("@u") == "Tokyo, Japan"


def test_location_show_after_set():
    ctx = Ctx()
    ctx.history.set_location("@u", "Berlin, Germany")
    asyncio.run(handle_location(ctx, "!r", "@u", "Alice", ""))
    body = ctx.matrix.send_text.call_args[0][1]
    assert "Berlin, Germany" in body


def test_location_clear():
    ctx = Ctx()
    ctx.history.set_location("@u", "Paris")
    asyncio.run(handle_location(ctx, "!r", "@u", "Alice", "clear"))
    assert ctx.history.get_location("@u") is None
    body = ctx.matrix.send_text.call_args[0][1]
    assert "cleared" in body.lower()


def test_location_clear_variants():
    for word in ("remove", "reset", "none"):
        ctx = Ctx()
        ctx.history.set_location("@u", "Paris")
        asyncio.run(handle_location(ctx, "!r", "@u", "Alice", word))
        assert ctx.history.get_location("@u") is None
