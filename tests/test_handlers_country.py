"""Tests for handle_country (cmd_country.py).

Coverage strategy:
- status query (empty args, "status" arg)
- enable with "on"/"enable"/"enabled"
- disable with "off"/"disable"/"disabled"
- toggle with unrecognized arg
- no country configured shows informative message
- verify sent message contains the configured country code
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_country import handle_country


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(search_country_enabled=True, country="US"):
    cfg = SimpleNamespace(llm=SimpleNamespace(web_search_country=country))
    return SimpleNamespace(
        search_country_enabled=search_country_enabled,
        cfg=cfg,
        matrix=FakeMatrix(),
        render=lambda s: None,
    )


def test_country_status_shows_enabled():
    ctx = _make_ctx(search_country_enabled=True)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", ""))
    assert "enabled" in ctx.matrix.sent[-1][1]
    assert "US" in ctx.matrix.sent[-1][1]


def test_country_status_shows_disabled():
    ctx = _make_ctx(search_country_enabled=False)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "status"))
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_country_enable_on():
    ctx = _make_ctx(search_country_enabled=False)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "on"))
    assert ctx.search_country_enabled is True
    assert "enabled" in ctx.matrix.sent[-1][1]


def test_country_enable_with_enable_keyword():
    ctx = _make_ctx(search_country_enabled=False)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "enable"))
    assert ctx.search_country_enabled is True


def test_country_disable_off():
    ctx = _make_ctx(search_country_enabled=True)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "off"))
    assert ctx.search_country_enabled is False
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_country_disable_with_disabled_keyword():
    ctx = _make_ctx(search_country_enabled=True)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "disabled"))
    assert ctx.search_country_enabled is False


def test_country_toggle_from_enabled():
    ctx = _make_ctx(search_country_enabled=True)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "toggle"))
    assert ctx.search_country_enabled is False
    assert "disabled" in ctx.matrix.sent[-1][1]


def test_country_toggle_from_disabled():
    ctx = _make_ctx(search_country_enabled=False)
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "anything"))
    assert ctx.search_country_enabled is True
    assert "enabled" in ctx.matrix.sent[-1][1]


def test_country_no_country_configured():
    ctx = _make_ctx(country="")
    asyncio.run(handle_country(ctx, "!r", "@u", "Admin", "on"))
    assert "No search country configured" in ctx.matrix.sent[-1][1]
    # Should not modify the flag
    assert ctx.search_country_enabled is True
