"""Tests for handle_whitelist (cmd_whitelist.py).

Coverage strategy:
- add: requires an argument, adds to the set, auto-enables the whitelist
- remove: requires an argument, discards from the set (no error if absent)
- list: shows entries when present, friendly message when empty
- bare command defaults to list
- unknown subcommand shows usage
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_whitelist import handle_whitelist


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(whitelist=None, enabled=False):
    return SimpleNamespace(
        video_whitelist=set(whitelist or []),
        video_whitelist_enabled=enabled,
        matrix=FakeMatrix(),
        render=lambda s: None,
    )


def test_whitelist_add_inserts_user():
    ctx = _make_ctx()
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "add @alice:server"))
    assert "@alice:server" in ctx.video_whitelist
    assert "Added" in ctx.matrix.sent[-1][1]


def test_whitelist_add_auto_enables_when_disabled():
    ctx = _make_ctx(enabled=False)
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "add @alice:server"))
    assert ctx.video_whitelist_enabled is True


def test_whitelist_add_keeps_enabled_state():
    ctx = _make_ctx(enabled=True)
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "add @bob:server"))
    assert ctx.video_whitelist_enabled is True


def test_whitelist_add_without_arg_shows_usage():
    ctx = _make_ctx()
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "add"))
    assert "Usage" in ctx.matrix.sent[-1][1]
    assert ctx.video_whitelist == set()


def test_whitelist_remove_discards_user():
    ctx = _make_ctx(whitelist=["@alice:server"])
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "remove @alice:server"))
    assert "@alice:server" not in ctx.video_whitelist
    assert "Removed" in ctx.matrix.sent[-1][1]


def test_whitelist_remove_absent_user_is_noop():
    ctx = _make_ctx(whitelist=["@alice:server"])
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "remove @ghost:server"))
    assert ctx.video_whitelist == {"@alice:server"}


def test_whitelist_remove_without_arg_shows_usage():
    ctx = _make_ctx(whitelist=["@alice:server"])
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "remove"))
    assert "Usage" in ctx.matrix.sent[-1][1]
    assert ctx.video_whitelist == {"@alice:server"}


def test_whitelist_list_shows_sorted_entries():
    ctx = _make_ctx(whitelist=["@bob:server", "@alice:server"])
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "list"))
    body = ctx.matrix.sent[-1][1]
    assert "@alice:server, @bob:server" in body


def test_whitelist_list_empty_shows_friendly_message():
    ctx = _make_ctx()
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "list"))
    assert "empty" in ctx.matrix.sent[-1][1].lower()


def test_whitelist_bare_command_defaults_to_list():
    ctx = _make_ctx(whitelist=["@alice:server"])
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", ""))
    assert "@alice:server" in ctx.matrix.sent[-1][1]


def test_whitelist_unknown_subcommand_shows_usage():
    ctx = _make_ctx()
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "frobnicate x"))
    assert "Usage" in ctx.matrix.sent[-1][1]


def test_whitelist_subcommand_is_case_insensitive():
    ctx = _make_ctx()
    asyncio.run(handle_whitelist(ctx, "!r", "@u", "Admin", "ADD @alice:server"))
    assert "@alice:server" in ctx.video_whitelist
