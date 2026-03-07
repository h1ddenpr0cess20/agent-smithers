import pytest

from agent_smithers.handlers.cmd_reset import handle_reset
from agent_smithers.history import HistoryStore


class Ctx:
    def __init__(self):
        # Mirror reference prompt handling
        self.history = HistoryStore(prompt_prefix="you are ", prompt_suffix=".", personality="helper", max_items=8)
        self.bot_id = "Bot"
        self.model = "gpt-4o"
        self.default_model = "gpt-4o"
        self.personality = "helper"
        self.default_personality = "helper"
        self._sent = []

    def render(self, body):
        return None

    async def matrix_send(self, room_id, body, html=None):
        self._sent.append(body)

    @property
    def matrix(self):
        class M:
            def __init__(self, outer):
                self._o = outer

            async def send_text(self, room_id, body, html=None):
                await self._o.matrix_send(room_id, body, html)

        return M(self)

    def log(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_reset_seeds_default_persona():
    ctx = Ctx()
    room = "!r"
    user = "@u"
    # Call reset without stock
    await handle_reset(ctx, room, user, "User", "")
    msgs = ctx.history.get(room, user)
    assert msgs and msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("you are ")


@pytest.mark.asyncio
async def test_reset_stock_clears_history():
    ctx = Ctx()
    room = "!r"
    user = "@u"
    ctx.history.add(room, user, "user", "hello")
    await handle_reset(ctx, room, user, "User", "stock")
    # stock reset leaves history empty (no system prompt seeded)
    raw_msgs = ctx.history.messages[room][user]
    assert raw_msgs == []
    # Sent message should mention "Stock settings"
    assert any("Stock settings" in body for body in ctx._sent)


@pytest.mark.asyncio
async def test_reset_default_sends_bot_id_message():
    ctx = Ctx()
    await handle_reset(ctx, "!r", "@u", "User", "")
    assert any("Bot reset to default" in body for body in ctx._sent)


@pytest.mark.asyncio
async def test_clear_resets_all_and_restores_defaults():
    """handle_clear should clear all history and reset model/personality."""
    from agent_smithers.handlers.cmd_reset import handle_clear
    ctx = Ctx()
    ctx.model = "gpt-4o-mini"
    ctx.default_model = "gpt-4o"
    ctx.personality = "custom-persona"
    ctx.default_personality = "helper"
    ctx.history.add("!r1", "@u1", "user", "a")
    ctx.history.add("!r2", "@u2", "user", "b")
    await handle_clear(ctx, "!r", "@admin", "Admin", "")
    assert ctx.history.messages == {}
    assert ctx.model == "gpt-4o"
    assert ctx.personality == "helper"
    assert any("reset for everyone" in body for body in ctx._sent)

