import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_model import handle_model
from agent_smithers.handlers.cmd_ai import handle_ai
from agent_smithers.handlers.cmd_help import handle_help
from agent_smithers.history import HistoryStore


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def test_handle_model_show_set_reset():
    ctx = SimpleNamespace(
        model="gpt-4o",
        default_model="gpt-4o",
        models={"openai": ["gpt-4o", "gpt-4o-mini"], "xai": ["grok-4"]},
        render=lambda s: None,
        matrix=FakeMatrix(),
        log=lambda *a, **k: None,
    )
    asyncio.run(handle_model(ctx, "!r", "@u", "Admin", ""))
    sent_body = ctx.matrix.sent[-1][1]
    assert "Current model" in sent_body
    assert "**OpenAI**: gpt-4o, gpt-4o-mini" in sent_body
    assert "**xAI**: grok-4" in sent_body
    asyncio.run(handle_model(ctx, "!r", "@u", "Admin", "gpt-4o-mini"))
    assert ctx.model == "gpt-4o-mini"
    asyncio.run(handle_model(ctx, "!r", "@u", "Admin", "reset"))
    assert ctx.model == ctx.default_model


def test_handle_ai_strips_think_markers_and_trims():
    content = "<think>plan</think>  Hello  \n"

    async def generate_reply(messages, model=None, room_id=None):
        return content

    ctx = SimpleNamespace(
        history=HistoryStore("you are ", ".", "helper", max_items=8),
        matrix=FakeMatrix(),
        render=lambda s: None,
        model="gpt-4o",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text.replace("<think>plan</think>", "").strip(),
        log=lambda *a, **k: None,
        user_models={},
    )
    asyncio.run(handle_ai(ctx, "!r", "@u", "User", "hello"))
    sent_body = ctx.matrix.sent[-1][1]
    assert "<think>" not in sent_body
    assert sent_body.endswith("Hello")


def test_handle_help_with_admin_split(monkeypatch, tmp_path):
    help_file = tmp_path / "help.md"
    help_file.write_text("User Help~~~Admin Help")
    monkeypatch.chdir(tmp_path)
    ctx = SimpleNamespace(render=lambda s: None, matrix=FakeMatrix(), admins=["AdminUser"]) 
    asyncio.run(handle_help(ctx, "!r", "@u", "User", ""))
    assert ctx.matrix.sent[-1][1].strip() == "User Help"
    ctx.matrix.sent.clear()
    asyncio.run(handle_help(ctx, "!r", "@admin", "AdminUser", ""))
    assert len(ctx.matrix.sent) == 2
    assert ctx.matrix.sent[0][1].strip() == "User Help"
    assert ctx.matrix.sent[1][1].strip() == "Admin Help"
