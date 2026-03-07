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

    async def generate_reply(messages, model=None, room_id=None, thread_user=None):
        assert thread_user == "@u"
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


def test_handle_model_set_unknown_model_keeps_current():
    """Setting an unknown model should not change ctx.model."""
    ctx = SimpleNamespace(
        model="gpt-4o",
        default_model="gpt-4o",
        models={"openai": ["gpt-4o", "gpt-4o-mini"]},
        render=lambda s: None,
        matrix=FakeMatrix(),
        log=lambda *a, **k: None,
    )
    asyncio.run(handle_model(ctx, "!r", "@u", "Admin", "nonexistent-model"))
    assert ctx.model == "gpt-4o"  # unchanged
    body = ctx.matrix.sent[-1][1]
    assert "gpt-4o" in body  # shows current model in response


def test_handle_ai_with_empty_args_no_user_message_added():
    """When args is empty, no user message should be added to history."""
    async def generate_reply(messages, model=None, room_id=None, thread_user=None):
        assert thread_user == "@u"
        return "response"

    ctx = SimpleNamespace(
        history=HistoryStore("you are ", ".", "helper", max_items=8),
        matrix=FakeMatrix(),
        render=lambda s: None,
        model="gpt-4o",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text.strip(),
        log=lambda *a, **k: None,
        user_models={},
    )
    asyncio.run(handle_ai(ctx, "!r", "@u", "User", ""))
    msgs = ctx.history.get("!r", "@u")
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 0


def test_handle_ai_error_sends_error_message():
    """When generate_reply raises, handle_ai should send an error message."""
    async def generate_reply(messages, model=None, room_id=None, thread_user=None):
        assert thread_user == "@u"
        raise RuntimeError("API failure")

    ctx = SimpleNamespace(
        history=HistoryStore("you are ", ".", "helper", max_items=8),
        matrix=FakeMatrix(),
        render=lambda s: None,
        model="gpt-4o",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text.strip(),
        log=lambda *a, **k: None,
        user_models={},
    )
    asyncio.run(handle_ai(ctx, "!r", "@u", "User", "hello"))
    assert len(ctx.matrix.sent) == 1
    assert "Something went wrong" in ctx.matrix.sent[0][1]


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


def test_handle_help_no_file_shows_fallback(monkeypatch, tmp_path):
    """When no help.md or help.txt exists, fallback text is used."""
    monkeypatch.chdir(tmp_path)
    ctx = SimpleNamespace(render=lambda s: None, matrix=FakeMatrix(), admins=[])
    asyncio.run(handle_help(ctx, "!r", "@u", "User", ""))
    assert ctx.matrix.sent[-1][1] == "See README for usage."


def test_handle_help_non_admin_does_not_get_admin_section(monkeypatch, tmp_path):
    """Non-admin users should only see the public help section."""
    help_file = tmp_path / "help.md"
    help_file.write_text("Public~~~Secret")
    monkeypatch.chdir(tmp_path)
    ctx = SimpleNamespace(render=lambda s: None, matrix=FakeMatrix(), admins=["AdminOnly"])
    asyncio.run(handle_help(ctx, "!r", "@u", "RegularUser", ""))
    assert len(ctx.matrix.sent) == 1
    assert "Public" in ctx.matrix.sent[0][1]
    assert "Secret" not in ctx.matrix.sent[0][1]
