"""Tests for handle_persona and handle_custom (cmd_prompt.py).

Coverage strategy:
- handle_persona: sets persona on history, adds "introduce yourself", generates reply, sends response
- handle_persona with empty args: falls back to default_personality
- handle_custom: sets custom system prompt, generates reply
- handle_custom with empty args: returns immediately without sending
- _respond error path: generate_reply raises, sends error message
"""
import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_prompt import handle_persona, handle_custom
from agent_smithers.history import HistoryStore


class FakeMatrix:
    def __init__(self):
        self.sent = []

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))


def _make_ctx(*, generate_reply_result="I am Shakespeare", generate_reply_error=None):
    async def generate_reply(messages, model=None, room_id=None):
        if generate_reply_error:
            raise generate_reply_error
        return generate_reply_result

    cfg_llm = SimpleNamespace(prompt=["you are ", "."])
    cfg = SimpleNamespace(llm=cfg_llm)
    return SimpleNamespace(
        history=HistoryStore("you are ", ".", "helper", max_items=8),
        matrix=FakeMatrix(),
        render=lambda s: None,
        model="gpt-4o",
        default_personality="helper",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text.strip(),
        log=lambda *a, **k: None,
        user_models={},
        cfg=cfg,
    )


def test_handle_persona_sets_persona_and_generates_reply():
    ctx = _make_ctx()
    asyncio.run(handle_persona(ctx, "!r", "@u", "User", "Shakespeare"))
    # The system prompt should contain Shakespeare
    msgs = ctx.history.get("!r", "@u")
    system_msg = msgs[0]
    assert system_msg["role"] == "system"
    assert "Shakespeare" in system_msg["content"]
    # Should have added "introduce yourself" as user message
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert any("introduce yourself" in m["content"] for m in user_msgs)
    # Should have sent a response
    assert len(ctx.matrix.sent) == 1
    room_id, body, _ = ctx.matrix.sent[0]
    assert room_id == "!r"
    assert "I am Shakespeare" in body
    assert "**User**:" in body


def test_handle_persona_empty_args_uses_default_personality():
    ctx = _make_ctx()
    asyncio.run(handle_persona(ctx, "!r", "@u", "User", ""))
    msgs = ctx.history.get("!r", "@u")
    system_msg = msgs[0]
    assert "helper" in system_msg["content"]


def test_handle_persona_whitespace_args_uses_default_personality():
    ctx = _make_ctx()
    asyncio.run(handle_persona(ctx, "!r", "@u", "User", "   "))
    msgs = ctx.history.get("!r", "@u")
    system_msg = msgs[0]
    assert "helper" in system_msg["content"]


def test_handle_custom_sets_exact_system_prompt():
    ctx = _make_ctx()
    asyncio.run(handle_custom(ctx, "!r", "@u", "User", "Be a pirate captain"))
    msgs = ctx.history.get("!r", "@u")
    system_msg = msgs[0]
    assert system_msg["content"] == "Be a pirate captain"
    # Should have generated and sent reply
    assert len(ctx.matrix.sent) == 1
    assert "I am Shakespeare" in ctx.matrix.sent[0][1]


def test_handle_custom_empty_args_does_nothing():
    ctx = _make_ctx()
    asyncio.run(handle_custom(ctx, "!r", "@u", "User", ""))
    assert len(ctx.matrix.sent) == 0
    # History should not be modified beyond initial ensure
    raw = ctx.history.messages.get("!r", {}).get("@u")
    assert raw is None  # Never touched


def test_handle_custom_whitespace_only_does_nothing():
    ctx = _make_ctx()
    asyncio.run(handle_custom(ctx, "!r", "@u", "User", "   "))
    assert len(ctx.matrix.sent) == 0


def test_respond_sends_error_message_when_generate_reply_fails():
    ctx = _make_ctx(generate_reply_error=RuntimeError("API down"))
    asyncio.run(handle_persona(ctx, "!r", "@u", "User", "pirate"))
    # Should have sent "Something went wrong" as error message
    assert len(ctx.matrix.sent) == 1
    assert "Something went wrong" in ctx.matrix.sent[0][1]


def test_handle_persona_uses_per_user_model():
    """When a user has a per-user model set, _respond should pass it to generate_reply."""
    captured = {}

    async def generate_reply(messages, model=None, room_id=None):
        captured["model"] = model
        return "ok"

    ctx = _make_ctx()
    ctx.generate_reply = generate_reply
    ctx.user_models = {"!r": {"@u": "grok-4"}}
    asyncio.run(handle_persona(ctx, "!r", "@u", "User", "pirate"))
    assert captured["model"] == "grok-4"
