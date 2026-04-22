import asyncio
from types import SimpleNamespace

from agent_smithers.handlers.cmd_x import handle_x
from agent_smithers.history import HistoryStore


class FakeMatrix:
    def __init__(self):
        self.sent = []
        self.names = {}

    async def send_text(self, room_id, body, html=None):
        self.sent.append((room_id, body, html))

    async def display_name(self, user_id):
        return self.names.get(user_id)


def test_x_supports_display_names_with_spaces():
    matrix = FakeMatrix()
    matrix.names = {
        "@john:hs": "John Doe",
        "@jane:hs": "Jane",
    }
    history = HistoryStore("you are ", ".", "helper")
    history.init_prompt("!r", "@john:hs")
    history.init_prompt("!r", "@jane:hs")

    async def generate_reply(messages, model=None, room_id=None, thread_user=None):
        assert thread_user == "@john:hs"
        return "got it"

    ctx = SimpleNamespace(
        history=history,
        matrix=matrix,
        render=lambda s: None,
        model="gpt-4o",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text,
        log=lambda *a, **k: None,
        user_models={},
    )

    asyncio.run(handle_x(ctx, "!r", "@sender:hs", "Sender", "John Doe hello there"))

    assert matrix.sent
    assert matrix.sent[-1][1] == "**Sender**:\ngot it"
    assert history.get("!r", "@john:hs")[-2] == {"role": "user", "content": "hello there"}


def test_x_keeps_matrix_id_targeting():
    matrix = FakeMatrix()
    history = HistoryStore("you are ", ".", "helper")

    async def generate_reply(messages, model=None, room_id=None, thread_user=None):
        assert thread_user == "@target:hs"
        return "got it"

    ctx = SimpleNamespace(
        history=history,
        matrix=matrix,
        render=lambda s: None,
        model="gpt-4o",
        generate_reply=generate_reply,
        clean_response_text=lambda text, sender_display, sender_id: text,
        log=lambda *a, **k: None,
        user_models={},
    )

    asyncio.run(handle_x(ctx, "!r", "@sender:hs", "Sender", "@target:hs hello"))

    assert matrix.sent
    assert history.get("!r", "@target:hs")[-2] == {"role": "user", "content": "hello"}
