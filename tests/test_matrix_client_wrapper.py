import asyncio
import pytest

import agent_smithers.matrix_client as mc
from types import SimpleNamespace


class FakeAsyncClient:
    def __init__(self, server, username, device_id=None, store_path=None, config=None):
        self.server = server
        self.user = username
        self.device_id = device_id or "D1"
        self.user_id = username
        self.store_path = store_path
        self.config = config
        self.should_upload_keys = True
        self._callbacks = []
        self._to_device_callbacks = []

    async def login(self, password, device_name=None):
        return SimpleNamespace(device_id=self.device_id)

    async def keys_upload(self):
        self.keys_uploaded = True

    def load_store(self):
        self.store_loaded = True

    async def join(self, room_id):
        self.joined = getattr(self, "joined", []) + [room_id]

    async def room_send(self, room_id=None, message_type=None, content=None, ignore_unverified_devices=None):
        self.last_send = SimpleNamespace(room_id=room_id, message_type=message_type, content=content)

    async def upload(self, fp, content_type=None, filename=None, filesize=None):
        self.last_upload = SimpleNamespace(
            content=fp.read(),
            content_type=content_type,
            filename=filename,
            filesize=filesize,
        )
        return SimpleNamespace(content_uri="mxc://example/media"), None

    async def get_displayname(self, user_id):
        return SimpleNamespace(displayname=f"DN:{user_id}")

    def add_event_callback(self, cb, event_type):
        self._callbacks.append((cb, event_type))

    def add_to_device_callback(self, cb, event_types=None):
        self._to_device_callbacks.append((cb, event_types))

    async def sync(self, timeout=None, full_state=None):
        self.synced = True

    async def sync_forever(self, timeout=None, full_state=None):
        self.sync_loop = True


class FakeAsyncClientConfig:
    def __init__(self, encryption_enabled=True, store_sync_tokens=True):
        self.encryption_enabled = encryption_enabled
        self.store_sync_tokens = store_sync_tokens


def test_matrix_client_wrapper_basic(monkeypatch):
    monkeypatch.setattr(mc, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(mc, "AsyncClientConfig", FakeAsyncClientConfig)

    w = mc.MatrixClientWrapper(
        server="https://example.org",
        username="@bot:example.org",
        password="pw",
        device_id="",
        store_path="store",
        encryption_enabled=True,
    )

    asyncio.run(w.login())
    asyncio.run(w.ensure_keys())
    assert getattr(w.client, "keys_uploaded", False) is True
    asyncio.run(w.load_store())
    assert getattr(w.client, "store_loaded", False) is True
    asyncio.run(w.join("!r"))
    asyncio.run(w.send_text("!r", "hello"))
    assert w.client.last_send.content["body"] == "hello"
    asyncio.run(w.send_text("!r", "hello", html="<p>hello</p>"))
    content = w.client.last_send.content
    assert content["formatted_body"] == "<p>hello</p>"
    assert content["format"] == "org.matrix.custom.html"
    dn = asyncio.run(w.display_name("@user:example.org"))
    assert dn.startswith("DN:@user")
    seen = {}

    async def handler(room, event):
        seen["ok"] = True

    w.add_text_handler(handler)
    cb, _ = w.client._callbacks[-1]
    asyncio.run(cb(SimpleNamespace(room_id="!r"), SimpleNamespace(body="hi", sender="@u", server_timestamp=0)))
    assert seen.get("ok") is True
    w.add_to_device_callback(lambda *a, **k: None, None)
    assert w.client._to_device_callbacks


def test_matrix_client_wrapper_send_video(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(mc, "AsyncClientConfig", FakeAsyncClientConfig)

    w = mc.MatrixClientWrapper(
        server="https://example.org",
        username="@bot:example.org",
        password="pw",
        store_path=str(tmp_path / "store"),
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")

    asyncio.run(w.send_video("!r", str(video_path), None, log=lambda *_args, **_kwargs: None))

    assert w.client.last_upload.filename == "clip.mp4"
    assert w.client.last_send.content["msgtype"] == "m.video"
    assert w.client.last_send.content["body"] == "clip.mp4"
