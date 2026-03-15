import json

from cryptography.fernet import Fernet

from agent_smithers.history import HistoryStore


def _make_store(tmp_path, key=None, **kwargs):
    """Create a HistoryStore with encrypted persistence pointed at tmp_path."""
    key = key or Fernet.generate_key().decode()
    return HistoryStore(
        "you are ", ".", "helper",
        store_path=str(tmp_path),
        encryption_key=key,
        **kwargs,
    ), key


def test_save_creates_encrypted_file(tmp_path):
    hs, _ = _make_store(tmp_path)
    hs.add("!r", "@u", "user", "hello")

    enc_file = tmp_path / "history.enc"
    assert enc_file.exists()
    # The raw bytes should not contain the plaintext
    raw = enc_file.read_bytes()
    assert b"hello" not in raw


def test_load_restores_history(tmp_path):
    hs, key = _make_store(tmp_path)
    hs.add("!r", "@u", "user", "hello")
    hs.add("!r", "@u", "assistant", "hi there")

    # Create a new store with the same key — should restore
    hs2, _ = _make_store(tmp_path, key=key)
    msgs = hs2.messages.get("!r", {}).get("@u", [])
    assert len(msgs) == 3  # system + user + assistant
    assert msgs[1]["content"] == "hello"
    assert msgs[2]["content"] == "hi there"


def test_wrong_key_starts_empty(tmp_path):
    hs, _ = _make_store(tmp_path)
    hs.add("!r", "@u", "user", "secret")

    # Load with a different key
    wrong_key = Fernet.generate_key().decode()
    hs2, _ = _make_store(tmp_path, key=wrong_key)
    assert hs2.messages == {}


def test_clear_all_persists(tmp_path):
    hs, key = _make_store(tmp_path)
    hs.add("!r", "@u", "user", "hello")
    hs.clear_all()

    hs2, _ = _make_store(tmp_path, key=key)
    assert hs2.messages == {}


def test_reset_persists(tmp_path):
    hs, key = _make_store(tmp_path)
    hs.add("!r", "@u", "user", "hello")
    hs.reset("!r", "@u", stock=False)

    hs2, _ = _make_store(tmp_path, key=key)
    msgs = hs2.messages["!r"]["@u"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


def test_init_prompt_persists(tmp_path):
    hs, key = _make_store(tmp_path)
    hs.init_prompt("!r", "@u", custom="Be a pirate")

    hs2, _ = _make_store(tmp_path, key=key)
    msgs = hs2.messages["!r"]["@u"]
    assert msgs[0]["content"] == "Be a pirate"


def test_no_persistence_without_key(tmp_path):
    """Without encryption_key, no file is created."""
    hs = HistoryStore("you are ", ".", "helper", store_path=str(tmp_path))
    hs.add("!r", "@u", "user", "hello")
    assert not (tmp_path / "history.enc").exists()


def test_multiple_rooms_and_users_persist(tmp_path):
    hs, key = _make_store(tmp_path)
    hs.add("!r1", "@u1", "user", "msg1")
    hs.add("!r1", "@u2", "user", "msg2")
    hs.add("!r2", "@u1", "user", "msg3")

    hs2, _ = _make_store(tmp_path, key=key)
    assert hs2.messages["!r1"]["@u1"][1]["content"] == "msg1"
    assert hs2.messages["!r1"]["@u2"][1]["content"] == "msg2"
    assert hs2.messages["!r2"]["@u1"][1]["content"] == "msg3"
