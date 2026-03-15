from cryptography.fernet import Fernet

from agent_smithers.history import HistoryStore


def test_location_appended_to_system_prompt():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "Tokyo, Japan")
    msgs = hs.get("!r", "@u")
    assert msgs[0]["role"] == "system"
    assert "The user is located in Tokyo, Japan." in msgs[0]["content"]


def test_location_included_in_new_threads():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "Berlin")
    # New thread should include location
    msgs = hs.get("!r2", "@u")
    assert "Berlin" in msgs[0]["content"]


def test_location_clear_removes_from_prompt():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "Paris")
    msgs = hs.get("!r", "@u")
    assert "Paris" in msgs[0]["content"]

    hs.set_location("@u", "")
    msgs = hs.get("!r", "@u")
    assert "located in" not in msgs[0]["content"]


def test_location_updates_existing_threads():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r", "@u", "user", "hello")
    hs.set_location("@u", "NYC")
    msgs = hs.messages["!r"]["@u"]
    assert "NYC" in msgs[0]["content"]


def test_location_with_custom_prompt():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "London")
    hs.init_prompt("!r", "@u", custom="Be a pirate")
    msgs = hs.get("!r", "@u")
    assert msgs[0]["content"] == "Be a pirate The user is located in London."


def test_location_with_persona():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "Sydney")
    hs.init_prompt("!r", "@u", persona="Shakespeare")
    msgs = hs.get("!r", "@u")
    assert "Shakespeare" in msgs[0]["content"]
    assert "Sydney" in msgs[0]["content"]


def test_location_persists_with_encryption(tmp_path):
    key = Fernet.generate_key().decode()
    hs = HistoryStore("you are ", ".", "helper", store_path=str(tmp_path), encryption_key=key)
    hs.set_location("@u", "Tokyo")
    hs.add("!r", "@u", "user", "hello")

    # Reload from disk
    hs2 = HistoryStore("you are ", ".", "helper", store_path=str(tmp_path), encryption_key=key)
    assert hs2.get_location("@u") == "Tokyo"
    msgs = hs2.messages["!r"]["@u"]
    assert "Tokyo" in msgs[0]["content"]


def test_location_survives_clear_all():
    hs = HistoryStore("you are ", ".", "helper")
    hs.set_location("@u", "Berlin")
    hs.add("!r", "@u", "user", "hello")
    hs.clear_all()
    assert hs.get_location("@u") == "Berlin"
    # New thread after clear should still have location
    msgs = hs.get("!r", "@u")
    assert "Berlin" in msgs[0]["content"]


def test_no_location_no_suffix():
    hs = HistoryStore("you are ", ".", "helper")
    msgs = hs.get("!r", "@u")
    assert "located in" not in msgs[0]["content"]


def test_location_change_updates_all_rooms():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r1", "@u", "user", "hi")
    hs.add("!r2", "@u", "user", "hi")
    hs.set_location("@u", "Mars")
    assert "Mars" in hs.messages["!r1"]["@u"][0]["content"]
    assert "Mars" in hs.messages["!r2"]["@u"][0]["content"]
