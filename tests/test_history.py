from agent_smithers.history import HistoryStore


def test_history_prompt_and_trim():
    hs = HistoryStore("you are ", ".", "helper", max_items=5)
    room = "!r:server"
    user = "@u:server"
    msgs = hs.get(room, user)
    assert msgs[0]["role"] == "system"
    for i in range(10):
        hs.add(room, user, "user", f"m{i}")
    msgs = hs.get(room, user)
    assert len(msgs) <= 5
    assert msgs[0]["role"] in ("system", "user")


def test_system_prompt_content_matches_prefix_personality_suffix():
    hs = HistoryStore("assume the role of ", ".", "Bob")
    msgs = hs.get("!r", "@u")
    assert msgs[0]["content"] == "assume the role of Bob."


def test_init_prompt_with_persona_replaces_history():
    hs = HistoryStore("you are ", ".", "default")
    hs.add("!r", "@u", "user", "hello")
    hs.init_prompt("!r", "@u", persona="Shakespeare")
    msgs = hs.get("!r", "@u")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert "Shakespeare" in msgs[0]["content"]
    assert "default" not in msgs[0]["content"]


def test_init_prompt_with_custom_sets_exact_text():
    hs = HistoryStore("you are ", ".", "default")
    hs.init_prompt("!r", "@u", custom="Be a pirate captain")
    msgs = hs.get("!r", "@u")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Be a pirate captain"


def test_reset_with_stock_leaves_empty():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r", "@u", "user", "hello")
    hs.reset("!r", "@u", stock=True)
    msgs = hs.get("!r", "@u")
    # get() calls _ensure which re-adds a system message
    # but reset with stock=True clears to empty list
    assert hs.messages["!r"]["@u"] == []


def test_reset_without_stock_seeds_default():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r", "@u", "user", "hello")
    hs.reset("!r", "@u", stock=False)
    msgs = hs.messages["!r"]["@u"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert "helper" in msgs[0]["content"]


def test_clear_is_alias_for_stock_reset():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r", "@u", "user", "hello")
    hs.clear("!r", "@u")
    assert hs.messages["!r"]["@u"] == []


def test_clear_all_removes_everything():
    hs = HistoryStore("you are ", ".", "helper")
    hs.add("!r1", "@u1", "user", "a")
    hs.add("!r2", "@u2", "user", "b")
    hs.clear_all()
    assert hs.messages == {}


def test_set_verbose_omits_extra_suffix():
    hs = HistoryStore("you are ", ".", "helper", prompt_suffix_extra=" keep it brief")
    msgs_default = hs.get("!r", "@u")
    assert " keep it brief" in msgs_default[0]["content"]
    hs.set_verbose(True)
    # Need new room/user to regenerate prompt
    msgs_verbose = hs.get("!r2", "@u2")
    assert " keep it brief" not in msgs_verbose[0]["content"]
    assert msgs_verbose[0]["content"].endswith(".")


def test_set_verbose_false_includes_extra_suffix():
    hs = HistoryStore("you are ", ".", "helper", prompt_suffix_extra=" keep it brief")
    hs.set_verbose(True)
    hs.set_verbose(False)
    msgs = hs.get("!r", "@u")
    assert " keep it brief" in msgs[0]["content"]


def test_fixed_system_prompt_constructor():
    hs = HistoryStore(system_prompt="Fixed prompt text", history_size=10)
    msgs = hs.get("!r", "@u")
    assert msgs[0]["content"] == "Fixed prompt text"
    assert hs.max_items == 10


def test_trim_preserves_system_message():
    hs = HistoryStore("you are ", ".", "helper", max_items=3)
    room, user = "!r", "@u"
    hs.add(room, user, "user", "m1")
    hs.add(room, user, "assistant", "r1")
    hs.add(room, user, "user", "m2")
    hs.add(room, user, "assistant", "r2")
    msgs = hs.get(room, user)
    assert len(msgs) <= 3
    assert msgs[0]["role"] == "system"


def test_get_returns_copy():
    hs = HistoryStore("you are ", ".", "helper")
    msgs = hs.get("!r", "@u")
    msgs.append({"role": "user", "content": "injected"})
    assert len(hs.get("!r", "@u")) == 1  # unchanged

