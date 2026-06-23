"""Tests for runtime helpers (runtime.py).

Coverage strategy:
- build_router wires every documented command to the correct handler and
  to the correct (admin vs. non-admin) bucket.
- persist_device_id writes a freshly negotiated device_id back into the
  JSON config, but only when there is a device_id, no existing one, and a
  config path; failures are swallowed rather than raised.
"""
import json
from types import SimpleNamespace

from agent_smithers import runtime
from agent_smithers.handlers.cmd_ai import handle_ai
from agent_smithers.handlers.cmd_country import handle_country
from agent_smithers.handlers.cmd_help import handle_help
from agent_smithers.handlers.cmd_location import handle_location
from agent_smithers.handlers.cmd_model import handle_model
from agent_smithers.handlers.cmd_mymodel import handle_mymodel
from agent_smithers.handlers.cmd_prompt import handle_custom, handle_persona
from agent_smithers.handlers.cmd_reset import handle_clear, handle_reset
from agent_smithers.handlers.cmd_thinking import handle_thinking
from agent_smithers.handlers.cmd_tools import handle_tools
from agent_smithers.handlers.cmd_verbose import handle_verbose
from agent_smithers.handlers.cmd_whitelist import handle_whitelist
from agent_smithers.handlers.cmd_x import handle_x


def test_build_router_registers_regular_commands():
    router = runtime.build_router()
    expected = {
        ".ai": handle_ai,
        ".x": handle_x,
        ".persona": handle_persona,
        ".custom": handle_custom,
        ".reset": handle_reset,
        ".help": handle_help,
        ".location": handle_location,
        ".mymodel": handle_mymodel,
    }
    for cmd, fn in expected.items():
        assert router._handlers.get(cmd) is fn, cmd


def test_build_router_registers_admin_commands():
    router = runtime.build_router()
    expected = {
        ".thinking": handle_thinking,
        ".tools": handle_tools,
        ".verbose": handle_verbose,
        ".model": handle_model,
        ".clear": handle_clear,
        ".whitelist": handle_whitelist,
        ".country": handle_country,
    }
    for cmd, fn in expected.items():
        assert router._admin_handlers.get(cmd) is fn, cmd


def test_build_router_admin_commands_not_in_regular_bucket():
    router = runtime.build_router()
    for cmd in (".thinking", ".tools", ".verbose", ".model", ".clear", ".whitelist", ".country"):
        assert cmd not in router._handlers, cmd


def test_build_router_stock_is_registered_callable():
    router = runtime.build_router()
    assert ".stock" in router._handlers
    assert callable(router._handlers[".stock"])


def test_build_router_stock_dispatches_to_reset_with_stock(monkeypatch):
    captured = {}

    def fake_reset(ctx, room, sender, display, args):
        captured["args"] = args
        return "ok"

    monkeypatch.setattr(runtime, "handle_reset", fake_reset)
    router = runtime.build_router()
    result = router._handlers[".stock"]("ctx", "!r", "@u", "User", "ignored")
    assert result == "ok"
    assert captured["args"] == "stock"


def _ctx_with_device(device_id, configured_device_id, tmp_log=None):
    logs = tmp_log if tmp_log is not None else []
    return SimpleNamespace(
        matrix=SimpleNamespace(client=SimpleNamespace(device_id=device_id)),
        cfg=SimpleNamespace(matrix=SimpleNamespace(device_id=configured_device_id)),
        log=lambda msg: logs.append(msg),
        logger=SimpleNamespace(exception=lambda *a, **k: None),
    )


def test_persist_device_id_writes_when_missing(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"matrix": {"username": "u"}}))
    ctx = _ctx_with_device("DEVICE123", "")

    runtime.persist_device_id(ctx, str(config_file))

    data = json.loads(config_file.read_text())
    assert data["matrix"]["device_id"] == "DEVICE123"
    assert data["matrix"]["username"] == "u"


def test_persist_device_id_creates_matrix_section(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"llm": {}}))
    ctx = _ctx_with_device("DEVICE123", "")

    runtime.persist_device_id(ctx, str(config_file))

    data = json.loads(config_file.read_text())
    assert data["matrix"]["device_id"] == "DEVICE123"
    assert "llm" in data


def test_persist_device_id_skips_when_already_configured(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"matrix": {"device_id": "OLD"}}))
    ctx = _ctx_with_device("DEVICE123", "OLD")

    runtime.persist_device_id(ctx, str(config_file))

    data = json.loads(config_file.read_text())
    assert data["matrix"]["device_id"] == "OLD"


def test_persist_device_id_skips_when_no_config_path(tmp_path):
    ctx = _ctx_with_device("DEVICE123", "")
    runtime.persist_device_id(ctx, None)


def test_persist_device_id_skips_when_no_device_id(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"matrix": {}}))
    ctx = _ctx_with_device(None, "")

    runtime.persist_device_id(ctx, str(config_file))

    data = json.loads(config_file.read_text())
    assert "device_id" not in data["matrix"]


def test_persist_device_id_swallows_errors_on_bad_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("not valid json {{{")
    ctx = _ctx_with_device("DEVICE123", "")

    runtime.persist_device_id(ctx, str(config_file))
