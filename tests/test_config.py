from pathlib import Path

import pytest

from agent_smithers.config import (
    load_config,
    validate_config,
    AppConfig,
    LLMConfig,
    MatrixConfig,
    provider_for_model,
    _parse_bool,
    _parse_csv,
    _parse_json,
    _resolve_lmstudio_url,
    load_env_file,
)
from agent_smithers.exceptions import ConfigError


def test_load_config_and_validate(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "XAI_API_KEY=X",
                "XAI_MODELS=grok-4-1-fast-non-reasoning,grok-4",
                "DEFAULT_MODEL=grok-4-1-fast-non-reasoning",
                "BOT_PERSONALITY=helper",
                "BOT_PROMPT_PREFIX=you are ",
                "BOT_PROMPT_SUFFIX=.",
                "HISTORY_SIZE=8",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
                "MATRIX_ADMINS=@admin:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    ok, errs = validate_config(cfg)
    assert ok and not errs
    assert cfg.llm.models["xai"] == ["grok-4-1-fast-non-reasoning", "grok-4"]
    assert cfg.matrix.channels == ["!r:example.org"]


def test_validate_config_default_model_missing():
    llm = LLMConfig(models={"xai": []}, api_keys={}, default_model="x", personality="p", prompt=["you are ", "."])
    matrix = MatrixConfig(server="s", username="u", password="p", channels=["!r"], admin="a")
    cfg = AppConfig(llm=llm, matrix=matrix)
    ok, errs = validate_config(cfg)
    assert not ok and errs


def test_load_config_and_validate_xai(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "XAI_API_KEY=X",
                "XAI_MODELS=grok-4,grok-3-mini",
                "DEFAULT_MODEL=grok-4",
                "TOOLS_X_SEARCH=true",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    ok, errs = validate_config(cfg)
    assert ok and not errs
    assert cfg.llm.models["xai"] == ["grok-4", "grok-3-mini"]
    assert cfg.llm.api_keys["xai"] == "X"
    assert cfg.llm.tools["x_search"] is True
    assert cfg.llm.tools["video_generation"] is True


def test_load_config_reads_video_generation_toggle(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "XAI_API_KEY=X",
                "XAI_MODELS=grok-4",
                "DEFAULT_MODEL=grok-4",
                "TOOLS_VIDEO_GENERATION=false",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    assert cfg.llm.tools["video_generation"] is False


def test_load_config_reads_web_search_country(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "XAI_API_KEY=X",
                "XAI_MODELS=grok-4",
                "DEFAULT_MODEL=grok-4",
                "TOOLS_WEB_SEARCH_COUNTRY=us",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    assert cfg.llm.web_search_country == "US"


def test_load_config_supports_both_providers(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "XAI_API_KEY=X",
                "LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1",
                "XAI_MODELS=grok-4",
                "LMSTUDIO_MODELS=local-model",
                "DEFAULT_MODEL=grok-4",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    ok, errs = validate_config(cfg)
    assert ok and not errs
    assert cfg.llm.models["xai"] == ["grok-4"]
    assert cfg.llm.models["lmstudio"] == ["local-model"]


def test_load_config_and_validate_lmstudio(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1",
                "LMSTUDIO_MODELS=local-model",
                "DEFAULT_MODEL=local-model",
                "MATRIX_SERVER=https://example.org",
                "MATRIX_USERNAME=@bot:example.org",
                "MATRIX_PASSWORD=pw",
                "MATRIX_CHANNELS=!r:example.org",
            ]
        )
    )
    cfg = load_config(str(p))
    ok, errs = validate_config(cfg)
    assert ok and not errs
    assert cfg.llm.models["lmstudio"] == ["local-model"]
    assert cfg.llm.base_urls["lmstudio"] == "http://127.0.0.1:1234/v1"


# --- provider_for_model edge cases ---

def test_provider_for_model_returns_none_for_empty_string():
    assert provider_for_model("", {"xai": ["grok-4"]}) is None


def test_provider_for_model_returns_none_for_whitespace_only():
    assert provider_for_model("   ", {"xai": ["grok-4"]}) is None


def test_provider_for_model_matches_exact_model_in_list():
    models = {"xai": ["grok-4"], "lmstudio": ["local-model"]}
    assert provider_for_model("grok-4", models) == "xai"
    assert provider_for_model("local-model", models) == "lmstudio"


def test_provider_for_model_falls_back_to_prefix_heuristic_for_grok():
    assert provider_for_model("grok-99-turbo", {}) == "xai"


def test_provider_for_model_returns_none_for_unknown_model():
    assert provider_for_model("claude-3", {"xai": ["grok-4"]}) is None


# --- _parse_bool ---

def test_parse_bool_true_variants():
    for val in ("1", "true", "yes", "on", " True ", " YES "):
        assert _parse_bool(val) is True, f"Expected True for {val!r}"


def test_parse_bool_false_variants():
    for val in ("0", "false", "no", "off", "anything"):
        assert _parse_bool(val) is False, f"Expected False for {val!r}"


def test_parse_bool_none_returns_default():
    assert _parse_bool(None) is False
    assert _parse_bool(None, True) is True


# --- _parse_csv ---

def test_parse_csv_splits_values():
    assert _parse_csv("a, b , c") == ["a", "b", "c"]


def test_parse_csv_empty_and_none():
    assert _parse_csv("") == []
    assert _parse_csv(None) == []


def test_parse_csv_strips_trailing_empty():
    assert _parse_csv("a,,b, ,c") == ["a", "b", "c"]


# --- _parse_json ---

def test_parse_json_returns_default_for_empty():
    assert _parse_json("", {}) == {}
    assert _parse_json(None, []) == []


def test_parse_json_parses_valid_json():
    assert _parse_json('{"key": 42}', {}) == {"key": 42}


def test_parse_json_raises_config_error_on_invalid():
    with pytest.raises(ConfigError, match="Invalid JSON"):
        _parse_json("{bad json", {})


# --- load_env_file ---

def test_load_env_file_raises_on_missing(tmp_path):
    with pytest.raises(ConfigError, match="Missing env file"):
        load_env_file(str(tmp_path / "nonexistent.env"))


def test_load_env_file_strips_quotes_and_exports(tmp_path):
    env = tmp_path / ".env"
    env.write_text('export FOO="bar"\nBAZ=\'qux\'\nPLAIN=val\n# comment\nnoequalline\n')
    result = load_env_file(str(env))
    assert result["FOO"] == "bar"
    assert result["BAZ"] == "qux"
    assert result["PLAIN"] == "val"
    assert "noequalline" not in result


# --- validate_config error paths ---

def _base_llm(**overrides):
    defaults = dict(
        models={"xai": ["grok-4"]},
        api_keys={"xai": "X"},
        default_model="grok-4",
        personality="p",
        prompt=["you are ", "."],
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _base_matrix(**overrides):
    defaults = dict(server="s", username="u", password="p", channels=["!r"])
    defaults.update(overrides)
    return MatrixConfig(**defaults)


def test_validate_no_providers_configured():
    llm = LLMConfig(models={}, api_keys={}, default_model="x", personality="p", prompt=["a"])
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("At least one provider" in e for e in errs)


def test_validate_lmstudio_models_without_base_url():
    llm = LLMConfig(
        models={"lmstudio": ["local"]},
        api_keys={},
        default_model="local",
        personality="p",
        prompt=["a"],
    )
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("LMSTUDIO_BASE_URL" in e for e in errs)


def test_validate_missing_matrix_fields():
    llm = _base_llm()
    cfg = AppConfig(llm=llm, matrix=MatrixConfig(server="", username="", password="", channels=[]))
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("MATRIX_SERVER" in e for e in errs)
    assert any("MATRIX_USERNAME" in e for e in errs)
    assert any("MATRIX_PASSWORD" in e for e in errs)
    assert any("MATRIX_CHANNELS" in e for e in errs)


def test_validate_bad_prompt():
    llm = _base_llm(prompt=[])
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("Prompt" in e for e in errs)


def test_validate_tools_not_dict():
    llm = _base_llm(tools="bad")
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("tool settings" in e.lower() for e in errs)


def test_validate_mcp_not_dict():
    llm = _base_llm(mcp_servers="bad")
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("MCP_SERVERS" in e for e in errs)


def test_validate_default_model_no_matching_provider():
    """DEFAULT_MODEL that doesn't match any configured provider should error."""
    llm = LLMConfig(
        models={"xai": ["grok-4"]},
        api_keys={"xai": "X"},
        default_model="claude-3",
        personality="p",
        prompt=["a"],
    )
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("does not match any configured provider" in e for e in errs)


def test_validate_default_model_requires_api_key_for_provider():
    """DEFAULT_MODEL for xai should require matching API key."""
    llm = LLMConfig(
        models={"xai": ["grok-4"]},
        api_keys={"xai": ""},
        default_model="grok-4",
        personality="p",
        prompt=["a"],
    )
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("XAI_API_KEY" in e and "DEFAULT_MODEL" in e for e in errs)


def test_validate_default_model_requires_lmstudio_base_url():
    """DEFAULT_MODEL for lmstudio should require matching base URL."""
    llm = LLMConfig(
        models={"lmstudio": ["local"]},
        api_keys={},
        base_urls={"lmstudio": ""},
        default_model="local",
        personality="p",
        prompt=["a"],
    )
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("LMSTUDIO_BASE_URL" in e and "DEFAULT_MODEL" in e for e in errs)


def test_validate_xai_models_without_api_key():
    llm = LLMConfig(
        models={"xai": ["grok-4"]},
        api_keys={"xai": ""},
        default_model="grok-4",
        personality="p",
        prompt=["a"],
    )
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("XAI_API_KEY" in e for e in errs)


def test_validate_missing_default_model():
    llm = _base_llm(default_model="")
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert not ok
    assert any("DEFAULT_MODEL is required" in e for e in errs)


def test_validate_all_valid_passes():
    """A fully valid config should produce no errors."""
    llm = _base_llm()
    cfg = AppConfig(llm=llm, matrix=_base_matrix())
    ok, errs = validate_config(cfg)
    assert ok
    assert errs == []


def test_provider_for_model_case_insensitive_grok_prefix():
    """Grok prefix heuristic should be case-insensitive."""
    assert provider_for_model("Grok-future", {}) == "xai"
    assert provider_for_model("GROK-5", {}) == "xai"


def test_provider_for_model_none_input():
    """None-ish model input should return None."""
    assert provider_for_model(None, {"xai": ["grok-4"]}) is None


def test_resolve_lmstudio_url_outside_docker():
    """Outside Docker (no /.dockerenv), URL is returned unchanged."""
    # /.dockerenv does not exist in the test environment
    assert _resolve_lmstudio_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


def test_resolve_lmstudio_url_in_docker(tmp_path):
    """Inside Docker (/.dockerenv present), 127.0.0.1/localhost → host.docker.internal."""
    import agent_smithers.config as cfg_mod
    from unittest.mock import patch, MagicMock

    dockerenv = MagicMock()
    dockerenv.exists.return_value = True

    with patch.object(cfg_mod, "Path", return_value=dockerenv):
        assert _resolve_lmstudio_url("http://127.0.0.1:1234/v1") == "http://host.docker.internal:1234/v1"
        assert _resolve_lmstudio_url("http://localhost:1234/v1") == "http://host.docker.internal:1234/v1"


def test_resolve_lmstudio_url_empty():
    assert _resolve_lmstudio_url("") == ""


def test_load_env_file_skips_empty_keys(tmp_path):
    """Lines where key is empty after stripping should be skipped."""
    env = tmp_path / ".env"
    env.write_text("=value\nGOOD=ok\n")
    result = load_env_file(str(env))
    assert "" not in result
    assert result["GOOD"] == "ok"
