from pathlib import Path

from infinigpt.config import load_config, validate_config, AppConfig, LLMConfig, MatrixConfig


def test_load_config_and_validate(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=X",
                "OPENAI_MODELS=gpt-4o,gpt-4o-mini",
                "DEFAULT_MODEL=gpt-4o",
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
    assert cfg.llm.models["openai"] == ["gpt-4o", "gpt-4o-mini"]
    assert cfg.matrix.channels == ["!r:example.org"]


def test_validate_config_default_model_missing():
    llm = LLMConfig(models={"openai": []}, api_keys={}, default_model="x", personality="p", prompt=["you are ", "."])
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


def test_load_config_supports_both_providers(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=O",
                "XAI_API_KEY=X",
                "OPENAI_MODELS=gpt-5-mini",
                "XAI_MODELS=grok-4",
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
    assert cfg.llm.models["openai"] == ["gpt-5-mini"]
    assert cfg.llm.models["xai"] == ["grok-4"]


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
