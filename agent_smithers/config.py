"""Configuration loading, parsing, and validation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import ConfigError


SUPPORTED_PROVIDERS = {"openai", "xai", "lmstudio", "ollama"}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a truthy string from the environment into a bool.

    Args:
        value: Raw string value, or ``None`` when the variable is unset.
        default: Value to return when ``value`` is ``None``.

    Returns:
        ``True`` if the value is one of ``1``/``true``/``yes``/``on``
        (case-insensitive), otherwise ``False`` (or ``default`` when unset).
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> List[str]:
    """Parse a comma-separated string into a list of trimmed, non-empty items.

    Args:
        value: Raw comma-separated string, or ``None``.

    Returns:
        A list of trimmed entries, or an empty list when the value is empty.
    """
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_json(value: str | None, default: Any) -> Any:
    """Parse a JSON-encoded string, falling back to a default when empty.

    Args:
        value: Raw JSON string, or ``None``/empty for the default.
        default: Value returned when ``value`` is empty.

    Returns:
        The decoded JSON object, or ``default`` when no value is provided.

    Raises:
        ConfigError: If ``value`` is present but not valid JSON.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON value: {value}") from exc


def load_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from an env file."""
    env_path = Path(path) if path else Path(".env")
    if not env_path.exists():
        raise ConfigError(f"Missing env file: {env_path}")

    loaded: Dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
        os.environ[key] = value
    return loaded


@dataclass
class MatrixConfig:
    """Matrix connection, identity, and access-control settings."""

    server: str
    username: str
    password: str
    channels: List[str]
    admin: str = ""
    admins: List[str] = field(default_factory=list)
    device_id: str = ""
    store_path: str = "store"
    e2e: bool = True
    video_whitelist: List[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    """Provider, model, prompt, tool, and history settings for the LLM layer."""

    models: Dict[str, List[str]]
    api_keys: Dict[str, str]
    default_model: str
    personality: str
    prompt: List[str]
    base_urls: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    web_search_country: str = ""
    server_models: bool = True
    history_tokens: int = 8192
    history_encryption_key: str = ""
    mcp_servers: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300


@dataclass
class AppConfig:
    """Top-level application config bundling the LLM and Matrix sections."""

    llm: LLMConfig
    matrix: MatrixConfig
    markdown: bool = True
    thinking: bool = False


def provider_for_model(model: str, models: Dict[str, List[str]]) -> Optional[str]:
    """Resolve which provider owns a model id.

    Prefers an explicit match in the configured ``models`` map, then falls
    back to prefix heuristics (``grok-`` → xAI, ``gpt-``/``o1``/``o3``/``o4``
    → OpenAI).

    Args:
        model: Model identifier to resolve.
        models: Mapping of provider key to its list of configured model ids.

    Returns:
        The provider key, or ``None`` if the model cannot be resolved.
    """
    selected = str(model or "").strip()
    if not selected:
        return None
    for provider, provider_models in models.items():
        if selected in provider_models:
            return provider

    lowered = selected.lower()
    if lowered.startswith("grok-"):
        return "xai"
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def validate_config(cfg: AppConfig) -> Tuple[bool, List[str]]:
    """Validate a fully assembled application config.

    Checks that at least one provider is configured, that keyed providers
    have their required API keys or base URLs, and that a default model is set.

    Args:
        cfg: The assembled application configuration to validate.

    Returns:
        A tuple of ``(ok, errors)`` where ``ok`` is ``True`` only when
        ``errors`` is empty.
    """
    errors: List[str] = []

    configured_providers = []
    for provider in sorted(SUPPORTED_PROVIDERS):
        if cfg.llm.models.get(provider) or cfg.llm.api_keys.get(provider) or cfg.llm.base_urls.get(provider):
            configured_providers.append(provider)
        if provider in {"openai", "xai"} and cfg.llm.models.get(provider) and not cfg.llm.api_keys.get(provider):
            errors.append(f"{provider.upper()}_API_KEY is required when {provider.upper()}_MODELS is set")
        if provider == "lmstudio" and cfg.llm.models.get(provider) and not cfg.llm.base_urls.get(provider):
            errors.append("LMSTUDIO_BASE_URL is required when LMSTUDIO_MODELS is set")
        if provider == "ollama" and cfg.llm.models.get(provider) and not cfg.llm.base_urls.get(provider):
            errors.append("OLLAMA_BASE_URL is required when OLLAMA_MODELS is set")
    if not configured_providers:
        errors.append("At least one provider must be configured")
    if not cfg.llm.default_model:
        errors.append("DEFAULT_MODEL is required")
    else:
        provider = provider_for_model(cfg.llm.default_model, cfg.llm.models)
        if not provider:
            errors.append(
                f"DEFAULT_MODEL '{cfg.llm.default_model}' does not match any configured provider"
            )
        elif provider in {"openai", "xai"} and not cfg.llm.api_keys.get(provider):
            errors.append(f"{provider.upper()}_API_KEY is required for DEFAULT_MODEL '{cfg.llm.default_model}'")
        elif provider == "lmstudio" and not cfg.llm.base_urls.get(provider):
            errors.append(f"LMSTUDIO_BASE_URL is required for DEFAULT_MODEL '{cfg.llm.default_model}'")
        elif provider == "ollama" and not cfg.llm.base_urls.get(provider):
            errors.append(f"OLLAMA_BASE_URL is required for DEFAULT_MODEL '{cfg.llm.default_model}'")
    if not (isinstance(cfg.llm.prompt, list) and len(cfg.llm.prompt) >= 1):
        errors.append("Prompt settings must produce at least one prompt string")
    if not isinstance(cfg.llm.tools, dict):
        errors.append("Hosted tool settings must be a mapping")
    if not isinstance(cfg.llm.mcp_servers, dict):
        errors.append("MCP_SERVERS must be a JSON object when provided")
    if not cfg.matrix.server:
        errors.append("MATRIX_SERVER is required")
    if not cfg.matrix.username:
        errors.append("MATRIX_USERNAME is required")
    if not cfg.matrix.password:
        errors.append("MATRIX_PASSWORD is required")
    if not cfg.matrix.channels:
        errors.append("MATRIX_CHANNELS must contain at least one room")

    return (len(errors) == 0, errors)


def _resolve_lmstudio_url(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal when running inside Docker."""
    if not url or not Path("/.dockerenv").exists():
        return url
    return url.replace("127.0.0.1", "host.docker.internal").replace(
        "localhost", "host.docker.internal"
    )


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load application config from a .env file and environment variables."""
    load_env_file(path)

    openai_models = _parse_csv(os.getenv("OPENAI_MODELS"))
    xai_models = _parse_csv(os.getenv("XAI_MODELS"))
    lmstudio_models = _parse_csv(os.getenv("LMSTUDIO_MODELS"))
    ollama_models = _parse_csv(os.getenv("OLLAMA_MODELS"))
    default_model = os.getenv(
        "DEFAULT_MODEL",
        openai_models[0]
        if openai_models
        else (xai_models[0] if xai_models else (lmstudio_models[0] if lmstudio_models else (ollama_models[0] if ollama_models else ""))),
    )
    prompt_prefix = os.getenv("BOT_PROMPT_PREFIX", "assume the personality of ")
    prompt_suffix = os.getenv(
        "BOT_PROMPT_SUFFIX",
        ". roleplay and never break character, never say you are not actually this character or refuse to play the character.",
    )
    prompt_extra = os.getenv("BOT_PROMPT_SUFFIX_EXTRA", " keep your responses relatively short.")
    prompt = [prompt_prefix, prompt_suffix]
    if prompt_extra:
        prompt.append(prompt_extra)

    tools: Dict[str, Any] = {
        "web_search": _parse_bool(os.getenv("TOOLS_WEB_SEARCH"), True),
        "code_interpreter": _parse_bool(os.getenv("TOOLS_CODE_INTERPRETER"), True),
        "x_search": _parse_bool(os.getenv("TOOLS_X_SEARCH"), True),
        "image_generation": _parse_bool(os.getenv("TOOLS_IMAGE_GENERATION"), True),
        "video_generation": _parse_bool(os.getenv("TOOLS_VIDEO_GENERATION"), True),
    }

    llm = LLMConfig(
        models={
            "openai": openai_models,
            "xai": xai_models,
            "lmstudio": lmstudio_models,
            "ollama": ollama_models,
        },
        api_keys={
            "openai": os.getenv("OPENAI_API_KEY", "").strip(),
            "xai": os.getenv("XAI_API_KEY", "").strip(),
            "lmstudio": os.getenv("LMSTUDIO_API_KEY", "").strip(),
        },
        base_urls={
            "lmstudio": _resolve_lmstudio_url(os.getenv("LMSTUDIO_BASE_URL", "").strip()),
            "ollama": os.getenv("OLLAMA_BASE_URL", "").strip(),
        },
        default_model=default_model,
        personality=os.getenv(
            "BOT_PERSONALITY",
            "an AI that can assume any personality, named Agent Smithers",
        ),
        prompt=prompt,
        options=_parse_json(os.getenv("RESPONSES_OPTIONS"), {}),
        tools=tools,
        web_search_country=os.getenv("TOOLS_WEB_SEARCH_COUNTRY", "").strip().upper(),
        server_models=_parse_bool(os.getenv("SERVER_MODELS"), True),
        history_tokens=int(os.getenv("HISTORY_TOKENS", "8192")),
        history_encryption_key=os.getenv("HISTORY_ENCRYPTION_KEY", "").strip(),
        mcp_servers=_parse_json(os.getenv("MCP_SERVERS"), {}),
        timeout=int(
            os.getenv("LLM_TIMEOUT")
            or os.getenv("OPENAI_TIMEOUT")
            or os.getenv("XAI_TIMEOUT")
            or "180"
        ),
    )

    admins = _parse_csv(os.getenv("MATRIX_ADMINS"))
    video_whitelist = _parse_csv(os.getenv("VIDEO_WHITELIST"))
    matrix = MatrixConfig(
        server=os.getenv("MATRIX_SERVER", "").strip(),
        username=os.getenv("MATRIX_USERNAME", "").strip(),
        password=os.getenv("MATRIX_PASSWORD", "").strip(),
        channels=_parse_csv(os.getenv("MATRIX_CHANNELS")),
        admins=admins,
        admin=admins[0] if admins else "",
        device_id=os.getenv("MATRIX_DEVICE_ID", "").strip(),
        store_path=os.getenv("MATRIX_STORE_PATH", "store").strip() or "store",
        e2e=_parse_bool(os.getenv("MATRIX_E2E"), True),
        video_whitelist=video_whitelist,
    )

    cfg = AppConfig(llm=llm, matrix=matrix, markdown=_parse_bool(os.getenv("MARKDOWN"), True), thinking=_parse_bool(os.getenv("THINKING"), False))
    ok, errs = validate_config(cfg)
    if not ok:
        raise ConfigError("Invalid configuration: " + "; ".join(errs))
    return cfg
