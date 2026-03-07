from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import ConfigError


SUPPORTED_PROVIDERS = {"openai", "xai", "lmstudio"}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_json(value: str | None, default: Any) -> Any:
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
    server: str
    username: str
    password: str
    channels: List[str]
    admin: str = ""
    admins: List[str] = field(default_factory=list)
    device_id: str = ""
    store_path: str = "store"
    e2e: bool = True


@dataclass
class LLMConfig:
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
    history_size: int = 24
    mcp_servers: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 180


@dataclass
class AppConfig:
    llm: LLMConfig
    matrix: MatrixConfig
    markdown: bool = True


def provider_for_model(model: str, models: Dict[str, List[str]]) -> Optional[str]:
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
    errors: List[str] = []

    configured_providers = []
    for provider in sorted(SUPPORTED_PROVIDERS):
        if cfg.llm.models.get(provider) or cfg.llm.api_keys.get(provider) or cfg.llm.base_urls.get(provider):
            configured_providers.append(provider)
        if provider in {"openai", "xai"} and cfg.llm.models.get(provider) and not cfg.llm.api_keys.get(provider):
            errors.append(f"{provider.upper()}_API_KEY is required when {provider.upper()}_MODELS is set")
        if provider == "lmstudio" and cfg.llm.models.get(provider) and not cfg.llm.base_urls.get(provider):
            errors.append("LMSTUDIO_BASE_URL is required when LMSTUDIO_MODELS is set")
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


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load application config from a .env file and environment variables."""
    load_env_file(path)

    openai_models = _parse_csv(os.getenv("OPENAI_MODELS"))
    xai_models = _parse_csv(os.getenv("XAI_MODELS"))
    lmstudio_models = _parse_csv(os.getenv("LMSTUDIO_MODELS"))
    default_model = os.getenv(
        "DEFAULT_MODEL",
        openai_models[0]
        if openai_models
        else (xai_models[0] if xai_models else (lmstudio_models[0] if lmstudio_models else "")),
    )
    prompt_prefix = os.getenv("BOT_PROMPT_PREFIX", "assume the personality of ")
    prompt_suffix = os.getenv(
        "BOT_PROMPT_SUFFIX",
        ". roleplay and never break character, never say you are not actually this character or refuse to play the character. keep your responses relatively short.",
    )
    prompt_extra = os.getenv("BOT_PROMPT_SUFFIX_EXTRA", "")
    prompt = [prompt_prefix, prompt_suffix]
    if prompt_extra:
        prompt.append(prompt_extra)

    tools: Dict[str, Any] = {
        "web_search": _parse_bool(os.getenv("TOOLS_WEB_SEARCH"), True),
        "code_interpreter": _parse_bool(os.getenv("TOOLS_CODE_INTERPRETER"), True),
        "x_search": _parse_bool(os.getenv("TOOLS_X_SEARCH"), True),
        "image_generation": _parse_bool(os.getenv("TOOLS_IMAGE_GENERATION"), True),
    }

    llm = LLMConfig(
        models={
            "openai": openai_models,
            "xai": xai_models,
            "lmstudio": lmstudio_models,
        },
        api_keys={
            "openai": os.getenv("OPENAI_API_KEY", "").strip(),
            "xai": os.getenv("XAI_API_KEY", "").strip(),
            "lmstudio": os.getenv("LMSTUDIO_API_KEY", "").strip(),
        },
        base_urls={
            "lmstudio": os.getenv("LMSTUDIO_BASE_URL", "").strip(),
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
        history_size=int(os.getenv("HISTORY_SIZE", "24")),
        mcp_servers=_parse_json(os.getenv("MCP_SERVERS"), {}),
        timeout=int(
            os.getenv("LLM_TIMEOUT")
            or os.getenv("OPENAI_TIMEOUT")
            or os.getenv("XAI_TIMEOUT")
            or "180"
        ),
    )

    admins = _parse_csv(os.getenv("MATRIX_ADMINS"))
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
    )

    cfg = AppConfig(llm=llm, matrix=matrix, markdown=_parse_bool(os.getenv("MARKDOWN"), True))
    ok, errs = validate_config(cfg)
    if not ok:
        raise ConfigError("Invalid configuration: " + "; ".join(errs))
    return cfg
