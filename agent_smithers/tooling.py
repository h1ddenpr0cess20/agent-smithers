from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .config import provider_for_model

if TYPE_CHECKING:
    from .context import AppContext


XAI_HOSTED_TOOL_TYPES = {"web_search", "x_search", "code_interpreter", "mcp"}


def initialize_hosted_tools(ctx: "AppContext") -> Tuple[Dict[str, List[Dict[str, Any]]], set[str]]:
    hosted_tools_by_provider = {
        provider: build_tools(ctx, provider)
        for provider in configured_providers(ctx)
    }
    auto_approve = {
        str((tool.get("server_label") or "")).strip()
        for provider_tools in hosted_tools_by_provider.values()
        for tool in provider_tools
        if tool.get("type") == "mcp" and bool(tool.pop("_auto_approve", False))
    }
    return hosted_tools_by_provider, auto_approve


def configured_providers(ctx: "AppContext") -> List[str]:
    providers: List[str] = []
    for provider in ("openai", "xai", "lmstudio"):
        if provider == "lmstudio":
            if ctx.cfg.llm.base_urls.get(provider):
                providers.append(provider)
            continue
        if ctx.cfg.llm.api_keys.get(provider):
            providers.append(provider)
    return providers


def provider_for_context_model(ctx: "AppContext", model: str) -> str:
    provider = provider_for_model(model, ctx.models)
    if not provider:
        raise ValueError(f"Unable to resolve provider for model '{model}'")
    return provider


def tools_for_model(ctx: "AppContext", model: str) -> List[Dict[str, Any]]:
    provider = provider_for_context_model(ctx, model)
    tools = list(ctx.hosted_tools_by_provider.get(provider, []))
    return [tool for tool in tools if tool_supported_for_model(provider, model, tool)]


def tool_supported_for_model(provider: str, model: str, tool: Dict[str, Any]) -> bool:
    if provider != "xai":
        return True
    tool_type = str(tool.get("type") or "")
    if tool_type in XAI_HOSTED_TOOL_TYPES:
        return xai_model_supports_hosted_tools(model)
    if tool_type == "function":
        return xai_model_supports_function_tools(model)
    return True


def xai_model_supports_hosted_tools(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return lowered.startswith("grok-4")


def xai_model_supports_function_tools(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return lowered.startswith(("grok-4", "grok-3", "grok-code-fast-1"))


async def refresh_models(ctx: "AppContext") -> None:
    """Refresh the available model list from configured providers."""
    merged_models = dict(ctx.cfg.llm.models)
    for provider in configured_providers(ctx):
        try:
            fetched = await ctx.llm.list_models(provider)
        except Exception:
            ctx.logger.exception("Failed to refresh model list from %s; keeping configured models", provider)
            continue
        if not fetched:
            ctx.logger.warning("%s model list was empty; keeping configured models", provider)
            continue
        configured = list(ctx.cfg.llm.models.get(provider, []))
        merged_models[provider] = sorted(dict.fromkeys([*fetched, *configured]))
        ctx.logger.info("Loaded %d %s models from server", len(fetched), provider)
    ctx.models = merged_models
    ctx.cfg.llm.models = merged_models
    ctx.hosted_tools = tools_for_model(ctx, ctx.model)


def build_tools(ctx: "AppContext", provider: str) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    hosted_config = dict(getattr(ctx.cfg.llm, "tools", {}) or {})
    defaults = {
        "web_search": True,
        "code_interpreter": True,
    }
    if provider == "lmstudio":
        defaults = {}
    elif provider == "xai":
        defaults["x_search"] = True
        defaults["image_generation"] = True
    elif provider == "openai":
        defaults["image_generation"] = True
    for tool_name, default_value in defaults.items():
        tool = build_hosted_tool(ctx, provider, tool_name, hosted_config.get(tool_name, default_value))
        if tool:
            tools.append(tool)
    for name, spec in (ctx.cfg.llm.mcp_servers or {}).items():
        tool = build_mcp_tool(ctx, provider, name, spec)
        if tool:
            tools.append(tool)
    return tools


def build_hosted_tool(
    ctx: "AppContext",
    provider: str,
    tool_name: str,
    spec: Any,
) -> Optional[Dict[str, Any]]:
    if spec in (None, False):
        return None
    if tool_name == "image_generation" and provider == "xai":
        if spec is False:
            return None
        return {
            "type": "function",
            "name": "generate_image",
            "description": "Generate an image from a text description using the Grok Imagine API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image to generate.",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Image aspect ratio.",
                        "enum": [
                            "1:1", "16:9", "9:16", "4:3", "3:4",
                            "3:2", "2:3", "2:1", "1:2",
                            "19.5:9", "9:19.5", "20:9", "9:20", "auto",
                        ],
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Output resolution: '1k' (default) or '2k'.",
                        "enum": ["1k", "2k"],
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of images to generate (1-10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        }
    tool: Dict[str, Any] = {"type": tool_name}
    if isinstance(spec, dict):
        tool.update(spec)
    elif spec is not True:
        ctx.logger.warning("Ignoring invalid tool config for %s", tool_name)
        return None
    if (
        tool_name == "web_search"
        and provider == "openai"
        and ctx.cfg.llm.web_search_country
        and "user_location" not in tool
    ):
        tool["user_location"] = {
            "type": "approximate",
            "country": ctx.cfg.llm.web_search_country,
        }
    if (
        tool_name == "code_interpreter"
        and provider == "openai"
        and "container" not in tool
    ):
        tool["container"] = {"type": "auto"}
    return tool


def build_mcp_tool(
    ctx: "AppContext",
    provider: str,
    name: str,
    spec: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict):
        ctx.logger.warning("Ignoring invalid MCP config for %s", name)
        return None
    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": str(spec.get("server_label") or name),
    }
    if provider == "xai":
        field_map = {
            "server_url": "server_url",
            "server_description": "server_description",
            "allowed_tools": "allowed_tool_names",
            "authorization": "authorization",
            "headers": "extra_headers",
        }
    else:
        field_map = {
            "server_url": "server_url",
            "connector_id": "connector_id",
            "server_description": "server_description",
            "allowed_tools": "allowed_tools",
            "require_approval": "require_approval",
            "authorization": "authorization",
            "headers": "headers",
        }
    for source_key, target_key in field_map.items():
        if source_key in spec and spec[source_key] is not None:
            tool[target_key] = spec[source_key]
    auth_env = spec.get("authorization_env")
    if auth_env and "authorization" not in tool:
        token = os.getenv(str(auth_env), "").strip()
        if token:
            tool["authorization"] = f"Bearer {token}"
    if "server_url" not in tool and "connector_id" not in tool:
        ctx.logger.warning("Skipping MCP server '%s' without server_url or connector_id", name)
        return None
    if spec.get("auto_approve") is True:
        tool["_auto_approve"] = True
    return tool


def apply_search_country_policy(
    ctx: "AppContext",
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    country = ctx.cfg.llm.web_search_country
    if not country or provider != "xai":
        return list(messages)
    uses_search = any(
        isinstance(tool, dict) and str(tool.get("type") or "") in {"web_search", "x_search"}
        for tool in tools
    )
    if not uses_search:
        return list(messages)
    policy = (
        "When using web_search or x_search, prioritize results and sources from "
        f"{country}. If the provider tool cannot enforce country filtering directly, "
        "state that limitation and keep the answer focused on US sources where possible."
    )
    if messages and str(messages[0].get("role") or "") == "system":
        merged = dict(messages[0])
        merged["content"] = f"{messages[0].get('content', '').rstrip()}\n\n{policy}".strip()
        return [merged, *messages[1:]]
    return [{"role": "system", "content": policy}, *messages]
