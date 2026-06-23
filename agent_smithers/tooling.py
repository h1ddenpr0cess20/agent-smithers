"""Tool configuration: hosted, local media, and MCP tool assembly.

Builds the per-provider tool definitions :class:`~.context.AppContext` offers
to the model — hosted tools (web/X search, code interpreter), the local Grok
image/video generation functions, and MCP server entries — and applies the
per-model gating, MCP reachability probing, and web-search country policy.
"""
from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import httpx

from .config import provider_for_model

if TYPE_CHECKING:
    from .context import AppContext


XAI_HOSTED_TOOL_TYPES = {"web_search", "x_search", "code_interpreter", "mcp"}
XAI_IMAGE_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "2:1", "1:2",
    "19.5:9", "9:19.5", "20:9", "9:20", "auto",
]
XAI_VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]
XAI_VIDEO_RESOLUTIONS = ["480p", "720p"]
GROK_GENERATE_IMAGE_TOOL = "grok_generate_image"
GROK_EDIT_IMAGE_TOOL = "grok_edit_image"
GROK_GENERATE_VIDEO_TOOL = "grok_generate_video"


_MCP_PROBE_TIMEOUT = 5.0


async def _probe_url(url: str) -> bool:
    """Check whether an MCP server URL is reachable.

    Issues a short-timeout GET and treats any response below HTTP 500 as
    reachable; connection errors and timeouts count as unreachable.

    Args:
        url: The server URL to probe.

    Returns:
        ``True`` if the server responded with a status below 500.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_MCP_PROBE_TIMEOUT)) as client:
            response = await client.get(url)
            return response.status_code < 500
    except Exception:
        return False


async def probe_mcp_servers(ctx: "AppContext") -> None:
    """Check configured MCP servers and remove unreachable ones from the tool list.

    Probes each server_url in parallel at startup. Servers that fail to respond
    are logged and dropped so they don't cause errors during generation.
    """
    mcp_tools_by_label: Dict[str, str] = {}
    for provider_tools in ctx.hosted_tools_by_provider.values():
        for tool in provider_tools:
            if tool.get("type") == "mcp":
                url = str(tool.get("server_url") or "").strip()
                label = str(tool.get("server_label") or "").strip()
                if url and label:
                    mcp_tools_by_label[label] = url

    if not mcp_tools_by_label:
        return

    results = await asyncio.gather(
        *(_probe_url(url) for url in mcp_tools_by_label.values()),
        return_exceptions=True,
    )
    offline: Set[str] = set()
    for label, result in zip(mcp_tools_by_label, results):
        if result is True:
            ctx.logger.info("MCP server '%s' is reachable", label)
        else:
            ctx.logger.warning("MCP server '%s' is unreachable — skipping", label)
            offline.add(label)

    if not offline:
        return

    for provider, provider_tools in ctx.hosted_tools_by_provider.items():
        ctx.hosted_tools_by_provider[provider] = [
            t for t in provider_tools
            if not (t.get("type") == "mcp" and str(t.get("server_label") or "").strip() in offline)
        ]
    ctx._mcp_auto_approve -= offline
    ctx.hosted_tools = tools_for_model(ctx, ctx.model)


def initialize_hosted_tools(ctx: "AppContext") -> Tuple[Dict[str, List[Dict[str, Any]]], set[str]]:
    """Build the per-provider tool map and the MCP auto-approve set.

    Pops the internal ``_auto_approve`` marker off each MCP tool, collecting
    the server labels that should be auto-approved.

    Args:
        ctx: Application context providing config and configured providers.

    Returns:
        A tuple of ``(tools_by_provider, auto_approve_labels)``.
    """
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
    """List providers usable given the current credentials/base URLs.

    OpenAI and xAI require an API key; LM Studio and Ollama require a base URL.

    Args:
        ctx: Application context holding the LLM config.

    Returns:
        The ordered list of provider keys that are actually configured.
    """
    providers: List[str] = []
    for provider in ("openai", "xai", "lmstudio", "ollama"):
        if provider in {"lmstudio", "ollama"}:
            if ctx.cfg.llm.base_urls.get(provider):
                providers.append(provider)
            continue
        if ctx.cfg.llm.api_keys.get(provider):
            providers.append(provider)
    return providers


def provider_for_context_model(ctx: "AppContext", model: str) -> str:
    """Resolve the provider for a model using the context's model map.

    Args:
        ctx: Application context holding the resolved model map.
        model: Model identifier to resolve.

    Returns:
        The owning provider key.

    Raises:
        ValueError: If the model cannot be resolved to a provider.
    """
    provider = provider_for_model(model, ctx.models)
    if not provider:
        raise ValueError(f"Unable to resolve provider for model '{model}'")
    return provider


def tools_for_model(ctx: "AppContext", model: str) -> List[Dict[str, Any]]:
    """Return the tool definitions enabled for a specific model.

    Filters the provider's tools by per-model support and strips the
    web-search ``user_location`` when the country toggle is disabled.

    Args:
        ctx: Application context.
        model: Model the tools will be offered to.

    Returns:
        The list of tool definitions for that model.
    """
    provider = provider_for_context_model(ctx, model)
    tools = list(ctx.hosted_tools_by_provider.get(provider, []))
    tools = [tool for tool in tools if tool_supported_for_model(provider, model, tool)]
    if not getattr(ctx, "search_country_enabled", True):
        tools = _strip_search_country(tools)
    return tools


def _strip_search_country(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the ``user_location`` hint from web-search tools.

    Applied when the search-country toggle is off so location bias is not sent
    to the provider. Non-search tools pass through unchanged.

    Args:
        tools: The tool definitions to sanitize.

    Returns:
        A new list where ``web_search`` tools have no ``user_location`` key.
    """
    result: List[Dict[str, Any]] = []
    for tool in tools:
        if str(tool.get("type") or "") == "web_search" and "user_location" in tool:
            tool = {k: v for k, v in tool.items() if k != "user_location"}
        result.append(tool)
    return result


def tool_supported_for_model(provider: str, model: str, tool: Dict[str, Any]) -> bool:
    """Report whether a tool is usable by a given model.

    Only xAI is gated: its hosted tools require a Grok-4 class model, while
    local function tools are always allowed. Other providers support all tools.

    Args:
        provider: Provider owning the model.
        model: Model identifier.
        tool: The tool definition under consideration.

    Returns:
        ``True`` if the model can use the tool.
    """
    if provider != "xai":
        return True
    tool_type = str(tool.get("type") or "")
    if tool_type in XAI_HOSTED_TOOL_TYPES:
        return xai_model_supports_hosted_tools(model)
    if tool_type == "function":
        return True
    return True


def xai_model_supports_hosted_tools(model: str) -> bool:
    """Report whether an xAI model supports hosted tools (Grok-4 family).

    Args:
        model: xAI model identifier.

    Returns:
        ``True`` for ``grok-4``-prefixed models, else ``False``.
    """
    lowered = str(model or "").strip().lower()
    return lowered.startswith("grok-4")


async def refresh_models(ctx: "AppContext") -> None:
    """Merge server-reported models into the context's model map.

    Queries each configured provider's model list and unions it with the
    configured models (configured ids are always retained). Providers that
    error or return nothing are left as configured. Rebuilds the active tool
    list afterward.

    Args:
        ctx: Application context whose model map and tools are updated.
    """
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
    """Assemble the full tool list for a provider from config.

    Combines configured hosted tools (with sensible defaults), the local Grok
    media tools, and any MCP server entries.

    Args:
        ctx: Application context holding the tool configuration.
        provider: Provider to build tools for.

    Returns:
        The assembled list of tool definitions.
    """
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
        defaults["video_generation"] = True
    elif provider == "openai":
        defaults["image_generation"] = True
        defaults["video_generation"] = True
    for tool_name, default_value in defaults.items():
        if provider in {"xai", "openai"} and tool_name == "video_generation":
            continue
        if provider == "xai" and tool_name == "image_generation":
            continue
        tool = build_hosted_tool(ctx, provider, tool_name, hosted_config.get(tool_name, default_value))
        if tool:
            tools.append(tool)
    if provider in {"openai", "xai"}:
        tools.extend(build_local_media_tools(ctx, hosted_config))
    if provider == "openai":
        pass
    for name, spec in (ctx.cfg.llm.mcp_servers or {}).items():
        tool = build_mcp_tool(ctx, provider, name, spec)
        if tool:
            tools.append(tool)
    return tools


def build_local_media_tools(ctx: "AppContext", hosted_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the local Grok image/video function-tool definitions.

    These are only emitted when an xAI API key is present and the
    corresponding ``image_generation``/``video_generation`` toggles are not
    disabled.

    Args:
        ctx: Application context holding API keys.
        hosted_config: The resolved ``tools`` config controlling toggles.

    Returns:
        The list of local media function-tool definitions (possibly empty).
    """
    tools: List[Dict[str, Any]] = []
    xai_available = bool(ctx.cfg.llm.api_keys.get("xai"))
    if xai_available and hosted_config.get("image_generation", True) not in (None, False):
        tools.extend(
            [
                {
                    "type": "function",
                    "name": GROK_GENERATE_IMAGE_TOOL,
                    "description": (
                        "Generate an image with xAI Grok Imagine. Prefer the model's native image tool "
                        "unless the user explicitly asks for Grok or xAI."
                    ),
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
                                "enum": XAI_IMAGE_ASPECT_RATIOS,
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
                },
                {
                    "type": "function",
                    "name": GROK_EDIT_IMAGE_TOOL,
                    "description": "Edit one or more source images with xAI Grok Imagine.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed description of the requested image edit.",
                            },
                            "image_url": {
                                "type": "string",
                                "description": "Public URL or data URI for a single source image.",
                            },
                            "image_urls": {
                                "type": "array",
                                "description": "Optional list of up to 3 public URLs or data URIs for source images.",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                            "aspect_ratio": {
                                "type": "string",
                                "description": "Image aspect ratio.",
                                "enum": XAI_IMAGE_ASPECT_RATIOS,
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Output resolution: '1k' (default) or '2k'.",
                                "enum": ["1k", "2k"],
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of edited images to return (1-10).",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            ]
        )
    if hosted_config.get("video_generation", True) not in (None, False):
        if xai_available:
            tools.append(
                {
                    "type": "function",
                    "name": GROK_GENERATE_VIDEO_TOOL,
                    "description": "Generate a new video, animate an image, or edit a video with xAI Grok Imagine.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed description of the video to create or the edit to apply.",
                            },
                            "image_url": {
                                "type": "string",
                                "description": "Optional public URL or data URI for image-to-video generation.",
                            },
                            "video_url": {
                                "type": "string",
                                "description": "Optional public URL for editing an existing video.",
                            },
                            "duration": {
                                "type": "integer",
                                "description": "Video duration in seconds for new generations (1-15).",
                                "minimum": 1,
                                "maximum": 15,
                            },
                            "aspect_ratio": {
                                "type": "string",
                                "description": "Video aspect ratio for new generations.",
                                "enum": XAI_VIDEO_ASPECT_RATIOS,
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Video output resolution.",
                                "enum": XAI_VIDEO_RESOLUTIONS,
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                }
            )
    return tools


def build_hosted_tool(
    ctx: "AppContext",
    provider: str,
    tool_name: str,
    spec: Any,
) -> Optional[Dict[str, Any]]:
    """Build a single hosted-tool definition from its config spec.

    Adds provider-specific extras (OpenAI web-search ``user_location`` and
    code-interpreter ``container``) and merges any dict overrides.

    Args:
        ctx: Application context (for logging and search-country config).
        provider: Provider the tool targets.
        tool_name: Hosted tool name (becomes the ``type``).
        spec: ``True`` to enable with defaults, a dict of overrides, or a
            falsy/invalid value to disable.

    Returns:
        The tool definition, or ``None`` when disabled or invalid.
    """
    if spec in (None, False):
        return None
    if tool_name == "image_generation" and provider == "xai":
        return None
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
    """Build an MCP server tool definition from its config spec.

    Maps config keys to the provider-specific field names (xAI vs. OpenAI),
    resolves a bearer token from ``authorization_env`` when present, and marks
    auto-approve servers with an internal ``_auto_approve`` flag.

    Args:
        ctx: Application context (for logging and env lookups).
        provider: Provider the MCP tool targets.
        name: Default server label when the spec omits one.
        spec: The MCP server configuration dict.

    Returns:
        The MCP tool definition, or ``None`` if the spec is invalid or lacks
        both a ``server_url`` and a ``connector_id``.
    """
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
    """Prepend or merge a web-search country-preference policy note.

    Applies only for xAI requests that include a search tool and only when a
    country is configured and the toggle is enabled; the note steers results
    toward the configured country.

    Args:
        ctx: Application context holding the country config and toggle.
        messages: The chat messages about to be sent.
        provider: Provider the request targets.
        tools: Tools attached to the request.

    Returns:
        A new message list with the policy merged into (or prepended as) a
        system message, or a copy of ``messages`` when the policy does not
        apply.
    """
    country = ctx.cfg.llm.web_search_country
    if not country or not getattr(ctx, "search_country_enabled", True) or provider != "xai":
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
