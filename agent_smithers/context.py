"""Application runtime context and service container.

:class:`AppContext` owns the shared services (Matrix client, LLM client,
history store, executor) and per-conversation state (active model, persona,
tool configuration, generated-media cache, thinking-indicator bookkeeping).
Most heavy lifting is delegated to the :mod:`responses` and :mod:`tooling`
modules; the methods here are the stable entry points those modules call back
through.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .history import HistoryStore
from .llm_client import LLMClient
from .logging_conf import spinner_status
from .markdown_utils import render_markdown
from .matrix_client import MatrixClientWrapper
from . import responses, tooling


class AppContext:
    """Application runtime context and service container."""

    INLINE_CITATION_RE = responses.INLINE_CITATION_RE
    _extract_text = staticmethod(responses.extract_text)
    _strip_inline_citations = staticmethod(responses.strip_inline_citations)
    _walk_image_results = staticmethod(responses.walk_image_results)
    _iter_image_sources = staticmethod(responses.iter_image_sources)
    _decode_base64_image = staticmethod(responses.decode_base64_image)

    def __init__(self, cfg: AppConfig, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Construct the context and all shared services from config.

        Builds the Matrix client, history store, and LLM client, derives the
        active model/persona/tool configuration, and prepares the artifact
        output directory.

        Args:
            cfg: Fully resolved application configuration.
            executor: Optional thread pool for offloading blocking work; a
                4-worker pool is created when not supplied.
        """
        self.cfg = cfg
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-smithers")
        self.logger = logging.getLogger(__name__)
        self.log = self.logger.info

        self.matrix = MatrixClientWrapper(
            server=cfg.matrix.server,
            username=cfg.matrix.username,
            password=cfg.matrix.password,
            device_id=cfg.matrix.device_id,
            store_path=cfg.matrix.store_path,
            encryption_enabled=bool(getattr(cfg.matrix, "e2e", True)),
        )

        prompt = list(cfg.llm.prompt or ["you are ", "."])
        prefix = prompt[0] if len(prompt) >= 1 else "you are "
        suffix = prompt[1] if len(prompt) >= 2 else "."
        extra = prompt[2] if len(prompt) >= 3 else ""
        self.history = HistoryStore(
            prompt_prefix=prefix,
            prompt_suffix=suffix,
            personality=cfg.llm.personality,
            prompt_suffix_extra=extra,
            max_tokens=cfg.llm.history_tokens,
            store_path=cfg.matrix.store_path if cfg.llm.history_encryption_key else None,
            encryption_key=cfg.llm.history_encryption_key or None,
        )

        self.models = cfg.llm.models
        self.default_model = cfg.llm.default_model
        self.model = cfg.llm.default_model
        self.default_personality = cfg.llm.personality
        self.personality = cfg.llm.personality
        self.options = cfg.llm.options
        self.timeout = cfg.llm.timeout
        self.admins = list(getattr(cfg.matrix, "admins", []))
        self.bot_id = "Agent Smithers"
        self.user_models: Dict[str, Dict[str, str]] = {}
        self.verbose = False
        self.generated_media: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        self.video_whitelist: set[str] = set(getattr(cfg.matrix, "video_whitelist", []))
        self.video_whitelist_enabled: bool = bool(self.video_whitelist)
        self.search_country_enabled: bool = bool(cfg.llm.web_search_country)
        self.thinking: bool = bool(getattr(cfg, "thinking", False))

        self.llm = LLMClient(cfg)
        self.hosted_tools_by_provider, self._mcp_auto_approve = tooling.initialize_hosted_tools(self)
        self.hosted_tools = self._tools_for_model(self.model)
        self.tools_enabled = any(bool(tools) for tools in self.hosted_tools_by_provider.values())

        artifact_root = Path(cfg.matrix.store_path).expanduser().resolve().parent / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = artifact_root

        if self.hosted_tools:
            self.logger.info("Hosted tools enabled with %d tool definitions", len(self.hosted_tools))
        else:
            self.logger.info("Tool calling disabled: no hosted tools configured")

    def is_video_allowed(self, sender_id: str, sender_display: str) -> bool:
        """Return True if the user is allowed to generate video."""
        if not self.video_whitelist_enabled:
            return True
        if sender_display in self.admins or sender_id in self.admins:
            return True
        return sender_id in self.video_whitelist or sender_display in self.video_whitelist

    def _configured_providers(self) -> List[str]:
        """Return the providers that are actually usable given the config.

        Returns:
            Provider keys that have credentials or a base URL configured.
        """
        return tooling.configured_providers(self)

    def _provider_for_model(self, model: str) -> str:
        """Resolve the provider that serves ``model``.

        Args:
            model: Model identifier to resolve.

        Returns:
            The owning provider key.
        """
        return tooling.provider_for_context_model(self, model)

    def _tools_for_model(self, model: str) -> List[Dict[str, Any]]:
        """Return the tool definitions enabled for a given model.

        Args:
            model: Model the tools will be offered to.

        Returns:
            The list of hosted/local/MCP tool definitions for that model.
        """
        return tooling.tools_for_model(self, model)

    async def refresh_models(self) -> None:
        """Refresh each provider's model list from its ``/models`` endpoint."""
        await tooling.refresh_models(self)

    async def probe_mcp_servers(self) -> None:
        """Probe configured MCP servers to validate connectivity and tools."""
        await tooling.probe_mcp_servers(self)

    def _build_tools(self, provider: str) -> List[Dict[str, Any]]:
        """Build the full tool list for a provider.

        Args:
            provider: Provider key to build tools for.

        Returns:
            The assembled tool definitions for that provider.
        """
        return tooling.build_tools(self, provider)

    def _build_hosted_tool(self, provider: str, tool_name: str, spec: Any) -> Optional[Dict[str, Any]]:
        """Build a single hosted-tool definition from its spec.

        Args:
            provider: Provider the tool targets.
            tool_name: Hosted tool name (e.g. ``"web_search"``).
            spec: The tool's configuration value (bool or dict).

        Returns:
            The tool definition, or ``None`` if the spec disables it.
        """
        return tooling.build_hosted_tool(self, provider, tool_name, spec)

    def _build_mcp_tool(self, provider: str, name: str, spec: Any) -> Optional[Dict[str, Any]]:
        """Build a single MCP server tool definition from its spec.

        Args:
            provider: Provider the MCP tool targets.
            name: Server label for the MCP entry.
            spec: The MCP server configuration dict.

        Returns:
            The MCP tool definition, or ``None`` if the spec is invalid.
        """
        return tooling.build_mcp_tool(self, provider, name, spec)

    def _apply_search_country_policy(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: str,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Inject the configured web-search country hint into the messages.

        Args:
            messages: The chat messages about to be sent.
            provider: Provider the request targets (policy only applies to xAI).
            tools: The tools attached to the request (policy applies only when
                a search tool is present).

        Returns:
            The messages, possibly with a system note prepended or merged.
        """
        return tooling.apply_search_country_policy(self, messages, provider=provider, tools=tools)

    async def to_thread(self, fn, *args, **kwargs) -> Any:
        """Run a blocking callable on the context's thread pool.

        Args:
            fn: The callable to execute off the event loop.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            The return value of ``fn``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))

    def render(self, body: str) -> Optional[str]:
        """Render Markdown body text to HTML when Markdown output is enabled.

        Args:
            body: The Markdown source to render.

        Returns:
            The rendered HTML, or ``None`` when Markdown is disabled or
            rendering fails.
        """
        if not self.cfg.markdown:
            return None
        rendered = render_markdown(body)
        if rendered is None:
            self.logger.exception("Markdown rendering failed")
        return rendered

    def status(self, message: str, *, spinner: str = "dots"):
        """Create a console status spinner context manager.

        Args:
            message: Initial status text to display.
            spinner: Rich spinner animation name.

        Returns:
            A status context manager (no-op unless INFO logging is enabled).
        """
        return spinner_status(
            message,
            spinner=spinner,
            enabled=self.logger.isEnabledFor(logging.INFO),
        )

    def clean_response_text(self, text: str, *, sender_display: str, sender_id: str) -> str:
        """Strip thinking/solution scaffolding from model output.

        Args:
            text: Raw model response text.
            sender_display: Display name of the requesting user.
            sender_id: Matrix ID of the requesting user.

        Returns:
            The cleaned, user-facing response text.
        """
        return responses.clean_response_text(
            self,
            text,
            sender_display=sender_display,
            sender_id=sender_id,
        )

    def _write_artifact(self, data: bytes, suffix: str) -> str:
        """Persist binary artifact data to the artifact directory.

        Args:
            data: Raw bytes to write.
            suffix: File extension/suffix for the artifact.

        Returns:
            The filesystem path of the written artifact.
        """
        return responses.write_artifact(self, data, suffix)

    async def _download_image_bytes(
        self,
        *,
        provider: str,
        file_id: str,
        container_id: Optional[str],
    ) -> bytes:
        """Download an image referenced by a hosted-tool file id.

        Args:
            provider: Provider that produced the file.
            file_id: Identifier of the file to download.
            container_id: Optional container id when the file lives in a
                code-interpreter container.

        Returns:
            The raw image bytes.
        """
        return await responses.download_image_bytes(
            self,
            provider=provider,
            file_id=file_id,
            container_id=container_id,
        )

    async def _send_response_artifacts(
        self,
        response: Dict[str, Any],
        room_id: Optional[str],
        *,
        provider: str,
        thread_user: Optional[str] = None,
    ) -> bool:
        """Extract and send any images embedded in a model response.

        Args:
            response: The decoded Responses API payload.
            room_id: Destination Matrix room, or ``None`` to skip sending.
            provider: Provider that produced the response (for auth/downloads).
            thread_user: Optional user id used to remember generated media.

        Returns:
            ``True`` if at least one image was sent.
        """
        return await responses.send_response_artifacts(
            self,
            response,
            room_id,
            provider=provider,
            thread_user=thread_user,
        )

    def _approval_items(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pending MCP approval-request items from a response.

        Args:
            response: The decoded Responses API payload.

        Returns:
            The list of approval-request output items (possibly empty).
        """
        return responses.approval_items(response)

    def _should_auto_approve(self, item: Dict[str, Any]) -> bool:
        """Decide whether an MCP approval request should be auto-approved.

        Args:
            item: A single MCP approval-request item.

        Returns:
            ``True`` if the item's server label is on the auto-approve set.
        """
        return responses.should_auto_approve(self, item)

    async def _maybe_continue_after_approvals(
        self,
        *,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Continue a response after auto-approving MCP requests, if any.

        Args:
            model: Model to continue the turn with.
            tools: Tools to keep attached to the follow-up request.
            response: The response carrying the approval requests.

        Returns:
            The continued response, or the original when nothing was approved.
        """
        return await responses.maybe_continue_after_approvals(
            self,
            model=model,
            tools=tools,
            response=response,
        )

    async def _handle_generate_image_calls(
        self,
        response: Dict[str, Any],
        *,
        model: str,
        room_id: Optional[str],
        thread_user: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute local image/video generation tool calls in a response.

        Args:
            response: The decoded Responses API payload to inspect.
            model: Model that issued the tool calls.
            room_id: Destination room for generated media.
            thread_user: Optional user id for remembering generated media.

        Returns:
            The ``function_call_output`` items to feed back, or ``None`` when
            there were no media tool calls to handle.
        """
        return await responses.handle_generate_image_calls(
            self,
            response,
            model=model,
            room_id=room_id,
            thread_user=thread_user,
        )

    def _remember_generated_media(
        self,
        room_id: Optional[str],
        user_id: Optional[str],
        *,
        kind: str,
        reference: str,
        mime_type: str,
    ) -> None:
        """Cache the most recent generated media for a room/user thread.

        Args:
            room_id: Room the media was generated in.
            user_id: User the media belongs to within the room thread.
            kind: Media kind, ``"image"`` or ``"video"``.
            reference: URL or data URI pointing at the media.
            mime_type: MIME type of the media.
        """
        if not room_id or not user_id or not reference:
            return
        room_media = self.generated_media.setdefault(room_id, {})
        thread_media = room_media.setdefault(user_id, {})
        thread_media[kind] = {
            "reference": reference,
            "mime_type": mime_type,
        }

    def _latest_generated_media(
        self,
        room_id: Optional[str],
        user_id: Optional[str],
        *,
        kind: str,
    ) -> Optional[str]:
        """Return the cached reference for the latest media of a kind.

        Args:
            room_id: Room to look up.
            user_id: User thread within the room.
            kind: Media kind, ``"image"`` or ``"video"``.

        Returns:
            The cached media reference, or ``None`` if none is remembered.
        """
        if not room_id or not user_id:
            return None
        return (
            self.generated_media
            .get(room_id, {})
            .get(user_id, {})
            .get(kind, {})
            .get("reference")
        )

    def _thread_media_prompt_note(
        self,
        room_id: Optional[str],
        user_id: Optional[str],
        *,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Build a system note telling the model about cached thread media.

        Lets the model call edit/animate tools without the user re-supplying a
        media URL, since the runtime injects the cached reference.

        Args:
            room_id: Room to look up cached media for.
            user_id: User thread within the room.
            provider: Active provider; video notes are omitted for OpenAI.

        Returns:
            A note string, or ``None`` when no relevant media is cached.
        """
        image_ref = self._latest_generated_media(room_id, user_id, kind="image")
        video_ref = self._latest_generated_media(room_id, user_id, kind="video")
        notes: List[str] = []
        if image_ref:
            notes.append(
                "If the user asks to edit, vary, or animate the most recently generated image in this thread, "
                "call the relevant tool without requiring an explicit image URL; the runtime will supply it."
            )
        if video_ref and provider != "openai":
            notes.append(
                "If the user asks to edit the most recently generated video in this thread, "
                "call the video tool without requiring an explicit video URL; the runtime will supply it."
            )
        if not notes:
            return None
        return "Recent generated media is available in this thread. " + " ".join(notes)

    def _clear_generated_media(self, room_id: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """Evict cached generated media at varying granularity.

        Args:
            room_id: Room to clear; when ``None``, clears the entire cache.
            user_id: User thread to clear within the room; when ``None``,
                clears the whole room.
        """
        if room_id is None:
            self.generated_media.clear()
            return
        if room_id not in self.generated_media:
            return
        if user_id is None:
            self.generated_media.pop(room_id, None)
            return
        room_media = self.generated_media.get(room_id, {})
        room_media.pop(user_id, None)
        if not room_media:
            self.generated_media.pop(room_id, None)

    async def generate_reply(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        use_tools: Optional[bool] = None,
        thread_user: Optional[str] = None,
    ) -> str:
        """Generate a chat reply, optionally running tools and sending media.

        Args:
            messages: The chat history to respond to.
            model: Model override; defaults to the active model.
            room_id: Room context for tool output and media delivery.
            use_tools: Force-enable or disable tools; ``None`` uses the
                model's default tool configuration.
            thread_user: Optional user id for thread media context.

        Returns:
            The final user-facing reply text.
        """
        return await responses.generate_reply(
            self,
            messages,
            model=model,
            room_id=room_id,
            use_tools=use_tools,
            thread_user=thread_user,
        )

    async def respond_with_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        tool_choice: str = "auto",
        thread_user: Optional[str] = None,
    ) -> str:
        """Run a full tool-calling turn and return the final reply.

        Drives the request/response loop including hosted tools, local media
        tools, and MCP approval handling.

        Args:
            messages: The chat history to respond to.
            model: Model override; defaults to the active model.
            room_id: Room context for tool output and media delivery.
            tool_choice: Tool-choice policy passed to the provider.
            thread_user: Optional user id for thread media context.

        Returns:
            The final user-facing reply text.
        """
        return await responses.respond_with_tools(
            self,
            messages,
            model=model,
            room_id=room_id,
            tool_choice=tool_choice,
            thread_user=thread_user,
        )

    async def _cancel_thinking_animation(self) -> None:
        """Cancel and await the running thinking-animation task, if any."""
        task = getattr(self, "thinking_animation_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.thinking_animation_task = None

    async def clear_thinking_indicator(self) -> None:
        """Cancel the thinking animation and redact the placeholder (used on error)."""
        await self._cancel_thinking_animation()
        event_id = getattr(self, "thinking_placeholder_event_id", None)
        room_id = getattr(self, "thinking_placeholder_room_id", None)
        self.thinking_placeholder_event_id = None
        self.thinking_placeholder_room_id = None
        if event_id and room_id:
            await self.matrix.redact_event(room_id, event_id)

    async def send_response(self, room_id: str, body: str, html: Optional[str] = None) -> None:
        """Send a reply, editing the thinking placeholder in-place if one exists."""
        await self._cancel_thinking_animation()
        event_id = getattr(self, "thinking_placeholder_event_id", None)
        self.thinking_placeholder_event_id = None
        self.thinking_placeholder_room_id = None
        if event_id:
            await self.matrix.edit_message(room_id, event_id, body, html=html)
        else:
            await self.matrix.send_text(room_id, body, html=html)
