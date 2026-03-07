from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .history import HistoryStore
from .llm_client import LLMClient
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
            max_items=cfg.llm.history_size,
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

    def _configured_providers(self) -> List[str]:
        return tooling.configured_providers(self)

    def _provider_for_model(self, model: str) -> str:
        return tooling.provider_for_context_model(self, model)

    def _tools_for_model(self, model: str) -> List[Dict[str, Any]]:
        return tooling.tools_for_model(self, model)

    async def refresh_models(self) -> None:
        await tooling.refresh_models(self)

    def _build_tools(self, provider: str) -> List[Dict[str, Any]]:
        return tooling.build_tools(self, provider)

    def _build_hosted_tool(self, provider: str, tool_name: str, spec: Any) -> Optional[Dict[str, Any]]:
        return tooling.build_hosted_tool(self, provider, tool_name, spec)

    def _build_mcp_tool(self, provider: str, name: str, spec: Any) -> Optional[Dict[str, Any]]:
        return tooling.build_mcp_tool(self, provider, name, spec)

    def _apply_search_country_policy(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: str,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return tooling.apply_search_country_policy(self, messages, provider=provider, tools=tools)

    async def to_thread(self, fn, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))

    def render(self, body: str) -> Optional[str]:
        if not self.cfg.markdown:
            return None
        try:
            import markdown as _md

            return _md.markdown(
                body,
                extensions=["extra", "fenced_code", "nl2br", "sane_lists", "tables", "codehilite"],
            )
        except Exception:
            self.logger.exception("Markdown rendering failed")
            return None

    def clean_response_text(self, text: str, *, sender_display: str, sender_id: str) -> str:
        return responses.clean_response_text(
            self,
            text,
            sender_display=sender_display,
            sender_id=sender_id,
        )

    def _write_artifact(self, data: bytes, suffix: str) -> str:
        return responses.write_artifact(self, data, suffix)

    async def _download_image_bytes(
        self,
        *,
        provider: str,
        file_id: str,
        container_id: Optional[str],
    ) -> bytes:
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
    ) -> bool:
        return await responses.send_response_artifacts(self, response, room_id, provider=provider)

    def _approval_items(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        return responses.approval_items(response)

    def _should_auto_approve(self, item: Dict[str, Any]) -> bool:
        return responses.should_auto_approve(self, item)

    async def _maybe_continue_after_approvals(
        self,
        *,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
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
    ) -> Optional[List[Dict[str, Any]]]:
        return await responses.handle_generate_image_calls(
            self,
            response,
            model=model,
            room_id=room_id,
        )

    async def generate_reply(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        use_tools: Optional[bool] = None,
    ) -> str:
        return await responses.generate_reply(
            self,
            messages,
            model=model,
            room_id=room_id,
            use_tools=use_tools,
        )

    async def respond_with_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        tool_choice: str = "auto",
    ) -> str:
        return await responses.respond_with_tools(
            self,
            messages,
            model=model,
            room_id=room_id,
            tool_choice=tool_choice,
        )
