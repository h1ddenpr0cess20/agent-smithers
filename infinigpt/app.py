from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import AppConfig, provider_for_model
from .handlers.cmd_ai import handle_ai
from .handlers.cmd_help import handle_help
from .handlers.cmd_model import handle_model
from .handlers.cmd_mymodel import handle_mymodel
from .handlers.cmd_prompt import handle_custom, handle_persona
from .handlers.cmd_reset import handle_clear, handle_reset
from .handlers.cmd_tools import handle_tools
from .handlers.cmd_x import handle_x
from .handlers.router import Router
from .history import HistoryStore
from .llm_client import LLMClient
from .matrix_client import MatrixClientWrapper
from .security import Security


class AppContext:
    """Application runtime context and service container."""

    def __init__(self, cfg: AppConfig, executor: Optional[ThreadPoolExecutor] = None) -> None:
        self.cfg = cfg
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="infinigpt")
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
        try:
            self.admins = list(getattr(cfg.matrix, "admins", []))
        except Exception:
            self.admins = []
        self.bot_id = "InfiniGPT"
        self.user_models: Dict[str, Dict[str, str]] = {}

        self.llm = LLMClient(cfg)
        self.hosted_tools_by_provider = {
            provider: self._build_tools(provider)
            for provider in self._configured_providers()
        }
        self.hosted_tools = self._tools_for_model(self.model)
        self.tools_enabled = any(bool(tools) for tools in self.hosted_tools_by_provider.values())
        self._mcp_auto_approve = {
            str((tool.get("server_label") or "")).strip()
            for provider_tools in self.hosted_tools_by_provider.values()
            for tool in provider_tools
            if tool.get("type") == "mcp" and bool(tool.pop("_auto_approve", False))
        }

        artifact_root = Path(cfg.matrix.store_path).expanduser().resolve().parent / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = artifact_root

        if self.hosted_tools:
            self.logger.info("Hosted tools enabled with %d tool definitions", len(self.hosted_tools))
        else:
            self.logger.info("Tool calling disabled: no hosted tools configured")

    def _configured_providers(self) -> List[str]:
        providers: List[str] = []
        for provider in ("openai", "xai", "lmstudio"):
            if provider == "lmstudio":
                if self.cfg.llm.base_urls.get(provider):
                    providers.append(provider)
                continue
            if self.cfg.llm.api_keys.get(provider):
                providers.append(provider)
        return providers

    def _provider_for_model(self, model: str) -> str:
        provider = provider_for_model(model, self.models)
        if not provider:
            raise ValueError(f"Unable to resolve provider for model '{model}'")
        return provider

    def _tools_for_model(self, model: str) -> List[Dict[str, Any]]:
        return list(self.hosted_tools_by_provider.get(self._provider_for_model(model), []))

    async def refresh_models(self) -> None:
        """Refresh the available model list from configured providers."""
        merged_models = dict(self.cfg.llm.models)
        for provider in self._configured_providers():
            try:
                fetched = await self.llm.list_models(provider)
            except Exception:
                self.logger.exception("Failed to refresh model list from %s; keeping configured models", provider)
                continue
            if not fetched:
                self.logger.warning("%s model list was empty; keeping configured models", provider)
                continue
            configured = list(self.cfg.llm.models.get(provider, []))
            merged_models[provider] = sorted(dict.fromkeys([*fetched, *configured]))
            self.logger.info("Loaded %d %s models from server", len(fetched), provider)
        self.models = merged_models
        self.cfg.llm.models = merged_models
        self.hosted_tools = self._tools_for_model(self.model)

    def _build_tools(self, provider: str) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        hosted_config = dict(getattr(self.cfg.llm, "tools", {}) or {})
        defaults = {
            "web_search": True,
            "code_interpreter": True,
        }
        if provider == "lmstudio":
            defaults = {}
        elif provider == "xai":
            defaults["x_search"] = True
        elif provider == "openai":
            defaults["image_generation"] = True
        for tool_name, default_value in defaults.items():
            tool = self._build_hosted_tool(provider, tool_name, hosted_config.get(tool_name, default_value))
            if tool:
                tools.append(tool)
        for name, spec in (self.cfg.llm.mcp_servers or {}).items():
            tool = self._build_mcp_tool(provider, name, spec)
            if tool:
                tools.append(tool)
        return tools

    def _build_hosted_tool(self, provider: str, tool_name: str, spec: Any) -> Optional[Dict[str, Any]]:
        if spec in (None, False):
            return None
        tool: Dict[str, Any] = {"type": tool_name}
        if isinstance(spec, dict):
            tool.update(spec)
        elif spec is not True:
            self.logger.warning("Ignoring invalid tool config for %s", tool_name)
            return None
        if (
            tool_name == "code_interpreter"
            and provider == "openai"
            and "container" not in tool
        ):
            tool["container"] = {"type": "auto"}
        return tool

    def _build_mcp_tool(self, provider: str, name: str, spec: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(spec, dict):
            self.logger.warning("Ignoring invalid MCP config for %s", name)
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
            self.logger.warning("Skipping MCP server '%s' without server_url or connector_id", name)
            return None
        if spec.get("auto_approve") is True:
            tool["_auto_approve"] = True
        return tool

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
            return None

    def clean_response_text(self, text: str, *, sender_display: str, sender_id: str) -> str:
        cleaned = text or ""
        if "</think>" in cleaned and "<think>" in cleaned:
            try:
                thinking, rest = cleaned.split("</think>", 1)
                thinking = thinking.replace("<think>", "").strip()
                self.log(f"Model thinking for {sender_display} ({sender_id}): {thinking}")
                cleaned = rest.strip()
            except Exception:
                pass
        if "<|begin_of_thought|>" in cleaned and "<|end_of_thought|>" in cleaned:
            try:
                parts = cleaned.split("<|end_of_thought|>")
                if len(parts) > 1:
                    thinking = (
                        parts[0]
                        .replace("<|begin_of_thought|>", "")
                        .replace("<|end_of_thought|>", "")
                        .strip()
                    )
                    self.log(f"Model thinking for {sender_display} ({sender_id}): {thinking}")
                    cleaned = parts[1].strip()
            except Exception:
                pass
        if "<|begin_of_solution|>" in cleaned and "<|end_of_solution|>" in cleaned:
            try:
                cleaned = cleaned.split("<|begin_of_solution|>", 1)[1].split(
                    "<|end_of_solution|>", 1
                )[0].strip()
            except Exception:
                pass
        return cleaned.strip()

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        parts: List[str] = []
        output_text = str(response.get("output_text") or "").strip()
        if output_text:
            return output_text
        for item in response.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text":
                    text = str(content.get("text") or "").strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _walk_image_results(value: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(value, str):
            yield {"inline": value}
            return
        if isinstance(value, list):
            for entry in value:
                yield from AppContext._walk_image_results(entry)
            return
        if not isinstance(value, dict):
            return

        inline = value.get("b64_json") or value.get("result") or value.get("data")
        if isinstance(inline, str) and inline:
            yield {"inline": inline}

        file_id = value.get("file_id")
        if isinstance(file_id, str) and file_id:
            yield {
                "file_id": file_id,
                "container_id": value.get("container_id"),
            }

        nested_file = value.get("file")
        if isinstance(nested_file, dict):
            nested_file_id = nested_file.get("id") or nested_file.get("file_id")
            if isinstance(nested_file_id, str) and nested_file_id:
                yield {
                    "file_id": nested_file_id,
                    "container_id": nested_file.get("container_id") or value.get("container_id"),
                }

    @staticmethod
    def _iter_image_sources(response: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        for item in response.get("output", []) or []:
            item_type = item.get("type")
            if item_type == "image_generation_call":
                container_id = item.get("container_id")
                result = item.get("result")
                for source in AppContext._walk_image_results(result):
                    if container_id and source.get("file_id") and not source.get("container_id"):
                        source = dict(source)
                        source["container_id"] = container_id
                    yield source
                direct_file_id = item.get("file_id")
                if isinstance(direct_file_id, str) and direct_file_id:
                    yield {"file_id": direct_file_id, "container_id": container_id}
                continue

            if item_type != "message":
                continue
            for content in item.get("content", []) or []:
                content_type = content.get("type")
                if content_type == "output_image":
                    file_id = content.get("file_id")
                    if isinstance(file_id, str) and file_id:
                        yield {
                            "file_id": file_id,
                            "container_id": content.get("container_id"),
                        }
                    image_url = content.get("image_url")
                    if isinstance(image_url, str) and image_url.startswith("data:"):
                        yield {"inline": image_url}
                elif content_type == "output_text":
                    annotations = content.get("annotations") or []
                    for annotation in annotations:
                        if not isinstance(annotation, dict):
                            continue
                        file_id = annotation.get("file_id")
                        if isinstance(file_id, str) and file_id:
                            yield {
                                "file_id": file_id,
                                "container_id": annotation.get("container_id"),
                            }

    @staticmethod
    def _decode_base64_image(data: str) -> bytes:
        encoded = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
        return base64.b64decode(encoded)

    def _write_artifact(self, data: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="infinigpt-",
            suffix=suffix,
            dir=self.artifact_dir,
            delete=False,
        ) as handle:
            handle.write(data)
            return handle.name

    async def _download_image_bytes(
        self,
        *,
        provider: str,
        file_id: str,
        container_id: Optional[str],
    ) -> bytes:
        return await self.llm.download_file(file_id, provider=provider, container_id=container_id)

    async def _send_response_artifacts(
        self,
        response: Dict[str, Any],
        room_id: Optional[str],
        *,
        provider: str,
    ) -> bool:
        if not room_id:
            return False
        sent_any = False
        for source in self._iter_image_sources(response):
            try:
                if source.get("inline"):
                    image_bytes = self._decode_base64_image(str(source["inline"]))
                else:
                    file_id = str(source.get("file_id") or "")
                    if not file_id:
                        continue
                    image_bytes = await self._download_image_bytes(
                        provider=provider,
                        file_id=file_id,
                        container_id=source.get("container_id"),
                    )
                path = self._write_artifact(image_bytes, ".png")
                await self.matrix.send_image(room_id=room_id, path=path, filename=None, log=self.log)
                sent_any = True
            except Exception:
                self.logger.exception("Failed to send generated image")
        return sent_any

    def _approval_items(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in response.get("output", []) or []:
            item_type = str(item.get("type") or "")
            if item_type in {"mcp_approval_request", "mcp_approval_request_item"}:
                items.append(item)
        return items

    def _should_auto_approve(self, item: Dict[str, Any]) -> bool:
        label = str(item.get("server_label") or item.get("mcp_server_label") or "").strip()
        return bool(label and label in self._mcp_auto_approve)

    async def _maybe_continue_after_approvals(
        self,
        *,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        current = response
        while True:
            approval_items = self._approval_items(current)
            auto_items = [item for item in approval_items if self._should_auto_approve(item)]
            if not auto_items:
                return current
            approval_input = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in auto_items
                if item.get("id")
            ]
            if not approval_input:
                return current
            current = await self.llm.create_response(
                model=model,
                previous_response_id=current.get("id"),
                input_items=approval_input,
                tools=tools if self.tools_enabled else None,
                tool_choice="auto" if self.tools_enabled and tools else None,
                options=self.options,
            )

    async def generate_reply(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        use_tools: Optional[bool] = None,
    ) -> str:
        use_model = model or self.model
        provider = self._provider_for_model(use_model)
        active_tools = self._tools_for_model(use_model)
        self.hosted_tools = active_tools
        tools_enabled = self.tools_enabled if use_tools is None else bool(use_tools and active_tools)
        response = await self.llm.create_response(
            model=use_model,
            messages=messages,
            tools=active_tools if tools_enabled else None,
            tool_choice="auto" if tools_enabled and active_tools else None,
            options=self.options,
        )
        response = await self._maybe_continue_after_approvals(
            model=use_model,
            tools=active_tools if tools_enabled else None,
            response=response,
        )
        sent_artifacts = await self._send_response_artifacts(response, room_id, provider=provider)
        text = self._extract_text(response)
        if text:
            return text
        if sent_artifacts:
            return "Generated output attached."
        return ""

    async def respond_with_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        room_id: Optional[str] = None,
        tool_choice: str = "auto",
    ) -> str:
        del tool_choice
        return await self.generate_reply(messages, model=model, room_id=room_id, use_tools=True)


async def run(cfg: AppConfig, config_path: Optional[str] = None) -> None:
    """Run the Matrix bot runtime loop."""
    ctx = AppContext(cfg)
    if cfg.llm.server_models:
        await ctx.refresh_models()

    router = Router()
    router.register(".ai", handle_ai)
    router.register(".x", handle_x)
    router.register(".persona", handle_persona)
    router.register(".custom", handle_custom)
    router.register(".reset", handle_reset)
    router.register(".stock", lambda c, r, s, d, a: handle_reset(c, r, s, d, "stock"))
    router.register(".help", handle_help)
    router.register(".mymodel", handle_mymodel)
    router.register(".tools", handle_tools, admin=True)
    try:
        from .handlers.cmd_verbose import handle_verbose

        router.register(".verbose", handle_verbose, admin=True)
    except Exception:
        pass
    router.register(".model", handle_model, admin=True)
    router.register(".clear", handle_clear, admin=True)

    ctx.log(f"Model set to {ctx.model}")

    await ctx.matrix.load_store()
    login_resp = await ctx.matrix.login()
    try:
        ctx.log(login_resp)
    except Exception:
        pass
    await ctx.matrix.ensure_keys()
    await ctx.matrix.initial_sync()

    try:
        ctx.bot_id = await ctx.matrix.display_name(cfg.matrix.username)
    except Exception:
        ctx.bot_id = cfg.matrix.username

    try:
        device_id = getattr(ctx.matrix.client, "device_id", None)
        if device_id and hasattr(cfg.matrix, "device_id") and not cfg.matrix.device_id and config_path:
            with open(config_path, "r+") as handle:
                data = json.load(handle)
                data.setdefault("matrix", {})["device_id"] = device_id
                handle.seek(0)
                json.dump(data, handle, indent=4)
                handle.truncate()
            ctx.log(f"Persisted device_id to {config_path}")
    except Exception:
        pass

    for room in cfg.matrix.channels:
        try:
            await ctx.matrix.join(room)
            ctx.log(f"{ctx.bot_id} joined {room}")
        except Exception:
            ctx.log(f"Couldn't join {room}")

    import datetime as _dt

    security = Security(ctx.matrix, logger=ctx.logger)
    try:
        from nio import KeyVerificationEvent  # type: ignore
    except Exception:
        KeyVerificationEvent = None  # type: ignore
    try:
        if KeyVerificationEvent:
            ctx.matrix.add_to_device_callback(security.emoji_verification_callback, (KeyVerificationEvent,))
        ctx.matrix.add_to_device_callback(security.log_to_device_event, None)
    except Exception:
        pass

    join_time = _dt.datetime.now()

    async def on_text(room, event) -> None:
        try:
            message_time = getattr(event, "server_timestamp", 0) / 1000.0
            message_time = _dt.datetime.fromtimestamp(message_time)
            if message_time <= join_time:
                return
            text = getattr(event, "body", "")
            sender = getattr(event, "sender", "")
            if sender == cfg.matrix.username:
                return
            sender_display = await ctx.matrix.display_name(sender)
            is_admin = sender_display in ctx.admins or sender in ctx.admins
            handler, args = router.dispatch(
                ctx,
                room.room_id,
                sender,
                sender_display,
                text,
                is_admin,
                bot_name=ctx.bot_id,
                timestamp=message_time,
            )
            if handler is None:
                return
            try:
                ctx.log(f"{sender_display} ({sender}) sent {text} in {room.room_id}")
            except Exception:
                pass
            try:
                await security.allow_devices(sender)
            except Exception:
                pass
            result = handler(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            ctx.log(exc)

    ctx.matrix.add_text_handler(on_text)

    import signal as _signal

    stop = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except Exception:
                pass
    except Exception:
        pass
    sync_task = asyncio.create_task(ctx.matrix.sync_forever())
    stop_task = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({sync_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        pass
    finally:
        for task in (sync_task, stop_task):
            if not task.done():
                task.cancel()
        try:
            if hasattr(ctx.matrix, "shutdown"):
                await ctx.matrix.shutdown()
        except Exception:
            pass
        try:
            ctx.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
