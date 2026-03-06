from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from .config import AppConfig, provider_for_model


class LLMClient:
    """Thin Responses API client for OpenAI-compatible providers."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
 
    @staticmethod
    def _fallback_base_url(provider: str) -> str:
        if provider == "lmstudio":
            return "http://127.0.0.1:1234/v1"
        if provider == "xai":
            return "https://api.x.ai/v1"
        return "https://api.openai.com/v1"

    def _base_url(self, provider: str) -> str:
        configured = str(self.cfg.llm.base_urls.get(provider, "") or "").strip()
        return configured or self._fallback_base_url(provider)

    def _provider_for_model(self, model: str) -> str:
        provider = provider_for_model(model, self.cfg.llm.models)
        if not provider:
            raise ValueError(f"Unable to resolve provider for model '{model}'")
        return provider

    @staticmethod
    def _supports_instructions(provider: str) -> bool:
        return provider != "xai"

    def _headers(self, provider: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = str(self.cfg.llm.api_keys.get(provider, "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _is_chat_model(provider: str, model_id: str) -> bool:
        lowered = model_id.lower()
        if provider == "lmstudio":
            blocked_models = {
                "text-embedding-nomic-embed-text-v1.5",
            }
            if lowered in blocked_models:
                return False
            return bool(model_id.strip())
        if provider == "xai":
            if not lowered.startswith("grok-"):
                return False
            blocked_fragments = ("imagine", "image", "video", "voice", "vision")
            return not any(fragment in lowered for fragment in blocked_fragments)

        prefixes = ("gpt-", "o1", "o3", "o4")
        if not model_id.startswith(prefixes):
            return False

        blocked_fragments = (
            "preview",
            "audio",
            "computer-use",
            "transcribe",
            "tts",
            "image",
        )
        if any(fragment in lowered for fragment in blocked_fragments):
            return False

        if re.search(r"-\d{4}-\d{2}-\d{2}$", lowered):
            return False

        return True

    @staticmethod
    def build_input_items(
        messages: Iterable[Dict[str, Any]],
        *,
        include_system: bool = False,
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Convert chat-style history to Responses API instructions/input."""
        instructions: List[str] = []
        input_items: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if not role or not content:
                continue
            if role == "system":
                if include_system:
                    input_items.append({"role": role, "content": content})
                else:
                    instructions.append(content)
                continue
            if role in {"user", "assistant"}:
                input_items.append({"role": role, "content": content})
        joined = "\n\n".join(part.strip() for part in instructions if part.strip()).strip()
        return (joined or None, input_items)

    def build_request_payload(
        self,
        *,
        model: str,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        input_items: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a Responses API request body."""
        provider = self._provider_for_model(model)
        payload: Dict[str, Any] = {"model": model}
        instructions = None
        derived_input: List[Dict[str, Any]] = []
        supports_instructions = self._supports_instructions(provider)
        if messages is not None:
            instructions, derived_input = self.build_input_items(
                messages,
                include_system=not supports_instructions,
            )
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if supports_instructions and instructions and not previous_response_id:
            payload["instructions"] = instructions
        final_input = input_items if input_items is not None else derived_input
        if final_input:
            payload["input"] = final_input
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if options:
            for key, value in options.items():
                if value is not None:
                    payload[key] = value
        return payload

    async def create_response(
        self,
        *,
        model: str,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        input_items: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a response via the configured provider's Responses API."""
        provider = self._provider_for_model(model)
        payload = self.build_request_payload(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id,
            input_items=input_items,
            options=options,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.post(
                f"{self._base_url(provider)}/responses",
                headers=self._headers(provider),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def download_file(
        self,
        file_id: str,
        *,
        provider: str,
        container_id: Optional[str] = None,
    ) -> bytes:
        """Download a file or container file returned by a hosted tool."""
        if container_id:
            url = f"{self._base_url(provider)}/containers/{container_id}/files/{file_id}/content"
        else:
            url = f"{self._base_url(provider)}/files/{file_id}/content"
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.get(url, headers=self._headers(provider))
            response.raise_for_status()
            return response.content

    async def list_models(self, provider: str) -> List[str]:
        """List response-capable models from the requested provider."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.get(
                f"{self._base_url(provider)}/models",
                headers=self._headers(provider),
            )
            response.raise_for_status()
            payload = response.json()
        model_ids = [
            str(item.get("id") or "").strip()
            for item in payload.get("data", [])
            if isinstance(item, dict)
        ]
        filtered = sorted({model_id for model_id in model_ids if self._is_chat_model(provider, model_id)})
        return filtered or sorted({model_id for model_id in model_ids if model_id})
