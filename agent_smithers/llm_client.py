from __future__ import annotations

import asyncio
import base64
import mimetypes
from io import BytesIO
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from .config import AppConfig, provider_for_model


class LLMClient:
    """Thin Responses API client for OpenAI-compatible providers."""

    LMSTUDIO_FALLBACK_USER_PROMPT = "Please continue the conversation."
    OPENAI_VIDEO_MODEL = "sora-2"
    OPENAI_SORA_SIZES = ("720x1280", "1280x720", "1024x1792", "1792x1024")
    XAI_IMAGE_MODEL = "grok-imagine-image"
    XAI_VIDEO_MODEL = "grok-imagine-video"
    VIDEO_POLL_INTERVAL_SECONDS = 5.0

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

    @staticmethod
    def _has_user_message(items: Iterable[Dict[str, Any]]) -> bool:
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "").strip() == "user" and str(item.get("content") or "").strip():
                return True
        return False

    @classmethod
    def _ensure_lmstudio_user_message(
        cls,
        items: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        final_items = [dict(item) for item in items if isinstance(item, dict)]
        if cls._has_user_message(final_items):
            return final_items
        final_items.append({"role": "user", "content": cls.LMSTUDIO_FALLBACK_USER_PROMPT})
        return final_items

    @staticmethod
    def _merge_include_items(existing: Any, additions: Iterable[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in existing if isinstance(existing, list) else []:
            if isinstance(value, str) and value and value not in seen:
                merged.append(value)
                seen.add(value)
        for value in additions:
            if value and value not in seen:
                merged.append(value)
                seen.add(value)
        return merged

    @staticmethod
    def _xai_image_ref(url: str) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "url": url,
        }

    @staticmethod
    def _xai_video_ref(url: str) -> Dict[str, Any]:
        return {"url": url}

    @staticmethod
    def _has_video_url(payload: Dict[str, Any]) -> bool:
        direct_url = payload.get("url")
        if isinstance(direct_url, str) and direct_url:
            return True
        for key in ("video", "result", "output"):
            value = payload.get(key)
            if isinstance(value, dict):
                nested_url = value.get("url")
                if isinstance(nested_url, str) and nested_url:
                    return True
        return False

    @staticmethod
    def _video_backend_label(provider: str) -> str:
        return "Sora" if provider == "openai" else "Grok"

    def _headers(self, provider: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = str(self.cfg.llm.api_keys.get(provider, "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _multipart_headers(self, provider: str) -> Dict[str, str]:
        headers = self._headers(provider)
        headers.pop("Content-Type", None)
        return headers

    @staticmethod
    def _decode_data_uri(reference: str) -> Tuple[bytes, str]:
        header, encoded = reference.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1] if ":" in header else "application/octet-stream"
        return base64.b64decode(encoded), mime_type or "application/octet-stream"

    async def _load_media_reference(
        self,
        reference: str,
        *,
        provider: Optional[str],
        default_name: str,
    ) -> Tuple[str, bytes, str]:
        if reference.startswith("data:"):
            payload, mime_type = self._decode_data_uri(reference)
            extension = mimetypes.guess_extension(mime_type) or ""
            filename = f"{default_name}{extension}" if extension else default_name
            return filename, payload, mime_type
        payload = await self.download_url(reference, provider=provider)
        path = urlparse(reference).path
        filename = path.rsplit("/", 1)[-1] or default_name
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, payload, mime_type

    @classmethod
    def _parse_size(cls, size: str) -> Tuple[int, int]:
        width_str, height_str = size.split("x", 1)
        return int(width_str), int(height_str)

    @classmethod
    def _choose_openai_sora_size(cls, width: int, height: int, requested_size: Optional[str]) -> str:
        if requested_size:
            return requested_size
        source_ratio = width / height if height else 1.0
        candidates = []
        for candidate in cls.OPENAI_SORA_SIZES:
            candidate_width, candidate_height = cls._parse_size(candidate)
            candidate_ratio = candidate_width / candidate_height if candidate_height else 1.0
            candidates.append((abs(candidate_ratio - source_ratio), candidate))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    async def _prepare_openai_video_reference(
        self,
        reference: str,
        *,
        size: Optional[str],
    ) -> Tuple[str, str, bytes, str]:
        filename, payload, _mime_type = await self._load_media_reference(
            reference,
            provider=None,
            default_name="input_reference",
        )
        with Image.open(BytesIO(payload)) as image:
            source = image.convert("RGBA") if image.mode in {"RGBA", "LA", "P"} else image.convert("RGB")
            selected_size = self._choose_openai_sora_size(source.width, source.height, size)
            target_width, target_height = self._parse_size(selected_size)
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            prepared = ImageOps.fit(source, (target_width, target_height), method=resampling)
            output = BytesIO()
            prepared.save(output, format="PNG")
        return selected_size, f"{filename.rsplit('.', 1)[0]}.png", output.getvalue(), "image/png"

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
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Responses API request body."""
        provider = self._provider_for_model(model)
        payload: Dict[str, Any] = {"model": model}
        derived_instructions = instructions
        derived_input: List[Dict[str, Any]] = []
        supports_instructions = self._supports_instructions(provider)
        if messages is not None:
            derived_instructions, derived_input = self.build_input_items(
                messages,
                include_system=not supports_instructions,
            )
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if supports_instructions and derived_instructions and not previous_response_id:
            payload["instructions"] = derived_instructions
        final_input = input_items if input_items is not None else derived_input
        if provider == "lmstudio" and messages is not None and input_items is None:
            final_input = self._ensure_lmstudio_user_message(final_input)
        if final_input:
            payload["input"] = final_input
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if options:
            for key, value in options.items():
                if value is not None:
                    payload[key] = value
        if provider == "xai" and any(
            isinstance(tool, dict) and tool.get("type") in {"web_search", "x_search"}
            for tool in (tools or [])
        ):
            payload["include"] = self._merge_include_items(
                payload.get("include"),
                ["no_inline_citations"],
            )
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
        instructions: Optional[str] = None,
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
            instructions=instructions,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.post(
                f"{self._base_url(provider)}/responses",
                headers=self._headers(provider),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        provider_override: Optional[str] = None,
        n: int = 1,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image via xAI POST /v1/images/generations."""
        provider = provider_override or self._provider_for_model(model)
        payload: Dict[str, Any] = {
            "model": self.XAI_IMAGE_MODEL,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.post(
                f"{self._base_url(provider)}/images/generations",
                headers=self._headers(provider),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def edit_image(
        self,
        *,
        prompt: str,
        image_urls: List[str],
        model: str,
        provider_override: Optional[str] = None,
        n: int = 1,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Edit one or more images via xAI POST /v1/images/edits."""
        provider = provider_override or self._provider_for_model(model)
        cleaned_urls = [url.strip() for url in image_urls if str(url).strip()]
        if not cleaned_urls:
            raise ValueError("At least one image URL is required")
        payload: Dict[str, Any] = {
            "model": self.XAI_IMAGE_MODEL,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if len(cleaned_urls) == 1:
            payload["image"] = self._xai_image_ref(cleaned_urls[0])
        else:
            payload["images"] = [self._xai_image_ref(url) for url in cleaned_urls]
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.post(
                f"{self._base_url(provider)}/images/edits",
                headers=self._headers(provider),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def _poll_video_generation(
        self,
        *,
        client: httpx.AsyncClient,
        provider: str,
        request_id: str,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + float(self.cfg.llm.timeout)
        while True:
            response = await client.get(
                f"{self._base_url(provider)}/videos/{request_id}",
                headers=self._headers(provider),
            )
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status") or "").strip().lower()
            if on_status:
                label = status or "pending"
                on_status(f"Generating video with {self._video_backend_label(provider)} [{label}]")
            if status in {"done", "completed", "succeeded", "success"} or self._has_video_url(payload):
                return payload
            if status in {"expired", "failed", "error", "cancelled"}:
                raise RuntimeError(f"Video generation ended with status '{status}'")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for video generation request '{request_id}'")
            await asyncio.sleep(self.VIDEO_POLL_INTERVAL_SECONDS)

    async def generate_video(
        self,
        *,
        prompt: str,
        model: str,
        backend: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seconds: Optional[int] = None,
        size: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Generate or edit a video via xAI or OpenAI Sora."""
        provider = self._provider_for_model(model)
        backend_name = str(backend or "").strip().lower()
        if backend_name in {"grok", "xai"}:
            provider = "xai"
        elif backend_name in {"sora", "openai"}:
            provider = "openai"
        if provider == "openai":
            if video_url:
                raise ValueError("OpenAI Sora remix is not implemented in this client")
            multipart_fields: List[Tuple[str, Tuple[Optional[str], Any, Optional[str]]]] = [
                ("model", (None, self.OPENAI_VIDEO_MODEL, None)),
                ("prompt", (None, prompt, None)),
            ]
            if seconds is not None:
                multipart_fields.append(("seconds", (None, str(seconds), None)))
            if image_url:
                selected_size, filename, payload, mime_type = await self._prepare_openai_video_reference(
                    image_url,
                    size=size,
                )
                size = selected_size
                multipart_fields.append(("input_reference", (filename, payload, mime_type)))
            if size:
                multipart_fields.append(("size", (None, size, None)))
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
                response = await client.post(
                    f"{self._base_url(provider)}/videos",
                    headers=self._multipart_headers(provider),
                    files=multipart_fields,
                )
                response.raise_for_status()
                created = response.json()
                status = str(created.get("status") or "").strip().lower()
                if status in {"done", "completed", "succeeded", "success"}:
                    return created
                request_id = str(created.get("id") or "").strip()
                if not request_id:
                    raise RuntimeError("OpenAI video response did not include an id")
                if on_status:
                    on_status(f"Generating video with {self._video_backend_label(provider)} [{status or 'queued'}]")
                return await self._poll_video_generation(
                    client=client,
                    provider=provider,
                    request_id=request_id,
                    on_status=on_status,
                )
        payload: Dict[str, Any] = {
            "model": self.XAI_VIDEO_MODEL,
            "prompt": prompt,
        }
        if video_url:
            payload["video_url"] = video_url
        else:
            if image_url:
                payload["image"] = self._xai_video_ref(image_url)
            if duration is not None:
                payload["duration"] = duration
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            if resolution:
                payload["resolution"] = resolution
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.post(
                f"{self._base_url(provider)}/videos/generations",
                headers=self._headers(provider),
                json=payload,
            )
            response.raise_for_status()
            created = response.json()
            status = str(created.get("status") or "").strip().lower()
            if status in {"done", "completed", "succeeded", "success"} or self._has_video_url(created):
                return created
            request_id = str(created.get("id") or created.get("request_id") or "").strip()
            if not request_id:
                raise RuntimeError("Video generation response did not include a request id")
            if on_status:
                on_status(f"Generating video with {self._video_backend_label(provider)} [{status or 'queued'}]")
            return await self._poll_video_generation(
                client=client,
                provider=provider,
                request_id=request_id,
                on_status=on_status,
            )

    async def download_video_content(self, video_id: str, *, provider: str) -> bytes:
        """Download a generated video from a provider-specific content endpoint."""
        if provider == "openai":
            url = f"{self._base_url(provider)}/videos/{video_id}/content"
        else:
            url = f"{self._base_url(provider)}/videos/{video_id}/content"
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.get(url, headers=self._headers(provider))
            response.raise_for_status()
            return response.content

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

    async def download_url(self, url: str, *, provider: Optional[str] = None) -> bytes:
        """Download a direct media URL, using provider auth only for first-party URLs."""
        headers: Optional[Dict[str, str]] = None
        if provider and url.startswith(self._base_url(provider)):
            headers = self._headers(provider)
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.llm.timeout)) as client:
            response = await client.get(url, headers=headers)
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
