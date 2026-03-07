from __future__ import annotations

import base64
import json
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .context import AppContext

from .llm_client import LLMClient
from .tooling import (
    GROK_EDIT_IMAGE_TOOL,
    GROK_GENERATE_IMAGE_TOOL,
    GROK_GENERATE_VIDEO_TOOL,
    SORA_GENERATE_VIDEO_TOOL,
)


INLINE_CITATION_RE = None
IMAGE_GENERATION_TOOL_NAMES = {GROK_GENERATE_IMAGE_TOOL}
IMAGE_EDIT_TOOL_NAMES = {GROK_EDIT_IMAGE_TOOL}
VIDEO_GENERATION_TOOL_NAMES = {GROK_GENERATE_VIDEO_TOOL, SORA_GENERATE_VIDEO_TOOL}
LOCAL_MEDIA_TOOL_NAMES = IMAGE_GENERATION_TOOL_NAMES | IMAGE_EDIT_TOOL_NAMES | VIDEO_GENERATION_TOOL_NAMES


def clean_response_text(
    ctx: "AppContext",
    text: str,
    *,
    sender_display: str,
    sender_id: str,
) -> str:
    cleaned = text or ""
    if "</think>" in cleaned and "<think>" in cleaned:
        try:
            thinking, rest = cleaned.split("</think>", 1)
            thinking = thinking.replace("<think>", "").strip()
            ctx.log(f"Model thinking for {sender_display} ({sender_id}): {thinking}")
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
                ctx.log(f"Model thinking for {sender_display} ({sender_id}): {thinking}")
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


def extract_text(response: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                text = str(content.get("text") or "")
                if text:
                    parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    return str(response.get("output_text") or "")


def _video_backend_label(provider: str) -> str:
    return "Sora" if provider == "openai" else "Grok"


def strip_inline_citations(text: str, annotations: Any = None) -> str:
    del annotations
    return str(text or "")


def walk_image_results(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, str):
        yield {"inline": value}
        return
    if isinstance(value, list):
        for entry in value:
            yield from walk_image_results(entry)
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


def iter_image_sources(response: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in response.get("output", []) or []:
        item_type = item.get("type")
        if item_type == "image_generation_call":
            container_id = item.get("container_id")
            result = item.get("result")
            for source in walk_image_results(result):
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


def decode_base64_image(data: str) -> bytes:
    encoded = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    return base64.b64decode(encoded)


def write_artifact(ctx: "AppContext", data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="agent-smithers-",
        suffix=suffix,
        dir=ctx.artifact_dir,
        delete=False,
    ) as handle:
        handle.write(data)
        return handle.name


async def download_image_bytes(
    ctx: "AppContext",
    *,
    provider: str,
    file_id: str,
    container_id: Optional[str],
) -> bytes:
    return await ctx.llm.download_file(file_id, provider=provider, container_id=container_id)


async def send_response_artifacts(
    ctx: "AppContext",
    response: Dict[str, Any],
    room_id: Optional[str],
    *,
    provider: str,
    thread_user: Optional[str] = None,
) -> bool:
    if not room_id:
        return False
    sent_any = False
    for source in iter_image_sources(response):
        try:
            if source.get("inline"):
                inline_ref = str(source["inline"])
                image_bytes = decode_base64_image(inline_ref)
                ctx._remember_generated_media(
                    room_id,
                    thread_user,
                    kind="image",
                    reference=inline_ref if inline_ref.startswith("data:") else f"data:image/png;base64,{inline_ref}",
                    mime_type="image/png",
                )
            else:
                file_id = str(source.get("file_id") or "")
                if not file_id:
                    continue
                image_bytes = await ctx._download_image_bytes(
                    provider=provider,
                    file_id=file_id,
                    container_id=source.get("container_id"),
                )
                ctx._remember_generated_media(
                    room_id,
                    thread_user,
                    kind="image",
                    reference=f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
                    mime_type="image/png",
                )
            path = ctx._write_artifact(image_bytes, ".png")
            await ctx.matrix.send_image(room_id=room_id, path=path, filename=None, log=ctx.log)
            sent_any = True
        except Exception:
            ctx.logger.exception("Failed to send generated image")
    return sent_any


def approval_items(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in response.get("output", []) or []:
        item_type = str(item.get("type") or "")
        if item_type in {"mcp_approval_request", "mcp_approval_request_item"}:
            items.append(item)
    return items


def should_auto_approve(ctx: "AppContext", item: Dict[str, Any]) -> bool:
    label = str(item.get("server_label") or item.get("mcp_server_label") or "").strip()
    return bool(label and label in ctx._mcp_auto_approve)


async def maybe_continue_after_approvals(
    ctx: "AppContext",
    *,
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    current = response
    while True:
        pending = approval_items(current)
        auto_items = [item for item in pending if should_auto_approve(ctx, item)]
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
        current = await ctx.llm.create_response(
            model=model,
            previous_response_id=current.get("id"),
            input_items=approval_input,
            tools=tools if ctx.tools_enabled else None,
            tool_choice="auto" if ctx.tools_enabled and tools else None,
            options=ctx.options,
        )


async def settle_response(
    ctx: "AppContext",
    response: Dict[str, Any],
    *,
    model: str,
    room_id: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    tools_enabled: bool,
    thread_user: Optional[str] = None,
    followup_instructions: Optional[str] = None,
    followup_input_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Continue approvals and local function calls until the response settles."""
    current = response
    accumulated_input = list(followup_input_items) if followup_input_items is not None else None
    while True:
        if accumulated_input is not None:
            accumulated_input.extend(
                item for item in (current.get("output") or [])
                if isinstance(item, dict)
            )
            pending = approval_items(current)
            auto_items = [item for item in pending if should_auto_approve(ctx, item)]
            if auto_items:
                approval_input = [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": item["id"],
                        "approve": True,
                    }
                    for item in auto_items
                    if item.get("id")
                ]
                if approval_input:
                    accumulated_input.extend(approval_input)
                    current = await ctx.llm.create_response(
                        model=model,
                        input_items=list(accumulated_input),
                        tools=tools if tools_enabled else None,
                        tool_choice="auto" if tools_enabled and tools else None,
                        options=ctx.options,
                        instructions=followup_instructions,
                    )
                    continue
        else:
            current = await maybe_continue_after_approvals(
                ctx,
                model=model,
                tools=tools if tools_enabled else None,
                response=current,
            )
        image_outputs = await handle_generate_image_calls(
            ctx,
            current,
            model=model,
            room_id=room_id,
            thread_user=thread_user,
        )
        if not image_outputs:
            return current
        with ctx.status(f"Generating follow-up reply with {model}"):
            if accumulated_input is not None:
                accumulated_input.extend(image_outputs)
                current = await ctx.llm.create_response(
                    model=model,
                    input_items=list(accumulated_input),
                    tools=tools if tools_enabled else None,
                    tool_choice="auto" if tools_enabled and tools else None,
                    options=ctx.options,
                    instructions=followup_instructions,
                )
            else:
                current = await ctx.llm.create_response(
                    model=model,
                    previous_response_id=current.get("id"),
                    input_items=image_outputs,
                    tools=tools if tools_enabled else None,
                    tool_choice="auto" if tools_enabled and tools else None,
                    options=ctx.options,
                )


async def handle_generate_image_calls(
    ctx: "AppContext",
    response: Dict[str, Any],
    *,
    model: str,
    room_id: Optional[str],
    thread_user: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Execute local xAI media function calls and return tool output items."""
    calls = [
        item for item in (response.get("output") or [])
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") in LOCAL_MEDIA_TOOL_NAMES
    ]
    if not calls:
        return None
    output_items: List[Dict[str, Any]] = []
    for call in calls:
        call_id = str(call.get("call_id") or call.get("id") or "")
        try:
            args = json.loads(str(call.get("arguments") or "{}"))
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Empty prompt")
            name = str(call.get("name") or "")
            if name in IMAGE_GENERATION_TOOL_NAMES:
                result = await _execute_generate_image_call(
                    ctx,
                    model=model,
                    room_id=room_id,
                    thread_user=thread_user,
                    prompt=prompt,
                    args=args,
                )
            elif name in IMAGE_EDIT_TOOL_NAMES:
                result = await _execute_edit_image_call(
                    ctx,
                    model=model,
                    room_id=room_id,
                    thread_user=thread_user,
                    prompt=prompt,
                    args=args,
                )
            else:
                result = await _execute_generate_video_call(
                    ctx,
                    model=model,
                    room_id=room_id,
                    thread_user=thread_user,
                    prompt=prompt,
                    args=args,
                    provider="openai" if name == SORA_GENERATE_VIDEO_TOOL else "xai",
                )
        except Exception:
            ctx.logger.exception("%s tool call failed", call.get("name"))
            if call.get("name") in VIDEO_GENERATION_TOOL_NAMES:
                result = "Video generation failed."
            elif call.get("name") in IMAGE_EDIT_TOOL_NAMES:
                result = "Image editing failed."
            else:
                result = "Image generation failed."
        if call_id:
            output_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            })
    return output_items or None


async def _send_base64_images(
    ctx: "AppContext",
    response: Dict[str, Any],
    *,
    room_id: Optional[str],
    thread_user: Optional[str],
) -> bool:
    sent_any = False
    for entry in response.get("data", []) or []:
        b64 = entry.get("b64_json")
        if not b64:
            continue
        image_bytes = base64.b64decode(b64)
        ctx._remember_generated_media(
            room_id,
            thread_user,
            kind="image",
            reference=f"data:image/png;base64,{b64}",
            mime_type="image/png",
        )
        path = ctx._write_artifact(image_bytes, ".png")
        if room_id:
            await ctx.matrix.send_image(room_id=room_id, path=path, filename=None, log=ctx.log)
            sent_any = True
    return sent_any


def _extract_image_edit_urls(args: Dict[str, Any]) -> List[str]:
    image_urls = [
        str(value).strip()
        for value in (args.get("image_urls") or [])
        if str(value).strip()
    ]
    image_url = str(args.get("image_url") or "").strip()
    if image_url:
        image_urls.insert(0, image_url)
    unique_urls: List[str] = []
    seen = set()
    for value in image_urls:
        if value not in seen:
            unique_urls.append(value)
            seen.add(value)
    return unique_urls


def _guess_media_suffix(url: str, default: str) -> str:
    suffix = urlparse(url).path.rsplit("/", 1)[-1]
    if "." not in suffix:
        return default
    ext = "." + suffix.rsplit(".", 1)[-1].lower()
    if not ext or len(ext) > 8:
        return default
    return ext


def _extract_video_url(payload: Dict[str, Any]) -> Optional[str]:
    direct_url = payload.get("url")
    if isinstance(direct_url, str) and direct_url:
        return direct_url
    for key in ("video", "result", "output"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested_url = value.get("url")
            if isinstance(nested_url, str) and nested_url:
                return nested_url
    return None


async def _execute_generate_image_call(
    ctx: "AppContext",
    *,
    model: str,
    room_id: Optional[str],
    thread_user: Optional[str],
    prompt: str,
    args: Dict[str, Any],
) -> str:
    with ctx.status("Generating image with Grok"):
        img_response = await ctx.llm.generate_image(
            prompt=prompt,
            model=model,
            provider_override="xai",
            n=int(args["n"]) if args.get("n") else 1,
            aspect_ratio=str(args["aspect_ratio"]) if args.get("aspect_ratio") else None,
            resolution=str(args["resolution"]) if args.get("resolution") else None,
        )
    sent_any = await _send_base64_images(ctx, img_response, room_id=room_id, thread_user=thread_user)
    return "Image generated and sent." if sent_any else "Image generated."


async def _execute_edit_image_call(
    ctx: "AppContext",
    *,
    model: str,
    room_id: Optional[str],
    thread_user: Optional[str],
    prompt: str,
    args: Dict[str, Any],
) -> str:
    image_urls = _extract_image_edit_urls(args)
    if not image_urls:
        latest_image = ctx._latest_generated_media(room_id, thread_user, kind="image")
        if latest_image:
            image_urls = [latest_image]
        else:
            raise ValueError("No image_url or image_urls provided")
    with ctx.status("Editing image with Grok"):
        img_response = await ctx.llm.edit_image(
            prompt=prompt,
            image_urls=image_urls,
            model=model,
            provider_override="xai",
            n=int(args["n"]) if args.get("n") else 1,
            aspect_ratio=str(args["aspect_ratio"]) if args.get("aspect_ratio") else None,
            resolution=str(args["resolution"]) if args.get("resolution") else None,
        )
    sent_any = await _send_base64_images(ctx, img_response, room_id=room_id, thread_user=thread_user)
    return "Image edited and sent." if sent_any else "Image edited."


async def _execute_generate_video_call(
    ctx: "AppContext",
    *,
    model: str,
    room_id: Optional[str],
    thread_user: Optional[str],
    prompt: str,
    args: Dict[str, Any],
    provider: str,
) -> str:
    backend = "sora" if provider == "openai" else "grok"
    image_url = str(args.get("image_url") or "").strip() or None
    video_url = str(args.get("video_url") or "").strip() or None
    if not image_url and not video_url:
        image_url = ctx._latest_generated_media(room_id, thread_user, kind="image")
        if not image_url and provider != "openai":
            video_url = ctx._latest_generated_media(room_id, thread_user, kind="video")
    if image_url and video_url:
        raise ValueError("Only one of image_url or video_url may be provided")
    action = "Animating image" if image_url else "Editing video" if video_url else "Generating video"
    backend_label = _video_backend_label(provider)
    with ctx.status(f"{action} with {backend_label}") as status:
        video_response = await ctx.llm.generate_video(
            prompt=prompt,
            model=model,
            backend=backend,
            image_url=image_url,
            video_url=video_url,
            duration=int(args["duration"]) if args.get("duration") is not None else None,
            aspect_ratio=str(args["aspect_ratio"]) if args.get("aspect_ratio") else None,
            resolution=str(args["resolution"]) if args.get("resolution") else None,
            seconds=int(args["seconds"]) if args.get("seconds") is not None else None,
            size=str(args["size"]) if args.get("size") else None,
            on_status=status.update,
        )
        suffix = ".mp4"
        if provider == "openai":
            video_id = str(video_response.get("id") or "").strip()
            if not video_id:
                raise ValueError("Video response did not include a downloadable id")
            ctx._remember_generated_media(
                room_id,
                thread_user,
                kind="video",
                reference=video_id,
                mime_type="video/mp4",
            )
            status.update(f"Downloading video from {backend_label}")
            video_bytes = await ctx.llm.download_video_content(video_id, provider=provider)
        else:
            final_url = _extract_video_url(video_response)
            if not final_url:
                raise ValueError("Video response did not include a downloadable URL")
            suffix = _guess_media_suffix(final_url, ".mp4")
            ctx._remember_generated_media(
                room_id,
                thread_user,
                kind="video",
                reference=final_url,
                mime_type="video/mp4",
            )
            status.update(f"Downloading video from {backend_label}")
            video_bytes = await ctx.llm.download_url(final_url, provider=provider)
    path = ctx._write_artifact(video_bytes, suffix)
    if room_id:
        await ctx.matrix.send_video(room_id=room_id, path=path, filename=None, log=ctx.log)
        return "Video generated and sent."
    return "Video generated."


async def generate_reply(
    ctx: "AppContext",
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    room_id: Optional[str] = None,
    use_tools: Optional[bool] = None,
    thread_user: Optional[str] = None,
) -> str:
    use_model = model or ctx.model
    provider = ctx._provider_for_model(use_model)
    active_tools = ctx._tools_for_model(use_model)
    prepared_messages = list(messages)
    media_note = ctx._thread_media_prompt_note(room_id, thread_user, provider=provider)
    if media_note:
        if prepared_messages and str(prepared_messages[0].get("role") or "") == "system":
            prepared_messages[0] = dict(prepared_messages[0])
            prepared_messages[0]["content"] = f'{prepared_messages[0]["content"]}\n\n{media_note}'
        else:
            prepared_messages.insert(0, {"role": "system", "content": media_note})
    prepared_messages = ctx._apply_search_country_policy(
        prepared_messages,
        provider=provider,
        tools=active_tools,
    )
    followup_instructions = None
    followup_input_items = None
    if provider == "openai":
        followup_instructions, followup_input_items = LLMClient.build_input_items(prepared_messages)
    ctx.hosted_tools = active_tools
    tools_enabled = ctx.tools_enabled if use_tools is None else bool(use_tools and active_tools)
    with ctx.status(f"Generating reply with {use_model}"):
        response = await ctx.llm.create_response(
            model=use_model,
            messages=prepared_messages,
            tools=active_tools if tools_enabled else None,
            tool_choice="auto" if tools_enabled and active_tools else None,
            options=ctx.options,
        )
    response = await settle_response(
        ctx,
        response,
        model=use_model,
        room_id=room_id,
        tools=active_tools,
        tools_enabled=tools_enabled,
        thread_user=thread_user,
        followup_instructions=followup_instructions,
        followup_input_items=followup_input_items,
    )
    sent_artifacts = await send_response_artifacts(
        ctx,
        response,
        room_id,
        provider=provider,
        thread_user=thread_user,
    )
    text = extract_text(response)
    if text:
        return text
    if sent_artifacts:
        return "Generated output attached."
    return ""


async def respond_with_tools(
    ctx: "AppContext",
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    room_id: Optional[str] = None,
    tool_choice: str = "auto",
    thread_user: Optional[str] = None,
) -> str:
    del tool_choice
    return await generate_reply(
        ctx,
        messages,
        model=model,
        room_id=room_id,
        use_tools=True,
        thread_user=thread_user,
    )
