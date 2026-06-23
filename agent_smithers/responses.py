"""Response generation, tool execution, and artifact handling.

Implements the core request/response loop that :class:`~.context.AppContext`
delegates to: cleaning model output, extracting text and image/video sources,
downloading and sending media artifacts, handling MCP approval requests, and
executing the local Grok image/video generation tools.
"""
from __future__ import annotations

import base64
import json
import os
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
)


INLINE_CITATION_RE = None
IMAGE_GENERATION_TOOL_NAMES = {GROK_GENERATE_IMAGE_TOOL}
IMAGE_EDIT_TOOL_NAMES = {GROK_EDIT_IMAGE_TOOL}
VIDEO_GENERATION_TOOL_NAMES = {GROK_GENERATE_VIDEO_TOOL}
LOCAL_MEDIA_TOOL_NAMES = IMAGE_GENERATION_TOOL_NAMES | IMAGE_EDIT_TOOL_NAMES | VIDEO_GENERATION_TOOL_NAMES


def _is_video_allowed(ctx: "AppContext", user_id: Optional[str]) -> bool:
    """Check whether a user may generate video, per the allowlist.

    Everyone is allowed when the allowlist is disabled; otherwise the user
    must be an admin or an explicit allowlist entry. A missing user id is
    treated as not allowed.

    Args:
        ctx: Application context holding the allowlist and admins.
        user_id: The requesting user's id, if known.

    Returns:
        ``True`` if the user is allowed to generate video.
    """
    if not ctx.video_whitelist_enabled:
        return True
    if not user_id:
        return False
    if user_id in ctx.admins:
        return True
    return user_id in ctx.video_whitelist


def clean_response_text(
    ctx: "AppContext",
    text: str,
    *,
    sender_display: str,
    sender_id: str,
) -> str:
    """Strip reasoning/solution scaffolding from a model response.

    Removes ``<think>``, ``<|begin_of_thought|>``, and ``<|begin_of_solution|>``
    style blocks, logging any extracted thinking, and returns only the final
    user-facing answer.

    Args:
        ctx: Application context, used for logging extracted reasoning.
        text: Raw model output text.
        sender_display: Display name of the requesting user (for logs).
        sender_id: Matrix ID of the requesting user (for logs).

    Returns:
        The cleaned, stripped response text.
    """
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
    """Extract the assistant's text from a Responses API payload.

    Concatenates ``output_text`` content across message items, falling back to
    the top-level ``output_text`` convenience field.

    Args:
        response: The decoded Responses API payload.

    Returns:
        The combined assistant text, or an empty string when none is present.
    """
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


def strip_inline_citations(text: str, annotations: Any = None) -> str:
    """Return the text unchanged (inline citations are now preserved).

    Kept as a stable no-op so callers and tests have a single seam should
    citation stripping ever be reintroduced.

    Args:
        text: The response text.
        annotations: Ignored; accepted for backward compatibility.

    Returns:
        The original text, coerced to ``str``.
    """
    del annotations
    return str(text or "")


def walk_image_results(value: Any) -> Iterable[Dict[str, Any]]:
    """Recursively yield image sources from an arbitrary result value.

    Handles base64 strings, lists, and dicts carrying inline data, ``file_id``
    references, or nested ``file`` objects.

    Args:
        value: A string, list, or dict from an image tool result.

    Yields:
        Dicts describing each image source, either ``{"inline": ...}`` or
        ``{"file_id": ..., "container_id": ...}``.
    """
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
    """Yield every image source embedded in a Responses API payload.

    Walks ``image_generation_call`` results plus ``output_image`` content and
    file-citation annotations on message items, propagating container ids.

    Args:
        response: The decoded Responses API payload.

    Yields:
        Image source dicts (inline data or file references) suitable for
        :func:`walk_image_results` consumers.
    """
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
    """Decode a base64 image, accepting raw or ``data:`` URI input.

    Args:
        data: A base64 string or a ``data:...;base64,...`` URI.

    Returns:
        The decoded image bytes.
    """
    encoded = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    return base64.b64decode(encoded)


def write_artifact(ctx: "AppContext", data: bytes, suffix: str) -> str:
    """Write bytes to a uniquely named file in the artifact directory.

    Args:
        ctx: Application context providing the artifact directory.
        data: Raw bytes to persist.
        suffix: File suffix/extension (e.g. ``".png"``).

    Returns:
        The path to the written temporary artifact file.
    """
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
    """Download an image file produced by a hosted tool.

    Args:
        ctx: Application context providing the LLM client.
        provider: Provider that produced the file.
        file_id: Identifier of the file to download.
        container_id: Optional code-interpreter container id.

    Returns:
        The raw image bytes.
    """
    return await ctx.llm.download_file(file_id, provider=provider, container_id=container_id)


async def send_response_artifacts(
    ctx: "AppContext",
    response: Dict[str, Any],
    room_id: Optional[str],
    *,
    provider: str,
    thread_user: Optional[str] = None,
) -> bool:
    """Send every image embedded in a response to a Matrix room.

    Decodes inline images and downloads file-referenced ones, caches each as
    the thread's latest generated image, writes a temp artifact, and uploads it.

    Args:
        ctx: Application context.
        response: The decoded Responses API payload.
        room_id: Destination room, or ``None`` to skip sending.
        provider: Provider used to authenticate file downloads.
        thread_user: Optional user id for remembering generated media.

    Returns:
        ``True`` if at least one image was sent.
    """
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
            try:
                await ctx.matrix.send_image(room_id=room_id, path=path, filename=None, log=ctx.log)
                sent_any = True
            finally:
                os.unlink(path)
        except Exception:
            ctx.logger.exception("Failed to send generated image")
    return sent_any


def approval_items(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect MCP approval-request items from a response payload.

    Args:
        response: The decoded Responses API payload.

    Returns:
        The list of ``mcp_approval_request``/``mcp_approval_request_item``
        output items (possibly empty).
    """
    items: List[Dict[str, Any]] = []
    for item in response.get("output", []) or []:
        item_type = str(item.get("type") or "")
        if item_type in {"mcp_approval_request", "mcp_approval_request_item"}:
            items.append(item)
    return items


def should_auto_approve(ctx: "AppContext", item: Dict[str, Any]) -> bool:
    """Decide whether an MCP approval request is auto-approved.

    Args:
        ctx: Application context holding the auto-approve server set.
        item: A single MCP approval-request item.

    Returns:
        ``True`` if the item's server label is configured for auto-approval.
    """
    label = str(item.get("server_label") or item.get("mcp_server_label") or "").strip()
    return bool(label and label in ctx._mcp_auto_approve)


async def maybe_continue_after_approvals(
    ctx: "AppContext",
    *,
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Auto-approve pending MCP requests and continue until none remain.

    Repeatedly approves the auto-approvable requests in the current response
    and re-calls the model, looping until the response carries no further
    auto-approvable requests.

    Args:
        ctx: Application context (LLM client, options, auto-approve set).
        model: Model to continue the turn with.
        tools: Tools to keep attached when tools are enabled.
        response: The response carrying the initial approval requests.

    Returns:
        The settled response once no auto-approvable requests remain.
    """
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
    """Drive a response to completion through approvals and local tool calls.

    Loops: auto-approves any pending MCP requests and executes local media
    tool calls, feeding their outputs back to the model, until a turn yields
    no further approvals or tool calls. Either threads state via
    ``previous_response_id`` or accumulates explicit input items, depending on
    whether ``followup_input_items`` was supplied (OpenAI follow-up path).

    Args:
        ctx: Application context.
        response: The initial model response to settle.
        model: Model used for any continuation requests.
        room_id: Room for tool output and media delivery.
        tools: Tools to keep attached on continuation requests.
        tools_enabled: Whether tools should be sent on continuations.
        thread_user: Optional user id for thread media context.
        followup_instructions: Instructions for accumulated-input continuations.
        followup_input_items: Seed input items; when provided, state is carried
            explicitly rather than via ``previous_response_id``.

    Returns:
        The settled response once no approvals or tool calls remain.
    """
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
    """Execute local Grok media tool calls and collect their outputs.

    Finds ``function_call`` items naming the local image/edit/video tools,
    runs each (enforcing the video allowlist), and returns the matching
    ``function_call_output`` items to feed back to the model. Per-call
    failures are caught and reported as a short error output string.

    Args:
        ctx: Application context.
        response: The model response that may contain media tool calls.
        model: Model that issued the calls.
        room_id: Room for generated media delivery.
        thread_user: Optional user id for allowlist checks and media context.

    Returns:
        The list of tool-output items, or ``None`` when there were no local
        media calls to handle.
    """
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
            elif not _is_video_allowed(ctx, thread_user):
                result = "Video generation is restricted. You are not on the whitelist."
            else:
                result = await _execute_generate_video_call(
                    ctx,
                    model=model,
                    room_id=room_id,
                    thread_user=thread_user,
                    prompt=prompt,
                    args=args,
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
    """Send base64 images from an image API ``data`` array to a room.

    Args:
        ctx: Application context.
        response: An image generation/edit API payload with a ``data`` list.
        room_id: Destination room, or ``None`` to skip sending.
        thread_user: Optional user id for remembering generated media.

    Returns:
        ``True`` if at least one image was sent.
    """
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
        try:
            if room_id:
                await ctx.matrix.send_image(room_id=room_id, path=path, filename=None, log=ctx.log)
                sent_any = True
        finally:
            os.unlink(path)
    return sent_any


def _extract_image_edit_urls(args: Dict[str, Any]) -> List[str]:
    """Collect and de-duplicate source image URLs from edit-tool arguments.

    Merges a singular ``image_url`` (first) with any ``image_urls`` list,
    dropping blanks and duplicates while preserving order.

    Args:
        args: The decoded tool-call arguments.

    Returns:
        An ordered, de-duplicated list of source image URLs.
    """
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
    """Guess a file suffix from a media URL's path.

    Args:
        url: The media URL to inspect.
        default: Suffix to fall back to when none can be derived or the
            candidate looks implausible (missing dot or over 8 chars).

    Returns:
        A lowercase extension including the leading dot, or ``default``.
    """
    suffix = urlparse(url).path.rsplit("/", 1)[-1]
    if "." not in suffix:
        return default
    ext = "." + suffix.rsplit(".", 1)[-1].lower()
    if not ext or len(ext) > 8:
        return default
    return ext


def _extract_video_url(payload: Dict[str, Any]) -> Optional[str]:
    """Pull the downloadable video URL out of a generation payload.

    Checks the top-level ``url`` then the nested ``video``/``result``/
    ``output`` containers.

    Args:
        payload: The decoded video generation/poll response.

    Returns:
        The first non-empty video URL found, or ``None``.
    """
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
    """Run a Grok image-generation tool call and send the result.

    Args:
        ctx: Application context.
        model: Model that issued the call (provider is forced to xAI).
        room_id: Destination room for the generated image.
        thread_user: Optional user id for remembering generated media.
        prompt: The image prompt.
        args: Remaining tool-call arguments (``n``, ``aspect_ratio``,
            ``resolution``).

    Returns:
        A short status string reported back to the model as tool output.
    """
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
    """Run a Grok image-edit tool call and send the result.

    Falls back to the thread's most recently generated image when no source
    URL is supplied.

    Args:
        ctx: Application context.
        model: Model that issued the call (provider is forced to xAI).
        room_id: Destination room for the edited image.
        thread_user: Optional user id for media context and remembering.
        prompt: The edit instruction prompt.
        args: Remaining tool-call arguments (image URLs, ``n``, etc.).

    Returns:
        A short status string reported back to the model as tool output.

    Raises:
        ValueError: If no source image can be resolved.
    """
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
) -> str:
    """Run a Grok video-generation/edit tool call and send the result.

    Resolves the source image or video (falling back to the thread's latest
    generated media), generates the video, downloads it, and uploads it.

    Args:
        ctx: Application context.
        model: Model that issued the call (provider is forced to xAI).
        room_id: Destination room for the generated video.
        thread_user: Optional user id for media context and remembering.
        prompt: The video prompt.
        args: Remaining tool-call arguments (image/video URL, ``duration``,
            ``aspect_ratio``, ``resolution``).

    Returns:
        A short status string reported back to the model as tool output.

    Raises:
        ValueError: If both image and video sources are given, or the
            response carries no downloadable URL.
    """
    image_url = str(args.get("image_url") or "").strip() or None
    video_url = str(args.get("video_url") or "").strip() or None
    if not image_url and not video_url:
        image_url = ctx._latest_generated_media(room_id, thread_user, kind="image")
        if not image_url:
            video_url = ctx._latest_generated_media(room_id, thread_user, kind="video")
    if image_url and video_url:
        raise ValueError("Only one of image_url or video_url may be provided")
    action = "Animating image" if image_url else "Editing video" if video_url else "Generating video"
    with ctx.status(f"{action} with Grok") as status:
        video_response = await ctx.llm.generate_video(
            prompt=prompt,
            model=model,
            backend="grok",
            image_url=image_url,
            video_url=video_url,
            duration=int(args["duration"]) if args.get("duration") is not None else None,
            aspect_ratio=str(args["aspect_ratio"]) if args.get("aspect_ratio") else None,
            resolution=str(args["resolution"]) if args.get("resolution") else None,
            on_status=status.update,
        )
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
        status.update("Downloading video from Grok")
        video_bytes = await ctx.llm.download_url(final_url, provider="xai")
    path = ctx._write_artifact(video_bytes, suffix)
    try:
        if room_id:
            await ctx.matrix.send_video(room_id=room_id, path=path, filename=None, log=ctx.log)
            return "Video generated and sent."
        return "Video generated."
    finally:
        os.unlink(path)


async def generate_reply(
    ctx: "AppContext",
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    room_id: Optional[str] = None,
    use_tools: Optional[bool] = None,
    thread_user: Optional[str] = None,
) -> str:
    """Generate a chat reply, running tools and sending media as needed.

    Prepares the messages (thread media note, search-country policy), calls the
    provider, settles any tool/approval follow-ups, sends image/video
    artifacts, and returns the cleaned text.

    Args:
        ctx: Application context.
        messages: The chat history to respond to.
        model: Model override; defaults to the active model.
        room_id: Room context for tool output and media delivery.
        use_tools: Force-enable/disable tools; ``None`` uses the model default.
        thread_user: Optional user id for thread media context.

    Returns:
        The final user-facing reply text.
    """
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
    """Generate a reply with tools force-enabled.

    Thin wrapper over :func:`generate_reply` that always enables tools; the
    ``tool_choice`` argument is accepted for API compatibility but ignored.

    Args:
        ctx: Application context.
        messages: The chat history to respond to.
        model: Model override; defaults to the active model.
        room_id: Room context for tool output and media delivery.
        tool_choice: Accepted for compatibility; currently ignored.
        thread_user: Optional user id for thread media context.

    Returns:
        The final user-facing reply text.
    """
    del tool_choice
    return await generate_reply(
        ctx,
        messages,
        model=model,
        room_id=room_id,
        use_tools=True,
        thread_user=thread_user,
    )
