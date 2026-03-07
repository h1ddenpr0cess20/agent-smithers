from __future__ import annotations

import base64
import json
import re
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from .context import AppContext


INLINE_CITATION_RE = re.compile(r"\s*[【〖][^】〗\n]*?†source[】〗]")


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
                text = strip_inline_citations(
                    str(content.get("text") or ""),
                    annotations=content.get("annotations"),
                )
                if text:
                    parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    return strip_inline_citations(str(response.get("output_text") or ""))


def strip_inline_citations(text: str, annotations: Any = None) -> str:
    cleaned = str(text or "")
    for annotation in annotations or []:
        if not isinstance(annotation, dict):
            continue
        annotation_type = str(annotation.get("type") or "").lower()
        if "citation" not in annotation_type:
            continue
        annotation_text = str(annotation.get("text") or "")
        if annotation_text:
            cleaned = cleaned.replace(annotation_text, "")
    cleaned = INLINE_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
) -> bool:
    if not room_id:
        return False
    sent_any = False
    for source in iter_image_sources(response):
        try:
            if source.get("inline"):
                image_bytes = decode_base64_image(str(source["inline"]))
            else:
                file_id = str(source.get("file_id") or "")
                if not file_id:
                    continue
                image_bytes = await ctx._download_image_bytes(
                    provider=provider,
                    file_id=file_id,
                    container_id=source.get("container_id"),
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


async def handle_generate_image_calls(
    ctx: "AppContext",
    response: Dict[str, Any],
    *,
    model: str,
    room_id: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    """Execute any generate_image function calls in the response and return tool output items."""
    calls = [
        item for item in (response.get("output") or [])
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") == "generate_image"
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
            img_response = await ctx.llm.generate_image(
                prompt=prompt,
                model=model,
                n=int(args["n"]) if args.get("n") else 1,
                aspect_ratio=str(args["aspect_ratio"]) if args.get("aspect_ratio") else None,
                resolution=str(args["resolution"]) if args.get("resolution") else None,
            )
            sent_any = False
            for entry in img_response.get("data", []) or []:
                b64 = entry.get("b64_json")
                if not b64:
                    continue
                image_bytes = base64.b64decode(b64)
                path = ctx._write_artifact(image_bytes, ".png")
                if room_id:
                    await ctx.matrix.send_image(room_id=room_id, path=path, filename=None, log=ctx.log)
                    sent_any = True
            result = "Image generated and sent." if sent_any else "Image generated."
        except Exception:
            ctx.logger.exception("generate_image tool call failed")
            result = "Image generation failed."
        if call_id:
            output_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            })
    return output_items or None


async def generate_reply(
    ctx: "AppContext",
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    room_id: Optional[str] = None,
    use_tools: Optional[bool] = None,
) -> str:
    use_model = model or ctx.model
    provider = ctx._provider_for_model(use_model)
    active_tools = ctx._tools_for_model(use_model)
    prepared_messages = ctx._apply_search_country_policy(
        messages,
        provider=provider,
        tools=active_tools,
    )
    ctx.hosted_tools = active_tools
    tools_enabled = ctx.tools_enabled if use_tools is None else bool(use_tools and active_tools)
    response = await ctx.llm.create_response(
        model=use_model,
        messages=prepared_messages,
        tools=active_tools if tools_enabled else None,
        tool_choice="auto" if tools_enabled and active_tools else None,
        options=ctx.options,
    )
    response = await maybe_continue_after_approvals(
        ctx,
        model=use_model,
        tools=active_tools if tools_enabled else None,
        response=response,
    )
    image_outputs = await handle_generate_image_calls(ctx, response, model=use_model, room_id=room_id)
    if image_outputs:
        response = await ctx.llm.create_response(
            model=use_model,
            previous_response_id=response.get("id"),
            input_items=image_outputs,
            tools=active_tools if tools_enabled else None,
            tool_choice="auto" if tools_enabled and active_tools else None,
            options=ctx.options,
        )
    sent_artifacts = await send_response_artifacts(ctx, response, room_id, provider=provider)
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
) -> str:
    del tool_choice
    return await generate_reply(ctx, messages, model=model, room_id=room_id, use_tools=True)
