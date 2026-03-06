from __future__ import annotations

from typing import Any


async def handle_ai(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Primary chat command: add user text and reply with model output.

    Args:
        ctx: App context.
        room_id: Matrix room ID.
        sender_id: Matrix user ID.
        sender_display: Sender display name.
        args: Optional text to add before generating a reply.
    """
    history = ctx.history
    matrix = ctx.matrix
    if args:
        history.add(room_id, sender_id, "user", args)
    messages = history.get(room_id, sender_id)
    model = ctx.user_models.get(room_id, {}).get(sender_id, ctx.model)
    try:
        response_text = await ctx.generate_reply(messages, model=model, room_id=room_id)
    except Exception as e:
        try:
            await matrix.send_text(room_id, "Something went wrong", html=ctx.render("Something went wrong"))
            ctx.log(e)
        except Exception:
            pass
        return
    text = ctx.clean_response_text(response_text or "", sender_display=sender_display, sender_id=sender_id)
    history.add(room_id, sender_id, "assistant", text)
    body = f"**{sender_display}**:\n{text}"
    html = ctx.render(body)
    try:
        ctx.log(f"Sending response to {sender_display} in {room_id}: {body}")
    except Exception:
        pass
    await matrix.send_text(room_id, body, html=html)
