from __future__ import annotations

from typing import Any


async def handle_persona(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Set a persona and seed a response for the user.

    Args:
        ctx: App context.
        room_id: Matrix room ID.
        sender_id: Matrix user ID.
        sender_display: Sender display name.
        args: Persona string to apply.
    """
    persona = args.strip()
    # Initialize a fresh system prompt using persona
    try:
        ctx.history.init_prompt(room_id, sender_id, persona=persona or ctx.default_personality)
        ctx.log(
            f"System prompt for {sender_display} ({sender_id}) set to '{(ctx.cfg.llm.prompt[0] if ctx.cfg.llm.prompt else 'you are ')}{persona or ctx.default_personality}{(ctx.cfg.llm.prompt[1] if len(ctx.cfg.llm.prompt) > 1 else '.')}"  # noqa: E501
        )
    except Exception:
        pass
    ctx.history.add(room_id, sender_id, "user", "introduce yourself")
    await _respond(ctx, room_id, sender_id, sender_display)


async def handle_custom(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Set a custom system prompt for the user and seed a response.

    Args:
        ctx: App context.
        room_id: Matrix room ID.
        sender_id: Matrix user ID.
        sender_display: Sender display name.
        args: Custom system prompt text.
    """
    custom = args.strip()
    if not custom:
        return
    # Replace system prompt with custom text
    try:
        ctx.history.init_prompt(room_id, sender_id, custom=custom)
        ctx.log(f"System prompt for {sender_display} ({sender_id}) set to '{custom}'")
    except Exception:
        pass
    ctx.history.add(room_id, sender_id, "user", "introduce yourself")
    await _respond(ctx, room_id, sender_id, sender_display)


async def _respond(ctx: Any, room_id: str, user_id: str, header_display: str) -> None:
    """Helper to request a reply for the current user and send it."""
    messages = ctx.history.get(room_id, user_id)
    model = ctx.user_models.get(room_id, {}).get(user_id, ctx.model)
    try:
        response_text = await ctx.generate_reply(messages, model=model, room_id=room_id)
    except Exception as e:
        try:
            await ctx.matrix.send_text(room_id, "Something went wrong", html=ctx.render("Something went wrong"))
            ctx.log(e)
        except Exception:
            pass
        return
    response_text = ctx.clean_response_text(
        response_text or "",
        sender_display=header_display,
        sender_id=user_id,
    )
    ctx.history.add(room_id, user_id, "assistant", response_text)
    body = f"**{header_display}**:\n{response_text}"
    html = ctx.render(body)
    try:
        ctx.log(f"Sending response to {header_display} in {room_id}: {body}")
    except Exception:
        pass
    await ctx.matrix.send_text(room_id, body, html=html)
