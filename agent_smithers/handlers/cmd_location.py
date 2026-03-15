from __future__ import annotations

from typing import Any


async def handle_location(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Show or set the user's location.

    When set, the location is included in the system prompt so the model
    can tailor responses (e.g. local time, weather, recommendations).

    Args:
        ctx: App context.
        room_id: Matrix room ID.
        sender_id: Matrix user ID.
        sender_display: Sender display name.
        args: Location string, "clear" to remove, or empty to show current.
    """
    location = args.strip()

    if not location:
        current = ctx.history.get_location(sender_id)
        if current:
            body = f"**{sender_display}**, your location is set to: {current}"
        else:
            body = f"**{sender_display}**, you have no location set. Use `.location <place>` to set one."
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return

    if location.lower() in {"clear", "remove", "reset", "none"}:
        ctx.history.set_location(sender_id, "")
        body = f"Location cleared for {sender_display}"
        ctx.log(f"Location cleared for {sender_display} ({sender_id})")
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return

    ctx.history.set_location(sender_id, location)
    ctx.log(f"Location for {sender_display} ({sender_id}) set to '{location}'")
    body = f"Location for {sender_display} set to: {location}"
    await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
