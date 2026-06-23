"""Handler for the ``.thinking`` admin command: toggle the typing placeholder."""
from __future__ import annotations

from typing import Any


async def handle_thinking(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Admin command to view or toggle the thinking placeholder.

    Usage: .thinking [on|off|toggle]
    """
    arg = args.strip().lower()
    if not arg:
        state = "ON" if getattr(ctx, "thinking", False) else "OFF"
        body = f"Thinking placeholder is **{state}**"
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return
    if arg == "on":
        new_state = True
    elif arg == "off":
        new_state = False
    elif arg == "toggle":
        new_state = not bool(getattr(ctx, "thinking", False))
    else:
        body = "Usage: .thinking [on|off|toggle]"
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return
    ctx.thinking = new_state
    state = "ON" if new_state else "OFF"
    body = f"Thinking placeholder set to **{state}**"
    await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
