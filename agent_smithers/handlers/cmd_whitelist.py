from __future__ import annotations

from typing import Any


async def handle_whitelist(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Admin: manage the video generation whitelist.

    Subcommands:
        .whitelist add <user>    — add a user (ID or display name)
        .whitelist remove <user> — remove a user
        .whitelist list          — show current whitelist
    """
    parts = (args or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "add":
        if not arg:
            body = "Usage: `.whitelist add <user>`"
        else:
            ctx.video_whitelist.add(arg)
            if not ctx.video_whitelist_enabled:
                ctx.video_whitelist_enabled = True
            body = f"Added **{arg}** to video whitelist"

    elif sub == "remove":
        if not arg:
            body = "Usage: `.whitelist remove <user>`"
        else:
            ctx.video_whitelist.discard(arg)
            body = f"Removed **{arg}** from video whitelist"

    elif sub == "list":
        if ctx.video_whitelist:
            entries = ", ".join(sorted(ctx.video_whitelist))
            body = f"Video whitelist: {entries}"
        else:
            body = "Video whitelist is empty — all users can generate video"

    else:
        body = "Usage: `.whitelist add|remove|list`"

    await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
