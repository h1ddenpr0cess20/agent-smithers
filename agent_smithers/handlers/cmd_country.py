from __future__ import annotations

from typing import Any


async def handle_country(ctx: Any, room_id: str, sender_id: str, sender_display: str, args: str) -> None:
    """Admin: toggle search country filtering or show status.

    Args:
        ctx: App context.
        room_id: Matrix room ID.
        sender_id: Matrix user ID.
        sender_display: Sender display name.
        args: on/off/toggle/status.
    """
    country = ctx.cfg.llm.web_search_country
    if not country:
        body = "No search country configured (set TOOLS_WEB_SEARCH_COUNTRY)"
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return
    arg = (args or "").strip().lower()
    if arg in ("", "status"):
        state = "enabled" if ctx.search_country_enabled else "disabled"
        body = f"{country} search filtering is currently {state}"
        await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
        return
    if arg in ("on", "enable", "enabled"):
        ctx.search_country_enabled = True
    elif arg in ("off", "disable", "disabled"):
        ctx.search_country_enabled = False
    else:
        ctx.search_country_enabled = not ctx.search_country_enabled
    state = "enabled" if ctx.search_country_enabled else "disabled"
    body = f"{country} search filtering is now {state}"
    await ctx.matrix.send_text(room_id, body, html=ctx.render(body))
