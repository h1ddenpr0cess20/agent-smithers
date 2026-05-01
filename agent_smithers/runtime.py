from __future__ import annotations

import asyncio
import datetime as dt
import json
import signal
from typing import TYPE_CHECKING, Optional

from .config import AppConfig
from .context import AppContext

if TYPE_CHECKING:
    from .matrix_client import MatrixClientWrapper
from .handlers.cmd_ai import handle_ai
from .handlers.cmd_country import handle_country
from .handlers.cmd_help import handle_help
from .handlers.cmd_location import handle_location
from .handlers.cmd_model import handle_model
from .handlers.cmd_mymodel import handle_mymodel
from .handlers.cmd_prompt import handle_custom, handle_persona
from .handlers.cmd_reset import handle_clear, handle_reset
from .handlers.cmd_tools import handle_tools
from .handlers.cmd_verbose import handle_verbose
from .handlers.cmd_whitelist import handle_whitelist
from .handlers.cmd_x import handle_x
from .handlers.router import Router
from .security import Security


_THINKING_EMOJIS = ["🤔", "💭", "🧠"]
_THINKING_INTERVAL = 4.5
_GENERATING_HANDLERS = {handle_ai, handle_x, handle_persona, handle_custom}


def build_router() -> Router:
    router = Router()
    router.register(".ai", handle_ai)
    router.register(".x", handle_x)
    router.register(".persona", handle_persona)
    router.register(".custom", handle_custom)
    router.register(".reset", handle_reset)
    router.register(".stock", lambda c, r, s, d, _a: handle_reset(c, r, s, d, "stock"))
    router.register(".help", handle_help)
    router.register(".location", handle_location)
    router.register(".mymodel", handle_mymodel)
    router.register(".tools", handle_tools, admin=True)
    router.register(".verbose", handle_verbose, admin=True)
    router.register(".model", handle_model, admin=True)
    router.register(".clear", handle_clear, admin=True)
    router.register(".whitelist", handle_whitelist, admin=True)
    router.register(".country", handle_country, admin=True)
    return router


def persist_device_id(ctx: AppContext, config_path: Optional[str]) -> None:
    try:
        device_id = getattr(ctx.matrix.client, "device_id", None)
        if device_id and hasattr(ctx.cfg.matrix, "device_id") and not ctx.cfg.matrix.device_id and config_path:
            with open(config_path, "r+") as handle:
                data = json.load(handle)
                data.setdefault("matrix", {})["device_id"] = device_id
                handle.seek(0)
                json.dump(data, handle, indent=4)
                handle.truncate()
            ctx.log(f"Persisted device_id to {config_path}")
    except Exception:
        ctx.logger.exception("Failed to persist device_id to config")


def register_security_callbacks(ctx: AppContext, security: Security) -> None:
    try:
        from nio import KeyVerificationEvent  # type: ignore
    except ImportError:
        KeyVerificationEvent = None  # type: ignore
    try:
        if KeyVerificationEvent:
            ctx.matrix.add_to_device_callback(security.emoji_verification_callback, (KeyVerificationEvent,))
        ctx.matrix.add_to_device_callback(security.log_to_device_event, None)
    except Exception:
        pass


def install_signal_handlers(stop: asyncio.Event) -> None:
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except Exception:
                pass
    except Exception:
        pass


async def thinking_indicator(matrix: MatrixClientWrapper, room_id: str, target_event_id: str) -> None:
    """Add thinking emojis while the bot processes, then clean them up.

    Emojis are added one at a time (🤔, 💭, 🧠) with a delay between each,
    then all are redacted when the handler completes.

    Args:
        matrix: MatrixClientWrapper instance.
        room_id: Target room ID.
        target_event_id: The user's message event ID to react to.
    """
    # reaction_ids[i] is the currently-active reaction event id for _THINKING_EMOJIS[i],
    # or None if that slot is temporarily redacted.
    reaction_ids: list[Optional[str]] = [None] * len(_THINKING_EMOJIS)
    try:
        # Phase 1: accumulate all three emojis with a delay between each.
        for idx, emoji in enumerate(_THINKING_EMOJIS):
            reaction_id = await matrix.send_reaction(room_id, target_event_id, emoji)
            reaction_ids[idx] = reaction_id
            await asyncio.sleep(_THINKING_INTERVAL)
        # Phase 2: cycle — for each slot, redact briefly then re-add, in sequence.
        idx = 0
        half = _THINKING_INTERVAL / 2
        while True:
            slot = idx % len(_THINKING_EMOJIS)
            current = reaction_ids[slot]
            if current:
                await matrix.redact_event(room_id, current)
            await asyncio.sleep(half)
            new_id = await matrix.send_reaction(room_id, target_event_id, _THINKING_EMOJIS[slot])
            if new_id:
                reaction_ids[slot] = new_id
            await asyncio.sleep(half)
            idx += 1
    except asyncio.CancelledError:
        pending = [rid for rid in reaction_ids if rid]
        if pending:
            await asyncio.gather(
                *(matrix.redact_event(room_id, rid) for rid in pending),
                return_exceptions=True,
            )
        raise


async def run(cfg: AppConfig, config_path: Optional[str] = None) -> None:
    """Run the Matrix bot runtime loop."""
    ctx = AppContext(cfg)
    if cfg.llm.server_models:
        await ctx.refresh_models()

    router = build_router()
    ctx.log(f"Model set to {ctx.model}")

    await ctx.matrix.load_store()
    login_resp = await ctx.matrix.login()
    try:
        ctx.log(login_resp)
    except Exception:
        pass
    await ctx.matrix.ensure_keys()
    await ctx.matrix.initial_sync()

    ctx.bot_id = await ctx.matrix.display_name(cfg.matrix.username)
    persist_device_id(ctx, config_path)

    for room in cfg.matrix.channels:
        try:
            await ctx.matrix.join(room)
            ctx.log(f"{ctx.bot_id} joined {room}")
        except Exception:
            ctx.log(f"Couldn't join {room}")

    security = Security(ctx.matrix, logger=ctx.logger)
    register_security_callbacks(ctx, security)
    join_time = dt.datetime.now()

    async def on_text(room, event) -> None:
        try:
            message_time = getattr(event, "server_timestamp", 0) / 1000.0
            message_time = dt.datetime.fromtimestamp(message_time)
            if message_time <= join_time:
                return
            text = getattr(event, "body", "")
            sender = getattr(event, "sender", "")
            if sender == cfg.matrix.username:
                return
            sender_display = await ctx.matrix.display_name(sender)
            is_admin = sender_display in ctx.admins or sender in ctx.admins
            handler, args = router.dispatch(
                ctx,
                room.room_id,
                sender,
                sender_display,
                text,
                is_admin,
                bot_name=ctx.bot_id,
                timestamp=message_time,
            )
            if handler is None:
                return
            try:
                ctx.log(f"{sender_display} ({sender}) sent {text} in {room.room_id}")
            except Exception:
                pass
            try:
                await security.allow_devices(sender)
            except Exception:
                pass
            user_event_id = getattr(event, "event_id", None)
            should_indicate = handler in _GENERATING_HANDLERS and user_event_id
            if should_indicate:
                indicator = asyncio.create_task(
                    thinking_indicator(ctx.matrix, room.room_id, user_event_id)
                )
                ctx.thinking_indicator = indicator
            else:
                indicator = None
            try:
                result = handler(*args)
                if asyncio.iscoroutine(result):
                    await result
            finally:
                if indicator:
                    indicator.cancel()
                    try:
                        await indicator
                    except asyncio.CancelledError:
                        pass
                ctx.thinking_indicator = None
        except Exception as exc:
            ctx.log(exc)

    ctx.matrix.add_text_handler(on_text)

    stop = asyncio.Event()
    install_signal_handlers(stop)
    sync_task = asyncio.create_task(ctx.matrix.sync_forever())
    stop_task = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({sync_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        pass
    finally:
        for task in (sync_task, stop_task):
            if not task.done():
                task.cancel()
        try:
            if hasattr(ctx.matrix, "shutdown"):
                await ctx.matrix.shutdown()
        except Exception:
            pass
        try:
            ctx.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
