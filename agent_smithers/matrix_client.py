"""Async wrapper around the matrix-nio client.

:class:`MatrixClientWrapper` centralizes login, store/key management, sync,
message sending/editing/redaction, media uploads, and event-callback
registration so the rest of the app talks to a small, intention-revealing
surface instead of nio directly.
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
from typing import Any, Awaitable, Callable, Optional

from nio import AsyncClient, AsyncClientConfig, MatrixRoom, MegolmEvent, RoomMessageText, KeyVerificationEvent

from .markdown_utils import render_markdown


TextHandler = Callable[[Any, Any], Awaitable[None]]


class MatrixClientWrapper:
    """Convenience wrapper around ``nio.AsyncClient``.

    Owns a configured ``AsyncClient`` and exposes a small, intention-revealing
    surface for the operations the bot needs — login, store/key management,
    one-shot and long-polling sync, sending/editing/redacting messages,
    reactions, media uploads, display-name lookup, and event-callback
    registration. Network operations swallow and log errors so a transient
    Matrix failure never crashes the runtime.
    """
    def __init__(
        self,
        server: str,
        username: str,
        password: str,
        device_id: str = "",
        store_path: str = "store",
        encryption_enabled: bool = True,
    ) -> None:
        """Initialize the underlying nio client and store configuration.

        Args:
            server: Homeserver base URL.
            username: Bot user ID.
            password: Account password.
            device_id: Optional persisted device ID for E2EE.
            store_path: Directory for SQLite store files.
            encryption_enabled: Whether to enable E2EE features.
        """
        # Ensure store path exists for nio's SQLite store (peewee) to open DB
        try:
            os.makedirs(store_path, exist_ok=True)
        except Exception:
            # If creation fails, nio will surface an error; continue
            pass
        try:
            cfg = AsyncClientConfig(encryption_enabled=encryption_enabled, store_sync_tokens=True)
        except ImportWarning:
            cfg = AsyncClientConfig(encryption_enabled=False, store_sync_tokens=True)
        self.client = AsyncClient(server, username, device_id=device_id or None, store_path=store_path, config=cfg)
        try:
            self.client.user_id = username
        except Exception:
            pass
        self.password = password

    async def login(self) -> Any:
        """Log in to the homeserver with the configured password.

        Uses the persisted device id as the device name when available.

        Returns:
            The nio login response object.
        """
        return await self.client.login(self.password, device_name=self.client.device_id or "agent-smithers")

    async def ensure_keys(self) -> None:
        """Upload device keys when the client reports they are needed.

        No-op unless nio indicates ``should_upload_keys``; relevant only when
        end-to-end encryption is enabled.
        """
        if getattr(self.client, "should_upload_keys", False):
            await self.client.keys_upload()

    async def load_store(self) -> None:
        """Load the local nio store if the client supports it.

        Tolerates clients without a ``load_store`` method and handles both
        sync and async implementations.
        """
        result = getattr(self.client, "load_store", None)
        if callable(result):
            maybe = result()
            if asyncio.iscoroutine(maybe):
                await maybe

    async def join(self, room_id: str) -> None:
        """Join a room by ID or alias.

        Args:
            room_id: The room ID or alias to join.
        """
        await self.client.join(room_id)

    async def send_text(self, room_id: str, body: str, html: Optional[str] = None) -> Optional[str]:
        """Send a text message, optionally with HTML formatting.

        Args:
            room_id: Target room ID.
            body: Plaintext body.
            html: Optional formatted body; when provided, sends custom HTML.

        Returns:
            The event ID of the sent message, or None on failure.
        """
        content = {"msgtype": "m.text", "body": body}
        if html is not None:
            content.update({"format": "org.matrix.custom.html", "formatted_body": html})
        try:
            resp = await self.client.room_send(room_id=room_id, message_type="m.room.message", content=content, ignore_unverified_devices=True)
            return getattr(resp, "event_id", None)
        except Exception:
            return None

    async def edit_message(self, room_id: str, event_id: str, body: str, html: Optional[str] = None) -> None:
        """Edit an existing message using the m.replace relation.

        Args:
            room_id: Target room ID.
            event_id: Event ID of the message to replace.
            body: New plain-text body.
            html: Optional new HTML body.
        """
        new_content: dict = {"msgtype": "m.text", "body": body}
        if html is not None:
            new_content.update({"format": "org.matrix.custom.html", "formatted_body": html})
        content = {
            **new_content,
            "body": f"* {body}",
            "m.relates_to": {"rel_type": "m.replace", "event_id": event_id},
            "m.new_content": new_content,
        }
        try:
            await self.client.room_send(room_id=room_id, message_type="m.room.message", content=content, ignore_unverified_devices=True)
        except Exception:
            pass

    async def send_markdown(self, room_id: str, message: str) -> None:
        """Render Markdown to HTML and send it as a formatted message.

        Args:
            room_id: Target room ID.
            message: Markdown source; sent as the plain-text body with the
                rendered HTML as the formatted body.
        """
        html = render_markdown(message)
        await self.send_text(room_id, message, html=html)

    async def _send_media(
        self,
        room_id: str,
        path: str,
        filename: str | None,
        log,
        *,
        msgtype: str,
        missing_label: str,
    ) -> None:
        """Upload a local media file and send it to a room.

        Args:
            room_id: Target room ID.
            path: Local filesystem path to the media file.
            filename: Optional display filename; defaults to basename of path.
            log: Logging callable for status/error output.
            msgtype: Matrix message type, for example `m.image` or `m.video`.
            missing_label: Human-readable media label for user-facing errors.
        """
        if not path or not os.path.exists(path):
            log(f"Error sending {missing_label}: Invalid path '{path}'")
            await self.send_markdown(room_id, f"Error: Could not find {missing_label} file at {path}")
            return
        if not filename:
            filename = os.path.basename(path)
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        file_stat = os.stat(path)
        try:
            with open(path, "rb") as fp:
                upload_response, _ = await self.client.upload(fp, content_type=mime_type, filename=filename, filesize=file_stat.st_size)
            if not upload_response or not hasattr(upload_response, "content_uri"):
                log(f"Failed to upload {missing_label}: Invalid response {upload_response}")
                await self.send_markdown(room_id, f"Failed to upload {missing_label} '{filename}'.")
                return
            content_uri = upload_response.content_uri
            content = {"body": filename, "info": {"mimetype": mime_type, "size": file_stat.st_size}, "msgtype": msgtype, "url": content_uri}
            await self.client.room_send(room_id=room_id, message_type="m.room.message", content=content, ignore_unverified_devices=True)
        except Exception as e:
            log(f"Error sending {missing_label} to {room_id}: {e}")
            await self.send_markdown(room_id, f"Sorry, an error occurred while trying to send the {missing_label}: {e}")

    async def send_image(self, room_id: str, path: str, filename: str | None, log) -> None:
        """Upload a local image file and send it as an ``m.image`` message.

        Args:
            room_id: Target room ID.
            path: Local filesystem path to the image.
            filename: Optional display filename; defaults to the path basename.
            log: Logging callable for status/error output.
        """
        await self._send_media(room_id, path, filename, log, msgtype="m.image", missing_label="image")

    async def send_video(self, room_id: str, path: str, filename: str | None, log) -> None:
        """Upload a local video file and send it as an ``m.video`` message.

        Args:
            room_id: Target room ID.
            path: Local filesystem path to the video.
            filename: Optional display filename; defaults to the path basename.
            log: Logging callable for status/error output.
        """
        await self._send_media(room_id, path, filename, log, msgtype="m.video", missing_label="video")

    async def display_name(self, user_id: str) -> str:
        """Look up a user's display name, falling back to the user ID.

        Args:
            user_id: The Matrix user ID to resolve.

        Returns:
            The display name, or ``user_id`` when it is unset or the lookup
            fails.
        """
        try:
            res = await self.client.get_displayname(user_id)
            return getattr(res, "displayname", None) or user_id
        except Exception:
            return user_id

    async def send_reaction(self, room_id: str, event_id: str, key: str) -> Optional[str]:
        """Send a reaction emoji to an event.

        Args:
            room_id: Target room ID.
            event_id: Event ID to react to.
            key: Emoji or unicode string to react with.

        Returns:
            The reaction event ID, or None on failure.
        """
        content = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": key,
            }
        }
        try:
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.reaction",
                content=content,
                ignore_unverified_devices=True,
            )
            return getattr(resp, "event_id", None)
        except Exception as e:
            import logging
            logging.debug(f"Failed to send reaction {key} to {event_id}: {e}")
            return None

    async def redact_event(self, room_id: str, event_id: str) -> None:
        """Redact (delete) an event by its ID.

        Args:
            room_id: Target room ID.
            event_id: Event ID to redact.
        """
        try:
            await self.client.room_redact(room_id=room_id, event_id=event_id)
        except Exception:
            pass

    def add_text_handler(self, handler: TextHandler) -> None:
        """Register a callback for incoming room text messages.

        Wraps ``handler`` in a nio ``RoomMessageText`` event callback.

        Args:
            handler: Async callable invoked with ``(room, event)`` per message.
        """
        async def _cb(room: MatrixRoom, event: RoomMessageText) -> None:  # type: ignore
            """Forward the nio text event to the registered handler.

            Args:
                room: The room the event arrived in.
                event: The nio text message event.
            """
            await handler(room, event)
        self.client.add_event_callback(_cb, RoomMessageText)  # type: ignore

    def add_megolm_handler(self, handler: TextHandler) -> None:
        """Register a callback for undecryptable (Megolm) messages.

        Wraps ``handler`` in a nio ``MegolmEvent`` callback so the runtime can
        react to messages it cannot yet decrypt.

        Args:
            handler: Async callable invoked with ``(room, event)`` per event.
        """
        async def _cb(room: MatrixRoom, event: MegolmEvent) -> None:  # type: ignore
            """Forward the nio undecryptable event to the registered handler.

            Args:
                room: The room the event arrived in.
                event: The undecryptable Megolm event.
            """
            await handler(room, event)
        self.client.add_event_callback(_cb, MegolmEvent)  # type: ignore

    async def request_room_key(self, event: Any) -> None:
        """Request the missing Megolm session key for an event.

        Asks the sender's devices to re-share the key so the event can be
        decrypted on a later sync. Errors are swallowed.

        Args:
            event: The undecryptable event whose room key is needed.
        """
        try:
            await self.client.request_room_key(event)
        except Exception:
            pass

    def add_to_device_callback(self, callback, event_types=None) -> None:
        """Register a to-device event callback, tolerating unsupported clients.

        Args:
            callback: The callback to register with nio.
            event_types: Optional iterable of event types to filter on; passes
                ``None`` through to receive all to-device events.
        """
        try:
            self.client.add_to_device_callback(callback, event_types)
        except Exception:
            pass

    async def initial_sync(self, timeout_ms: int = 3000) -> None:
        """Perform a single full-state sync after login.

        Args:
            timeout_ms: Long-poll timeout for the sync, in milliseconds.
        """
        await self.client.sync(timeout=timeout_ms, full_state=True)

    async def sync_forever(self, timeout_ms: int = 30000) -> None:
        """Run the long-polling sync loop until the task is cancelled.

        Args:
            timeout_ms: Per-iteration long-poll timeout, in milliseconds.
        """
        await self.client.sync_forever(timeout=timeout_ms, full_state=True)

    async def shutdown(self) -> None:
        """Log out and close the client, ignoring any failures.

        Best-effort cleanup invoked during runtime shutdown; both logout and
        close are attempted independently and never raise.
        """
        try:
            if hasattr(self.client, "logout"):
                await self.client.logout()  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            if hasattr(self.client, "close"):
                await self.client.close()  # type: ignore[arg-type]
        except Exception:
            pass
