"""Per-room conversation history with optional encrypted persistence.

Defines :class:`HistoryStore`, which keeps a system-seeded message list per
room/user thread, composes the system prompt from the persona/prefix/suffix
(plus optional per-user location), trims threads to a token budget, and can
transparently persist everything to an encrypted file on disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class HistoryStore:
    """In-memory history per room and user with system prompt support.

    When ``store_path`` and ``encryption_key`` are provided, history is
    persisted to an encrypted file and restored on startup.
    """

    def __init__(
        self,
        prompt_prefix: str = "you are ",
        prompt_suffix: str = ".",
        personality: str = "",
        *,
        prompt_suffix_extra: str = "",
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        store_path: Optional[str] = None,
        encryption_key: Optional[str] = None,
    ) -> None:
        """Initialize the history store.

        Args:
            prompt_prefix: Text placed before the personality in the system
                prompt (ignored when ``system_prompt`` is given).
            prompt_suffix: Text placed after the personality in the system
                prompt (ignored when ``system_prompt`` is given).
            personality: The persona description woven into the system prompt.
            prompt_suffix_extra: Optional extra suffix appended unless verbose
                mode is enabled.
            max_tokens: Soft cap on retained history tokens per conversation.
            system_prompt: If provided, used verbatim as the system prompt,
                bypassing the prefix/suffix/personality composition.
            store_path: Directory for the encrypted history file; enables
                persistence only when combined with ``encryption_key``.
            encryption_key: Fernet key used to encrypt persisted history.
        """
        # Allow alternate constructor via system_prompt
        if system_prompt is not None:
            self.prompt_prefix = ""
            self.prompt_suffix = ""
            self.prompt_suffix_extra = ""
            self.personality = ""
            self._fixed_system_prompt = system_prompt
        else:
            self.prompt_prefix = prompt_prefix
            self.prompt_suffix = prompt_suffix
            self.prompt_suffix_extra = prompt_suffix_extra
            self.personality = personality
            self._fixed_system_prompt = None
        self.max_tokens = max_tokens
        self._include_extra = True
        self._messages: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        self._locations: Dict[str, str] = {}  # user_id -> location
        # For per-user model override parity with app
        self.user_models: Dict[str, Dict[str, str]] = {}

        # Encrypted persistence
        self._store_file: Optional[Path] = None
        self._fernet: Optional[Fernet] = None
        if store_path and encryption_key:
            self._store_file = Path(store_path) / "history.enc"
            self._store_file.parent.mkdir(parents=True, exist_ok=True)
            self._fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
            self._load()

    @property
    def messages(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """The raw nested history mapping.

        Exposed primarily for inspection and tests; mutating it bypasses
        trimming and persistence.

        Returns:
            The ``{room: {user: [message, ...]}}`` mapping backing the store.
        """
        return self._messages

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose mode for system prompt suffix.

        When verbose is True, omit the extra suffix in the system prompt.

        Args:
            verbose: Verbosity flag.
        """
        self._include_extra = not bool(verbose)

    def _full_suffix(self) -> str:
        """Compose the prompt suffix, including the extra clause when active.

        The extra clause (typically a brevity instruction) is appended unless
        verbose mode has been enabled via :meth:`set_verbose`.

        Returns:
            The suffix string to append after the personality.
        """
        return f"{self.prompt_suffix}{self.prompt_suffix_extra if self._include_extra and self.prompt_suffix_extra else ''}"

    def _location_suffix(self, user: str) -> str:
        """Build the location sentence appended to a user's system prompt.

        Args:
            user: Matrix user ID to look up.

        Returns:
            A leading-space location note, or an empty string when the user
            has no stored location.
        """
        loc = self._locations.get(user)
        if loc:
            return f" The user is located in {loc}."
        return ""

    def _system_for(self, room: str, user: str) -> str:
        """Compose the system prompt for a room/user thread.

        Uses the fixed system prompt when one was configured, otherwise builds
        it from prefix + personality + suffix; either way the user's location
        note is appended.

        Args:
            room: Matrix room ID (reserved for future per-room prompts).
            user: Matrix user ID, used for the location note.

        Returns:
            The full system prompt string for the thread.
        """
        if self._fixed_system_prompt is not None:
            return self._fixed_system_prompt + self._location_suffix(user)
        return f"{self.prompt_prefix}{self.personality}{self._full_suffix()}{self._location_suffix(user)}"

    def _ensure(self, room: str, user: str) -> None:
        """Create a thread seeded with a system message if absent.

        Idempotent: existing threads are left untouched.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
        """
        if room not in self._messages:
            self._messages[room] = {}
        if user not in self._messages[room]:
            self._messages[room][user] = [{"role": "system", "content": self._system_for(room, user)}]

    def init_prompt(self, room: str, user: str, persona: Optional[str] = None, custom: Optional[str] = None) -> None:
        """Initialize or replace the system prompt for a thread.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
            persona: Optional persona to apply using configured prefix/suffix.
            custom: Optional custom system prompt string to use instead.
        """
        self._ensure(room, user)
        loc = self._location_suffix(user)
        if custom:
            self._messages[room][user] = [{"role": "system", "content": custom + loc}]
        else:
            p = persona if (persona is not None and persona != "") else self.personality
            self._messages[room][user] = [
                {"role": "system", "content": f"{self.prompt_prefix}{p}{self._full_suffix()}{loc}"}
            ]
        self._save()

    def add(self, room: str, user: str, role: str, content: str) -> None:
        """Append a message and trim to max history length.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
            role: Message role ("user"/"assistant"/"system").
            content: Message content.
        """
        self._ensure(room, user)
        self._messages[room][user].append({"role": role, "content": content})
        self._trim(room, user)
        self._save()

    def get(self, room: str, user: str) -> List[Dict[str, str]]:
        """Return a shallow copy of a thread's messages.

        Seeds the thread with its system message first if it does not yet
        exist. The copy is safe to pass to generation without mutating store
        state.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.

        Returns:
            A new list of the thread's message dicts.
        """
        self._ensure(room, user)
        return list(self._messages[room][user])

    def reset(self, room: str, user: str, stock: bool = False) -> None:
        """Reset a user's history for a room.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
            stock: When True, leave history empty; otherwise seed default prompt.
        """
        if room not in self._messages:
            self._messages[room] = {}
        self._messages[room][user] = []
        if not stock:
            self.init_prompt(room, user, persona=self.personality)
        self._save()

    # alias used by our handlers
    def clear(self, room: str, user: str) -> None:
        """Clear a thread's history without re-seeding a prompt.

        Convenience alias for ``reset(room, user, stock=True)`` to match the
        naming used by the command handlers.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
        """
        self.reset(room, user, stock=True)

    def clear_all(self) -> None:
        """Clear all histories across rooms and users.

        Locations are preserved since they are user preferences, not
        conversation state.
        """
        self._messages.clear()
        self._save()

    @staticmethod
    def count_tokens(msgs: List[Dict[str, str]]) -> int:
        """Estimate the token count of a message list.

        Uses a cheap ~4-characters-per-token heuristic rather than a real
        tokenizer, which is sufficient for budgeting history length.

        Args:
            msgs: The messages to estimate.

        Returns:
            The approximate total token count.
        """
        return sum(len(m.get("content", "")) for m in msgs) // 4

    def _trim(self, room: str, user: str) -> None:
        """Drop oldest messages until a thread fits the token budget.

        Preserves the leading system message, removing the next-oldest
        messages first; stops if only the system message remains.

        Args:
            room: Matrix room ID.
            user: Matrix user ID.
        """
        msgs = self._messages[room][user]
        while self.count_tokens(msgs) > self.max_tokens:
            if msgs and msgs[0].get("role") == "system":
                if len(msgs) > 1:
                    msgs.pop(1)
                else:
                    break
            else:
                msgs.pop(0)

    # -- User locations --------------------------------------------------------

    def set_location(self, user: str, location: str) -> None:
        """Set or clear the location for a user.

        When set, the location is appended to system prompts across all rooms.
        Existing threads are updated to reflect the new location.

        Args:
            user: Matrix user ID.
            location: Location string, or empty to clear.
        """
        if location:
            self._locations[user] = location
        else:
            self._locations.pop(user, None)
        # Update system prompts in all existing threads for this user
        for room in self._messages:
            if user in self._messages[room]:
                msgs = self._messages[room][user]
                if msgs and msgs[0].get("role") == "system":
                    # Rebuild the system prompt with the new location
                    old_content = msgs[0]["content"]
                    # Strip any existing location suffix
                    marker = " The user is located in "
                    idx = old_content.find(marker)
                    base = old_content[:idx] if idx != -1 else old_content
                    msgs[0]["content"] = base + self._location_suffix(user)
        self._save()

    def get_location(self, user: str) -> Optional[str]:
        """Return a user's stored location.

        Args:
            user: Matrix user ID to look up.

        Returns:
            The stored location string, or ``None`` if the user has none.
        """
        return self._locations.get(user)

    # -- Encrypted persistence -------------------------------------------------

    def _save(self) -> None:
        """Persist messages and locations to the encrypted store file.

        No-op when persistence is not configured (no Fernet key or store
        path). Failures are logged rather than raised so a write error never
        breaks message handling.
        """
        if not self._fernet or not self._store_file:
            return
        try:
            payload = {"messages": self._messages, "locations": self._locations}
            data = json.dumps(payload, separators=(",", ":")).encode()
            self._store_file.write_bytes(self._fernet.encrypt(data))
        except Exception:
            logger.exception("Failed to save encrypted history")

    def _load(self) -> None:
        """Restore messages and locations from the encrypted store file.

        No-op when persistence is unconfigured or the file is absent. Supports
        both the legacy bare-messages format and the current
        ``{messages, locations}`` format. Failures are logged, not raised.
        """
        if not self._fernet or not self._store_file or not self._store_file.exists():
            return
        try:
            encrypted = self._store_file.read_bytes()
            data = self._fernet.decrypt(encrypted)
            parsed = json.loads(data)
            # Support both old format (bare messages dict) and new format
            if isinstance(parsed, dict) and "messages" in parsed:
                self._messages = parsed["messages"]
                self._locations = parsed.get("locations", {})
            else:
                self._messages = parsed
        except InvalidToken:
            logger.error(
                "Failed to decrypt history file — wrong key or corrupted data. "
                "Starting with empty history."
            )
        except Exception:
            logger.exception("Failed to load encrypted history")
