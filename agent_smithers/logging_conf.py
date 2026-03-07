import logging
import logging.config
import re
from typing import Any, Optional


_RICH_CONSOLE: Any = None
_STATUS_CONTENT_PADDING = 20


class _NoopStatus:
    """Fallback status object when Rich status output is unavailable."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __enter__(self) -> "_NoopStatus":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    def update(
        self,
        status: Optional[str] = None,
        *,
        spinner: Optional[str] = None,
        spinner_style: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        del spinner, spinner_style, speed
        if status:
            self.message = status


class _RichStatus:
    """Rich-backed spinner aligned with the log message content column."""

    def __init__(self, console: Any, message: str, *, spinner: str) -> None:
        from rich.live import Live
        from rich.padding import Padding
        from rich.status import Status

        self._status = Status(message, spinner=spinner)
        self._padding = Padding
        self._live = Live(
            self._renderable(),
            console=console,
            transient=True,
            refresh_per_second=12,
        )

    def _renderable(self) -> Any:
        return self._padding(self._status, (0, 0, 0, _STATUS_CONTENT_PADDING))

    def __enter__(self) -> "_RichStatus":
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._live.stop()
        del exc_type, exc, tb
        return False

    def update(
        self,
        status: Optional[str] = None,
        *,
        spinner: Optional[str] = None,
        spinner_style: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        self._status.update(
            status=status,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
        )
        self._live.update(self._renderable(), refresh=True)


class MatrixHighlighter:
    """Rich highlighter for Matrix-specific patterns in log lines.

    Highlights user IDs, room IDs, model names, tool calls, and common
    bot log phrases to make logs easier to scan in the terminal.
    """
    _user_re = re.compile(r"@[A-Za-z0-9_.\-:]+\b")
    _room_re = re.compile(r"[#!][^\s:]+:[A-Za-z0-9_.\-]+")
    _model_re = re.compile(r"\bModel set to\s+(?P<model>\S+)")
    _sent_msg_re = re.compile(r"\bsent\s+(?P<msg>.+?)\s+in\s+")
    _joined_re = re.compile(r"^(?P<bot>.+?)\s+joined\s+(?P<room>\S+)")
    _sent_line_re = re.compile(r"^(?P<display>.+?)\s+\((?P<id>@[^)]+)\)\s+sent\s+(?P<msg>.+?)\s+in\s+(?P<room>\S+)", re.S)
    _sending_resp_re = re.compile(r"Sending response to\s+(?P<name>.+?)\s+in\s+(?P<room>\S+):\s+(?P<body>.*)", re.S)
    _thinking_re = re.compile(r"Model thinking for\s+(?P<who>.+?):\s+(?P<thinking>.*)", re.S)
    _sys_prompt_re = re.compile(r"System prompt for\s+(?P<who>.+?)\s+\(.*?\)\s+set to\s+'(?P<prompt>.*)'")
    _verified_re = re.compile(r"\bverified device\s+(?P<dev>\S+)")
    _persist_re = re.compile(r"\bPersisted device_id to\s+(?P<path>\S+)")
    _tool_call_re = re.compile(r"(?P<tool>Tool)\s+\((?P<origin>MCP|builtin)\):\s+(?P<name>\S+)\s+args=(?P<args>.*)")

    def __call__(self, value):
        """Return a Rich Text instance with highlights applied.

        Args:
            value: The original log record message (str or Text-like).

        Returns:
            A Rich ``Text`` instance with style spans applied when Rich is
            available; otherwise returns the original value.
        """
        try:
            from rich.text import Text
        except Exception:
            return value
        text = value if hasattr(value, "stylize") else Text(str(value))
        self.highlight(text)
        return text

    def highlight(self, text) -> None:
        """Apply in-place style spans to a Rich ``Text`` instance.

        Args:
            text: A Rich ``Text`` instance to stylize.
        """
        s = text.plain
        for m in self._user_re.finditer(s):
            text.stylize("bold cyan", m.start(), m.end())
        for m in self._room_re.finditer(s):
            text.stylize("magenta", m.start(), m.end())
        for m in self._model_re.finditer(s):
            span = m.span("model")
            text.stylize("bold yellow", span[0], span[1])
        for m in self._sent_msg_re.finditer(s):
            span = m.span("msg")
            text.stylize("white", span[0], span[1])
        for m in self._sent_line_re.finditer(s):
            dspan = m.span("display")
            rsp = m.span("room")
            text.stylize("bold cyan", dspan[0], dspan[1])
            text.stylize("magenta", rsp[0], rsp[1])
        for m in self._joined_re.finditer(s):
            bspan = m.span("bot")
            rsp = m.span("room")
            text.stylize("bold cyan", bspan[0], bspan[1])
            text.stylize("magenta", rsp[0], rsp[1])
        for m in self._sending_resp_re.finditer(s):
            nspan = m.span("name")
            text.stylize("bold cyan", nspan[0], nspan[1])
            bsp = m.span("body")
            body_text = s[bsp[0]:bsp[1]]
            nl = body_text.find("\n")
            if nl >= 0:
                text.stylize("bold", bsp[0] + nl + 1, bsp[1])
            else:
                text.stylize("bold", bsp[0], bsp[1])
        for m in self._thinking_re.finditer(s):
            wsp = m.span("who")
            tsp = m.span("thinking")
            text.stylize("bold cyan", wsp[0], wsp[1])
            text.stylize("dim italic", tsp[0], tsp[1])
        for m in self._sys_prompt_re.finditer(s):
            wsp = m.span("who")
            text.stylize("bold cyan", wsp[0], wsp[1])
        for m in self._verified_re.finditer(s):
            span = m.span("dev")
            text.stylize("green", span[0], span[1])
        for m in self._persist_re.finditer(s):
            span = m.span("path")
            text.stylize("green", span[0], span[1])
        for m in self._tool_call_re.finditer(s):
            tsp = m.span("tool")
            osp = m.span("origin")
            nsp = m.span("name")
            asp = m.span("args")
            text.stylize("bold cyan", tsp[0], tsp[1])
            text.stylize("cyan", osp[0], osp[1])
            text.stylize("bold yellow", nsp[0], nsp[1])
            text.stylize("dim", asp[0], asp[1])


def setup_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure package logging with Rich fallback.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO").
        json: When True, use a simpler formatter suitable for JSON-like
            ingestion; otherwise use a human-friendly rich console.
    """
    global _RICH_CONSOLE

    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})
    try:
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.traceback import install as rich_traceback_install

        rich_traceback_install(show_locals=False)
        console = Console(highlight=False, force_terminal=True)
        _RICH_CONSOLE = console
        highlighter = MatrixHighlighter()
        handler = RichHandler(console=console, rich_tracebacks=True, markup=True, show_level=True, show_time=True, show_path=False, highlighter=highlighter)
        datefmt = "[%X]"
        fmt = "%(message)s" if not json else "%(name)s - %(message)s"
        root = logging.getLogger(); root.handlers = []; root.setLevel(logging.ERROR)
        pkg_logger = logging.getLogger("agent_smithers"); pkg_logger.handlers = []; pkg_logger.setLevel(lvl)
        logging.Formatter(fmt=fmt, datefmt=datefmt)
        pkg_logger.addHandler(handler); pkg_logger.propagate = False
    except Exception:
        _RICH_CONSOLE = None
        fmt = ("%(asctime)s %(levelname)s %(name)s %(message)s" if json else "%(asctime)s - %(levelname)s - %(message)s")
        root = logging.getLogger(); root.handlers = []; root.setLevel(logging.ERROR)
        pkg_logger = logging.getLogger("agent_smithers"); pkg_logger.handlers = []; pkg_logger.setLevel(lvl)
        handler = logging.StreamHandler(); handler.setFormatter(logging.Formatter(fmt))
        pkg_logger.addHandler(handler); pkg_logger.propagate = False


def configure_logging(level: int = logging.INFO) -> None:
    """Helper to configure logging from a numeric level.

    Args:
        level: Numeric logging level from the ``logging`` module.
    """
    setup_logging(logging.getLevelName(level))


def spinner_status(
    message: str,
    *,
    spinner: str = "dots",
    enabled: bool = True,
):
    """Return a Rich status spinner when available, otherwise a no-op context manager."""
    if not enabled or _RICH_CONSOLE is None:
        return _NoopStatus(message)
    return _RichStatus(_RICH_CONSOLE, message, spinner=spinner)
