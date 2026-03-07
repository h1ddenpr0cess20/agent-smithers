from __future__ import annotations

from typing import Optional

import markdown


MATRIX_MARKDOWN_EXTENSIONS = [
    "extra",
    "fenced_code",
    "nl2br",
    "sane_lists",
    "tables",
]


def render_markdown(body: str) -> Optional[str]:
    """Render Markdown to Matrix-safe HTML.

    Avoid ``codehilite`` here. It emits ``div``/``span``-heavy HTML that Matrix
    clients frequently sanitize in ways that break fenced code blocks.
    """
    try:
        return markdown.markdown(body, extensions=MATRIX_MARKDOWN_EXTENSIONS)  # type: ignore[arg-type]
    except Exception:
        return None
