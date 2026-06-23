"""Public façade for the application.

Re-exports the two stable entry points other code (and tests) depend on:
:class:`~agent_smithers.context.AppContext`, the service container, and
:func:`~agent_smithers.runtime.run`, the async runtime loop.
"""
from __future__ import annotations

from .context import AppContext
from .runtime import run

__all__ = ["AppContext", "run"]
