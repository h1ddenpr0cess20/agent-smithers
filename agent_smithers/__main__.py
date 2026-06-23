"""Package execution entry point.

Allows the bot to be started with ``python -m agent_smithers`` by delegating
to the console-script :func:`agent_smithers.cli.main`.
"""
from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

