"""Package entry point so ``python -m agent_smithers`` launches the CLI."""
from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

