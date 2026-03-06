from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import List, Optional

from .logging_conf import setup_logging
from .config import load_config
from .app import run as run_app


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the bot.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(prog="infinigpt-matrix", description="InfiniGPT Matrix bot (modular)", add_help=True)
    parser.add_argument("-L", "--log-level", default=os.getenv("INFINIGPT_LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("-e", "--env-file", default=os.getenv("INFINIGPT_ENV_FILE", ".env"), help="Path to env file (default: ./.env)")
    parser.add_argument("-E", "--e2e", action="store_true", help="Enable end-to-end encryption (overrides config)")
    parser.add_argument("-N", "--no-e2e", action="store_true", help="Disable end-to-end encryption (overrides config)")
    parser.add_argument("-m", "--model", help="Override default model")
    parser.add_argument("-s", "--store-path", help="Override store path")
    parser.add_argument("-S", "--server-models", action="store_true", help="Refresh the model list from the configured provider on startup")
    parser.add_argument("-v", "--verbose", dest="verbose_mode", action="store_true", help="Enable verbose mode (omit brevity clause)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for running the Matrix bot.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 for success).
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    os.environ["INFINIGPT_ENV_FILE"] = args.env_file or ".env"
    cfg = load_config(args.env_file)
    if args.model:
        try:
            cfg.llm.default_model = args.model
        except Exception:
            pass
    if args.store_path:
        try:
            cfg.matrix.store_path = args.store_path
        except Exception:
            pass
    if args.server_models:
        try:
            cfg.llm.server_models = True
        except Exception:
            pass
    if args.e2e:
        try:
            cfg.matrix.e2e = True
        except Exception:
            pass
    if args.no_e2e:
        try:
            cfg.matrix.e2e = False
        except Exception:
            pass
    logging.getLogger(__name__).info(
        "Loaded config. OpenAI models: %s; xAI models: %s; LM Studio models: %s; Default model: %s",
        ", ".join(cfg.llm.models.get("openai", [])),
        ", ".join(cfg.llm.models.get("xai", [])),
        ", ".join(cfg.llm.models.get("lmstudio", [])),
        cfg.llm.default_model,
    )
    asyncio.run(run_app(cfg))
    return 0
