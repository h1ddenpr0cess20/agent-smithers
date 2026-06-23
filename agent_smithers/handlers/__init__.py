"""Command handlers and the message router for the Matrix bot.

Each ``cmd_*`` module implements one chat command; :class:`Router` maps
incoming message prefixes and bot mentions to those handlers.
"""
from .router import Router

