from agent_smithers.handlers.router import Router


def test_router_dispatch_ai():
    r = Router()
    called = {}

    async def h(ctx, room, sender, display, args):
        called["ok"] = (room, sender, display, args)

    r.register(".ai", h)
    fn, args = r.dispatch(object(), "!r", "@u", "User", ".ai hello world", False, bot_name="Bot")
    assert fn is h
    assert args[-1] == "hello world"


def test_router_dispatch_botname():
    r = Router()
    async def h(ctx, room, sender, display, args):
        pass
    r.register(".ai", h)
    fn, args = r.dispatch(object(), "!r", "@u", "User", "Bot: hi", False, bot_name="Bot")
    assert fn is h
    assert args[-1] == "hi"


def test_router_dispatch_empty_text():
    r = Router()
    async def h(ctx, room, sender, display, args):
        pass
    r.register(".ai", h)
    fn, args = r.dispatch(object(), "!r", "@u", "User", "", False)
    assert fn is None
    assert args == tuple()


def test_router_dispatch_unknown_command():
    r = Router()
    async def h(ctx, room, sender, display, args):
        pass
    r.register(".ai", h)
    fn, args = r.dispatch(object(), "!r", "@u", "User", ".unknown hello", False)
    assert fn is None
    assert args == tuple()


def test_router_dispatch_admin_command_denied_for_non_admin():
    r = Router()
    async def admin_h(ctx, room, sender, display, args):
        pass
    r.register(".model", admin_h, admin=True)
    fn, args = r.dispatch(object(), "!r", "@u", "User", ".model gpt-4o", False)
    assert fn is None


def test_router_dispatch_admin_command_allowed_for_admin():
    r = Router()
    async def admin_h(ctx, room, sender, display, args):
        pass
    r.register(".model", admin_h, admin=True)
    fn, args = r.dispatch(object(), "!r", "@u", "Admin", ".model gpt-4o", True)
    assert fn is admin_h
    assert args[-1] == "gpt-4o"


def test_router_dispatch_botname_without_ai_registered():
    r = Router()
    fn, args = r.dispatch(object(), "!r", "@u", "User", "Bot: hi", False, bot_name="Bot")
    assert fn is None


def test_router_dispatch_whitespace_only():
    r = Router()
    fn, args = r.dispatch(object(), "!r", "@u", "User", "   ", False)
    assert fn is None
    assert args == tuple()


def test_router_dispatch_regular_takes_priority_over_admin():
    """When a command is registered as both regular and admin, regular wins."""
    r = Router()
    async def regular(ctx, room, sender, display, args):
        pass
    async def admin(ctx, room, sender, display, args):
        pass
    r.register(".test", regular)
    r.register(".test", admin, admin=True)
    # Non-admin gets regular
    fn, _ = r.dispatch(object(), "!r", "@u", "User", ".test foo", False)
    assert fn is regular
    # Admin also gets regular (it's checked first)
    fn, _ = r.dispatch(object(), "!r", "@u", "Admin", ".test foo", True)
    assert fn is regular
