import asyncio
from types import SimpleNamespace

from agent_smithers.security import Security


class FakeMatrix:
    def __init__(self):
        self.client = SimpleNamespace()


def test_security_allows_devices_noop():
    sec = Security(FakeMatrix())
    asyncio.run(sec.allow_devices("@u:example.org"))
