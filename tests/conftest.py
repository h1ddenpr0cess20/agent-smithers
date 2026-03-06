import os
import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setenv("INFINIGPT_ENV_FILE", os.environ.get("INFINIGPT_ENV_FILE", ".env"))
    yield
