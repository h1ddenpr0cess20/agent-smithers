import os
import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    env_file = os.environ.get("AGENT_SMITHERS_ENV_FILE", os.environ.get("INFINIGPT_ENV_FILE", ".env"))
    monkeypatch.setenv("AGENT_SMITHERS_ENV_FILE", env_file)
    monkeypatch.setenv("INFINIGPT_ENV_FILE", env_file)
    yield
