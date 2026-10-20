import socket
import urllib.request
import pytest

@pytest.fixture(autouse=True)
def refuse_live_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("CI N702 diagnostic attempted live network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
