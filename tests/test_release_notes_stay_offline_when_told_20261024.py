"""``SPACR_NO_NETWORK`` keeps the News refresh off the network.

Without the check the refresh opened a socket in every test window, and
closing that window waited for the name lookup (3.5 s here, longer on
Windows) inside the embedded-overview budget.
"""
from __future__ import annotations

import urllib.request

from spacr import updater


def test_no_socket_is_opened_when_the_network_is_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_NO_NETWORK", "1")
    monkeypatch.setattr(updater, "news_cache_path", lambda: tmp_path / "news.json")

    def refuse(*_args, **_kwargs):
        raise AssertionError("opened a socket with SPACR_NO_NETWORK set")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert updater.fetch_release_notes() == []


def test_a_cleared_switch_still_asks_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_NO_NETWORK", "0")
    monkeypatch.setattr(updater, "news_cache_path", lambda: tmp_path / "news.json")
    asked = []

    def answer(*_args, **_kwargs):
        asked.append(True)
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    assert updater.fetch_release_notes() == [] and asked
