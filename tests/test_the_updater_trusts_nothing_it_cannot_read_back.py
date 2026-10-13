"""Item 288: the updater's news cache and version check, when they fail.

* A release link that appears twice in a release body is listed once.
* A news cache whose ``releases`` is not a list is not a cache: the next
  launch fetches again rather than showing nothing.
* A news cache that cannot be written (a read-only home) does not stop the
  news being shown; the failure is logged.
* A pinned upgrade whose installed version does not even parse as a
  version is reported as NOT verified, rather than raising or passing.
"""
from __future__ import annotations

import json
import logging
import time

from spacr import updater


def test_a_link_repeated_in_a_release_body_is_listed_once():
    body = ("See https://x.org/a. And again https://x.org/a, then "
            "https://x.org/b")
    assert updater._news_links(body) == ["https://x.org/a", "https://x.org/b"]


def test_a_cache_whose_releases_are_not_a_list_is_not_used(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv(updater.ENV_NEWS_CACHE, str(tmp_path))
    updater.news_cache_path().write_text(json.dumps(
        {"fetched": time.time(), "releases": {"tag": "v1"}}), encoding="utf-8")
    assert updater._read_news_cache(3600.0) is None


def test_a_cache_that_cannot_be_written_is_logged_not_raised(tmp_path,
                                                             monkeypatch,
                                                             caplog):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file where the cache folder should be")
    monkeypatch.setenv(updater.ENV_NEWS_CACHE, str(blocker))
    with caplog.at_level(logging.DEBUG, logger=updater.LOG.name):
        updater._write_news_cache([{"tag": "v1"}])
    assert "could not write the news cache" in caplog.text
    assert blocker.read_text() == "a file where the cache folder should be"


def test_an_unparseable_installed_version_is_not_verified(monkeypatch):
    calls = []

    def run(args, timeout=1800.0):
        calls.append(args)
        if len(calls) == 1:
            return 0, "Successfully installed spacr\n"
        return 0, "not-a-version\n"

    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "run_install_command", run)
    code, output = updater.run_pip_upgrade(target_version="2.0.0")
    assert code == 1
    assert "did not verify spaCR 2.0.0" in output
    assert "not-a-version" in output
    assert len(calls) == 2
