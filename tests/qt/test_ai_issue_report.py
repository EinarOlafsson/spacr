"""Tests for spacr.qt.ai.issue_report — sanitization + URL builder."""
from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest


def test_sanitize_path_collapses_home(monkeypatch, tmp_path):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    text = f"traceback at {tmp_path}/scratch/foo.py line 3"
    out = ir.sanitize_path(text)
    assert str(tmp_path) not in out
    assert "~/scratch/foo.py" in out


def test_sanitize_path_redacts_db_paths():
    from spacr.qt.ai import issue_report as ir
    text = "opening /var/lib/exp/patient_ABC.db failed"
    assert "patient_ABC.db" not in ir.sanitize_path(text)
    assert "<DB>" in ir.sanitize_path(text)


def test_sanitize_settings_recurses_into_lists(monkeypatch, tmp_path):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    settings = {
        "src":  f"{tmp_path}/plates/plate1",
        "srcs": [f"{tmp_path}/a", f"{tmp_path}/b", 42],
        "n":    5,
    }
    out = ir.sanitize_settings(settings)
    assert out["src"].startswith("~/")
    assert all(
        (isinstance(s, str) and s.startswith("~/")) or s == 42
        for s in out["srcs"]
    )
    assert out["n"] == 5


def test_traceback_hash_is_deterministic():
    from spacr.qt.ai import issue_report as ir
    tb = (
        'Traceback (most recent call last):\n'
        '  File "spacr/core.py", line 42, in preprocess\n'
        '    x = 1 / 0\n'
        'ZeroDivisionError: division by zero'
    )
    h1 = ir._traceback_hash(tb)
    h2 = ir._traceback_hash(tb)
    assert h1 == h2
    assert len(h1) == 6


def test_build_report_contains_all_sections(monkeypatch, tmp_path):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    tb = 'ZeroDivisionError: division by zero'
    r = ir.build_report(tb, active_app="mask",
                          settings={"src": f"{tmp_path}/foo"})
    assert "auto" in r["title"]
    assert "[mask]" in r["title"]
    assert "ZeroDivisionError" in r["title"]
    body = r["body"]
    assert "### Traceback" in body
    assert "### Environment" in body
    assert "Pipeline settings" in body
    assert "spaCR" in body


def test_build_report_omits_settings_when_none():
    from spacr.qt.ai import issue_report as ir
    r = ir.build_report("SomeError: boom", active_app="")
    assert "Pipeline settings" not in r["body"]


def test_issue_url_is_valid_github_url():
    from spacr.qt.ai import issue_report as ir
    url = ir.issue_url("test title", "hello **world**")
    assert url.startswith("https://github.com/EinarOlafsson/spacr/issues/new?")
    p = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(p.query)
    assert q["title"] == ["test title"]
    assert q["body"] == ["hello **world**"]
    assert q["labels"] == ["auto-filed"]


def test_issue_url_truncates_when_too_long():
    from spacr.qt.ai import issue_report as ir
    big_body = "x" * 20000
    url = ir.issue_url("short title", big_body)
    assert len(url) <= 8000
    # Decode + confirm truncation notice is present
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert "truncated" in q["body"][0]


def test_file_issue_returns_url_without_opening(monkeypatch):
    from spacr.qt.ai import issue_report as ir
    called = {}
    monkeypatch.setattr(
        ir, "open_issue_in_browser",
        lambda url: called.setdefault("url", url) is None,
    )
    url = ir.file_issue("Error: boom", active_app="mask")
    assert url.startswith("https://github.com/EinarOlafsson/spacr/issues/new?")
    assert called["url"] == url
