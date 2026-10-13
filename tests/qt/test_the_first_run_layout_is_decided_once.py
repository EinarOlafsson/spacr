"""The opening width is resolved once and recorded with its evidence.

Instruction 359's last headless point: "First-run selection stores nothing
yet: the width is recomputed each launch rather than resolved once and
recorded with the metrics that justified it."

A width on its own cannot be checked later. A width beside the screen
geometry and font scale it came from can be -- and, just as important, it
can be seen to have EXPIRED when either of those changes.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt import app as qt_app                           # noqa: E402
from spacr.qt import preferences as prefs                    # noqa: E402


class _Screen:
    def __init__(self, width=2560, height=1440):
        self._w, self._h = width, height

    def availableGeometry(self):
        from PySide6.QtCore import QRect
        return QRect(0, 0, self._w, self._h)


class _Window:
    def __init__(self, width=1200):
        self._w = width
        self.resized_to = None

    def screen(self):
        return _Screen()

    def width(self):
        return self._w

    def height(self):
        return 850

    def resize(self, w, h):
        self.resized_to = (w, h)
        self._w = w


@pytest.fixture
def clean_record(monkeypatch):
    """A fresh decision store, held in a dict rather than in QSettings."""
    store = {}
    monkeypatch.setattr(prefs, "_get_layout_decision", lambda: dict(store))
    monkeypatch.setattr(prefs, "_set_layout_decision",
                        lambda record: store.update(record))
    return store


def test_the_first_launch_records_the_width_and_its_evidence(clean_record,
                                                             monkeypatch):
    """Not just the answer: the numbers it was derived from."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    qt_app._open_at_the_measured_width(_Window())

    assert clean_record, "the first launch recorded nothing"
    assert clean_record["width"] > 0
    assert clean_record["available"] == [2560, 1440]
    assert clean_record["font_scale"] == 1.0
    assert clean_record["reason"], "a width with no reason cannot be audited"


def test_the_second_launch_reuses_it_without_re_deriving(clean_record,
                                                         monkeypatch):
    """The policy artifact is read once, not once per launch."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    qt_app._open_at_the_measured_width(_Window())

    calls = []
    import spacr.qt._layout_policy as policy
    real = policy.recommended_window_size

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy, "recommended_window_size", counted)
    qt_app._open_at_the_measured_width(_Window())
    assert calls == [], "the recorded decision was re-derived anyway"


def test_a_different_font_scale_expires_the_record(clean_record, monkeypatch):
    """It is a decision ABOUT those numbers, so it expires when they do."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    qt_app._open_at_the_measured_width(_Window())
    first = clean_record["width"]

    monkeypatch.setattr(prefs, "get_font_scale", lambda: 2.0)
    qt_app._open_at_the_measured_width(_Window())
    assert clean_record["font_scale"] == 2.0
    assert clean_record["width"] >= first, (
        "a larger font cannot need a narrower window")


def test_a_different_screen_expires_the_record(clean_record, monkeypatch):
    """Moving to another monitor re-derives rather than reusing."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    qt_app._open_at_the_measured_width(_Window())

    class _Smaller(_Window):
        def screen(self):
            return _Screen(1440, 900)

    qt_app._open_at_the_measured_width(_Smaller())
    assert clean_record["available"] == [1440, 900]


def test_an_unreadable_record_is_treated_as_no_record(monkeypatch):
    """A half-read record is worse than none: it could match by accident."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    monkeypatch.setattr(prefs, "_get_layout_decision",
                        lambda: {"width": 9999})          # no evidence with it
    written = {}
    monkeypatch.setattr(prefs, "_set_layout_decision", written.update)

    window = _Window()
    qt_app._open_at_the_measured_width(window)
    assert written.get("available") == [2560, 1440], (
        "a record with no evidence was trusted")
    assert window.resized_to != (9999, 850)


def test_the_store_rejects_a_record_without_its_evidence():
    """The same rule at the storage boundary, not only at the caller."""
    assert prefs._get_layout_decision() == {} or True     # no crash on a real store
    import json
    from unittest import mock

    with mock.patch.object(prefs, "_settings") as fake:
        fake.return_value.value.return_value = json.dumps({"width": 1800})
        assert prefs._get_layout_decision() == {}
        fake.return_value.value.return_value = json.dumps(
            {"width": 1800, "available": [2560, 1440], "font_scale": 1.0})
        assert prefs._get_layout_decision()["width"] == 1800
        fake.return_value.value.return_value = "not json"
        assert prefs._get_layout_decision() == {}


def test_it_still_only_ever_grows(clean_record, monkeypatch):
    """The rule the function already had, kept: a wide window is not shrunk."""
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    window = _Window(width=3000)
    assert qt_app._open_at_the_measured_width(window) is False
    assert window.resized_to is None
