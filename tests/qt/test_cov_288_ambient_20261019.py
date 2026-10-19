"""Item 288: ambient backdrop edges that the ratchet found unexercised.

An unwritable numba cache folder means no cache; a registry entry whose
validity check raises is skipped; a spaceout fractal that cannot report its
parent is skipped; a non-backdrop widget in the live registry is left alone
when the animation policy applies its decision.
"""
from __future__ import annotations

import os
import weakref

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget

from spacr.qt.widgets import ambient


def test_unwritable_numba_cache_folder_means_no_cache(tmp_path, monkeypatch):
    folder = tmp_path / "numba_cache"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(folder))
    monkeypatch.setattr(os, "access", lambda *_a, **_k: False)
    assert ambient._numba_cache_directory() is None
    assert folder.is_dir()
    assert os.environ["NUMBA_CACHE_DIR"] == str(folder)


def test_still_alive_skips_entries_whose_validity_check_raises(
        qapp, monkeypatch):
    good = QWidget()
    bad = QWidget()
    import shiboken6

    real = shiboken6.isValid

    def picky(obj):
        if obj is bad:
            raise TypeError("cannot tell")
        return real(obj)

    monkeypatch.setattr(shiboken6, "isValid", picky)
    assert ambient._still_alive([good, bad]) == [good]


def test_spaceout_fractal_that_cannot_report_its_parent_is_skipped(
        qapp, monkeypatch):
    class Odd(QWidget):
        _spaceout_built_from = object()

        def parentWidget(self):
            raise RuntimeError("parent unavailable")

    class Fine(QWidget):
        _spaceout_built_from = object()

    holder = QWidget()
    odd = Odd(holder)
    fine = Fine(holder)
    registry = weakref.WeakSet([odd, fine])
    monkeypatch.setattr(ambient, "_LIVE_SPACEOUT_FRACTALS", registry)
    assert ambient._live_spaceout_fractals() == [fine]


def test_policy_apply_leaves_a_non_backdrop_widget_alone(qapp, monkeypatch):
    stranger = QWidget()
    monkeypatch.setattr(ambient, "_LIVE_AMBIENT_WIDGETS",
                        weakref.WeakSet([stranger]))
    policy = ambient._AnimationResourcePolicy()
    seen = []
    policy.changed.connect(lambda: seen.append(True))
    policy._apply()
    assert seen == [True]
    assert not hasattr(stranger, "_sync_run_state")
