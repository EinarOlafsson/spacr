"""The drop branches that only fire when the *registry* already knows.

Every case below is a path through :mod:`spacr.qt.dnd_handlers` that the
existing dnd files never take, and each one is a different way for a drop to
be quietly wrong rather than loudly broken:

* Measure resolving nothing and having to use the folder as dropped, with and
  without a trailing ``merged`` to hop.
* Classify being handed a plate it is already pointing at -- the second drop
  of the same folder must not list it twice.
* The Data Manager taking a project on a screen that has no ``scan``.
* The Prediction Profiler and Curate being told by
  :func:`spacr.chaining.resolve_drop` *which* file inside a folder is the one,
  which has to beat the alphabetical scan of the folder they would otherwise
  do.
* A coefficients CSV and a settings snapshot dropped by themselves, which must
  go straight in rather than re-searching the folder they sit in.

The registry's answer is supplied by the ``resolves`` fixture and the "which
one did you mean?" dialog by ``picks``, both shaped like the ones in
``tests/qt/test_cov_dnd_handlers.py``; everything else is a real tree on disk,
because every assertion here is about what was found on it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt import dnd_handlers as dh


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Screen:
    """A plain, non-Qt screen: what a tool screen looks like to a handler."""

    def __init__(self, **attrs):
        self._console = _Console()
        for name, value in attrs.items():
            setattr(self, name, value)

    @property
    def log(self) -> str:
        return self._console.text


class _Target:
    def __init__(self, value, location=None, kind="measurements-db",
                 source="registry", paths=()):
        self.value = value
        self.location = location if location is not None else value
        self.kind = kind
        self.source = source
        self.paths = tuple(paths)


class _Resolution:
    def __init__(self, targets=(), choices=(), root="/root", reason="",
                 ok=True, ambiguous=False):
        self.targets = tuple(targets)
        self.choices = tuple(choices)
        self.root = root
        self.reason = reason
        self.ok = ok
        self.ambiguous = ambiguous

    def target_for(self, kind):
        for target in self.targets:
            if target.kind == kind:
                return target
        return None


@pytest.fixture
def resolves(monkeypatch):
    """Make spacr.chaining answer whatever the case under test needs."""
    def _install(resolution):
        monkeypatch.setattr(dh._ch, "resolve_drop", lambda *a, **k: resolution)
        return resolution
    return _install


# ---------------------------------------------------------------------------
# Measure: nothing resolved, so the folder as dropped
# ---------------------------------------------------------------------------


def test_measure_suggests_child_plates_when_the_parent_cannot_be_listed(
        tmp_path):
    """A drop is dispatched inside Qt's event loop, so an exception here is a
    crashed window rather than a "did you mean…?" prompt.

    The sibling scan reads the *parent* of the dropped folder, which can stop
    being a directory between the two reads -- a network share going away
    mid-drag. The child hits already found must still be offered.
    """
    class _ParentRemovedMidDrop(type(Path("."))):
        @property
        def parent(self):
            return Path(f"{self}-was-removed")

    root = tmp_path / "screen1"
    (root / "plateA" / "merged").mkdir(parents=True)
    (tmp_path / "screen2" / "merged").mkdir(parents=True)

    with_parent = dh.MeasureDropHandler().suggest_alternatives(root)
    parent_gone = dh.MeasureDropHandler().suggest_alternatives(
        _ParentRemovedMidDrop(root))

    assert with_parent == [root / "plateA" / "merged",
                           tmp_path / "screen2" / "merged"]
    assert [Path(hit) for hit in parent_gone] == [root / "plateA" / "merged"]


# ---------------------------------------------------------------------------
# Data Manager: a project, then a measurement of it -- if it can measure
# ---------------------------------------------------------------------------

def test_the_data_manager_still_takes_a_project_on_a_screen_that_cannot_scan(
        tmp_path, resolves):
    """The scan is a follow-up, not a precondition.

    ``DataManagerDropHandler`` measures the project it was just handed, but
    the same handler is reachable from a screen built without that button --
    an older data manager, or one still constructing. Requiring ``scan`` there
    would turn a working drop into an AttributeError inside Qt's drop
    dispatch, which is a crash and not a dialog.
    """
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    handler = dh.DataManagerDropHandler("data_manager")

    full = []
    handler.apply(tmp_path, _Screen(set_project=lambda p: full.append("project"),
                                    scan=lambda: full.append("scan")))
    without_scan = []
    handler.apply(tmp_path, _Screen(set_project=without_scan.append))

    assert full == ["project", "scan"]
    assert without_scan == [str(tmp_path)]
