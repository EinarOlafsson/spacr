"""Two refusals on the Outliers screen that a finished scan cannot reach.

Both are ``if frame is None: return``, and both sit downstream of the code
that guarantees a frame. Neither is dead weight -- each is one line
standing between a background job and an AttributeError on the GUI thread
-- but neither can fire while the producing side stays as it is written,
so this file pins the producing side. If someone gives ``set_frame`` a
None default, or lets ``object_frame`` return None, these tests fail
rather than the guard quietly coming alive.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.outliers import OutliersScreen
from spacr.qt.widgets import outlier_model

pytestmark = pytest.mark.qt


def _plate(n_wells: int = 6, per_well: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for well in range(n_wells):
        factor = 1.6 if well == 2 else 1.0
        area = factor * rng.lognormal(0.0, 0.2, per_well)
        perimeter = rng.lognormal(0.0, 0.2, per_well)
        for i in range(per_well):
            rows.append(("p1", "r1", f"c{well + 1}", "f1", i,
                         area[i], perimeter[i]))
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label",
        "cell_area", "cell_perimeter"])


@pytest.fixture
def screen(qtbot):
    widget = OutliersScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class TestTheFrameCannotVanishMidJob:
    """``_on_scanned`` re-reads ``self._frame`` after the worker returns."""

    def test_a_finished_scan_reaches_both_tables(self, screen):
        """The live branch: the frame is there, so the result is drawn."""
        screen.set_frame(_plate())
        screen.scan()

        assert screen.result is not None
        assert screen.objects_frame() is not None
        assert screen.report.toPlainText().strip(), (
            "a finished scan wrote no report")

    def test_nothing_ever_puts_none_back_into_the_frame(self):
        """THE PIN.

        ``self._frame`` is set to None once, in ``__init__``, before any
        scan can be submitted. Every other assignment is ``set_frame``
        storing its own required argument. So a scan that got past
        ``scan()``'s refusal cannot find None waiting for it when the
        worker comes back.

        The failure this catches is a future ``set_frame(frame=None)``
        to mean "unload": that would make the guard live, and it would
        need a test of its own rather than this pin.
        """
        source = inspect.getsource(OutliersScreen)
        tree = ast.parse("class _X:\n" + "\n".join(
            "    " + line for line in source.splitlines()[1:]))

        assigning_functions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and target.attr == "_frame"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    assigning_functions.append(ast.unparse(node.value))

        assert assigning_functions, "no assignment to self._frame found"
        assert set(assigning_functions) <= {"None", "frame"}, (
            f"self._frame is assigned from {assigning_functions}")

        signature = inspect.signature(OutliersScreen.set_frame)
        assert signature.parameters["frame"].default is inspect.Parameter.empty, (
            "set_frame's frame argument gained a default, so a caller can "
            "now unload the table -- the guard in _on_scanned is live and "
            "needs its own test")

    def test_a_scan_with_no_table_refuses_before_submitting(self, screen):
        """The producing side: nothing is submitted without a frame."""
        submitted = []
        screen._jobs.submit = lambda *a, **k: submitted.append(a)

        screen.scan()
        assert submitted == [], "a scan was submitted with no table loaded"
        assert "Load a table" in screen._source.text()


class TestTheObjectFrameIsSetImmediatelyBefore:
    """``_fill_object_table`` reads ``self._objects``, assigned one line up."""

    def test_object_frame_returns_a_frame_for_every_result(self, screen):
        """THE PIN, from the producing side.

        ``_on_scanned`` assigns ``self._objects = result.object_frame(frame)``
        and calls ``_fill_object_table`` on the next line. The guard there
        can only fire if ``object_frame`` returns None, so that is what is
        asserted -- against the engine, not against the screen.
        """
        frame = _plate()
        screen.set_frame(frame)
        screen.scan()

        result = screen.result
        rebuilt = result.object_frame(frame)
        assert isinstance(rebuilt, pd.DataFrame), (
            "object_frame no longer returns a frame")
        assert len(rebuilt) == len(frame)

        annotation = inspect.signature(
            outlier_model.OutlierResult.object_frame).return_annotation
        assert "Optional" not in str(annotation), (
            "object_frame may now return None, so _fill_object_table's "
            "guard is live and needs its own test")

    def test_a_failed_job_clears_the_object_frame_and_the_export(self, screen):
        """Why the field is Optional at all.

        It is None between construction and the first result, and again
        after a failure -- but neither of those states runs the fill,
        because the fill is only ever called from a *success*.
        """
        screen.set_frame(_plate())
        screen.scan()
        assert screen._export.isEnabled()

        screen._on_job_failed("the table could not be read")
        assert screen.objects_frame() is None
        assert screen._export.isEnabled() is False
        assert "could not be read" in screen.report.toPlainText()
