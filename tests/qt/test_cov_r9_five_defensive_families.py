"""Five defensive handlers whose premise nothing else checks.

Each is marked ``pragma: no cover`` with a reason, and the reason is a
claim about something OUTSIDE the function -- Qt's clipboard, numpy's
singular values, a hard dependency, an ordering two methods apart. A
comment cannot notice when one of those stops holding; these can.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestThePcaRankGuards:

    def test_a_matrix_with_variance_has_a_positive_largest_singular_value(
            self):
        """THE PIN, for ``largest <= 0``.

        Constant columns are removed before this point, so the matrix
        that reaches the decomposition has variance somewhere -- and a
        largest singular value of zero means none at all. Asked of numpy
        over the shapes the screen produces.
        """
        rng = np.random.default_rng(0)
        for rows, columns in ((5, 2), (20, 4), (3, 3)):
            matrix = rng.normal(size=(rows, columns))
            singular = np.linalg.svd(matrix, compute_uv=False)
            assert float(singular.max()) > 0.0

    def test_an_all_constant_matrix_is_what_would_trip_it(self):
        """The half that says the guard is not nonsense: standardising a
        constant column gives zeros, and its singular values are zero
        too. The constants are removed EARLIER, which is the premise."""
        constant = np.zeros((5, 3))
        singular = np.linalg.svd(constant, compute_uv=False)

        assert float(singular.max()) == 0.0

    def test_rank_is_at_least_one_once_the_largest_is_positive(self):
        """THE PIN, for ``rank < 1``.

        The tolerance is ``largest * max(n, p) * eps`` and the largest
        singular value is compared against it -- so with ``largest > 0``
        at least that one exceeds its own epsilon-scaled tolerance.
        Driven, because it is an arithmetic claim about floats.
        """
        rng = np.random.default_rng(1)
        for rows, columns in ((5, 2), (20, 4), (2, 8), (50, 3)):
            singular = np.linalg.svd(rng.normal(size=(rows, columns)),
                                     compute_uv=False)
            largest = float(singular.max())
            assert largest > 0.0
            tolerance = largest * max(rows, columns) * float(np.finfo(float).eps)
            assert int((singular > tolerance).sum()) >= 1, (
                f"a {rows}x{columns} matrix with a positive largest singular "
                f"value came out rank 0, so the second guard is live")

    def test_both_refusals_say_what_the_matrix_was(self):
        from spacr.qt.widgets import pca_model as P

        source = _source(P)
        assert "no variance left to decompose" in source
        assert "has rank 0" in source
        assert source.index("if largest <= 0:") < source.index("if rank < 1:"), (
            "the rank check now runs first, so its 'implied by largest > 0' "
            "reason no longer holds")


class TestTheOutlierScanFrames:

    def test_the_frame_is_set_before_the_scan_is_launched(self):
        """THE PIN, for the first ``if frame is None``.

        A scan cannot be started without a frame, and the callback runs
        on the GUI thread -- so the frame cannot vanish between the two.
        Held by ORDER, which is the only thing that could change.
        """
        from spacr.qt.screens import outliers as O

        source = _source(O)
        handler = source.index("def _on_scanned(self, result) -> None:")
        assert "GUI thread only" in source[handler:handler + 200], (
            "the callback is no longer documented as GUI-thread only, so the "
            "frame really could change under it")
        assert "frame = self._frame" in source[handler:handler + 300]

    def test_the_object_frame_is_assigned_before_the_table_is_filled(self):
        """THE PIN, for the second: ``_objects`` is set on the line
        before ``_fill_object_table`` is called, so it cannot be None
        inside it."""
        from spacr.qt.screens import outliers as O

        source = _source(O)
        assign = source.index("self._objects = result.object_frame(frame)")
        call = source.index("self._fill_object_table(result)", assign)

        assert assign < call, (
            "the object table is filled before the frame it reads is "
            "assigned, so its None check is live")


class TestTheMontageGraphPanel:

    def test_the_panel_is_ensured_before_it_is_indexed(self):
        """THE PIN, for ``self._graph_panel is None``.

        ``_ensure_graph_tab`` creates it, so the line after cannot find
        it missing -- and ``indexOf(None)`` would be a type error rather
        than a missing tab.
        """
        from spacr.qt.widgets import cell_montage_view as C

        source = _source(C)
        ensure = source.index("self._ensure_graph_tab()")
        guard = source.index("if self._graph_panel is None:", ensure)
        index = source.index("self._tabs.indexOf(self._graph_panel)", guard)

        assert ensure < guard < index

    def test_nothing_picked_says_so_before_any_of_that(self):
        """The arm that DOES run, and the message a user actually meets:
        the comparison groups by what the picker chose, so with nothing
        picked there is nothing to group."""
        from spacr.qt.widgets import cell_montage_view as C

        source = _source(C)
        assert "the comparison groups them by what" in source
        assert "nothing is picked yet" in source


class TestTheSummaryClipboard:

    def test_qt_gives_an_application_a_clipboard(self, qtbot):
        """THE PIN, for ``clipboard is None``.

        ``QApplication.clipboard()`` answers None only without a running
        application -- and this is a widget method, so there is one.
        Asked of Qt.
        """
        from PySide6.QtWidgets import QApplication

        assert QApplication.instance() is not None
        assert QApplication.clipboard() is not None

    def test_an_empty_summary_is_not_copied_at_all(self):
        """The arm above it, and the reason it comes first: putting an
        empty string on the clipboard would silently destroy whatever the
        user had copied before."""
        from spacr.qt.widgets import folding_summary as F

        source = _source(F)
        empty = source.index("if not text.strip():")
        clipboard = source.index("clipboard = QApplication.clipboard()", empty)

        assert empty < clipboard, (
            "the clipboard is read before the empty check, so copying an "
            "empty summary now clears what the user had")


class TestTheDefaultPngMapping:

    def test_the_fallback_matches_what_crops_actually_declares(self):
        """THE PIN, for ``except Exception`` over a hard dependency.

        ``spacr.crops`` cannot fail to import, so the handler cannot run
        -- but its FALLBACK is a literal copy of the real default, and a
        copy that drifted would reintroduce exactly the preview/run
        disagreement the import exists to prevent.
        """
        from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING
        from spacr.qt.widgets import measure_preview as M

        source = _source(M)
        assert "from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING" in source

        fallback = {"r": 2, "g": 1, "b": 0}
        assert dict(DEFAULT_PNG_CHANNEL_MAPPING) == fallback, (
            f"the preview's fallback {fallback} no longer matches crops' "
            f"default {dict(DEFAULT_PNG_CHANNEL_MAPPING)}, so a run that hit "
            f"it would disagree with the pipeline")
        assert 'return {"r": 2, "g": 1, "b": 0}' in source

    def test_the_mapping_is_copied_rather_than_handed_out(self):
        """A caller that edited the returned dict would change the
        package-level default for every later preview."""
        from spacr.qt.widgets import measure_preview as M

        assert "dict(DEFAULT_PNG_CHANNEL_MAPPING)" in _source(M)

    def test_the_reason_for_importing_rather_than_copying_is_recorded(self):
        from spacr.qt.widgets import measure_preview as M

        source = _source(M)
        assert "how the preview and the run came to" in source
        assert "disagree in the first place" in source
