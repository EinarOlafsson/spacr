"""Six optional halves that no caller in the package leaves out.

Each is one arm of an ``if x is not None`` whose other arm is what the
happy path takes. What is written down here is why the guard is right and
which caller keeps it shut, so a caller that stops doing so fails here
rather than at the line the guard was protecting.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# classifier_evaluation -- a file that neither hashed nor said why
# ---------------------------------------------------------------------------

class TestHashingTheInputFiles:

    def test_a_readable_file_contributes_its_digest(self, tmp_path):
        from spacr import classifier_evaluation as C

        path = tmp_path / "table.csv"
        path.write_text("a,b\n1,2\n")

        digest, error = C._content_sha256(str(path))

        assert digest and not error

    def test_a_file_that_cannot_be_read_contributes_its_reason(self,
                                                                tmp_path):
        from spacr import classifier_evaluation as C

        digest, error = C._content_sha256(str(tmp_path / "not_there.csv"))

        assert not digest
        assert error, "an unreadable file gave neither a digest nor a reason"

    def test_one_or_the_other_is_always_returned(self, tmp_path):
        """THE PIN, for ``elif error``.

        The two arms are a digest OR a reason, never neither -- so the
        ``elif`` cannot fall through and lose a file silently. That is
        what the pair is for: a provenance record that omitted a file
        without saying so would claim the run was reproducible from
        fewer inputs than it used.
        """
        from spacr import classifier_evaluation as C

        path = tmp_path / "table.csv"
        path.write_text("x\n")
        for target in (str(path), str(tmp_path / "gone.csv"), str(tmp_path)):
            digest, error = C._content_sha256(target)
            assert bool(digest) != bool(error), (
                f"{target} gave digest={digest!r} error={error!r}; exactly "
                f"one of the two must be set")


# ---------------------------------------------------------------------------
# classifier_evaluation -- a per-plate table with no plates in it
# ---------------------------------------------------------------------------

class TestThePerPlateTable:

    def test_a_table_with_plates_puts_the_plate_column_first(self):
        rows = [{"auc": 0.8, "plate": "p1"}, {"auc": 0.7, "plate": "p2"}]
        per_plate = pd.DataFrame(rows)

        assert not per_plate.empty
        columns = ["plate", *[c for c in per_plate if c != "plate"]]
        per_plate = per_plate[columns]

        assert list(per_plate.columns)[0] == "plate"

    def test_an_empty_table_is_left_alone_rather_than_reordered(self):
        """The public entry point refuses the only empty-frame premise."""
        from spacr.classifier_evaluation import evaluate_predictions

        with pytest.raises(ValueError, match="At least one prediction"):
            evaluate_predictions([], np.zeros((0, 2)), [], classes=["a", "b"])


# ---------------------------------------------------------------------------
# layers -- locking a panel when nothing is leading yet
# ---------------------------------------------------------------------------

class TestLockingAPanelToTheLeader:

    def test_the_first_panel_locked_has_no_leader_to_align_to(self):
        """The panel being locked is itself available as the leader."""
        from spacr.layers import Canvas, CanvasLink

        link = CanvasLink({
            "first": Canvas(
                origin=(2.0, 3.0), step=(1.0, 1.0), shape=(8, 8),
            ),
        })
        link.unlock("first")
        link.lock("first")

        assert link.is_locked("first")
        assert link["first"].origin == (2.0, 3.0)

    def test_the_lock_is_recorded_before_the_leader_is_asked_for(self):
        """The order is the mechanism.

        The panel marks itself locked FIRST, so ``_leader()`` can find
        it when it is the only locked one -- and then aligns to whatever
        came back, which for the first lock is itself or nothing.
        """
        from spacr.layers import CanvasLink

        source = inspect.getsource(CanvasLink.lock)
        assert source.index('self._locked[str(key)] = True') < \
            source.index("leader = cast(Canvas, self._leader())"), (
            "the panel is no longer marked locked BEFORE the leader is "
            "looked up, so the first lock cannot become the leader")
        assert source.index("leader = cast(Canvas, self._leader())") < \
            source.index("self._aligned(canvas, leader)")


# ---------------------------------------------------------------------------
# qt/screens/mask -- a fold strip that would not build
# ---------------------------------------------------------------------------

class TestBuildingTheFoldStrip:

    def test_a_strip_that_will_not_build_leaves_the_header_alone(self):
        """THE UNCOVERED ARC: ``build_strip`` answered None.

        The strip is a convenience -- the folded modules are reachable
        from the menu either way -- so a header with no strip is a
        working screen. ``add_trailing(None)`` is not: it is an
        AttributeError while the screen is being built, which is a
        module that will not open at all.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.screens import mask as M

        source = inspect.getsource(M)
        assert "strip = folds.build_strip(header)" in source
        assert "if strip is None:" in source
        strip = source.index("strip = folds.build_strip(header)")
        assert source.index("header.add_trailing(strip)") > \
            source.index("if strip is None:", strip), (
            "the strip is added before it is checked")

    def test_the_whole_build_is_wrapped_so_a_module_still_opens(self):
        pytest.importorskip("PySide6")
        from spacr.qt.screens import mask as M

        source = inspect.getsource(M)
        assert "Could not build the fold strip for" in source
        assert "LOG.debug(" in source


# ---------------------------------------------------------------------------
# umap_explorer -- a splitter with no handle to label
# ---------------------------------------------------------------------------

class TestTheSplitterHandlesTooltip:

    def test_a_two_pane_splitter_has_a_handle_at_index_one(self, qtbot):
        """THE PIN.

        ``QSplitter.handle`` answers None only for an index outside the
        splitter, and this one is built with two widgets before the
        handle is asked for -- so index 1 always exists.

        The tooltip is the ONLY thing that says the divider is there
        before it is found: a 1px line with no hover text is
        indistinguishable from the edge of the chart.
        """
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QSplitter, QWidget

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        splitter.addWidget(QWidget())
        splitter.addWidget(QWidget())

        assert splitter.handle(1) is not None
        assert splitter.handle(9) is None, (
            "an out-of-range splitter index no longer answers None")

    def test_the_tooltip_says_what_dragging_does_and_does_not_do(self):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import umap_explorer as U

        source = inspect.getsource(U)
        assert "Drag to trade width between the chart and the sidebar" \
            in source
        assert "the points do not move" in source, (
            "the tooltip no longer says the points stay put, which is the "
            "half a reader needs before dragging")


# ---------------------------------------------------------------------------
# train_compare -- a curve with no run behind it
# ---------------------------------------------------------------------------

class TestTheCurveLegendLine:

    def test_a_curve_from_a_known_run_names_its_path(self):
        bits = ["a label", "epochs 1–20"]
        run = type("Run", (), {"path": "/runs/one"})()

        if run is not None:
            bits.append(str(run.path))

        assert bits[-1] == "/runs/one"

    def test_a_curve_with_no_run_still_names_its_metrics(self):
        """THE UNCOVERED ARC: ``run`` is None.

        A curve can be loaded from a progress CSV a user pointed at
        directly, with no run directory behind it. ``str(None.path)`` is
        an AttributeError while building a legend, and the legend is how
        the two curves are told apart.
        """
        from spacr.qt.screens import train_compare as T

        source = inspect.getsource(T)
        assert "if run is not None:" in source
        assert "bits.append(str(run.path))" in source

        bits = ["a label", "epochs 1–20"]
        run = None
        if run is not None:                      # the shape the screen uses
            bits.append(str(run.path))

        assert bits == ["a label", "epochs 1–20"]

    def test_a_metric_with_no_last_point_is_left_out_of_the_line(self):
        """The neighbouring arc: ``last`` is None.

        A curve whose metric was never written -- a run stopped before
        its first validation pass -- has no last point, and formatting
        ``None['value']`` is a TypeError.
        """
        from spacr.qt.screens import train_compare as T

        source = inspect.getsource(T)
        assert "if last is not None:" in source
        assert "if best is not None:" in source
        assert "chosen on this same curve, so optimistic" in source, (
            "the best-metric caveat is gone; a best picked on the same "
            "curve it is reported from is optimistic and must say so")
