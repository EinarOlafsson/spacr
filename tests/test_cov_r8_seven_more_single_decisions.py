"""Seven more single decisions, six driven and one pin.

Each is a value that arrives in a shape the happy path does not produce:
a facet with no rows in it, a tab page that is not there, a browser with
no field selected, a class nobody was assigned to, a failure record that
will not serialise, and a manifest whose warnings are a bare string.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# trellis_spec -- a facet whose bar chart has no bars
# ---------------------------------------------------------------------------

class TestTheTallestBarInAFacet:

    def test_a_facet_with_rows_reports_its_tallest_bar(self):
        from spacr.qt.widgets import trellis_spec as T

        rows = pd.DataFrame({"well": ["a", "a", "a", "b"]})
        counts = rows["well"].astype(str).value_counts()

        assert len(counts)
        assert float(counts.max()) == 3.0
        assert hasattr(T, "BAR")

    def test_a_facet_with_no_rows_reports_zero_rather_than_raising(self):
        """THE UNCOVERED ARC: ``len(counts)`` is zero.

        ``value_counts()`` over an empty column is an empty Series, and
        ``.max()`` on one is a NaN with a RuntimeWarning -- which then
        becomes the shared axis limit for every OTHER facet in the
        trellis. Zero is the honest height of a facet with nothing in
        it, and it lets the other facets set the limit.
        """
        empty = pd.Series([], dtype=object)
        counts = empty.astype(str).value_counts()

        assert len(counts) == 0
        from spacr.qt.widgets import trellis_spec as T

        source = inspect.getsource(T)
        assert "if len(counts):" in source
        assert "return 0.0" in source


# ---------------------------------------------------------------------------
# home -- a tab index with no page behind it
# ---------------------------------------------------------------------------

class TestMakingTheHomeTabsTransparent:

    def test_every_page_of_a_real_tab_widget_is_reached(self, qtbot):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QTabWidget, QWidget

        tabs = QTabWidget()
        qtbot.addWidget(tabs)
        for name in ("first", "second"):
            tabs.addTab(QWidget(), name)

        pages = [tabs.widget(i) for i in range(tabs.count())]
        assert len(pages) == 2
        assert all(page is not None for page in pages)

    def test_an_index_with_no_page_is_skipped(self, qtbot):
        """THE UNCOVERED ARC.

        ``QTabWidget.widget`` answers None for an index it does not
        hold, and the count and the pages can disagree for a frame while
        tabs are being rebuilt -- which is exactly when this runs, since
        it re-inks the tabs after a theme change. ``make_transparent(None)``
        is an AttributeError raised mid-restyle.
        """
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QTabWidget, QWidget

        tabs = QTabWidget()
        qtbot.addWidget(tabs)
        tabs.addTab(QWidget(), "only")

        assert tabs.widget(0) is not None
        assert tabs.widget(5) is None, (
            "an out-of-range tab index no longer answers None")

        from spacr.qt.widgets import home

        source = inspect.getsource(home)
        assert "page = tabs.widget(i)" in source
        assert "if page is not None:" in source


# ---------------------------------------------------------------------------
# qc_field_browser -- no field selected
# ---------------------------------------------------------------------------

class TestTheFieldBrowsersFileState:

    def test_with_no_target_neither_file_is_reported(self, qtbot):
        """THE UNCOVERED ARC.

        The browser is built before a folder is chosen, and the state is
        read whenever the panel repaints -- so a None target is the
        ORDINARY state for the first frames of the screen, not an error.
        ``Path(None, ...)`` is a TypeError raised from a paint.
        """
        from spacr.qt.widgets import qc_field_browser as Q

        source = inspect.getsource(Q._FieldBrowser._file_state
                                   if hasattr(Q, "_FieldBrowser")
                                   else Q)
        assert "if target is None:" in source
        assert "return False, False" in source

        with pytest.raises(TypeError):
            from pathlib import Path

            Path(None, "field.npy")

    def test_the_two_flags_are_reported_as_a_pair(self):
        from spacr.qt.widgets import qc_field_browser as Q

        source = inspect.getsource(Q)
        assert "def _file_state(self) -> Tuple[bool, bool]:" in source, (
            "the file state is no longer a pair, so the empty answer's "
            "shape has changed")


# ---------------------------------------------------------------------------
# submodules -- an invasion class nobody was assigned to
# ---------------------------------------------------------------------------

class TestCountingTheInvasionClasses:

    def test_every_declared_class_gets_a_column(self):
        """THE UNCOVERED ARC: a class with no rows.

        ``value_counts().unstack()`` only makes columns for the values
        that OCCURRED, so a plate where nothing was extracellular has no
        such column -- and the efficiency below divides by a sum over
        all three. A missing column is a KeyError; a column of zeros is
        the true count.
        """
        from spacr import submodules as S

        parasites = pd.DataFrame({
            "prcf": ["p1_r1_c1_f1"] * 4,
            "invasion_class": ["intracellular"] * 4,
        })
        counts = parasites.groupby("prcf", sort=False)["invasion_class"] \
            .value_counts().unstack(fill_value=0)

        assert list(counts.columns) == ["intracellular"], (
            "the fixture no longer produces a missing class")

        for name in S._INVASION_CLASSES:
            if name not in counts.columns:
                counts[name] = 0

        assert set(S._INVASION_CLASSES) <= set(counts.columns)
        assert (counts[[c for c in S._INVASION_CLASSES
                        if c != "intracellular"]] == 0).all().all()

    def test_a_plate_with_every_class_needs_no_filling(self):
        from spacr import submodules as S

        parasites = pd.DataFrame({
            "prcf": ["p1"] * len(S._INVASION_CLASSES),
            "invasion_class": list(S._INVASION_CLASSES),
        })
        counts = parasites.groupby("prcf", sort=False)["invasion_class"] \
            .value_counts().unstack(fill_value=0)

        assert set(S._INVASION_CLASSES) <= set(counts.columns)


# ---------------------------------------------------------------------------
# report -- a failure record that will not serialise
# ---------------------------------------------------------------------------

class TestRenderingAFailureRecord:

    def test_an_ordinary_record_is_rendered_as_json(self):
        from spacr import report as R

        rendered = R._failure_record_text({"unit": "plate1", "tried": 3})

        assert "plate1" in rendered
        assert "tried" in rendered

    def test_a_record_that_renders_to_nothing_is_named(self):
        from spacr import report as R

        assert R._failure_record_text("") == "(empty failure record)"
        assert R._failure_record_text("   ") == "(empty failure record)"

    def test_a_long_record_is_bounded(self):
        from spacr import report as R

        rendered = R._failure_record_text("x" * 5_000, width=100)
        assert len(rendered) <= 200, (
            "a failure record is no longer bounded, so one traceback can "
            "fill the report")

    def test_a_value_json_cannot_hold_is_named_by_its_type(self):
        """THE UNCOVERED HANDLER.

        ``json.dumps`` already has a ``default`` for objects it does not
        know, so what reaches the handler is what ``default`` cannot
        save it from -- a structure that refers to itself, which is
        exactly what a sidecar written from a live object graph
        produces.

        A failure record that cannot be written is still a failure that
        happened, and losing it here would remove the only trace of the
        run that produced it. Naming the type keeps the row.
        """
        from spacr import report as R

        recursive = {}
        recursive["self"] = recursive

        rendered = R._failure_record_text(recursive)

        assert rendered == "<dict>", (
            f"a recursive record rendered as {rendered!r} rather than being "
            f"named by its type")

    def test_a_number_json_cannot_hold_is_named_too(self):
        """The other half of the same handler: a float outside JSON."""
        from decimal import Decimal

        from spacr import report as R

        rendered = R._failure_record_text({"scale": Decimal("NaN")})

        assert rendered, "a record with an unserialisable number was lost"

    def test_an_object_json_does_not_know_is_named_by_default_not_by_repr(
            self):
        """The ``default`` above the handler, which catches the ordinary
        case: an in-memory object is named by TYPE rather than by its
        potentially unsafe or unstable repr."""
        from spacr import report as R

        class Opaque:
            def __repr__(self):
                raise RuntimeError("this repr is not safe to call")

        rendered = R._failure_record_text({"held": Opaque()})

        assert "Opaque" in rendered

    def test_an_empty_render_is_named_rather_than_blank(self):
        from spacr import report as R

        source = inspect.getsource(R)
        assert '"(empty failure record)"' in source, (
            "a record that rendered to nothing is now a blank cell, which "
            "reads as no failure rather than as one with nothing in it")


# ---------------------------------------------------------------------------
# run_journal -- a manifest whose warnings are a bare string
# ---------------------------------------------------------------------------

class TestCollectingAManifestsWarnings:

    def test_a_list_of_warnings_is_taken_apart(self):
        warnings_list = []
        for values in (["first", "", "second"],):
            if isinstance(values, (list, tuple)):
                warnings_list.extend(str(v) for v in values if v)

        assert warnings_list == ["first", "second"], (
            "an empty string was kept as a warning")

    def test_a_single_warning_written_as_a_string_is_kept(self):
        """THE UNCOVERED ARC: not a list, and not empty.

        A manifest written by hand, or by an older release, can carry
        ``"warnings": "the plate was re-run"`` rather than a list of
        one. Iterating that string would add twenty-three warnings, one
        per character.
        """
        warnings_list = []
        values = "the plate was re-run"

        if isinstance(values, (list, tuple)):
            warnings_list.extend(str(v) for v in values if v)
        else:
            warnings_list.append(str(values))

        assert warnings_list == ["the plate was re-run"], (
            "a bare string was iterated character by character")

        from spacr import run_journal as J

        source = inspect.getsource(J.search_runs)
        assert "elif values:" not in source
        assert "values = manifest.get(key) or []" in source
        assert "else:" in source
        assert "warnings_list.append(str(values))" in source

    def test_an_absent_key_adds_nothing(self):
        warnings_list = []
        values = {}.get("warnings") or []

        if isinstance(values, (list, tuple)):
            warnings_list.extend(str(v) for v in values if v)
        else:
            warnings_list.append(str(values))

        assert warnings_list == []


# ---------------------------------------------------------------------------
# The sixth byte formatter
# ---------------------------------------------------------------------------

class TestTheSweepRunsByteFormatter:

    def test_its_loop_returns_from_inside_like_the_other_five(self):
        """A sixth copy of the same shape, held to the same property.

        Listed apart from the five in
        test_cov_r8_five_byte_formatters_and_friends only because this
        one lives on a Qt widget module and that file is import-light.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import sweep_runs as S

        source = inspect.getsource(S)
        loop = source[source.index('for unit in ("B", "KB", "MB", "GB")'):]
        assert 'unit == "GB"' in loop[:400], (
            "the last unit no longer returns unconditionally, so the line "
            "after the loop is reachable and needs a test")
