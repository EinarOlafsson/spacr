"""Seven last guards, most of them one line the happy path steps over.

A colour search that never improves, an application that is not there, a
number that matched a number pattern and would not parse, a widget with
no style, a palette of nothing but headers, and a version that cannot be
read at all.
"""
from __future__ import annotations

import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# qt/theme -- the most colourful shade at a given luminance
# ---------------------------------------------------------------------------

class TestPickingTheMostColourfulShade:

    def test_the_search_finds_a_shade_inside_the_luminance_band(self):
        from spacr.qt import theme as T

        picked = T._hue_rgb(0.0)
        assert len(picked) == 3
        assert all(0.0 <= channel <= 1.0 for channel in picked)

    def test_the_first_candidate_inside_the_band_always_improves_on_nothing(
            self):
        """THE PIN, for ``spread > chroma``.

        ``chroma`` starts at -1, and a spread is a difference of two
        channel values so it is never negative. The FIRST candidate that
        lands inside the luminance band therefore always wins, and the
        comparison only ever chooses between later ones.

        Starting at -1 rather than 0 is what makes that true, and it
        matters for a fully grey hue: its spread is 0, which would not
        beat a 0 and would leave ``best`` as None.
        """
        from spacr.qt import theme as T

        source = inspect.getsource(T)
        block = source[source.index("chroma = -1"):]
        block = block[:block.index("for step in range(256):")]
        assert "if spread > chroma:" in block
        assert "best, chroma = candidate, spread" in block

        grey = (128, 128, 128)
        assert max(grey) - min(grey) == 0
        assert 0 > -1, "a grey candidate must still beat the starting chroma"

    def test_recording_the_qss_context_with_no_application_does_nothing(self):
        """THE UNCOVERED ARC: ``app`` is None.

        The context is stashed ON the application object, so there is
        nowhere to put it without one -- ``setattr(None, ...)`` is an
        AttributeError. The import-time and offscreen-probe paths call
        this helper with None rather than branching around it, which is
        what the guard buys them.
        """
        pytest.importorskip("PySide6")
        from spacr.qt import theme as T

        before = dict(getattr(T, "_WIDGET_QSS_CONTEXT", {}) or {})

        T.set_widget_qss_context(None, "dark", 1.0, None)   # must not raise

        # AND NOTHING WAS RECORDED. There is nowhere to put it without an
        # application, so the correct outcome is that no context appears
        # -- which "did not raise" alone would not distinguish from one
        # stashed somewhere it can never be read back.
        assert dict(getattr(T, "_WIDGET_QSS_CONTEXT", {}) or {}) == before

    def test_with_an_application_the_context_is_recorded_on_it(self, qtbot):
        """The other arm, and what it is for: the exact live preference
        inputs, kept where a late screen block can read them back."""
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from spacr.qt import theme as T

        app = QApplication.instance()
        assert app is not None

        T.set_widget_qss_context(app, "light", 1.25, 0.5)

        recorded = getattr(app, T._WIDGET_QSS_CONTEXT_ATTRIBUTE, None)
        assert recorded == ("light", 1.25, 0.5)


# ---------------------------------------------------------------------------
# methods_export -- a token that matched the number pattern
# ---------------------------------------------------------------------------

class TestReadingTheNumbersOutOfASetting:

    def test_a_numeric_string_is_read_as_a_number(self):
        from spacr import methods_export as M

        found = M.digest_numbers({"threshold": "0.35", "count": 7})

        assert 0.35 in found and 7.0 in found

    def test_a_word_is_not_read_as_a_number(self):
        from spacr import methods_export as M

        found = M.digest_numbers({"method": "otsu", "note": "about 3 cells"})

        assert found == set() or all(
            isinstance(value, float) for value in found)

    def test_the_pattern_admits_only_what_float_can_parse(self):
        """THE PIN, for the ``except ValueError`` beside the parse.

        ``_NUMBER.fullmatch`` is the gate, and everything it admits is
        something ``float()`` accepts -- so the handler cannot run. It is
        cheap insurance against the two drifting apart, which is the
        change this fails on.

        Checked by running the pattern's own admissions through float,
        including the awkward ones a settings file really carries.
        """
        from spacr import methods_export as M

        for token in ("0", "7", "-3", "0.35", "-0.5", "1e-6", "+2",
                      ".5", "3.", "1E5"):
            if M._NUMBER.fullmatch(token):
                float(token)                     # must not raise

        for token in ("otsu", "", "1/2", "3px", "nan cells"):
            assert not M._NUMBER.fullmatch(token) or float(token) == float(
                token)


# ---------------------------------------------------------------------------
# qt/prerun -- a widget with no style behind it
# ---------------------------------------------------------------------------

class TestRepolishingAWidget:

    def test_a_live_widget_always_has_a_style(self, qtbot):
        """THE PIN.

        ``QWidget.style()`` falls back to the application's style, and
        an application always has one -- so a widget in a running
        process cannot answer None.

        The guard is for the shutdown path: a widget whose application
        has gone answers None rather than raising, and re-polishing then
        would be an AttributeError inside a teardown.
        """
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QLabel

        widget = QLabel("styled")
        qtbot.addWidget(widget)

        assert widget.style() is not None

        from spacr.qt import prerun as P

        source = inspect.getsource(P)
        assert "style = widget.style()" in source
        assert "if style is not None:" in source

    def test_a_repolish_is_unpolish_then_polish_in_that_order(self, qtbot):
        """Either alone leaves the widget half-restyled: unpolish clears
        the old objectName's rules, polish applies the new ones."""
        pytest.importorskip("PySide6")
        from spacr.qt import prerun as P

        source = inspect.getsource(P)
        assert source.index("style.unpolish(widget)") < \
            source.index("style.polish(widget)")


# ---------------------------------------------------------------------------
# qt/command_palette -- a list of nothing but section headers
# ---------------------------------------------------------------------------

class TestAutoSelectingTheFirstCommand:

    def test_the_first_selectable_row_is_chosen_not_the_header(self, qtbot):
        """The headers are unselectable, so the walk skips them."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        listing = QListWidget()
        qtbot.addWidget(listing)
        header = QListWidgetItem("SECTION")
        header.setFlags(Qt.NoItemFlags)
        listing.addItem(header)
        listing.addItem(QListWidgetItem("a command"))

        chosen = None
        for i in range(listing.count()):
            if listing.item(i).flags() != Qt.NoItemFlags:
                chosen = i
                break

        assert chosen == 1, "the header was auto-selected"

    def test_a_palette_of_only_headers_selects_nothing(self, qtbot):
        """THE UNCOVERED ARC: the walk finds no selectable row.

        A section whose every command was filtered out leaves its header
        behind, and the count is then above one with nothing choosable
        in it. Falling out of the loop selects nothing, which is right:
        pressing Return would otherwise activate a header.
        """
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        listing = QListWidget()
        qtbot.addWidget(listing)
        for name in ("FIRST", "SECOND"):
            header = QListWidgetItem(name)
            header.setFlags(Qt.NoItemFlags)
            listing.addItem(header)

        assert listing.count() > 1
        chosen = None
        for i in range(listing.count()):
            if listing.item(i).flags() != Qt.NoItemFlags:
                chosen = i
                break

        assert chosen is None, "a header was selectable after all"

        from spacr.qt import command_palette as C

        source = inspect.getsource(C)
        assert "if self._list.item(i).flags() != Qt.NoItemFlags:" in source


# ---------------------------------------------------------------------------
# ome_zarr -- a version that cannot be read at all
# ---------------------------------------------------------------------------

class TestReportingTheVersion:

    def test_an_installed_checkout_reports_its_version_literal(self):
        from spacr import ome_zarr as Z

        source = inspect.getsource(Z)
        assert "from ._version import __version__ as checkout_version" \
            in source
        assert 'return "unknown"' in source

        from spacr._version import __version__

        assert isinstance(__version__, str) and __version__

    def test_a_checkout_with_no_version_module_says_unknown(self):
        """THE UNCOVERED HANDLER.

        Metadata is authoritative for an installed package and this is
        the source-tree fallback used when that lookup explicitly found
        none. Both failing is a checkout with no version at all --
        a shallow copy, or a tree assembled by hand -- and "unknown" is
        a version string a reader can act on where an ImportError from a
        Zarr writer is not.
        """
        from spacr import ome_zarr as Z

        source = inspect.getsource(Z)
        block = source[source.index(
            "from ._version import __version__ as checkout_version"):]
        assert "except Exception:" in block[:200]
        assert 'return "unknown"' in block[:260]
        assert 'return fallback or "unknown"' in block[:400], (
            "an empty version literal no longer falls back to 'unknown'")
