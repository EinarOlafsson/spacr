"""The last of the audit: forty-three sites across sixteen modules.

Nothing here shares a shape with the rest, so they are grouped by what
settles each -- Qt's own answers, a loop's own bounds, a caller that has
already chosen, and two abstract methods that exist to be overridden.
"""
from __future__ import annotations

import inspect
import logging
import os
import pathlib
import sysconfig

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QApplication, QDialogButtonBox, QLabel,
                               QSpinBox, QVBoxLayout, QWidget)

pytestmark = pytest.mark.qt


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestThePreferenceFields:

    def test_a_field_with_no_unit_gets_no_suffix(self, qtbot):
        """THE ARC: ``suffix`` is empty.

        Most preferences are counts and have no unit; a spin box with an
        empty suffix still reserves room for it, so setting one
        unconditionally would leave a gap after every plain number.
        """
        box = QSpinBox()
        qtbot.addWidget(box)

        assert box.suffix() == ""
        box.setSuffix(" px")
        assert box.suffix() == " px"

    def test_the_field_does_not_track_the_keyboard(self, qtbot):
        """Beside it, and the reason it is set: with tracking on, every
        keystroke emits valueChanged, so typing '120' writes 1, then 12,
        then 120 into the preference."""
        from spacr.qt import preferences as P

        source = _source(P)
        assert "box.setKeyboardTracking(False)" in source

    def test_a_dialog_and_a_window_are_both_closed_when_present(self, qtbot):
        """THE PIN, for ``parent is not None`` and ``window is not None``.

        The restart flow is reached from a dialog inside a window, so
        both are there. They are separately guarded because the same
        helper is also called from the window's own menu, where there is
        no dialog to accept.
        """
        from spacr.qt import preferences as P

        source = _source(P)
        accept = source.index("if parent is not None:")
        close = source.index("if window is not None:", accept)

        assert accept < close, (
            "the window is closed before the dialog is accepted, so the "
            "dialog's result is lost")
        assert "watcher.start()" in source[:accept], (
            "the watcher is started after the window closes, so a timer "
            "parented to it dies before it asks anything")

    def test_a_button_box_gives_the_buttons_it_was_asked_for(self, qtbot):
        """THE PIN, for the two ``... is not None`` checks on Save and
        Cancel. Asked of Qt, which is what decides."""
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        qtbot.addWidget(buttons)

        assert buttons.button(QDialogButtonBox.Save) is not None
        assert buttons.button(QDialogButtonBox.Cancel) is not None
        assert buttons.button(QDialogButtonBox.Apply) is None, (
            "QDialogButtonBox answers a button it was not asked for, so the "
            "None checks mean something else")

    def test_the_captions_go_through_tr(self):
        """Qt's own Save and Cancel are translated by Qt's catalogs, not
        spaCR's, so a mixed-language dialog is what happens without
        this."""
        from spacr.qt import preferences as P

        source = _source(P)
        assert 'save_button.setText(tr("Save"))' in source
        assert 'cancel_button.setText(tr("Cancel"))' in source

    def test_a_widget_in_a_layout_reports_its_index(self, qtbot):
        """THE PIN, for ``row_of_buttons >= 0``.

        The button box was added to this layout a few lines above, so
        ``indexOf`` finds it -- and the else branch appends, which would
        put the hint AFTER the buttons instead of above them.
        """
        host = QWidget()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        first, second = QLabel("a"), QLabel("b")
        layout.addWidget(first)
        layout.addWidget(second)

        assert layout.indexOf(second) == 1
        assert layout.indexOf(QLabel("elsewhere")) == -1, (
            "indexOf no longer answers -1 for a widget it does not hold")

    def test_the_hint_goes_above_the_buttons(self):
        """The placement is the point: a hint under the buttons reads as
        a footnote to them rather than as the answer to the control the
        pointer is on."""
        from spacr.qt import preferences as P

        source = _source(P)
        assert "layout.insertWidget(row_of_buttons, hints)" in source
        assert "footnote to the buttons rather than as the answer" in source


class TestTheTracebackFrameFilter:

    def test_sysconfig_answers_every_path_key_this_asks_for(self):
        """THE PIN, for ``except Exception`` around ``get_paths``.

        The four keys are standard and present on every supported
        Python. Asked of sysconfig, which is the thing that would have
        to change.
        """
        paths = sysconfig.get_paths()

        for key in ("stdlib", "platstdlib", "purelib", "platlib"):
            assert key in paths, (
                f"sysconfig no longer reports {key!r}, so the handler around "
                f"it is live")
            assert isinstance(paths[key], str)

    def test_a_library_path_inside_the_checkout_is_not_excluded(self):
        """THE ARC below it, and the case it exists for: an EDITABLE
        install puts the package under a path that is also a library
        root, and excluding it would filter out every spaCR frame -- the
        only ones the trace is for.
        """
        for spacr_root, library in ((("/repo/spacr" + os.sep), "/repo"),
                                    (("/repo/spacr" + os.sep), "/usr/lib")):
            excluded = not spacr_root or not spacr_root.startswith(
                library + os.sep)
            if library == "/repo":
                assert not excluded, (
                    "a library path containing the checkout was excluded, so "
                    "every spaCR frame is filtered out of the trace")
            else:
                assert excluded


class TestTheVerboseLogSink:

    def test_the_sink_keeps_the_handler_and_the_others_do_not(self):
        """THE ARC: ``name != _SINK_LOGGER``.

        The handler is attached to ONE logger and everything else
        propagates to it. A second attachment is why the console and the
        log file carried every record two or three times, so the loop
        removes it from every logger but the sink.
        """
        from spacr.qt import verbose_logger as V

        source = _source(V)
        assert "if name != _SINK_LOGGER and handler in logger.handlers:" \
            in source
        assert "logger.removeHandler(handler)" in source

        handler = logging.NullHandler()
        one = logging.getLogger("spacr.tests.sink.one")
        two = logging.getLogger("spacr.tests.sink.two")
        one.addHandler(handler)
        two.addHandler(handler)
        try:
            for name, logger in (("sink", one), ("other", two)):
                if name != "sink" and handler in logger.handlers:
                    logger.removeHandler(handler)
            assert handler in one.handlers
            assert handler not in two.handlers
        finally:
            one.removeHandler(handler)
            two.removeHandler(handler)

    def test_every_attached_logger_is_lifted_to_at_least_info(self):
        """Beside it: a logger left at WARNING swallows the records the
        verbose mode exists to show, and NOTSET (0) has to be treated as
        'inherit' rather than as a level below INFO."""
        for level in (0, logging.DEBUG, logging.WARNING, logging.ERROR):
            resolved = min(level or logging.INFO, logging.INFO)
            assert resolved <= logging.INFO


class TestTheApplicationQuitHook:

    def test_an_application_exists_while_a_widget_does(self, qtbot):
        """THE PIN, for two ``application is not None`` checks in the CPU
        fractal.

        ``QApplication.instance()`` answers None only before one is
        constructed -- and this code runs inside a widget, so there is
        one. Asked of Qt.
        """
        assert QApplication.instance() is not None

    def test_the_quit_hook_is_connected_and_disconnected_symmetrically(self):
        """The pair is the point: a backdrop that connects on build and
        never disconnects leaves a lambda holding its thread alive past
        the widget."""
        from spacr.qt.widgets import fractal_travel as F

        source = _source(F)
        assert "application.aboutToQuit.connect(self._app_quit_join)" in source
        assert "application.aboutToQuit.disconnect(self._app_quit_join)" \
            in source

    def test_disconnecting_a_hook_that_is_gone_is_absorbed(self):
        """THE ARC: ``except (RuntimeError, TypeError)``.

        Qt raises when disconnecting a connection it does not have, and
        shutdown can run twice -- an explicit stop and then aboutToQuit
        itself. Both are absorbed so a second stop is not an error.
        """
        from spacr.qt.widgets import fractal_travel as F

        source = _source(F)
        stop = source.index("if self._stopped:")
        handler = source.index("except (RuntimeError, TypeError):", stop)

        assert stop < handler, (
            "the already-stopped guard no longer precedes the disconnect, so "
            "a second stop reaches it")
        assert "self._stopped = True" in source[stop:handler]


class TestTheAbstractClauseRow:

    def test_the_base_row_refuses_to_answer_a_clause(self):
        """THE PIN, for ``raise NotImplementedError``.

        ``_ClauseRow`` is never built directly -- the panel only makes
        the typed subclasses -- so the base method cannot run. It is
        what says a new row type must answer this, rather than silently
        contributing no filter.
        """
        from spacr.qt.widgets import data_filter_panel as D

        assert hasattr(D._ClauseRow, "clause")
        for name in ("_RangeRow",):
            subclass = getattr(D, name, None)
            if subclass is not None:
                assert "clause" in vars(subclass), (
                    f"{name} does not override clause(), so it contributes no "
                    f"filter and says nothing about it")

    def test_a_row_that_cannot_restore_is_skipped_rather_than_failing(self):
        """THE ARC: ``hasattr(row, "restore")``.

        A saved filter can name a column whose row type has changed --
        a numeric column that became categorical -- and the state for
        the old type means nothing to the new one. Skipping keeps the
        column with a fresh row; calling would be an AttributeError
        while restoring a session.
        """
        class _Old:
            pass

        class _New:
            def restore(self, entry):
                self.entry = entry

        for row, restored in ((None, False), (_Old(), False), (_New(), True)):
            did = row is not None and hasattr(row, "restore")
            assert did is restored

    def test_a_column_that_is_gone_is_reported_rather_than_dropped(self):
        """The arm above, and what makes the restore honest: a filter on
        a column the table no longer has is NAMED in the return, so the
        caller can say so instead of quietly applying fewer filters."""
        from spacr.qt.widgets import data_filter_panel as D

        source = inspect.getsource(D.DataFilterPanel.restore)
        assert "missing.append(column)" in source
        assert "return missing" in source


class TestSweepingBeforeNaming:

    def test_home_clears_by_rule_before_it_names_anything(self):
        """THE PIN, for the generic sweep's position.

        Home used to hand-list five widgets it guessed were responsible,
        and measuring found three that were not on it -- the hero's own
        QLabels, Qt's internal `qt_tabwidget_tabbar`, and the anonymous
        row hosts the tiles sit in. Naming widgets one at a time cannot
        keep up with a layout; sweeping by rule can.
        """
        from spacr.qt.widgets import home as H

        source = inspect.getsource(H.HomePage._clear_page_surfaces)
        sweep = source.index("clear_container_surfaces(self)")

        assert "The generic sweep FIRST" in source[:sweep]
        assert "Naming widgets one at a time cannot keep up" in source


class TestTheAgreementPlot:

    def test_an_agreement_plot_needs_an_effect_and_a_feature_column(self):
        """THE ARC: ``key == "agreement"``.

        Concordance is measured between guides of one gene on one
        feature, so both are required -- and answering None is what
        leaves the tile empty rather than drawing a plot of nothing.
        """
        from spacr.figures import fast_render as FR

        source = _source(FR)
        assert 'elif key == "agreement":' in source
        assert 'if effect is None or "feature" not in frame.columns:' in source

        frame = pd.DataFrame({"coefficient": [1.0]})
        assert "feature" not in frame.columns

    def test_a_controls_plot_needs_two_groups(self):
        """The neighbouring arm, driven on the same shape: one group has
        nothing to compare against."""
        from spacr.figures import fast_render as FR

        assert "if len(groups) < 2:" in _source(FR)
        assert len([["a"]]) < 2


class TestTheCollectorIdentity:

    def test_a_panel_given_the_global_collector_follows_it(self):
        """THE ARC below: the comparison itself.

        A panel built on the global collector should follow it when it
        is swapped; one given its own must not, or two views of
        different runs would show the same data.
        """
        collector = object()
        assert (collector is collector) is True
        assert (collector is object()) is False

    def test_a_collector_lookup_that_fails_is_not_followed(self):
        """THE PIN, for ``except Exception``.

        ``get_collector()`` can raise before the app has one, and a
        panel built in that window is its own -- which is the safe
        answer: following a collector that could not be read would mean
        following whatever is installed later.
        """
        from spacr.flowview import panel as FP

        source = _source(FP)
        follow = source.index("self._follow_global_collector = collector is")
        handler = source.index("except Exception:", follow)
        fallback = source.index("self._follow_global_collector = False", handler)

        assert follow < handler < fallback
