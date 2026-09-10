"""Guards in the console panel and the montage tab that no input can trip.

Round 6 asked for eight arcs across :mod:`spacr.qt.widgets.console_panel` and
:mod:`spacr.qt.widgets.cell_montage_view`. Every one of them turned out to be
a defensive arm whose condition was already decided a few lines earlier, so
none of them is faked here. Each test drives the code that makes the arm dead
and asserts the guarantee itself, so the day a guarantee stops holding, the
failure lands here -- on the invariant -- rather than silently waking a branch
nobody has ever executed.

What is pinned:

* **console_panel** -- ``_TopicBar._copy_section``'s second ``panel is None``
  test, and the ``widget is not None`` tests in ``section_body`` and
  ``clear``. The first cannot fire because the bounded walk above it has
  already returned for a missing parent; the other two cannot fire because
  the entries layout holds exactly one item that is not a widget -- the
  trailing stretch -- and both loops stop before it.
* **cell_montage_view** -- the ``try: pass`` block left behind in ``load``
  (its two handlers can never run, because a bare ``pass`` cannot raise), the
  fourth-type fall-through in ``_write_back`` / ``_read_widgets`` (every
  mirrored control is one of the three types the chain names), and
  ``_announce``'s ``not self._name`` arm (``reason()`` has already returned a
  sentence in exactly that case).

Every "nothing happened" assertion is paired, in the same test, with the input
that makes something happen.
"""
from __future__ import annotations

import ast
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QLineEdit, QSpinBox)

from spacr.qt.widgets import cell_montage_view as cmv                   # noqa: E402
from spacr.qt.widgets import console_panel as cp                        # noqa: E402
from spacr.qt.widgets.console_panel import ConsolePanel                 # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# console_panel
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    return widget


def _bars(panel):
    """Every topic bar currently in the console, in order."""
    return [panel._entries.itemAt(i).widget()
            for i in range(panel._entries.count())
            if panel._entries.itemAt(i).widget() is not None
            and isinstance(panel._entries.itemAt(i).widget(), cp._TopicBar)]


def test_a_topic_bar_copies_its_section_only_while_it_still_has_a_panel(
        panel, qtbot):
    """``_copy_section``'s walk is the only thing that can find no panel.

    The second ``if panel is None: return`` (line 397) is dead: the bounded
    walk above it returns for a missing parent (line 391), ``break``s only
    when ``panel`` answers ``section_text`` -- and an object that answers it
    is not None -- and otherwise leaves through the ``for`` ... ``else``. So
    by the time line 397 is evaluated, ``panel`` is the console.

    Both halves are driven here: the bar that is on the panel copies its
    section, and the one that has been taken off it copies nothing.
    """
    panel.begin_topic("spaCR output — mask")
    panel.append_stdout("one\ntwo\n")
    bar = _bars(panel)[0]

    clipboard = QApplication.clipboard()
    clipboard.setText("untouched")
    bar._copy_section()
    copied = clipboard.text()
    assert copied != "untouched"
    assert copied == panel.section_text(bar)
    assert "one" in copied and "two" in copied

    # The same bar, taken off the console: the walk starts at a parent of
    # None and returns before anything is put on the clipboard.
    orphan = cp._TopicBar("spaCR output — mask")
    qtbot.addWidget(orphan)
    assert orphan.parent() is None
    clipboard.setText("untouched")
    orphan._copy_section()
    assert clipboard.text() == "untouched"


def test_the_only_entry_that_is_not_a_widget_is_the_trailing_stretch(panel):
    """Why ``widget is not None`` cannot be false in ``section_body`` (1580)
    or in ``clear`` (1671).

    ``_entries`` is given exactly one non-widget item -- ``addStretch(1)`` in
    ``__init__`` -- and every entry afterwards is inserted with
    ``insertWidget(count - 1, w)``, which keeps that stretch last.
    ``section_body`` stops at ``count - 1`` and ``clear`` stops while
    ``count > 1``, so neither loop ever reaches it.
    """
    panel.begin_topic("spaCR output — mask")
    panel.append_stdout("first\n")
    panel.begin_topic("spaCR output — measure")
    panel.append_stdout("second\n")

    count = panel._entries.count()
    assert count > 3
    for index in range(count - 1):
        item = panel._entries.itemAt(index)
        assert item.widget() is not None, (
            f"entry {index} is not a widget, so the guard could fire")
    last = panel._entries.itemAt(count - 1)
    assert last.widget() is None and last.spacerItem() is not None

    # The body really is read through that loop, and it really returns the
    # widgets between the bar and the next one.
    body = panel.section_body(_bars(panel)[0])
    assert body and all(w is not None for w in body)
    assert any("first" in w.toPlainText() for w in body
               if hasattr(w, "toPlainText"))

    panel.clear()
    assert panel._entries.count() == 1
    assert panel._entries.itemAt(0).spacerItem() is not None


# ---------------------------------------------------------------------------
# cell_montage_view
# ---------------------------------------------------------------------------

@pytest.fixture()
def view(qtbot):
    widget = cmv.CellMontageView(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def test_the_fraction_reader_reports_both_failures_from_the_live_path(
        monkeypatch):
    """The two answers ``load`` gives when the fractions cannot be read.

    A :class:`MontageError` becomes the montage's own sentence, because
    that error was written for the user; anything else becomes the
    "Could not read" sentence with the original text appended. Both mark
    the result unavailable, so the view says nothing is there rather than
    drawing an empty panel.

    This test used to also pin a second ``try`` whose body was a bare
    ``pass``, making its two handlers unreachable. That husk has since
    been deleted -- the pin fired when it was, which is what a pin is
    for -- and the live reader above it is the only path now. The AST
    check below keeps it deleted.
    """
    from spacr import cell_montage as cm

    husks = [node for node in ast.walk(ast.parse(open(cmv.__file__).read()))
             if isinstance(node, ast.Try)
             and len(node.body) == 1 and isinstance(node.body[0], ast.Pass)]
    assert not husks, (
        f"a try whose body is a pass is back at line(s) "
        f"{[node.lineno for node in husks]}; its handlers cannot run")

    request = cmv.MontageRequest(
        name="GRA14_1", level="grna", effect=1.0,
        results_path="/nowhere/at/all", databases=("/nowhere/measurements.db",))

    answers = {}
    for label, error in (("montage", cm.MontageError("no fractions here")),
                         ("other", RuntimeError("the disk went away"))):
        def _raise(_folder, _error=error):
            raise _error

        monkeypatch.setattr(cm, "read_well_guide_fractions", _raise)
        answers[label] = cmv.load(request)

    assert answers["montage"].unavailable is True
    assert answers["montage"].error == "no fractions here"
    assert answers["other"].unavailable is True
    assert answers["other"].error == (
        "Could not read the per-well guide fractions: the disk went away")


def test_every_mirrored_setting_is_one_of_the_three_kinds_of_control(view):
    """Why the ``elif isinstance(widget, QLineEdit)`` fall-through is dead.

    ``_write_back`` (2820) and ``_read_widgets`` (2843) end their chain on
    ``QLineEdit``; falling past it needs a fourth kind of control. There is
    none: each of the eight names in ``_MIRRORED`` is assigned exactly once,
    in ``__init__``, to a ``QComboBox``, ``QLineEdit``, ``QSpinBox`` or
    ``QDoubleSpinBox``.

    Pinned by the types themselves and by a round trip through all three
    arms: what the settings window writes is what the widgets read back.
    """
    kinds = (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit)
    for key, name in cmv.CellMontageView._MIRRORED.items():
        widget = getattr(view, name)
        assert isinstance(widget, kinds), f"{key} is a {type(widget).__name__}"

    before = view._read_widgets()
    assert set(before) == set(cmv.CellMontageView._MIRRORED)

    view._write_back({"channels": "0,1", "cap": 7, "half_widths": 2.5,
                      "object_type": before["object_type"]})
    after = view._read_widgets()

    assert after["channels"] == "0,1"            # the QLineEdit arm
    assert after["cap"] == 7                     # the QSpinBox arm
    assert after["half_widths"] == 2.5           # the QDoubleSpinBox arm
    assert after["object_type"] == before["object_type"]   # the QComboBox arm


def test_a_montage_with_no_coefficient_name_never_reaches_the_status_line(
        view):
    """Why ``if not self._name`` in ``_announce`` (3264) cannot be true.

    ``_announce`` only reaches that line when ``reason()`` returned ``''``,
    and ``reason()`` returns a sentence at its own line 2146 whenever
    ``_name`` is empty -- ``NOTHING_SELECTED`` when no coefficient is chosen
    at all, and the "names neither a gene nor a guide" sentence when one is
    chosen but is a nuisance term. So a nameless view never gets past the
    first arm of ``_announce``.
    """
    assert view._name == ""
    view._announce()
    assert view.reason() == cmv.CellMontageView.NOTHING_SELECTED
    assert view._status_text == cmv.CellMontageView.NOTHING_SELECTED

    # A key that names no gene or guide is the other nameless state, and it
    # too is answered by `reason()` rather than by the guard below it.
    view._key = "plate[p1]"
    view._name = ""
    view._announce()
    said = view._status_text
    assert said != cmv.CellMontageView.NOTHING_SELECTED
    assert "neither a gene nor a guide" in said

    # The implication that makes the guard dead, stated over the states a
    # view can be in: `reason()` is never empty while `_name` is, so
    # `_announce` never reaches line 3264 with an empty name.
    for key in ("", "plate[p1]", "fraction:grna[GRA14_1]"):
        view._key, view._name = key, ""
        assert view.reason() != ""

    # And a NAMED view is answered by `reason()` as well -- with a different
    # sentence, which is what says the first arm of `_announce` is doing the
    # work rather than the guard below it.
    view._key, view._name, view._effect = "fraction:grna[GRA14_1]", "GRA14_1", 1.0
    view._announce()
    assert view._status_text == cmv.CellMontageView.NO_RUN_LOADED
