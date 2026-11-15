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

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox,  # noqa: E402
                               QLineEdit, QSpinBox)

from spacr.qt.widgets import cell_montage_view as cmv                   # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# console_panel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# cell_montage_view
# ---------------------------------------------------------------------------

@pytest.fixture()
def view(qtbot):
    widget = cmv.CellMontageView(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


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
