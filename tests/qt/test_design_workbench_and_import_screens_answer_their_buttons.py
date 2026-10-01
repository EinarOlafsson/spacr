"""Three screens answer the buttons their users press at the edges.

Pinned behaviour:

* :class:`spacr.qt.screens.experiment_design.ExperimentDesignScreen` lists
  its shipped templates by key and refuses a key it does not have, leaving
  the design as it was;
* :class:`spacr.qt.screens.train_cellpose.CellposeWorkbenchScreen` stays
  open when one of its module pages refuses to close (a page finishing a
  write), and closes once the page lets it;
* :class:`spacr.qt.screens.foreign.ForeignScreen`'s **Load test data**
  button offers the test-data chooser, and cancelling it changes nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

pytestmark = pytest.mark.qt


def test_the_template_menu_lists_keys_and_refuses_unknown_ones(qtbot):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen
    from spacr.qt.widgets.plate_layout import plate_templates

    screen = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(screen)
    shipped = [template.key for template in plate_templates()]

    assert screen.template_keys() == shipped
    before = screen.design()
    assert screen.load_template("zz_no_such_template") is False
    assert screen.design() == before


def test_the_workbench_stays_open_while_a_page_refuses_to_close(
        qtbot, monkeypatch):
    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen

    workbench = CellposeWorkbenchScreen()
    qtbot.addWidget(workbench)
    workbench.show()
    qtbot.waitExposed(workbench)
    refusals = [True]

    def _busy_page():
        if refusals:
            refusals.pop()
            return False
        return True

    monkeypatch.setattr(workbench.train_screen, "close", _busy_page)

    assert workbench.close() is False
    assert workbench.isVisible()

    assert workbench.close() is True
    assert not workbench.isVisible()


def test_load_test_data_offers_the_chooser_and_a_cancel_changes_nothing(
        qtbot, monkeypatch):
    from spacr.qt import import_demo
    from spacr.qt.screens.foreign import ForeignScreen

    screen = ForeignScreen(threaded=False)
    qtbot.addWidget(screen)
    offered = []

    def _cancelled(dialog):
        offered.append(dialog.parent())
        return 0

    monkeypatch.setattr(import_demo.ImportTestDataChooser, "exec", _cancelled)
    status_before = screen.status_text()

    qtbot.mouseClick(screen._btn_test_data, Qt.MouseButton.LeftButton)

    assert offered == [screen]
    assert screen.status_text() == status_before
