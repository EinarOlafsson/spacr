"""The metadata wizard stops fitting its readouts once their preview table is gone.

The preview lives in the settings content a mounted Overview lends to its
module, which can be torn down before the wizard. A resize then queued the
readout fit against the deleted table and raised "Internal C++ object
(QTableWidget) already deleted" in the event loop, failing whichever
Overview test was tearing down.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt


def test_a_resize_after_the_preview_is_destroyed_fits_nothing(tmp_path, qtbot):
    tifffile.imwrite(tmp_path / "A01_C1.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard._startup_timer.stop()
    wizard.show()
    wizard.preview.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    wizard.resize(wizard.width() + 40, wizard.height() + 40)
    QApplication.processEvents()
    assert not wizard._readout_timer.isActive()
    wizard.reject()
