"""The Overview metadata step's senders, readout fitting and stale-owner guards."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt

PATTERN = (r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)"
           r"-c(?P<chanID>\d+)\.tif")


class _Sender:
    def __init__(self, path):
        self._path = path

    def property(self, name):
        assert name == "platePath"
        return self._path


def _stub(sender):
    class Stub:
        calls = []

        def sender(self):
            return sender

        def _use_plate_subfolders(self, path, on):
            self.calls.append(("subfolders", path, on))

        def _apply_to_plate(self, path):
            self.calls.append(("apply", path))

    stub = Stub()
    stub.calls = []
    return stub


def test_a_switch_flipped_acts_on_the_plate_it_stands_for():
    stub = _stub(_Sender("/screen/p1"))
    browser._OverviewMetadataWizard._plate_subfolders_toggled(stub, True)
    assert stub.calls == [("subfolders", "/screen/p1", True)]


def test_a_switch_with_no_plate_property_acts_on_the_empty_path():
    stub = _stub(_Sender(None))
    browser._OverviewMetadataWizard._plate_subfolders_toggled(stub, False)
    assert stub.calls == [("subfolders", "", False)]


def test_a_signal_with_no_sender_changes_no_plate():
    stub = _stub(None)
    browser._OverviewMetadataWizard._plate_subfolders_toggled(stub, True)
    browser._OverviewMetadataWizard._apply_plate_button_clicked(stub)
    assert stub.calls == []


def test_the_apply_button_applies_to_its_own_plate():
    stub = _stub(_Sender("/screen/p2"))
    browser._OverviewMetadataWizard._apply_plate_button_clicked(stub, False)
    assert stub.calls == [("apply", "/screen/p2")]


def _wizard(qtbot, tmp_path):
    plate = tmp_path / "screen" / "plate1"
    plate.mkdir(parents=True)
    for well in ("A01", "A02"):
        for channel in (1, 2):
            tifffile.imwrite(plate / f"p1-{well}-f1-c{channel}.tif",
                             np.full((24, 32), 100, dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(
        str(tmp_path / "screen"), {"custom_regex": PATTERN})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    wizard.resize(1200, 800)
    return wizard


def test_readouts_are_fitted_while_the_preview_table_lives(qtbot, tmp_path):
    wizard = _wizard(qtbot, tmp_path)
    wizard.show()
    wizard._fit_readouts()
    assert wizard._readout_timer is not None


def test_a_wizard_whose_preview_table_is_gone_stops_fitting_its_readouts(
        qtbot, tmp_path, monkeypatch):
    wizard = _wizard(qtbot, tmp_path)
    wizard.show()
    started = []
    monkeypatch.setattr(wizard._readout_timer, "start", started.append)
    wizard._readouts_gone()
    assert wizard._readouts_alive is False
    wizard._fit_readouts()
    wizard.resize(1000, 700)
    assert started == []


def test_a_wizard_other_than_the_current_one_is_ignored(qtbot):
    class Owner:
        wizard = object()

        def _drop_wizard(self, _wizard):
            raise AssertionError("must not drop a stale wizard")

        def _retire_browser(self):
            raise AssertionError("must not retire for a stale wizard")

    owner = Owner()
    stale = object()
    browser._OverviewModulePage._wizard_closed(owner, stale)
    browser._OverviewModulePage._show_browser(owner, stale)
