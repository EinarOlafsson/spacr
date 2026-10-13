"""The compound option of Control Charts' hit scoring, behind the alpha gate.

Hidden with Preferences -> Show alpha features off and shown with it on; a
compound table loaded while the option is hidden still reaches the SAR
export, because hiding is a display decision only.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from spacr.qt.screens.control_chart import (             # noqa: E402
    _CHEMISTRY_ALPHA_WIDGETS, ControlChartScreen)

from tests.test_chemistry_aware_hits import _screen     # noqa: E402


@pytest.fixture
def screen(qtbot):
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


@pytest.fixture
def gate(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _score(screen):
    frame, layout, _host = _screen()
    screen.set_frame(frame)
    screen._value.setCurrentText("signal")
    screen._control_column.setCurrentText("well_type")
    screen._on_control_column("well_type")
    screen._negative.setCurrentText("neg")
    screen._positive.setCurrentText("pos")
    screen._hit_score.setChecked(True)
    return layout


def test_registered_alpha_and_follows_the_switch(screen, gate):
    from PySide6.QtCore import QObject

    from spacr.settings import ALPHA_FEATURES, _alpha_names

    assert set(_CHEMISTRY_ALPHA_WIDGETS) == set(ALPHA_FEATURES[584]["widgets"])
    assert set(_CHEMISTRY_ALPHA_WIDGETS) <= _alpha_names("widgets")
    gate._apply_alpha_widgets(screen)
    for name in _CHEMISTRY_ALPHA_WIDGETS:
        assert screen.findChild(QObject, name).isHidden(), name
    gate._set_show_alpha_features(True)
    gate._apply_alpha_widgets(screen)
    for name in _CHEMISTRY_ALPHA_WIDGETS:
        assert not screen.findChild(QObject, name).isHidden(), name


def test_a_table_loaded_while_hidden_still_reaches_the_export(
        screen, gate, tmp_path):
    from PySide6.QtCore import QObject

    gate._apply_alpha_widgets(screen)
    assert screen.findChild(QObject, "ControlChartChemistry").isHidden()
    layout = _score(screen)
    assert screen._load_compounds(layout)
    chem = screen._chemistry_result
    assert chem is not None
    assert int(chem.sar["hit"].sum()) == 5
    assert screen.sar_table.rowCount() == len(chem.sar)
    written = screen.export_hits(str(tmp_path / "hits"))
    assert os.path.exists(written["hit_table"])
    assert os.path.exists(written["sar_table"])


def test_structures_are_drawn_with_rdkit(screen):
    pytest.importorskip("rdkit")
    screen._load_compounds(_score(screen))
    chem = screen._chemistry_result
    assert chem.clustered
    assert len(screen.chem_figure.axes) == 5
    screen._chem_similarity.setValue(0.95)
    assert len(screen._chemistry_result.clusters) == 5


def test_a_table_without_smiles_says_so(screen):
    import pandas as pd

    _score(screen)
    assert not screen._load_compounds(pd.DataFrame({"well": ["A02"]}))
    assert "SMILES" in screen._compound_label.text()
    assert screen._chemistry_result is None
