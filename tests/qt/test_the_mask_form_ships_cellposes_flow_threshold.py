"""428, driven the way a user meets it: the Mask form, Import, Run.

GitHub #123 (jak18015, spaCR 1.5.0.8): every Mask run warned that the three
flow thresholds were 100, because 100 was the default, and the warning's fix
line said "spaCR ships 1.0". The default is Cellpose's own 0.4 since
2026-09-19.

The pipeline itself is stubbed; everything in front of it is real -- the
form's widgets, the Import settings button reading a CSV, and the bridge
entry point the Run button calls, which prints the ``[settings] WARNING``
lines the reporter saw.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

MASK_KEYS = ("nucleus_flow_threshold", "cell_flow_threshold",
             "pathogen_flow_threshold")


@pytest.fixture
def mask_screen(qtbot):
    """A Mask screen as it opens."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


@pytest.fixture
def run_entry(monkeypatch):
    """What the Run button calls, with the segmentation stubbed out."""
    import spacr.core
    from spacr.qt.bridge import resolve_pipeline_entry

    received = []
    monkeypatch.setattr(spacr.core, "preprocess_generate_masks",
                        lambda settings: received.append(dict(settings)))
    return resolve_pipeline_entry("mask"), received


def _flow_warning_lines(printed: str):
    return [line for line in printed.splitlines()
            if line.startswith("[settings]") and "_flow_threshold" in line]


def _import_csv(screen, monkeypatch, path):
    """Press Import settings and choose ``path`` in the file dialog."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    screen._on_import_settings()


def test_the_form_opens_at_cellposes_own_default(mask_screen):
    """The three boxes the user sees hold 0.4, and so does what Run sends."""
    model = mask_screen._settings_model
    collected = model.collect()

    for key in MASK_KEYS:
        assert collected[key] == pytest.approx(0.4), (key, collected[key])
        assert model._widgets[key].value() == pytest.approx(0.4), key


def test_running_the_untouched_form_prints_no_flow_warning(
        mask_screen, run_entry, capsys):
    """#123: the shipped default was warned about on every run."""
    entry, received = run_entry
    capsys.readouterr()

    entry(mask_screen._settings_model.collect())

    printed = capsys.readouterr().out
    assert not _flow_warning_lines(printed), _flow_warning_lines(printed)
    assert received and received[0]["cell_flow_threshold"] == pytest.approx(0.4)


@pytest.mark.parametrize("spelling", ["cell_flow_threshold", "cell_FT"])
def test_an_imported_100_is_kept_and_the_warning_tells_the_truth(
        mask_screen, run_entry, monkeypatch, tmp_path, capsys, spelling):
    """A settings file saved by 1.5.0.8 keeps its 100, and is told why.

    ``cell_FT`` is how every file written before 2026-09-02 spells it.
    """
    path = tmp_path / "gen_mask_settings.csv"
    path.write_text(f"Key,Value\n{spelling},100\n", encoding="utf-8")
    entry, received = run_entry

    _import_csv(mask_screen, monkeypatch, path)
    held = mask_screen._settings_model.collect()
    capsys.readouterr()
    entry(held)
    printed = capsys.readouterr().out

    assert float(held["cell_flow_threshold"]) == 100.0, "the saved value was rewritten"
    assert float(received[0]["cell_flow_threshold"]) == 100.0
    assert held["nucleus_flow_threshold"] == pytest.approx(0.4)
    warned = _flow_warning_lines(printed)
    assert any("[cell_flow_threshold]" in line for line in warned), printed
    assert "spaCR and Cellpose both default to 0.4" in printed
    assert "Set cell_flow_threshold to 0.4" in printed
    assert "spaCR ships 1.0" not in printed
