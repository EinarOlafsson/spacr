"""Issue 134: recover stale forms explicitly, retain wrong-plane refusal."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from spacr.crops import (MERGED_LAYOUT_SIDECAR, PlaneLayoutConflict,
                         reconcile_merged_mask_dims)
from spacr.qt.screens import app_screen


def _source(tmp_path, pathogen=False):
    merged = tmp_path / "merged"
    merged.mkdir(parents=True)
    order = ["cell", "pathogen"] if pathogen else ["cell"]
    manifest = merged / MERGED_LAYOUT_SIDECAR
    manifest.write_text(json.dumps({
        "version": 1, "intensity_channels": [0, 1],
        "mask_plane_order": order,
        "mask_dims": {role: 2 + index for index, role in enumerate(order)},
    }))
    return merged, manifest


def _settings(source):
    return {"src": str(source), "channels": [0, 1],
            "cell_mask_dim": "2", "pathogen_mask_dim": "3",
            "nucleus_mask_dim": None, "uninfected": True, "timelapse": False}


def test_real_measure_entry_refuses_reported_explicit_plane(tmp_path, monkeypatch):
    from spacr import measure
    merged, manifest = _source(tmp_path)
    before = manifest.read_bytes()
    monkeypatch.setattr(measure, "run_context", lambda *a: nullcontext(None))
    with pytest.raises(PlaneLayoutConflict, match="Set pathogen_mask_dim to None"):
        measure.measure_crop(_settings(merged))
    assert manifest.read_bytes() == before
    assert not (tmp_path / "measurements").exists()


def test_defaults_use_manifest_but_explicit_conflicts_still_fail(tmp_path):
    merged, _ = _source(tmp_path)
    settings = _settings(merged)
    resolved = reconcile_merged_mask_dims(settings, merged)
    assert resolved["pathogen_mask_dim"] is None
    assert resolved["cell_mask_dim"] == 2
    assert settings["pathogen_mask_dim"] == "3"
    with pytest.raises(PlaneLayoutConflict):
        reconcile_merged_mask_dims(settings, merged, explicit_keys=settings)


def test_recovered_no_pathogen_settings_measure_real_cell_rows(tmp_path):
    import sqlite3
    import numpy as np
    from spacr.measure import measure_crop

    merged, manifest = _source(tmp_path)
    before = manifest.read_bytes()
    data = np.zeros((32, 32, 3), dtype=np.uint16)
    data[..., 0] = 100
    data[..., 1] = 200
    data[5:20, 5:20, 2] = 1
    np.save(merged / "plate1_A01_F001.npy", data)
    settings = reconcile_merged_mask_dims(_settings(merged), merged)
    settings.update(cell_min_size=0, nucleus_min_size=0, pathogen_min_size=0,
                    save_png=False, plot=False, verbose=False, n_jobs=1,
                    test_mode=False, timelapse=False, normalize=False,
                    save_measurements=True, crop_mode=["cell"])
    measure_crop(settings)
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as db:
        assert db.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 1
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "pathogen" not in tables
    assert manifest.read_bytes() == before


class _Dialog:
    Warning, Cancel, ActionRole = 1, 2, 3
    accept = False
    created = []

    def __init__(self, parent):
        self.button = None
        self.created.append(self)

    def setWindowTitle(self, text): pass
    def setIcon(self, icon): pass
    def setText(self, text): self.text = text
    def setStandardButtons(self, buttons): pass
    def setInformativeText(self, text): self.info = text
    def addButton(self, text, role):
        self.button = object()
        return self.button
    def exec(self): pass
    def clickedButton(self): return self.button if self.accept else None


@pytest.mark.parametrize("accept", [False, True])
def test_form_recovery_requires_explicit_choice_and_another_run(
        tmp_path, monkeypatch, accept):
    merged, manifest = _source(tmp_path)
    before = manifest.read_bytes()
    applied = []
    screen = SimpleNamespace(apply_settings_dict=applied.append)
    monkeypatch.setattr(app_screen, "QMessageBox", _Dialog)
    monkeypatch.setattr(_Dialog, "accept", accept)
    settings = _settings(merged)
    assert not app_screen.AppScreen._confirm_measure_plane_layout(screen, settings)
    assert bool(applied) == accept
    assert settings["pathogen_mask_dim"] == "3"
    if accept:
        assert applied[0]["pathogen_mask_dim"] is None
        settings.update(applied[0])
        assert app_screen.AppScreen._confirm_measure_plane_layout(screen, settings)
    assert manifest.read_bytes() == before


def test_mixed_sources_do_not_offer_one_layout_for_all(tmp_path, monkeypatch):
    first, _ = _source(tmp_path / "one")
    second, _ = _source(tmp_path / "two", pathogen=True)
    settings = _settings(first)
    settings["src"] = [str(first), str(second)]
    monkeypatch.setattr(app_screen, "QMessageBox", _Dialog)
    monkeypatch.setattr(_Dialog, "accept", True)
    applied = []
    screen = SimpleNamespace(apply_settings_dict=applied.append)
    assert not app_screen.AppScreen._confirm_measure_plane_layout(screen, settings)
    assert not applied and _Dialog.created[-1].button is None
