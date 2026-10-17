"""The local alpha example loads aligned labels and selects CPU inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.qt


@pytest.fixture
def demo_screen(qtbot, monkeypatch, tmp_path):
    from spacr import example_archives
    from spacr.qt import preferences
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    prepare = example_archives._prepare_embeddings_example
    monkeypatch.setattr(example_archives, "_prepare_embeddings_example",
                        lambda: prepare(tmp_path / "demo"))
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def test_the_demo_is_an_alpha_action(demo_screen, monkeypatch):
    from spacr.qt import preferences
    from spacr.settings import ALPHA_FEATURES

    button = demo_screen._demo_button
    assert button.objectName() == "EmbeddingsDemoButton"
    assert button.objectName() in ALPHA_FEATURES[560]["widgets"]
    assert not button.isHidden()
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    preferences._apply_alpha_widgets(demo_screen)
    assert button.isHidden()


def test_pressing_demo_replaces_old_vectors_and_loads_aligned_labels(demo_screen):
    from spacr.crop_loader import CROP_SOURCE_FOLDER
    from spacr.qt.screens.embeddings import DIMENSION_CAVEAT
    from spacr.tabular import read_table

    demo_screen.set_crops(np.ones((2, 8, 8, 3), dtype=np.uint8))
    demo_screen._frame = pd.DataFrame({"old_embedding": [1.0, 2.0]})
    demo_screen._labels = {"0": "old", "1": "old"}
    demo_screen._foundation.setCurrentIndex(
        demo_screen._foundation.findData("cell_dino"))
    demo_screen._limit.setValue(1)
    loaded = []
    demo_screen.crops_loaded.connect(loaded.append)

    demo_screen._demo_button.click()

    assert loaded == [24]
    assert demo_screen._crops.shape == (24, 64, 64, 3)
    assert demo_screen.crop_source() == CROP_SOURCE_FOLDER
    assert demo_screen.crop_record()["loaded"] == 24
    expected = read_table(
        str(demo_screen._plan.query.path) + "/../labels.csv", report=None)
    assert list(demo_screen._labels.values()) == expected["label"].tolist()
    assert len(set(demo_screen._labels.values())) == 3
    assert demo_screen._frame is None
    assert demo_screen._result is None
    spec = demo_screen.spec()
    assert (spec.backbone, spec.device, spec.channel_policy, spec.batch_size) == (
        "vit_small_patch14_dinov2.lvd142m", "cpu", "project", 8)
    assert demo_screen._foundation.currentData() == "dino_auto"
    assert "DINOv2" in demo_screen._dino_in_use.text()
    assert demo_screen._run.isEnabled()
    assert not demo_screen._save_similar.isEnabled()
    assert DIMENSION_CAVEAT in demo_screen._caveat.text()
    assert "not biological validation" in demo_screen._caveat.text()
    assert "Press Embed" in demo_screen._status.text()


def test_other_crops_do_not_inherit_demo_labels_or_cpu_choice(demo_screen):
    from spacr.qt.screens.embeddings import DIMENSION_CAVEAT

    demo_screen._demo_button.click()
    assert demo_screen._labels
    assert demo_screen.spec().device == "cpu"

    demo_screen.set_crops(np.zeros((3, 8, 8, 3), dtype=np.uint8))

    assert demo_screen._labels == {}
    assert demo_screen.spec().device is None
    assert demo_screen._caveat.text() == DIMENSION_CAVEAT


def test_a_demo_preparation_error_keeps_the_existing_selection(
        demo_screen, monkeypatch):
    from spacr import example_archives

    demo_screen.set_crops(np.ones((2, 8, 8, 3), dtype=np.uint8))
    crops = demo_screen._crops
    demo_screen._labels = {"0": "left", "1": "right"}
    frame = pd.DataFrame({"embedding": [1.0, 2.0]})
    demo_screen._frame = frame

    def fail():
        raise OSError("example directory is not writable")

    monkeypatch.setattr(example_archives, "_prepare_embeddings_example", fail)
    demo_screen._demo_button.click()

    assert demo_screen._crops is crops
    assert demo_screen._frame is frame
    assert demo_screen._labels == {"0": "left", "1": "right"}
    assert demo_screen._demo_button.isEnabled()
    assert demo_screen._run.isEnabled()
    assert not demo_screen.is_loading()
    assert "example directory is not writable" in demo_screen._status.text()


@pytest.mark.parametrize("language", [
    "sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is",
])
def test_the_demo_uses_installed_translations_and_control_captions(
        qtbot, monkeypatch, tmp_path, language):
    from spacr import example_archives
    from spacr.qt import preferences
    from spacr.qt.i18n import tr
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    monkeypatch.setenv("SPACR_LANGUAGE", language)
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    prepare = example_archives._prepare_embeddings_example
    monkeypatch.setattr(example_archives, "_prepare_embeddings_example",
                        lambda: prepare(tmp_path / "demo"))
    records_path = (Path(__file__).resolve().parents[2]
                    / "docs/i18n/reviewed/runtime" / language
                    / "2026-10-16-embeddings-synthetic-demo.json")
    records = json.loads(records_path.read_text())["records"]
    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)

    assert screen._demo_button.text() == records[0]["translation"]
    dino = json.loads((records_path.parent
                       / "2026-10-17-560-dinov2-drop-in.json").read_text())["records"]
    assert screen._demo_button.toolTip() == next(
        row["translation"] for row in dino
        if row["source"].startswith("Load 24 generated shapes"))
    screen._demo_button.click()

    assert records[3]["translation"] in screen._caveat.text()
    assert screen._status.text() == records[4]["translation"].format(
        embed=tr("Embed"), classifier=tr("Classifier"), run=tr("Run"),
        folder=tmp_path / "demo")


def _fake_example_plate(folder):
    import sqlite3

    from PIL import Image

    rows = []
    for index, value in enumerate((1, 2, 11, 2, 1, None, 2)):
        path = folder / "data" / f"crop_{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 40 * index, dtype=np.uint8)).save(path)
        rows.append((str(path), path.name, "plate1", f"plate1_r1_c1_f1_o{index + 1}",
                     f"o{index + 1}", value))
    database = folder / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE png_list (png_path TEXT, file_name TEXT, "
                     "plateID TEXT, prcfo TEXT, cell_id TEXT, infected INTEGER)")
        conn.executemany("INSERT INTO png_list VALUES (?, ?, ?, ?, ?, ?)", rows)
    return database


def test_the_cell_example_loads_annotated_cells_with_aligned_labels(
        demo_screen, monkeypatch, tmp_path):
    from spacr import example_archives
    from spacr.qt.widgets import measurements_example
    from spacr.settings import ALPHA_FEATURES
    from spacr.tabular import read_table

    database = _fake_example_plate(tmp_path / "plate1")
    monkeypatch.setattr(measurements_example, "example_measurements_db",
                        lambda: database)
    monkeypatch.setattr(example_archives, "example_plate_folder",
                        lambda: tmp_path / "plate1")
    button = demo_screen._cell_example_button
    assert button.objectName() in ALPHA_FEATURES[560]["widgets"]
    assert "EmbeddingsDinoInUseLabel" in ALPHA_FEATURES[560]["widgets"]

    button.click()

    assert demo_screen._crops.shape[0] == 5
    assert list(demo_screen._labels.values()) == ["1", "2", "2", "1", "2"]
    written = read_table(tmp_path / "embeddings_cell_example" / "labels.csv",
                         report=None)
    assert written["label"].astype(str).tolist() == ["1", "2", "2", "1", "2"]
    assert written["prcfo"].tolist() == [
        f"plate1_r1_c1_f1_o{n}" for n in (1, 2, 4, 5, 7)]
    spec = demo_screen.spec()
    assert (spec.backbone, spec.device) == ("vit_small_patch14_dinov2.lvd142m",
                                            "cpu")
    assert "not a scientific comparison" in demo_screen._caveat.text()
    assert "Press Embed" in demo_screen._status.text()
    assert not demo_screen._dino_in_use.isHidden()


def test_an_installed_mapped_cell_dino_replaces_dinov2(
        demo_screen, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from spacr import embeddings

    monkeypatch.setenv("SPACR_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setattr(embeddings, "_CELL_DINO_SEEN", {})
    folder = tmp_path / "models" / "cell_dino"
    folder.mkdir(parents=True)
    torch.save({"patch_embed.proj.weight": torch.zeros(8, 4, 16, 16)},
               folder / "hpa.pth")
    demo_screen.set_crops(np.zeros((3, 8, 8, 4), dtype=np.float32))
    demo_screen._foundation.setCurrentIndex(
        demo_screen._foundation.findData("dino_auto"))

    assert "not mapped" in demo_screen._dino_in_use.text()
    assert demo_screen.spec().backbone == "vit_small_patch14_dinov2.lvd142m"
    assert demo_screen._policy.isEnabled()

    demo_screen._cell_dino_channels = (0, 1, 2, 3)
    demo_screen._sync_subcell_controls()

    assert "official Cell-DINO (cell_dino_hpa_vitl16)" in demo_screen._dino_in_use.text()
    spec = demo_screen.spec()
    assert (spec.backbone, spec.cell_dino_factory, spec.channels) == (
        "cell_dino", "cell_dino_hpa_vitl16", (0, 1, 2, 3))
    assert len(spec.checkpoint_sha256) == 64


def test_the_dino_line_is_alpha_gated(demo_screen, monkeypatch):
    from spacr.qt import preferences

    demo_screen._foundation.setCurrentIndex(
        demo_screen._foundation.findData("dino_auto"))
    assert not demo_screen._dino_in_use.isHidden()
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    preferences._apply_alpha_widgets(demo_screen)
    assert demo_screen._dino_in_use.isHidden()
    assert demo_screen._cell_example_button.isHidden()
