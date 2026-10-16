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
        "resnet18", "cpu", "project", 8)
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
    assert screen._demo_button.toolTip() == records[1]["translation"]
    screen._demo_button.click()

    assert records[4]["translation"] in screen._caveat.text()
    assert screen._status.text() == records[5]["translation"].format(
        embed=tr("Embed"), classifier=tr("Classifier"), run=tr("Run"),
        folder=tmp_path / "demo")
