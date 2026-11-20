"""Items 983/984: Annotate draws OME-Zarr crops with any-channel image and outline roles."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("zarr")

from spacr.crop_store import (CropPoolWriter, OmeZarrCropSource, composite_crop,
                              cut_crops, read_crop, save_channel_preset)
from spacr.qt import annotate_engine as engine
from spacr.qt.screens import annotate as annotate_mod

pytestmark = pytest.mark.qt

ROLES = [{"channel": 0, "role": "image", "colour": "blue"},
         {"channel": 3, "role": "image", "colour": "magenta", "gamma": 1.5},
         {"channel": "mask", "role": "outline", "colour": "yellow", "width": 1}]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with four-channel OME-Zarr crops and their ssCell_Crops table."""
    rng = np.random.default_rng(2)
    image = rng.integers(0, 60000, (4, 50, 50)).astype(np.uint16)
    labels = np.zeros((50, 50), np.int32)
    labels[4:20, 6:26] = 1
    labels[28:46, 30:48] = 2
    writer = CropPoolWriter(tmp_path, "plate1", "A01", "cell", 4)
    writer.add(cut_crops(image, labels, margin=2), field=1)
    (tmp_path / "measurements").mkdir()
    with sqlite3.connect(str(tmp_path / "measurements" / "measurements.db")) as conn:
        pd.DataFrame(writer.commit()["rows"]).to_sql("ssCell_Crops", conn, index=False)
    return tmp_path


def _settings(project: Path, roles=None) -> engine.AnnotateSettings:
    s = engine.AnnotateSettings()
    s.src = str(project)
    s.db_path = str(project / "measurements" / "measurements.db")
    s.image_size = (32, 32)
    s.channels = []
    s.normalize_channels = []
    s.channel_roles = roles
    return s


def test_the_thumbnail_worker_draws_four_channels_with_an_outline(project):
    source = OmeZarrCropSource(str(project))
    row = {"object_id": "plate1_A01_1_cell_1", "annotation": None}
    img, _annotation = annotate_mod._load_thumb_image_worker(
        row, source, _settings(project, ROLES))
    pixels, mask = read_crop(project, source._crop_row(row))
    expected = composite_crop(pixels, ROLES, mask=mask)
    from PIL import Image

    settings = _settings(project, ROLES)
    settings.normalize_channels = ["r", "g", "b"]
    settings.channels = ["r"]
    settings.outline = ["g"]
    img, _annotation = annotate_mod._load_thumb_image_worker(row, source, settings)
    assert np.array_equal(np.asarray(img),
                          np.asarray(Image.fromarray(expected).resize((32, 32))))
    plain, _a = annotate_mod._load_thumb_image_worker(row, source, _settings(project))
    assert np.asarray(plain).shape == np.asarray(img).shape
    assert not np.array_equal(np.asarray(plain), np.asarray(img))


def test_the_dialog_offers_project_presets_and_returns_parsed_roles(qtbot, project):
    save_channel_preset(project, "nuclei+outline", ROLES)
    dialog = annotate_mod._SettingsDialog(_settings(project))
    qtbot.addWidget(dialog)
    combo = dialog._role_preset
    index = combo.findData("nuclei+outline")
    assert index > 0
    combo.setCurrentIndex(index)
    chosen = dialog.collect()
    assert chosen.channel_roles == ROLES
    dialog._channel_roles.setText("")
    assert dialog.collect().channel_roles is None


def test_the_roles_example_is_valid_syntax_and_not_a_translatable_string(qtbot, project):
    from spacr.crop_store import format_channel_roles, parse_channel_roles

    text = format_channel_roles(annotate_mod._CHANNEL_ROLES_EXAMPLE)
    assert text == "0:image:blue; 1:image:green:gamma=1.5; mask:outline:yellow:width=2"
    assert parse_channel_roles(text) == [dict(r) for r in annotate_mod._CHANNEL_ROLES_EXAMPLE]
    dialog = annotate_mod._SettingsDialog(_settings(project))
    qtbot.addWidget(dialog)
    assert dialog._channel_roles.placeholderText() == text


def test_native_annotation_page_save_and_reopen_use_object_ids(project):
    """Real table pages and asynchronous saves work without png_list or PNG paths."""
    from spacr.crops import resolve_crop_source

    settings = _settings(project, ROLES)
    settings.png_table = "ssCell_Crops"
    engine.ensure_annotation_column(settings.db_path, "annotate", table=settings.png_table)
    assert engine.count_rows(settings.db_path, "cell_png", table=settings.png_table) == 2
    page = engine.fetch_page(settings.db_path, "annotate", 0, 1, "cell_png",
                             table=settings.png_table)
    assert len(page) == 1
    identity, annotation = page[0]
    assert identity.startswith("plate1_A01_1_cell_")
    assert annotation is None
    source = resolve_crop_source({"src": str(project), "crop_source": "ome-zarr"})
    thumbnail, annotation = annotate_mod._load_thumb_image_worker(page[0], source, settings)
    pixels, mask = source.get_raw({"object_id": identity})
    expected = Image.fromarray(composite_crop(pixels, ROLES, mask=mask)).resize(settings.image_size)
    np.testing.assert_array_equal(np.asarray(thumbnail), np.asarray(expected))
    worker = engine.SaveWorker(settings.db_path, "annotate", table=settings.png_table)
    worker.start()
    other_identity = engine.fetch_page(settings.db_path, "annotate", 1, 1, "cell_png",
                                       table=settings.png_table)[0][0]
    worker.submit({identity: 2, other_identity: 12})
    worker.stop(wait=True)
    assert worker.last_error is None
    assert engine.fetch_page(settings.db_path, "annotate", 0, 1, "cell_png",
                             table=settings.png_table) == [(identity, 2)]
    assert engine.fetch_page(settings.db_path, "annotate", 1, 1, "cell_png",
                             table=settings.png_table) == [(other_identity, 12)]
    assert engine.class_counts(settings.db_path, "annotate", table=settings.png_table) == [(2, 1)]
    assert engine.find_last_annotated_offset(settings.db_path, "annotate", 1, "cell_png",
                                             table=settings.png_table) == 1
    with sqlite3.connect(settings.db_path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='png_list'").fetchone() is None


def test_native_annotation_screen_opens_the_real_crop_table(qtbot, project, monkeypatch):
    """The normal source-open path selects ssCell_Crops before creating its save worker."""
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    def defer_population(**kwargs):
        """Keep this source-opening assertion independent of background pagination."""
        return None
    monkeypatch.setattr(screen, "_refresh_total", defer_population)
    screen._settings = _settings(project)
    screen._settings.crop_source = "ome-zarr"
    screen._open_source(str(project))
    assert screen._settings.png_table == "ssCell_Crops"
    assert screen._worker.table == "ssCell_Crops"
    assert screen._crop_source().kind == "ome-zarr"
    screen._worker.stop(wait=True)
    screen._worker = None


def test_annotation_settings_keep_existing_automatic_selection(qtbot, project):
    """Opening and saving settings preserves the inherited automatic resolver."""
    settings = _settings(project)
    settings.crop_source = "auto"
    dialog = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dialog)
    assert dialog.collect().crop_source == "auto"
