"""Import's "Load test data…": pick a variant, fetch it, see it read back.

Ledger item 462: "the used presses test zeis import, which downloads the zeis
formated images ... populates the src with those image paths on disk and
allows the user to import them and see that they are propperly imported."

Every download here is mocked: the "download" builds the set from synthetic
pixels with the same builder the published archive came from, straight into
the folder the real worker unpacks into.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_import_example as builder  # noqa: E402

sys.path.remove(str(ROOT / "tools"))

from spacr import import_examples as ix  # noqa: E402
from spacr.qt import import_demo as demo  # noqa: E402


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.foreign import ForeignScreen

    s = ForeignScreen(threaded=False)
    qtbot.addWidget(s)
    return s


@pytest.fixture
def chooser(qtbot):
    d = demo.ImportTestDataChooser()
    qtbot.addWidget(d)
    return d


def _build_into(plate: Path, keys=None) -> Path:
    """What a finished download leaves: the set unpacked in the plate."""
    return builder.build(builder.synthetic_fields(),
                         ix.import_example_folder(plate), keys=keys)


class _Download:
    """Stands in for the worker: builds the set, then reports success."""

    def __init__(self, keys=None, error=""):
        self.calls = []
        self.keys = keys
        self.error = error

    def __call__(self, parent, plate, on_done):
        self.calls.append(Path(plate))
        if self.error:
            on_done(None, self.error)
            return
        root = _build_into(Path(plate), self.keys)
        on_done(demo.DownloadResult(root, Path(plate) / "settings"), "")


def test_the_chooser_offers_every_variant(chooser):
    assert set(chooser._buttons) == {v.key for v in ix.IMPORT_VARIANTS}
    assert chooser._buttons["zeiss_czi"].text() == "Test Zeiss CZI import"
    assert chooser._buttons["nikon_nd2"].text() == "Test Nikon ND2 import"


def test_the_buttons_are_laid_out_in_a_grid_not_one_row(chooser):
    from PySide6.QtWidgets import QGridLayout

    grids = chooser.findChildren(QGridLayout)
    assert grids and grids[0].columnCount() == chooser.COLUMNS


def test_hovering_a_variant_describes_its_format_and_naming(chooser):
    button = chooser._buttons["opera_phenix"]
    chooser.eventFilter(button, QEvent(QEvent.Enter))
    text = chooser.description_text()
    assert "r05c01" in text and "opera_phenix" in text
    chooser.eventFilter(button, QEvent(QEvent.Leave))
    assert chooser.description_text() == chooser.RESTING_TEXT
    assert "285 MB" in chooser.RESTING_TEXT


def test_the_import_screen_has_the_button_and_every_convention(screen):
    assert screen._btn_test_data.text() == "Load test data…"
    offered = {screen._naming_box.itemData(i)
               for i in range(screen._naming_box.count())}
    from spacr.regex_infer import _metadata_convention_keys
    assert set(_metadata_convention_keys()) <= offered


def test_choosing_zeiss_downloads_fills_and_previews(screen, tmp_path):
    download = _Download(keys=["zeiss_czi"])
    assert demo.load_import_test_data(screen, "zeiss_czi", ask=download,
                                      plate=tmp_path) is False
    assert download.calls == [tmp_path]
    root = ix.import_example_folder(tmp_path)
    assert screen.images_path() == str(root / "variants" / "zeiss_czi"
                                       / "plate1")
    assert set(screen.mask_folders()) == {"cell", "nucleus", "pathogen"}
    assert screen.measurements_path().endswith("measurements.db")
    assert screen.metadata_type() == "zeiss_zen_split_tiles"
    plan = screen.plan()
    assert plan is not None and plan.ok, screen.report_text()
    assert sorted(plan.stems) == ["plate1_E01_10", "plate1_E01_9",
                                  "plate1_E02_10", "plate1_E02_9"]
    assert plan.join.rows_matched == plan.join.rows_total > 0
    assert screen.can_import()
    assert screen._btn_test_data.isEnabled()


def test_a_second_variant_opens_from_the_cache(screen, tmp_path):
    _build_into(tmp_path, keys=["arrayscan"])
    download = _Download()
    assert demo.load_import_test_data(screen, "arrayscan", ask=download,
                                      plate=tmp_path) is True
    assert download.calls == []
    assert screen.metadata_type() == "arrayscan"


def test_custom_fills_its_pattern(screen, tmp_path):
    _build_into(tmp_path, keys=["custom"])
    assert demo.load_import_test_data(screen, "custom", plate=tmp_path)
    assert screen.custom_regex() == ix.import_variant("custom").custom_regex
    assert screen.plan() is not None and screen.plan().ok


def test_loading_again_does_not_inherit_the_last_masks_or_project(
        screen, tmp_path):
    _build_into(tmp_path, keys=["cellvoyager", "cq1"])
    demo.load_import_test_data(screen, "cellvoyager", plate=tmp_path)
    first = screen.destination_path()
    Path(first).mkdir(parents=True)
    demo.load_import_test_data(screen, "cq1", plate=tmp_path)
    assert all("cq1" in folder for folder in screen.mask_folders().values())
    assert screen.destination_path() != first


def test_the_chosen_variant_is_what_loads(screen, tmp_path):
    _build_into(tmp_path, keys=["incell"])

    class Picked:
        chosen = "incell"

        def exec(self):
            return 1

    assert demo.choose_import_test_data(screen, chooser=Picked(),
                                        plate=tmp_path)
    assert screen.metadata_type() == "incell"


def test_closing_the_chooser_loads_nothing(screen, tmp_path):
    class Closed:
        chosen = ""

        def exec(self):
            return 0

    download = _Download()
    assert not demo.choose_import_test_data(screen, chooser=Closed(),
                                            ask=download, plate=tmp_path)
    assert download.calls == []


def test_a_failed_download_says_why_and_gives_the_button_back(
        screen, tmp_path):
    demo.load_import_test_data(screen, "zeiss_czi",
                               ask=_Download(error="network is unreachable"),
                               plate=tmp_path)
    assert "network is unreachable" in screen.status_text()
    assert screen.last_error
    assert screen._btn_test_data.isEnabled()
    assert screen._btn_test_data.text() == "Load test data…"


def test_a_cancel_is_not_reported_as_a_failure(screen, tmp_path):
    demo.load_import_test_data(screen, "zeiss_czi",
                               ask=_Download(error=demo.CANCELLED),
                               plate=tmp_path)
    assert "cancelled" in screen.status_text()
    assert not screen.last_error


def test_a_download_that_cannot_start_is_reported(screen, tmp_path):
    def _boom(parent, plate, on_done):
        raise ConnectionError("no route to host")

    demo.load_import_test_data(screen, "zeiss_czi", ask=_boom,
                               plate=tmp_path)
    assert screen.last_error


def test_an_incomplete_copy_is_refused_not_opened(screen, tmp_path):
    root = _build_into(tmp_path, keys=["cellvoyager"])
    victim = next((root / "variants" / "cellvoyager" / "plate1").glob("*"))
    victim.unlink()
    assert not demo.apply_variant(screen, root, "cellvoyager")


def test_an_images_only_sample_goes_to_the_format_converter(
        screen, tmp_path, monkeypatch, qtbot):
    import tifffile

    from spacr.qt.screens.convert import ConvertScreen

    samples = tmp_path / "samples"
    samples.mkdir()
    tifffile.imwrite(str(samples / "one.tif"), np.zeros((8, 8), np.uint16))
    monkeypatch.setattr(builder, "SAMPLES", {"nikon_nd2": [
        ("one.tif", "A01/WellA01_ChannelBF_Seq0001.tif", "test")]})
    root = ix.import_example_folder(tmp_path / "plate")
    builder.build(builder.synthetic_fields(), root, samples=samples,
                  keys=["cellvoyager"])
    converter = ConvertScreen(threaded=False)
    qtbot.addWidget(converter)
    monkeypatch.setattr(demo, "_converter_for", lambda _screen: converter)
    assert demo.load_import_test_data(screen, "nikon_nd2",
                                      plate=tmp_path / "plate")
    assert converter.source_path().endswith("nikon_nd2/plate1")
    assert converter.preview_row_count() == 1


def test_the_worker_fetches_the_import_archive():
    assert demo._ImportTarWorker.repo == "einarolafsson/spacr-example-import"
    assert demo._ImportTarWorker.archive == "spacr-example-import.tar"
    assert demo._ImportTarWorker(Path("/x")).dataset_root("/x") == Path(
        "/x/import_example")


def test_the_set_is_registered_with_the_other_example_sets():
    from spacr.example_archives import EXAMPLE_ARCHIVES, example_set

    entry = example_set("import")
    assert entry.repo == ix.IMPORT_EXAMPLE_REPO
    assert entry.archive == ix.IMPORT_EXAMPLE_ARCHIVE
    assert EXAMPLE_ARCHIVES[ix.IMPORT_EXAMPLE_REPO] == entry.archive
    assert entry.markers == ("import_example/manifest.csv",)


def test_custom_naming_with_no_pattern_is_refused_inline(screen, tmp_path):
    _build_into(tmp_path, keys=["custom"])
    inputs = ix.variant_inputs(ix.import_example_folder(tmp_path), "custom")
    screen.set_images(inputs["images"])
    for role, folder in inputs["masks"].items():
        screen.add_mask_folder(role, folder)
    screen.set_measurements(inputs["measurements"])
    screen.set_metadata_type("custom")
    assert screen._regex_edit.isEnabled()
    assert screen.preview() is False
    assert "pattern" in screen.status_text()


def test_an_unknown_convention_is_refused_inline(screen):
    assert screen.set_metadata_type("zeiss") is False
    assert screen.last_error
