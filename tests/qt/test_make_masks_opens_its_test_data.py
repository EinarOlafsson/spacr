"""Make Masks' "Load test data…" downloads ten fields and opens the first.

Ledger item 412. The maintainer's words: "add a button for loading this
dataset which automatically triggers opening the first image with make masks".

NO NETWORK ANYWHERE IN HERE. The download is replaced through the `ask` seam
the other example buttons use, or, for the one test of the real worker,
`requests.get` is replaced by a local archive served in chunks. What is proved
is the sequence the button owns: the cache check, the button's state while the
download runs, the folder opening on its first image BY NAME, a second press
that fetches nothing, and a failure that says why.
"""
from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import make_masks_demo as demo
from spacr.qt.hf_download import DownloadResult, explain_download_failure
from spacr.qt.screens.make_masks import MakeMasksScreen

#: Written OUT of name order, so a screen that opened the first file written,
#: or the first the manifest lists, would show the wrong one.
_NAMES = ("plate7_N21_6", "plate1_A02_12", "plate3_B18_17")
_FIRST_BY_NAME = "plate1_A02_12.tif"


def _pixels(index: int) -> np.ndarray:
    """A small uint16 field that differs from every other one."""
    rng = np.random.default_rng(412 + index)
    return rng.integers(0, 65535, (40, 48), dtype=np.uint16)


def _write_dataset(folder: Path) -> None:
    """Lay the dataset out as the archive unpacks: images, truth, manifest."""
    (folder / "ground_truth_masks").mkdir(parents=True, exist_ok=True)
    (folder / "masks").mkdir(exist_ok=True)
    rows = []
    for index, stem in enumerate(_NAMES):
        imageio.imwrite(folder / f"{stem}.tif", _pixels(index))
        truth = np.zeros((40, 48), dtype=np.uint16)
        truth[5:15, 5:15] = 1
        imageio.imwrite(folder / "ground_truth_masks" / f"{stem}.tif", truth)
        rows.append({"name": stem, "host": "?", "dataset": "toxoplasma_pv",
                     "image": f"{stem}.tif",
                     "ground_truth_mask": f"ground_truth_masks/{stem}.tif"})
    with (folder / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class _FakeDownload:
    """Stands in for the Hugging Face download: counts calls, then finishes.

    ``outcome`` is ``"ok"`` to unpack the dataset, or an error message to
    report a failure the way the real worker does.
    """

    def __init__(self, outcome: str = "ok"):
        self.outcome = outcome
        self.calls = 0
        self.button_enabled_during = None
        self.button_text_during = None

    def __call__(self, screen, dest, on_done):
        self.calls += 1
        button = screen._btn_test_data
        self.button_enabled_during = button.isEnabled()
        self.button_text_during = button.text()
        if self.outcome == "ok":
            _write_dataset(Path(dest))
            on_done(DownloadResult(dataset_path=Path(dest),
                                   settings_path=Path(dest) / "settings"), "")
        else:
            on_done(None, self.outcome)


@pytest.fixture
def screen(qtbot):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    return widget


def test_the_button_is_on_the_make_masks_screen_and_says_what_it_loads(screen):
    button = screen._btn_test_data
    assert button.text() == "Load test data…"
    assert "Toxoplasma" in button.toolTip()
    assert "ground_truth_masks" in button.toolTip()
    assert screen._btn_open.parentWidget() is button.parentWidget(), (
        "the button is not in the row beside Open folder…")


def test_the_download_then_opens_the_first_image_by_name(screen, tmp_path):
    folder = tmp_path / "cache" / "make_masks_toxo_pv"
    fake = _FakeDownload()

    demo.load_the_test_data(screen, ask=fake, folder=folder)

    assert fake.calls == 1
    assert fake.button_enabled_during is False, (
        "the button stayed pressable while its download ran")
    assert fake.button_text_during == "Fetching test data…"
    assert screen._btn_test_data.isEnabled()
    assert screen._btn_test_data.text() == "Load test data…"

    assert Path(screen._folder) == folder
    assert screen._image_files == sorted(f"{n}.tif" for n in _NAMES)
    assert screen._image_files[screen._current_index] == _FIRST_BY_NAME
    first = _pixels(_NAMES.index("plate1_A02_12"))
    assert np.array_equal(screen._canvas.image, first), (
        "the field on the canvas is not the first image by name")
    assert screen._body_stack.currentWidget() is screen._body_splitter


def test_the_ground_truth_is_not_offered_as_a_field_and_not_loaded(screen,
                                                                    tmp_path):
    folder = tmp_path / "data"
    demo.load_the_test_data(screen, ask=_FakeDownload(), folder=folder)

    assert all("/" not in name for name in screen._image_files)
    assert not np.any(screen._canvas.mask), (
        "the editor opened on the ground truth instead of an empty draft")


def test_a_second_press_opens_from_the_cache_without_downloading(screen,
                                                                 tmp_path):
    folder = tmp_path / "data"
    fake = _FakeDownload()
    assert demo.load_the_test_data(screen, ask=fake, folder=folder) is False

    screen._open_folder(str(tmp_path))          # somewhere else in between
    assert demo.load_the_test_data(screen, ask=fake, folder=folder) is True

    assert fake.calls == 1, "the second press downloaded again"
    assert Path(screen._folder) == folder
    assert screen._image_files[screen._current_index] == _FIRST_BY_NAME


def test_the_real_button_click_goes_through_the_same_load(screen, tmp_path,
                                                          monkeypatch):
    fake = _FakeDownload()
    folder = tmp_path / "clicked"
    monkeypatch.setattr(demo, "download_make_masks_example", fake)
    monkeypatch.setattr(demo, "make_masks_example_folder", lambda: folder)

    screen._btn_test_data.click()

    assert fake.calls == 1
    assert screen._image_files[screen._current_index] == _FIRST_BY_NAME


def test_offline_it_says_why_and_opens_nothing(screen, tmp_path):
    reason = explain_download_failure(ConnectionError("Network is unreachable"))
    fake = _FakeDownload(outcome=reason)

    opened = demo.load_the_test_data(screen, ask=fake, folder=tmp_path / "x")

    assert opened is False
    status = screen._status_label.text()
    assert "Test data not loaded" in status
    assert "Could not reach huggingface.co" in status
    assert screen._folder == ""
    assert screen._btn_test_data.isEnabled(), (
        "a failed download left the button disabled for good")


def test_a_download_that_cannot_start_is_reported_not_raised(screen, tmp_path):
    def broken(_screen, _dest, _on_done):
        raise PermissionError("read-only cache")

    assert demo.load_the_test_data(screen, ask=broken,
                                   folder=tmp_path / "x") is False
    assert "read-only cache" in screen._status_label.text()
    assert screen._btn_test_data.isEnabled()


def test_a_cancel_is_not_reported_as_a_failure(screen, tmp_path):
    fake = _FakeDownload(outcome=demo.CANCELLED)
    demo.load_the_test_data(screen, ask=fake, folder=tmp_path / "x")
    assert screen._status_label.text() == (
        "The test data download was cancelled.")


def test_a_half_unpacked_copy_is_not_a_cache_hit(tmp_path):
    _write_dataset(tmp_path)
    assert demo.is_present(tmp_path)
    (tmp_path / "ground_truth_masks" / "plate3_B18_17.tif").unlink()
    assert not demo.is_present(tmp_path)
    (tmp_path / "manifest.csv").unlink()
    assert demo.listed_files(tmp_path) == []
    assert not demo.is_present(tmp_path)


def test_an_incomplete_download_is_reported_rather_than_opened(screen,
                                                               tmp_path):
    def unpacks_nothing(_screen, dest, on_done):
        on_done(DownloadResult(dataset_path=Path(dest),
                               settings_path=Path(dest)), "")

    assert demo.load_the_test_data(screen, ask=unpacks_nothing,
                                   folder=tmp_path / "x") is False
    assert "incomplete" in screen._status_label.text()
    assert screen._folder == ""


def test_the_worker_fetches_this_sets_own_archive_and_unpacks_it(tmp_path,
                                                                 monkeypatch):
    """The real worker, with `requests.get` serving a local archive.

    Proves the `archive` override reaches the URL and that the unpacked
    folder is one `is_present` accepts. The manifest goes in LAST, as the
    published archive has it.
    """
    source = tmp_path / "source"
    source.mkdir()
    _write_dataset(source)
    archive = tmp_path / demo.MAKE_MASKS_EXAMPLE_ARCHIVE
    with tarfile.open(archive, "w") as tar:
        for path in sorted(source.rglob("*")):
            if path.name != "manifest.csv":
                tar.add(path, arcname=str(path.relative_to(source)),
                        recursive=False)
        tar.add(source / "manifest.csv", arcname="manifest.csv")
    payload = archive.read_bytes()
    requested = []

    class _Response:
        headers = {"Content-Length": str(len(payload))}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            for start in range(0, len(payload), chunk_size):
                yield payload[start:start + chunk_size]

    import requests

    def fake_get(url, **_kwargs):
        requested.append(url)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "dest"
    worker = demo._MakeMasksTarWorker(dest)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))

    worker.run()

    assert requested == [
        "https://huggingface.co/datasets/einarolafsson/spacr-example-make-masks"
        "/resolve/main/spacr-example-make-masks.tar?download=true"]
    assert finished and finished[0][0] is True, finished
    assert demo.is_present(dest)
    assert not (dest / demo.MAKE_MASKS_EXAMPLE_ARCHIVE).exists(), (
        "the archive was kept after unpacking")


def test_the_cache_is_its_own_folder_not_the_shared_plate():
    """The Mask demo decides it is present by finding `*.tif` in the plate
    folder, so ten fields unpacked there would pass for that demo."""
    from spacr.qt.hf_download import example_plate_folder

    folder = demo.make_masks_example_folder()
    assert folder != example_plate_folder()
    assert example_plate_folder() not in folder.parents
