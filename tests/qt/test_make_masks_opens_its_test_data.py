"""Make Masks' "Load test data…" downloads ten fields and opens the first.

Ledger item 412. The maintainer's words: "add a button for loading this
dataset which automatically triggers opening the first image with make masks".

NO NETWORK ANYWHERE IN HERE. The download is replaced through the `ask` seam
the other example buttons use, or, for the tests of the real worker and its
progress dialog, `requests.get` is replaced by a local archive served in
chunks. What is proved is the sequence the button owns: the cache check, the
button's state while the download runs, the folder opening on its first image
BY NAME, a second press that fetches nothing, and a failure that says why.
"""
from __future__ import annotations

import csv
import tarfile
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QThread

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


def _edit_manifest(folder: Path, index: int, **fields) -> None:
    """Rewrite one manifest row, the way a hand-edited or damaged copy reads."""
    manifest = folder / "manifest.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[index].update(fields)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _archive_of(tmp_path: Path, leave_out=()) -> bytes:
    """The published archive's bytes: the dataset, manifest as LAST member.

    ``leave_out`` names members (relative paths) to omit, which is what an
    archive that lost a file on its way to the repository looks like.
    """
    source = tmp_path / "source"
    source.mkdir()
    _write_dataset(source)
    archive = tmp_path / demo.MAKE_MASKS_EXAMPLE_ARCHIVE
    with tarfile.open(archive, "w") as tar:
        for path in sorted(source.rglob("*")):
            name = path.relative_to(source).as_posix()
            if path.name != "manifest.csv" and name not in leave_out:
                tar.add(path, arcname=name, recursive=False)
        tar.add(source / "manifest.csv", arcname="manifest.csv")
    return archive.read_bytes()


class _Hub:
    """`requests.get` serving one archive in chunks, with no network.

    ``hold`` keeps the worker waiting before its first chunk until
    :attr:`release` is set, so a test can look at the screen, or press
    Cancel, while the download is genuinely in flight. ``error`` is raised
    by the request itself, the way an offline machine fails.
    """

    def __init__(self, monkeypatch, payload=b"", *, hold=False, error=None):
        import requests

        self.requested = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        hub = self

        class _Response:
            headers = {"Content-Length": str(len(payload))}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                hub.started.set()
                assert hub.release.wait(20), "the test never released it"
                for start in range(0, len(payload), chunk_size):
                    yield payload[start:start + chunk_size]

        def fake_get(url, **_kwargs):
            hub.requested.append(url)
            if error is not None:
                raise error
            return _Response()

        monkeypatch.setattr(requests, "get", fake_get)


@pytest.fixture
def screen(qtbot):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    yield widget
    # A real download runs on a QThread parented to the screen, and a QThread
    # destroyed while running aborts the process. The wait outlasts `_Hub`'s
    # own 20-second hold, so even a test that failed mid-transfer is joined.
    thread = getattr(widget, "_hf_download_thread", None)
    if isinstance(thread, QThread) and thread.isRunning():
        thread.quit()
        thread.wait(30000)


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
    hub = _Hub(monkeypatch, _archive_of(tmp_path))
    dest = tmp_path / "dest"
    worker = demo._MakeMasksTarWorker(dest)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))

    worker.run()

    assert hub.requested == [
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


def test_a_manifest_row_that_names_no_image_makes_the_folder_unrecognised(
        tmp_path):
    """A row with no image cannot say which field it describes. The folder is
    therefore not taken for the dataset even with every file in place, and
    that answer fetches a fresh copy instead of opening a damaged one."""
    _write_dataset(tmp_path)
    _edit_manifest(tmp_path, 1, image="   ")

    assert demo.listed_files(tmp_path) == []
    assert not demo.is_present(tmp_path)


def test_a_row_without_a_ground_truth_mask_is_complete_with_its_image_alone(
        tmp_path):
    """A mask is required only where the manifest names one. A field published
    without curated truth must not make the set read as absent, or every
    press would download it again."""
    _write_dataset(tmp_path)
    _edit_manifest(tmp_path, 2, ground_truth_mask="")
    (tmp_path / "ground_truth_masks" / "plate3_B18_17.tif").unlink()

    listed = demo.listed_files(tmp_path)
    assert listed[-1] == "plate3_B18_17.tif"
    assert "ground_truth_masks/plate3_B18_17.tif" not in listed
    assert len(listed) == 2 * len(_NAMES) - 1
    assert demo.is_present(tmp_path)


class _BareScreen:
    """A screen that never installed the button and has no status line.

    `load_the_test_data` reads both through `getattr`, so anything that can
    open a folder and show a warning can load the data.
    """

    def __init__(self):
        self.opened = []
        self.warnings = []

    def _open_folder(self, folder):
        self.opened.append(folder)
        return True

    def _warn(self, title, text):
        self.warnings.append((title, text))


def test_a_screen_without_the_button_or_a_status_line_still_loads(tmp_path):
    """With no button to disable and no status line to write on, the data
    still downloads and opens, and the next call opens the cache."""
    bare = _BareScreen()
    folder = tmp_path / "data"
    fetched = []

    def download(_screen, dest, on_done):
        fetched.append(Path(dest))
        _write_dataset(Path(dest))
        on_done(DownloadResult(dataset_path=Path(dest),
                               settings_path=Path(dest)), "")

    assert demo.load_the_test_data(bare, ask=download, folder=folder) is False
    assert bare.opened == [str(folder)]
    assert demo.load_the_test_data(bare, ask=download, folder=folder) is True
    assert fetched == [folder], "the cached copy was fetched again"
    assert bare.warnings == []


def test_a_screen_without_a_status_line_warns_on_failure_but_not_on_cancel(
        tmp_path):
    """A cancel is reported only on the status line, so a screen without one
    shows nothing. A failure still reaches the user, as a warning."""
    bare = _BareScreen()

    def cancelled(_screen, _dest, on_done):
        on_done(None, demo.CANCELLED)

    def failed(_screen, _dest, on_done):
        on_done(None, "the disk is full")

    demo.load_the_test_data(bare, ask=cancelled, folder=tmp_path / "x")
    assert bare.warnings == []
    assert bare.opened == []

    demo.load_the_test_data(bare, ask=failed, folder=tmp_path / "x")
    assert bare.warnings == [(
        "Test data not loaded",
        "The test data could not be downloaded: the disk is full")]
    assert bare.opened == []


def test_the_real_download_keeps_the_button_busy_then_opens_the_first_image(
        screen, tmp_path, monkeypatch, qtbot):
    """The button's whole path, with only the transport replaced.

    It runs through `download_make_masks_example`'s progress dialog and
    QThread, the tar worker, and the screen. While the archive is still
    arriving, the button is disabled and says so, and pressing it starts
    nothing. Once the archive lands, the first image by name is on the
    canvas. The next press opens the cache without a request.
    """
    hub = _Hub(monkeypatch, _archive_of(tmp_path), hold=True)
    folder = tmp_path / "cache" / "make_masks_toxo_pv"
    monkeypatch.setattr(demo, "make_masks_example_folder", lambda: folder)
    button = screen._btn_test_data

    button.click()
    qtbot.waitUntil(hub.started.is_set, timeout=10000)

    assert not button.isEnabled()
    assert button.text() == "Fetching test data…"
    assert screen._status_label.text() == (
        "Downloading the Make Masks test data")
    assert screen._hf_download_dialog.windowTitle() == (
        "Downloading the Make Masks test data")
    button.click()
    assert len(hub.requested) == 1, "a press while fetching started another"

    hub.release.set()
    qtbot.waitUntil(lambda: screen._folder != "", timeout=20000)

    assert button.isEnabled()
    assert button.text() == "Load test data…"
    assert Path(screen._folder) == folder
    assert screen._image_files[screen._current_index] == _FIRST_BY_NAME
    assert not hasattr(screen, "_hf_download_dialog"), (
        "the progress dialog outlived the download")

    screen._open_folder(str(tmp_path / "source"))    # somewhere else
    button.click()
    assert len(hub.requested) == 1, "the cached copy was fetched again"
    assert Path(screen._folder) == folder


def test_cancel_in_the_real_progress_dialog_is_not_reported_as_a_failure(
        screen, tmp_path, monkeypatch, qtbot):
    """The real dialog's Cancel, pressed mid-transfer.

    The worker's own cancel text has to be the one `CANCELLED` names.
    Otherwise a cancel the user asked for arrives as "could not be
    downloaded", and the wait below never ends. Nothing is opened, and no
    partial file is left behind.
    """
    hub = _Hub(monkeypatch, _archive_of(tmp_path), hold=True)
    folder = tmp_path / "cache"

    assert demo.load_the_test_data(screen, folder=folder) is False
    qtbot.waitUntil(hub.started.is_set, timeout=10000)
    screen._hf_download_dialog._cancel.click()
    hub.release.set()

    cancelled = "The test data download was cancelled."
    qtbot.waitUntil(lambda: screen._status_label.text() == cancelled,
                    timeout=20000)

    assert screen._folder == ""
    assert screen._btn_test_data.isEnabled()
    assert list(folder.iterdir()) == [], "the partial download was kept"
    assert not hasattr(screen, "_hf_download_thread")


def test_offline_the_real_download_says_why_on_the_screen(screen, tmp_path,
                                                          monkeypatch, qtbot):
    """The worker's explanation reaches the screen unchanged, so the user
    reads that huggingface.co was unreachable, not a transport error."""
    failure = ConnectionError("Network is unreachable")
    hub = _Hub(monkeypatch, error=failure)

    demo.load_the_test_data(screen, folder=tmp_path / "cache")
    qtbot.waitUntil(lambda: screen._btn_test_data.isEnabled(), timeout=20000)

    assert len(hub.requested) == 1
    assert screen._status_label.text() == (
        "Test data not loaded: The test data could not be downloaded: "
        + explain_download_failure(failure))
    assert screen._folder == ""


def test_an_archive_missing_a_mask_is_reported_as_incomplete_not_opened(
        screen, tmp_path, monkeypatch, qtbot):
    """The transfer succeeds, but the archive lacks one file its manifest
    lists. The other fields would pass for the whole set, so the screen says
    the copy is incomplete and opens none of it."""
    missing = "ground_truth_masks/plate3_B18_17.tif"
    _Hub(monkeypatch, _archive_of(tmp_path, leave_out={missing}))
    folder = tmp_path / "cache"

    demo.load_the_test_data(screen, folder=folder)
    qtbot.waitUntil(lambda: "incomplete" in screen._status_label.text(),
                    timeout=20000)

    assert screen._status_label.text() == (
        f"Test data not loaded: The downloaded test data is incomplete: "
        f"{folder}")
    assert (folder / "manifest.csv").is_file()
    assert (folder / "plate3_B18_17.tif").is_file()
    assert not (folder / missing).exists()
    assert screen._folder == ""
    assert screen._btn_test_data.isEnabled()
