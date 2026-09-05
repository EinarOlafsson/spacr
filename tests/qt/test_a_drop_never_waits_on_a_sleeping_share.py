"""Dropping a path never stat-s it on the GUI thread.

THE FREEZE, 2026-09-04. `install_dropzone` puts a `_DropzoneFilter` on the
AppScreen itself, so `QEvent.Drop` is delivered by Qt's event dispatch and
everything it calls runs ON THE GUI THREAD. `_on_drop` then asked the
handler, inline:

    handler.can_accept(p)
      -> MaskDropHandler.can_accept   -> Path.is_dir() + has_images_in()
                                                        -> Path.iterdir()
      -> MapBarcodesDropHandler.can_accept -> Path.is_file() / is_dir()
                                                        -> Path.iterdir()

and on a rejection went on to `suggest_alternatives`, which walked the
parent and every child looking for images.

The maintainer's sequencing folder lives under ``/nas_mnt``, an ``autofs``
mount with ``timeout=600``. Measured on that machine: a single
``os.path.exists`` on a path there had NOT RETURNED AFTER TWENTY SECONDS,
because the stat is what triggers the automount and the share was asleep.
Dropping that folder on Map Barcodes therefore froze the whole window with
no traceback -- a stalled event loop is not a crash -- and was reported as
"opening map barcodes crashes spacr", plus hover flicker and glimpses of
other module screens repainting late.

The folder scans behind ``apply`` had already been moved to a worker. The
accept tests had not, because they have no callback to report into: the
boolean has to come back before the drop can be routed. `_decide` is how
they come back anyway -- the walk happens on a thread, the GUI thread waits
`DECISION_BUDGET_S` for it and no longer, and a share that is still waking
up gets the optimistic answer while the worker behind ``apply`` reports
what is actually there.

MEASURED AFTER: every call below returns in a few milliseconds with the
same filesystem that used to hold it for twenty seconds.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLineEdit, QWidget

from spacr.qt import dnd_handlers as dh
from spacr.qt.dnd_handlers import MapBarcodesDropHandler, MaskDropHandler

#: Longer than any human calls responsive, shorter than the twenty seconds
#: actually measured -- a test that waited the real duration is a test
#: nobody runs.
SLOW_S = 8.0

#: What "did not wait" means here. The budget is 0.25 s, so anything under a
#: second proves the sleeping filesystem was not waited out; the margin is
#: for a loaded CI box, not for a second stat.
FAST_S = 1.0


@pytest.fixture(autouse=True)
def _fresh_decisions():
    dh.forget_decisions()
    yield
    dh.forget_decisions()


@pytest.fixture(autouse=True)
def _drain_folder_scans(monkeypatch):
    """No test may walk away from a running folder scan.

    Qt aborts the process when a running QThread is destroyed with the object
    that owns it, and half of what is asserted below is that a drop hands its
    filesystem work to exactly such a thread.
    """
    made = []
    real_init = dh._DropScanner.__init__

    def _spy(self, screen):
        real_init(self, screen)
        made.append(self)

    monkeypatch.setattr(dh._DropScanner, "__init__", _spy)
    yield made
    for scanner in made:
        try:
            scanner.shutdown()
        except Exception:
            pass


class _SleepingPath:
    """A path from a share that is asleep: every question takes SLOW_S.

    Deliberately not a `Path` subclass. The point of the test is that the
    handler asks this object nothing on the calling thread, and a real
    `Path` cannot be made to prove that.
    """

    def __init__(self, text: str, asked: list):
        self._text = str(text)
        self.asked = asked

    def __str__(self) -> str:
        return self._text

    @property
    def name(self) -> str:
        return self._text.rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        name = self.name
        return "." + name.rsplit(".", 1)[-1] if "." in name else ""

    @property
    def parent(self) -> "_SleepingPath":
        return _SleepingPath(self._text.rsplit("/", 1)[0] or "/", self.asked)

    def _stall(self, what: str):
        self.asked.append(what)
        time.sleep(SLOW_S)

    def is_dir(self) -> bool:
        self._stall("is_dir")
        return True

    def is_file(self) -> bool:
        self._stall("is_file")
        return False

    def iterdir(self):
        self._stall("iterdir")
        return iter(())


class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Model:
    def __init__(self, widgets):
        self._widgets = dict(widgets)


class _Screen(QWidget):
    """AppScreen-shaped double: a settings model of real widgets + console."""

    def __init__(self, keys=("src",)):
        super().__init__()
        self._settings_model = _Model({k: QLineEdit() for k in keys})
        self._console = _Console()

    def w(self, key):
        return self._settings_model._widgets[key]


@pytest.fixture
def screen(qtbot):
    s = _Screen(keys=("src", "custom_regex", "fastq"))
    qtbot.addWidget(s)
    return s


def _mkimg(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(p), np.zeros((4, 5), np.uint16))
    return p


def _timed(fn):
    started = time.monotonic()
    value = fn()
    return value, time.monotonic() - started


# ---------------------------------------------------------------------------
# Mask -- the accept test and the suggestion walk behind it
# ---------------------------------------------------------------------------

def test_mask_can_accept_answers_before_the_share_wakes_up():
    """The call `_on_drop` makes inline, against a path that never answers."""
    asked: list = []
    answer, elapsed = _timed(
        lambda: MaskDropHandler().can_accept(_SleepingPath("/nas_mnt/plate1",
                                                           asked)))

    assert elapsed < FAST_S, (
        f"can_accept took {elapsed:.1f} s -- it is stat-ing the dropped path "
        "on the GUI thread again")
    assert answer is True, "an undecided drop is accepted, not refused"
    assert asked, "the scan should still have been started, on a thread"


def test_mask_suggestions_do_not_walk_the_neighbourhood_inline(tmp_path,
                                                               monkeypatch):
    """A rejected drop used to walk the parent and every child, inline."""
    def asleep(*_a, **_k):
        time.sleep(SLOW_S)
        raise AssertionError("the GUI thread waited for the image walk")

    monkeypatch.setattr(dh, "has_images_in", asleep)
    monkeypatch.setattr(dh, "find_image_folders_nearby", asleep)
    folder = tmp_path / "plate1"
    folder.mkdir()

    handler = MaskDropHandler()
    _, accept_s = _timed(lambda: handler.can_accept(folder))
    _, suggest_s = _timed(lambda: handler.suggest_alternatives(folder))

    assert accept_s < FAST_S, f"can_accept took {accept_s:.1f} s"
    assert suggest_s < FAST_S, f"suggest_alternatives took {suggest_s:.1f} s"


def test_mask_apply_does_not_wait_for_the_decision(screen, tmp_path,
                                                   monkeypatch):
    """`apply` re-asked is_file/is_dir after can_accept had already asked.

    The share is released at the end rather than left asleep for SLOW_S: an
    undecided drop now hands the question it could not answer to the drop
    scanner, and a QThread still running when Qt destroys the widget that
    owns it aborts the process rather than failing the test.
    """
    woke = threading.Event()

    def asleep(_path):
        assert not woke.wait(SLOW_S), "the share was released early"
        raise AssertionError("the GUI thread waited for the drop decision")

    monkeypatch.setattr(dh, "scan_mask_drop", asleep)
    folder = tmp_path / "plate1"
    _mkimg(folder / "plate1_A01_T0001F001L01A01Z01C01.tif")

    try:
        _, elapsed = _timed(lambda: MaskDropHandler().apply(folder, screen))

        assert elapsed < FAST_S, f"apply took {elapsed:.1f} s"
        assert screen.w("src").text() == str(folder)
    finally:
        woke.set()


# ---------------------------------------------------------------------------
# Map Barcodes -- the drop the maintainer actually reported
# ---------------------------------------------------------------------------

def test_map_barcodes_accepts_a_fastq_by_name_alone():
    """The common drop costs no filesystem call whatsoever."""
    asked: list = []
    path = _SleepingPath("/nas_mnt/data/sequencing/seq_3/reads.fastq.gz",
                         asked)

    answer, elapsed = _timed(lambda: MapBarcodesDropHandler().can_accept(path))

    assert answer is True
    assert asked == [], (
        f"the filesystem was asked {asked} for a question the filename had "
        "already answered")
    assert elapsed < FAST_S


def test_map_barcodes_can_accept_a_folder_that_never_lists():
    """`/nas_mnt/data/sequencing/seq_3` itself: the gesture that froze it."""
    asked: list = []
    answer, elapsed = _timed(
        lambda: MapBarcodesDropHandler().can_accept(
            _SleepingPath("/nas_mnt/data/sequencing/seq_3", asked)))

    assert elapsed < FAST_S, (
        f"can_accept took {elapsed:.1f} s -- it is listing the dropped "
        "folder on the GUI thread again")
    assert answer is True


def test_map_barcodes_apply_touches_the_filesystem_not_at_all(screen):
    """Both halves of the drop are decided by the name."""
    asked: list = []
    path = _SleepingPath("/nas_mnt/data/sequencing/seq_3/reads.fq.gz", asked)

    _, elapsed = _timed(lambda: MapBarcodesDropHandler().apply(path, screen))

    assert asked == [], f"apply stat-ed the dropped path: {asked}"
    assert elapsed < FAST_S
    assert screen.w("src").text() == "/nas_mnt/data/sequencing/seq_3"
    assert screen.w("fastq").text() == str(path)


# ---------------------------------------------------------------------------
# ...and the answers are still the answers, on a disk that can give them
# ---------------------------------------------------------------------------

def test_a_disk_that_answers_is_decided_exactly_as_before(tmp_path):
    """Not waiting is not guessing. A filesystem that replies is obeyed."""
    with_images = tmp_path / "plate1"
    _mkimg(with_images / "plate1_A01_T0001F001L01A01Z01C01.tif")
    empty = tmp_path / "notes"
    empty.mkdir()
    (empty / "readme.txt").write_text("hi")

    mask = MaskDropHandler()
    assert mask.can_accept(with_images) is True
    assert mask.can_accept(empty) is False
    assert mask.can_accept(tmp_path / "gone") is False
    assert mask.suggest_alternatives(empty) == [with_images]

    barcodes = MapBarcodesDropHandler()
    (with_images / "reads.fastq").write_bytes(b"@id\nACGT\n+\nIIII\n")
    dh.forget_decisions()
    assert barcodes.can_accept(with_images) is True
    assert barcodes.can_accept(empty) is False
    assert barcodes.can_accept(tmp_path / "gone") is False


def test_a_slow_answer_is_still_remembered_once_it_lands(tmp_path,
                                                         monkeypatch):
    """Optimism is temporary: the truth is cached when it finally arrives."""
    calls = {"n": 0}
    real = dh.has_images_in

    def slow_once(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(dh.DECISION_BUDGET_S * 4)
        return real(path, *a, **k)

    monkeypatch.setattr(dh, "has_images_in", slow_once)
    empty = tmp_path / "notes"
    empty.mkdir()

    handler = MaskDropHandler()
    assert handler.can_accept(empty) is True          # optimistic, no wait

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and handler.can_accept(empty) is True:
        time.sleep(0.02)
    assert handler.can_accept(empty) is False, "the scan never landed"


# ---------------------------------------------------------------------------
# The sibling sites in the same flow
# ---------------------------------------------------------------------------

def test_a_container_drop_never_opens_the_file_on_the_gui_thread(
        screen, tmp_path, monkeypatch, qtbot):
    """The branch one step past the one that was fixed.

    ``apply`` decided folder-or-container without stat-ing, and then handed
    the container branch to ``QTimer.singleShot(0, ...)`` under the note that
    reading a header "stays on the GUI thread". A single-shot timer does not
    leave the GUI thread; it runs its callback on it one turn later with the
    event loop stopped. And a header read is a FILE OPEN -- the very call
    that had not returned after twenty seconds on ``/nas_mnt``.
    """
    where = []
    monkeypatch.setattr(dh, "_open_metadata_table",
                        lambda rows, dst, screen: None)
    real = dh.scan_mask_container

    def watched(path):
        where.append(threading.current_thread())
        return real(path)

    monkeypatch.setattr(dh, "scan_mask_container", watched)
    stack = tmp_path / "plate.tif"
    tifffile.imwrite(str(stack), np.zeros((3, 8, 9), np.uint16))

    _, elapsed = _timed(lambda: MaskDropHandler().apply(stack, screen))

    assert elapsed < FAST_S
    qtbot.waitUntil(
        lambda: "single-file dataset" in screen._console.text, timeout=10000)
    assert where and where[0] is not threading.main_thread(), (
        "the container header was read on the GUI thread")
    assert screen.w("src").text() == str(tmp_path), (
        "a container's src is the folder holding it, not the file")


def test_an_undecided_drop_does_not_read_a_container_as_a_folder(
        screen, tmp_path, monkeypatch, qtbot):
    """Accepting a guess is free. ROUTING on one is not.

    When the budget runs out, the record `_decide` hands back says "folder,
    acceptable" -- which is the right thing to ACCEPT and the wrong thing to
    act on. Acting on it put the container FILE in ``src`` where its parent
    belongs, skipped the header read that sets ``metadata_type = auto``, and
    finished by printing "no images found in the top level of plate.tif".
    """
    monkeypatch.setattr(dh, "_open_metadata_table",
                        lambda rows, dst, screen: None)
    stack = tmp_path / "plate.tif"
    tifffile.imwrite(str(stack), np.zeros((3, 8, 9), np.uint16))

    real = dh.scan_mask_drop

    def slow(path):
        time.sleep(dh.DECISION_BUDGET_S * 3)
        return real(path)

    monkeypatch.setattr(dh, "scan_mask_drop", slow)

    _, elapsed = _timed(lambda: MaskDropHandler().apply(stack, screen))
    assert elapsed < FAST_S, f"apply took {elapsed:.1f} s"

    qtbot.waitUntil(
        lambda: "single-file dataset" in screen._console.text, timeout=15000)
    log = screen._console.text
    assert "no images found in the top level" not in log, (
        "the guess was acted on: a container was read as an empty folder")
    assert screen.w("src").text() == str(tmp_path)


def test_a_scan_that_raises_still_reports_to_the_console(
        screen, tmp_path, monkeypatch, qtbot):
    """`JobRunner` runs `on_done` only for a job that SUCCEEDED.

    So a scan that raised used to take the whole report with it: the drop had
    already written "[drop] mask src = ..." and nothing ever followed it.
    """
    def boom(path, *a, **k):
        raise OSError("the share went away")

    monkeypatch.setattr(dh, "scan_mask_folder", boom)
    folder = tmp_path / "plate1"
    _mkimg(folder / "plate1_A01_T0001F001L01A01Z01C01.tif")

    MaskDropHandler().apply(folder, screen)

    qtbot.waitUntil(
        lambda: "could not read" in screen._console.text, timeout=10000)
    assert "the share went away" in screen._console.text


def test_a_late_answer_never_overwrites_a_newer_one(tmp_path, monkeypatch):
    """A stat parked on a sleeping share outlives the drop that asked for it.

    Drop an empty folder: the GUI thread gives up after DECISION_BUDGET_S and
    that stat stays parked. Put the images in and drop it again: the second
    question answers "yes" and caches it -- and then the first one finally
    returns "no" and, unguarded, wrote it over the top. The next question got
    an empty folder that was not empty.
    """
    folder = tmp_path / "plate1"
    folder.mkdir()
    released = threading.Event()
    landed = threading.Event()
    asked = []

    def scan(path):
        asked.append(path)
        if len(asked) == 1:
            assert released.wait(10), "the first scan was never released"
            landed.set()
            return {"is_dir": True, "is_file": False,
                    "accepted": False, "alternatives": []}
        return {"is_dir": True, "is_file": False,
                "accepted": True, "alternatives": []}

    monkeypatch.setattr(dh, "scan_mask_drop", scan)
    handler = MaskDropHandler()

    assert handler.can_accept(folder) is True, "the first answer is the guess"
    assert handler.can_accept(folder) is True, "the second one is real"

    released.set()
    assert landed.wait(10), "the parked scan never came back"
    time.sleep(0.2)                      # give it time to do the damage

    assert handler.can_accept(folder) is True, (
        "the stat still parked from the first drop overwrote the answer the "
        "second drop had already cached")
    assert len(asked) == 2, (
        "the answer above came from a third scan, not from the cache -- this "
        "test proves nothing unless it is a cache hit")


def test_the_drop_scanner_never_claims_a_run_banner(screen):
    """A drop is a gesture, not a run.

    `home._on_runs_changed` filters its blue "<module> - running" banners on
    `user_visible`, so a scanner that claimed one made every drop on every
    screen flash "folder scan - running" across the top of Home -- the same
    mistake the usage poller and the home-journal walk each made once.
    """
    scanner = dh._scanner_for(screen)
    assert scanner is not None
    assert scanner._runner._user_visible is False
