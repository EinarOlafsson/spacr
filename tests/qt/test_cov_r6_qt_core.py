"""``R6/C`` — the last unwalked turnings in eight top-level ``spacr.qt`` modules.

One file, eight modules, and every branch here is a *quiet* one: none of them
raises, so a regression in any of them looks exactly like success from the
outside. What is pinned:

* :mod:`spacr.qt.plate_queue` — ``PlateQueue.update`` writes the snapshot only
  when a field actually moved (an idempotent patch, and a patch naming a field
  the item does not have, must both leave the file alone); and
  ``import_plates_from_csv``'s coercion ladder falling all the way through, for
  a cell that is plain text and for a cell the CSV header promised but the row
  never supplied.
* :mod:`spacr.qt.curation_tool` — a ``BrushPanel`` handed a session adopts it
  instead of minting a second one over the same labels; ``save_mask`` treats a
  ``str`` as a path and a ``clicked`` bool as "no path"; and
  ``TrackCurationPanel.save`` with nowhere to write asks, then honours the
  answer.
* :mod:`spacr.qt.synthetic` — a field renders only the channels asked for while
  still building every mask, and an unpaired FASTQ run writes R1 alone.
* :mod:`spacr.qt.prefs` — a recent-source list stored as one newline-joined
  string, and the migration meeting a legacy namespace that *is* the canonical
  store.
* :mod:`spacr.qt.job_runner` — an unthreaded job with no completion callback
  still reports itself finished.
* :mod:`spacr.qt.crop_thumbs` — a cached value whose size cannot be measured
  costs zero bytes rather than throwing out of the memory sweep.
* :mod:`spacr.qt.command_palette` — Down at the last command stays put.

Two guards reached from here are dead, and are proved rather than driven:
``JobRunner._on_worker_error_text``'s loop can never take a second turn (see
``test_the_last_non_blank_line_is_the_first_one_looked_at``) and
``CommandPalette._render``'s auto-select loop can never run off the end (see
``test_a_rendered_list_always_holds_something_selectable``).

:mod:`spacr.qt.annotate_engine`'s two remaining arcs — ``parse_image_type``'s
empty-token guard and ``fetch_filtered_paths``'s re-check of the annotation
column it just created — are also unreachable, and were already proved in
``tests/qt/test_cov_r4_annotate_engine.py``
(``test_every_readable_filter_yields_at_least_one_token`` and
``test_the_annotation_column_survives_every_step_after_it_is_added``). They are
not duplicated here.

Offscreen, offline, no sleeps.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QFileDialog, QMainWindow, QStackedWidget

from spacr.curation import MaskCuration
from spacr.layers import LayerStack, Spacing
from spacr.qt import command_palette as cp
from spacr.qt import crop_thumbs as ctm
from spacr.qt import curation_tool as ct
from spacr.qt import job_runner as jr
from spacr.qt import layer_viewer as lv
from spacr.qt import plate_queue as pq
from spacr.qt import prefs
from spacr.qt import synthetic as sy

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack(size=32):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, 1.0, units="px")
    stack.add_image(np.zeros((size, size), np.uint16), name="image",
                    spacing=spacing)
    stack.add_labels(np.zeros((size, size), np.int64), name="mask",
                     spacing=spacing)
    return stack


def _canvas(qtbot, stack):
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(120, 120)
    canvas._ensure_canvas()
    return canvas


def _tracks():
    rows = []
    for track_id in (1, 2):
        for frame in range(3):
            rows.append({"frame": frame, "track_id": track_id,
                         "original_label": 100 + frame,
                         "x": float(frame), "y": float(track_id)})
    return pd.DataFrame(rows, columns=["frame", "track_id", "original_label",
                                       "x", "y"])


# ---------------------------------------------------------------------------
# spacr.qt.plate_queue
# ---------------------------------------------------------------------------

def test_a_patch_that_changes_nothing_does_not_rewrite_the_queue(tmp_path):
    """``update`` saves on a change and only on a change.

    The queue file is the crash-recovery record for a run that can take
    hours, and the screen calls ``update`` on every poll tick. Rewriting it
    for a no-op patch is a synchronous JSON dump per tick; skipping it for a
    real one loses the plate that was running when the machine went down.

    Detected by *deleting* the file after the first save: a save recreates
    it, so its absence is a positive fact about the second call rather than
    an inference from a timestamp.
    """
    path = tmp_path / "queue.json"
    queue = pq.PlateQueue(path)
    item = pq.QueueItem.build("mask", {"src": "/plates/A"})
    queue.add(item)
    assert path.is_file(), "add() should have written the snapshot"
    path.unlink()

    # Same values as the item already carries -> no write.
    queue.update(item.id, status=pq.Status.QUEUED, label="/plates/A")
    assert not path.exists()

    # A field the item does not have -> no write, and no attribute invented.
    queue.update(item.id, not_a_queue_field=17)
    assert not path.exists()
    assert not hasattr(item, "not_a_queue_field")

    # ... and one real change writes the whole snapshot back.
    queue.update(item.id, status=pq.Status.RUNNING)
    assert item.status is pq.Status.RUNNING
    payload = json.loads(path.read_text())
    assert [row["status"] for row in payload["items"]] == ["running"]


def test_a_csv_cell_that_is_text_or_absent_is_carried_through_unconverted(
        tmp_path):
    """The coercion ladder's bottom two rungs.

    ``import_plates_from_csv`` tries int, then float, then the boolean and
    null spellings, and a cell matching none of them has to arrive at the
    settings dict as the string the user typed. A row that stops short of the
    header hands ``csv.DictReader``'s ``None`` in, which is a missing value
    and must not be stringified into ``"None"`` either.
    """
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text(
        "src,cell_size,note\n"
        "/plates/A,25,two channels\n"
        "/plates/B,30\n"
        ",ignored,ignored\n")

    items = pq.import_plates_from_csv(
        csv_path, {"cell_size": 0, "note": "", "channels": [0, 1]},
        app_key="mask")

    assert [i.label for i in items] == ["/plates/A", "/plates/B"]
    # numbers still coerce...
    assert items[0].settings["cell_size"] == 25
    # ... text falls all the way through as text ...
    assert items[0].settings["note"] == "two channels"
    # ... and the column the short row never supplied is None, not "None".
    assert items[1].settings["note"] is None
    # base settings are copied per row, not shared
    assert items[0].settings["channels"] == [0, 1]
    assert items[0].settings["channels"] is not items[1].settings["channels"]


# ---------------------------------------------------------------------------
# spacr.qt.curation_tool
# ---------------------------------------------------------------------------

def test_a_brush_panel_adopts_the_session_it_is_given(qtbot, qt_theme_applied):
    """Two sessions over one mask means two ledgers, each half the truth.

    A screen that already opened a ``MaskCuration`` hands it to the panel;
    the panel must paint into that one, not mint a second over the same
    labels.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    layer = stack["mask"]
    session = MaskCuration(layer, artifact="given.tif")

    adopted = ct.BrushPanel(canvas, session=session)
    qtbot.addWidget(adopted)
    assert adopted.session is session

    minted = ct.BrushPanel(canvas)
    qtbot.addWidget(minted)
    assert minted.session is not session

    # the adopted one records into the caller's ledger, not into a private one
    adopted.label_spin.setValue(4)
    tool = adopted.start_painting()
    tool.press(canvas, canvas.canvas.world_at(16, 16), _Event(Qt.LeftButton))
    tool.release(canvas, canvas.canvas.world_at(16, 16), _Event(Qt.LeftButton))
    assert len(session.log) == 1
    assert len(minted.session.log) == 0


class _Event:
    """The two accessors ``BrushTool`` asks a mouse event for."""

    def __init__(self, button, buttons=None):
        self._button = button
        self._buttons = button if buttons is None else buttons

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons


def test_save_mask_reads_a_string_as_a_path_and_a_bool_as_no_path(
        qtbot, qt_theme_applied, tmp_path):
    """``clicked`` hands a slot the checked state.

    Wired straight to a button, ``save_mask`` receives ``False`` — and a bool
    read as a path would write the curated labels to a file called "False".
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    artifact = tmp_path / "from_the_panel.tif"
    panel = ct.BrushPanel(canvas, artifact=str(artifact))
    qtbot.addWidget(panel)

    elsewhere = tmp_path / "explicit.tif"
    written = panel.save_mask(str(elsewhere))
    assert written == str(elsewhere)
    assert elsewhere.is_file()
    assert not artifact.exists()

    # the bool the signal supplies means "the artefact this panel was opened
    # on", so the second write lands on the other file.
    assert panel.save_mask(False) == str(artifact)
    assert artifact.is_file()


def test_a_track_save_with_nowhere_to_write_asks_and_honours_the_answer(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Cancelling the dialog must write nothing, and naming a file must write
    it — the two halves of the same guard."""
    panel = ct.TrackCurationPanel(tracks=_tracks())
    qtbot.addWidget(panel)

    cancelled = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    cancelled.append(panel.save())
    assert cancelled == [None]
    assert not list(tmp_path.glob("*.csv"))

    target = tmp_path / "chosen.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "CSV")))
    assert panel.save() == str(target)
    assert target.is_file()
    assert set(pd.read_csv(target)["track_id"]) == {1, 2}


# ---------------------------------------------------------------------------
# spacr.qt.synthetic
# ---------------------------------------------------------------------------

def test_a_demo_renders_only_the_channels_asked_for(tmp_path):
    """The cell and nucleus masks are built whatever is rendered.

    ``_synth_field`` paints each role's mask and only *then* asks whether its
    channel was wanted, because the measure and crop demos need the truth
    masks for roles whose image plane nobody asked to write. Rendering a
    single non-cell channel walks that skip for both concentric roles.
    """
    layout = sy.generate_mask_demo(tmp_path / "demo", wells=("A01",), fields=1,
                                   channels=(sy.CHANNEL_LAYOUT[
                                       "pathogen_channel"],))

    names = sorted(p.name for p in layout.image_files)
    assert len(names) == 1
    assert names[0].endswith("C02.tif")
    assert layout.settings_csv.is_file()

    settings = pd.read_csv(layout.settings_csv)
    written = dict(zip(settings["Key"], settings["Value"]))
    assert "2" in str(written["channels"])


def test_an_unpaired_fastq_run_writes_r1_alone(tmp_path):
    """``paired=False`` skips the mate everywhere — the write and the close.

    A stray R2 from a previous paired run left in the folder is worse than no
    R2 at all: spacr.sequencing pairs by filename, so it would be consumed as
    this run's mate.
    """
    grnas = ["ACGT" * 5 + "A", "TTGC" * 5 + "G"]
    rows = ["ACGTACGT"]
    columns = ["TTTTGGGG", "AAAACCCC"]

    single = sy.generate_synthetic_fastq(tmp_path / "single", grnas, rows,
                                         columns, n_reads=4, paired=False)
    assert [p.name for p in single] == ["demo_R1_001.fastq.gz"]
    assert not (tmp_path / "single" / "demo_R2_001.fastq.gz").exists()
    with gzip.open(single[0], "rt") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 4 * 4                     # 2 wells x 2 reads/well
    assert lines[0].startswith("@")
    assert lines[2] == "+"

    # ... and the paired run, for contrast, writes both and closes both.
    both = sy.generate_synthetic_fastq(tmp_path / "pair", grnas, rows, columns,
                                       n_reads=4, paired=True)
    assert [p.name for p in both] == ["demo_R1_001.fastq.gz",
                                      "demo_R2_001.fastq.gz"]
    with gzip.open(both[1], "rt") as fh:
        mate = fh.read().splitlines()
    assert len(mate) == 4 * 4
    assert mate[1] == sy._reverse_complement(
        gzip.open(both[0], "rt").read().splitlines()[1])


# ---------------------------------------------------------------------------
# spacr.qt.prefs
# ---------------------------------------------------------------------------

def test_a_recent_list_stored_as_one_string_reads_back_as_a_list():
    """``push_recent_source`` writes ``"\\n".join(...)``, so the string form is
    the *normal* one on any backend that does not preserve a QStringList."""
    app_key = "r6_recent_as_string"
    store = QSettings(prefs.ORG, prefs.APP)
    store.setValue(f"recent/{app_key}/list", "/plates/a\n/plates/b\n\n/plates/c")
    store.sync()

    assert prefs.get_recent_sources(app_key) == ["/plates/a", "/plates/b",
                                                 "/plates/c"]
    assert prefs.get_recent_sources(app_key, limit=2) == ["/plates/a",
                                                          "/plates/b"]

    # the list form and the absent form, for contrast
    store.setValue(f"recent/{app_key}/list", ["/plates/x", "/plates/y"])
    store.sync()
    assert prefs.get_recent_sources(app_key) == ["/plates/x", "/plates/y"]
    assert prefs.get_recent_sources("r6_never_written") == []


def test_the_canonical_store_is_not_migrated_onto_itself(monkeypatch):
    """A legacy namespace can resolve to the file being migrated INTO.

    ``ORG``/``APP`` are what a repackaged build changes, and pointing them at
    one of the historical namespaces makes ``current`` and ``legacy`` the same
    file. The copy has to skip that one and still perform the other, or a
    build shipped under the old organisation name silently loses the recent
    paths held by its sibling store.
    """
    monkeypatch.setattr(prefs, "ORG", "Olafsson Lab")
    monkeypatch.setattr(prefs, "APP", "spaCR")
    monkeypatch.setattr(prefs, "_MIGRATED_FILES", set())

    canonical = QSettings(prefs.ORG, prefs.APP)
    # The setup is only meaningful while these really are one file; assert it
    # rather than assume it, so a platform where they diverge fails loudly.
    assert canonical.fileName() == QSettings(*prefs._LEGACY_NAMESPACES[0]).fileName()
    other_org, other_app = prefs._LEGACY_NAMESPACES[1]
    assert QSettings(other_org, other_app).fileName() != canonical.fileName()

    canonical.setValue("recent/r6_own/last", "/plates/own")
    canonical.sync()
    sibling = QSettings(other_org, other_app)
    sibling.setValue("recent/r6_sibling/last", "/plates/sibling")
    sibling.sync()

    assert prefs.get_last_source("r6_own") == "/plates/own"
    assert prefs.get_last_source("r6_sibling") == "/plates/sibling"


# ---------------------------------------------------------------------------
# spacr.qt.job_runner
# ---------------------------------------------------------------------------

def test_an_unthreaded_job_without_a_callback_still_reports_finished(qtbot):
    """``on_done`` is optional; ``job_finished`` is not.

    The unthreaded runner is what the tests and the headless entry points
    use, and a caller that only wants the side effect passes no callback. The
    busy indicator is driven by ``job_finished``, so skipping the emit would
    leave the spinner turning for ever.
    """
    runner = jr.JobRunner(threaded=False)
    seen = []
    finished = []
    runner.job_finished.connect(finished.append)

    assert runner.submit(lambda: seen.append("ran") or "result") is True
    assert seen == ["ran"]
    assert finished == [True]

    # ... and with a callback, the result reaches it.
    got = []
    assert runner.submit(lambda: "payload", got.append) is True
    assert got == ["payload"]
    assert finished == [True, True]


def test_the_last_non_blank_line_is_the_first_one_looked_at(qtbot):
    """Why ``_on_worker_error_text``'s loop can never take a second turn.

    The loop walks ``str(text).strip().splitlines()`` in reverse looking for a
    non-blank line, and breaks on the first one it finds. But ``str.strip()``
    has already removed every trailing whitespace character, and every
    character ``str.splitlines()`` treats as a line boundary is whitespace
    (asserted below over the whole code space), so the LAST element of
    ``splitlines()`` — the first candidate visited — always ends in a
    non-whitespace character. Its ``strip()`` is therefore always truthy and
    the loop always breaks on iteration one; the arc back to the top of the
    loop is dead. The only other case, an all-whitespace ``text``, strips to
    ``""`` whose ``splitlines()`` is ``[]``, and the body never runs at all.

    What is pinned is the invariant, not the guard: the traceback's last line
    is the one the user is shown, and blank lines before it are skipped by
    ``strip`` rather than by the loop.
    """
    boundaries = "\n\r\v\f\x1c\x1d\x1e\x85  "
    assert all(ch.isspace() for ch in boundaries)
    assert not [cp for cp in range(0x110000)
                if chr(cp).isspace() and ("x" + chr(cp)).strip() != "x"]

    runner = jr.JobRunner(threaded=False)
    failures = []
    runner.job_failed.connect(failures.append)

    runner._on_worker_error_text(
        "Traceback (most recent call last):\n  File ...\n"
        "ValueError: the plate has no wells\n\n   \n")
    assert failures == ["ValueError: the plate has no wells"]

    runner._on_worker_error_text("   \n \n")
    assert failures[-1] == "unknown error"


# ---------------------------------------------------------------------------
# spacr.qt.crop_thumbs
# ---------------------------------------------------------------------------

def test_a_pixmap_whose_size_cannot_be_read_costs_zero_bytes(qtbot,
                                                             qt_theme_applied):
    """The memory sweep asks every cache what it is holding.

    ``cache_budget_entries`` calls ``_pixmap_bytes`` for every live entry, and
    it runs on a timer across every cache in the process. One entry that
    cannot answer must be worth zero, not an exception that kills the sweep
    for all the others.

    Driven on the static helper directly: only a real ``QPixmap`` or ``None``
    can reach the cache through ``pixmap()``, so the failure this guard exists
    for cannot be staged through the public door.
    """
    from PySide6.QtGui import QPixmap

    real = QPixmap(8, 4)
    real.fill()
    assert ctm.CropThumbnails._pixmap_bytes(real) > 0
    assert ctm.CropThumbnails._pixmap_bytes(None) == 0
    assert ctm.CropThumbnails._pixmap_bytes(QPixmap()) == 0

    class _Unmeasurable:
        def isNull(self):
            return False

        def width(self):
            raise ValueError("the paint device is gone")

        def height(self):
            return 4

        def depth(self):
            return 32

    assert ctm.CropThumbnails._pixmap_bytes(_Unmeasurable()) == 0

    # ... and the sweep, which evaluates it for every entry, survives one.
    cache = ctm.CropThumbnails()
    cache._cache[("staged",)] = _Unmeasurable()
    rows = cache.cache_budget_entries()
    assert [(key, size) for key, size, _used, _pinned in rows] == [
        (("staged",), 0)]


# ---------------------------------------------------------------------------
# spacr.qt.command_palette
# ---------------------------------------------------------------------------

class _Window(QMainWindow):
    """A main window with only the attributes the palette reaches for."""

    def __init__(self):
        super().__init__()
        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)
        self._screens = {}
        self.navigated = []

    def _on_nav_selected(self, key):
        self.navigated.append(key)


def _key(code):
    return QKeyEvent(QKeyEvent.KeyPress, code, Qt.NoModifier)


@pytest.fixture
def palette(qtbot, qt_theme_applied):
    window = _Window()
    qtbot.addWidget(window)
    pal = cp.CommandPalette(window)
    qtbot.addWidget(pal)
    return pal


def test_down_at_the_last_command_stays_on_it(palette):
    """Down past the end must not deselect.

    ``_render`` selects the first command and the arrow keys move between
    them; falling off the bottom with nothing selected would make the next
    Enter a no-op, which is indistinguishable from a command that ran and did
    nothing.

    The list is reached directly because the palette exposes no "go to the
    last row" seam — sixty Down presses would be the same reach with more
    steps.
    """
    last = palette._list.count() - 1
    assert last > 0
    palette._list.setCurrentRow(last)
    assert palette._list.item(last).flags() != Qt.NoItemFlags

    palette.keyPressEvent(_key(Qt.Key_Down))
    assert palette._list.currentRow() == last

    # ... while from anywhere else Down still advances to the next command.
    palette._list.setCurrentRow(1)
    palette.keyPressEvent(_key(Qt.Key_Down))
    moved = palette._list.currentRow()
    assert moved > 1
    assert palette._list.item(moved).flags() != Qt.NoItemFlags


def test_a_rendered_list_always_holds_something_selectable(palette):
    """Why ``_render``'s auto-select loop can never run off the end.

    ``_render`` appends exactly one selectable item per command, and a header
    only ever immediately before one. So ``self._list.count() > 1`` implies at
    least one command was rendered, its item's flags are the default ones
    rather than ``Qt.NoItemFlags``, and the loop always breaks — the arc that
    leaves it without selecting anything is dead.

    The invariant is what is pinned: one selectable row per command, headers
    never selected, and the selection landing on the first command.
    """
    ran = []
    commands = [cp.Command("Alpha", "one", lambda: ran.append("a")),
                cp.Command("Beta", "two", lambda: ran.append("b"))]
    palette._render(commands)

    assert palette._list.count() == 4              # two headers, two commands
    flags = [palette._list.item(i).flags() for i in range(4)]
    assert [f != Qt.NoItemFlags for f in flags] == [False, True, False, True]
    assert palette._list.currentRow() == 1

    # the row it selected is a command, and Enter runs that command
    palette._on_activate()
    assert ran == ["a"]

    # an empty render selects nothing at all, and never asks the loop to
    assert palette._render([]) is None
    assert palette._list.count() == 0
