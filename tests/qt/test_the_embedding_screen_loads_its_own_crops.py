"""366's blocker, pressed rather than described.

Instruction 366's Embeddings landing page could not be written because
nothing in `spacr/` called `EmbeddingsScreen.set_crops`: the screen had no
in-GUI way to load crops, so a "what it needs" paragraph would have
described a path that did not exist. This file is the evidence that it does
now -- every test drives the widget, and the one that matters presses Load
and checks that the SCREEN called `set_crops`, not the test.

Lesson 0b of the handoff is why this file exists beside the model-layer one:
instruction 52 was closed on 97 passing tests and the controls turned out to
be unreachable, because no test pressed one. `tests/test_choosing_crops_is_
not_reading_them.py` is the loader; this is the panel.
"""

import os
import sqlite3

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _png(path, size=24, value=7):
    """One constant-valued crop PNG."""
    from PIL import Image

    Image.fromarray(
        np.full((size, size, 3), int(value) % 256, dtype=np.uint8)).save(path)


def _plate(tmp_path, plate_names=("plate1", "plate2"), per_plate=3):
    """A throwaway screen: crops under data/, and a png_list naming them."""
    root = tmp_path / "screen"
    data = root / "data" / "cell_png"
    measurements = root / "measurements"
    data.mkdir(parents=True, exist_ok=True)
    measurements.mkdir(parents=True, exist_ok=True)
    rows = []
    for plate in plate_names:
        for label in range(1, per_plate + 1):
            name = f"{plate}_A01_1_{label}.png"
            _png(str(data / name), value=30 + label)
            rows.append((str(data / name), name, plate, "A", "01", "1",
                         f"o{label}", float(100 * label)))
    db = measurements / "measurements.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE png_list (png_path TEXT, file_name TEXT, plateID TEXT,"
        " rowID TEXT, columnID TEXT, fieldID TEXT, cell_id TEXT,"
        " cell_area REAL)")
    conn.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db)


def _screen(qtbot):
    """The screen, unthreaded, so a press settles before the next line."""
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def _pointed_at(qtbot, path, source_index=0):
    """A screen with its path box filled in, as typing it would leave it."""
    screen = _screen(qtbot)
    screen._where.setCurrentIndex(source_index)
    screen._path.setText(str(path))
    screen._on_path_changed()
    return screen


def test_the_screen_opens_on_a_sentence_not_an_empty_table(qtbot):
    """An empty table reads as a run that produced nothing.

    The screen has never been run when it opens, which is a different thing
    and has a different next step, so it says which one it is.
    """
    from spacr.qt.screens.embeddings import NOTHING_LOADED

    screen = _screen(qtbot)

    assert screen._pages.currentWidget() is screen._state
    assert screen._state.title() == NOTHING_LOADED
    assert "Load crops" in screen._state.detail()


def test_load_is_refused_until_there_is_something_to_load(qtbot):
    """A button that can only fail is worse than one that is greyed."""
    screen = _screen(qtbot)

    assert not screen._load.isEnabled()
    assert "Choose" in screen._load.toolTip()


def test_pressing_load_is_what_calls_set_crops(qtbot, tmp_path):
    """366'S BLOCKER, AND THIS IS THE LINE THAT CLEARS IT.

    Nothing in `spacr/` called `set_crops`; the crops now arrive because the
    user pressed a button. The spy is on the screen's own method, so a load
    that filled `_crops` by some other route would not pass this.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    handed = []
    original = screen.set_crops

    def spy(crops, *, label=""):
        handed.append((crops, label))
        return original(crops, label=label)

    screen.set_crops = spy
    screen._load.click()

    assert len(handed) == 1
    crops, label = handed[0]
    assert crops.shape == (6, 24, 24, 3)
    assert "6 objects" in label


def test_a_finished_load_lets_the_encoder_run(qtbot, tmp_path):
    """Embed is disabled with nothing to embed, and enabled by a load."""
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)

    assert not screen._run.isEnabled()
    screen._load.click()

    assert screen._run.isEnabled()
    assert screen._run.toolTip() == "Encode every object"


def test_the_progress_signal_reports_once_per_page(qtbot, tmp_path):
    """A progress line with one update is a frozen one."""
    db = _plate(tmp_path, per_plate=4)
    screen = _pointed_at(qtbot, db)
    screen._limit.setValue(8)
    seen = []
    screen.crops_progress.connect(lambda done, total: seen.append(
        (done, total)))

    screen._load.click()

    assert len(seen) == len(screen._plan.pages())
    assert seen[-1] == (8, 8)
    assert [done for done, _total in seen] == sorted(
        done for done, _total in seen)


def test_the_screen_says_which_plate_it_does_have(qtbot, tmp_path):
    """The loader's empty sentence reaches the panel unedited.

    It is the one the user needs and the screen has no better one; wrapping
    it in "load failed" would bury the plate names that are the whole
    answer.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._filter.setText("cell_area > 99999")

    screen._load.click()

    assert screen._pages.currentWidget() is screen._state
    assert screen._state.title() == "No crops"
    assert "none of them satisfies this filter" in screen._state.detail()
    assert screen._state.detail() in screen._status.text()


def test_an_empty_answer_does_not_unload_what_is_already_there(qtbot,
                                                               tmp_path):
    """The screen must not say two things at once.

    A query that matched nothing leaves the previous stack where it was,
    with Embed still enabled and the header still naming it. A panel reading
    "No crops" over that contradicts the rest of the screen, so it says what
    is still loaded.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._load.click()
    assert screen._run.isEnabled()

    screen._filter.setText("cell_area > 99999")
    screen._load.click()

    assert screen._run.isEnabled()
    assert "6 crops already loaded" in screen._state.detail()
    assert "6 objects" in screen._source.text()


def test_the_cap_is_on_the_screen_and_not_only_in_the_record(qtbot, tmp_path):
    """A subset that reads as the whole plate is the failure to avoid."""
    db = _plate(tmp_path, per_plate=5)
    screen = _pointed_at(qtbot, db)
    screen._limit.setValue(4)

    screen._load.click()

    assert screen.crop_record()["loaded"] == 4
    assert screen.crop_record()["matched"] == 10
    assert "first of 10" in screen._status.text()
    assert "At most" in screen._status.text()


def test_the_classes_and_plates_offered_are_the_ones_there_are(qtbot,
                                                               tmp_path):
    """A choice that can only ever return nothing is not offered.

    Reading them costs a database round trip, so it happens off the GUI
    thread like everything else here, and only when the path changes.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)

    classes = [screen._object.itemData(i)
               for i in range(screen._object.count())]
    plates = [screen._plate.itemData(i)
              for i in range(screen._plate.count())]

    assert classes == ["cell"]
    assert plates == ["", "plate1", "plate2"]


def test_the_plate_picker_selects(qtbot, tmp_path):
    """And picking one actually narrows the load."""
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._plate.setCurrentIndex(screen._plate.findData("plate2"))

    screen._load.click()

    assert screen.crop_query().plate == "plate2"
    assert screen.crop_record()["loaded"] == 3


def test_a_folder_greys_the_controls_it_cannot_read(qtbot, tmp_path):
    """A folder has no classes, no plates and no columns to filter on.

    The house rule from `spacr.crop_source`: a control the user can edit
    that changes nothing is worse than one that is not there.
    """
    folder = tmp_path / "loose_png"
    folder.mkdir()
    for index in range(3):
        _png(str(folder / f"c{index}.png"), value=index)
    screen = _pointed_at(qtbot, str(folder), source_index=1)

    assert not screen._object.isEnabled()
    assert not screen._plate.isEnabled()
    assert not screen._filter.isEnabled()
    assert screen._load.isEnabled()

    screen._load.click()
    assert screen.crop_record()["loaded"] == 3


def test_a_load_that_refused_puts_the_button_back(qtbot, tmp_path):
    """Load reads "Stop" while a load is in flight.

    Leaving it saying Stop after the job it would have stopped has died is a
    button that does nothing, and the screen looks hung when it is idle.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._filter.setText("no_such_column > 1")

    screen._load.click()

    assert screen._load.text() == "Load crops"
    assert screen._load.isEnabled()
    assert not screen.is_loading()
    assert screen._where.isEnabled()
    assert screen._state.title() == "Crops could not be loaded"


def test_a_stop_during_planning_is_headed_as_a_stop(qtbot, tmp_path):
    """The user's own Stop must not come back as "could not be loaded".

    Planning is one count and one select; it cannot be interrupted, so a
    Stop pressed while it is in flight is answered when it returns. Before,
    the load started anyway and then refused on the first page, and the
    panel headed the user's own deliberate action as a failure of the
    machine -- which sends somebody looking for a broken database.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    from spacr.crop_loader import plan_crops

    screen._stop.clear()
    screen._set_loading(True)
    plan = plan_crops(screen.crop_query())
    screen.stop_loading()
    screen._on_planned(plan)

    assert screen._state.title() == "Loading stopped"
    assert "stopped" in screen._state.detail()
    assert not screen.is_loading()
    assert screen._load.text() == "Load crops"


def test_a_stop_before_the_first_page_is_headed_as_a_stop(qtbot, tmp_path):
    """The same, one step later: the loader raises when it read nothing.

    A stop taken before the first page finishes leaves no crops to return,
    so `load_crops` says so by raising, and that arrives on the failure
    handler. It is told apart from a real refusal by the flag the USER set,
    not by reading the message.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)

    screen._stop.clear()
    screen._set_loading(True)
    screen._stop.set()
    screen._on_job_failed("loading cell crops was stopped before the first "
                          "page finished, so there are no crops to show")

    assert screen._state.title() == "Loading stopped"
    assert not screen.is_loading()
    assert screen._where.isEnabled()


def test_a_dropped_row_does_not_tell_the_user_to_raise_the_limit(qtbot):
    """"Raise 'At most'" is an instruction, and it has to work.

    Rows the merged route cannot cut leave `loaded` below `matched` with no
    limit anywhere near it. Printing the cap sentence there hands the reader
    an action that returns the same crops and reprints the same sentence.
    """
    screen = _screen(qtbot)

    capped = screen._loaded_sentence(
        {"loaded": 4, "matched": 10, "selected": 4, "dropped": 0,
         "capped": True, "source": "png crop source"})
    complete = screen._loaded_sentence(
        {"loaded": 7, "matched": 10, "selected": 10, "dropped": 3,
         "capped": False, "source": "merged crop source"})

    assert "raise" in capped
    assert "raise" not in complete
    assert "3 more matched but could not be cut" in complete


def test_the_path_box_does_not_stat_on_the_gui_thread(qtbot, tmp_path,
                                                      monkeypatch):
    """A stat is a blocking call, and this one ran on the GUI thread.

    It ran on every edit of the path box AND at the end of every load, when
    the user had done nothing -- so a path pointing at a hung network mount
    froze the window for as long as the mount took to time out, unprompted.
    The check belongs on the worker that was already reading the database.

    The job is caught rather than run, because unthreaded both halves are
    one thread and the test could not otherwise tell them apart. What is
    asserted is the split itself: nothing before the submit touches the
    filesystem, and the caught job is what does.
    """
    import os.path as ospath

    screen = _screen(qtbot)
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit",
                        lambda fn, on_done=None: submitted.append(fn) or True)

    def explode(*_args, **_kwargs):
        raise AssertionError("the GUI thread stat'ed the path")

    missing = str(tmp_path / "on-a-hung-mount" / "measurements.db")
    screen._path.setText(missing)
    monkeypatch.setattr(ospath, "exists", explode)
    monkeypatch.setattr(ospath, "isfile", explode)
    try:
        screen._on_path_changed()
    finally:
        monkeypatch.undo()

    assert screen._load.isEnabled()
    assert "Read these crops" in screen._load.toolTip()
    assert submitted, "the database read was never handed to a worker"
    assert submitted[0]() == ((), ())


def test_the_query_the_controls_describe_is_the_one_that_runs(qtbot,
                                                              tmp_path):
    """The panel is testable without pressing anything, and the two agree."""
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._plate.setCurrentIndex(screen._plate.findData("plate1"))
    screen._filter.setText("cell_area > 150")
    screen._limit.setValue(3)

    query = screen.crop_query()

    assert query.path == db
    assert query.plate == "plate1"
    assert query.where == "cell_area > 150"
    assert query.limit == 3
    assert query.object_type == "cell"


def test_the_preview_replaces_the_panel_only_once_there_is_one(qtbot,
                                                               tmp_path):
    """The table appears when it has numbers in it, and not before.

    Showing an empty grid between the load and the embedding is the exact
    thing the opening state avoids, and it would be worse there -- crops ARE
    loaded, so an empty table reads as a run that produced nothing.
    """
    import spacr.embeddings as engine

    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._load.click()
    assert screen._pages.currentWidget() is screen._state
    assert "press Embed" in screen._state.detail()

    real = engine.embed_array
    engine.embed_array = lambda crops, spec=None, **kw: real(
        crops, spec, encoder=lambda batch: np.asarray(
            batch, dtype="float32").reshape(len(batch), -1)[:, :4])
    try:
        screen.embed()
    finally:
        engine.embed_array = real

    assert screen._pages.currentWidget() is screen._table
    assert screen._table.rowCount() == 6


def test_a_load_in_flight_counts_as_busy(qtbot, tmp_path):
    """A plate of crops off a network mount is minutes.

    A screen that reported itself idle through it would let the window close
    over a live read.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)

    assert not screen.is_busy()
    screen._set_loading(True)
    assert screen.is_busy()
    screen._set_loading(False)
    assert not screen.is_busy()


def test_closing_the_screen_stops_a_load_rather_than_waiting_for_it(qtbot,
                                                                    tmp_path):
    """`shutdown` parks a worker that outlasts its budget.

    Without the flag that worker reads the rest of the plate into an array
    whose screen has gone.
    """
    db = _plate(tmp_path)
    screen = _pointed_at(qtbot, db)
    screen._set_loading(True)

    assert not screen._stop.is_set()
    screen.close()
    assert screen._stop.is_set()


def test_the_load_runs_off_the_gui_thread(qtbot, tmp_path):
    """The claim the whole two-job design is for, driven rather than argued.

    MEASURED WITH A BASELINE, because a tick count with nothing to compare
    it to is not a measurement: the same crops are read once on the GUI
    thread and once through the screen, with one instrument. On the
    threaded path a 5 ms timer keeps firing; on the blocking one it does
    not fire at all.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from spacr.crop_loader import CropQuery, load_crops, plan_crops
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    db = _plate(tmp_path, plate_names=tuple(f"p{i}" for i in range(12)),
                per_plate=25)

    def ticks_during(run):
        """How many times a 5 ms timer fires while ``run`` is in progress."""
        counted = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: counted.append(1))
        timer.start()
        try:
            run()
        finally:
            timer.stop()
        return len(counted)

    plan = plan_crops(CropQuery(path=db, limit=300, page_size=16))
    blocking = ticks_during(lambda: load_crops(plan))

    screen = EmbeddingsScreen(threaded=True)
    qtbot.addWidget(screen)
    screen._path.setText(db)
    screen._on_path_changed()
    screen._limit.setValue(300)
    landed = []
    screen.crops_loaded.connect(landed.append)

    def press_and_pump():
        """Press Load, then pump the event loop until the crops land."""
        screen._load.click()
        deadline = 60_000
        waited = 0
        while not landed and waited < deadline:
            QApplication.processEvents()
            qtbot.wait(1)
            waited += 1

    threaded = ticks_during(press_and_pump)

    assert landed == [300]
    assert blocking == 0
    assert threaded > blocking
    screen.close()
