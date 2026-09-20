"""The crop loader's two halves, and the promises each one makes.

`spacr.crop_loader` exists because the Embeddings screen had no way to load
crops: `EmbeddingsScreen.set_crops` was defined and nothing in `spacr/` called
it, so instruction 366's Embeddings landing page could not describe an input
path that did not exist.

THE DESIGN CLAIM THIS FILE HOLDS DOWN IS THE SPLIT. Planning answers "which
objects" and reads NO pixels; loading reads them a page at a time. Everything
a screen can do with a long read -- say how many there are before committing
to them, move a progress line, stop between pages -- follows from planning
being cheap, and nothing here is true if `plan_crops` quietly opens a file.

Every test builds its own throwaway plate under `tmp_path`. Nothing here
reads the user's data, and nothing needs Qt.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr.crop_loader import (CropLoadError, CropQuery, conform_crop,
                               load_crops, object_classes, plan_crops, plates)


def _png(path, size=24, value=None, channels=3):
    """Write one crop PNG, with a distinguishable constant value."""
    from PIL import Image

    if value is None:
        value = 7
    array = np.full((size, size, channels), int(value) % 256, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _folder(tmp_path, sizes=(24, 24, 24), name="cell_png"):
    """A bare crop folder: PNGs and nothing else."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    for index, size in enumerate(sizes):
        _png(str(folder / f"c{index:03d}.png"), size=size, value=10 + index)
    return str(folder)


def _plate(tmp_path, plate_names=("plate1", "plate2"), per_plate=3,
           with_object_table=True):
    """A plate folder: crops under data/, a png_list, and the object table.

    ``with_object_table=False`` is a database whose measurement tables were
    never written, which is what makes the PNG route's independence from the
    merged join testable.
    """
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
    if with_object_table:
        conn.execute(
            "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT,"
            " fieldID TEXT, path_name TEXT, file_name TEXT)")
    conn.commit()
    conn.close()
    return str(db)


def test_planning_reads_no_pixels(tmp_path, monkeypatch):
    """The claim the whole design rests on, asserted rather than assumed.

    If planning opened even one crop, a screen could not run it on a click:
    a plate of sixty thousand objects on a network mount would freeze on the
    button press, which is exactly what the two-job split avoids. So every
    route into reading a crop is booby-trapped for the duration of the plan.
    """
    import spacr.crops as crops

    def explode(*_args, **_kwargs):
        raise AssertionError("planning opened a crop")

    monkeypatch.setattr(crops, "read_crop_png", explode)
    monkeypatch.setattr(crops, "extract_crop", explode)
    monkeypatch.setattr(crops, "extract_crops", explode)

    db = _plate(tmp_path)
    plan = plan_crops(CropQuery(path=db))

    assert plan.count == 6
    assert not plan.is_empty


def test_a_page_is_read_at_a_time_and_says_so(tmp_path):
    """Progress arrives per page, rises, and ends at the total.

    A screen reports what it is told. A loader that reported only at the end
    would give a progress line one update, which is a frozen one.
    """
    db = _plate(tmp_path)
    plan = plan_crops(CropQuery(path=db, page_size=2))
    seen = []

    crops = load_crops(plan, progress=lambda done, total: seen.append(
        (done, total)))

    assert len(plan.pages()) == 3
    assert seen == [(2, 6), (4, 6), (6, 6)]
    assert crops.shape[0] == 6
    assert [start for start, _stop in plan.pages()] == [0, 2, 4]


def test_every_row_is_in_exactly_one_page(tmp_path):
    """The pages partition the plan. An overlap embeds a crop twice; a gap
    leaves a row of the array never written and full of whatever numpy's
    allocator had."""
    db = _plate(tmp_path, per_plate=4)
    plan = plan_crops(CropQuery(path=db, page_size=3))

    covered = [index for start, stop in plan.pages()
               for index in range(start, stop)]

    assert covered == list(range(plan.count))


def test_the_stack_is_the_layout_the_engine_documents(tmp_path):
    """``(objects, height, width, channels)``, channels LAST.

    The one thing `set_crops` refuses and `embed_array` requires, and the
    loader is now the main producer of it.
    """
    db = _plate(tmp_path)

    crops = load_crops(plan_crops(CropQuery(path=db)))

    assert crops.ndim == 4
    assert crops.shape == (6, 24, 24, 3)


def test_a_cap_is_reported_rather_than_applied_silently(tmp_path):
    """A subset that looks like the whole plate is the failure this avoids.

    Sixty thousand crops do not fit beside a backbone, so a cap is right.
    A cap nobody is told about turns "embed this plate" into "embed the
    first part of this plate, sorted however SQLite felt like".
    """
    db = _plate(tmp_path, per_plate=5)
    plan = plan_crops(CropQuery(path=db, limit=4))

    assert plan.count == 4
    assert plan.matched == 10
    assert plan.capped
    assert "first 4 of 10" in plan.describe()

    record = {}
    load_crops(plan, record=record)
    assert record["capped"] is True
    assert record["matched"] == 10
    assert record["loaded"] == 4


def test_a_load_stops_between_pages_and_keeps_what_it_read(tmp_path):
    """A stop costs at most one page, and never half a crop.

    Checked between pages rather than inside one, so the array is never
    left with a partly written row -- and what was read is returned, because
    two thousand crops already in memory are worth more than a refusal.
    """
    db = _plate(tmp_path, per_plate=5)
    plan = plan_crops(CropQuery(path=db, page_size=2))
    asked = []

    def stop_after_two_pages():
        asked.append(1)
        return len(asked) > 2

    record = {}
    crops = load_crops(plan, cancelled=stop_after_two_pages, record=record)

    assert crops.shape[0] == 4
    assert record["stopped"] is True
    assert record["loaded"] == 4
    assert record["matched"] == 10


def test_a_load_stopped_before_its_first_page_says_so(tmp_path):
    """Zero crops is not an array of zero crops; it is a sentence."""
    db = _plate(tmp_path)
    plan = plan_crops(CropQuery(path=db, page_size=2))

    with pytest.raises(CropLoadError, match="stopped before the first page"):
        load_crops(plan, cancelled=lambda: True)


def test_the_object_class_actually_selects(tmp_path):
    """Asking for nuclei out of a cell crop table returns NOTHING.

    Not cells. The class picks the label column, and a loader that fell back
    to whatever column was there would embed the wrong objects under the
    right name, which no downstream check could catch.
    """
    db = _plate(tmp_path)

    assert object_classes(db) == ("cell",)
    assert plan_crops(CropQuery(path=db, object_type="cell")).count == 6
    assert plan_crops(CropQuery(path=db, object_type="nucleus")).is_empty


def test_the_plate_filter_selects_one_plate(tmp_path):
    """And the plates on offer are the ones the table holds."""
    db = _plate(tmp_path, plate_names=("plate1", "plate2"), per_plate=3)

    assert plates(db) == ("plate1", "plate2")
    assert plan_crops(CropQuery(path=db, plate="plate2")).count == 3
    assert plan_crops(CropQuery(path=db)).count == 6


def test_a_condition_over_the_tables_own_columns_selects(tmp_path):
    """The third way of choosing, and the one a user reaches for last."""
    db = _plate(tmp_path, per_plate=3)

    plan = plan_crops(CropQuery(path=db, where="cell_area > 150"))

    assert plan.count == 4
    assert "cell_area > 150" in plan.describe()


def test_an_empty_answer_names_the_plate_the_screen_does_have(tmp_path):
    """"No crops" alone sends a reader to look for a broken install.

    Three shapes of nothing, three different places to go, and this is the
    one that costs the most: a mistyped plate name against a screen that is
    entirely healthy.
    """
    db = _plate(tmp_path, plate_names=("plate1", "plate2"))

    plan = plan_crops(CropQuery(path=db, plate="plate9"))

    assert plan.is_empty
    assert "plate1, plate2" in plan.empty_reason
    assert "plate9" in plan.empty_reason
    assert plan.describe() == plan.empty_reason


def test_an_empty_answer_names_the_classes_the_screen_does_have(tmp_path):
    """The second shape: the right plate, the wrong object."""
    db = _plate(tmp_path)

    reason = plan_crops(CropQuery(path=db, object_type="pathogen")).empty_reason

    assert "pathogen" in reason
    assert "It has cell." in reason


def test_an_empty_table_is_told_apart_from_an_unmatched_query(tmp_path):
    """The third shape: nothing was ever written, and no filter is to blame.

    A reader who is told their filter matched nothing goes and loosens the
    filter. If the table is empty that is wasted time, so it is a different
    sentence.
    """
    db = _plate(tmp_path, plate_names=(), per_plate=0)

    reason = plan_crops(CropQuery(path=db)).empty_reason

    assert "is empty" in reason
    assert "Measure" in reason


def test_a_folder_of_the_wrong_files_says_which_files(tmp_path):
    """The commonest folder miss is pointing at the images, not the crops."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "field.tif").write_bytes(b"")

    reason = plan_crops(CropQuery(source="folder", path=str(folder)
                                  )).empty_reason

    assert ".tif" in reason
    assert "*_png" in reason


def test_an_empty_folder_is_told_apart_from_a_full_one(tmp_path):
    """And an empty folder says it is empty, not that it holds the wrong
    thing -- which would send a reader looking for files that are not
    there."""
    folder = tmp_path / "nothing"
    folder.mkdir()

    assert "the folder is empty" in plan_crops(
        CropQuery(source="folder", path=str(folder))).empty_reason


def test_a_folder_loads_every_png_in_it(tmp_path):
    """The second source, and the one that needs no database at all."""
    folder = _folder(tmp_path, sizes=(24,) * 5)

    plan = plan_crops(CropQuery(source="folder", path=folder))
    crops = load_crops(plan)

    assert plan.count == 5
    assert crops.shape == (5, 24, 24, 3)


def test_crops_of_two_sizes_become_one_stack_and_the_count_is_kept(tmp_path):
    """A stack has to be rectangular. What that cost is recorded.

    Padding rather than rescaling, because rescaling changes the apparent
    size of the object, which is a phenotype; padding changes the
    background, which is not.
    """
    folder = _folder(tmp_path, sizes=(24, 30, 18, 24))

    record = {}
    crops = load_crops(plan_crops(CropQuery(source="folder", path=folder)),
                       record=record)

    assert crops.shape == (4, 24, 24, 3)
    assert record["conformed"] == 2
    assert record["crop_shape"] == [24, 24, 3]


def test_conforming_centres_rather_than_rescales():
    """The pixels keep their scale, and the object keeps its middle.

    A 4x4 of ones in an 8x8 frame comes back as a 4x4 of ones in the middle
    of zeros -- not as an 8x8 of ones, which is what a resize would give and
    would mean the object had doubled in size.
    """
    small = np.ones((4, 4, 3), dtype=np.uint8)

    framed = conform_crop(small, 8, 8)

    assert framed.shape == (8, 8, 3)
    assert framed[2:6, 2:6].all()
    assert not framed[0].any()
    assert not framed[:, 0].any()


def test_a_crop_that_already_fits_is_not_copied():
    """The common case does no work at all."""
    crop = np.zeros((16, 16, 3), dtype=np.uint8)

    assert conform_crop(crop, 16, 16) is crop


def test_two_crops_with_different_channels_are_not_one_stack(tmp_path):
    """Padding a missing channel would invent a stain.

    A crop with two channels and a crop with three are two different
    experiments, and the honest answer is a refusal that says so rather than
    a third channel of zeros nobody asked for.
    """
    folder = tmp_path / "mixed_png"
    folder.mkdir()
    _png(str(folder / "a.png"), channels=3)
    from PIL import Image

    Image.fromarray(np.zeros((24, 24), dtype=np.uint8)).convert("L").save(
        str(folder / "b.png"))

    plan = plan_crops(CropQuery(source="folder", path=str(folder)))
    crops = load_crops(plan)

    assert crops.shape == (2, 24, 24, 3)


def test_the_png_route_does_not_lose_rows_the_merged_join_would_drop(tmp_path):
    """A database with no measurement tables still loads its PNGs.

    `crop_rows_from_png_list` DROPS every row it cannot recover a merged
    path and an integer label for. That is right for a streamed crop, which
    cannot be cut without them, and wrong for a pre-generated one, which
    needs only the `png_path` the row already carries. Running the join for
    both routes silently discarded six crops that were sitting on disk.
    """
    db = _plate(tmp_path, with_object_table=False)

    plan = plan_crops(CropQuery(path=db))
    crops = load_crops(plan)

    assert plan.count == 6
    assert crops.shape[0] == 6


def test_a_filter_carrying_a_second_statement_is_refused(tmp_path):
    """The outer of two guards. The inner one is the read-only connection."""
    db = _plate(tmp_path)

    with pytest.raises(CropLoadError, match="not several statements"):
        plan_crops(CropQuery(path=db, where="1=1; DROP TABLE png_list"))


def test_the_database_is_opened_where_nothing_can_write_to_it(tmp_path):
    """The guard that holds when the first one is wrong.

    A predicate box is a fragment of the user's own query language and
    refusing it outright would mean inventing a worse one. So the engine is
    what refuses a write, not a pattern.
    """
    from spacr.crop_loader import _connect

    db = _plate(tmp_path)
    conn = _connect(db)
    try:
        with pytest.raises(sqlite3.Error):
            conn.execute("CREATE TABLE trouble (a INTEGER)")
    finally:
        conn.close()


def test_a_database_with_no_crop_table_says_what_to_do(tmp_path):
    """A measurements.db from a run that saved no crops is a common file to
    be handed, and "no png_list table" is not an instruction."""
    db = tmp_path / "bare.db"
    sqlite3.connect(str(db)).close()

    with pytest.raises(CropLoadError, match="crop folder"):
        plan_crops(CropQuery(path=str(db)))


def test_a_path_that_is_not_there_is_named(tmp_path):
    """Both sources, because both are typed by hand into the same box."""
    with pytest.raises(CropLoadError, match="not found"):
        plan_crops(CropQuery(path=str(tmp_path / "absent.db")))
    with pytest.raises(CropLoadError, match="not found"):
        plan_crops(CropQuery(source="folder", path=str(tmp_path / "absent")))


def test_an_unknown_source_names_the_ones_there_are(tmp_path):
    """A typo in a stored setting should not read as a missing file."""
    with pytest.raises(CropLoadError, match="database"):
        plan_crops(CropQuery(source="magic", path=str(tmp_path)))


def test_loading_an_empty_plan_raises_with_the_reason_it_carries(tmp_path):
    """The empty state is the plan's, and it survives into the exception so
    a caller that skipped the check still gets the useful sentence."""
    db = _plate(tmp_path)
    plan = plan_crops(CropQuery(path=db, plate="plate9"))

    with pytest.raises(CropLoadError, match="plate1"):
        load_crops(plan)


def test_the_console_is_not_flooded_by_a_bulk_load(tmp_path):
    """`spacr.crops` announces every crop path, once per path, by default.

    Right for one crop that failed to open; wrong for sixty thousand, which
    in the app are sixty thousand appends to a console widget on the GUI
    thread. It is turned off for the duration of a load and put back
    exactly as it was.
    """
    import spacr.crops as crops

    db = _plate(tmp_path)
    was = crops.PRINT_CROP_PATHS
    seen = []
    plan = plan_crops(CropQuery(path=db, page_size=2))

    def watch(done, total):
        seen.append(crops.PRINT_CROP_PATHS)

    try:
        load_crops(plan, progress=watch)
    finally:
        crops.say_crop_paths(was)

    assert seen and not any(seen)
    assert crops.PRINT_CROP_PATHS is was


def test_a_record_is_json_serialisable(tmp_path):
    """It is kept beside the matrix, like the embedding's scale record, so it
    has to survive being written down."""
    import json

    db = _plate(tmp_path)
    record = {}
    load_crops(plan_crops(CropQuery(path=db)), record=record)

    assert json.loads(json.dumps(record))["loaded"] == 6
