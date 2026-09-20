"""A curation session is a queue with a memory, and it must not lie about it.

Ledger item 396: Make Masks is the better editor and the tool outside the
repository is the better workflow. :mod:`spacr.curation_queue` is the
workflow half, and every invariant below is one the external tool learned the
hard way on a real 500-bundle set.

The four that matter most, because each has a silent failure mode:

**A folder full of work must never load as empty.** Three layouts are in use
-- images with ``masks/`` beneath them, sibling ``images/`` and ``masks/``,
and Cellpose ``*_seg.npy`` pickles -- and assuming one is how a folder of the
other two opens with nothing in it. An empty queue reads as "all done", so a
folder that matches none of the three, or two of them at once, is refused out
loud instead.

**done, skip and recropped are three states, not two.** ``skip`` records
"this one cannot be curated"; folding it into ``done`` is harmless for one
session and hands the same unusable field back every session after.

**Resuming means resuming.** The status file is read, the reviewed stems drop
out, what is left is ordered, and only THEN is the limit applied -- so
``--limit 20`` means the twenty most worthwhile fields still to do. A row
about a stem that is not in this folder is kept: the set syncs between
machines, and dropping those rows offers back work already finished.

**A reorder must never be silent.** ``prob`` and ``easy`` need probabilities
and fall back to ``value`` without them. The fallback is only safe because it
says so: otherwise the curator believes they are working the uncertain fields
first when they are not.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spacr.curation_queue import (
    DEFAULT_ORDER,
    DRAFTS_FILENAME,
    IMAGE_EXTS,
    LAYOUT_NESTED,
    LAYOUT_SEG,
    LAYOUT_SIBLING,
    MIN_DIAMETER_FOR_VALUE,
    MIN_OBJECTS_FOR_VALUE,
    ORDERS,
    REVIEWED,
    SCORES_FILENAME,
    STATUS_FIELDS,
    STATUS_FILENAME,
    CurationQueueError,
    LayoutError,
    QueueItem,
    QueueSummary,
    StatusFileError,
    StatusRow,
    build_queue,
    count_draft_objects,
    detect_layout,
    discover_items,
    is_reviewed,
    mark_state,
    median_equivalent_diameter,
    order_items,
    read_status,
    summarize,
    value_key,
    write_status,
)
from spacr.mask_io import save_mask

ROOT = Path(__file__).resolve().parents[1]

#: A draft that is on disk but cannot be read: the ``value`` order's fourth
#: rank exists for these, and reading one must not take the session down.
BROKEN = "broken"

#: A field with no draft mask at all -- work, not an error: it means drawing
#: from scratch, which is why ``easy`` offers it last.
NO_DRAFT = None


def _labels(n_objects: int, side: int = 10, shape=(256, 256)) -> np.ndarray:
    """A label image holding ``n_objects`` squares of ``side`` pixels.

    Distinct ids, laid out in a grid so none of them touch: the object count
    and the median equivalent diameter are then both known exactly.
    """
    mask = np.zeros(shape, dtype=np.uint16)
    per_row = max(1, shape[1] // (side + 2))
    for index in range(n_objects):
        row, column = divmod(index, per_row)
        y, x = row * (side + 2), column * (side + 2)
        assert y + side <= shape[0], "the test grid does not fit the image"
        mask[y:y + side, x:x + side] = index + 1
    return mask


def _write_draft(path: Path, spec) -> None:
    """Write one draft mask: a count, a ``(count, side)`` pair, or BROKEN."""
    if spec is NO_DRAFT:
        return
    if spec == BROKEN:
        path.write_bytes(b"this is not a TIFF and never was")
        return
    n_objects, side = spec if isinstance(spec, tuple) else (spec, 10)
    save_mask(path, _labels(n_objects, side))


def _nested_layout(root: Path, drafts) -> Path:
    """``<folder>/*.tif`` with the masks in ``<folder>/masks/``."""
    folder = root / "nested"
    (folder / "masks").mkdir(parents=True)
    for stem, spec in drafts.items():
        save_mask(folder / f"{stem}.tif", _labels(0, shape=(8, 8)))
        _write_draft(folder / "masks" / f"{stem}.tif", spec)
    return folder


def _sibling_layout(root: Path, drafts) -> Path:
    """``<folder>/images/*.tif`` beside ``<folder>/masks/*.tif``."""
    folder = root / "sibling"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir(parents=True)
    for stem, spec in drafts.items():
        save_mask(folder / "images" / f"{stem}.tif", _labels(0, shape=(8, 8)))
        _write_draft(folder / "masks" / f"{stem}.tif", spec)
    return folder


def _seg_layout(root: Path, drafts) -> Path:
    """``<folder>/*_seg.npy`` -- the Cellpose pickles the external tool edits."""
    folder = root / "seg"
    folder.mkdir(parents=True)
    for stem, spec in drafts.items():
        path = folder / f"{stem}_seg.npy"
        if spec == BROKEN:
            path.write_bytes(b"not a pickle")
            continue
        n_objects, side = spec if isinstance(spec, tuple) else (spec, 10)
        np.save(path, {"img": _labels(0, shape=(8, 8)),
                       "masks": _labels(n_objects, side)}, allow_pickle=True)
    return folder


def _scores(folder: Path, probabilities) -> Path:
    """Write ``curate_scores.csv`` for a queue."""
    path = folder / SCORES_FILENAME
    lines = ["stem,prob"] + [f"{stem},{value}"
                             for stem, value in probabilities.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stems(items) -> list:
    """The stems of a queue, in the order it offers them."""
    return [item.stem for item in items]


def _loose(stems, layout=LAYOUT_SEG):
    """Items with no files behind them, for the pure ordering tests."""
    return [QueueItem(stem=stem, layout=layout) for stem in stems]


# ---------------------------------------------------------------------------
# All three layouts, and a loud refusal when it is not one of them
# ---------------------------------------------------------------------------

def test_images_with_a_masks_folder_beneath_them_are_the_nested_layout(tmp_path):
    """What Make Masks documents, and what its save path already writes."""
    folder = _nested_layout(tmp_path, {"field_a": 2, "field_b": 3})

    layout = detect_layout(folder)

    assert layout.kind == LAYOUT_NESTED
    assert _stems(layout.items) == ["field_a", "field_b"]
    assert [item.image.parent for item in layout.items] == [folder, folder]
    assert [item.mask.parent.name for item in layout.items] == ["masks", "masks"]
    assert layout.mask_destination("field_a") == folder / "masks" / "field_a.tif"


def test_sibling_image_and_mask_folders_are_the_sibling_layout(tmp_path):
    """How the curation sets in the field are actually laid out."""
    folder = _sibling_layout(tmp_path, {"field_a": 2, "field_b": 3})

    layout = detect_layout(folder)

    assert layout.kind == LAYOUT_SIBLING
    assert _stems(layout.items) == ["field_a", "field_b"]
    assert [item.image.parent.name for item in layout.items] == ["images"] * 2
    assert [item.mask.parent.name for item in layout.items] == ["masks"] * 2
    assert layout.mask_destination("field_b") == folder / "masks" / "field_b.tif"


def test_a_folder_of_cellpose_pickles_is_the_seg_layout(tmp_path):
    """The external tool's own layout, opened unconverted."""
    folder = _seg_layout(tmp_path, {"field_a": 2, "field_b": 3})

    layout = detect_layout(folder)

    assert layout.kind == LAYOUT_SEG
    assert _stems(layout.items) == ["field_a", "field_b"]
    assert [item.bundle.name for item in layout.items] == [
        "field_a_seg.npy", "field_b_seg.npy"]
    assert [item.draft for item in layout.items] == [
        item.bundle for item in layout.items]
    assert layout.mask_destination("field_a") == folder / "field_a_seg.npy"


def test_a_folder_matching_no_layout_is_refused_rather_than_read_as_empty(tmp_path):
    """The failure this refusal exists for: an empty queue reads as done."""
    folder = tmp_path / "images_only"
    folder.mkdir()
    for stem in ("field_a", "field_b"):
        save_mask(folder / f"{stem}.tif", _labels(0, shape=(8, 8)))

    with pytest.raises(LayoutError) as excinfo:
        detect_layout(folder)

    message = str(excinfo.value)
    assert str(folder) in message
    assert "2 image(s)" in message, "the refusal does not say what it found"
    for layout in ("nested", "sibling", "seg"):
        assert layout in message, f"the refusal does not explain {layout}"


def test_a_folder_matching_two_layouts_is_refused_rather_than_guessed(tmp_path):
    """Guessing between two layouts loads half the work and reports none."""
    folder = _nested_layout(tmp_path, {"field_a": 2})
    (folder / "images").mkdir()
    save_mask(folder / "images" / "field_c.tif", _labels(0, shape=(8, 8)))

    with pytest.raises(LayoutError) as excinfo:
        detect_layout(folder)

    message = str(excinfo.value)
    assert "nested" in message and "sibling" in message
    assert "ambiguous" in message


def test_a_folder_that_is_not_there_is_refused_not_reported_as_finished(tmp_path):
    """A mistyped path is the commonest way to be told there is no work."""
    # THE CLASS IS NOT ENOUGH. `LayoutError` is also what the no-layout
    # fall-through raises, so asserting the class alone still passes with the
    # is_dir() guard removed -- and the answer a curator then gets is the long
    # "found: 0 image(s), 0 *_seg.npy bundles, masks/ absent" report, which
    # reads as "this folder is empty" rather than "you typed the path wrong".
    # Telling those two apart is the entire reason the guard exists.
    with pytest.raises(LayoutError) as raised:
        detect_layout(tmp_path / "no_such_folder")
    assert "not a directory" in str(raised.value).lower(), (
        f"a missing folder must not be reported as an empty one: {raised.value}")


def test_two_files_sharing_one_stem_are_refused_because_one_status_row_serves_both(tmp_path):
    """One stem is one status row: curating either would retire both."""
    folder = _nested_layout(tmp_path, {"field_a": 2})
    (folder / "field_a.png").write_bytes(b"a second file claiming that stem")

    with pytest.raises(CurationQueueError) as excinfo:
        detect_layout(folder)

    assert "field_a" in str(excinfo.value)
    assert "stem" in str(excinfo.value)


def test_a_field_whose_draft_is_missing_is_still_offered_as_work(tmp_path):
    """No draft means drawing from scratch, which is work, not absence."""
    folder = _sibling_layout(tmp_path, {"has_draft": 3, "no_draft": NO_DRAFT})

    queue = build_queue(folder, order="name")

    assert _stems(queue.items) == ["has_draft", "no_draft"]
    assert queue.summary.total == 2
    assert dict(zip(_stems(queue.all_items),
                    (count_draft_objects(i) for i in queue.all_items))) == {
        "has_draft": 3, "no_draft": 0}


# ---------------------------------------------------------------------------
# The resume record
# ---------------------------------------------------------------------------

def test_done_skip_and_recropped_all_survive_a_round_trip(tmp_path):
    """Three reviewed states, written and read back as three."""
    rows = {
        "one": StatusRow("one", "done", 12, "2026-09-12T10:00:00"),
        "two": StatusRow("two", "skip", 0, "2026-09-12T10:01:00"),
        "three": StatusRow("three", "recropped", 4, "2026-09-12T10:02:00"),
    }

    write_status(tmp_path, rows)
    back = read_status(tmp_path)

    assert back == rows
    assert {row.state for row in back.values()} == {"done", "skip", "recropped"}
    assert all(row.reviewed for row in back.values())
    assert REVIEWED == {"done", "skip", "recropped"}


def test_the_status_file_is_written_with_the_columns_both_tools_declare(tmp_path):
    """The Tk ancestor writes ``updated`` too; the union is what is written."""
    write_status(tmp_path, {"one": StatusRow("one", "done", 3, "2026-09-12")})

    header = (tmp_path / STATUS_FILENAME).read_text(
        encoding="utf-8").splitlines()[0]

    # PINNED AS LITERALS. Comparing the header to STATUS_FIELDS compares the
    # constant to itself through csv: appending a column to STATUS_FIELDS
    # changes the on-disk format of a file shared with the Tk ancestor and
    # this assertion would not notice.
    assert header.split(",") == ["stem", "state", "n_objects", "updated"]
    assert STATUS_FIELDS == ["stem", "state", "n_objects", "updated"]


def test_the_status_file_keeps_the_name_a_resumed_session_looks_for(tmp_path):
    """The one string that makes "resumes across machines" work.

    Every other use of STATUS_FILENAME in this suite compares the constant to
    itself -- the expected value and the actual value both come from
    `curation_queue.py`. Rename it and nothing here goes red, while the
    maintainer's measured 500-bundle set reopens as 0 done on the next
    machine, because the session looks for a file that is no longer written.
    The literal is the assertion.
    """
    assert STATUS_FILENAME == "curate_status.csv"
    write_status(tmp_path, {"one": StatusRow("one", "done", 3, "2026-09-12")})
    assert (tmp_path / "curate_status.csv").is_file()


def test_skip_is_not_a_lesser_done(tmp_path):
    """Collapsing them hands the same unusable field back every session."""
    folder = _seg_layout(tmp_path, {"good": 3, "hopeless": 1})
    mark_state(folder, "good", "done", 3)
    mark_state(folder, "hopeless", "skip", 0)

    queue = build_queue(folder, order="name")

    assert queue.items == ()
    assert queue.summary.done == 1
    assert queue.summary.skip == 1
    assert read_status(folder)["hopeless"].state == "skip"
    assert is_reviewed("skip") and is_reviewed("done")


def test_a_status_row_for_an_image_that_is_not_here_survives_the_rewrite(tmp_path):
    """Resuming on a machine that has only half the set must not lose the rest."""
    folder = _seg_layout(tmp_path, {"here": 2})
    elsewhere = StatusRow("elsewhere", "done", 7, "2026-09-12T09:00:00")
    write_status(folder, {"elsewhere": elsewhere})

    mark_state(folder, "here", "done", 2, "2026-09-13T09:00:00")

    on_disk = read_status(folder)
    assert on_disk["elsewhere"] == elsewhere
    assert on_disk["here"].state == "done"
    summary = build_queue(folder, order="name").summary
    assert summary.total == 1 and summary.remaining == 0
    assert summary.unknown == 1, "a row about an absent stem went unreported"


def test_a_state_typed_by_hand_in_the_wrong_case_still_ends_that_stems_turn(tmp_path):
    """The file is edited by hand between sessions; ``Done`` means done."""
    folder = _seg_layout(tmp_path, {"a": 1, "b": 1})
    (folder / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\na, Done ,3,\n", encoding="utf-8")

    assert read_status(folder)["a"].state == "done"
    assert _stems(build_queue(folder, order="name").items) == ["b"]


def test_a_status_file_with_no_stem_column_is_refused(tmp_path):
    """A CSV whose rows cannot be matched to a field is not a resume record."""
    (tmp_path / STATUS_FILENAME).write_text(
        "field,state\nfield_a,done\n", encoding="utf-8")

    with pytest.raises(StatusFileError) as excinfo:
        read_status(tmp_path)

    assert "stem" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Resuming, and the limit
# ---------------------------------------------------------------------------

def test_a_resumed_session_offers_exactly_the_stems_that_were_not_reviewed(tmp_path):
    """One of each reviewed state drops out; the rest are still waiting."""
    folder = _seg_layout(tmp_path, {stem: 2 for stem in "abcde"})
    mark_state(folder, "a", "done", 3)
    mark_state(folder, "b", "skip")
    mark_state(folder, "c", "recropped", 0)

    queue = build_queue(folder, order="name")

    assert _stems(queue.items) == ["d", "e"]
    assert queue.summary == QueueSummary(total=5, done=1, skip=1, recropped=1,
                                         remaining=2, unknown=0)


def test_the_limit_is_applied_after_the_ordering_not_before(tmp_path):
    """Otherwise a limit means "the first N found", which is not a session."""
    folder = _seg_layout(tmp_path, {"aaa": 4, "bbb": 4, "ccc": 4})
    _scores(folder, {"aaa": 0.10, "bbb": 0.90, "ccc": 0.50})

    queue = build_queue(folder, order="prob", limit=2)

    assert _stems(queue.items) == ["bbb", "ccc"]
    assert _stems(build_queue(folder, order="name", limit=2).items) == [
        "aaa", "bbb"], "the fixture cannot tell the two orders apart"


def test_the_limit_counts_only_the_work_that_is_left(tmp_path):
    """Applied before the reviewed stems drop out, a limit of two yields none."""
    folder = _seg_layout(tmp_path, {stem: 1 for stem in "abcd"})
    mark_state(folder, "a", "done", 1)
    mark_state(folder, "b", "skip")

    queue = build_queue(folder, order="name", limit=2)

    assert _stems(queue.items) == ["c", "d"]
    assert queue.limit == 2 and queue.summary.total == 4


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_an_unscored_bundle_sorts_last_under_prob():
    """The -1.0 default: unscored is not the same as probability zero."""
    items = _loose(["scored_high", "scored_low", "unscored"])
    probabilities = {"scored_high": 0.9, "scored_low": 0.1}

    assert _stems(order_items(items, "prob", probs=probabilities)) == [
        "scored_high", "scored_low", "unscored"]


def test_an_unscored_bundle_sorts_behind_one_scored_zero():
    """The sharpest statement of it: zero is a score, absent is not."""
    items = _loose(["absent", "zero"])

    assert _stems(order_items(items, "prob", probs={"zero": 0.0})) == [
        "zero", "absent"]


def test_an_empty_draft_sorts_last_under_easy_even_at_the_highest_probability():
    """An empty draft means drawing from scratch, whatever it is a picture of."""
    items = _loose(["empty", "likely", "unlikely"])
    probabilities = {"empty": 0.99, "likely": 0.80, "unlikely": 0.20}
    counts = {"empty": 0, "likely": 3, "unlikely": 7}

    ordered = order_items(items, "easy", probs=probabilities, counts=counts)

    assert _stems(ordered) == ["likely", "unlikely", "empty"]
    assert _stems(order_items(items, "prob", probs=probabilities))[0] == (
        "empty"), "probability alone would have offered the empty one first"


def test_easy_breaks_a_tie_on_probability_with_the_fuller_draft():
    """Within a group: most likely first, then most objects."""
    items = _loose(["few", "many"])
    probabilities = {"few": 0.5, "many": 0.5}

    ordered = order_items(items, "easy", probs=probabilities,
                          counts={"few": 1, "many": 9})

    assert _stems(ordered) == ["many", "few"]


def test_name_order_loads_no_probabilities_at_all():
    """It returns early: nothing is read, so nothing can be slow or missing."""
    def refuse():
        """A probability source that fails loudly if it is ever consulted."""
        raise AssertionError("name order asked for probabilities")

    ordered = order_items(_loose(["b", "a", "c"]), "name",
                          probs=refuse, counts=refuse)

    assert _stems(ordered) == ["a", "b", "c"]


def test_value_order_runs_from_the_richest_draft_to_the_unreadable_one(tmp_path):
    """The four ranks, in one folder, against a deliberately hostile alphabet."""
    folder = _nested_layout(tmp_path, {
        "aaa_broken": BROKEN,
        "bbb_empty": 0,
        "ccc_sparse": (2, 10),
        "ddd_rich": (5, 36),
    })

    ordered = order_items(discover_items(folder), "value")

    assert _stems(ordered) == ["ddd_rich", "ccc_sparse", "bbb_empty",
                               "aaa_broken"]


def test_the_richness_thresholds_are_parameters_not_baked_in_numbers(tmp_path):
    """Three objects and forty pixels are plaque-specific, so they are named."""
    folder = _nested_layout(tmp_path, {"rich": (5, 36)})
    item = discover_items(folder)[0]

    assert value_key(item)[0] == 0
    assert value_key(item, min_objects=6)[0] == 1
    assert value_key(item, min_diameter=100.0)[0] == 1
    assert (MIN_OBJECTS_FOR_VALUE, MIN_DIAMETER_FOR_VALUE) == (3, 40.0)


def test_prob_order_without_probabilities_falls_back_to_value_order(tmp_path):
    """Half of the behaviour: what the curator is actually given."""
    folder = _nested_layout(tmp_path, {
        "aaa_sparse": (2, 10), "bbb_rich": (5, 36), "ccc_empty": 0})
    items = discover_items(folder)

    fell_back = order_items(items, "prob", probs={}, announce=lambda _: None)

    assert _stems(fell_back) == _stems(order_items(items, "value"))
    assert _stems(fell_back) != _stems(order_items(items, "name")), (
        "the fixture cannot tell value order from name order")


def test_the_fallback_to_value_order_says_so_on_stdout(tmp_path, capsys):
    """The other half: a silent reorder is the failure mode here."""
    folder = _nested_layout(tmp_path, {"aaa": (2, 10), "bbb": (5, 36)})

    queue = build_queue(folder, order="prob")

    printed = capsys.readouterr().out
    assert "falling back to value order" in printed
    # NAMING THE ORDER THAT WAS ASKED FOR IS THE WHOLE POINT of the notice: a
    # curator who asked for "prob" and silently got "value" believes they are
    # working the uncertain fields first when they are not. Asserting `"prob"
    # in printed` does NOT test that -- "prob" is a substring of
    # "probabilities", which the notice already contains, so the assertion
    # passed before {order} was interpolated at all.
    assert "instead of prob" in printed, (
        "the notice does not name the order that was asked for")
    assert SCORES_FILENAME in printed, "the notice does not name the fix"
    assert queue.effective_order == "value"
    assert queue.order == "prob"


def test_easy_with_real_scores_does_not_announce_a_fallback(tmp_path, capsys):
    """The notice has to mean something, so it must not fire when all is well."""
    folder = _seg_layout(tmp_path, {"empty": 0, "full": 4})
    _scores(folder, {"empty": 0.99, "full": 0.10})

    queue = build_queue(folder, order="easy")

    assert "falling back" not in capsys.readouterr().out
    assert queue.effective_order == "easy" == DEFAULT_ORDER
    assert _stems(queue.items) == ["full", "empty"]


def test_a_mistyped_order_is_refused_rather_than_quietly_becoming_another(tmp_path):
    """``--order esay`` must not silently sort by value for an hour."""
    folder = _seg_layout(tmp_path, {"a": 1})

    with pytest.raises(ValueError) as excinfo:
        order_items(_loose(["a"]), "esay")
    assert "esay" in str(excinfo.value)

    with pytest.raises(ValueError):
        build_queue(folder, order="esay")

    assert set(ORDERS) == {"easy", "prob", "value", "name"}


# ---------------------------------------------------------------------------
# The summary, the diameter, and the import
# ---------------------------------------------------------------------------

def test_the_summary_expresses_the_state_the_ledger_could_not(tmp_path):
    """Measured 2026-09-12: 500 bundles, 366 done, 29 skip, 105 remaining."""
    items = _loose([f"bundle_{index:03d}" for index in range(500)])
    status = {item.stem: StatusRow(item.stem, "done", 5, "")
              for item in items[:366]}
    status.update({item.stem: StatusRow(item.stem, "skip", 0, "")
                   for item in items[366:395]})

    summary = summarize(items, status)

    assert summary == QueueSummary(total=500, done=366, skip=29, recropped=0,
                                   remaining=105, unknown=0)
    assert summary.describe() == "500 bundles, 366 done, 29 skip, 105 remaining"


def test_the_median_diameter_is_the_one_skimage_reports():
    """``value`` order ranks on it, so it has to be that number, not a proxy."""
    regionprops = pytest.importorskip("skimage.measure").regionprops
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[2:8, 2:8] = 1
    mask[10:30, 10:30] = 2
    mask[40:52, 40:52] = 3

    expected = float(np.median([region.equivalent_diameter_area
                                for region in regionprops(mask.astype(np.int32))]))

    assert expected > 0, "the fixture has no objects to measure"
    assert median_equivalent_diameter(mask) == pytest.approx(expected)


def test_a_draft_is_counted_by_its_labels_not_by_its_largest_id(tmp_path):
    """Deleting an object leaves a gap in the ids; ``max()`` would overcount."""
    folder = _nested_layout(tmp_path, {"gappy": 5})
    mask_path = folder / "masks" / "gappy.tif"
    gappy = _labels(5)
    gappy[gappy == 2] = 0
    gappy[gappy == 4] = 0
    save_mask(mask_path, gappy)

    assert int(gappy.max()) == 5, "the fixture no longer has a gap in its ids"
    assert count_draft_objects(discover_items(folder)[0]) == 3


def test_the_queue_accepts_the_same_image_extensions_as_make_masks():
    """Two answers to "is this an image" is how a folder half loads."""
    engine = pytest.importorskip("spacr.qt.mask_engine")

    assert set(IMAGE_EXTS) == set(engine.IMAGE_EXTS)
    assert ".tif" in IMAGE_EXTS


def test_the_queue_module_carries_no_qt_and_no_numpy_into_the_process():
    """The point of the split: this logic is testable with no GUI at all."""
    code = textwrap.dedent(
        """
        import sys
        import spacr.curation_queue as queue
        heavy = sorted({name.split('.')[0] for name in sys.modules} &
                       {'PySide6', 'PyQt5', 'PyQt6', 'numpy', 'skimage',
                        'imageio', 'tifffile', 'torch', 'cv2', 'cellpose'})
        print('loaded=' + queue.DEFAULT_ORDER + ' heavy=' + ','.join(heavy))
        """
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True, env=environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "loaded=easy heavy="


# ---------------------------------------------------------------------------
# The external tool's files beside the folder, and scores that are missing
# ---------------------------------------------------------------------------
#
# Added 2026-09-19 with the editor learning the sibling and seg layouts. The
# external curation tool keeps a queue's progress and scores BESIDE the
# folder, as <parent>/<name>_status.csv and <parent>/<name>_scores.csv, and
# the seg sets it curated have no record inside. Measured that day on
# round3_queue: 373 done and 29 skip beside it, nothing inside -- so a session
# that looked only inside would have offered all 402 again.

def _beside(folder: Path, suffix: str, text: str) -> Path:
    """Write the external tool's ``<parent>/<name><suffix>``."""
    path = folder.parent / f"{folder.name}{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_folder_with_only_the_external_record_resumes_from_it(tmp_path,
                                                                capsys):
    """The external tool's done and skip hold, and a save goes back there."""
    folder = _seg_layout(tmp_path, {"aaa": 2, "bbb": 3, "ccc": 1})
    beside = _beside(folder, "_status.csv",
                     "stem,state,n_objects,updated\n"
                     "aaa,done,2,2026-08-24T10:07:45\n"
                     "bbb,skip,,2026-08-24T10:09:50\n")

    queue = build_queue(folder, order="name", cache_counts=False)

    assert _stems(queue.items) == ["ccc"]
    assert (queue.summary.done, queue.summary.skip) == (1, 1)
    assert queue.layout.status_path == beside
    assert str(beside) in capsys.readouterr().out, "the adoption was silent"
    assert any(str(beside) in notice and "written back" in notice
               for notice in queue.notices), (
        "the editor is never told that its saves go to a file outside the "
        "folder it opened")

    mark_state(folder, "ccc", "done", n_objects=1)

    assert not (folder / STATUS_FILENAME).exists(), (
        "a second record was started inside the folder")
    written = beside.read_text(encoding="utf-8")
    assert "aaa,done" in written and "bbb,skip" in written
    assert "ccc,done" in written


def test_the_folders_own_record_wins_over_the_one_beside_it(tmp_path):
    """Once a folder has its own record, the external one is not read."""
    folder = _nested_layout(tmp_path, {"aaa": 1, "bbb": 1})
    write_status(folder, {"aaa": StatusRow("aaa", "done", 1, "2026-09-19")})
    _beside(folder, "_status.csv",
            "stem,state,n_objects,updated\nbbb,done,1,2026-09-01\n")

    queue = build_queue(folder, order="name", cache_counts=False)

    assert _stems(queue.items) == ["bbb"]
    assert queue.layout.status_path == folder / STATUS_FILENAME


def test_a_file_beside_the_folder_that_is_no_status_record_is_ignored(tmp_path):
    """Only a ``stem``/``state`` header makes a sibling CSV a resume record."""
    folder = _nested_layout(tmp_path, {"aaa": 1})
    _beside(folder, "_status.csv", "well,colour\nA1,red\n")

    queue = build_queue(folder, order="name", cache_counts=False)

    assert _stems(queue.items) == ["aaa"]
    assert queue.layout.status_path == folder / STATUS_FILENAME


def test_the_external_scores_file_orders_prob_and_is_named(tmp_path, capsys):
    """``plaque_prob`` beside the folder is what the plaque project writes."""
    folder = _seg_layout(tmp_path, {"aaa": 2, "bbb": 3})
    beside = _beside(folder, "_scores.csv",
                     "stem,plaque_prob\naaa,0.2\nbbb,0.9\n")

    queue = build_queue(folder, order="prob", cache_counts=False)

    assert queue.effective_order == "prob"
    assert _stems(queue.items) == ["bbb", "aaa"]
    assert f"probabilities from {beside}" in capsys.readouterr().out
    assert queue.notices == ()


def test_a_csv_beside_the_folder_with_no_probability_is_passed_over(tmp_path,
                                                                   capsys):
    """Not a scores file, so not a reason to stop: the fallback says so."""
    folder = _seg_layout(tmp_path, {"aaa": 2})
    _beside(folder, "_scores.csv", "stem,area\naaa,5\n")

    queue = build_queue(folder, order="prob", cache_counts=False)

    assert queue.effective_order == "value"
    assert "instead of prob" in capsys.readouterr().out


def test_no_scores_names_both_places_it_looked_and_is_kept_on_the_queue(
        tmp_path, capsys):
    """The absence is explicit: where it looked, and what the file needs."""
    folder = _nested_layout(tmp_path, {"aaa": (2, 10), "bbb": (5, 36)})

    queue = build_queue(folder, order="easy", cache_counts=False)

    printed = capsys.readouterr().out
    assert str(folder / SCORES_FILENAME) in printed
    assert str(tmp_path / "nested_scores.csv") in printed
    assert "plaque_prob" in printed, "the notice does not say what columns"
    assert len(queue.notices) == 1
    assert queue.notices[0] in printed
    assert "instead of easy" in queue.notices[0]
    assert queue.order_phrase == "sorted by value (asked for easy)"


def test_fields_the_scores_file_does_not_name_are_counted_aloud(tmp_path,
                                                                capsys):
    """A partial scores file used to rank the rest last without a word."""
    folder = _nested_layout(tmp_path, {"aaa": 1, "bbb": 1, "ccc": 1})
    _scores(folder, {"bbb": 0.5})

    queue = build_queue(folder, order="prob", cache_counts=False)

    printed = capsys.readouterr().out
    assert queue.effective_order == "prob"
    assert _stems(queue.items)[0] == "bbb"
    assert "2 of 3 field(s) have no probability" in printed
    assert len(queue.notices) == 1 and "2 of 3" in queue.notices[0]


def test_the_easy_notice_describes_the_order_easy_actually_uses(tmp_path,
                                                                capsys):
    """An unscored draft still beats a scored empty one under ``easy``.

    The notice used to say every order "ranks them below every scored
    field", which is ``prob``'s rule. ``easy`` sorts populated drafts first
    whatever their probability, so an unscored populated field is offered
    BEFORE a scored empty one, and the sentence must not claim otherwise.
    """
    folder = _nested_layout(tmp_path, {"aaa": 1, "bbb": 0, "ccc": 1})
    _scores(folder, {"bbb": 0.9, "ccc": 0.1})

    queue = build_queue(folder, order="easy", cache_counts=False)

    assert queue.effective_order == "easy"
    assert _stems(queue.items) == ["ccc", "aaa", "bbb"]
    notice, = queue.notices
    assert "1 of 3 field(s) have no probability" in notice
    assert "populated drafts before empty ones" in notice
    assert "within each group" in notice
    assert notice in capsys.readouterr().out


def test_the_prob_notice_says_unscored_fields_come_last(tmp_path):
    """Under ``prob`` the unscored ARE below every scored field."""
    folder = _nested_layout(tmp_path, {"aaa": 1, "bbb": 0})
    _scores(folder, {"bbb": 0.01})

    queue = build_queue(folder, order="prob", cache_counts=False)

    assert _stems(queue.items) == ["bbb", "aaa"]
    notice, = queue.notices
    assert notice.endswith("prob order ranks them below every scored field")


def test_a_fully_scored_queue_carries_no_notice(tmp_path, capsys):
    """The notices have to mean something, so a clean queue has none."""
    folder = _nested_layout(tmp_path, {"aaa": 1, "bbb": 1})
    _scores(folder, {"aaa": 0.1, "bbb": 0.5})

    queue = build_queue(folder, order="prob", cache_counts=False)

    assert queue.notices == ()
    assert "!" not in capsys.readouterr().out


# -- the draft-count cache, which is an optimisation and must behave like one


def _counts(folder, **kwargs):
    """``load_draft_counts`` over everything the folder holds."""
    from spacr.curation_queue import load_draft_counts

    return load_draft_counts(folder, discover_items(folder), **kwargs)


def test_the_second_launch_reads_the_cache_instead_of_the_drafts(tmp_path):
    """The whole reason the cache exists: three hundred TIFFs at every launch.

    The drafts are DELETED between the two calls, so a second read of the
    masks could not possibly return the same numbers -- which is the only
    way to show the cache was used rather than merely written.
    """
    folder = _nested_layout(tmp_path, {"aaa": 2, "bbb": 5})

    first = _counts(folder)
    assert first == {"aaa": 2, "bbb": 5}
    assert (folder / DRAFTS_FILENAME).is_file()

    for draft in (folder / "masks").glob("*.tif"):
        draft.write_bytes(b"this is not a TIFF and never was")

    assert _counts(folder) == {"aaa": 2, "bbb": 5}


def test_a_field_added_after_the_cache_was_written_is_counted_and_added(
    tmp_path,
):
    """Only the missing stems are read, and the cache grows to hold them."""
    folder = _nested_layout(tmp_path, {"aaa": 2})
    _counts(folder)

    _write_draft(folder / "masks" / "bbb.tif", 4)
    save_mask(folder / "bbb.tif", _labels(0, shape=(8, 8)))

    assert _counts(folder) == {"aaa": 2, "bbb": 4}
    rows = (folder / DRAFTS_FILENAME).read_text(encoding="utf-8")
    assert "aaa,2" in rows and "bbb,4" in rows


@pytest.mark.parametrize(
    ("damage", "why"),
    [
        (b"\x00\x01 not a csv \xff\xfe", "not valid UTF-8 at all"),
        (b"stem,n_objects\naaa,2\nbbb,\xc3", "truncated mid-write by a crash"),
        (b'stem,n_objects\n"unclosed\n', "malformed CSV"),
    ],
)
def test_a_damaged_cache_is_recounted_rather_than_believed(
    tmp_path, damage, why,
):
    """Counts drive ORDERING only, so a bad cache mis-sorts and nothing more.

    It must not take the session down -- a launch that raises over an
    optimisation has cost the curator their whole session -- and it must not
    be trusted either: the drafts are on disk and are the truth.

    The middle case is the one that bit: ``curate_drafts.csv`` cut off
    mid-character is not valid UTF-8, and ``UnicodeDecodeError`` is not an
    ``OSError`` or a ``csv.Error``, so it came straight out of
    ``build_queue``.
    """
    folder = _nested_layout(tmp_path, {"aaa": 2, "bbb": 5})
    (folder / DRAFTS_FILENAME).write_bytes(damage)

    assert _counts(folder, write_cache=False) == {"aaa": 2, "bbb": 5}, why


def test_cache_rows_that_are_not_counts_are_dropped_not_guessed(tmp_path):
    """A row with no stem, or a count that is not a number, is ignored."""
    folder = _nested_layout(tmp_path, {"aaa": 2, "bbb": 5})
    (folder / DRAFTS_FILENAME).write_text(
        "stem,n_objects\n"
        ",7\n"
        "aaa,not a number\n"
        "bbb,99\n",
        encoding="utf-8",
    )

    counts = _counts(folder, write_cache=False)

    assert counts["aaa"] == 2, "an unreadable count is recounted from the draft"
    assert counts["bbb"] == 99, "a usable cached count is still used"


def test_a_read_only_folder_still_opens_and_says_the_cache_was_not_written(
    tmp_path,
):
    """A queue on a read-only share is a queue you can still curate.

    Failing to write an optimisation must not fail the session, and the
    curator has to be told rather than left wondering why every launch is
    slow.
    """
    folder = _nested_layout(tmp_path, {"aaa": 2})
    said = []

    def refuse_to_write(path, mode="r", *args, **kwargs):
        if "w" in mode and str(path).endswith(DRAFTS_FILENAME):
            raise OSError(30, "Read-only file system")
        return _real_open(path, mode, *args, **kwargs)

    import builtins

    _real_open = builtins.open
    original = builtins.open
    builtins.open = refuse_to_write
    try:
        counts = _counts(folder, announce=said.append)
    finally:
        builtins.open = original

    assert counts == {"aaa": 2}
    assert not (folder / DRAFTS_FILENAME).exists()
    assert any("cache not written" in line for line in said), said


def test_write_cache_false_leaves_no_file_behind(tmp_path):
    """`build_queue(cache_counts=False)` must not write into the folder."""
    folder = _nested_layout(tmp_path, {"aaa": 2})

    assert _counts(folder, write_cache=False) == {"aaa": 2}
    assert not (folder / DRAFTS_FILENAME).exists()
