"""Segmentation QC: the branches a real plate takes and nobody had driven.

Six behaviours live here, and each one is a way the QC report can mislead the
person reading it:

* a field whose own pixels already showed it fused must not have the same
  verdict stapled on twice by the plate comparison;
* the card must only claim to have been written when a file was written;
* a folder that merely *looks* like a scorecard, and a file that merely looks
  like a mask stack, must not be opened as one;
* a plate carrying two object types must be scanned for masks once, not once
  per card;
* a digest with no project folder must not print an empty ``project:`` line.

Everything here is CPU-only, offline and deterministic.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from spacr import seg_qc
from spacr.seg_qc import (
    CARD_DIR,
    CARD_PREFIX,
    FLAG_UNDER,
    MASK_STACK_SUFFIX,
    find_mask_stacks,
    find_scorecards,
    format_digest,
    read_digest,
    run_segmentation_qc,
    score_masks,
    write_scorecard,
)

# ---------------------------------------------------------------------------
# synthetic masks — the same builders tests/test_seg_qc.py uses
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    """Paint one filled disc of label ``value`` into ``labels``."""
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hit = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    labels[y0:y1, x0:x1][hit] = value
    return labels


def _grid_field(shape=(512, 512), radius=10, spacing=50, margin=40):
    """A grid of well-separated discs, none touching the border."""
    labels = np.zeros(shape, np.int32)
    value = 0
    for cy in range(margin, shape[0] - margin, spacing):
        for cx in range(margin, shape[1] - margin, spacing):
            value += 1
            _disc(labels, cy, cx, radius, value)
    return labels


def _fused_field(shape=(384, 384), radius=16, step=26, cluster=3, block=96):
    """Clusters of overlapping discs, each cluster welded into ONE label."""
    labels = np.zeros(shape, np.int32)
    value = 0
    for by in range(0, shape[0], block):
        for bx in range(0, shape[1], block):
            value += 1
            cy0 = by + block // 2 - (cluster - 1) * step // 2
            cx0 = bx + block // 2 - (cluster - 1) * step // 2
            for i in range(cluster):
                for j in range(cluster):
                    _disc(labels, cy0 + i * step, cx0 + j * step, radius, value)
    return labels


# ---------------------------------------------------------------------------
# the plate comparison meeting a field the pixels already condemned
# ---------------------------------------------------------------------------

def test_a_field_the_pixels_already_called_fused_is_not_condemned_twice():
    """One defect must produce one flag and one sentence, not two of each.

    A confluent field is caught twice over: ``score_field`` sees 68% foreground
    with a distance transform resolving nine objects per label, and the plate
    comparison then sees the same field holding a fifth of the plate's objects
    at four times its median diameter. If the plate rule appended its flag
    unconditionally the card would list ``under_segmented`` twice, the flag
    tally in the plate summary would double-count that field, and the note
    would explain the same fusion to the user in two different sets of numbers.
    The plate rule may only speak about fields the pixels did not already
    convict — and the field below proves it still speaks about those.
    """
    fields = {
        "plate1_A01_f1": _grid_field(),
        "plate1_A02_f1": _grid_field(),
        "plate1_A03_f1": _grid_field(),
        # convicted by its own pixels AND by the plate comparison
        "plate1_B01_f1": _fused_field(),
        # convicted by the plate comparison ALONE: big objects, few of them,
        # on a field only 2% covered, so no confluence rule can fire.
        "plate1_B02_f1": _grid_field(shape=(512, 512), radius=15, spacing=170,
                                     margin=60),
    }

    scored = {q.field: q for q in score_masks(fields, object_type="cell")}
    both = scored["plate1_B01_f1"]
    plate_only = scored["plate1_B02_f1"]

    # the pixel-level conviction, stated once
    assert both.flags.count(FLAG_UNDER) == 1
    assert "objects look fused" in both.note
    assert "merged in pairs" not in both.note
    # the plate rule is not silent in general: this field hears it
    assert plate_only.flags.count(FLAG_UNDER) == 1
    assert "merged in pairs" in plate_only.note
    assert "objects look fused" not in plate_only.note
    # and the plate comparison did run for the fused field: it carries the
    # count reason from the same pass that skipped the duplicate flag
    assert both.metrics["count_ratio"] < 0.25
    assert "where the plate median is 81" in both.note


# ---------------------------------------------------------------------------
# run_segmentation_qc: claiming a file only when there is a file
# ---------------------------------------------------------------------------

def _small_plate():
    return {
        f"plate1_A0{i}_f1": _grid_field(shape=(128, 128), radius=6, spacing=30,
                                        margin=20)
        for i in (1, 2, 3)
    }


def test_the_card_names_the_file_it_wrote_and_stays_quiet_when_it_wrote_none(tmp_path):
    """A path in the log has to be a path the user can open.

    ``run_segmentation_qc`` is called with ``dst=None`` whenever the caller
    wants the verdict without leaving a file behind — the v2 pipeline scores a
    channel of a merged stack this way. Printing "scorecard written to None"
    there would send the user hunting for a file that was never created, and
    returning a csv_path for it would make the digest reader open a path that
    does not exist. Both runs below score the same three fields, so the only
    difference in what is printed is the one line about the file.
    """
    printed_without = []
    result_without = run_segmentation_qc(
        _small_plate(), object_type="cell", dst=None, verbose=True,
        print_fn=printed_without.append,
    )
    printed_with = []
    result_with = run_segmentation_qc(
        _small_plate(), object_type="cell", dst=str(tmp_path), verbose=True,
        print_fn=printed_with.append,
    )

    card_without = "\n".join(printed_without)
    card_with = "\n".join(printed_with)
    # the verdict itself is printed in both cases
    assert "plate1_A01_f1" in card_without and "plate1_A01_f1" in card_with
    assert result_without["csv_path"] is None
    assert "scorecard written to" not in card_without
    # and when there IS a file, it is named and it exists
    written = result_with["csv_path"]
    assert written == str(tmp_path / "qc" / "segmentation_qc_cell.csv")
    assert os.path.isfile(written)
    assert f"scorecard written to {written}" in card_with
    assert len(result_without["field_qcs"]) == 3


# ---------------------------------------------------------------------------
# things on disk that only look like cards and stacks
# ---------------------------------------------------------------------------

def test_a_folder_named_like_a_scorecard_is_not_handed_back_as_one(tmp_path):
    """A directory cannot be parsed as CSV, so it must never reach the parser.

    ``qc/`` is a folder users and other spaCR steps write into; a directory
    called ``segmentation_qc_cell.csv`` is exactly what an interrupted export
    or a rsync of a card-shaped tree leaves. Passing it on as a card path makes
    ``read_scorecard`` fail with IsADirectoryError far from here, and the
    banner then reports the plate as an error rather than reading the real
    card sitting beside it.
    """
    qc = tmp_path / CARD_DIR
    qc.mkdir()
    real = qc / f"{CARD_PREFIX}cell.csv"
    real.write_text("field,object_type,n_objects,severity,flags,note\n")
    (qc / f"{CARD_PREFIX}nucleus.csv").mkdir()

    found = find_scorecards(str(tmp_path))

    assert found == (str(real),)
    assert os.path.isdir(str(qc / f"{CARD_PREFIX}nucleus.csv"))


def test_a_file_named_like_a_mask_stack_is_not_offered_as_a_stack(tmp_path):
    """A stack is a folder of ``.npy`` masks; a file of that name is not one.

    ``find_mask_stacks`` hands what it finds to ``mask_stack_mtime``, which
    scans the folder to date the card against the masks. A stray file — a
    ``cell_mask_stack`` tarball or a leftover lock — accepted as a stack would
    date every card against one irrelevant timestamp and could mark a fresh
    card OUT OF DATE for a stack that does not exist.
    """
    (tmp_path / f"cell{MASK_STACK_SUFFIX}").write_bytes(b"not a folder")
    real = tmp_path / f"nucleus{MASK_STACK_SUFFIX}"
    real.mkdir()
    np.save(real / "plate1_A01_f1.npy", np.zeros((8, 8), np.uint16))

    stacks = find_mask_stacks(str(tmp_path))

    assert stacks == {"nucleus": str(real)}
    assert "cell" not in stacks
    assert os.path.isfile(str(tmp_path / f"cell{MASK_STACK_SUFFIX}"))


# ---------------------------------------------------------------------------
# read_digest: one plate, two cards, one directory scan
# ---------------------------------------------------------------------------

def _plate_with_two_cards(tmp_path):
    """A plate folder holding a cell card, a nucleus card and both stacks."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    for object_type in ("cell", "nucleus"):
        stack = plate / f"{object_type}{MASK_STACK_SUFFIX}"
        stack.mkdir()
        fields = {}
        for i in (1, 2, 3):
            name = f"plate1_A0{i}_f1"
            mask = _grid_field(shape=(128, 128), radius=6, spacing=30, margin=20)
            np.save(stack / f"{name}.npy", mask.astype(np.uint16))
            fields[name] = mask
        field_qcs = score_masks(fields, object_type=object_type)
        write_scorecard(field_qcs, str(plate), object_type)
    return plate


def test_two_cards_on_one_plate_scan_that_plate_for_masks_once(tmp_path, monkeypatch):
    """The screen reads the digest on every visit; it must not re-walk the disk.

    Every object type a plate was segmented for leaves its own card in the same
    ``qc/`` folder, and each card needs the mask stack that belongs to it in
    order to be dated. Looking the stacks up per card means listing the plate
    directory once per object type — on a network share holding a 1536-field
    plate that is the difference between a banner that appears and a banner
    that hangs the window. The lookup is cached per plate root, and both cards
    still have to come back correctly dated from that one scan.
    """
    plate = _plate_with_two_cards(tmp_path)
    calls = []
    real_find = seg_qc.find_mask_stacks

    def counted(root):
        calls.append(root)
        return real_find(root)

    monkeypatch.setattr(seg_qc, "find_mask_stacks", counted)

    digest = read_digest(str(plate))

    assert digest.object_types == ("cell", "nucleus")
    assert calls == [str(plate)]
    assert [os.path.basename(c.path) for c in digest.scorecards] == [
        f"{CARD_PREFIX}cell.csv", f"{CARD_PREFIX}nucleus.csv"
    ]
    # the cached lookup served BOTH cards: each one found its own stack and so
    # carries a real mask timestamp rather than the 0.0 of a stack not found
    assert all(card.masks_mtime > 0.0 for card in digest.scorecards)
    assert digest.n_fields == 6
    assert digest.stale is False


# ---------------------------------------------------------------------------
# format_digest: no project folder, no project line
# ---------------------------------------------------------------------------

def test_a_digest_with_no_project_folder_prints_no_project_line(tmp_path):
    """An empty "project:" line tells the user their masks are nowhere.

    A digest is built with an empty root whenever nothing on disk could be
    located — a settings dict whose ``src`` is still the placeholder, or a
    caller assembling a digest from cards it already holds. Printing
    ``project: `` there reads as a project folder that is the empty string,
    which is worse than saying nothing: it is the line a user copies into a
    file browser. The same digest with a root must still name it, or the line
    would be useless on the normal path.
    """
    plate = _plate_with_two_cards(tmp_path)
    with_root = read_digest(str(plate))
    without_root = dataclasses.replace(with_root, root="")

    printed_with = format_digest(with_root).splitlines()
    printed_without = format_digest(without_root).splitlines()

    project_line = f"  project: {str(plate)}"
    assert project_line in printed_with
    assert printed_without == [ln for ln in printed_with if ln != project_line]
    assert not any(ln.startswith("  project:") for ln in printed_without)
    # the rest of the card is unchanged, so the missing line is the only edit
    assert any("cell:" in ln for ln in printed_without)
    assert printed_without[0].startswith("Segmentation QC:")
