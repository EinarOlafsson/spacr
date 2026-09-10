"""Two guards in the montage that no input can trip, and why.

Round 4 asked for ``cell_montage`` lines 1306-1307 (``sudoku: no cell sits in
a well holding this guide``) and the false side of ``if root:`` at line 2575.
Neither is reachable: each is a re-check of something the code just above it
has already guaranteed. Rather than fake them, this file pins the two
invariants that make them dead, so that if either guarantee is ever broken the
failure lands here -- on the invariant -- instead of silently waking a branch
nobody has ever executed.

The proofs are written out beside each test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spacr import cell_montage as cm


# One well holds GRA14_1. Of the other three, exactly one holds a rival at or
# above the anchoring share, so the sudoku scope is a strict subset of the
# plate -- which is what makes the "kept wells" assertions below meaningful.
FRACTIONS = {
    "r1_c1": {"GRA14_1": 0.4, "OTHER_1": 0.6},    # the guide's own well
    "r1_c2": {"OTHER_2": 0.7, "OTHER_3": 0.3},    # no rival of GRA14_1 here
    "r1_c3": {"OTHER_1": 0.55, "OTHER_2": 0.45},  # OTHER_1 anchors at >= 0.5
    "r1_c4": {"OTHER_2": 1.0},                    # no rival of GRA14_1 here
}
PER_WELL = 8


def _counts(fractions=FRACTIONS):
    """A ``regression_data.csv``-shaped count frame."""
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id,
                "grna": guide, "gene": guide.split("_")[0],
                "fraction": fraction, "cell_count": PER_WELL,
            })
    return pd.DataFrame(rows)


def _objects(fractions=FRACTIONS):
    """The per-object frame, ``PER_WELL`` scored cells in every well."""
    rows = []
    for index, well in enumerate(fractions):
        row_id, column_id = well.split("_")
        for label in range(1, PER_WELL + 1):
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id, "fieldID": "f1",
                "object_label": label,
                "pred": round(0.05 + 0.9 * (label - 1) / (PER_WELL - 1), 4),
                "area": 100.0 + label + 10 * index,
                "perimeter": 40.0 + 2 * label,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# `_sudoku_calls`: the scope is always wells the frame has cells in
# ---------------------------------------------------------------------------
#
# UNREACHABLE, with the proof, for cell_montage.py lines 1305-1307:
#
#   1246  wells = [str(w) for w in frame["_montage_well"]]
#   1248  for label in sorted(set(wells)):
#   1251      if here: fractions[label] = here
#   1284  mine = [label for label, here in fractions.items() if ...]
#   1286  if not mine: return None
#   1297  keep = set(mine) | anchoring
#   1304  rows = [i for i, w in enumerate(wells) if w in keep]
#   1305  if not rows:              <-- never true
#
# `fractions` is keyed only by labels drawn from `set(wells)` (line 1248), so
# every key is a label some row of `frame` carries. `mine` is a subset of
# those keys, and line 1286 has already returned when it is empty. `keep` is a
# superset of `mine`. So line 1304 finds, at minimum, every row belonging to
# each well in `mine` -- at least one -- and `rows` cannot be empty. `frame`
# is not rebound between 1246 and 1304 (1226 is the last rebinding, 1308 the
# next), so `wells` still describes the rows being indexed.
#
# This is the "defensive re-check after a call that already guarantees the
# condition" family. The test pins the guarantee.

def test_sudokus_scope_is_the_guides_wells_plus_the_wells_that_anchor_it():
    """The scope is a subset of the wells that have cells, and never empty.

    Trimming to the guide's own wells alone was the first version, and it was
    wrong: the graph learns what a guide looks like from every well it is in,
    so a rival characterised only from the share it happens to hold here is
    sampled at its weakest. The scope is therefore the guide's wells plus the
    wells where a rival is large enough to anchor -- and, because those labels
    are read off the frame's own rows, it always selects some cells.
    """
    objects = _objects()
    notes: list = []

    # A private call: the public `select_montage` reports which cells were
    # ringed, but not which wells sudoku was allowed to learn from, and the
    # scope is the thing under test.
    result = cm._sudoku_calls(objects, _counts(), ["prc"], "grna", "fraction",
                              "pred", "GRA14_1", notes)

    assert result is not None
    kept_wells = {objects.loc[index, "prc"] for index in result}
    assert kept_wells == {"plate1_r1_c1", "plate1_r1_c3"}, (
        "the scope is not the guide's well plus its rival's anchoring well"
    )
    # Present and absent in the same breath: the guide's own well is in, the
    # two wells holding none of its rivals are out.
    assert "plate1_r1_c1" in kept_wells
    assert "plate1_r1_c2" not in kept_wells
    assert "plate1_r1_c4" not in kept_wells

    # The invariant the dead branch rests on: every kept well is a well the
    # frame has rows for, so the row selection can never come back empty.
    assert set(result) <= set(objects.index)
    assert len(result) == 2 * PER_WELL
    assert any("1 well(s) hold GRA14_1" in note for note in notes)


# ---------------------------------------------------------------------------
# `resolve_montage_crop_source`: a source with no root has already returned
# ---------------------------------------------------------------------------
#
# UNREACHABLE, with the proof, for cell_montage.py line 2575:
#
#   2559  source = resolve_crop_source(src, ...)      # raises CropError
#   2561  except CropError as exc: return CropSourceChoice(available=False...)
#   2572  root = src.get("src") if isinstance(src, Mapping) else src
#   2573  if isinstance(root, (list, tuple)): root = root[0] if root else None
#   2575  if root:                                    <-- never false
#
# `spacr/crops.py:resolve_crop_source` computes the SAME value by the same
# three steps (crops.py lines 3564-3571: `settings.get("src")` for a Mapping,
# else the argument; then the list/tuple unwrap) and raises `CropError` at
# crops.py line 3572-3573 -- `if not src` -- before returning anything. So by
# the time line 2575 runs, that value has already been proved truthy by the
# call on line 2559; a falsy one left through the `except` on line 2561.
#
# Same family as above. The test pins the guarantee at the boundary.

def test_a_settings_map_with_no_source_root_is_answered_not_raised(tmp_path):
    """A missing source arrives as a sentence the tab can display.

    That is the whole reason this wrapper exists over
    ``crops.resolve_crop_source``: the montage tab has to stay on screen and
    say why there are no crops, rather than catch an exception. Every spelling
    of "no root" is refused at that boundary, which is why the channel lookup
    below it never sees one.
    """
    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    np.save(root / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((8, 8, 3), np.uint16))

    usable = cm.resolve_montage_crop_source({"src": str(root)})
    assert usable.available is True
    assert usable.kind == "merged"

    for empty in ({"src": ""}, {"src": []}, {}, ""):
        choice = cm.resolve_montage_crop_source(empty)
        assert choice.available is False, f"{empty!r} was accepted as a source"
        assert "no 'src'" in choice.reason
        assert choice.source is None
        assert choice.requirements.route == "none"
        # And it never reached the channel lookup: that is the only thing
        # past the guard that can add a requirement note.
        assert choice.requirements.missing == (choice.reason,)


def test_a_run_with_no_measurements_database_still_resolves_its_crops(tmp_path):
    """The channel lookup is optional; the source is not.

    A plate whose ``measurements.db`` has not been written yet -- or was left
    behind -- still has pixels to cut, and the montage has to draw them. What
    changes is only whether the run's own channel mapping could be read.
    """
    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((8, 8, 3), np.uint16))
    assert not (root / "measurements" / "measurements.db").exists()

    choice = cm.resolve_montage_crop_source({"src": str(root)})

    assert choice.available is True
    assert choice.kind == "merged"
    # With no database to read the run's own mapping from, the caption has to
    # say which planes were assumed rather than chosen.
    assert any("no channel list" in note
               for note in choice.requirements.assumed)

    # Asking for channels explicitly answers the same question the other way,
    # and the assumption disappears -- which is what makes the assertion above
    # a statement about the lookup rather than about this route.
    declared = cm.resolve_montage_crop_source({"src": str(root)},
                                              channels=[0, 1, 2])
    assert declared.available is True
    assert not any("no channel list" in note
                   for note in declared.requirements.assumed)
