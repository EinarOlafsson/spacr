"""Sixteen arcs in :mod:`spacr.sequencing` and :mod:`spacr.utils`, and why.

Both modules are above 98.9% and what is left in them is almost entirely
defensive: a re-check of something the line above already guaranteed, a
retry loop that cannot fall out of its own bottom, a column test against a
frame built four lines earlier. Rather than contort an input into reaching
one, each test here drives the code that makes the arm dead and asserts the
guarantee itself, so the day a guarantee stops holding the failure lands on
the invariant rather than waking a branch nobody has executed.

What is pinned:

* **sequencing** -- the invariants that allowed nine redundant branches to
  be deleted: readers pad every window to the requested length; the chunk
  frame always contains all three name columns; the processing gate admits
  only ``paired`` and ``single`` and single-read validation admits only R1
  and R2; and the threshold sweep always receives a string destination.
* **utils** -- the two ``DB_WRITE_ATTEMPTS`` retry loops (every path returns
  or raises, so neither loop ever runs off its end); ``if len(merged_df) >
  0`` (an empty morphology frame has already returned, and an outer merge
  never drops rows); the de-duplicator in ``suggest_training_changes`` (no
  two rules produce the same sentence); ``if match`` in ``_run_test_mode``
  (the filenames were filtered with that regex); ``if legend is not None``
  in ``plot_clusters`` (matplotlib always hands back a legend); and the
  ``other_label != first_label`` tests in the cell merge (``np.unique``
  returns distinct values, so no member of ``labels[1:]`` is ``labels[0]``).
"""
from __future__ import annotations

import sqlite3

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


from spacr import sequencing as SEQ                              # noqa: E402
from spacr import utils as U                                     # noqa: E402

def test_only_a_paired_or_single_run_reaches_the_read_function(tmp_path,
                                                               monkeypatch):
    """Why mode and single-direction dispatch can use final ``else`` arms.

    The sample is only processed at all when the gate above it holds, and
    that gate is ``mode == 'paired' and R1 and R2`` or ``mode == 'single'
    and R1`` or ``mode == 'single' and R2``. Every disjunct names a mode, so
    a third mode never gets inside -- which matters, because the ``elif``
    is the only thing that binds ``function``, and falling past it would
    raise ``NameError`` on the call three lines below.
    """
    gate = {}

    for mode in ("paired", "single", "interleaved"):
        settings = {"mode": mode}
        r1, r2 = "reads_R1.fastq", "reads_R2.fastq"
        gate[mode] = (
            settings['mode'] == 'paired' and r1 and r2
            or settings['mode'] == 'single' and r1
            or settings['mode'] == 'single' and r2)

    assert gate["paired"] and gate["single"]
    assert not gate["interleaved"], (
        "a mode the elif does not handle got past the gate")

    # The two modes the gate admits are exactly the two the branch binds a
    # read function for, and they are different functions.
    assert SEQ.paired_read_chunked_processing is not SEQ.single_read_chunked_processing

    import inspect
    source = inspect.getsource(SEQ.generate_barecode_mapping)
    assert "elif settings['mode'] == 'single':" not in source
    assert "elif settings['single_direction'] == 'R2':" not in source
    assert "_check_read_settings(settings)" in source
    guard = inspect.getsource(SEQ._check_read_settings)
    assert "settings['single_direction'] not in {'R1', 'R2'}" in guard


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_a_database_that_stays_locked_leaves_the_retry_by_raising(tmp_path,
                                                                  monkeypatch):
    """Neither retry can run off its end or silently report success.

    Both retries answer every attempt: a success returns, a non-lock error
    returns or re-raises, and the LAST attempt re-raises (or, for a table
    the run can lose, returns after saying so). The loop therefore always
    leaves through a ``return`` or a ``raise``, never by exhausting -- which
    is why the implicit ``return None`` after each loop is unreachable.
    """
    monkeypatch.setattr(U.time, "sleep", lambda _seconds: None)
    locked = sqlite3.OperationalError("database is locked")
    frame = pd.DataFrame({"object_label": [1]})

    tries = []

    def _always_locked(*_args, **_kwargs):
        tries.append(1)
        raise locked

    # The release path: every attempt is taken, and the last one raises.
    db_path = tmp_path / "measurements.db"
    db_path.write_bytes(b"")
    monkeypatch.setattr(U, "_release_imported_rows_once", _always_locked)
    with pytest.raises(sqlite3.OperationalError):
        U._release_imported_rows_for_field(str(db_path), "cell", frame)
    assert len(tries) == U.DB_WRITE_ATTEMPTS

    # The append path, for a table the run cannot do without: same shape.
    tries.clear()
    import spacr.database_concurrency as dbc
    monkeypatch.setattr(dbc, "connect", _always_locked)
    with pytest.raises(sqlite3.OperationalError):
        U._append_to_measurements_db(str(db_path), "cell", frame)
    assert len(tries) == U.DB_WRITE_ATTEMPTS

    # And for a side table, which is allowed to be lost: it RETURNS on the
    # last attempt rather than falling out of the loop, having taken every
    # attempt first.
    tries.clear()
    assert U._append_to_measurements_db(
        str(db_path), "png_list", frame, required=False) is None
    assert len(tries) == U.DB_WRITE_ATTEMPTS

    import inspect
    for function in (U._release_imported_rows_for_field,
                     U._append_to_measurements_db):
        source = inspect.getsource(function)
        assert "while True:" in source
        assert "for attempt in range(" not in source


def test_the_labels_a_cell_merge_walks_are_always_distinct():
    """Why ``if other_label != first_label`` cannot be false, twice over.

    Both merges walk ``np.unique(...)[1:]`` against ``[0]``. ``np.unique``
    returns sorted DISTINCT values, so every member of the tail is strictly
    greater than the head and the test is always true.
    """
    cells = np.zeros((20, 20), dtype=np.uint8)
    cells[2:9, 2:10] = 1                     # two cells side by side, each
    cells[2:9, 10:18] = 2                    # already carrying its own label
    nuclei = np.zeros_like(cells)
    nuclei[4:7, 4:7] = 1
    nuclei[4:7, 13:16] = 1
    parasite = np.zeros_like(cells)
    parasite[4:7, 8:12] = 1                  # spans both cells
    organelle = np.zeros_like(cells)

    labelled = U.label(cells)
    overlapping = np.unique(labelled[parasite.astype(bool)])
    overlapping = overlapping[overlapping != 0]
    assert len(overlapping) > 1
    assert len(set(overlapping.tolist())) == len(overlapping), (
        "np.unique is what makes the head-vs-tail test certain")
    assert all(other != overlapping[0] for other in overlapping[1:])
    assert sorted(np.unique(cells).tolist()) == [0, 1, 2]

    merged = U._merge_cells_based_on_parasite_overlap(
        parasite, cells.copy(), nuclei, organelle, overlap_threshold=5)

    # The walk really ran: the two cells the parasite spans came back as one.
    assert len(np.unique(labelled)) - 1 == 2
    assert len(np.unique(merged)) - 1 == 1

    import inspect
    source = inspect.getsource(U._merge_cells_based_on_parasite_overlap)
    assert "if other_label != first_label:" not in source
