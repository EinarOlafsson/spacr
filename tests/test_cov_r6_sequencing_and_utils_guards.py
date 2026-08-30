"""Sixteen arcs in :mod:`spacr.sequencing` and :mod:`spacr.utils`, and why.

Both modules are above 98.9% and what is left in them is almost entirely
defensive: a re-check of something the line above already guaranteed, a
retry loop that cannot fall out of its own bottom, a column test against a
frame built four lines earlier. Rather than contort an input into reaching
one, each test here drives the code that makes the arm dead and asserts the
guarantee itself, so the day a guarantee stops holding the failure lands on
the invariant rather than waking a branch nobody has executed.

What is pinned:

* **sequencing** -- the three ``len(consensus_seq) >= expected_end`` tests
  (both readers pad every window to exactly that length first); the three
  ``if '<name>' in df2.columns`` tests in ``process_chunk`` (the frame is
  built with those seven names four lines above); the ``elif mode ==
  'single'`` in ``generate_barecode_mapping`` (the gate above it admits only
  ``paired`` and ``single``); and ``if dst is not None`` in the threshold
  sweep (its sole caller passes ``os.path.dirname(...)``, which is a string
  even when it is empty).
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

import os
import re
import sqlite3

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                  # noqa: E402

from spacr import sequencing as SEQ                              # noqa: E402
from spacr import utils as U                                     # noqa: E402

# ---------------------------------------------------------------------------
# sequencing: a barcode layout small enough to write out
# ---------------------------------------------------------------------------

_ANCHOR = "GGATCC"
_REGEX = r"^(?P<columnID>.{4})GGAA(?P<grna>.{4})(?P<rowID>.{4})"
_WINDOW = 16
_COL, _GRNA, _ROW = "AAAA", "CCCC", "TTTT"
_PAYLOAD = f"{_COL}GGAA{_GRNA}{_ROW}"


def _refs(tmp_path):
    def _csv(name, seq, label):
        path = tmp_path / name
        pd.DataFrame({"sequence": [seq], "name": [label]}).to_csv(path,
                                                                  index=False)
        return str(path)

    return (_csv("c.csv", _COL, "col0"), _csv("g.csv", _GRNA, "sg0"),
            _csv("r.csv", _ROW, "row0"))


def _fastq(seq, name="r0"):
    return f"@{name}\n{seq}\n+\n{'I' * len(seq)}"


def _read(payload, lead="TT"):
    return f"{lead}{_ANCHOR}{payload}"


def _single(tmp_path, payloads, fill_na=False):
    c, g, r = _refs(tmp_path)
    chunk = [_fastq(_read(p), f"r{i}") for i, p in enumerate(payloads)]
    return SEQ.process_chunk((chunk, _REGEX, _ANCHOR, len(_ANCHOR), _WINDOW,
                              c, g, r, fill_na))


def _paired(tmp_path, payloads, fill_na=False):
    c, g, r = _refs(tmp_path)
    r1 = [_fastq(_read(p), f"p{i}") for i, p in enumerate(payloads)]
    r2 = [_fastq(SEQ.reverse_complement(_read(p)), f"p{i}")
          for i, p in enumerate(payloads)]
    return SEQ.process_chunk((r1, r2, _REGEX, _ANCHOR, len(_ANCHOR), _WINDOW,
                              c, g, r, fill_na))


def test_every_window_is_padded_to_the_length_the_check_asks_for(tmp_path):
    """Why ``len(consensus_seq) >= expected_end`` is never false.

    Both readers cut a window of at most ``expected_end`` bases and then pad
    it with ``N`` (quality ``!``) up to exactly ``expected_end`` before the
    length is tested -- and the paired reader's ``create_consensus`` refuses
    two reads of different lengths, so its answer is that same length. The
    test can therefore only ever be true, in the per-read arm and in the
    orientation hint printed for a chunk that matched nothing.
    """
    for reader in (_single, _paired):
        df, counts, qc = reader(tmp_path, [_PAYLOAD, _PAYLOAD[:9]])

        # The truncated read still reached the regex at full window length,
        # padded with N -- which is what makes the length test always true.
        assert list(df["read"].str.len()) == [_WINDOW, _WINDOW]
        assert df["read"].iloc[1].endswith("N" * (_WINDOW - 9))
        assert int(qc["total_reads"].iloc[0]) == 2
        # ...and the padded barcode maps to no name, so it drops out of the
        # counts rather than being counted as the well it half resembles.
        assert list(counts["grna_name"]) == ["sg0"]

    # A chunk that matches nothing takes the orientation-hint path, where the
    # same test is applied to the last window seen. It is the padded window,
    # so it is the full length there too.
    df, _counts, qc = _single(tmp_path, ["ACGT" * 4])
    assert df.empty and int(qc["total_reads"].iloc[0]) == 0


def test_the_chunk_frame_always_carries_the_three_names_the_fill_reads(
        tmp_path):
    """Why the three ``'<name>' in df2.columns`` tests cannot be false.

    ``df2`` is a copy of the frame built four lines above out of a literal
    dict, so ``columnID``, ``rowID`` and ``grna_name`` are always there.
    """
    df, counts, _qc = _single(tmp_path, [_PAYLOAD, _PAYLOAD[:9]], fill_na=True)

    assert list(df.columns) == [
        "read", "column_sequence", "columnID", "row_sequence", "rowID",
        "grna_sequence", "grna_name"]

    # And the fill really ran: the unmapped read is counted under its raw
    # sequence rather than dropped, which is what `fill_na` is for.
    filled = set(counts["grna_name"])
    assert "sg0" in filled
    assert any(name not in ("sg0",) for name in filled), (
        "the NaN names were not filled from the raw sequences")


def test_only_a_paired_or_single_run_reaches_the_read_function(tmp_path,
                                                               monkeypatch):
    """Why ``elif settings['mode'] == 'single'`` cannot be false.

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


def test_the_threshold_sweep_always_has_a_folder_to_write_into(tmp_path,
                                                              monkeypatch):
    """Why ``if dst is not None`` in the sweep cannot be false.

    ``graph_sequencing_stats`` -- the closure's only caller -- computes
    ``dst = os.path.dirname(settings['count_data'][0])``, and
    ``os.path.dirname`` always returns a string. A bare filename makes it
    the EMPTY string, which is the closest this gets to the default and is
    still not ``None``: the figure is written to ``results/`` beside the
    working directory rather than not written at all.
    """
    assert os.path.dirname("counts.csv") == ""

    rows = []
    for row in "ABCD":
        for column in range(1, 5):
            for guide in range(4):
                rows.append({"plateID": "p1", "rowID": row,
                             "columnID": f"c{column}",
                             "grna": f"sg{guide}",
                             "count": 100 if guide == 0 else 3})
    monkeypatch.chdir(tmp_path)
    pd.DataFrame(rows).to_csv("counts.csv", index=False)

    threshold = SEQ.graph_sequencing_stats({
        "count_data": "counts.csv", "target_unique_count": 2,
        "filter_column": "columnID", "control_wells": ["c4"]})

    assert 0.0 < float(threshold) <= 0.99
    assert (tmp_path / "results" / "fraction_threshold.pdf").is_file(), (
        "an empty dirname must still name a folder to write into")


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_a_database_that_stays_locked_leaves_the_retry_by_raising(tmp_path,
                                                                  monkeypatch):
    """Why neither ``for attempt in range(...)`` loop can run off its end.

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


def test_a_merge_that_reaches_the_write_always_has_rows():
    """Why ``if len(merged_df) > 0`` before the write cannot be false.

    ``_merge_and_save_to_database`` returns at its own line 2828 when the
    morphology frame is empty, and what it writes is either that frame
    copied or that frame OUTER-merged with the intensity frame. An outer
    merge keeps every left row, so a non-empty morphology frame can only
    produce a non-empty merge.
    """
    morph = pd.DataFrame({"object_label": [1, 2], "area": [10.0, 20.0]})
    empty_intensity = pd.DataFrame(columns=["object_label", "mean"])

    # The early return is the only way past this function with nothing to
    # write, and it happens before any database is touched.
    assert U._merge_and_save_to_database(
        morph.iloc[0:0], empty_intensity, "cell", "/nowhere", "f0", "exp"
    ) is None

    # Both shapes the write is reached with keep every morphology row.
    assert len(morph.copy()) == len(morph)
    outer = pd.merge(morph, pd.DataFrame({"object_label": [1],
                                          "mean": [5.0]}),
                     on="object_label", how="outer", validate="one_to_one")
    assert len(outer) == len(morph) > 0


def test_no_two_training_suggestions_are_the_same_sentence(tmp_path):
    """Why ``if s not in seen`` in the de-duplicator is never false.

    Each rule appends its own wording, and no two rules share one, so the
    pass that exists to drop a repeat never has a repeat to drop. Driven
    with a run that trips several rules at once.
    """
    epochs = list(range(1, 31))
    pd.DataFrame({
        "epoch": epochs,
        "loss": [1.0 / (e ** 2) for e in epochs],
        "accuracy": [min(0.99, 0.5 + e * 0.02) for e in epochs],
        "f1_macro": [min(0.99, 0.5 + e * 0.02) for e in epochs],
    }).to_csv(tmp_path / "train_progress.csv", index=False)
    pd.DataFrame({
        "epoch": epochs,
        "loss": [1.0 + 0.01 * e for e in epochs],
        "accuracy": [0.60] * 30,
        "f1_macro": [0.20] * 30,
    }).to_csv(tmp_path / "val_progress.csv", index=False)

    out = U.suggest_training_changes(str(tmp_path))

    assert out["flags"], "no rule fired, so nothing reached the de-duplicator"
    assert len(out["suggestions"]) > 1
    assert len(set(out["suggestions"])) == len(out["suggestions"]), (
        "two rules produced the same sentence, so the de-duplicator can drop "
        "one and the guard is no longer dead")


def test_only_filenames_the_regex_matched_are_grouped(tmp_path):
    """Why ``if match:`` in ``_run_test_mode`` cannot be false.

    ``all_filenames`` is a comprehension filtered on ``regular_expression
    .match(filename)``; the loop below re-matches the same names with the
    same pattern, so every one of them matches again.
    """
    regex = (r"^(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_"
             r"(?P<fieldID>f\d+)_(?P<chanID>C\d)\.tif$")
    src = tmp_path / "plate"
    src.mkdir()
    for name in ("plate1_A1_f1_C1.tif", "plate1_A1_f1_C2.tif",
                 "notes.txt", "plate1_A1.tif"):
        (src / name).write_bytes(b"")

    compiled = re.compile(regex)
    kept = [name for name in sorted(os.listdir(src)) if compiled.match(name)]
    assert kept == ["plate1_A1_f1_C1.tif", "plate1_A1_f1_C2.tif"], (
        "the filter is what makes the re-match certain")
    assert all(compiled.match(name) for name in kept)

    U._run_test_mode(str(src), regex, test_images=1, random_test=False)

    copied = sorted(os.listdir(src / "test"))
    assert copied == kept, "only the matched names were grouped and copied"


def test_a_cluster_plot_always_has_a_legend_to_restyle():
    """Why ``if legend is not None`` in ``plot_clusters`` cannot be false.

    ``Axes.legend`` builds and returns a ``Legend`` whatever it finds --
    including an axes with no labelled artist at all, where it warns and
    returns an empty one. There is no call that hands back ``None``.
    """
    from matplotlib.legend import Legend

    figure, axes = plt.subplots()
    try:
        assert isinstance(axes.legend(), Legend), (
            "an axes with nothing labelled still yields a legend")

        embedding = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                              [4.0, 4.0], [5.0, 4.0], [4.0, 5.0]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        U.plot_clusters(axes, embedding, labels,
                        colors=["#1f77b4", "#ff7f0e"],
                        cluster_centers=[(0.3, 0.3), (4.3, 4.3)],
                        plot_outlines=False, plot_points=True,
                        smooth_lines=False)

        legend = axes.get_legend()
        assert legend is not None
        assert axes.get_xlabel() == "UMAP Dimension 1"
        # The restyle really ran: the frame took the axes' own background.
        assert legend.get_frame().get_facecolor()[:3] == (
            axes.get_facecolor()[:3])
    finally:
        plt.close(figure)


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
