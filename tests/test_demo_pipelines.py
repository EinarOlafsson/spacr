"""The Demos menu, run for real and asserted on its output.

Every entry under **Demos** in the Qt GUI generates a dataset with
``spacr.qt.synthetic`` and then feeds it to a real pipeline. Until this module
existed, the demos were covered only by ``tests/qt/test_demo_menu.py``, which
checks that a generator returns a layout, that its settings CSV clears
pre-flight, and that the settings survive the widget round trip. All three are
worth having and none of them runs a pipeline — so three demos were broken in
the field while that file was green:

* **Timelapse** died in ``preprocess_generate_masks`` with
  ``no such table: object_counts``. ``spacr.io._rename_and_organize_image_files``
  dropped the timeID from its stack filenames whenever ``timelapse=True``, which
  max-projected all N frames of a field into one array *and* made
  ``_generate_time_lists`` skip the whole plate, so no normalised stack and no
  mask was ever written.
* **Timelapse**, once past that, hit ``'list' object has no attribute 'shape'``
  100 times inside ``_facilitate_trackin_with_adaptive_removal``'s retry loop
  and reported it as ``Failed to track after 100 attempts`` — a message about
  displacement for a bug about a type.
* **Classify** hung forever at the end of epoch 1.
  ``deep_spacr._plot_training_curves`` called a blocking ``plt.show()`` from
  inside the training loop, and matplotlib picks the interactive 'qtagg'
  backend on every machine with PySide6 installed.

So the rule for this module: assert on what the run *produced*. File counts,
label counts in the mask planes, table names and row counts in the database,
tracks per frame, reads mapped. Never "it did not raise".

The generator-output tests are offline and fast. The pipeline tests carry
``slow``/``gpu`` and skip themselves when the resource is missing; the mask,
timelapse and classify pipelines additionally need model weights (Cellpose
``cpsam``, torchvision ResNet-50) that are fetched once and cached, so they
also want a network on a cold machine.
"""
from __future__ import annotations

import gzip
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import synthetic as syn
from tests.resource_capabilities import cuda_available, package_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_demo_settings(layout):
    """Read a demo's settings CSV exactly the way the Demos menu does."""
    from spacr.utils import load_settings
    return load_settings(str(layout.settings_csv),
                         setting_key="Key", setting_value="Value")


def _tables(db_path):
    """``{table_name: row_count}`` for a measurements database."""
    with sqlite3.connect(str(db_path)) as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                for n in names}


def _columns(db_path, table):
    with sqlite3.connect(str(db_path)) as conn:
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def _label_count(plane):
    """Number of distinct non-zero labels in a mask plane."""
    return int(len(np.unique(plane)) - (1 if (plane == 0).any() else 0))


def _requires_gpu():
    if not cuda_available():
        pytest.skip("no CUDA device: the segmentation demos need Cellpose 4")
    if not package_available("cellpose"):
        pytest.skip("cellpose is not installed")


# ---------------------------------------------------------------------------
# What the generators actually write — offline, no GPU
# ---------------------------------------------------------------------------

def test_mask_demo_writes_a_full_yokogawa_plate(tmp_path):
    """2 wells x 2 fields x 4 channels, every name parsed by spaCR's own regex.

    The filename convention is not decoration: ``metadata_type='cellvoyager'``
    in the demo's settings makes ``_rename_and_organize_image_files`` parse
    these names with exactly this regex, and a name it cannot parse is silently
    skipped rather than reported.
    """
    from spacr.utils import _get_regex

    layout = syn.generate_mask_demo(tmp_path / "mask")
    tifs = sorted(p.name for p in Path(layout.image_dir).glob("*.tif"))
    assert len(tifs) == 16, tifs

    pattern = re.compile(_get_regex("cellvoyager", "tif"))
    unparsed = [name for name in tifs if not pattern.match(name)]
    assert not unparsed, (
        f"{unparsed} do not match spaCR's cellvoyager regex, so the mask "
        "pipeline would skip them without saying so")

    wells = {pattern.match(n).group("wellID") for n in tifs}
    channels = {pattern.match(n).group("chanID") for n in tifs}
    assert wells == {"A01", "A02"}
    assert channels == {"00", "01", "02", "03"}

    # Non-degenerate pixels: 16-bit, real dynamic range, no blank channel.
    import tifffile
    for name in tifs:
        img = tifffile.imread(str(Path(layout.image_dir) / name))
        assert img.dtype == np.uint16 and img.shape == syn.FIELD_SHAPE
        assert img.max() > 10 * syn._BACKGROUND, (
            f"{name} carries no objects — peak {img.max()} is at the "
            f"background level ({syn._BACKGROUND})")


def test_measure_demo_merged_stacks_carry_nested_non_degenerate_masks(tmp_path):
    """measure_crop reads ``merged/*.npy``; assert the label planes are real.

    A stack whose mask planes are empty (or hold one giant blob) still "runs" —
    it produces a measurements database with no rows. The counts and the
    nesting below are what makes the demo's cell/nucleus/pathogen *links*
    measurable at all.
    """
    layout = syn.generate_measure_demo(tmp_path / "measure")
    assert len(layout.merged_files) == 4, layout.merged_files

    roles = syn.MASK_ROLE_ORDER          # cell, nucleus, pathogen, organelle
    for path in layout.merged_files:
        arr = np.load(path)
        assert arr.shape == (*syn.FIELD_SHAPE, 8) and arr.dtype == np.uint16

        planes = {role: arr[..., 4 + i] for i, role in enumerate(roles)}
        counts = {r: _label_count(p) for r, p in planes.items()}
        assert counts["cell"] == syn.CELL_GRID ** 2 == 16, counts
        assert counts["nucleus"] == 16, counts
        # 12 infected cells x 1-2 pathogens each.
        assert 12 <= counts["pathogen"] <= 24, counts
        assert counts["organelle"] == 16 * syn._ORGANELLES_PER_CELL == 64, counts

        cell = planes["cell"]
        for child in ("nucleus", "pathogen", "organelle"):
            inside = (cell[planes[child] > 0] > 0).mean()
            assert inside == 1.0, (
                f"{Path(path).name}: {1 - inside:.1%} of the {child} pixels "
                "fall outside every cell, so measure_crop cannot link them")


def test_crop_demo_ships_one_settings_file_and_turns_pngs_on(tmp_path):
    layout = syn.generate_crop_demo(tmp_path / "crop")
    settings = _load_demo_settings(layout)
    assert settings["save_png"] is True
    assert settings["png_size"] == [64, 64]
    assert settings["crop_mode"] == ["cell"]
    found = sorted(p.name for p in Path(layout.src).glob("settings*.csv"))
    assert found == ["settings_crop.csv"], found


def test_classify_demo_writes_labelled_crops_and_a_png_list(tmp_path):
    """Crops on disk AND the png_list rows that select them into classes.

    ``generate_training_dataset`` filters ``png_path.str.contains('cell_png')``
    and builds classes from the ``annotate`` column, so both halves have to
    agree or the run dies on "got 0 classes".
    """
    layout = syn.generate_classify_demo(tmp_path / "classify")
    pngs = sorted(Path(layout.image_dir).rglob("*.png"))
    assert len(pngs) == 64, len(pngs)
    assert all(syn.CROP_FOLDER in p.parts for p in pngs), (
        "crops outside a cell_png/ folder are filtered away before any class "
        "is built")

    from PIL import Image
    sample = np.asarray(Image.open(pngs[0]))
    assert sample.shape == (64, 64, 3) and sample.max() > sample.min()

    with sqlite3.connect(str(layout.db_path)) as conn:
        rows = conn.execute(
            "SELECT annotate, COUNT(*), COUNT(DISTINCT wellID) "
            "FROM png_list GROUP BY annotate").fetchall()
    assert {r[0] for r in rows} == {1, 2}, rows
    assert sum(r[1] for r in rows) == 64, rows
    # Four wells so the well-grouped train/test holdout still leaves two
    # distinct groups on the training side (io.make_validation_holdout).
    assert all(r[2] == 4 for r in rows), rows


def test_timelapse_demo_holds_the_same_cells_moving_across_every_frame(tmp_path):
    """8 timepoints per field, and the cells persist rather than being redrawn.

    A tracker needs objects that are *the same objects* a few pixels away. If
    each frame were an independent random field, the demo would produce 8x16
    one-frame tracks and still "complete".
    """
    layout = syn.generate_timelapse_demo(tmp_path / "timelapse")
    tifs = sorted(p.name for p in Path(layout.image_dir).glob("*.tif"))
    assert len(tifs) == 16, tifs          # 8 timepoints x 2 channels
    times = {n.split("_T")[1][:2] for n in tifs}
    assert times == {f"{t:02d}" for t in range(1, 9)}, times

    settings = _load_demo_settings(layout)
    assert settings["timelapse"] is True
    # A slice of frame indices, not an inclusive 1-based range: [1, 8] would
    # throw away frame 0 of every field.
    assert settings["timelapse_frame_limits"] == [0, 8]
    assert settings["timelapse_mode"] == "iou"

    # The cell channel of consecutive frames must differ (they moved) but stay
    # highly correlated (they are the same cells).
    import tifffile
    frames = [
        tifffile.imread(str(Path(layout.image_dir) / syn.cellvoyager_filename(
            well="A01", time=t, field=1, chan=1)))
        for t in range(1, 9)
    ]
    for a, b in zip(frames, frames[1:]):
        assert not np.array_equal(a, b), "frames are identical — nothing moved"
        corr = np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1]
        assert corr > 0.5, (
            f"consecutive frames correlate at {corr:.2f}; these are different "
            "fields, not the same field a moment later")


def test_map_barcodes_demo_reads_carry_the_planted_triplets(tmp_path):
    """Illumina FASTQ pair + three barcode references, all self-consistent.

    Every read is checked against the same regex the shipped
    ``generate_barecode_mapping`` defaults use, and the recovered barcodes are
    checked against the reference CSVs — so a demo whose reads no longer parse
    fails here rather than 5,000 reads later with an empty output table.
    """
    import csv as _csv

    layout = syn.generate_map_barcodes_demo(tmp_path / "seq", n_reads=200)
    names = sorted(p.name for p in Path(layout.src).glob("*.fastq.gz"))
    assert names == ["demo_R1_001.fastq.gz", "demo_R2_001.fastq.gz"], names

    def _refs(name):
        with open(Path(layout.src) / syn.BARCODE_DIRNAME / name) as fh:
            return {row["sequence"]: row["name"] for row in _csv.DictReader(fh)}

    grnas, rows, columns = _refs("grna.csv"), _refs("row.csv"), _refs("column.csv")
    assert len(grnas) == 12 and len(rows) == 4 and len(columns) == 6

    from spacr.settings import DEFAULT_BARCODE_REGEX
    pattern = re.compile(DEFAULT_BARCODE_REGEX)
    settings = _load_demo_settings(layout)
    anchor = settings["target_sequence"]

    n_reads = 0
    with gzip.open(Path(layout.src) / "demo_R1_001.fastq.gz", "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                assert line.startswith("@"), line
                continue
            if i % 4 != 1:
                continue
            read = line.strip()
            n_reads += 1
            assert len(read) == syn.FASTQ_READ_LENGTH, read
            start = read.index(anchor) + settings["offset_start"]
            window = read[start:start + settings["expected_end"]]
            match = pattern.match(window)
            assert match, f"read {n_reads} does not parse: {window}"
            assert match.group("columnID") in columns
            assert match.group("rowID") in rows
            assert match.group("grna") in grnas
    assert n_reads == 192, n_reads       # 200 // 24 wells = 8 reads x 24


# ---------------------------------------------------------------------------
# The pipelines, run on the demo folders
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_measure_demo_writes_every_expected_measurement_table(tmp_path):
    """Run ``measure_crop`` on the measure demo and read the database.

    One row per object per field: 4 fields x 16 cells = 64, and the same for
    nuclei and cytoplasms. Pathogens are 12-24 per field, so the total is
    bounded rather than fixed.
    """
    from spacr.measure import measure_crop

    layout = syn.generate_measure_demo(tmp_path / "measure")
    measure_crop(_load_demo_settings(layout))

    db = Path(layout.src) / "measurements" / "measurements.db"
    assert db.is_file(), "measure_crop wrote no measurements database"
    counts = _tables(db)
    for table in ("cell", "nucleus", "cytoplasm", "pathogen"):
        assert table in counts, sorted(counts)
    assert counts["cell"] == 64, counts
    assert counts["nucleus"] == 64, counts
    assert counts["cytoplasm"] == 64, counts
    assert 48 <= counts["pathogen"] <= 96, counts

    cols = _columns(db, "cell")
    for expected in ("object_label", "plateID", "rowID", "columnID",
                     "fieldID", "prcf", "file_name",
                     "cell_area", "cell_eccentricity", "cell_solidity",
                     "cell_channel_0_mean_intensity",
                     "cell_channel_3_mean_intensity"):
        assert expected in cols, (expected, cols[:20])

    # Non-degenerate geometry: the demo draws cells at _RADIUS_CELL, so the
    # measured areas have to be in the right order of magnitude and vary.
    import pandas as pd
    with sqlite3.connect(str(db)) as conn:
        area = pd.read_sql("SELECT cell_area FROM cell", conn)["cell_area"]
    assert area.min() > 100 and area.max() < 5000, area.describe()
    assert area.std() > 0, "every cell measured the identical area"


@pytest.mark.slow
def test_crop_demo_writes_one_png_per_measured_cell(tmp_path):
    """Crop is Measure with ``save_png``; assert the crops exist and carry signal."""
    from PIL import Image
    from spacr.measure import measure_crop

    layout = syn.generate_crop_demo(tmp_path / "crop")
    measure_crop(_load_demo_settings(layout))

    pngs = sorted((Path(layout.src) / "data").rglob(f"{syn.CROP_FOLDER}/*.png"))
    assert len(pngs) == 64, f"expected one crop per cell, got {len(pngs)}"
    # np.ptp(arr), not arr.ptp(): the ndarray method was removed in NumPy 2.0.
    blank = [p for p in pngs if np.ptp(np.asarray(Image.open(p))) == 0]
    assert not blank, f"{len(blank)} crops are a single flat value: {blank[:3]}"


@pytest.mark.slow
def test_map_barcodes_demo_maps_every_read_it_planted(tmp_path):
    """Run the sequencing pipeline and check nothing was lost.

    ``qc.csv`` counts unmapped reads per field; every one of them must be zero,
    because the demo planted barcodes that are in the reference CSVs. The
    combination table then has to account for every read.
    """
    import pandas as pd
    from spacr.sequencing import generate_barecode_mapping

    layout = syn.generate_map_barcodes_demo(tmp_path / "seq", n_reads=480)
    generate_barecode_mapping(_load_demo_settings(layout))

    out = Path(layout.src) / f"{syn.FASTQ_SAMPLE}_paired"
    combos = pd.read_csv(out / "unique_combinations.csv")
    qc = pd.read_csv(out / "qc.csv")

    assert qc["total_reads"].sum() == 480, qc
    for column in ("read", "column_sequence", "columnID", "row_sequence",
                   "rowID", "grna_sequence", "grna_name"):
        assert qc[column].sum() == 0, (
            f"{int(qc[column].sum())} reads failed on '{column}' although "
            "every planted barcode is in the reference CSVs")

    assert combos["count"].sum() == 480, combos["count"].sum()
    assert set(combos["rowID"]) == {f"r{i + 1}" for i in range(4)}
    assert set(combos["columnID"]) == {f"c{i + 1}" for i in range(6)}
    assert combos["grna_name"].nunique() == 12, combos["grna_name"].nunique()


@pytest.mark.gpu
@pytest.mark.slow
def test_mask_demo_segments_every_object_it_drew(tmp_path):
    """Run the real Mask pipeline (Cellpose 4 / cpsam) on the mask demo.

    The demo draws a 4x4 lattice of cells with one nucleus each, so a working
    run comes back with 16 of each per field. Anything much lower means the
    diameters or the normalisation the demo advertises stopped matching the
    pixels it draws.
    """
    _requires_gpu()
    from spacr.core import preprocess_generate_masks

    layout = syn.generate_mask_demo(tmp_path / "mask")
    preprocess_generate_masks(_load_demo_settings(layout))

    merged = sorted((Path(layout.src) / "merged").glob("*.npy"))
    assert len(merged) == 4, [p.name for p in merged]
    for path in merged:
        arr = np.load(path)
        assert arr.shape == (*syn.FIELD_SHAPE, 8), arr.shape
        counts = {role: _label_count(arr[..., 4 + i])
                  for i, role in enumerate(syn.MASK_ROLE_ORDER)}
        assert counts["cell"] == 16, (path.name, counts)
        assert counts["nucleus"] == 16, (path.name, counts)
        assert counts["pathogen"] >= 12, (path.name, counts)
        assert counts["organelle"] >= 48, (path.name, counts)


@pytest.mark.gpu
@pytest.mark.slow
def test_timelapse_demo_tracks_the_same_cells_through_every_frame(tmp_path):
    """Run the Timelapse pipeline and assert on the tracks, not on completion.

    This is the regression guard for both timelapse defects. Before the io.py
    fix the run died in ``_pivot_counts_table`` on ``no such table:
    object_counts``; before the timelapse.py fix it died on ``Failed to track
    after 100 attempts``. A run that produced one merged stack instead of eight
    — the silent half of the first bug — is caught by the count below.
    """
    _requires_gpu()
    import pandas as pd
    from spacr.core import preprocess_generate_masks_timelapse

    layout = syn.generate_timelapse_demo(tmp_path / "timelapse")
    preprocess_generate_masks_timelapse(_load_demo_settings(layout))

    merged = sorted((Path(layout.src) / "merged").glob("*.npy"))
    assert len(merged) == 8, (
        f"expected one merged stack per timepoint, got "
        f"{[p.name for p in merged]}")

    tracks_files = sorted((Path(layout.src) / "tracks").glob("*.csv"))
    assert len(tracks_files) == 1, [p.name for p in tracks_files]
    tracks = pd.read_csv(tracks_files[0])
    assert sorted(tracks["frame"].unique()) == list(range(8)), (
        "a track table that does not span all 8 frames means the frames were "
        "collapsed before tracking")
    assert tracks["track_id"].nunique() == 16, tracks["track_id"].nunique()
    # Every cell present in every frame: 16 x 8.
    assert len(tracks) == 128, len(tracks)

    movies = sorted((Path(layout.src) / "movies").rglob("*"))
    assert any(p.suffix in {".mp4", ".gif"} for p in movies), movies


@pytest.mark.gpu
@pytest.mark.slow
def test_classify_demo_trains_a_model_and_predicts_every_crop(tmp_path):
    """Run ``deep_spacr`` on the classify demo.

    The regression guard for the blocking ``plt.show()``: with plotting on and
    an interactive backend this used to park in the Qt main loop at the end of
    epoch 1 and never return, so the assertions below were unreachable.
    """
    _requires_gpu()
    from spacr.deep_spacr import deep_spacr

    layout = syn.generate_classify_demo(tmp_path / "classify")
    settings = _load_demo_settings(layout)
    assert settings["plot"] is False, (
        "the classify demo must ship plot=False like every other demo")
    deep_spacr(settings)

    weights = sorted((Path(layout.src) / "datasets").rglob("*.pth"))
    assert weights, "training wrote no model weights"
    assert all(p.stat().st_size > 1_000_000 for p in weights), (
        [(p.name, p.stat().st_size) for p in weights])

    with sqlite3.connect(str(layout.db_path)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("png_list")')]
        assert "pred" in cols, cols
        n_pred = conn.execute(
            "SELECT COUNT(*) FROM png_list WHERE pred IS NOT NULL").fetchone()[0]
    assert n_pred == 64, f"only {n_pred}/64 crops got a prediction"
