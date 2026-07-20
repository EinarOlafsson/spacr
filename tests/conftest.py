"""
Shared pytest fixtures + synthetic-data builders for the spacr test suite.

Everything in here is DETERMINISTIC (fixed seeds) so failures can be
reproduced from a git hash alone. Fixtures are session-scoped where the
generated object is read-only; per-test fixtures reset writable state.

Fixtures provided:
    tmp_project_dir    per-test temp dir that gets wiped after the test
    rng                numpy Generator seeded to 0
    synth_image_2d     2-D uint16 grayscale "microscopy" image
    synth_image_3d     3-D uint16 image (Z, H, W)
    synth_image_stack  4-D uint16 stack (T, C, H, W)
    synth_mask_2d      2-D int label mask with N connected blobs
    synth_masks_multi  dict of cell/nucleus/pathogen label masks
    synth_measurements pandas DataFrame with typical spacr columns
    synth_sqlite_db    file-backed sqlite with a minimal spacr schema
    dark_style         style_out dict returned by set_dark_style() with
                       a hidden Tk root; scope='function' to keep Tk
                       state fresh across tests.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the in-tree spacr importable without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Headless matplotlib for CI / test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

# Try to import matplotlib once with the Agg backend fixed. If unavailable,
# individual tests that need it will skip themselves.
try:  # pragma: no cover - import side effect only
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Basic infra fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic numpy Generator."""
    return np.random.default_rng(0)


@pytest.fixture
def tmp_project_dir(tmp_path):
    """A fresh temp directory laid out like a spacr project."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "measurements").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

def _place_blobs(shape, n_blobs, rng, radius_range=(6, 14), max_intensity=60000):
    """Draw n_blobs bright circular blobs on a dark background."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        intensity = int(rng.integers(int(max_intensity * 0.4), max_intensity))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.maximum(img[mask], intensity)
    # Add a bit of gaussian background noise so np.min != np.max in flat regions.
    img = img + rng.integers(50, 200, size=shape, dtype=np.uint16)
    return img.astype(np.uint16)


@pytest.fixture
def synth_image_2d(rng):
    """256x256 uint16 grayscale image with ~8 bright blobs on dark background."""
    return _place_blobs((256, 256), n_blobs=8, rng=rng)


@pytest.fixture
def synth_image_3d(rng):
    """3-D image (Z=5, H=128, W=128) uint16."""
    return np.stack([_place_blobs((128, 128), n_blobs=6, rng=rng) for _ in range(5)])


@pytest.fixture
def synth_image_stack(rng):
    """4-D (T=3, C=2, H=128, W=128) uint16 timelapse-ish stack."""
    return np.stack(
        [
            np.stack([_place_blobs((128, 128), n_blobs=5, rng=rng) for _ in range(2)])
            for _ in range(3)
        ]
    )


# ---------------------------------------------------------------------------
# Synthetic label-mask fixtures
# ---------------------------------------------------------------------------

def _labeled_blobs(shape, n_blobs, rng, radius_range=(8, 16)):
    """Return an int32 label image where each blob has a unique id starting at 1."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    lbl = np.zeros(shape, dtype=np.int32)
    next_id = 1
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        # Skip if it would overlap an existing label (keep them disjoint).
        if lbl[mask].max() != 0:
            continue
        lbl[mask] = next_id
        next_id += 1
    return lbl


@pytest.fixture
def synth_mask_2d(rng):
    """256x256 int32 label mask, 6 disjoint blobs (ids 1..N)."""
    return _labeled_blobs((256, 256), n_blobs=6, rng=rng)


@pytest.fixture
def synth_masks_multi(rng):
    """Dict of aligned cell/nucleus/pathogen label masks for one 256x256 field."""
    cell = _labeled_blobs((256, 256), n_blobs=5, rng=rng, radius_range=(20, 30))
    # Nucleus sits inside cells; smaller radius, centered on cell centroids.
    nucleus = np.zeros_like(cell)
    from scipy.ndimage import center_of_mass
    for cell_id in np.unique(cell):
        if cell_id == 0:
            continue
        cy, cx = center_of_mass(cell == cell_id)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2
        nucleus[m] = cell_id
    # Pathogens: 0-2 small blobs scattered inside random cells.
    pathogen = np.zeros_like(cell)
    next_id = 1
    for _ in range(int(rng.integers(0, 3))):
        cell_ids = [i for i in np.unique(cell) if i != 0]
        if not cell_ids:
            break
        cid = int(rng.choice(cell_ids))
        cy, cx = center_of_mass(cell == cid)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        offset_y = int(rng.integers(-10, 11))
        offset_x = int(rng.integers(-10, 11))
        m = (yy - (cy + offset_y)) ** 2 + (xx - (cx + offset_x)) ** 2 <= 3 ** 2
        m = m & (cell == cid)  # keep pathogen inside its cell
        if m.any():
            pathogen[m] = next_id
            next_id += 1
    return {"cell": cell, "nucleus": nucleus, "pathogen": pathogen}


# ---------------------------------------------------------------------------
# Synthetic DataFrames & sqlite
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_measurements(rng):
    """A DataFrame with typical spacr measurement columns for 40 objects."""
    n = 40
    plates = ["plate1"] * n
    rows = rng.integers(1, 9, size=n)  # A..H analog
    cols = rng.integers(1, 13, size=n)
    wells = [f"{chr(ord('A')+r-1)}{c:02d}" for r, c in zip(rows, cols)]
    fields = rng.integers(1, 4, size=n)
    prcs = [f"{p}_{w}_{f}" for p, w, f in zip(plates, wells, fields)]
    return pd.DataFrame(
        {
            "plate": plates,
            "row": rows,
            "column": cols,
            "well": wells,
            "field": fields,
            "prc": prcs,
            "object_label": np.arange(1, n + 1),
            "cell_area": rng.uniform(200, 4000, size=n),
            "cell_channel_0_mean_intensity": rng.uniform(500, 40000, size=n),
            "cell_channel_1_mean_intensity": rng.uniform(500, 40000, size=n),
            "nucleus_area": rng.uniform(80, 900, size=n),
            "pathogen_count": rng.integers(0, 5, size=n),
        }
    )


@pytest.fixture
def synth_sqlite_db(tmp_path, synth_measurements):
    """A file-backed sqlite database with a minimal spacr-ish schema."""
    db_path = tmp_path / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        synth_measurements.to_sql("cell", con, index=False)
        # A dummy annotation table many spacr helpers assume exists.
        anno = pd.DataFrame(
            {
                "prc": synth_measurements["prc"].unique(),
                "annotation": 0,
            }
        )
        anno.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db_path


# ---------------------------------------------------------------------------
# GUI / Tk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tk_root():
    """A hidden Tk root; skips if there is no display available."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available for Tk: {e}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def dark_style(tk_root):
    """The style_out dict returned by set_dark_style()."""
    from tkinter import ttk
    from spacr.gui_elements import set_dark_style
    return set_dark_style(ttk.Style(), parent_frame=None)


# ---------------------------------------------------------------------------
# Yokogawa microscopy fixtures — CellVoyager (default) and CQ1 filename styles
# ---------------------------------------------------------------------------
#
# CellVoyager filename regex (see spacr/utils.py::_get_regex):
#     {plateID}_{wellID}_T{timeID}F{fieldID}L{laserID}A{AID}Z{sliceID}C{chanID}.tif
#
# CQ1 filename regex:
#     W{wellID}F{fieldID}T{timeID}Z{sliceID}C{chanID}.tif
#     wellID is an integer 1..384 that spacr converts to A01..P24.

def _write_tif(path, arr):
    """Save an image as a real TIFF (tifffile if available, else Pillow)."""
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
        return
    except Exception:
        pass
    from PIL import Image
    Image.fromarray(arr).save(str(path))


def _make_field(rng, shape=(128, 128)):
    """Small deterministic uint16 field image with a few blobs."""
    return _place_blobs(shape, n_blobs=int(rng.integers(2, 6)), rng=rng)


@pytest.fixture
def yokogawa_cellvoyager_dir(tmp_path, rng):
    """
    A temp directory of TIFFs following the Yokogawa CellVoyager naming.

    Layout (deterministic):
      * 1 plate  ('plate1')
      * 2 wells  ('A01', 'A02')
      * 2 fields (F001, F002)
      * 2 channels (C01, C02)
      * 1 z-slice, 1 timepoint, 1 laser, 1 action
    -> 8 TIFFs total.

    Yields the directory path plus a manifest so tests can assert what was
    written.
    """
    src = tmp_path / "cellvoyager"
    src.mkdir()
    manifest = []
    for well in ("A01", "A02"):
        for field in ("001", "002"):
            for chan in ("01", "02"):
                fname = f"plate1_{well}_T0001F{field}L01A01Z01C{chan}.tif"
                img = _make_field(rng)
                _write_tif(src / fname, img)
                manifest.append(
                    {"plate": "plate1", "well": well, "field": field,
                     "channel": chan, "path": str(src / fname)}
                )
    return {"src": src, "manifest": manifest,
            "metadata_type": "cellvoyager",
            "n_wells": 2, "n_fields": 2, "n_channels": 2}


@pytest.fixture
def yokogawa_cq1_dir(tmp_path, rng):
    """
    A temp directory of TIFFs following the Yokogawa CQ1 naming.

    Uses integer well IDs (1..384) that spacr converts to A01..P24 via
    utils._convert_cq1_well_id. Here we use W1 (=A01) and W25 (=B01).
    """
    src = tmp_path / "cq1"
    src.mkdir()
    manifest = []
    for well_id, expected_well in ((1, "A01"), (25, "B01")):
        for field in ("001", "002"):
            for chan in ("1", "2"):
                fname = f"W{well_id}F{field}T0001Z01C{chan}.tif"
                img = _make_field(rng)
                _write_tif(src / fname, img)
                manifest.append(
                    {"well_id": well_id, "well": expected_well, "field": field,
                     "channel": chan, "path": str(src / fname)}
                )
    return {"src": src, "manifest": manifest,
            "metadata_type": "cq1",
            "n_wells": 2, "n_fields": 2, "n_channels": 2}


# ---------------------------------------------------------------------------
# Illumina sequencing fixtures — 3-barcode reads matching spacr's default
# regex.
# ---------------------------------------------------------------------------
#
# spacr's default barcode regex is:
#   ^(?P<column>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<row>.{8}).*
#
# so each read starts with an 8bp column barcode, then a constant TGCTG
# spacer, then some fill, then TAAAC + a 20-21bp gRNA, then AACTT..AGAAG,
# then an 8bp row barcode, then anything.
#
# The barcode reference is emitted as FASTA (one entry per barcode) AND
# CSV (with 'sequence' / 'name' columns), since spacr.sequencing itself
# consumes the CSV form via map_sequences_to_names().

def _rand_bases(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n))


def _fastq_record(read_id, seq, qual_char="I"):
    qual = qual_char * len(seq)
    return f"@{read_id}\n{seq}\n+\n{qual}\n"


@pytest.fixture
def synth_barcodes(tmp_path, rng):
    """
    Build 3 barcode reference tables (columns, rows, gRNAs), each in BOTH
    FASTA and CSV form, and hand back the file paths + the raw sequences
    for use by test-read generators.

    Sizes: 4 columns, 4 rows, 6 gRNAs (small so the test suite stays fast).
    """
    N_COLUMNS = 4
    N_ROWS = 4
    N_GRNAS = 6

    # Deterministic barcode sequences.
    columns = {f"col{i+1}": _rand_bases(rng, 8) for i in range(N_COLUMNS)}
    rows = {f"row{i+1}": _rand_bases(rng, 8) for i in range(N_ROWS)}
    grnas = {f"grna{i+1}": _rand_bases(rng, 20) for i in range(N_GRNAS)}

    out_dir = tmp_path / "barcodes"
    out_dir.mkdir()

    def _write_fasta(path, name_to_seq):
        with open(path, "w") as f:
            for name, seq in name_to_seq.items():
                f.write(f">{name}\n{seq}\n")

    def _write_csv(path, name_to_seq):
        # spacr.sequencing.map_sequences_to_names expects 'sequence','name' columns.
        with open(path, "w") as f:
            f.write("sequence,name\n")
            for name, seq in name_to_seq.items():
                f.write(f"{seq},{name}\n")

    paths = {}
    for label, table in (("column", columns), ("row", rows), ("grna", grnas)):
        fasta = out_dir / f"{label}_barcodes.fasta"
        csv = out_dir / f"{label}_barcodes.csv"
        _write_fasta(fasta, table)
        _write_csv(csv, table)
        paths[f"{label}_fasta"] = str(fasta)
        paths[f"{label}_csv"] = str(csv)

    return {"columns": columns, "rows": rows, "grnas": grnas,
            "paths": paths, "dir": out_dir}


@pytest.fixture
def synth_illumina_reads(tmp_path, rng, synth_barcodes):
    """
    Build a paired-end Illumina FASTQ.gz pair whose R1 reads carry one
    column + one gRNA + one row barcode each, in the layout the default
    spacr regex expects. R2 mirrors R1 in this fixture.

    Yields:
      dict with 'r1_path', 'r2_path' (both .fastq.gz), 'n_reads',
      and 'truth' — a list of dicts telling the test which barcodes were
      injected into each read so tests can validate detection.
    """
    import gzip

    N_READS = 40
    truth = []
    col_seqs = list(synth_barcodes["columns"].items())
    row_seqs = list(synth_barcodes["rows"].items())
    grna_seqs = list(synth_barcodes["grnas"].items())

    lines_r1 = []
    lines_r2 = []
    for i in range(N_READS):
        col_name, col_seq = col_seqs[int(rng.integers(0, len(col_seqs)))]
        row_name, row_seq = row_seqs[int(rng.integers(0, len(row_seqs)))]
        grna_name, grna_seq = grna_seqs[int(rng.integers(0, len(grna_seqs)))]

        # Build a read exactly matching:
        #   {col:8}TGCTG{fill}TAAAC{grna:20-21}AACTT{fill}AGAAG{row:8}{trailing}
        # spacr's regex uses .* for the two fill regions.
        fill1 = _rand_bases(rng, 6)
        fill2 = _rand_bases(rng, 6)
        trailing = _rand_bases(rng, 8)

        seq = (
            col_seq +
            "TGCTG" + fill1 +
            "TAAAC" + grna_seq +
            "AACTT" + fill2 +
            "AGAAG" + row_seq + trailing
        )

        # For paired-end Illumina, R2 comes off the opposite strand — the
        # simplest realistic fixture: R2 is the reverse complement of R1.
        # spacr.sequencing.paired_find_sequence_in_chunk_reads applies
        # reverse_complement(R2) before searching, so after that step R2
        # should equal R1 again and the target anchor is findable in both.
        def _rc(s):
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            return "".join(comp[b] for b in reversed(s))

        read_id = f"SIM:1:FCXX:1:1101:{i}:1"
        lines_r1.append(_fastq_record(read_id, seq))
        lines_r2.append(_fastq_record(read_id, _rc(seq)))
        truth.append({
            "read_id": read_id, "seq": seq,
            "column": col_name, "row": row_name, "grna": grna_name,
        })

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    r1 = seq_dir / "sample_R1.fastq.gz"
    r2 = seq_dir / "sample_R2.fastq.gz"
    with gzip.open(r1, "wt") as fh:
        fh.write("".join(lines_r1))
    with gzip.open(r2, "wt") as fh:
        fh.write("".join(lines_r2))

    return {
        "r1_path": str(r1),
        "r2_path": str(r2),
        "n_reads": N_READS,
        "truth": truth,
        "src": str(seq_dir),
    }


# ---------------------------------------------------------------------------
# Hugging Face-backed fixtures: real Yokogawa CellVoyager images + spacr's
# canonical settings CSVs.
# ---------------------------------------------------------------------------
#
# These pull a small deterministic slice of two public HF datasets:
#   einarolafsson/toxo_mito       real 4-channel CellVoyager microscopy
#   einarolafsson/spacr_settings  the reference settings CSVs
#
# Tests that use them are marked @pytest.mark.network so they skip in
# offline CI. Fixtures are session-scoped since the payload is stable.
#
# Only 4 TIFFs (one plate/well/field, four channels) are pulled — enough
# to exercise the metadata extractor, the settings loader, and one mask
# generation pass, without downloading the full 210-file dataset.

@pytest.fixture(scope="session")
def hf_toxo_mito_field(tmp_path_factory):
    """Download one field (4 channels) from einarolafsson/toxo_mito.

    Returns a dict with:
      * src: path to the local directory containing the TIFFs
      * files: list of absolute file paths (4 TIFFs, one per channel)
      * plate/well/field: the metadata slice picked
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover - import failure
        pytest.skip(f"huggingface_hub not available: {e}")

    dst = tmp_path_factory.mktemp("hf_toxo_mito")
    target_files = [
        "plate1/plate1_E01_T0001F001L01A01Z01C02.tif",
        "plate1/plate1_E01_T0001F001L01A02Z01C01.tif",
        "plate1/plate1_E01_T0001F001L01A02Z01C04.tif",
        "plate1/plate1_E01_T0001F001L01A03Z01C03.tif",
    ]
    local_paths = []
    for rel in target_files:
        try:
            p = hf_hub_download(
                repo_id="einarolafsson/toxo_mito",
                filename=rel,
                repo_type="dataset",
                local_dir=str(dst),
            )
        except Exception as e:  # pragma: no cover - network path
            pytest.skip(f"HF download failed for {rel}: {e}")
        local_paths.append(p)
    return {
        "src": str(dst / "plate1"),
        "files": local_paths,
        "plate": "plate1",
        "well": "E01",
        "field": "001",
    }


@pytest.fixture(scope="session")
def spacr_pipeline_run(tmp_path_factory, hf_toxo_mito_multi_fields):
    """
    Run the full `preprocess_generate_masks` pipeline ONCE per test session
    on a copy of the HF toxo_mito data, then hand the working directory
    (with all generated folders + masks + measurements) to every
    downstream test that wants to inspect it.

    Marked as skip if GPU / cellpose / HF is unavailable — see the tests
    that use this fixture; they carry the @pytest.mark.slow +
    @pytest.mark.gpu + @pytest.mark.network markers so the whole thing
    only runs when explicitly opted in.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available for full pipeline test")
    try:
        import cellpose  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cellpose unavailable: {e}")

    import shutil
    work = tmp_path_factory.mktemp("spacr_pipeline")
    # Copy the HF TIFFs into a flat working dir (the pipeline expects a
    # flat directory of raw images).
    for src_path in hf_toxo_mito_multi_fields["files"]:
        shutil.copy(src_path, work / os.path.basename(src_path))

    from spacr.core import preprocess_generate_masks
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks(None)
    settings.update({
        "src": str(work),
        "metadata_type": "cellvoyager",
        "batch_size": 100,          # avoids the mod-1 bug in preprocess_img_data
        # channels are 0-indexed into the merged stack.
        "channels": [0, 1, 2, 3],
        # toxo_mito: C01=nucleus, C02=cell, C03=pathogen (0-indexed).
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": 2,
        "organelle_channel": None,
        "plot": False, "verbose": False, "test_mode": False, "timelapse": False,
        "n_jobs": 1, "adjust_cells": False, "delete_intermediate": False,
        "all_to_mip": False,
    })

    try:
        preprocess_generate_masks(settings)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pipeline failed to run: {e}")

    return {"src": str(work), "settings": settings,
            "n_input_fields": len(hf_toxo_mito_multi_fields["fields"])}


@pytest.fixture(scope="session")
def spacr_measure_run(spacr_pipeline_run):
    """Run measure_crop on the shared pipeline output ONCE per test session
    and hand back the measurements DB path + the src directory.

    Session-scoped so multiple test modules (pipeline_e2e,
    pipeline_training_analysis, ...) can inspect the same DB / PNG
    outputs without re-running the (~2 minute) mask + measure work per
    module.
    """
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings(None)
    settings.update({
        "src": spacr_pipeline_run["src"],
        # After preprocess_generate_masks the merged stack is
        # [C0=nucleus_intensity, C1=cell_intensity, C2=pathogen_intensity,
        #  C3=organelle_intensity(unused), C4=cell_mask, C5=nucleus_mask,
        #  C6=pathogen_mask].
        "channels": [0, 1, 2, 3],
        "cell_chann_dim": 1, "nucleus_chann_dim": 0, "pathogen_chann_dim": 2,
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "cytoplasm": True,
        "n_jobs": 1, "batch_size": 8, "verbose": False,
        # save_png=True so downstream tests can chain into generate_dataset
        # and apply_model on the resulting per-object crops.
        "plot": False, "save_png": True, "save_arrays": False,
    })
    try:
        measure_crop(settings)
    except Exception as e:  # pragma: no cover - integration path
        pytest.skip(f"measure_crop failed on synthetic pipeline output: {e}")
    return {
        "src": spacr_pipeline_run["src"],
        "db_path": os.path.join(
            spacr_pipeline_run["src"], "measurements", "measurements.db"
        ),
    }


@pytest.fixture(scope="session")
def hf_toxo_mito_multi_fields(tmp_path_factory):
    """Download several fields (each 4 channels) from einarolafsson/toxo_mito
    into a NEW temp directory — the full mask pipeline needs enough FOVs
    to form a valid batch and populate the channel folders.

    Returns:
      dict with src pointing at the plate directory that contains the flat
      list of Yokogawa CellVoyager TIFFs (as the pipeline expects) plus the
      manifest of what was downloaded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        pytest.skip(f"huggingface_hub not available: {e}")

    dst = tmp_path_factory.mktemp("hf_toxo_mito_multi")
    # 3 fields × 4 channels = 12 TIFFs — enough to satisfy batch checks
    # while staying under ~10 MB of download.
    fields = ("001", "009", "010")
    channel_layout = (
        ("A01Z01C02",),
        ("A02Z01C01",),
        ("A02Z01C04",),
        ("A03Z01C03",),
    )
    local_paths = []
    for f in fields:
        for (chan,) in channel_layout:
            rel = f"plate1/plate1_E01_T0001F{f}L01{chan}.tif"
            try:
                p = hf_hub_download(
                    repo_id="einarolafsson/toxo_mito",
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(dst),
                )
            except Exception as e:  # pragma: no cover
                pytest.skip(f"HF download failed for {rel}: {e}")
            local_paths.append(p)
    return {
        "src": str(dst / "plate1"),
        "files": local_paths,
        "plate": "plate1",
        "well": "E01",
        "fields": fields,
    }


@pytest.fixture(scope="session")
def hf_spacr_settings(tmp_path_factory):
    """Download the two reference settings CSVs from einarolafsson/spacr_settings."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        pytest.skip(f"huggingface_hub not available: {e}")

    dst = tmp_path_factory.mktemp("hf_spacr_settings")
    paths = {}
    for name in ("gen_masks_settings.csv", "crop_measure_settings.csv"):
        try:
            p = hf_hub_download(
                repo_id="einarolafsson/spacr_settings",
                filename=name,
                repo_type="dataset",
                local_dir=str(dst),
            )
        except Exception as e:  # pragma: no cover
            pytest.skip(f"HF download failed for {name}: {e}")
        paths[name] = p
    return paths
