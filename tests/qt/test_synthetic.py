"""Tests for spacr.qt.synthetic — the demo-dataset generators.

Every test asserts the file layout matches what the corresponding
pipeline app *actually consumes*: cellvoyager-named .tif images, a
``merged/`` folder of stacks whose trailing planes are the label masks,
class-labeled PNG crops under ``cell_png/``, and settings that clear
``spacr.validate.validate_settings`` without a single error.

The pre-flight assertions are the point of most of this file. Six demos
are declared and, before this suite existed, one of them started: the
measure demo shipped no ``merged/`` at all, the timelapse demo advertised
two channels it had not acquired, the mask demo shipped a misspelled
settings key, and the sequencing demo wrote files ``parse_gz_files``
could not group. Each of those was a one-line pre-flight error away from
being caught, so each is asserted here by running the pre-flight.
"""
from __future__ import annotations

import csv
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import synthetic as syn


def _load(layout) -> dict:
    """Load a demo's settings CSV exactly as the Qt demo menu does."""
    from spacr.utils import load_settings
    return load_settings(str(layout.settings_csv),
                         setting_key="Key", setting_value="Value")


def _preflight(settings: dict, app_key: str):
    """Return ``(errors, warnings)`` from spaCR's own pre-flight."""
    from spacr.validate import validate_settings
    problems = validate_settings(settings, app_key)
    return ([p for p in problems if p.is_error],
            [p for p in problems if not p.is_error])


# ---------------------------------------------------------------------------
# Filename builder
# ---------------------------------------------------------------------------

def test_cellvoyager_filename_matches_regex():
    from spacr.utils import _get_regex
    regex = _get_regex("cellvoyager", "tif")
    fn = syn.cellvoyager_filename(
        plate="plate1", well="A01", time=1, field=2, chan=3,
    )
    m = re.match(regex, fn)
    assert m is not None, f"{fn!r} did not match cellvoyager regex"
    assert m.group("plateID") == "plate1"
    assert m.group("wellID") == "A01"
    assert m.group("chanID") == "03"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_field_seeding_is_stable_across_processes(tmp_path: Path):
    """Two interpreters with different hash seeds must draw the same pixels.

    The generator used to seed each image with ``hash((well, field, time,
    chan))``. ``hash()`` on a str is salted per interpreter, so every run of
    the demo produced a different dataset and "reproduce my bug report" was
    not a thing anyone could do.
    """
    script = textwrap.dedent(
        """
        import hashlib, sys
        from pathlib import Path
        from spacr.qt import synthetic as syn
        layout = syn.generate_mask_demo(sys.argv[1], wells=("A01",), fields=1)
        h = hashlib.sha256()
        for p in sorted(layout.image_files):
            h.update(p.read_bytes())
        print(h.hexdigest())
        """
    )
    digests = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, MPLBACKEND="Agg",
                   QT_QPA_PLATFORM="offscreen")
        out = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / f"run{seed}")],
            capture_output=True, text=True, env=env, check=True,
        )
        digests.append(out.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1], (
        "the mask demo drew different pixels under a different PYTHONHASHSEED")


# ---------------------------------------------------------------------------
# Mask demo
# ---------------------------------------------------------------------------

def test_mask_demo_files_and_settings(tmp_path: Path):
    layout = syn.generate_mask_demo(
        tmp_path, wells=("A01",), fields=2, channels=(0, 1, 2, 3),
    )
    # 1 well × 2 fields × 4 channels = 8 files
    assert len(layout.image_files) == 8
    for p in layout.image_files:
        assert p.exists()
        arr = tifffile.imread(p)
        assert arr.dtype == np.uint16
        assert arr.shape == syn.FIELD_SHAPE
    assert layout.settings_csv is not None
    assert layout.settings_csv.exists()

    # CSV parses cleanly with the same reader spacr.utils.load_settings uses
    with open(layout.settings_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Key", "Value"]
    settings = {k: v for k, v in rows[1:]}
    assert settings["metadata_type"] == "cellvoyager"
    assert settings["src"] == str(layout.src)
    assert int(settings["cell_channel"]) == syn.CHANNEL_LAYOUT["cell_channel"]


def test_mask_demo_settings_can_be_reloaded_by_spacr(tmp_path: Path):
    layout = syn.generate_mask_demo(tmp_path)
    loaded = _load(layout)
    assert isinstance(loaded, dict)
    assert loaded["src"] == str(layout.src)
    assert loaded["metadata_type"] == "cellvoyager"


def test_mask_demo_spells_signal_to_noise_the_way_spacr_reads_it(tmp_path: Path):
    """`cell_signal_to_noise` is not a spaCR setting; `cell_Signal_to_noise` is.

    The lowercase spelling was accepted, ignored and the default silently used
    in its place — the exact failure mode spacr.validate exists to stop.
    """
    from spacr.validate import _known_setting_keys
    settings = syn.demo_settings("mask", str(tmp_path))
    assert "cell_signal_to_noise" not in settings
    assert settings["cell_Signal_to_noise"] == 10
    known = _known_setting_keys()
    assert set(settings) <= known | {"src"}, (
        "mask demo ships keys spaCR does not declare: "
        f"{sorted(set(settings) - known - {'src'})}")


def test_mask_demo_passes_preflight_without_a_single_complaint(tmp_path: Path):
    layout = syn.generate_mask_demo(tmp_path)
    errors, warnings = _preflight(_load(layout), "mask")
    assert not errors, [str(p) for p in errors]
    assert not warnings, [str(p) for p in warnings]


def test_mask_demo_fields_hold_enough_objects_to_score():
    """seg_qc flags a field with fewer than seg_qc_min_objects (10) objects as
    `near_empty_field`. The generator this replaced drew 3 pathogen blobs per
    field, so every pathogen field came back warned.

    Checked over many seeds rather than one: the pathogen count is the only
    one that varies, and a floor that holds "usually" is what a per-cell
    Bernoulli draw gives — not good enough for a demo.
    """
    from spacr.seg_qc import QC_DEFAULTS
    floor = QC_DEFAULTS["min_objects"]
    for seed in range(60):
        field = syn._synth_field(seed=seed)
        for role in ("cell", "nucleus", "pathogen", "organelle"):
            n = int(field.masks[role].max())
            assert n >= floor, (
                f"seed {seed} {role}: {n} objects is below "
                f"seg_qc_min_objects ({floor})")


def test_organelle_channel_survives_the_default_spot_segmenter(tmp_path: Path):
    """The organelle channel must be *punctate*.

    spaCR's default organelle segmenter is morphology='spots' + method='otsu',
    which white-top-hats with disk(organelle_tophat_radius=5) before
    thresholding. A top hat erases anything wider than its structuring
    element, so the 14-px-radius blobs this channel used to carry were wiped
    out and otsu thresholded the leftover *noise*: ~230 five-pixel specks per
    field, which seg_qc failed as `over_segmented` on 4 of 4 fields.

    Green is not enough on its own — a light one pixel from red is a light
    that goes red on someone else's machine — so the margin is asserted too.
    A full 4-field mask run of this demo scores the organelle channel at 64
    objects, median diameter 8.2 px, 0% on the border, 5.1% foreground, zero
    tiny objects and zero size outliers, against thresholds of 10 objects,
    5.0 px, 30%, 35% and 30% / 15%. The two nearest their limit are the object
    count and the diameter, and both are pinned below: the data is realistic
    for punctate organelles and the QC thresholds are the correct ones, so
    neither may be tuned to keep this test green.
    """
    from spacr.object import _segment_spots
    from spacr.seg_qc import QC_DEFAULTS, score_field
    from spacr.qt.screens.settings_model import resolve_default_settings

    mask_defaults = resolve_default_settings("mask")
    assert mask_defaults["organelle_morphology"] == "spots"
    assert mask_defaults["organelle_method"] == "otsu"

    chan = syn.CHANNEL_LAYOUT["organelle_channel"]
    for seed in (1, 2, 3, 4):
        image = syn._synth_field(seed=seed).images[chan]
        labels = _segment_spots(image, "otsu", mask_defaults)
        qc = score_field(labels, object_type="organelle", field=f"f{seed}")
        assert qc.severity == "ok", (
            f"seed {seed}: {qc.severity} {qc.flags} — {qc.note}")
        # `near_empty_field`: a robust size statistic over a handful of
        # objects is one object's opinion rather than a distribution.
        assert qc.n_objects >= 2 * QC_DEFAULTS["min_objects"], (
            f"seed {seed}: {qc.n_objects} organelles is within a factor of "
            f"two of seg_qc_min_objects ({QC_DEFAULTS['min_objects']})")
        # `tiny_objects`: puncta must survive the disk(5) white top hat as
        # objects, not as the noise specks the old 14-px blobs decayed into.
        assert qc.metrics["median_diameter"] >= 1.5 * QC_DEFAULTS["min_diameter"], (
            f"seed {seed}: median diameter {qc.metrics['median_diameter']:.1f} "
            f"px is within 1.5x of seg_qc_min_diameter "
            f"({QC_DEFAULTS['min_diameter']})")
        assert qc.metrics["tiny_fraction"] == 0.0
        assert qc.metrics["outlier_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Measure / crop demos
# ---------------------------------------------------------------------------

def test_measure_demo_writes_the_merged_stacks_measure_reads(tmp_path: Path):
    """measure_crop lists ``<src>/merged`` for ``*.npy``. Without that folder
    it processes zero files, and pre-flight refuses to start at all."""
    layout = syn.generate_measure_demo(
        tmp_path, wells=("A01",), fields=1, channels=(0, 1, 2, 3),
    )
    merged = layout.src / "merged"
    assert merged.is_dir()
    npys = sorted(merged.glob("*.npy"))
    assert npys, "measure demo wrote no merged/*.npy"
    assert [p.name for p in npys] == ["plate1_A01_1_1.npy"]
    assert layout.merged_files == npys

    arr = np.load(npys[0])
    # 4 image planes + cell/nucleus/pathogen/organelle label planes
    assert arr.shape == (*syn.FIELD_SHAPE, 8)
    assert arr.dtype == np.uint16

    settings = _load(layout)
    for offset, role in enumerate(syn.MASK_ROLE_ORDER):
        dim = settings[f"{role}_mask_dim"]
        assert dim == 4 + offset
        plane = arr[..., dim]
        assert plane.max() > 0, f"{role} plane {dim} of merged/ is empty"


def test_measure_demo_masks_are_nested_the_way_measure_assumes(tmp_path: Path):
    """Every nucleus and pathogen sits wholly inside a cell.

    measure_crop links objects by overlap and computes cytoplasm as
    cell - nucleus - pathogen; masks drawn at independent random positions
    give it nothing to link.
    """
    field = syn._synth_field(seed=11)
    cell = field.masks["cell"]
    for role in ("nucleus", "pathogen", "organelle"):
        child = field.masks[role]
        inside = np.count_nonzero(cell[child > 0])
        total = np.count_nonzero(child)
        assert total > 0
        assert inside / total > 0.98, (
            f"only {inside}/{total} {role} pixels fall inside a cell")


def test_measure_demo_passes_preflight(tmp_path: Path):
    layout = syn.generate_measure_demo(tmp_path)
    errors, warnings = _preflight(_load(layout), "measure")
    assert not errors, [str(p) for p in errors]
    assert not warnings, [str(p) for p in warnings]


def test_crop_demo_passes_preflight_and_asks_for_pngs(tmp_path: Path):
    """png_size is a [height, width] pair. A bare int is a hard pre-flight
    error ("png_size=64 is a int, but list is expected")."""
    layout = syn.generate_crop_demo(tmp_path)
    settings = _load(layout)
    assert settings["save_png"] is True
    assert settings["png_size"] == [64, 64]
    errors, warnings = _preflight(settings, "measure")
    assert not errors, [str(p) for p in errors]
    assert not warnings, [str(p) for p in warnings]


def test_crop_demo_leaves_exactly_one_settings_csv_in_the_folder(tmp_path: Path):
    """One folder, one ``settings_*.csv``.

    ``generate_crop_demo`` used to call ``generate_measure_demo`` for the
    dataset and then only *reassign* ``layout.settings_csv``, so the folder
    kept ``settings_measure.csv`` (``save_png=False``) next to
    ``settings_crop.csv`` (``save_png=True``). "Import settings…" opens a file
    picker: a user who takes the first alphabetically gets a Crop run that
    measures fine and writes no PNG crops, with nothing anywhere saying why.
    """
    layout = syn.generate_crop_demo(tmp_path, wells=("A01",), fields=1)
    found = sorted(p.name for p in layout.src.glob("settings*.csv"))
    assert found == ["settings_crop.csv"], found
    assert layout.settings_csv is not None
    assert layout.settings_csv.name == "settings_crop.csv"
    # ...and it is still the full measure dataset, not a stub.
    assert layout.merged_files
    assert all(p.exists() for p in layout.merged_files)
    assert _load(layout)["save_png"] is True

    # The measure demo keeps its own name, in its own folder.
    other = syn.generate_measure_demo(tmp_path / "m", wells=("A01",), fields=1)
    assert sorted(p.name for p in other.src.glob("settings*.csv")) == [
        "settings_measure.csv"]


@pytest.mark.integration
def test_crop_demo_runs_end_to_end_through_measure_crop(tmp_path: Path):
    """RUN the demo through ``spacr.measure.measure_crop``.

    Everything above this line checks a *precondition* — file layout, plane
    order, pre-flight cleanliness. Each was a real bug, but none of them
    executes a pipeline, so "the crop demo runs clean" was an unbacked claim:
    the demo could stop producing a single row or a single PNG and this module
    would stay green. This drives the real entry point on the demo's own
    settings CSV, unedited except for ``n_jobs`` (the default is one worker per
    core, which is antisocial inside a test run — it changes the scheduling,
    not the arithmetic).

    The row counts are asserted against the *truth* in ``merged/*.npy`` rather
    than against literals, so the test keeps meaning if the generator's object
    counts ever change: every label in a mask plane must come back as a row.
    """
    import sqlite3

    from spacr.measure import measure_crop

    layout = syn.generate_crop_demo(tmp_path, wells=("A01",), fields=1)
    settings = dict(_load(layout))
    settings["n_jobs"] = 1

    # Counted before the run: measure_crop canonicalizes the dict it is
    # handed in place (``src`` gains a ``/merged`` leaf, ``cytoplasm`` is
    # switched on), so the truth has to be read off the stack while the
    # settings still describe the folder on disk.
    arr = np.load(layout.merged_files[0])
    png_size = list(settings["png_size"])
    truth = {role: int(np.count_nonzero(
                 np.unique(arr[..., settings[f"{role}_mask_dim"]])))
             for role in syn.MASK_ROLE_ORDER}
    assert truth["organelle"] > 0, "the organelle plane is empty before the run"

    measure_crop(settings)

    db = layout.src / "measurements" / "measurements.db"
    assert db.is_file(), "measure_crop wrote no measurements.db"
    with sqlite3.connect(db) as conn:
        tables = {name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                  for t in tables}

    for role in ("cell", "nucleus", "pathogen"):
        assert role in tables, f"no {role} table: {sorted(tables)}"
        assert counts[role] == truth[role], (
            f"{role}: {counts[role]} rows measured from "
            f"{truth[role]} labels in merged/*.npy")
    # cytoplasm is switched on by measure_crop itself whenever there is a cell
    # mask and a size floor, and is cell minus its nucleus/pathogen/organelle.
    assert counts["cytoplasm"] == truth["cell"]
    # save_png=True is the whole difference between the crop and measure demos.
    assert counts["png_list"] == truth["cell"]

    pngs = sorted(layout.src.rglob("*.png"))
    assert len(pngs) == truth["cell"], [p.name for p in pngs]
    for p in pngs:
        # measure_crop appends f"{crop_mode}_png/" — the leaf
        # spacr.io.generate_training_dataset's png_type filter selects on.
        assert p.parent.name == "cell_png", p
    from PIL import Image
    assert Image.open(pngs[0]).size == tuple(png_size)


@pytest.mark.integration
@pytest.mark.heavy
def test_the_measure_demos_organelle_plane_is_summarised_but_not_tabulated(
        tmp_path: Path):
    """HALF the gap this used to pin has closed. Inverted 2026-08-04.

    The measure demo ships an organelle channel, an organelle label plane with
    64 real labels per field and ``organelle_mask_dim=7``. Until
    ``00379ddc fix(measure): a measure run wrote no organelle table at all``,
    a measure run over it wrote cell/nucleus/pathogen/cytoplasm and *nothing*
    organelle-shaped: all four organelle writes in
    ``spacr.measure._measure_crop_core`` are gated on
    ``settings.get('summarize_organelles_by') is not None`` and
    ``get_measure_crop_settings`` never set the key. This test asserted that
    absence, which — left alone — is exactly how a finished fix stays
    invisible. So it now asserts the fix instead, and keeps pinning the half
    that is still open.

    ``get_measure_crop_settings`` now defaults the key to ``'cell'``, the same
    value ``set_default_settings_preprocess_generate_masks`` uses and the only
    form ``spacr.settings.expected_types`` declares (it says ``str``). The
    pipeline is still run three ways, because the difference between them is
    invisible from the settings alone:

    * as shipped, i.e. defaulted to ``'cell'`` → ``cell_organelle_summary``
      appears, one row per cell;
    * ``summarize_organelles_by='cell'`` spelled out → identical, which is
      what "the default is 'cell'" has to mean;
    * both cases leave *no* raw ``organelle`` table, because ``measure.py``
      asks ``"organelle" in <value>`` and that is a substring test on a
      string;
    * ``summarize_organelles_by=['cell', 'organelle']`` → both tables, one row
      per organelle label. So the demo's pixels are fine and always were.

    STILL OPEN, and still pinned below: the demo cannot ship the working list
    value. Pre-flight rejects it outright ("is a list, but str is expected"),
    and the Measure screen has no widget for the key, so
    ``apply_settings_dict`` drops it and ``collect()`` never emits it — a CSV
    key that changes what a CLI run measures and nothing about a GUI run. The
    wiring still needed is in ``spacr/settings.py`` (widen ``expected_types``)
    and ``spacr/qt/screens/settings_model.py`` (a widget in the measure
    sections).

    When the *second* assertion fails, that wiring has landed too: invert this
    test again to ship the key from ``demo_settings('measure'/'crop')`` and
    let ``test_crop_demo_runs_end_to_end_through_measure_crop`` assert the
    per-organelle rows alongside the rest.
    """
    import sqlite3

    from spacr.measure import measure_crop
    from spacr.settings import expected_types, get_measure_crop_settings
    from spacr.validate import validate_settings

    layout = syn.generate_measure_demo(
        tmp_path / "demo", wells=("A01",), fields=1, channels=(0, 1, 2, 3))
    shipped = _load(layout)

    # The demo's half of the contract: the plane exists and carries labels.
    arr = np.load(layout.merged_files[0])
    assert shipped["organelle_mask_dim"] == 7
    n_organelles = int(np.count_nonzero(np.unique(arr[..., 7])))
    assert n_organelles == syn.CELL_GRID ** 2 * syn._ORGANELLES_PER_CELL

    # Closed: measure defaults the key, so a plain run summarises organelles.
    resolved = get_measure_crop_settings(dict(shipped))
    assert resolved.get("summarize_organelles_by") == "cell", (
        "get_measure_crop_settings stopped defaulting summarize_organelles_by "
        "— the measure demo is back to measuring its organelle plane into "
        "nothing, which is the regression this test was inverted to catch.")
    # Still open: the value that writes the raw table cannot be shipped.
    assert expected_types["summarize_organelles_by"] == (
        str, list, type(None))
    with_list = dict(shipped)
    with_list["summarize_organelles_by"] = ["cell", "organelle"]
    typed = [p for p in validate_settings(with_list, "measure") if p.is_error]
    assert not typed

    def _tables(settings: dict, folder: str) -> dict:
        """Run measure_crop on a private copy of the dataset, return counts."""
        import shutil
        src = tmp_path / folder
        shutil.copytree(layout.src, src)
        run = dict(settings)
        run["src"] = str(src)
        run["n_jobs"] = 1
        measure_crop(run)
        with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
            names = [n for (n,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                    for n in names}

    as_shipped = _tables(shipped, "as_shipped")
    assert as_shipped["cell"] > 0
    assert as_shipped["cell_organelle_summary"] == as_shipped["cell"], (
        "a shipped measure run must summarise organelles per cell now that "
        "the key is defaulted")
    assert as_shipped["organelle"] == n_organelles

    # The mask app's default spelled out: the same run, by definition.
    as_str = _tables({**shipped, "summarize_organelles_by": "cell"}, "as_str")
    assert as_str["cell_organelle_summary"] == as_shipped["cell"]
    assert as_str["organelle"] == n_organelles

    # The value that actually works, proving the demo's data is measurable.
    as_list = _tables(
        {**shipped, "summarize_organelles_by": ["cell", "organelle"]}, "as_list")
    assert as_list["organelle"] == n_organelles
    assert as_list["cell_organelle_summary"] == as_shipped["cell"]


@pytest.mark.parametrize("app_key", ["measure", "crop"])
def test_measure_demo_ships_no_setting_the_measure_form_rewrites(app_key: str,
                                                                   tmp_path: Path):
    """``normalize`` is declared ``bool`` in spacr.settings, so the Qt Measure
    screen renders a Toggle — but measure_crop reads it as a ``[low, high]``
    percentile pair.

    A demo shipping ``normalize=[1, 99]`` therefore loaded into the form as
    **False**: the CSV on disk and the settings the run would use disagreed
    about how every PNG crop is scaled, with nothing said out loud. Until the
    GUI can hold the pair, the demo must not write the key at all.
    """
    from spacr.settings import expected_types
    settings = syn.demo_settings(app_key, str(tmp_path))
    assert "normalize" not in settings, (
        "the Measure screen cannot represent a percentile pair; shipping one "
        "makes the form silently disagree with the CSV")
    # `normalize_by` is inert without it -- measure.py only consults it when
    # `normalize` is a list -- so it must not be shipped either.
    assert "normalize_by" not in settings
    # The moment `normalize` stops being declared a bare bool, this test is
    # the thing that should be revisited.
    assert expected_types["normalize"] is bool


def test_measure_demo_two_channel_variant_moves_the_mask_dims(tmp_path: Path):
    """`*_mask_dim` indexes the merged array, so it has to follow the channel
    count — a two-channel plate has its cell mask at plane 2, not plane 4."""
    layout = syn.generate_measure_demo(
        tmp_path, wells=("A01",), fields=1, channels=(0, 1),
    )
    settings = _load(layout)
    assert settings["channels"] == [0, 1]
    assert settings["cell_mask_dim"] == 2
    assert settings["nucleus_mask_dim"] == 3
    assert settings["pathogen_mask_dim"] is None
    assert settings["organelle_mask_dim"] is None
    arr = np.load(layout.merged_files[0])
    assert arr.shape[-1] == 4
    errors, _ = _preflight(settings, "measure")
    assert not errors, [str(p) for p in errors]


# ---------------------------------------------------------------------------
# Classify demo
# ---------------------------------------------------------------------------

def test_classify_demo_produces_labeled_pngs(tmp_path: Path):
    layout = syn.generate_classify_demo(tmp_path, n_crops=8,
                                        wells=("A01", "A02"))
    assert len(layout.image_files) == 8
    for p in layout.image_files:
        assert p.suffix == ".png"
    # DB has annotate column with alternating 1/2 labels
    with sqlite3.connect(layout.db_path) as conn:
        rows = conn.execute('SELECT annotate FROM "png_list"').fetchall()
    assert sorted(v for (v,) in rows) == [1, 1, 1, 1, 2, 2, 2, 2]


def test_classify_demo_crops_live_where_the_dataset_builder_looks(tmp_path: Path):
    """generate_training_dataset filters png_list on
    ``png_path.str.contains(png_type)`` with png_type='cell_png'. Crops at
    ``data/crop_000.png`` are dropped before a single class is built."""
    layout = syn.generate_classify_demo(tmp_path, n_crops=8,
                                        wells=("A01", "A02"))
    for p in layout.image_files:
        assert syn.CROP_FOLDER in p.parts, p
    settings = syn.demo_settings("classify", str(tmp_path))
    assert settings["png_type"] == syn.CROP_FOLDER
    # 'annotation' mode, because the labels are in png_list rather than in
    # well metadata; the shipped default ('metadata') cannot see them.
    assert settings["dataset_mode"] == "annotation"
    assert settings["annotation_column"] == "annotate"


def test_classify_demo_model_type_is_a_real_torchvision_model(tmp_path: Path):
    """model_type is handed to torchvision. 'cnn' is not a model name, and the
    Classify screen's combo offers only names from this list."""
    from spacr.gui_utils import _torchvision_model_names
    settings = syn.demo_settings("classify", str(tmp_path))
    assert settings["model_type"] in _torchvision_model_names()


def test_classify_demo_spans_enough_wells_to_hold_one_out(tmp_path: Path):
    """The classifier's leakage-safe validation split groups folds by well and
    stops on a single-group training half."""
    layout = syn.generate_classify_demo(tmp_path)
    with sqlite3.connect(layout.db_path) as conn:
        wells = {w for (w,) in conn.execute('SELECT DISTINCT wellID FROM "png_list"')}
    assert len(wells) >= 4, wells


def test_classify_demo_settings_carry_annotation_column(tmp_path: Path):
    layout = syn.generate_classify_demo(tmp_path, n_crops=4,
                                        wells=("A01", "A02"))
    with open(layout.settings_csv) as f:
        settings = {k: v for k, v in csv.reader(f)}
    assert settings["annotation_column"] == "annotate"


def test_classify_demo_names_the_channel_key_the_classifier_reads(tmp_path: Path):
    """`train_channels`, not `channels`.

    Classify runs ``spacr.deep_spacr.deep_spacr``, which selects crop planes
    with ``settings['train_channels']``. ``channels`` is not in
    ``deep_spacr_defaults`` at all, so the Classify screen has no widget for
    it and importing the demo dropped it silently.
    """
    from spacr.settings import deep_spacr_defaults
    settings = syn.demo_settings("classify", str(tmp_path))
    assert settings["train_channels"] == ["r", "g", "b"]
    assert "channels" not in settings
    declared = deep_spacr_defaults({})
    assert "train_channels" in declared
    assert "channels" not in declared


def test_classify_demo_ships_no_setting_the_classify_form_rewrites(tmp_path: Path):
    """``channel_of_interest`` is a spacr.ml setting rendered as a QSpinBox.

    Shipping ``None`` for it meant the CSV said None and the form said 3 —
    and deep_spacr never reads the key either way.
    """
    settings = syn.demo_settings("classify", str(tmp_path))
    assert "channel_of_interest" not in settings


def test_classify_demo_layout_is_not_measure_crops_layout(tmp_path: Path):
    """The docstring used to say the crops were laid out "exactly as
    measure_crop lays them out". They are not. Pin the real difference.

    What matches: the file name, and the ``cell_png/`` leaf that
    ``generate_training_dataset``'s ``png_type`` filter selects on. What does
    not: measure_crop buckets crops by what they contain first
    (``data/single_nucleus/multiple_pathogens/<plate>_<well>/cell_png/``); the
    demo writes ``data/<plate>_<well>/cell_png/`` with no bucket folders.
    """
    import numpy as np
    from spacr.utils import _generate_names

    layout = syn.generate_classify_demo(tmp_path, n_crops=8,
                                        wells=("A01", "A02"))
    demo_crop = layout.image_files[0]
    demo_parts = demo_crop.relative_to(tmp_path).parts

    # measure_crop's own naming, for one cell with one nucleus and one
    # pathogen in field 1 / time 1 of plate1_A01.
    img_name, fldr, _ = _generate_names(
        file_name="plate1_A01_1_1",
        cell_id=np.array([1]),
        cell_nucleus_ids=np.array([1]),
        cell_pathogen_ids=np.array([1]),
        source_folder=str(tmp_path),
        crop_mode="cell",
    )
    real_parts = Path(fldr, "cell_png", img_name).relative_to(tmp_path).parts

    # Same shape of file name...
    assert re.fullmatch(r"plate1_A0\d_\d+_\d+_\d+\.png", demo_crop.name)
    assert re.fullmatch(r"plate1_A01_\d+_\d+_\d+\.png", img_name)
    # ...same cell_png leaf, which is what png_type selects on...
    assert demo_parts[-2] == real_parts[-2] == syn.CROP_FOLDER
    # ...and a genuinely shorter path: no infection-status buckets.
    assert "single_nucleus" in real_parts and "single_pathogen" in real_parts
    assert not {"single_nucleus", "single_pathogen"} & set(demo_parts)
    assert len(demo_parts) < len(real_parts)

    doc = syn.generate_classify_demo.__doc__ or ""
    assert "multiple_pathogens" in doc and "_nucleus" in doc, (
        "the docstring must name measure_crop's real bucket folders, so a "
        "reader can see this layout is deliberately different")
    assert "filepaths_to_database" in doc and "annotate" in doc, (
        "the docstring must say png_list's columns are not the ones "
        "filepaths_to_database writes, and why")
    assert "exactly as measure_crop lays them out" not in doc, (
        "that claim was false: measure_crop buckets crops by infection "
        "status and this generator does not")


# ---------------------------------------------------------------------------
# Timelapse demo
# ---------------------------------------------------------------------------

def test_timelapse_demo_has_multiple_frames_per_well(tmp_path: Path):
    layout = syn.generate_timelapse_demo(
        tmp_path, wells=("A01",), fields=1, times=6, channels=(0, 1),
    )
    # 1 well × 1 field × 6 times × 2 channels = 12 files
    assert len(layout.image_files) == 12
    # Every filename should carry T01..T06
    times = {
        int(re.search(r"_T(\d+)F", p.name).group(1))
        for p in layout.image_files
    }
    assert times == {1, 2, 3, 4, 5, 6}
    with open(layout.settings_csv) as f:
        settings = {k: v for k, v in csv.reader(f)}
    assert settings["timelapse"] == "True"


def test_timelapse_demo_settings_name_only_the_acquired_channels(tmp_path: Path):
    """The demo emits channels (0, 1). Advertising a pathogen_channel=2 and
    organelle_channel=3 it never acquired was two hard pre-flight errors —
    the run could not start."""
    layout = syn.generate_timelapse_demo(tmp_path, times=4)
    settings = _load(layout)
    assert settings["channels"] == [0, 1]
    assert settings["nucleus_channel"] == 0
    assert settings["cell_channel"] == 1
    assert settings["pathogen_channel"] is None
    assert settings["organelle_channel"] is None
    errors, warnings = _preflight(settings, "timelapse")
    assert not errors, [str(p) for p in errors]
    assert not warnings, [str(p) for p in warnings]


def test_timelapse_demo_uses_a_tracker_that_is_always_installed(tmp_path: Path):
    """'trackastra' is the shipped default and an *optional* dependency. A
    demo that needs a pip install before it runs is not a demo."""
    layout = syn.generate_timelapse_demo(tmp_path, times=4)
    assert _load(layout)["timelapse_mode"] == "iou"


def test_timelapse_frame_limits_keep_every_frame(tmp_path: Path):
    """`timelapse_frame_limits` is a *slice* — spacr.object does
    ``stack[limits[0]:limits[1]]``. Shipping ``[1, times]`` read as a 1-based
    inclusive range and silently discarded the first frame of every field."""
    layout = syn.generate_timelapse_demo(tmp_path, times=5)
    limits = _load(layout)["timelapse_frame_limits"]
    assert limits == [0, 5]
    assert len(range(*limits)) == 5


def test_timelapse_frames_hold_the_same_cells_moving(tmp_path: Path):
    """Consecutive frames must be the same cells a few pixels away.

    Each timepoint used to be seeded independently, so frame 2 was an
    unrelated field: nothing for a tracker to link, which is the entire point
    of the timelapse demo.
    """
    from scipy import ndimage

    def centroids(frame: int):
        mask = syn._synth_field(seed=42, channels=(0, 1), frame=frame).masks["cell"]
        n = int(mask.max())
        return np.array(ndimage.center_of_mass(mask > 0, mask, range(1, n + 1)))

    first, second, last = centroids(0), centroids(1), centroids(7)
    assert first.shape == second.shape == last.shape

    step = np.linalg.norm(second - first, axis=1)
    assert np.all(step > 0), "cells did not move at all between frames"
    # Small enough that an IoU tracker still overlaps them frame to frame.
    assert np.all(step < syn._RADIUS_CELL), f"cells jumped {step.max():.1f} px"
    # And they keep drifting rather than resetting.
    assert np.linalg.norm(last - first, axis=1).mean() > step.mean()


# ---------------------------------------------------------------------------
# save_settings_csv + demo_settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["mask", "measure", "crop",
                                     "classify", "timelapse",
                                     "map_barcodes"])
def test_demo_settings_include_src(app_key, tmp_path: Path):
    settings = syn.demo_settings(app_key, str(tmp_path))
    assert settings["src"] == str(tmp_path)


def test_demo_settings_default_to_the_full_four_channel_layout(tmp_path: Path):
    settings = syn.demo_settings("mask", str(tmp_path))
    assert settings["channels"] == [0, 1, 2, 3]
    for key, index in syn.CHANNEL_LAYOUT.items():
        assert settings[key] == index


def test_save_settings_csv_roundtrip(tmp_path: Path):
    settings = {"src": str(tmp_path), "n": 42, "flag": True, "opt": None}
    p = syn.save_settings_csv(tmp_path / "s.csv", settings)
    with open(p) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Key", "Value"]
    kv = {k: v for k, v in rows[1:]}
    assert kv["src"] == str(tmp_path)
    assert kv["n"] == "42"
    assert kv["flag"] == "True"
    assert kv["opt"] == ""     # None → empty string


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_generates_all_demos(tmp_path: Path):
    rc = syn.main(["all", str(tmp_path)])
    assert rc == 0
    for name in ("mask", "measure", "crop", "classify", "timelapse",
                 "map_barcodes"):
        assert (tmp_path / name).is_dir(), f"missing {name} demo dir"
