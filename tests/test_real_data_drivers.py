"""The contract the real-data drivers in ``spacr_drivers/`` keep.

Those scripts are how a claim about a real run -- "measure reproduces the
reference database", "the wrong barcode orientation maps zero reads and says
nothing" -- can be re-checked instead of taken on trust. They only earn that
if three things hold on a machine that has none of the datasets:

* a driver pointed at data that is not there REFUSES, names every input it
  could not find, and exits 2. Half a run leaves a tree that looks like a
  result;
* the refusal costs nothing, because nothing heavy is imported until the
  preconditions hold;
* a driver never writes into a dataset. Inputs are copied out to scratch, and
  staging into the dataset root is refused outright rather than remembered
  not to be done.

Every test here runs with no dataset present, which is the only way this file
means anything anywhere but the machine the runs were recorded on.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO_ROOT / "spacr_drivers"


def _driver_paths():
    """Every driver script, so a new one is held to the contract automatically."""
    return sorted(DRIVER_DIR.glob("drive_*.py"))


DRIVERS = _driver_paths()
DRIVER_IDS = [path.stem for path in DRIVERS]


@pytest.fixture(scope="module")
def support():
    """The shared driver support module, loaded from its path.

    ``spacr_drivers`` is deliberately not a package: it is a folder of
    scripts, and giving it an ``__init__.py`` would make ``find_packages``
    ship it inside the wheel.
    """
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        "spacr_drivers_support", DRIVER_DIR / "_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path[:] = saved


def _load_driver(path):
    """Import one driver script under its own module name."""
    spec = importlib.util.spec_from_file_location(f"driver_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Refusing, rather than half-running
# ---------------------------------------------------------------------------


def test_require_names_every_missing_input_not_just_the_first(support, tmp_path):
    """One refusal has to be enough to fix the whole problem.

    Reporting the first missing file turns an unmounted disk into as many
    failed runs as the driver has inputs.
    """
    with pytest.raises(support.MissingData) as refusal:
        support.require(tmp_path, ("reads/R1.fastq.gz", "barcodes/row.csv",
                                   "settings/run.csv"))
    message = str(refusal.value)
    assert "reads/R1.fastq.gz" in message
    assert "barcodes/row.csv" in message
    assert "settings/run.csv" in message
    assert "3 of 3" in message


def test_require_says_when_the_dataset_root_itself_is_absent(support, tmp_path):
    """An unmounted disk and a missing file are different problems."""
    absent = tmp_path / "not_mounted"
    with pytest.raises(support.MissingData) as refusal:
        support.require(absent, ("anything",), what="the sequencing lane")
    message = str(refusal.value)
    assert "the sequencing lane" in message
    assert str(absent) in message
    assert "does not exist" in message


def test_require_accepts_a_glob_that_matches_at_least_once(support, tmp_path):
    """A driver declares "at least one field" without naming the fields."""
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "plate1_E01_9_1.npy").write_bytes(b"")
    assert support.require(tmp_path, ("merged/*.npy",)) == tmp_path.resolve()


def test_require_refuses_a_glob_that_matches_nothing(support, tmp_path):
    """An empty folder is not the same as a folder full of fields."""
    (tmp_path / "merged").mkdir()
    with pytest.raises(support.MissingData, match=r"merged/\*\.npy"):
        support.require(tmp_path, ("merged/*.npy",))


def test_missing_settings_file_names_every_place_it_looked(support, tmp_path):
    """"Not found" is not actionable; "look in one of these" is."""
    with pytest.raises(support.MissingData) as refusal:
        support.settings_file(tmp_path, ("settings/a.csv", "../settings/b.csv"),
                              what="the measure run")
    message = str(refusal.value)
    assert "settings/a.csv" in message
    assert "settings/b.csv" in message
    assert "the measure run" in message


def test_settings_file_returns_the_first_candidate_that_exists(support, tmp_path):
    """Order is preference: the nearest recorded location wins."""
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "b.csv").write_text("Key,Value\n")
    found = support.settings_file(tmp_path, ("settings/a.csv", "settings/b.csv"))
    assert found == (tmp_path / "settings" / "b.csv").resolve()


# ---------------------------------------------------------------------------
# Never writing into the dataset
# ---------------------------------------------------------------------------


def test_staging_into_the_dataset_root_is_refused(support, tmp_path):
    """The one mistake these drivers must not be able to make.

    A run whose scratch tree is inside the dataset writes results into the
    data it was supposed to only read, and the reference stops being a
    reference.
    """
    dataset = tmp_path / "plate1"
    (dataset / "merged").mkdir(parents=True)
    (dataset / "merged" / "field.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inside the dataset root"):
        support.stage(dataset, ("merged/field.npy",), dataset / "scratch")


def test_staging_leaves_the_dataset_exactly_as_it_found_it(support, tmp_path):
    """Copy out, never write in."""
    dataset = tmp_path / "plate1"
    (dataset / "merged").mkdir(parents=True)
    (dataset / "merged" / "field.npy").write_bytes(b"pixels")
    before = sorted(p.relative_to(dataset) for p in dataset.rglob("*"))

    work = tmp_path / "scratch"
    support.stage(dataset, ("merged/*.npy",), work)

    assert (work / "merged" / "field.npy").read_bytes() == b"pixels"
    assert sorted(p.relative_to(dataset) for p in dataset.rglob("*")) == before


def test_a_staged_copy_of_a_read_only_dataset_can_be_written(support, tmp_path):
    """Datasets are often read-only, and ``copy2`` preserves the mode.

    Without this the copy succeeds and the run then fails on the write it
    exists to make.
    """
    dataset = tmp_path / "plate1"
    dataset.mkdir()
    source = dataset / "measurements.db"
    source.write_bytes(b"db")
    source.chmod(0o444)

    work = tmp_path / "scratch"
    support.stage(dataset, ("measurements.db",), work)
    (work / "measurements.db").write_bytes(b"changed")
    assert source.read_bytes() == b"db"


def test_flattened_staging_drops_the_folder_layout(support, tmp_path):
    """The mask pipeline's ``src`` must hold the images directly.

    Pointed at a folder whose images are one level down it finds zero fields,
    so a driver that stages ``orig/*.tif`` has to flatten them.
    """
    dataset = tmp_path / "plate1"
    (dataset / "orig").mkdir(parents=True)
    (dataset / "orig" / "plate1_E01_F001.tif").write_bytes(b"tif")

    work = tmp_path / "scratch"
    support.stage(dataset, ("orig/*.tif",), work, flatten=True)
    assert (work / "plate1_E01_F001.tif").is_file()
    assert not (work / "orig").exists()


def test_scratch_empties_a_directory_it_reuses(support, tmp_path, monkeypatch):
    """A rerun must mean the same thing as a first run.

    spaCR pipelines skip work whose output is already there, so a scratch tree
    left over from a previous run makes the next one measure nothing and still
    report success.
    """
    monkeypatch.setenv("SPACR_DRIVER_SCRATCH", str(tmp_path))
    first = support.scratch("measure_on_plate1")
    (first / "leftover.csv").write_text("stale")
    second = support.scratch("measure_on_plate1")
    assert second == first
    assert not (second / "leftover.csv").exists()


def test_dataset_root_prefers_the_argument_over_the_recorded_default(support):
    """The recorded path is a default, not a requirement."""
    assert support.dataset_root(["driver.py", "/data/copy"], "/recorded") == \
        Path("/data/copy")
    assert support.dataset_root(["driver.py"], "/recorded") == Path("/recorded")
    assert str(support.dataset_root(["driver.py", "~/copy"], "/recorded")) \
        .startswith(os.path.expanduser("~"))


# ---------------------------------------------------------------------------
# Settings: loaded the way spaCR loads them
# ---------------------------------------------------------------------------


def test_a_blank_cell_loads_as_empty_and_not_as_a_number(support, tmp_path):
    """``hinge_threshold,`` is a box nobody filled in, not an answer.

    Reading a settings CSV with pandas turns every blank cell into ``nan``,
    and a pipeline then refuses the run over a value the user never set --
    naming a setting the chosen analysis does not even read.
    """
    path = tmp_path / "regression.csv"
    path.write_text("Key,Value\nhinge_threshold,\nregression_type,ols\n")
    settings = support.read_settings(path)
    assert settings["hinge_threshold"] is None
    assert settings["regression_type"] == "ols"


def test_settings_values_come_back_as_the_types_they_were_written_as(support,
                                                                     tmp_path):
    """A list has to survive the round trip as a list."""
    path = tmp_path / "measure.csv"
    path.write_text('Key,Value\nchannels,"[0, 1, 2, 3]"\nsave_png,True\n'
                    'png_size,"[[224, 224]]"\n')
    settings = support.read_settings(path)
    assert settings["channels"] == [0, 1, 2, 3]
    assert settings["save_png"] is True
    assert settings["png_size"] == [[224, 224]]


def test_undeclared_names_settings_nothing_reads_any_more(support):
    """A key spaCR no longer declares is a key whose value is ignored."""
    stale = support.undeclared({"channels": [0], "upscale_factor": 2}, "mask")
    assert stale == ["upscale_factor"]


def test_preflight_refuses_a_settings_dict_with_an_error(support, tmp_path):
    """The check the GUI makes before a run, which a bare script skips."""
    with pytest.raises(support.MissingData, match="pre-flight found"):
        support.preflight({"src": str(tmp_path / "absent"),
                           "cell_channel": 0}, "measure")


def test_preflight_lets_a_declared_false_positive_through(support, tmp_path,
                                                          capsys):
    """A wrong check must be overridable by name, and only by name.

    Regression fits score and count tables and never opens a measurements
    database, so requiring one refuses a run that would have worked. Waiving
    it wholesale would make the gate optional for everything else, so the
    waiver names the setting and prints the reason.
    """
    settings = {"src": str(tmp_path), "cell_channel": 0}
    with pytest.raises(support.MissingData):
        support.preflight(settings, "measure")
    support.preflight(settings, "measure",
                      {"src": "measure's database rule, applied here by hand"})
    assert "pre-flight error overridden" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Every driver keeps the contract
# ---------------------------------------------------------------------------


def test_there_are_drivers_to_check():
    """A glob that stops matching would make every test below vacuous."""
    assert len(DRIVERS) >= 7, f"only found {DRIVER_IDS} under {DRIVER_DIR}"


@pytest.mark.parametrize("path", DRIVERS, ids=DRIVER_IDS)
def test_a_driver_declares_what_it_reads_and_where_it_reads_it_from(path):
    """``DEFAULT_ROOT`` and ``REQUIRED`` are what make a driver checkable."""
    module = _load_driver(path)
    assert isinstance(module.DEFAULT_ROOT, str) and module.DEFAULT_ROOT
    assert module.REQUIRED, "a driver that declares no inputs cannot refuse"
    assert all(isinstance(item, str) for item in module.REQUIRED)
    assert callable(module.main)


@pytest.mark.parametrize("path", DRIVERS, ids=DRIVER_IDS)
def test_a_driver_refuses_when_the_data_is_absent(path, tmp_path):
    """Pointed at an empty folder, a driver names what it wanted and exits 2.

    This is the half of the contract that runs on a machine with none of the
    datasets on it, and the half that stops a driver from writing output for a
    run it could never have completed.
    """
    result = subprocess.run(
        [sys.executable, str(path), str(tmp_path)],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "REFUSED" in result.stderr
    module = _load_driver(path)
    first = module.REQUIRED[0]
    assert first in result.stderr, (
        f"the refusal does not name {first}:\n{result.stderr}")
    assert not list(tmp_path.iterdir()), (
        "the driver wrote into the dataset root it had just refused")


@pytest.mark.parametrize("path", DRIVERS, ids=DRIVER_IDS)
def test_a_driver_imports_nothing_heavy_before_it_has_checked(path):
    """The refusal has to cost a bare interpreter start.

    ``import torch`` alone is seconds, and a driver that pays it before
    looking at the disk turns "this data is not here" into a wait. The heavy
    imports belong inside ``main``, after ``require``.
    """
    banned = {"torch", "spacr", "numpy", "pandas", "cellpose", "matplotlib",
              "PySide6", "sklearn", "skimage"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if name.split(".")[0] in banned:
                offenders.append((node.lineno, name))
    assert not offenders, (
        f"{path.name} imports at module scope: {offenders}. Move them into "
        f"main(), after the preconditions have been checked.")


# ---------------------------------------------------------------------------
# The comparisons the drivers make, checked without their datasets
# ---------------------------------------------------------------------------


def test_a_merged_field_name_maps_to_the_database_field_id():
    """``plate1_E01_9_1.npy`` is field ``f9``.

    The comparison is restricted to the fields that were re-measured, so a
    wrong mapping here silently compares three fields against thirteen.
    """
    measure = _load_driver(DRIVER_DIR / "drive_measure_on_plate1.py")
    assert measure.field_ids(("1_1", "9_1", "10_1")) == {"f1", "f9", "f10"}


def test_a_column_name_that_carries_its_prefix_twice_is_folded():
    """The reference database spells blur ``cell_channel_0_cell_channel_0_blur``.

    The prefix was applied once by the measurement and again by the writer.
    Folding it is what lets the VALUES be compared instead of the column being
    reported as one the reference has and the run does not.
    """
    measure = _load_driver(DRIVER_DIR / "drive_measure_on_plate1.py")
    fold = measure.collapse_doubled_prefix
    assert fold("cell_channel_0_cell_channel_0_blur") == "cell_channel_0_blur"
    assert fold("nucleus_channel_3_nucleus_channel_3_blur") == \
        "nucleus_channel_3_blur"
    assert fold("cell_channel_0_frac_high90") == "cell_channel_0_frac_high90"
    assert fold("cell_area") == "cell_area"


def test_the_parent_of_an_object_is_the_label_that_covers_most_of_it():
    """``cell_id`` is checked against the masks, not against the reference.

    A child straddling two cells belongs to the one holding the majority of
    its pixels, which is the rule ``_map_child_to_parent`` uses; a child
    outside every cell has no parent.
    """
    import numpy as np

    measure = _load_driver(DRIVER_DIR / "drive_measure_on_plate1.py")
    cells = np.zeros((6, 6), dtype=np.uint16)
    cells[:, :4] = 7
    cells[:, 4:] = 9
    children = np.zeros((6, 6), dtype=np.uint16)
    children[0:2, 2:6] = 1          # 4 pixels in cell 7, 4 in cell 9
    children[3:5, 0:3] = 2          # entirely in cell 7
    children[5:6, 0:1] = 3
    cells[5, 0] = 0                 # child 3 sits outside every cell

    assert measure.dominant_cell(cells, children, 2) == 7
    assert measure.dominant_cell(cells, children, 3) == 0
    # A tie is broken toward the lower label, which is what argmax does; the
    # point of the assertion is that a straddling child gets a real parent.
    assert measure.dominant_cell(cells, children, 1) in (7, 9)


def test_flipping_the_barcode_orientation_toggles_the_rc_suffix():
    """Running with the forward files is how the silent zero count is shown.

    The recorded settings name the gRNA and row references as reverse
    complements and the column reference forward. The reproduction has to
    flip each one the right way round, or it tests nothing.
    """
    reads = _load_driver(DRIVER_DIR / "drive_map_barcodes_on_real_reads.py")
    flipped = reads.flip_orientation({
        "grna_csv": "/x/grna_barcodes_RC.csv",
        "row_csv": "/x/primers_3_row_barecodes_RC.csv",
        "column_csv": "/x/primers_3_column_barecodes.csv"})
    assert flipped["grna_csv"] == "/x/grna_barcodes.csv"
    assert flipped["row_csv"] == "/x/primers_3_row_barecodes.csv"
    assert flipped["column_csv"] == "/x/primers_3_column_barecodes_RC.csv"


def test_crops_are_labelled_by_the_folder_they_were_exported_into(tmp_path):
    """Dataset generation from ANNOTATIONS needs the annotations to exist.

    The class of a crop is the phenotype folder it sits in, and the counts
    come back so a project whose crops are not sorted that way says so instead
    of training on one class.
    """
    import sqlite3

    cv = _load_driver(DRIVER_DIR / "drive_classify_cv_on_plate1.py")
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE png_list (png_path TEXT, annotate INTEGER)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, NULL)",
            [("/p/data/single_nucleus/a.png",),
             ("/p/data/single_nucleus/b.png",),
             ("/p/data/multiple_nucleus/c.png",)])

    counts = cv.annotate_from_folders(database, ("single_nucleus",
                                                 "multiple_nucleus"))
    assert counts == {"single_nucleus": 2, "multiple_nucleus": 1}
    with sqlite3.connect(database) as connection:
        labels = dict(connection.execute(
            "SELECT png_path, annotate FROM png_list").fetchall())
    assert labels["/p/data/single_nucleus/a.png"] == 1
    assert labels["/p/data/multiple_nucleus/c.png"] == 2


def test_objects_are_counted_per_mask_plane_ignoring_the_background(tmp_path):
    """The mask comparison counts LABELS, and 0 is not an object.

    Counting unique values instead would report one object for an empty
    field, which is the answer that makes a segmentation that found nothing
    look like a segmentation that worked.
    """
    import numpy as np

    mask = _load_driver(DRIVER_DIR / "drive_mask_on_plate1.py")
    planes = np.zeros((4, 4, 7), dtype=np.uint16)
    planes[0:2, 0:2, 4] = 1
    planes[2:4, 2:4, 4] = 3          # labels need not be contiguous
    planes[0:1, 0:1, 5] = 1
    # plane 6 is left empty: no pathogen was segmented in this field.
    path = tmp_path / "plate1_E01_1_1.npy"
    np.save(path, planes)
    counts = mask.count_objects(path)
    assert counts == {"cell": 2, "nucleus": 1, "pathogen": 0}
