"""A CellProfiler pipeline runs as a Measure step, in its own environment.

Measure hands every field's channels and masks to CellProfiler as TIFFs,
CellProfiler runs the lab's own ``.cppipe`` headless in the backend's
environment, and each CellProfiler object's measurements come back into
``measurements.db:cellprofiler_<object>`` keyed by spaCR's own object ids.
The CellProfiler run itself is stood in for here; the real run is in
``test_the_example_pipeline_runs_in_the_installed_backend``, which runs only
where the backend is installed.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from spacr import _segmentation_backends as SB
from spacr import measure
from spacr.tabular import read_table

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "data", "cellprofiler", "example.cppipe")
STEM = "plate1_A01_1"


def _project(tmp_path):
    """One merged field: two channels, a cell mask and a nucleus mask."""
    merged = tmp_path / "merged"
    merged.mkdir()
    cells = np.zeros((40, 40), np.uint16)
    cells[2:18, 2:18] = 7
    cells[22:38, 22:38] = 9
    nuclei = np.zeros((40, 40), np.uint16)
    nuclei[6:14, 6:14] = 3
    nuclei[26:34, 26:34] = 5
    rng = np.random.default_rng(0)
    dna = (rng.random((40, 40)) * 100 + (nuclei > 0) * 1000).astype(np.uint16)
    actin = (rng.random((40, 40)) * 100 + (cells > 0) * 500).astype(np.uint16)
    np.save(merged / f"{STEM}.npy",
            np.stack([dna, actin, cells, nuclei], axis=-1))
    (tmp_path / "measurements").mkdir()
    settings = {"src": str(merged), "cell_mask_dim": 2,
                "nucleus_mask_dim": 3, "pathogen_mask_dim": None,
                "timelapse": False}
    return settings, str(tmp_path / "measurements" / "measurements.db")


def test_every_field_is_exported_under_names_a_pipeline_can_match(tmp_path):
    import tifffile

    settings, _db = _project(tmp_path)
    out = tmp_path / "tiffs"
    out.mkdir()
    written = measure._cellprofiler_export(settings["src"], settings, out)
    names = sorted(os.path.basename(p) for p in written)
    assert names == [f"{STEM}_cell_mask.tif", f"{STEM}_ch0.tif",
                     f"{STEM}_ch1.tif", f"{STEM}_nucleus_mask.tif"]
    cells = tifffile.imread(out / f"{STEM}_cell_mask.tif")
    assert cells.dtype == np.uint16 and set(np.unique(cells)) == {0, 7, 9}
    for name in names:
        match = measure._CELLPROFILER_FILE.match(name)
        assert match and match.group("stem") == STEM


def test_object_names_say_which_spacr_mask_they_mean():
    roles = ["cell", "nucleus", "pathogen"]
    assert measure._cellprofiler_role("Nuclei", roles) == "nucleus"
    assert measure._cellprofiler_role("nucleus", roles) == "nucleus"
    assert measure._cellprofiler_role("Cells", roles) == "cell"
    assert measure._cellprofiler_role("PathogenObjects", roles) == "pathogen"
    assert measure._cellprofiler_role("Speckles", roles) is None


def _fake_runner(objects):
    """A stand-in for the CellProfiler worker that reports ``objects``:
    ``{name: [(x, y, object number, area), ...]}`` for image set 1."""
    def run(pipeline, files, output):
        """Write each object's table where the worker would."""
        os.makedirs(output, exist_ok=True)
        assert all(os.path.isfile(f) for f in files)
        reply = {"image_sets": 1,
                 "images": {"1": [os.path.basename(f) for f in files
                                  if f.endswith("_ch0.tif")]},
                 "objects": {}}
        for i, (name, rows) in enumerate(objects.items()):
            table = np.array([[1, n, x, y, area] for x, y, n, area in rows],
                             dtype=float)
            path = os.path.join(output, f"objects_{i}.npy")
            np.save(path, table)
            reply["objects"][name] = {
                "columns": ["ImageNumber", "ObjectNumber",
                            "Location_Center_X", "Location_Center_Y",
                            "AreaShape_Area"],
                "path": path}
        return reply
    return run


def test_the_step_writes_cellprofiler_objects_keyed_by_spacr_ids(tmp_path):
    settings, db = _project(tmp_path)
    pipeline = tmp_path / "p.cppipe"
    pipeline.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    settings["cellprofiler_pipeline"] = str(pipeline)
    runner = _fake_runner({
        "Nuclei": [(10.0, 10.0, 1, 64.0), (30.0, 30.0, 2, 64.0),
                   (20.0, 1.0, 3, 4.0)],
        "Speckles": [(9.6, 9.6, 1, 256.0), (30.2, 29.8, 2, 256.0)],
    })
    counts = measure._run_cellprofiler_step(db, settings, runner=runner)
    assert counts == {"cellprofiler_nuclei": 3, "cellprofiler_speckles": 2}
    nuclei = read_table(db, table="cellprofiler_nuclei", canonicalise=False)
    assert list(nuclei["object_type"].unique()) == ["nucleus"]
    assert nuclei["object_label"].tolist()[:2] == [3, 5]
    assert pd_isna(nuclei["object_label"].iloc[2])
    assert nuclei["prcfo"].tolist()[:2] == ["plate1_r1_c1_f1_o3",
                                            "plate1_r1_c1_f1_o5"]
    assert nuclei["cp_AreaShape_Area"].tolist() == [64.0, 64.0, 4.0]
    speckles = read_table(db, table="cellprofiler_speckles", canonicalise=False)
    assert speckles["object_type"].iat[0] in ("cell", "nucleus")
    assert speckles["prcfo"].notna().all()
    assert not any(os.path.basename(p).startswith("input_")
                   for p in os.listdir(tmp_path / "cellprofiler"))


def pd_isna(value):
    """Whether a database cell is empty."""
    import pandas as pd

    return bool(pd.isna(value))


def test_a_failed_pipeline_is_reported_and_leaves_the_run_alone(tmp_path,
                                                               capsys):
    settings, db = _project(tmp_path)
    settings["cellprofiler_pipeline"] = str(tmp_path / "missing.cppipe")
    assert measure._run_cellprofiler_step(db, settings,
                                          runner=_fake_runner({})) is None
    assert "could not be run" in capsys.readouterr().out
    assert not os.path.exists(db)


def test_an_uninstalled_cellprofiler_says_how_to_install_it(tmp_path):
    with pytest.raises(ImportError, match="Model Zoo"):
        SB._run_cellprofiler("p.cppipe", ["a.tif"], str(tmp_path / "o"),
                             root=str(tmp_path / "backends"))


def test_the_install_fetches_java_before_it_builds_javabridge():
    spec = SB._SPECS[SB._CELLPROFILER]
    steps = SB._install_plan(spec, "/b/cellprofiler", ("/py3.8",))
    labels = [s.label for s in steps]
    assert labels.index("Fetch a Java development kit") < labels.index(
        "Build CellProfiler's extensions") < len(labels) - 2
    assert labels[-2:] == ["Install CellProfiler", "Check it loads"]
    build = steps[labels.index("Build CellProfiler's extensions")].argv
    assert "--no-build-isolation" in build
    assert "python-javabridge==4.0.3" in build
    fetch = steps[labels.index("Fetch a Java development kit")].argv
    assert fetch[-2:] == ("11", os.path.join("/b/cellprofiler", "jdk"))
    assert not spec.segments and spec.alpha and not spec.torch
    assert "wxPython" not in " ".join(spec.requirements)


def test_the_worker_uses_only_the_java_in_its_own_environment(tmp_path,
                                                              monkeypatch):
    env = tmp_path / "cellprofiler"
    monkeypatch.setenv("JAVA_HOME", "/usr/lib/jvm/elsewhere")
    assert "JAVA_HOME" not in SB._worker_env(SB._CELLPROFILER, str(env))
    home = env / "jdk" / "jdk-11.0.1+1"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "javac").write_text("")
    environ = SB._worker_env(SB._CELLPROFILER, str(env))
    assert environ["JAVA_HOME"] == str(home)
    assert environ["PATH"].startswith(str(home / "bin"))
    assert SB._worker_env(SB._CELLPOSE3, str(env)).get(
        "JAVA_HOME") == "/usr/lib/jvm/elsewhere"


def test_the_record_lists_the_pins_built_from_source():
    spec = SB._SPECS[SB._CELLPROFILER]
    record = {"requirements": list(spec.requirements
                                   + spec.without_dependencies)}
    assert SB._stale_requirements(SB._CELLPROFILER, record) == list(
        spec.built_here)


def _installed_state():
    """CellProfiler's backend state here, or None when it cannot run."""
    state = SB._backend_state(SB._CELLPROFILER)
    return state if state.ready and not state.in_process else None


@pytest.mark.skipif(_installed_state() is None,
                    reason="the CellProfiler backend is not installed here")
def test_the_example_pipeline_runs_in_the_installed_backend(tmp_path):
    settings, db = _project(tmp_path)
    pipeline = tmp_path / "example.cppipe"
    shutil.copy(EXAMPLE, pipeline)
    settings["cellprofiler_pipeline"] = str(pipeline)
    try:
        counts = measure._run_cellprofiler_step(db, settings)
    finally:
        SB._shutdown_workers(SB._CELLPROFILER)
    assert counts and counts["cellprofiler_nuclei"] == 2
    nuclei = read_table(db, table="cellprofiler_nuclei", canonicalise=False)
    assert sorted(nuclei["object_label"].tolist()) == [3, 5]
    assert sorted(nuclei["prcfo"]) == ["plate1_r1_c1_f1_o3",
                                       "plate1_r1_c1_f1_o5"]
    assert (nuclei["cp_AreaShape_Area"] == 64).all()
    cells = read_table(db, table="cellprofiler_cells", canonicalise=False)
    assert sorted(cells["object_label"].tolist()) == [7, 9]
    assert (cells["cp_Intensity_MeanIntensity_Actin"] > 0).all()
