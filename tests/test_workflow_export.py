"""A recorded run exported as a Snakemake or Nextflow workflow.

The workflow runs the run's module once per plate with ``spacr-run`` and one
settings file per plate, separating explicit output folders for multiple jobs. Whether the
exported workflow reproduces a run end to end was checked with real
Snakemake and Nextflow; these tests pin what is written.
"""
from __future__ import annotations

import json

import pytest

from spacr import cli_repro
from spacr import run_journal as journal


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(journal, "runs_root", lambda: root)
    monkeypatch.setattr(cli_repro, "runs_root", lambda: root)

    def record(app_key, settings):
        with journal.open_run(app_key, settings) as run:
            run.set_status("success")
        return run.dir

    return record


def _settings(folder):
    return {p.stem: json.loads(p.read_text())
            for p in sorted((folder / "settings").glob("*.json"))}


def test_every_plate_of_a_run_becomes_one_job_with_the_run_settings(
        recorded, tmp_path):
    run = recorded("mask", {"src": ["/data/plate 1", "/data/plate2",
                                    "/other/plate2"], "cell_channel": 1})
    main = cli_repro._export_workflow(run, tmp_path / "wf", "snakemake")
    assert main.name == "Snakefile"
    jobs = _settings(tmp_path / "wf")
    assert {k: v["src"] for k, v in jobs.items()} == {
        "plate_1": "/data/plate 1", "plate2": "/data/plate2",
        "plate2_2": "/other/plate2"}
    assert all(v["cell_channel"] == 1 for v in jobs.values())
    text = main.read_text()
    assert "rule spacr_mask:" in text
    assert "{params.spacr_run} mask --settings {input}" in text
    assert 'glob_wildcards("settings/{plate}.json")' in text
    config = (tmp_path / "wf" / "config.yaml").read_text()
    assert 'image: "docker://ghcr.io/einarolafsson/spacr:' in config
    assert 'spacr_run: "spacr-run"' in config


def test_nextflow_export_runs_the_module_per_settings_file(recorded, tmp_path):
    run = recorded("mask", {"src": "/data/plateA"})
    main = cli_repro._export_workflow(run, tmp_path / "nf", "nextflow",
                                      plates=["/x/p1", "/x/p2"],
                                      image="spacr:local",
                                      spacr_run="python -m spacr.cli")
    assert main.name == "main.nf"
    assert [v["src"] for v in _settings(tmp_path / "nf").values()] == [
        "/x/p1", "/x/p2"]
    text = main.read_text()
    assert "process SPACR_MASK" in text
    assert "${params.spacr_run} mask --settings ${settings}" in text
    config = (tmp_path / "nf" / "nextflow.config").read_text()
    assert 'image        = "spacr:local"' in config
    assert 'spacr_run    = "python -m spacr.cli"' in config
    assert "apptainer" in config and "slurm" in config


@pytest.mark.parametrize("engine", ["snakemake", "nextflow"])
@pytest.mark.parametrize("destination_key", ["dst", "dst_root"])
def test_plate_jobs_isolate_explicit_outputs_even_when_plate_names_collide(
        recorded, tmp_path, engine, destination_key):
    output = tmp_path / "results with spaces"
    original = {"src": ["/a/plate", "/b/plate", "/c/plate_2"],
                destination_key: str(output), "custom_model": "/models/shared",
                "reference_file": "/reference/shared.csv"}
    run = recorded("convert", original)
    cli_repro._export_workflow(run, tmp_path / engine, engine)
    jobs = _settings(tmp_path / engine)
    assert set(jobs) == {"plate", "plate_2", "plate_2_2"}
    destinations = {value[destination_key] for value in jobs.values()}
    assert destinations == {str(output / name) for name in jobs}
    assert len(destinations) == len(jobs)
    for name, settings in jobs.items():
        assert settings["custom_model"] == original["custom_model"]
        assert settings["reference_file"] == original["reference_file"]
        assert settings[destination_key].endswith(name)
    assert journal.load_run_settings(run) == original


@pytest.mark.parametrize("sources", [None, "/a/plate", ["/a/plate"]])
def test_single_job_keeps_both_output_roots_exactly(sources):
    original = {"dst": "relative/results", "dst_root": "/separate/results"}
    if sources is not None:
        original["src"] = sources
    jobs = cli_repro._workflow_plates(original)
    assert len(jobs) == 1
    job = next(iter(jobs.values()))
    assert job["dst"] == original["dst"]
    assert job["dst_root"] == original["dst_root"]


@pytest.mark.parametrize("destination", [None, ""])
def test_multi_plate_jobs_leave_unset_destinations_to_the_module(destination):
    original = {"src": ["/a/plate", "/b/plate"], "dst": destination}
    jobs = cli_repro._workflow_plates(original)
    assert all(job["dst"] == destination and "dst_root" not in job
               for job in jobs.values())


def test_replacement_plates_control_whether_destinations_are_split():
    original = {"src": "/recorded/plate", "dst": "/outputs",
                "dst_root": "/other-outputs"}
    jobs = cli_repro._workflow_plates(original, ["/a/plate", "/b/plate"])
    assert jobs["plate"]["dst"] == "/outputs/plate"
    assert jobs["plate_2"]["dst_root"] == "/other-outputs/plate_2"
    assert cli_repro._workflow_plates(original, ["/a/plate"])["plate"] == {
        **original, "src": "/a/plate"}
    assert original == {"src": "/recorded/plate", "dst": "/outputs",
                        "dst_root": "/other-outputs"}


@pytest.mark.parametrize("engine", ["snakemake", "nextflow"])
@pytest.mark.parametrize("module", ["convert", "align"])
def test_output_database_follows_each_module_plate_without_rewriting_inputs(
        recorded, tmp_path, engine, module):
    original = {"src": ["/a/plate", "/b/plate"], "dst": "/outputs",
                "db_path": "/outputs/measurements/result.sqlite",
                "checkpoint_path": "/checkpoints/progress.json",
                "reference_file": "/references/input.sqlite"}
    run = recorded(module, original)
    cli_repro._export_workflow(run, tmp_path / engine, engine)
    jobs = _settings(tmp_path / engine)
    assert set(jobs) == {"plate", "plate_2"}
    for name, job in jobs.items():
        assert job["db_path"] == f"/outputs/{name}/measurements/result.sqlite"
        assert job["reference_file"] == original["reference_file"]
        assert job["checkpoint_path"] == (
            f"/checkpoints/{name}/progress.json" if module == "convert"
            else original["checkpoint_path"])
    assert journal.load_run_settings(run) == original


@pytest.mark.parametrize("module", ["explain_cv", "investigate_hit", "mask"])
def test_input_database_and_model_checkpoint_are_not_output_files(
        recorded, tmp_path, module):
    original = {"src": ["/a/plate", "/b/plate"],
                "db_path": "/inputs/measurements.db",
                "checkpoint_path": "/models/model.ckpt", "resume": True}
    run = recorded(module, original)
    cli_repro._export_workflow(run, tmp_path / "workflow")
    for job in _settings(tmp_path / "workflow").values():
        assert job["db_path"] == original["db_path"]
        assert job["checkpoint_path"] == original["checkpoint_path"]


@pytest.mark.parametrize("engine", ["snakemake", "nextflow"])
def test_explicit_convert_resume_input_cannot_be_silently_relocated(
        recorded, tmp_path, engine):
    checkpoint = tmp_path / "existing.json"
    checkpoint.write_text('{"retained": true}')
    original = {"src": ["/a/plate", "/b/plate"], "dst": "/outputs",
                "checkpoint_path": str(checkpoint), "resume": True}
    run = recorded("convert", original)
    with pytest.raises(ValueError, match="explicit resume checkpoint"):
        cli_repro._export_workflow(run, tmp_path / engine, engine)
    assert checkpoint.read_text() == '{"retained": true}'
    assert not (tmp_path / engine).exists()
    single = cli_repro._workflow_plates(original, ["/a/plate"], module="convert")
    assert single["plate"]["checkpoint_path"] == str(checkpoint)
    defaults = cli_repro._workflow_plates(
        {**original, "checkpoint_path": None}, module="convert")
    assert all(job["checkpoint_path"] is None and job["resume"] for job in defaults.values())


@pytest.mark.parametrize("destination", ["state/db.sqlite", "outputs/../state/db.sqlite"])
def test_external_output_file_gets_a_unique_parent_after_path_normalization(destination):
    jobs = cli_repro._workflow_plates(
        {"src": ["a/plate", "b/plate"], "dst": "outputs", "db_path": destination},
        module="convert")
    assert jobs["plate"]["db_path"] == "state/plate/db.sqlite"
    assert jobs["plate_2"]["db_path"] == "state/plate_2/db.sqlite"


def test_single_plate_preserves_explicit_output_file_paths():
    original = {"src": "/plate", "dst": "/outputs",
                "db_path": "/existing/measurements.db",
                "checkpoint_path": "/existing/resume.json", "resume": True,
                "map_name": "/existing/map.csv"}
    assert cli_repro._workflow_plates(original, module="convert")["plate"] == original


@pytest.mark.parametrize("engine", ["snakemake", "nextflow"])
@pytest.mark.parametrize("map_name,expected", [
    ("/shared/map.csv", "/shared/{plate}/map.csv"),
    ("/outputs/maps/map.csv", "/outputs/{plate}/maps/map.csv"),
    ("../shared/map.csv", "/shared/{plate}/map.csv"),
    ("nested/../../shared/map.csv", "/shared/{plate}/map.csv"),
])
def test_convert_escaped_map_outputs_are_isolated(
        recorded, tmp_path, engine, map_name, expected):
    original = {"src": ["/a/plate", "/b/plate"], "dst": "/outputs",
                "map_name": map_name}
    run = recorded("convert", original)
    cli_repro._export_workflow(run, tmp_path / engine, engine)
    for name, job in _settings(tmp_path / engine).items():
        assert job["map_name"] == expected.format(plate=name)
    assert journal.load_run_settings(run) == original


@pytest.mark.parametrize("map_name", [None, "", "map.csv", "maps/map.csv",
                                     "maps/../map.csv"])
def test_convert_contained_map_names_keep_their_original_spelling(map_name):
    jobs = cli_repro._workflow_plates(
        {"src": ["/a/plate", "/b/plate"], "dst": "/outputs", "map_name": map_name},
        module="convert")
    assert all(job["map_name"] == map_name for job in jobs.values())


def test_convert_escaped_map_with_relative_or_default_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = {"src": ["a/plate", "b/plate"], "map_name": "../map.csv"}
    jobs = cli_repro._workflow_plates({**original, "dst": "outputs"}, module="convert")
    assert jobs["plate"]["map_name"] == str(tmp_path / "plate/map.csv")
    assert jobs["plate_2"]["map_name"] == str(tmp_path / "plate_2/map.csv")
    defaults = cli_repro._workflow_plates(original, module="convert")
    assert defaults["plate"]["map_name"] == str(tmp_path / "a/plate/map.csv")
    assert defaults["plate_2"]["map_name"] == str(tmp_path / "b/plate_2/map.csv")


def test_a_run_without_src_is_one_job_and_versions_pick_the_image(
        recorded, tmp_path):
    run = recorded("external_masks", {"inputs": ["/a", "/b"], "dst": "/o"})
    cli_repro._export_workflow(run, tmp_path / "wf")
    assert _settings(tmp_path / "wf") == {
        "run": {"inputs": ["/a", "/b"], "dst": "/o"}}
    assert cli_repro._workflow_image({"env": {"spacr": "1.5.0"}}).endswith(
        ":1.5.0-cpu")
    assert cli_repro._workflow_image({"env": {"spacr": "1.5.0.dev3"}}).endswith(
        ":cpu")


def test_refusals_are_sentences(recorded, tmp_path):
    run = recorded("mask", {"src": "/p"})
    with pytest.raises(ValueError, match="engine"):
        cli_repro._export_workflow(run, tmp_path / "x", "make")
    with pytest.raises(ValueError, match="not a run folder"):
        cli_repro._export_workflow(tmp_path, tmp_path / "x")
    gui_only = recorded("annotate", {"src": "/p"})
    with pytest.raises(ValueError, match="cannot run headless"):
        cli_repro._export_workflow(gui_only, tmp_path / "x")


def test_the_cli_exports_and_needs_an_out_folder(recorded, tmp_path, capsys):
    run = recorded("mask", {"src": "/p"})
    assert cli_repro.main([str(run), "--export", "nextflow"]) == 2
    assert cli_repro.main([str(run), "--export", "snakemake",
                           "--out", str(tmp_path / "wf")]) == 0
    assert "Snakefile" in capsys.readouterr().out
    assert (tmp_path / "wf" / "settings" / "p.json").is_file()


def test_a_run_named_by_its_folder_name_is_found_in_the_journal(
        recorded, tmp_path, monkeypatch):
    run = recorded("mask", {"src": "/p"})
    monkeypatch.chdir(tmp_path)
    main_file = cli_repro._export_workflow(run.name, tmp_path / "wf")
    assert main_file.name == "Snakefile" and main_file.is_file()


def test_the_cli_turns_an_export_refusal_into_exit_2(recorded, tmp_path,
                                                    capsys):
    gui_only = recorded("annotate", {"src": "/p"})
    assert cli_repro.main([str(gui_only), "--export", "snakemake",
                           "--out", str(tmp_path / "wf")]) == 2
    assert "cannot run headless" in capsys.readouterr().err
    assert not (tmp_path / "wf").exists()
