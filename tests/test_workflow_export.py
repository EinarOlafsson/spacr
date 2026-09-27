"""A recorded run exported as a Snakemake or Nextflow workflow.

The workflow runs the run's module once per plate with ``spacr-run`` and one
settings file per plate holding the run's exact settings. Whether the
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
