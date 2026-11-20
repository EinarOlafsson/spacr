"""Standalone OPS power simulations validate their calibration and output paths."""

import json

import pytest

from spacr import cli, validate


@pytest.fixture
def simulation_settings(tmp_path):
    """A simulation has a calibration file and a new destination, with no images."""
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    return {"ops_power_sim": True, "ops_power_calibration_run": str(calibration),
            "dst_root": str(tmp_path / "results")}


@pytest.mark.parametrize("layout", ["tiles", "cpg_cropped"])
@pytest.mark.parametrize("calibration_kind", ["file", "directory"])
def test_a_simulation_uses_its_calibration_without_image_inventory(
        simulation_settings, tmp_path, monkeypatch, layout, calibration_kind):
    """Both image layouts yield to the standalone simulation before scanning inputs."""
    settings = {**simulation_settings, "ops_layout": layout,
                "genotype_source": str(tmp_path / "unused-sequencing"),
                "cpg_root": str(tmp_path / "unused-cropped")}
    if calibration_kind == "directory":
        settings["ops_power_calibration_run"] = str(tmp_path)

    def refuse_inventory(*args, **kwargs):
        """A simulation pre-flight must not inspect image input folders."""
        raise AssertionError("image inputs were inspected")

    monkeypatch.setattr(validate, "_inventory", refuse_inventory)
    monkeypatch.setattr(validate, "_check_cpg_inputs", refuse_inventory)
    monkeypatch.setattr(validate, "_describe_cpg_inputs", refuse_inventory)
    problems = validate.validate_settings(settings, "ops")
    assert not [problem for problem in problems if problem.is_error], problems
    plan = validate.describe_plan(settings, "ops")
    assert "power_sim" in plan
    assert settings["ops_power_calibration_run"] in plan
    assert str(tmp_path / "results" / "results" / "ops_power_sim") in plan
    assert "unused-sequencing" not in plan and "unused-cropped" not in plan
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("missing", ["ops_power_calibration_run", "dst_root"])
def test_missing_simulation_inputs_are_reported(simulation_settings, missing):
    """The pre-flight still requires the two inputs consumed by the simulation."""
    settings = dict(simulation_settings)
    settings.pop(missing)
    errors = [problem for problem in validate.validate_settings(settings, "ops") if problem.is_error]
    assert {problem.setting for problem in errors} == {missing}
    assert errors[0].fix


def test_a_missing_calibration_path_is_reported(simulation_settings, tmp_path):
    """An unavailable calibration names the path before the simulation starts."""
    settings = {**simulation_settings, "ops_power_calibration_run": str(tmp_path / "gone.json")}
    errors = [problem for problem in validate.validate_settings(settings, "ops") if problem.is_error]
    assert {problem.setting for problem in errors} == {"ops_power_calibration_run"}
    assert "gone.json" in errors[0].message


def test_switching_simulation_off_keeps_the_image_source_check(simulation_settings):
    """The regular OPS path continues to require sequencing inputs."""
    settings = {**simulation_settings, "ops_power_sim": False}
    errors = [problem for problem in validate.validate_settings(settings, "ops") if problem.is_error]
    assert "genotype_source" in {problem.setting for problem in errors}


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
def test_cli_accepts_a_standalone_simulation_settings_file(
        simulation_settings, tmp_path, monkeypatch, capsys, dry_run):
    """The real CLI pre-flight reaches the simulation entry or its dry-run plan."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(simulation_settings), encoding="utf-8")
    calls = []

    def run(settings):
        """Record dispatch without performing a simulation."""
        calls.append(settings)

    monkeypatch.setattr(cli, "import_entry", lambda module: run)
    args = ["ops", "--settings", str(settings_file)]
    if dry_run:
        args.append("--dry-run")
    code = cli.main(args)
    output = capsys.readouterr().out
    assert code == cli.EXIT_OK, output
    assert len(calls) == (0 if dry_run else 1)
    assert "power_sim" in output
    if calls:
        assert calls[0]["ops_power_sim"] is True
        assert calls[0]["ops_power_calibration_run"] == simulation_settings["ops_power_calibration_run"]
