"""Pre-flight reads cpg_root on the cropped-site layout; --dry-run shows the effective read mode."""
from __future__ import annotations


from spacr import cli
from spacr.validate import describe_plan, validate_settings
from tests.test_cpg_cropped_reader_919 import make_tree, settings_for


def _errors(problems):
    return [p for p in problems if p.is_error]


def test_a_cpg_cropped_run_needs_cpg_root_not_src(tmp_path):
    root = make_tree(tmp_path / "images_corrected_cropped", wells=("CP1_Well1",), sites=(1, 2))
    settings = settings_for(root, tmp_path / "out")
    assert "src" not in settings
    assert not _errors(validate_settings(settings, ""))
    assert not _errors(validate_settings(settings, "ops"))
    plan = describe_plan(settings, "")
    assert "image files in 2 site(s) of 1 well(s)" in plan and "no image files found" not in plan


def test_a_missing_or_empty_cpg_root_is_still_refused(tmp_path):
    settings = settings_for(tmp_path / "nowhere", tmp_path / "out")
    errors = _errors(validate_settings(settings, ""))
    assert errors and all(e.setting == "cpg_root" for e in errors)
    no_key = {k: v for k, v in settings.items() if k != "cpg_root"}
    assert "cpg_root is missing" in _errors(validate_settings(no_key, ""))[0].message
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _errors(validate_settings(settings_for(empty, tmp_path / "out"), ""))


def test_cpg_sites_that_name_no_site_are_refused(tmp_path):
    root = make_tree(tmp_path / "t", wells=("CP1_Well1",), sites=(1,))
    assert _errors(validate_settings(settings_for(root, tmp_path / "o", cpg_sites="9"), ""))


def test_the_tiles_layout_still_needs_its_source(tmp_path):
    errors = _errors(validate_settings({"ops_layout": "tiles"}, ""))
    assert any("src" in e.message for e in errors)


def test_the_cli_preflight_passes_without_src(tmp_path, capsys):
    root = make_tree(tmp_path / "t", wells=("CP1_Well1",), sites=(1,))
    path = tmp_path / "s.csv"
    cli.write_settings_file(settings_for(root, tmp_path / "o"), path)
    code = cli.main(["ops", "--settings", str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK, out
    assert "src is missing" not in out


def test_dry_run_prints_the_effective_read_mode(tmp_path):
    from spacr.ops_prefix import _prefix_read_mode

    base = {"ops_layout": "cpg_cropped", "ops_read_mode": None,
            "ops_prefix_threshold_tuning": "per_well_decoy"}
    line = [l for l in cli.render_settings(base, "ops").splitlines() if "ops_read_mode" in l][0]
    assert line.endswith("None (effective: auto)")
    assert _prefix_read_mode(base, {"ops_prefix_threshold_tuning": "per_well_decoy"}) == "auto"
    log = dict(base, ops_prefix_threshold_tuning="none")
    shown = [l for l in cli.render_settings(log, "ops").splitlines() if "ops_read_mode" in l][0]
    assert shown.endswith("None (effective: log)")
    raw = dict(base, ops_read_mode="raw")
    assert "(effective: raw)" in cli.render_settings(raw, "ops")
    assert "effective" not in cli.render_settings(base, "mask")
