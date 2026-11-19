"""``spacr-run ops`` reads an OPS settings file the way the pre-flight and the run need.

Two bugs met while reproducing an OPS run from a settings file (item 977):
the ``ops`` module had no validate key, so the pre-flight asked for ``src``,
which OPS never reads, and refused every tile-layout OPS settings file; and a
one-site ``ops_sites`` / ``ops_ph_sites`` comes back from a settings CSV as an
int, which the types refused.
"""
from __future__ import annotations


def _problem_keys(problems):
    return {getattr(p, "setting", "") for p in problems}


def test_spacr_run_ops_preflights_genotype_source_not_src(tmp_path):
    from spacr import cli
    from spacr.validate import validate_settings

    assert cli.resolve_module("ops").validate_key == "ops"
    seq = tmp_path / "sequencing"
    seq.mkdir()
    problems = validate_settings({"genotype_source": str(seq)}, "ops")
    assert "src" not in _problem_keys(problems)
    missing = validate_settings({}, "ops")
    assert "genotype_source" in _problem_keys(missing)
    assert "src" not in _problem_keys(missing)


def test_the_cropped_site_layout_still_checks_cpg_root(tmp_path):
    """Item 974's path (d63b9bcaeb) is unchanged under the ops validate key."""
    from spacr.validate import validate_settings

    problems = validate_settings({"ops_layout": "cpg_cropped"}, "ops")
    keys = _problem_keys(problems)
    assert "cpg_root" in keys
    assert "src" not in keys and "genotype_source" not in keys


def test_a_single_site_read_back_from_a_settings_file_is_accepted(tmp_path):
    from spacr.cli import load_settings_file, write_settings_file
    from spacr.ops_engine import _ops_sites, _ph_sites
    from spacr.ops_settings import OPS_TYPES, ops_defaults
    from spacr.validate import validate_settings

    settings = ops_defaults({})
    settings.update({"ops_ph_sites": "1124", "ops_sites": "281"})
    path = tmp_path / "s.csv"
    write_settings_file(settings, path)
    back = load_settings_file(str(path))
    for key in ("ops_ph_sites", "ops_sites"):
        assert isinstance(back[key], OPS_TYPES[key])
    assert _ph_sites(back["ops_ph_sites"]) == {1124}
    assert _ops_sites(back) == {281}
    keys = _problem_keys(validate_settings(dict(back), "ops"))
    assert not keys & {"ops_ph_sites", "ops_sites"}
