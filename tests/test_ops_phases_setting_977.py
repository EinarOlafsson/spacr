"""Item 977: ``ops_phases`` lets a settings file name the OPS phases it runs.

``spacr-run ops <file>`` calls ``run_ops(settings)`` with no phase list, so
before this setting a file could not reproduce a run past decode. The
default (None or empty) keeps the four default phases; the caller's own
phases always win; an unknown phase is refused before anything runs.
"""
from __future__ import annotations

import pytest

from spacr import ops_engine
from spacr.ops_engine import _ALL_PHASES, _PHASES, _run_phases


@pytest.mark.parametrize("value", [None, "", [], " , "])
def test_an_empty_setting_keeps_the_default_phases(value):
    assert _run_phases({"ops_phases": value}, None) == _PHASES
    assert _run_phases({}, None) == _PHASES


def test_the_setting_names_the_phases_in_order():
    value = "stitch, phenotype;objects,decode,ph_segment,ph_measure,ph_masks,merge,aggregate"
    assert _run_phases({"ops_phases": value}, None) == (
        "stitch", "phenotype", "objects", "decode", "ph_segment", "ph_measure",
        "ph_masks", "merge", "aggregate")
    assert _run_phases({"ops_phases": ["merge", "aggregate"]}, None) == (
        "merge", "aggregate")


def test_the_callers_phases_win():
    assert _run_phases({"ops_phases": "merge"}, ("stitch",)) == ("stitch",)
    assert _run_phases({"ops_phases": "merge"}, ()) == ()


def test_an_unknown_phase_is_refused_by_name():
    with pytest.raises(ValueError, match=r"ops_phases names phases spaCR does not know: \['stich'\]"):
        _run_phases({"ops_phases": "stitch,stich"}, None)
    assert all(_run_phases({"ops_phases": name}, None) == (name,) for name in _ALL_PHASES)


def test_run_ops_reads_the_setting_and_refuses_before_any_work(tmp_path, monkeypatch):
    """The refusal comes before any folder is read or written."""
    with pytest.raises(ValueError, match="ops_phases"):
        ops_engine.run_ops({"ops_phases": "nonsense", "dst_root": str(tmp_path / "out")})
    assert not (tmp_path / "out").exists()


def test_spacr_run_hands_the_settings_to_run_ops_without_phases(monkeypatch):
    """The CLI path: ``_call_entry`` passes the settings alone, so the setting decides."""
    from spacr import cli

    seen = {}

    def fake_run_ops(settings, *, wells=None, phases=None, library=None):
        seen["phases"] = _run_phases(settings, phases)
        return {}

    module = cli.resolve_module("ops")
    cli._call_entry(module, fake_run_ops, {"ops_phases": "merge,aggregate"})
    assert seen["phases"] == ("merge", "aggregate")


def test_the_setting_is_registered_with_its_type_category_and_help():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import ALPHA_FEATURES

    assert OPS_DEFAULTS["ops_phases"] is None
    assert OPS_TYPES["ops_phases"] == (str, list, type(None))
    assert any("ops_phases" in keys for keys in OPS_CATEGORIES.values())
    assert "Default None. Alpha." in OPS_TOOLTIPS["ops_phases"]
    assert ALPHA_FEATURES[977] == {"settings": ("ops_phases",)}


def test_spacr_run_ops_preflights_genotype_source_not_src(tmp_path):
    """``spacr-run ops`` checks the OPS input (genotype_source), not ``src``.

    Before item 977 the ops module had no validate key, so the pre-flight
    asked for ``src``, which OPS never reads, and refused every OPS
    settings file.
    """
    from spacr import cli
    from spacr.validate import validate_settings

    assert cli.resolve_module("ops").validate_key == "ops"
    seq = tmp_path / "sequencing"
    seq.mkdir()
    problems = validate_settings({"genotype_source": str(seq)}, "ops")
    assert not any(getattr(p, "setting", getattr(p, "key", "")) == "src" for p in problems)


def test_a_single_site_read_back_from_a_settings_file_is_accepted(tmp_path):
    """A one-site list round-trips through a CSV as an int, and is still a site list."""
    from spacr.cli import load_settings_file, write_settings_file
    from spacr.ops_engine import _ph_sites, _ops_sites
    from spacr.ops_settings import OPS_TYPES, ops_defaults

    settings = ops_defaults({})
    settings.update({"ops_ph_sites": "1124", "ops_sites": "281"})
    path = tmp_path / "s.csv"
    write_settings_file(settings, path)
    back = load_settings_file(str(path))
    for key in ("ops_ph_sites", "ops_sites"):
        assert isinstance(back[key], OPS_TYPES[key])
    assert _ph_sites(back["ops_ph_sites"]) == {1124}
    assert _ops_sites(back) == {281}
