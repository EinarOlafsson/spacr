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


def _write_profiles(root):
    """A tiny profiles archive: two wells, two genes, non-targeting controls."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(977)
    rows = []
    for well in ("A1", "A2"):
        for gene, guides in (("G1", ("g1a", "g1b")), ("G2", ("g2a",)),
                             ("nontargeting", ("nt1", "nt2", "nt3"))):
            for guide in guides:
                for _ in range(6):
                    rows.append({"plate": "p", "well": well, "site": 1, "label": len(rows) + 1,
                                 "gene": gene, "sgRNA": guide, "mapped_guide": guide,
                                 "cell_area": float(rng.normal(100 + (gene == "G1") * 30, 5)),
                                 "nucleus_dapi_mean_intensity": float(rng.normal(50, 3))})
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "profiles_cells.parquet", index=False)
    (root / "references").mkdir()
    (root / "references" / "string_pair_benchmark.tsv").write_text(
        "gene_name\tpair\nG1\t0\nG2\t0\n", encoding="utf-8")


def test_the_profiles_archive_reduces_to_genes_with_the_aggregate_reducers(tmp_path, monkeypatch):
    from spacr import ops_live

    root = tmp_path / "ops_testdata_funk2022_A1_profiles"
    _write_profiles(root)
    genes = ops_live._profile_genes({}, root)
    assert set(genes["gene"]) == {"G1", "G2", "nontargeting"}
    assert int(genes.set_index("gene").loc["G1", "n_cells"]) == 24
    g1 = float(genes.set_index("gene").loc["G1", "cell_area"])
    g2 = float(genes.set_index("gene").loc["G2", "cell_area"])
    assert g1 > 3 and abs(g2) < 3, "G1's larger cells stand out against the controls"
    monkeypatch.setattr(ops_live, "_profiles_root", lambda: root)
    score = ops_live._string_score(["G1", "G2"], [0, 0], "nontargeting")
    assert score["precision"] == 1.0
    assert ops_live._string_score(["G1", "G2"], None, "nontargeting") is None
