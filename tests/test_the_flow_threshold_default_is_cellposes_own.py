"""428: the flow threshold spaCR ships is Cellpose's own, 0.4.

Reported as GitHub #123 by jak18015 on spaCR 1.5.0.8: "the mask generation
says it will run, but warns that the flow thresholds are all defaulting to
100, when cellpose defaults to 1.0". spaCR 1.5.0.5 to 1.5.0.8 filled
``nucleus_flow_threshold``, ``cell_flow_threshold`` and
``pathogen_flow_threshold`` -- and ``FT`` for the apply/test-model
submodules -- with 100, which keeps every mask Cellpose proposes. Its own
pre-flight check then warned about that default on every run, with a fix
line reading "spaCR ships 1.0", which it did not. The reporter's "cellpose
defaults to 1.0" came from that line.

The maintainer chose 0.4 on 2026-09-19. What is held here:

* every default is 0.4, the value Cellpose's own ``eval`` defaults to;
* the shipped defaults no longer trip the warning, through the same
  ``spacr-run validate`` path the report came from;
* a settings file carrying 100 keeps 100 -- a saved choice is never
  rewritten -- and the warning it gets now states the truth.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

import spacr.validate as V
from spacr.settings import (
    get_default_apply_cellpose_model_settings,
    get_default_test_cellpose_model_settings,
    get_timelapse_settings,
    set_default_settings_preprocess_generate_masks,
)

MASK_KEYS = ("nucleus_flow_threshold", "cell_flow_threshold",
             "pathogen_flow_threshold")


def _raw_plate(root: Path) -> str:
    """A folder of CellVoyager-named raw tifs the Mask pre-flight accepts."""
    plate = root / "plate1"
    plate.mkdir(parents=True, exist_ok=True)
    for field in range(1, 3):
        for chan in range(1, 5):
            (plate / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif"
             ).write_bytes(b"")
    return str(plate)


def _flow_problems(problems):
    """The problems about a flow threshold, and nothing else."""
    return [p for p in problems
            if p.setting and (p.setting.endswith("_flow_threshold")
                              or p.setting in ("FT", "flow_threshold"))]


def _cellpose_eval_default():
    """``flow_threshold``'s default in ``CellposeModel.eval``, read from source.

    Parsed rather than imported, so the test costs no torch import.
    """
    spec = importlib.util.find_spec("cellpose")
    if spec is None or not spec.origin:
        pytest.skip("cellpose is not installed")
    source = Path(spec.origin).with_name("models.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CellposeModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "eval":
                    args = item.args.args
                    defaults = item.args.defaults
                    offset = len(args) - len(defaults)
                    for index, arg in enumerate(args[offset:]):
                        if arg.arg == "flow_threshold":
                            return ast.literal_eval(defaults[index])
    pytest.skip("CellposeModel.eval has no flow_threshold default to read")


def test_every_flow_threshold_default_is_0_4():
    """Mask, Timelapse, and the apply/test-model submodules all ship 0.4."""
    mask = set_default_settings_preprocess_generate_masks({})
    timelapse = get_timelapse_settings()
    for key in MASK_KEYS:
        assert mask[key] == 0.4, f"Mask ships {key}={mask[key]}"
        assert timelapse[key] == 0.4, f"Timelapse ships {key}={timelapse[key]}"
    assert get_default_test_cellpose_model_settings({})["FT"] == 0.4
    assert get_default_apply_cellpose_model_settings({})["FT"] == 0.4


def test_0_4_is_what_cellpose_itself_defaults_to():
    """The tooltips say "Cellpose's own default"; this holds them to it."""
    assert _cellpose_eval_default() == 0.4


def test_the_shipped_mask_defaults_raise_no_flow_warning(tmp_path):
    """The defect: the checker warned about spaCR's own default, every run."""
    settings = set_default_settings_preprocess_generate_masks({})
    settings["src"] = _raw_plate(tmp_path)
    settings["cell_channel"] = 0

    found = _flow_problems(V.validate_settings(settings, "mask"))

    assert not found, [str(p) for p in found]


@pytest.mark.parametrize("factory", [
    get_default_test_cellpose_model_settings,
    get_default_apply_cellpose_model_settings,
])
def test_the_submodule_default_is_neither_out_of_range_nor_the_wrong_type(
        factory):
    """``FT`` was declared ``int``; 0.4 must not become a type error."""
    settings = factory({})

    assert not _flow_problems(V._check_numeric_sanity(settings))
    assert not [p for p in V._check_types(settings) if p.setting == "FT"]


def test_an_ft_read_back_from_a_csv_is_a_float_again():
    """A settings CSV round trip makes 0.4 the text '0.4'."""
    restored = V.coerce_expected_types({"FT": "0.4"})

    assert restored["FT"] == 0.4 and isinstance(restored["FT"], float)
    assert V.coerce_expected_types({"FT": "100"})["FT"] == 100.0


@pytest.mark.parametrize("value", [0, 0.4, 1.0, 3])
def test_a_value_inside_0_to_3_is_not_reported(value):
    """1.0 is what spaCR shipped before 1.5.0.5; it is a choice, not an error."""
    for key in MASK_KEYS + ("FT", "flow_threshold"):
        assert not _flow_problems(V._check_numeric_sanity({key: value}))


def test_100_is_still_reported_and_the_report_is_true():
    """The warning has to survive the new default, and stop lying."""
    found = _flow_problems(V._check_numeric_sanity({"cell_flow_threshold": 100}))

    assert len(found) == 1
    problem = found[0]
    assert problem.severity == V.WARNING
    assert "cell_flow_threshold=100" in problem.message
    assert "every mask Cellpose proposes is kept" in problem.message
    assert "spaCR and Cellpose both default to 0.4" in problem.fix
    assert "1.5.0.5 to 1.5.0.8" in problem.fix
    assert "Set cell_flow_threshold to 0.4" in problem.fix
    assert "ships 1.0" not in problem.fix
    assert "values above 0 and up to 3 filter" in problem.fix, (
        "0 switches Cellpose's check off, so the fix must not offer it as a "
        "filtering value")


def test_a_negative_value_says_cellpose_skips_the_filter():
    """0 or below: Cellpose does not run the check at all."""
    found = _flow_problems(V._check_numeric_sanity({"pathogen_flow_threshold": -1}))

    assert len(found) == 1
    assert "below 0" in found[0].message
    assert "spaCR and Cellpose both default to 0.4" in found[0].fix
    assert "above 0 and at most 3" in found[0].fix


def test_a_saved_100_is_kept_not_rewritten():
    """Filling defaults never touches a value the settings already hold.

    Both spellings a saved file can carry: the current key, and the
    ``<role>_FT`` name every file written before 2026-09-02 uses.
    """
    current = set_default_settings_preprocess_generate_masks(
        {"cell_flow_threshold": 100})
    legacy = set_default_settings_preprocess_generate_masks({"pathogen_FT": 100})

    assert current["cell_flow_threshold"] == 100
    assert legacy["pathogen_flow_threshold"] == 100
    assert current["nucleus_flow_threshold"] == 0.4


def _spacr_run_validate(tmp_path, rows, capsys):
    """Run ``spacr-run validate --module mask`` on a CSV of ``rows``."""
    from spacr.cli import main

    plate = _raw_plate(tmp_path)
    path = tmp_path / "gen_mask_settings.csv"
    body = "".join(f"{key},{value}\n" for key, value in rows)
    path.write_text(f"Key,Value\nsrc,{plate}\ncell_channel,0\n{body}",
                    encoding="utf-8")
    capsys.readouterr()
    code = main(["validate", "--settings", str(path), "--module", "mask"])
    return code, capsys.readouterr().out


def test_spacr_run_validate_is_quiet_about_the_defaults(tmp_path, capsys):
    """#123 as reported: ``spacr-run`` with default settings."""
    _code, out = _spacr_run_validate(tmp_path, [], capsys)

    flagged = [line for line in out.splitlines() if "_flow_threshold" in line]
    assert not flagged, flagged


def test_spacr_run_validate_keeps_a_saved_100_and_says_why_it_warns(
        tmp_path, capsys):
    """A file saved by 1.5.0.8 keeps its 100, and is told the truth."""
    rows = [(key, 100) for key in MASK_KEYS]
    _code, out = _spacr_run_validate(tmp_path, rows, capsys)

    for key in MASK_KEYS:
        assert f"{key}=100" in out, out
    assert "spaCR and Cellpose both default to 0.4" in out
    assert "spaCR ships 1.0" not in out


def test_the_example_pack_carries_the_default_flow_threshold():
    """Decision 2026-09-25 (item 428): the example pack
    spaCR_settings/1_generate_masks_settings.csv moves nucleus_/cell_/
    pathogen_FT from 100 to 0.4, matching the default, so loading it no
    longer draws the "comes from an old saved file" warning."""
    from spacr.cli import load_settings_file

    pack = (Path(__file__).resolve().parents[1] / "spaCR_settings"
            / "1_generate_masks_settings.csv")
    settings = load_settings_file(str(pack))
    for key in ("nucleus_FT", "cell_FT", "pathogen_FT"):
        assert float(settings[key]) == 0.4, key
        assert not V._flow_threshold_problems(key, settings[key],
                                              float(settings[key]))
