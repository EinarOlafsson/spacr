"""Tests for ``spacr.cli`` — the headless ``spacr-run`` entry point.

The load-bearing test in here is :func:`test_import_pulls_no_gui_or_torch`:
``spacr.cli`` exists so a settings file can be run on a compute node with no
display, and the moment importing it drags in Qt, Tk or torch that stops being
true. Everything else checks that a settings.csv written by the GUI runs
unchanged, that a typo'd override is loud rather than silent, and that the exit
codes a batch scheduler reads are the ones documented.

Nothing here runs a pipeline: the entry points are resolved through a synthetic
module registered in ``sys.modules``, so no test imports torch or cellpose.
"""
from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from spacr import cli


REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "spacr"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cli_state():
    """Drop the CLI's log handlers and restore the env vars it sets.

    ``setup_logging`` binds a handler to whatever ``sys.stdout`` is at the
    time; left in place it would write into a closed capsys buffer during a
    later test. ``_quiet_progress_bars`` mutates ``os.environ`` directly, which
    monkeypatch cannot undo for us.
    """
    import os

    watched = ("TQDM_DISABLE", "SPACR_NO_PROGRESS", "MPLBACKEND")
    before = {k: os.environ.get(k) for k in watched}
    yield
    for handler in list(cli.LOG.handlers):
        cli.LOG.removeHandler(handler)
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Register a synthetic module + registry entry the CLI can really run.

    Exercises :func:`spacr.cli.import_entry` for real — importlib, getattr and
    all — without importing a gramme of torch.
    """
    mod = types.ModuleType("spacr_cli_fake_pipeline")
    mod.calls = []

    def run(settings):
        mod.calls.append(settings)
        return "done"

    def boom(settings):
        mod.calls.append(settings)
        raise RuntimeError("pipeline exploded")

    mod.run = run
    mod.boom = boom
    monkeypatch.setitem(sys.modules, "spacr_cli_fake_pipeline", mod)

    for key, func in (("_fake", "run"), ("_fake_boom", "boom")):
        monkeypatch.setitem(cli.MODULES, key, cli.Module(
            key=key,
            summary="test-only module",
            entry=f"spacr_cli_fake_pipeline:{func}",
            defaults=None,
            validate_key="",
            requires=("src",),
            writes=("nothing",),
        ))
    return mod


def _write_csv(path, rows, columns=("Key", "Value")):
    """Write a two-column spaCR settings CSV, exactly as save_settings does."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for key, value in rows:
            writer.writerow([key, value])
    return str(path)


@pytest.fixture
def fake_settings(tmp_path):
    """A minimal settings CSV whose src exists, so pre-flight stays clean."""
    src = tmp_path / "plate"
    src.mkdir()
    return _write_csv(tmp_path / "s.csv", [("src", str(src)), ("verbose", True)])


# ---------------------------------------------------------------------------
# the point of the module: importing it must stay cheap
# ---------------------------------------------------------------------------


def _subprocess_modules(code: str) -> dict:
    """Run ``code`` in a fresh interpreter; it must print a JSON dict."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT),
             "HOME": "/tmp", "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_HEAVY = ("PySide6", "PyQt5", "PyQt6", "tkinter", "torch", "cellpose")

_PROBE = (
    "import json, sys\n"
    "{body}\n"
    "print(json.dumps({{m: (m in sys.modules) for m in %r}}))\n" % (_HEAVY,)
)


def test_import_pulls_no_gui_or_torch():
    """`import spacr.cli` must not pull Qt, Tk or torch — that is the feature.

    A compute node has no display; if importing the CLI imports PySide6 the
    whole headless path collapses at the first line. Checked in a *fresh*
    interpreter so nothing another test imported can mask a regression.
    """
    loaded = _subprocess_modules(_PROBE.format(body="import spacr.cli"))
    offenders = [m for m, present in loaded.items() if present]
    assert not offenders, f"spacr.cli imported heavy modules: {offenders}"


def test_help_pulls_no_gui_or_torch():
    """`spacr-run --help` answers from the standard library alone."""
    body = (
        "import io, contextlib, spacr.cli\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    rc = spacr.cli.main(['--help'])\n"
        "assert rc == 0, rc\n"
        "assert 'spacr-run' in buf.getvalue()\n"
    )
    loaded = _subprocess_modules(_PROBE.format(body=body))
    offenders = [m for m, present in loaded.items() if present]
    assert not offenders, f"--help imported heavy modules: {offenders}"


def test_list_pulls_no_gui_or_torch():
    """`--list` is a login-node command; it must not warm up a GPU stack."""
    body = (
        "import io, contextlib, spacr.cli\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    rc = spacr.cli.main(['--list'])\n"
        "assert rc == 0, rc\n"
        "assert 'measure' in buf.getvalue()\n"
    )
    loaded = _subprocess_modules(_PROBE.format(body=body))
    offenders = [m for m, present in loaded.items() if present]
    assert not offenders, f"--list imported heavy modules: {offenders}"


# ---------------------------------------------------------------------------
# the registry is real
# ---------------------------------------------------------------------------


def _top_level_defs(path: Path) -> set:
    """Names defined at the top level of a python file, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize("key", sorted(cli.MODULES))
def test_registered_entry_point_exists_on_disk(key):
    """Every module maps to a function that is really defined in spaCR.

    Checked with ast rather than an import so the whole registry is verified
    without loading torch — and so a renamed pipeline function fails here
    instead of at 3am on a compute node.
    """
    module = cli.MODULES[key]
    assert module.module_name.startswith("spacr."), module.module_name
    path = PKG_ROOT / (module.module_name.split(".", 1)[1].replace(".", "/") + ".py")
    assert path.is_file(), f"{key}: no such file {path}"
    assert module.func_name in _top_level_defs(path), \
        f"{key}: {path.name} does not define {module.func_name}"


@pytest.mark.parametrize("key", sorted(cli.MODULES))
def test_registered_defaults_helper_exists(key):
    """Each module's defaults helper is a real callable in spacr.settings."""
    module = cli.MODULES[key]
    if module.defaults is None:
        return
    from spacr import settings as spacr_settings
    assert callable(getattr(spacr_settings, module.defaults, None)), \
        f"{key}: spacr.settings has no {module.defaults}"


@pytest.mark.parametrize("key", sorted(cli.MODULES))
def test_registered_validate_key_is_known(key):
    """A module's pre-flight key is one spacr.validate actually has rules for."""
    from spacr.validate import APP_FUNCTIONS
    module = cli.MODULES[key]
    assert module.validate_key == "" or module.validate_key in APP_FUNCTIONS


def test_gui_only_modules_are_not_runnable():
    """The interactive apps must not sneak into the runnable registry."""
    assert not (set(cli.INTERACTIVE_ONLY) & set(cli.MODULES))


@pytest.mark.parametrize("alias,expected", sorted(cli.ALIASES.items()))
def test_aliases_resolve_to_a_real_module(alias, expected):
    """Every alias points at a registered module."""
    resolved = cli.resolve_module(alias)
    assert resolved is not None and resolved.key == expected


@pytest.mark.parametrize("name", ["MASK", "Measure", "map-barcodes", " mask "])
def test_module_names_are_forgiving(name):
    """Case, spaces and dashes all resolve — cluster scripts are copy-pasted."""
    assert cli.resolve_module(name) is not None


def test_resolve_module_rejects_non_strings():
    assert cli.resolve_module(None) is None
    assert cli.resolve_module(17) is None


# ---------------------------------------------------------------------------
# --list / --describe
# ---------------------------------------------------------------------------


def test_list_names_every_module(capsys):
    rc = cli.main(["--list"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    for key, module in cli.MODULES.items():
        if key.startswith("_"):
            continue
        assert key in out
        assert f"{module.module_name}.{module.func_name}()" in out
    for key in cli.INTERACTIVE_ONLY:
        assert key in out


@pytest.mark.parametrize("key", [k for k in cli.MODULES if not k.startswith("_")])
def test_describe_each_module(capsys, key):
    """--describe works for every registered module and names its callable."""
    rc = cli.main(["--describe", key])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    module = cli.MODULES[key]
    assert module.summary in out
    assert f"{module.module_name}.{module.func_name}" in out
    for requirement in module.requires:
        assert requirement in out
    for written in module.writes:
        assert written in out


def test_describe_reports_settings_count(capsys):
    cli.main(["--describe", "measure"])
    out = capsys.readouterr().out
    assert "spacr.settings.get_measure_crop_settings()" in out
    assert "keys" in out


def test_describe_module_without_defaults(capsys):
    """The simulator has no set_default_* helper; --describe says so."""
    cli.main(["--describe", "simulation"])
    out = capsys.readouterr().out
    assert "none — every setting must come from the settings file" in out


def test_describe_unknown_module_exits_2(capsys):
    rc = cli.main(["--describe", "meesure"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "meesure" in err
    assert "measure" in err  # the did-you-mean


def test_describe_gui_only_module_explains(capsys):
    """'annotate' is not a typo — say why it cannot run headless."""
    rc = cli.main(["--describe", "annotate"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "GUI-only" in err
    assert "spacr-qt" in err


def test_version(capsys):
    from spacr.version import __version__
    rc = cli.main(["--version"])
    assert rc == cli.EXIT_OK
    assert __version__ in capsys.readouterr().out


def test_no_module_exits_2(capsys):
    rc = cli.main([])
    assert rc == cli.EXIT_USAGE
    assert "no module given" in capsys.readouterr().err


def test_argparse_usage_error_exits_2(capsys):
    """A bad flag exits 2, not argparse's own SystemExit escaping main()."""
    rc = cli.main(["--not-a-flag"])
    assert rc == cli.EXIT_USAGE


# ---------------------------------------------------------------------------
# settings file round trip
# ---------------------------------------------------------------------------


ROUND_TRIP = {
    "src": "/data/plate01",
    "experiment": "exp1",
    "magnification": 20,
    "cell_CP_prob": -1.5,
    "preprocess": True,
    "masks": False,
    "channels": [0, 1, 2, 3],
    "normalize": [1, 99],
    "png_size": [[224, 224]],
    "cell_channel": None,
    "plate_dict": {"EO1": "plate1", "EO2": "plate2"},
}


def test_settings_csv_round_trip(tmp_path):
    """A settings.csv written the way the GUI writes it loads back identical.

    This is the whole feature: click through the GUI on a laptop, copy
    <src>/settings/*.csv to the cluster, run it unchanged.
    """
    path = _write_csv(tmp_path / "gen_mask_settings.csv", ROUND_TRIP.items())
    assert cli.load_settings_file(path) == ROUND_TRIP


def test_settings_csv_round_trip_setting_key_columns(tmp_path):
    """The other column spelling load_settings documents also works."""
    path = _write_csv(tmp_path / "s.csv", ROUND_TRIP.items(),
                      columns=("setting_key", "setting_value"))
    assert cli.load_settings_file(path) == ROUND_TRIP


def test_settings_json_round_trip(tmp_path):
    """A run journal's settings.json is a valid input too."""
    path = tmp_path / "settings.json"
    payload = {k: v for k, v in ROUND_TRIP.items() if k != "cell_channel"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cli.load_settings_file(str(path)) == payload


def test_settings_csv_empty_cell_is_none(tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", "/data"), ("custom_regex", "")])
    assert cli.load_settings_file(path)["custom_regex"] is None


def test_settings_csv_unquoted_list_is_rejoined(tmp_path):
    """A hand-edited `channels,[0, 1, 2]` must not load as the fragment '[0'."""
    path = tmp_path / "s.csv"
    path.write_text("Key,Value\nsrc,/data\nchannels,[0, 1, 2]\n", encoding="utf-8")
    assert cli.load_settings_file(str(path))["channels"] == [0, 1, 2]


def test_settings_csv_version_like_string_stays_a_string(tmp_path):
    """'1.5.2' has a dot but is not a float — it stays text, as load_settings does."""
    path = _write_csv(tmp_path / "s.csv", [("src", "/data"), ("experiment", "1.5.2")])
    assert cli.load_settings_file(path)["experiment"] == "1.5.2"


def test_settings_file_missing_exits_2(capsys, tmp_path):
    """A path typo on a cluster must produce a sentence, not a traceback."""
    rc = cli.main(["measure", "--settings", str(tmp_path / "nope.csv")])
    captured = capsys.readouterr()
    assert rc == cli.EXIT_USAGE
    assert "settings file not found" in captured.err
    assert "Traceback" not in captured.err


def test_settings_file_is_a_directory_exits_2(capsys, tmp_path):
    rc = cli.main(["measure", "--settings", str(tmp_path)])
    assert rc == cli.EXIT_USAGE
    assert "is a folder" in capsys.readouterr().err


def test_settings_file_wrong_columns_exits_2(capsys, tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    rc = cli.main(["measure", "--settings", str(path)])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "not a spaCR settings CSV" in err


def test_settings_json_not_an_object_exits_2(capsys, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    rc = cli.main(["measure", "--settings", str(path)])
    assert rc == cli.EXIT_USAGE
    assert "not a settings object" in capsys.readouterr().err


def test_settings_json_malformed_exits_2(capsys, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{oops", encoding="utf-8")
    rc = cli.main(["measure", "--settings", str(path)])
    assert rc == cli.EXIT_USAGE
    assert "could not parse" in capsys.readouterr().err


def test_load_settings_file_without_a_path():
    with pytest.raises(cli.SettingsError):
        cli.load_settings_file("")


def test_file_values_beat_defaults(tmp_path):
    """The settings file wins over the module defaults it does not mention."""
    path = _write_csv(tmp_path / "s.csv", [("src", "/data"), ("cell_mask_dim", 6)])
    resolved = cli.resolve_settings(cli.MODULES["measure"], path)
    assert resolved["cell_mask_dim"] == 6
    assert "nucleus_mask_dim" in resolved  # untouched default still present


# ---------------------------------------------------------------------------
# --set overrides and type coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,text,expected", [
    # bool, both spellings people actually type
    ("preprocess", "false", False),
    ("preprocess", "False", False),
    ("preprocess", "0", False),
    ("preprocess", "true", True),
    ("preprocess", "yes", True),
    ("preprocess", "1", True),
    # int
    ("magnification", "40", 40),
    ("magnification", "40.0", 40),
    ("cell_min_size", "-1", -1),
    # float — declared float, so a bare int becomes one
    ("nucleus_Signal_to_noise", "10", 10.0),
    ("nucleus_Signal_to_noise", "3.5", 3.5),
    # list, as a literal and as the bare comma form
    ("channels", "[0, 1, 2]", [0, 1, 2]),
    ("channels", "0,1,2", [0, 1, 2]),
    ("channels", "0", [0]),
    ("crop_mode", "cell,nucleus", ["cell", "nucleus"]),
    # str — '123' for a str setting stays text
    ("experiment", "screen_A", "screen_A"),
    ("experiment", "123", "123"),
    # optional int
    ("cell_channel", "none", None),
    ("cell_channel", "None", None),
    ("cell_channel", "2", 2),
    # optional str
    ("custom_regex", "null", None),
    ("custom_regex", "W(?P<wellID>.*)", "W(?P<wellID>.*)"),
    # (str, list) — src is both
    ("src", "/data/plate01", "/data/plate01"),
    ("src", "['/a', '/b']", ["/a", "/b"]),
    # (bool, list) — normalize is declared bool but measure needs percentiles
    ("normalize", "false", False),
    ("normalize", "[1, 99]", [1, 99]),
    # dict and tuple
    ("plate_dict", "{'EO1': 'plate1'}", {"EO1": "plate1"}),
    ("motility_xlim", "(0, 10)", (0, 10)),
])
def test_set_coercion_follows_expected_types(key, text, expected):
    """--set values are coerced with spacr.settings.expected_types, not stored raw.

    A settings CSV round trip already turns every value into a string once;
    letting --set do it again is how `cell_mask_dim='4'` reaches measure_crop.
    """
    from spacr.settings import expected_types
    result = cli.coerce_value(key, text, None, expected_types)
    assert result == expected
    assert type(result) is type(expected)


def test_set_coercion_infers_type_from_the_current_value():
    """Keys expected_types does not declare take the type they already hold."""
    assert cli.coerce_value("undeclared_flag", "true", False, {}) is True
    assert cli.coerce_value("undeclared_count", "7", 3, {}) == 7
    assert cli.coerce_value("undeclared_ratio", "2", 1.5, {}) == 2.0
    assert cli.coerce_value("undeclared_list", "a,b", ["x"], {}) == ["a", "b"]
    assert cli.coerce_value("undeclared_name", "abc", "old", {}) == "abc"


def test_set_coercion_with_no_type_information_parses_literally():
    """With neither a declaration nor a prior value, fall back to CSV parsing."""
    assert cli.coerce_value("brand_new", "12", None, {}) == 12
    assert cli.coerce_value("brand_new", "[1, 2]", None, {}) == [1, 2]
    assert cli.coerce_value("brand_new", "hello", None, {}) == "hello"


def test_set_applies_after_the_file(tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", "/from_file"), ("cell_mask_dim", 4)])
    resolved = cli.resolve_settings(cli.MODULES["measure"], path,
                                    ["src=/from_cli", "cell_mask_dim=6"])
    assert resolved["src"] == "/from_cli"
    assert resolved["cell_mask_dim"] == 6


def test_set_unknown_key_exits_2_and_names_it(capsys, fake_settings):
    """A typo'd override is an error: silently doing nothing is worse."""
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run",
                   "--set", "cell_min_sze=200"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "cell_min_sze" in err
    assert "cell_min_size" in err  # the did-you-mean


def test_set_unknown_key_without_a_close_match(capsys, fake_settings):
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run",
                   "--set", "zzzqqq=1"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "zzzqqq" in err
    assert "does not exist" in err


def test_set_without_equals_exits_2(capsys, fake_settings):
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run",
                   "--set", "cell_mask_dim"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "key=value" in err


def test_set_with_empty_key_exits_2(capsys, fake_settings):
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run", "--set", "=4"])
    assert rc == cli.EXIT_USAGE
    assert "empty key" in capsys.readouterr().err


def test_set_uncoercible_value_exits_2(capsys, fake_settings):
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run",
                   "--set", "cell_mask_dim=abc"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "cell_mask_dim" in err
    assert "int" in err


def test_set_malformed_list_exits_2(capsys, fake_settings):
    rc = cli.main(["measure", "--settings", fake_settings, "--dry-run",
                   "--set", "channels=[0, 1"])
    assert rc == cli.EXIT_USAGE
    assert "not a valid list" in capsys.readouterr().err


def test_set_malformed_dict_exits_2():
    from spacr.settings import expected_types
    with pytest.raises(cli.SettingsError, match="not a valid dict"):
        cli.coerce_value("plate_dict", "{'a': ", None, expected_types)


def test_apply_overrides_is_a_noop_without_overrides():
    settings = {"src": "/data"}
    assert cli.apply_overrides(settings, []) is settings


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_executes_nothing(capsys, fake_pipeline, fake_settings):
    """--dry-run prints the plan and never calls the pipeline."""
    rc = cli.main(["_fake", "--settings", fake_settings, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert fake_pipeline.calls == []
    assert "Resolved settings:" in out
    assert "Plan — what this run would do" in out
    assert "--dry-run: nothing was executed." in out
    assert "spacr_cli_fake_pipeline.run() was not called" in out


def test_dry_run_prints_the_resolved_settings(capsys, fake_pipeline, tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    path = _write_csv(tmp_path / "s.csv", [("src", str(src)), ("verbose", False)])
    rc = cli.main(["_fake", "--settings", path, "--dry-run", "--set", "verbose=true"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "verbose" in out
    assert str(src) in out


def test_dry_run_with_settings_errors_exits_2(capsys, fake_pipeline, tmp_path):
    """A dry run that finds errors fails the job — that is what it is for."""
    path = _write_csv(tmp_path / "s.csv", [("src", str(tmp_path / "gone"))])
    rc = cli.main(["_fake", "--settings", path, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_USAGE
    assert "does not exist" in out
    assert fake_pipeline.calls == []


def test_dry_run_on_a_real_module_does_not_import_it(capsys, tmp_path):
    """--dry-run on `measure` must not import spacr.measure (and so torch)."""
    src = tmp_path / "plate"
    (src / "merged").mkdir(parents=True)
    path = _write_csv(tmp_path / "s.csv", [("src", str(src))])
    sys.modules.pop("spacr.measure", None)
    cli.main(["measure", "--settings", path, "--dry-run"])
    assert "spacr.measure" not in sys.modules


# ---------------------------------------------------------------------------
# running for real (against the synthetic pipeline)
# ---------------------------------------------------------------------------


def test_run_calls_the_entry_point_with_resolved_settings(capsys, fake_pipeline,
                                                          fake_settings):
    rc = cli.main(["_fake", "--settings", fake_settings, "--set", "verbose=false"])
    assert rc == cli.EXIT_OK
    assert len(fake_pipeline.calls) == 1
    assert fake_pipeline.calls[0]["verbose"] is False
    assert "finished in" in capsys.readouterr().out


def test_run_that_raises_exits_1(capsys, fake_pipeline, fake_settings):
    """A pipeline exception is exit 1 — a batch job that exits 0 after failing
    is the classic headless footgun."""
    rc = cli.main(["_fake_boom", "--settings", fake_settings])
    captured = capsys.readouterr()
    assert rc == cli.EXIT_RUNTIME
    assert "pipeline exploded" in captured.out + captured.err


def test_run_verbose_logs_the_settings(capsys, fake_pipeline, fake_settings):
    rc = cli.main(["_fake", "--settings", fake_settings, "--verbose"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "resolved settings:" in out
    assert "DEBUG" in out


def test_run_logs_are_timestamped(capsys, fake_pipeline, fake_settings):
    """Cluster logs are read weeks later; every line carries a timestamp."""
    import re
    cli.main(["_fake", "--settings", fake_settings])
    out = capsys.readouterr().out
    stamped = [line for line in out.splitlines()
               if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", line)]
    assert stamped, out


def test_run_refuses_to_start_when_preflight_finds_errors(capsys, fake_pipeline,
                                                          tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", str(tmp_path / "gone"))])
    rc = cli.main(["_fake", "--settings", path])
    assert rc == cli.EXIT_USAGE
    assert fake_pipeline.calls == []
    assert "refusing to start" in capsys.readouterr().out


def test_force_runs_despite_preflight_errors(capsys, fake_pipeline, tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", str(tmp_path / "gone"))])
    rc = cli.main(["_fake", "--settings", path, "--force"])
    assert rc == cli.EXIT_OK
    assert len(fake_pipeline.calls) == 1
    assert "running anyway" in capsys.readouterr().out


def test_no_preflight_skips_the_check(capsys, fake_pipeline, tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", str(tmp_path / "gone"))])
    rc = cli.main(["_fake", "--settings", path, "--no-preflight"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "spaCR pre-flight check" not in out


def test_run_without_a_settings_file_exits_2(capsys):
    rc = cli.main(["measure"])
    assert rc == cli.EXIT_USAGE
    assert "no settings file given" in capsys.readouterr().err


def test_run_neutralises_plt_show(fake_pipeline, fake_settings, monkeypatch):
    """plt.show() inside a pipeline must not warn, block or leak the figure."""
    plt = pytest.importorskip("matplotlib.pyplot")
    seen = {}

    def _show_a_figure(settings):
        fig = plt.figure()
        seen["before"] = len(plt.get_fignums())
        plt.show()
        seen["after"] = len(plt.get_fignums())

    monkeypatch.setattr(sys.modules["spacr_cli_fake_pipeline"], "run", _show_a_figure)
    original_show = plt.show
    rc = cli.main(["_fake", "--settings", fake_settings])
    assert rc == cli.EXIT_OK
    assert seen["after"] == seen["before"] - 1  # the figure was closed, not shown
    assert plt.show is original_show           # and show() was put back


def test_run_restores_plt_show_after_a_failure(fake_pipeline, fake_settings):
    plt = pytest.importorskip("matplotlib.pyplot")
    original_show = plt.show
    cli.main(["_fake_boom", "--settings", fake_settings])
    assert plt.show is original_show


def test_import_entry_reports_a_missing_callable(monkeypatch):
    mod = types.ModuleType("spacr_cli_empty_module")
    monkeypatch.setitem(sys.modules, "spacr_cli_empty_module", mod)
    module = cli.Module(key="_x", summary="", entry="spacr_cli_empty_module:missing",
                        defaults=None, validate_key="")
    with pytest.raises(cli.SettingsError, match="has no callable"):
        cli.import_entry(module)


def test_import_entry_reports_a_missing_module():
    module = cli.Module(key="_x", summary="", entry="spacr_cli_no_such_module:run",
                        defaults=None, validate_key="")
    with pytest.raises(cli.SettingsError, match="could not import"):
        cli.import_entry(module)


def test_import_entry_failure_exits_2(capsys, monkeypatch, fake_settings):
    monkeypatch.setitem(cli.MODULES, "_broken", cli.Module(
        key="_broken", summary="", entry="spacr_cli_no_such_module:run",
        defaults=None, validate_key=""))
    rc = cli.main(["_broken", "--settings", fake_settings])
    assert rc == cli.EXIT_USAGE
    assert "could not import" in capsys.readouterr().err


def test_folder_call_style_passes_src_positionally(monkeypatch, tmp_path):
    """`convert` takes a bare folder, not a settings dict."""
    got = []
    module = cli.MODULES["convert"]
    cli._call_entry(module, lambda folder: got.append(folder), {"src": "/data"})
    assert got == ["/data"]


def test_folder_call_style_rejects_a_list_src():
    module = cli.MODULES["convert"]
    with pytest.raises(cli.SettingsError, match="single folder"):
        cli._call_entry(module, lambda folder: None, {"src": ["/a", "/b"]})


def test_run_that_calls_sys_exit_propagates_the_code(fake_pipeline, fake_settings,
                                                     monkeypatch):
    def _bail(settings):
        raise SystemExit(3)

    monkeypatch.setattr(sys.modules["spacr_cli_fake_pipeline"], "run", _bail)
    assert cli.main(["_fake", "--settings", fake_settings]) == 3


def test_keyboard_interrupt_exits_1(fake_pipeline, fake_settings, monkeypatch):
    def _interrupt(settings):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules["spacr_cli_fake_pipeline"], "run", _interrupt)
    assert cli.main(["_fake", "--settings", fake_settings]) == cli.EXIT_RUNTIME


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------


def test_validate_reports_clean_settings(capsys, tmp_path):
    src = tmp_path / "plate"
    (src / "merged").mkdir(parents=True)
    import numpy as np
    np.save(src / "merged" / "f1.npy", np.zeros((8, 8, 7), dtype=np.uint16))
    path = _write_csv(tmp_path / "s.csv", [
        ("src", str(src)), ("cell_mask_dim", 4), ("nucleus_mask_dim", 5),
        ("pathogen_mask_dim", 6), ("crop_mode", "['cell']"),
        ("normalize", "[1, 99]"), ("normalize_by", "png")])
    rc = cli.main(["validate", "--settings", path, "--module", "measure"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "validate: settings are runnable." in out


def test_validate_flags_bad_settings_and_exits_2(capsys, tmp_path):
    path = _write_csv(tmp_path / "s.csv", [("src", str(tmp_path / "gone"))])
    rc = cli.main(["validate", "--settings", path, "--module", "mask"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_USAGE
    assert "would not run" in out


def test_validate_without_a_module_runs_generic_checks(capsys, tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    path = _write_csv(tmp_path / "s.csv", [("src", str(src))])
    rc = cli.main(["validate", "--settings", path])
    assert rc == cli.EXIT_OK
    assert "spaCR pre-flight check" in capsys.readouterr().out


def test_validate_without_settings_exits_2(capsys):
    rc = cli.main(["validate"])
    assert rc == cli.EXIT_USAGE
    assert "needs a settings file" in capsys.readouterr().err


def test_validate_unknown_module_exits_2(capsys, fake_settings):
    rc = cli.main(["validate", "--settings", fake_settings, "--module", "nonsense"])
    assert rc == cli.EXIT_USAGE
    assert "unknown module" in capsys.readouterr().err


def test_validate_missing_file_exits_2(capsys, tmp_path):
    rc = cli.main(["validate", "--settings", str(tmp_path / "nope.csv")])
    assert rc == cli.EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_validate_rejects_a_bad_override(capsys, fake_settings):
    rc = cli.main(["validate", "--settings", fake_settings, "--set", "nope_key=1"])
    assert rc == cli.EXIT_USAGE
    assert "nope_key" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# headless environment handling
# ---------------------------------------------------------------------------


def test_agg_is_forced_when_there_is_no_display(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli.use_agg_if_headless() is True
    import os
    assert os.environ["MPLBACKEND"] == "Agg"
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_agg_is_not_forced_when_a_display_exists(monkeypatch):
    """Interactive local use must be left alone."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli.use_agg_if_headless() is False


def test_agg_is_not_forced_on_windows_or_macos(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        assert cli.use_agg_if_headless() is False


def test_an_explicit_backend_choice_wins(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "pdf")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli.use_agg_if_headless() is False


def test_progress_bars_are_disabled_off_a_tty(monkeypatch):
    import os
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.delenv("SPACR_NO_PROGRESS", raising=False)
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: False))
    assert cli._quiet_progress_bars() is True
    assert os.environ["TQDM_DISABLE"] == "1"


def test_progress_bars_are_left_alone_on_a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: True))
    assert cli._quiet_progress_bars() is False


def test_progress_bars_survive_a_stdout_without_isatty(monkeypatch):
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace())
    assert cli._quiet_progress_bars() is True


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def test_render_settings_aligns_and_sorts():
    text = cli.render_settings({"zeta": 1, "a": "x"})
    lines = text.splitlines()
    assert lines[0].strip().startswith("a")
    assert lines[1].strip().startswith("zeta")


def test_render_settings_handles_an_empty_dict():
    assert "no settings" in cli.render_settings({})


def test_render_settings_clips_long_values():
    text = cli.render_settings({"src": "/" + "x" * 500})
    assert max(len(line) for line in text.splitlines()) < 120


def test_module_defaults_for_a_module_without_a_helper():
    assert cli.module_defaults(cli.MODULES["simulation"]) == {}


def test_module_defaults_survive_a_renamed_helper():
    module = cli.Module(key="_x", summary="", entry="spacr.core:preprocess_generate_masks",
                        defaults="no_such_helper", validate_key="")
    assert cli.module_defaults(module) == {}


def test_module_defaults_accepts_a_zero_argument_helper():
    """Not every spacr.settings helper takes a dict — some build their own."""
    module = cli.Module(key="_x", summary="", entry="spacr.core:preprocess_generate_masks",
                        defaults="set_default_plot_merge_settings", validate_key="")
    defaults = cli.module_defaults(module)
    assert defaults and "cell_mask_dim" in defaults


def test_module_defaults_reject_a_non_dict_helper(monkeypatch):
    from spacr import settings as spacr_settings
    monkeypatch.setattr(spacr_settings, "_cli_probe", lambda *a: "not a dict",
                        raising=False)
    module = cli.Module(key="_x", summary="", entry="spacr.core:preprocess_generate_masks",
                        defaults="_cli_probe", validate_key="")
    assert cli.module_defaults(module) == {}


def test_describe_survives_a_broken_defaults_helper(capsys, monkeypatch):
    """A settings helper that cannot be called must not take --describe down.

    `check_settings` needs three arguments, so both the fn({}) and the bare
    fn() call in module_defaults raise — the real shape of this failure.
    """
    monkeypatch.setitem(cli.MODULES, "_broken_defaults", cli.Module(
        key="_broken_defaults", summary="test-only",
        entry="spacr.core:preprocess_generate_masks",
        defaults="check_settings", validate_key=""))
    rc = cli.main(["--describe", "_broken_defaults"])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "spacr.settings.check_settings()" in out
    assert "keys, all optional" not in out


def test_exit_codes_are_the_documented_ones():
    assert (cli.EXIT_OK, cli.EXIT_RUNTIME, cli.EXIT_USAGE) == (0, 1, 2)


def test_python_dash_m_spacr_cli_works(monkeypatch, capsys):
    """`python -m spacr.cli` is the fallback when the console script is not on PATH."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["spacr-run", "--list"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("spacr.cli", run_name="__main__")
    assert exc.value.code == cli.EXIT_OK
    assert "measure" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# defensive paths
# ---------------------------------------------------------------------------


def test_csv_dict_values_are_parsed_recursively(tmp_path):
    """A dict cell is literal-eval'd, then each of its values re-parsed."""
    path = _write_csv(tmp_path / "s.csv", [("plate_dict", "{'a': 1, 'b': 'plate2'}")])
    assert cli.load_settings_file(path)["plate_dict"] == {"a": 1, "b": "plate2"}


def test_csv_row_with_a_blank_key_is_skipped(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("Key,Value\nsrc,/data\n,\n   ,7\n", encoding="utf-8")
    assert cli.load_settings_file(str(path)) == {"src": "/data"}


def test_csv_broken_literal_is_kept_as_text(tmp_path):
    """`[0, 1` cannot be parsed; keep the text rather than guessing."""
    path = _write_csv(tmp_path / "s.csv", [("channels", "[0, 1")])
    assert cli.load_settings_file(path)["channels"] == "[0, 1"


def test_undecodable_settings_file_exits_2(capsys, tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"Key,Value\nsrc,\xff\xfe\x00binary\n")
    rc = cli.main(["measure", "--settings", str(path)])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "could not read" in err
    assert "Traceback" not in err


def test_coercion_infers_dict_from_the_current_value():
    assert cli.coerce_value("undeclared_map", "{'a': 1}", {"b": 2}, {}) == {"a": 1}


def test_comma_split_list_items_keep_their_types():
    """Each token of a bare comma list is parsed, not left as text."""
    from spacr.settings import expected_types
    assert cli.coerce_value("channels", "none,true,false,'x',2,1.5", None,
                            expected_types) == [None, True, False, "x", 2, 1.5]


def test_type_label_for_an_undeclared_setting():
    assert cli._type_label(()) == "any value"


def test_agg_is_skipped_when_matplotlib_is_unavailable(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert cli.use_agg_if_headless() is False


def test_noshow_is_inert_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with cli._NoShow() as guard:
        assert guard._plt is None


def test_noshow_survives_a_matplotlib_that_raises(monkeypatch):
    """Restoring plt.show must never be the thing that fails a completed run."""
    plt = pytest.importorskip("matplotlib.pyplot")

    def _boom(*args, **kwargs):
        raise RuntimeError("backend gone")

    with cli._NoShow():
        monkeypatch.setattr(plt, "close", _boom)
        plt.show()  # the shim swallows the failure


def test_setup_logging_replaces_its_handler():
    """Called twice, the CLI logger keeps exactly one handler, not two."""
    cli.setup_logging(False)
    cli.setup_logging(True)
    assert len(cli.LOG.handlers) == 1


def test_run_with_an_unknown_module_exits_2(capsys, fake_settings):
    rc = cli.main(["meesure", "--settings", fake_settings])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE
    assert "meesure" in err


def test_run_reports_a_forced_agg_backend(capsys, monkeypatch, fake_pipeline,
                                          fake_settings):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    rc = cli.main(["_fake", "--settings", fake_settings])
    assert rc == cli.EXIT_OK
    assert "backend forced to Agg" in capsys.readouterr().out


def test_bad_call_convention_exits_2(capsys, monkeypatch, tmp_path):
    """A folder-style entry point given a list of sources fails as a usage error."""
    mod = types.ModuleType("spacr_cli_folder_pipeline")
    mod.calls = []
    mod.run = lambda folder: mod.calls.append(folder)
    monkeypatch.setitem(sys.modules, "spacr_cli_folder_pipeline", mod)
    monkeypatch.setitem(cli.MODULES, "_folder", cli.Module(
        key="_folder", summary="test-only", entry="spacr_cli_folder_pipeline:run",
        defaults=None, validate_key="", call_style="folder"))
    path = _write_csv(tmp_path / "s.csv", [("src", "['/a', '/b']")])
    rc = cli.main(["_folder", "--settings", path, "--no-preflight"])
    assert rc == cli.EXIT_USAGE
    assert "single folder" in capsys.readouterr().err
    assert mod.calls == []
