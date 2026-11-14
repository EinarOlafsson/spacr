"""Table-driven tests of spaCR's command-line front doors.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: how ``spacr.cli.coerce_value`` reads a ``--set`` text
(paths, blanks, retired booleans, shapes that parse into the wrong kind of
object); what ``spacr.cli.main`` hands its pre-flight and its pipeline (the
``--hash-inputs`` flags, the analysis-lock verdict's log level, progress bars
on a terminal and off one); which tab ``python -m spacr`` opens for each
subcommand; and how ``spacr repro`` resolves a run named by its basename.
Pipelines are synthetic modules registered through ``sys.modules`` and the run
journal lives under ``tmp_path``, so nothing heavy is imported or written to
the user's home.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types
from contextlib import redirect_stderr, redirect_stdout

import pytest

from spacr import __main__ as M
from spacr import cli, cli_repro
from spacr.settings import expected_types


# ---------------------------------------------------------------------------
# coerce_value
# ---------------------------------------------------------------------------

_RAISES = object()

COERCE_CASES = [
    # case id, key, text, current value, declared types, app, expected
    ("checkpoint path survives", "custom_model", "/models/my_cellpose.pth",
     None, expected_types, "mask", "/models/my_cellpose.pth"),
    ("none means the stock model", "custom_model", "none", None,
     expected_types, "mask", None),
    ("retired True is refused", "custom_model", "True", None, expected_types,
     "mask", (_RAISES, "--set custom_model is a checkpoint path, not a boolean "
              "switch; use custom_model=none for the stock Cellpose model.")),
    ("retired False is refused", "custom_model", "False", None,
     expected_types, "mask",
     (_RAISES, "--set custom_model is a checkpoint path, not a boolean "
      "switch; use custom_model=none for the stock Cellpose model.")),
    ("blank src stays blank", "src", "", "path", {"src": (str, list)}, "", ""),
    ("spaces-only src stays as typed", "src", "   ", "path",
     {"src": (str, list)}, "", "   "),
    ("non-type declaration falls back to current int", "thing", "7", 5,
     {"thing": "int"}, "", 7),
    ("non-type declaration refuses non-int", "thing", "abc", 5,
     {"thing": "int"}, "",
     (_RAISES, "--set thing='abc' cannot be read as int.\n"
      "  thing expects int; the current value is 5.")),
    ("a real type declaration beats the current value", "thing", "abc", 5,
     {"thing": str}, "", "abc"),
    ("braces parsing to a set stay text", "k", "{1, 2}", None, {}, "",
     "{1, 2}"),
    ("braces parsing to a dict are a dict", "k", "{'a': 1}", None, {}, "",
     {"a": 1}),
    ("brackets around a scalar stay text", "k", "(1)", None, {}, "", "(1)"),
    ("brackets with a comma are a list", "k", "(1, 2)", None, {}, "", [1, 2]),
]


def test_coerce_value_reads_set_text_into_the_declared_kind(subtests):
    """``--set`` text becomes exactly the value (or refusal) each case names."""
    for case_id, key, text, current, declared, app, expected in COERCE_CASES:
        with subtests.test(msg=case_id):
            if isinstance(expected, tuple) and expected[0] is _RAISES:
                with pytest.raises(cli.SettingsError) as excinfo:
                    cli.coerce_value(key, text, current, declared, app)
                assert str(excinfo.value) == expected[1]
            else:
                got = cli.coerce_value(key, text, current, declared, app)
                assert got == expected
                assert type(got) is type(expected)


# ---------------------------------------------------------------------------
# cli.main
# ---------------------------------------------------------------------------

class _TtyOut(io.StringIO):
    """A writable stdout that reports itself as a terminal."""

    def isatty(self):
        """Yes -- which is what the CLI branches on."""
        return True


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """A synthetic runnable module, a clean settings file and a private journal.

    Restores the CLI logger and the progress-bar environment variables after
    the test.
    """
    from spacr import run_journal

    journal_root = tmp_path / "runs"
    journal_root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: journal_root)
    mod = types.ModuleType("spacr_cli_dispatch_fake")
    mod.calls = []
    mod.run = mod.calls.append
    monkeypatch.setitem(sys.modules, "spacr_cli_dispatch_fake", mod)
    monkeypatch.setitem(cli.MODULES, "_dispatch_fake", cli.Module(
        key="_dispatch_fake", summary="test-only module",
        entry="spacr_cli_dispatch_fake:run", defaults=None,
        validate_key="", requires=("src",), writes=("nothing",)))
    plate = tmp_path / "plate"
    plate.mkdir()
    settings = tmp_path / "s.csv"
    settings.write_text("Key,Value\nsrc,%s\nverbose,False\n" % plate,
                        encoding="utf-8")
    watched = ("TQDM_DISABLE", "SPACR_NO_PROGRESS")
    before = {k: os.environ.get(k) for k in watched}
    propagate, level = cli.LOG.propagate, cli.LOG.level
    yield mod, str(settings)
    _reset_cli_log()
    cli.LOG.propagate = propagate
    cli.LOG.setLevel(level)
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _reset_cli_log():
    for handler in list(cli.LOG.handlers):
        cli.LOG.removeHandler(handler)


def test_cli_validate_checks_the_settings_after_the_hashing_flag(
        subtests, fake_cli, monkeypatch):
    """``validate`` pre-flights the settings with ``--[no-]hash-inputs`` applied."""
    _mod, _settings = fake_cli
    # The lower-case header is the second column pair the CSV reader accepts.
    settings = os.path.join(os.path.dirname(_settings), "lower.csv")
    with open(settings, "w", encoding="utf-8") as handle:
        handle.write("key,value\nsrc,%s\n" % os.path.join(
            os.path.dirname(_settings), "plate"))
    cases = [
        ("--hash-inputs turns hashing on", ["--hash-inputs"], True),
        ("--no-hash-inputs turns hashing off", ["--no-hash-inputs"], False),
        ("no flag leaves the file's answer", [], "absent"),
    ]
    for case_id, flags, expected in cases:
        with subtests.test(msg=case_id):
            seen = []
            monkeypatch.setattr(
                cli, "_preflight",
                lambda s, key, **_: seen.append(dict(s)) or [])
            assert cli.main(["validate", "--settings", settings, *flags]) \
                == cli.EXIT_OK
            assert len(seen) == 1
            assert seen[0].get("hash_inputs", "absent") is expected
            _reset_cli_log()


def test_cli_run_reports_the_lock_and_quiets_bars_only_off_a_terminal(
        subtests, fake_cli, monkeypatch, capsys):
    """A run logs the lock verdict at its level and mutes tqdm only on a pipe."""
    from spacr import run_journal

    mod, settings = fake_cli
    lock_cases = [("verified lock logs INFO", "verified", "INFO"),
                  ("deviating lock logs WARNING", "deviation", "WARNING")]
    for case_id, status, level in lock_cases:
        with subtests.test(msg=case_id):
            mod.calls.clear()
            summary = f"the analysis lock says {status}"
            with monkeypatch.context() as mp:
                mp.setattr(run_journal, "_find_lock",
                           lambda app_key, src: {"lock_id": "lock-1"})
                mp.setattr(
                    run_journal, "check_analysis_lock",
                    lambda s, app_key=None, lock=None, _st=status, _sm=summary: {
                        "status": _st, "summary": _sm, "deviations": []})
                assert cli.main(["_dispatch_fake", "--settings", settings]) \
                    == cli.EXIT_OK
            assert len(mod.calls) == 1
            lines = [line for line in capsys.readouterr().out.splitlines()
                     if summary in line]
            assert len(lines) == 1 and f" {level} " in lines[0], lines
            _reset_cli_log()
    tty_cases = [("a terminal keeps the bars", _TtyOut, None),
                 ("a pipe mutes the bars", io.StringIO, "1")]
    for case_id, stream, expected in tty_cases:
        with subtests.test(msg=case_id):
            mod.calls.clear()
            for key in ("TQDM_DISABLE", "SPACR_NO_PROGRESS"):
                os.environ.pop(key, None)
            with contextlib.redirect_stdout(stream()):
                rc = cli.main(["_dispatch_fake", "--settings", settings])
            assert rc == cli.EXIT_OK
            assert len(mod.calls) == 1
            assert mod.calls[0]["verbose"] is False
            assert os.environ.get("TQDM_DISABLE") == expected
            assert os.environ.get("SPACR_NO_PROGRESS") == expected
            _reset_cli_log()


# ---------------------------------------------------------------------------
# python -m spacr
# ---------------------------------------------------------------------------

def test_python_dash_m_spacr_opens_each_subcommand_on_its_tab(
        subtests, monkeypatch, capsys):
    """Each subcommand opens its tab, ``version`` prints, a stray choice exits 2."""
    import spacr.qt as qt
    from spacr.version import version_str

    opened = []
    monkeypatch.setattr(qt, "run",
                        lambda argv=None: opened.append(list(argv or [])) or 0)
    window_cases = [
        ("bare call opens home", [], []),
        ("gui opens home", ["gui"], []),
        ("mask", ["mask"], ["mask"]),
        ("measure", ["measure"], ["measure"]),
        ("classify", ["classify"], ["classify_merged"]),
        ("annotate", ["annotate"], ["annotate"]),
        ("sequencing", ["sequencing"], ["map_barcodes"]),
        ("umap", ["umap"], ["umap"]),
        ("make-masks", ["make-masks"], ["make_masks"]),
    ]
    for case_id, argv, launcher_argv in window_cases:
        with subtests.test(msg=case_id):
            opened.clear()
            assert M.main(argv) == 0
            assert opened == [launcher_argv]

    with subtests.test(msg="version prints and opens nothing"):
        opened.clear()
        capsys.readouterr()
        assert M.main(["version"]) == 0
        assert capsys.readouterr().out == version_str + "\n"
        assert opened == []

    with subtests.test(msg="a parser choice the dispatch lacks exits 2"):
        def parser_with_an_undispatched_choice():
            parser = argparse.ArgumentParser(prog="spacr")
            parser.add_argument("command", nargs="?", default="gui",
                                choices=["gui", "brand_new_screen"])
            return parser

        monkeypatch.setattr(M, "build_parser",
                            parser_with_an_undispatched_choice)
        capsys.readouterr()
        with pytest.raises(SystemExit) as excinfo:
            M.main(["brand_new_screen"])
        assert excinfo.value.code == 2
        assert "Unknown command: brand_new_screen" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# spacr repro
# ---------------------------------------------------------------------------

def test_spacr_repro_resolves_a_basename_under_the_runs_root(
        subtests, tmp_path, monkeypatch):
    """A bare run name resolves under the runs root; an unknown one exits 2."""
    from spacr import run_journal as rj

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    monkeypatch.setattr(cli_repro, "runs_root", lambda: root)
    with rj.open_run("mask", {"src": str(tmp_path / "data"), "n": 5}) as run:
        pass
    manifest = json.loads((run.dir / "manifest.json").read_text(
        encoding="utf-8"))
    # Run from a folder where the basename does NOT exist relative to cwd.
    monkeypatch.chdir(tmp_path)
    cases = [
        ("known basename shows the run", [run.dir.name, "--show"], 0,
         [f"run:       {run.dir.name}", "app:       mask",
          f"n_settings:{manifest['n_settings']}"], []),
        ("unknown basename is an error", ["not-a-run"], 2, [],
         ["error: no such run folder: not-a-run"]),
    ]
    for case_id, argv, code, out_parts, err_parts in cases:
        with subtests.test(msg=case_id):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                assert cli_repro.main(argv) == code
            for part in out_parts:
                assert part in out.getvalue()
            for part in err_parts:
                assert part in err.getvalue()
            if code:
                assert out.getvalue() == ""
