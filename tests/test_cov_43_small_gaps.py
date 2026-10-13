"""Paths the coverage ratchet found unexercised on dispatch 36739819315.

Each test here drives one behaviour that had no test after the per-module
baseline was written (2026-09-15): a ledger listener that fails, an analysis
lock the command line reports, a filter cell that gives only a maximum, and
the type-only imports two screens defer for start-up speed.
"""
from __future__ import annotations

import ast
import csv
import logging
import sys
import types
from pathlib import Path

import pytest

from spacr import cli, errors
from spacr.object_settings_table import _parse_filter_text

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# errors.RunLedger.finalize: a listener that raises
# ---------------------------------------------------------------------------

def test_a_failing_finalize_listener_does_not_fail_the_run(monkeypatch, caplog):
    """A listener is a bystander: its exception is logged at DEBUG and the
    ledger still finishes, and the listeners after it still hear about it."""
    heard = []

    def broken(ledger):
        raise RuntimeError("listener fell over")

    monkeypatch.setattr(errors, "_FINALIZE_LISTENERS",
                        [broken, heard.append])
    ledger = errors.RunLedger("listened")
    ledger.record_success("a")
    with caplog.at_level(logging.DEBUG):
        assert ledger.finalize() is ledger
    assert heard == [ledger]
    assert "a ledger listener failed" in caplog.text


# ---------------------------------------------------------------------------
# cli: the analysis lock verdict is logged at the level it deserves
# ---------------------------------------------------------------------------

@pytest.fixture
def lock_run(monkeypatch, tmp_path):
    """A fake pipeline behind the real ``spacr-run`` path, with the run
    journal written under ``tmp_path`` and the CLI logger restored after."""
    from spacr import run_journal

    journal_root = tmp_path / "runs"
    journal_root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: journal_root)
    module = types.ModuleType("spacr_cli_lock_pipeline")
    module.calls = []
    module.run = module.calls.append
    monkeypatch.setitem(sys.modules, "spacr_cli_lock_pipeline", module)
    monkeypatch.setitem(cli.MODULES, "_lock_fake", cli.Module(
        key="_lock_fake", summary="test-only module",
        entry="spacr_cli_lock_pipeline:run", defaults=None,
        validate_key="", requires=("src",), writes=("nothing",)))
    src = tmp_path / "plate"
    src.mkdir()
    settings = tmp_path / "s.csv"
    with open(settings, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerow(["src", str(src)])
    propagate, level = cli.LOG.propagate, cli.LOG.level
    yield module, str(settings)
    for handler in list(cli.LOG.handlers):
        cli.LOG.removeHandler(handler)
    cli.LOG.propagate = propagate
    cli.LOG.setLevel(level)
    for key in ("TQDM_DISABLE", "SPACR_NO_PROGRESS"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("status,level", [("verified", "INFO"),
                                          ("deviation", "WARNING")])
def test_the_command_line_reports_the_analysis_lock(lock_run, monkeypatch,
                                                    capsys, status, level):
    from spacr import run_journal

    module, settings = lock_run
    summary = f"the analysis lock says {status}"
    monkeypatch.setattr(run_journal, "_find_lock",
                        lambda app_key, src: {"lock_id": "lock-1"})
    monkeypatch.setattr(run_journal, "check_analysis_lock",
                        lambda settings, app_key=None, lock=None: {
                            "status": status, "summary": summary,
                            "deviations": []})
    assert cli.main(["_lock_fake", "--settings", settings]) == cli.EXIT_OK
    assert len(module.calls) == 1
    lines = [line for line in capsys.readouterr().out.splitlines()
             if summary in line]
    assert lines and f" {level} " in lines[0], lines


# ---------------------------------------------------------------------------
# object_settings_table: a filter cell holding only a maximum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [("-.5", (None, 0.5)),
                                           ("- .25", (None, 0.25))])
def test_a_leading_dash_before_a_non_digit_is_a_maximum_alone(text, expected):
    """``-.5`` cannot be a negative minimum (the range regex refuses a leading
    dash), so it reads as "no minimum, at most 0.5"."""
    assert _parse_filter_text(text) == expected


def test_a_leading_dash_before_a_digit_is_still_a_number():
    assert _parse_filter_text("-5") == (-5.0, None)


# ---------------------------------------------------------------------------
# Type-only imports two screens defer (items 284 and 380)
# ---------------------------------------------------------------------------

def _run_type_checking_block(module):
    """Execute ``module``'s ``if TYPE_CHECKING:`` block as the type checker
    would see it, compiled against the module's own file and line numbers,
    and return the names it binds.

    At run time the block never executes, so a class renamed or moved under
    it breaks only the annotations, silently. Running it here proves each
    name it imports still exists where the annotation says.
    """
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and isinstance(node.test, ast.Name)
              and node.test.id == "TYPE_CHECKING"]
    assert len(guards) == 1, f"{module.__name__} has {len(guards)} guards"
    code = compile(ast.Module(body=guards, type_ignores=[]), str(path), "exec")
    namespace = {"__name__": module.__name__,
                 "__package__": module.__package__, "TYPE_CHECKING": True}
    exec(code, namespace)
    return namespace


def test_the_hit_list_screens_type_only_import_names_the_real_class():
    pytest.importorskip("PySide6")
    from spacr import hits
    from spacr.qt.screens import hit_list

    bound = _run_type_checking_block(hit_list)
    assert bound["HitList"] is hits.HitList


def test_linked_selections_type_only_import_names_pandas():
    pytest.importorskip("PySide6")
    import pandas

    from spacr.qt import linked_selection

    bound = _run_type_checking_block(linked_selection)
    assert bound["pd"] is pandas
    assert "pd" not in vars(linked_selection)
