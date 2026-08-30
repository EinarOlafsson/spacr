"""``spacr.cli``: the ``--set`` parses that back out, and a run on a real tty.

``coerce_value`` guesses a type before it parses, and three of its guesses
have to be able to be wrong without the parse going wrong with them:

* a key whose ``expected_types`` entry declares no *type* at all falls back to
  the type of the value the key already holds, rather than accepting anything;
* ``{...}`` is a dict *shape*, and ``{1, 2}`` is a set;
* ``(...)`` is a tuple *shape*, and ``(1)`` is an integer in brackets.

In each case the literal parse succeeds and produces the wrong kind of object,
and the branch that notices has to hand the text on to the next rule instead
of returning it.

The last test runs ``cmd_run`` with stdout attached to something that claims
to be a terminal, which is the branch the CLI never takes under pytest and the
only one where it leaves tqdm alone.
"""
from __future__ import annotations

import io
import os
import sys
import types

import pytest

from spacr import cli


# ---------------------------------------------------------------------------
# coerce_value: a shape that parses into the wrong kind of object
# ---------------------------------------------------------------------------

def test_a_declaration_that_names_no_type_falls_back_to_the_current_value():
    """``expected_types`` entries that are not types must not widen the key.

    ``expected_types`` spells its entries several ways -- ``type(None)``, a
    bare ``None``, tuples -- and normalising them can leave nothing behind. An
    empty result has to mean "I learned nothing here", so the type of the
    value the key already holds decides, and ``--set`` stays as strict as it
    would have been with no declaration at all.
    """
    # The declaration is the string 'int', not the type: nothing survives
    # normalisation, so the current value (5) makes this an int key.
    assert coerce("thing", "7", 5, {"thing": "int"}) == 7
    with pytest.raises(cli.SettingsError) as excinfo:
        coerce("thing", "abc", 5, {"thing": "int"})
    assert "cannot be read as int" in str(excinfo.value)

    # And a declaration that *is* a type still wins over the current value --
    # same key, same current value, opposite answer.
    assert coerce("thing", "abc", 5, {"thing": str}) == "abc"


def test_braces_that_parse_to_a_set_are_not_returned_as_a_dict():
    """``{1, 2}`` is a set literal, and a set is not a settings value.

    The dict rule matches on the leading brace, so it has to check what came
    back. Returning the set would put an unordered, unserialisable object into
    a settings CSV; falling through leaves it as the text the user typed.
    """
    assert coerce("k", "{1, 2}", None, {}) == "{1, 2}"
    # The same rule, one character different, really does yield a dict.
    assert coerce("k", "{'a': 1}", None, {}) == {"a": 1}


def test_brackets_that_parse_to_a_scalar_are_not_returned_as_a_list():
    """``(1)`` is an integer in brackets, not a one-element tuple.

    Python's own grammar says so, and ``ast.literal_eval`` agrees. The list
    rule matches on the leading bracket, so it has to check the result before
    handing back something a caller would iterate.
    """
    assert coerce("k", "(1)", None, {}) == "(1)"
    # The comma is what makes it a sequence, and then the rule does return one.
    assert coerce("k", "(1, 2)", None, {}) == [1, 2]


def coerce(key, text, current, expected_types):
    """``coerce_value`` with the module argument the tests never vary."""
    return cli.coerce_value(key, text, current, expected_types, "")


# ---------------------------------------------------------------------------
# cmd_run on a terminal
# ---------------------------------------------------------------------------

class _TtyOut(io.StringIO):
    """A writable stdout that reports itself as a terminal."""

    def isatty(self):
        """Yes -- which is what the CLI branches on."""
        return True


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """Register a synthetic runnable module and a clean settings file.

    Mirrors ``tests/test_cli.py``'s ``fake_pipeline`` / ``fake_settings`` pair:
    the entry point is resolved through ``sys.modules`` so no pipeline, and no
    torch, is imported.
    """
    from spacr import run_journal

    journal_root = tmp_path / "runs"
    journal_root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: journal_root)

    mod = types.ModuleType("spacr_cli_r5_fake")
    mod.calls = []
    mod.run = lambda settings: mod.calls.append(settings)
    monkeypatch.setitem(sys.modules, "spacr_cli_r5_fake", mod)
    monkeypatch.setitem(cli.MODULES, "_r5_fake", cli.Module(
        key="_r5_fake", summary="test-only module",
        entry="spacr_cli_r5_fake:run", defaults=None, validate_key="",
        requires=("src",), writes=("nothing",)))

    plate = tmp_path / "plate"
    plate.mkdir()
    settings = tmp_path / "s.csv"
    settings.write_text("Key,Value\nsrc,%s\nverbose,False\n" % plate,
                        encoding="utf-8")

    watched = ("TQDM_DISABLE", "SPACR_NO_PROGRESS")
    before = {k: os.environ.get(k) for k in watched}
    for key in watched:
        os.environ.pop(key, None)
    yield mod, str(settings)
    for handler in list(cli.LOG.handlers):
        cli.LOG.removeHandler(handler)
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_a_run_on_a_terminal_leaves_the_progress_bars_alone(monkeypatch,
                                                            fake_run):
    """tqdm is only muzzled when stdout is a pipe, not when a human is watching.

    ``_quiet_progress_bars`` sets ``TQDM_DISABLE`` and ``SPACR_NO_PROGRESS`` in
    the environment the pipeline then runs in, so on a terminal it must not
    fire: a user running ``spacr-run`` interactively should see the bars. This
    is the branch pytest itself can never take, because its stdout is captured.
    """
    mod, settings = fake_run

    monkeypatch.setattr(sys, "stdout", _TtyOut())
    rc = cli.main(["_r5_fake", "--settings", settings])

    assert rc == cli.EXIT_OK
    assert len(mod.calls) == 1
    assert "TQDM_DISABLE" not in os.environ
    assert "SPACR_NO_PROGRESS" not in os.environ

    # Off a terminal the same run really does set them, so the assertion above
    # is about the tty and not about the run having skipped the check.
    mod.calls.clear()
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert cli.main(["_r5_fake", "--settings", settings]) == cli.EXIT_OK
    assert len(mod.calls) == 1
    assert os.environ.get("TQDM_DISABLE") == "1"
    assert os.environ.get("SPACR_NO_PROGRESS") == "1"
