"""Table-driven tests of how ``spacr.macro`` writes, finishes and renders a chain.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: :meth:`spacr.macro.Macro.write` (where the script lands
and that it appears atomically), :func:`spacr.macro.finish_recording` (the
step is always recorded, whatever the disk does, and only copies that were
written are announced) and :func:`spacr.macro._render_value` (every settings
value becomes exact Python source, lossy ones named in ``coerced``).  The
macro folder, home and log folders are redirected under ``tmp_path``.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

import pytest

from spacr import macro
from spacr.macro import Macro, MacroStep, Recording


@pytest.fixture
def isolated_macros(tmp_path, monkeypatch):
    """Point the macro folder, the home dir and the run logs at ``tmp_path``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    macro.reset()
    yield tmp_path
    macro.reset()


def _demo_macro(src: str) -> Macro:
    """A one-step mask chain that renders without going through a run."""
    chain = Macro()
    chain.steps.append(MacroStep(module="mask", entry_module="spacr.core",
                                 entry_func="preprocess_generate_masks",
                                 settings={"src": src}, user_set=("src",),
                                 status="success"))
    return chain


class _Model:
    """Stands in for a fitted estimator that reached a settings dict."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __str__(self) -> str:
        return f"<fitted model {self.tag}>"


# ---------------------------------------------------------------------------
# Macro.write
# ---------------------------------------------------------------------------

def test_macro_write_lands_whole_where_asked(subtests, isolated_macros, monkeypatch):
    """The script lands at the given path (relative or nested) via a renamed temporary."""
    work = isolated_macros / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    chain = _demo_macro(str(isolated_macros))
    expected_text = chain.source()
    tmp_name = f"macro.py.{os.getpid()}.tmp"
    cases = [
        # (case id, path handed to write, file that must hold the script)
        ("bare filename beside the caller", "macro.py", work / "macro.py"),
        ("nested relative path creates its parents",
         os.path.join("deep", "sub", "macro.py"), work / "deep" / "sub" / "macro.py"),
        ("absolute path in a new folder", isolated_macros / "out" / "macro.py",
         isolated_macros / "out" / "macro.py"),
    ]
    for case_id, given, target in cases:
        with subtests.test(msg=case_id):
            assert chain.write(given) == str(given)
            text = target.read_text(encoding="utf-8")
            assert text == expected_text
            assert text.startswith("#!")
            compile(text, str(target), "exec")
            assert sorted(p.name for p in target.parent.iterdir()
                          if p.name.endswith(".tmp")) == []
    with subtests.test(msg="the working folder holds only what was asked for"):
        assert sorted(p.name for p in work.iterdir()) == ["deep", "macro.py"]

    with subtests.test(msg="a render failure leaves the last good script and no temporary"):
        target = isolated_macros / "macro.py"
        target.write_text("last good macro", encoding="utf-8")

        def _fail_render():
            raise ValueError("settings cannot be rendered")

        monkeypatch.setattr(chain, "source", _fail_render)
        with pytest.raises(ValueError, match="^settings cannot be rendered$"):
            chain.write(target)
        assert target.read_text(encoding="utf-8") == "last good macro"
        assert not (isolated_macros / tmp_name).exists()


# ---------------------------------------------------------------------------
# finish_recording
# ---------------------------------------------------------------------------

def _announced(messages):
    """The success lines (``macro <id> (<chain>) → <path>``) among log messages."""
    return [m for m in messages if m.startswith("macro ") and "→" in m]


def test_finish_recording_keeps_the_step_whatever_the_disk_does(
        subtests, isolated_macros, monkeypatch, caplog):
    """Every finished recording is appended; only copies actually written are announced."""
    root = logging.getLogger()
    blocker = isolated_macros / "not-a-folder"
    blocker.write_text("this is a file, not a directory\n", encoding="utf-8")
    good_dir = isolated_macros / "macros"
    runs = isolated_macros / "runs"
    present = runs / "mask_20260101-120000_ab12cd34"
    present.mkdir(parents=True)

    cases = [
        # (case id, macro folder, run_dir, build capture-less Recording by hand,
        #  stable copy written, run copy written, failure logged)
        ("no log capture: run id falls back to the journal tag",
         good_dir, present, True, True, True, False),
        ("both folders usable: both copies, run copy announced",
         good_dir, present, False, True, True, False),
        ("macro folder is a file: run copy still written and announced",
         blocker, present, False, False, True, True),
        ("macro folder is a file and no journal: nothing written or announced",
         blocker, "", False, False, False, True),
        ("run folder vanished: stable copy only, no complaint about it",
         good_dir, runs / "already-deleted", False, True, False, False),
    ]
    for case_id, mdir, run_dir, by_hand, stable, run_copy, failed in cases:
        with subtests.test(msg=case_id):
            macro.reset()
            for old in (good_dir.glob("*.py") if good_dir.is_dir() else ()):
                old.unlink()
            if Path(str(run_dir) or "/nonexistent").is_dir():
                (Path(run_dir) / macro.MACRO_FILENAME).unlink(missing_ok=True)
            monkeypatch.setenv(macro.MACRO_DIR_ENV, str(mdir))
            handlers_before = list(root.handlers)
            if by_hand:
                recording = Recording(module="mask",
                                      settings={"src": str(isolated_macros)},
                                      run_dir=str(run_dir), started=0.0,
                                      started_utc="2026-01-01T00:00:00Z",
                                      capture=None)
            else:
                recording = macro.begin_recording(
                    "mask", {"src": str(isolated_macros)}, run_dir=str(run_dir))
                assert root.handlers == handlers_before + [recording.capture]
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="spacr.macro"):
                step = macro.finish_recording(recording, status="success")
            messages = [r.getMessage() for r in caplog.records]

            assert (step.module, step.status, step.run_dir) == ("mask", "success",
                                                                str(run_dir))
            assert root.handlers == handlers_before
            chain = macro.current_macro()
            assert chain.modules == ("mask",)
            assert chain.steps[-1] is step
            if by_hand:
                assert (step.run_ids, step.run_id, step.run_id_source) == (
                    (), "ab12cd34", "journal")
            stable_path = good_dir / f"{chain.macro_id}.py"
            assert stable_path.is_file() is stable
            run_path = Path(str(run_dir)) / macro.MACRO_FILENAME
            assert (bool(str(run_dir)) and run_path.is_file()) is run_copy
            written = [p for p, ok in ((stable_path, stable), (run_path, run_copy)) if ok]
            for path in written:
                text = path.read_text(encoding="utf-8")
                assert text == chain.source()
                compile(text, str(path), "exec")
            if by_hand:
                assert "ab12cd34" in stable_path.read_text(encoding="utf-8")
            expected_lines = ([f"macro {chain.macro_id} (mask) → {written[-1]}"]
                              if written else [])
            assert _announced(messages) == expected_lines
            assert ((f"could not write the macro for chain {chain.macro_id}"
                     in messages) is failed)
            assert not [m for m in messages
                        if m.startswith("could not write the macro into")]

    with subtests.test(msg="steps chain by declared edge, shared project, else start anew"):
        monkeypatch.setenv(macro.MACRO_DIR_ENV, str(good_dir))
        macro.reset()
        project = isolated_macros / "plate"
        sibling = isolated_macros / "plate-b"
        steps = [
            # (module, src, link, number of chains afterwards)
            ("mask", project, "", 1),
            ("measure", project, "ports", 1),
            ("mask", project / "other", "project", 1),
            ("mask", sibling, "", 2),
        ]
        for module, src, link, chains in steps:
            step = macro.finish_recording(
                macro.begin_recording(module, {"src": str(src)}), status="success")
            assert (step.module, step.link, step.project_root) == (module, link, str(src))
            assert len(macro.macros()) == chains
        first = macro.macros()[0]
        assert first.modules == ("mask", "measure", "mask")
        assert [line for line in first.source().splitlines()
                if line.startswith(("# Reads", "# Ran", "PROJECT_"))] == [
            f"PROJECT_1 = {str(project)!r}",
            f"PROJECT_2 = {str(project / 'other')!r}",
            "# Reads what step 1 (mask) produced \u2014 spacr.ports declares the edge.",
            "# Ran on the same project as step 2.",
        ]
        # An idle chain is not continued, even on the same project.
        macro.current_macro().touched = 0.0
        step = macro.finish_recording(
            macro.begin_recording("measure", {"src": str(sibling)}), status="success")
        assert (step.link, len(macro.macros())) == ("", 3)

    with subtests.test(msg="a recording that never started is a no-op"):
        macro.reset()
        assert macro.finish_recording(None, status="success") is None
        assert macro.current_macro() is None


# ---------------------------------------------------------------------------
# _render_value
# ---------------------------------------------------------------------------

def test_render_value_writes_each_settings_value_as_python_source(subtests, tmp_path):
    """Literals round-trip, project paths become expressions, the rest is named as coerced."""
    project = str(tmp_path / "plate")
    threader = macro._Threader([("PROJECT_1", project)])
    model = _Model("solo")
    cases = [
        # (case id, value, key, rendered source, coerced after, literal value or None)
        ("plain string", "hello", "", "'hello'", [], "hello"),
        ("empty string", "", "", "''", [], ""),
        ("project root", project, "", "PROJECT_1", [], None),
        ("path inside the project", os.path.join(project, "masks", "a.tif"), "",
         "os.path.join(PROJECT_1, 'masks', 'a.tif')", [], None),
        ("True", True, "", "True", [], True),
        ("None", None, "", "None", [], None),
        ("int", 7, "", "7", [], 7),
        ("float", 0.5, "", "0.5", [], 0.5),
        ("bytes", b"ab", "", "b'ab'", [], b"ab"),
        ("empty dict", {}, "", "{}", [], {}),
        ("dict", {"a": 1}, "", "{\n    'a': 1,\n}", [], {"a": 1}),
        ("one-tuple", (1,), "", "(1,)", [], (1,)),
        ("two-tuple", (1, 2), "", "(1, 2)", [], (1, 2)),
        ("empty set", set(), "", "set()", [], set()),
        ("set", {3}, "", "{3}", [], {3}),
        ("list", [1, "x"], "", "[1, 'x']", [], [1, "x"]),
        ("object with no key is rendered but not named", model, "",
         repr("<fitted model solo>"), [], "<fitted model solo>"),
        ("object under a key is named", model, "model",
         repr("<fitted model solo>"), ["model"], "<fitted model solo>"),
        ("two objects under one key are named once", [model, _Model("b")], "models",
         "['<fitted model solo>', '<fitted model b>']", ["models"],
         ["<fitted model solo>", "<fitted model b>"]),
    ]
    for case_id, value, key, rendered, coerced_after, literal in cases:
        with subtests.test(msg=case_id):
            coerced = []
            assert macro._render_value(value, threader, "", coerced, key) == rendered
            assert coerced == coerced_after
            if literal is not None:
                assert ast.literal_eval(rendered) == literal
    with subtests.test(msg="no coerced list at all is accepted"):
        assert macro._render_value(model, threader, "") == repr("<fitted model solo>")
