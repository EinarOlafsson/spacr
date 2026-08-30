"""The macro recorder when the disk, not the run, is what goes wrong.

The recorder's whole promise is that it never costs a run: a script that
cannot be written is a log line and nothing more. These tests drive the
three ways writing actually fails on a real install -- a macro folder that
is not a folder, a run with no journal folder to keep a second copy in,
and a caller that hands ``Macro.write`` a bare filename -- and pin the
part users depend on: the pipeline still finishes, the step is still
recorded in the chain, and whatever copy *could* be written is on disk.

The rendering half is covered where the settings dict holds something
Python cannot round-trip (a fitted model object, a numpy array), which the
script must admit to rather than silently mangle.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from spacr import macro
from spacr.macro import Macro, MacroStep, Recording


@pytest.fixture
def isolated_macros(tmp_path, monkeypatch):
    """Point the macro folder, the home dir and the run logs at tmp_path.

    ``macros_dir()`` defaults to ``~/.spacr/macros``; without this a test
    would write scripts into the developer's real home and inherit chains
    recorded by whatever ran before it.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(tmp_path / "macros"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    macro.reset()
    yield tmp_path
    macro.reset()


def _demo_macro(module: str = "mask", **settings) -> Macro:
    """A one-step chain that renders, without going through a run."""
    chain = Macro()
    chain.steps.append(MacroStep(module=module,
                                 entry_module="spacr.core",
                                 entry_func="preprocess_generate_masks",
                                 settings=dict(settings),
                                 user_set=tuple(settings),
                                 status="success"))
    return chain


# ---------------------------------------------------------------------------
# Where the script is written


def test_a_bare_filename_writes_beside_the_caller_not_at_the_root(
        isolated_macros, monkeypatch):
    """`write('macro.py')` must land in the working directory.

    ``spacr-repro`` and the notebook exporter both write a macro out under
    a name the user typed. A relative name has no directory part, and if
    the writer treated that empty string as a directory to create it would
    raise ``FileNotFoundError`` on the most ordinary call there is. The
    nested name in the same test proves the directory *is* created when
    there really is one, so this is not passing by doing nothing.
    """
    chain = _demo_macro(src=str(isolated_macros))
    workdir = isolated_macros / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    bare = chain.write("macro.py")
    nested = chain.write(os.path.join("deep", "sub", "macro.py"))

    assert bare == "macro.py"
    assert (workdir / "macro.py").is_file()
    assert (workdir / "macro.py").read_text().startswith("#!")
    # The nested call created its parents; the bare one had none to create.
    assert (workdir / "deep" / "sub" / "macro.py").is_file()
    assert nested == os.path.join("deep", "sub", "macro.py")
    assert sorted(p.name for p in workdir.iterdir()) == ["deep", "macro.py"]


def test_the_script_appears_whole_or_not_at_all(isolated_macros):
    """A reader must never catch the file half-written.

    A chain is rewritten in place every time a step joins it, so a
    ``spacr-repro`` that opens the script while Measure is being appended
    would read a truncated file if the writer streamed into the target.
    The write goes to a neighbouring temp name and is renamed, so the temp
    file must be gone and the target must parse once ``write`` returns.
    """
    chain = _demo_macro(src=str(isolated_macros))
    target = isolated_macros / "out" / "macro.py"

    written = chain.write(target)

    assert written == str(target)
    compile(target.read_text(), str(target), "exec")
    leftovers = [p.name for p in (isolated_macros / "out").iterdir()
                 if p.name != "macro.py"]
    assert leftovers == [], f"temporary file left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Finishing a recording that never started a log capture


def test_a_recording_with_no_log_capture_still_records_its_step(
        isolated_macros):
    """A recorder that could not attach to the root logger must still emit.

    ``Recording.capture`` is optional -- a caller that builds one by hand,
    or a process where adding the handler failed, leaves it None. If
    ``finish_recording`` assumed the handler was there it would raise
    inside the run's teardown, which is exactly the thing this module
    promises never to do. The step is still appended and the script is
    still written; only the observed run id is missing.
    """
    handlers_before = list(logging.getLogger().handlers)
    run_dir = isolated_macros / "runs" / "mask_20260101-120000_ab12cd34"
    run_dir.mkdir(parents=True)
    recording = Recording(module="mask",
                          settings={"src": str(isolated_macros)},
                          run_dir=str(run_dir),
                          started=0.0,
                          started_utc="2026-01-01T00:00:00Z",
                          capture=None)

    step = macro.finish_recording(recording, status="success")

    assert step is not None and step.module == "mask"
    assert step.status == "success"
    # Nothing observed the run's own log records, so the id falls back to
    # the journal folder's tag and says so.
    assert step.run_ids == ()
    assert step.run_id == "ab12cd34"
    assert step.run_id_source == "journal"
    assert macro.current_macro().modules == ("mask",)
    # Nothing was attached, so nothing may be detached: the root logger is
    # exactly as it was, which a capture-holding recording would change.
    assert logging.getLogger().handlers == handlers_before
    scripts = list(Path(os.environ[macro.MACRO_DIR_ENV]).glob("*.py"))
    assert len(scripts) == 1
    assert "ab12cd34" in scripts[0].read_text()


def test_a_capture_holding_recording_is_detached_when_it_finishes(
        isolated_macros):
    """The run-id handler must come off the root logger when the run ends.

    It sits on the *root* logger for the life of the run, so a recorder
    that forgot to remove it would leave one handler per run behind and
    every later run would pay to walk them. This is the companion of the
    capture-less case above: same seam, handler present.
    """
    root = logging.getLogger()
    before = len(root.handlers)

    recording = macro.begin_recording("mask", {"src": str(isolated_macros)})

    assert len(root.handlers) == before + 1
    assert recording.capture in root.handlers

    step = macro.finish_recording(recording, status="success")

    assert step.module == "mask"
    assert len(root.handlers) == before
    assert recording.capture not in root.handlers


# ---------------------------------------------------------------------------
# When the script cannot be written


def test_a_macro_folder_that_is_a_file_still_leaves_the_run_its_copy(
        isolated_macros, monkeypatch, caplog):
    """A misconfigured SPACR_MACRO_DIR must not cost the run its script.

    Point the env var at a regular file -- a plausible typo, and what a
    cluster gets when a home directory is a read-only stub -- and the
    stable copy under the macro folder cannot be created. The copy beside
    the run's own manifest is written from a different path and must still
    appear, because that is the one ``spacr-repro`` is pointed at. The
    failure is a logged exception, not a raised one.
    """
    blocker = isolated_macros / "not-a-folder"
    blocker.write_text("this is a file, not a directory\n")
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(blocker))
    run_dir = isolated_macros / "runs" / "run-abc123"
    run_dir.mkdir(parents=True)

    recording = macro.begin_recording("mask", {"src": str(isolated_macros)},
                                      run_dir=str(run_dir))
    with caplog.at_level(logging.INFO, logger="spacr.macro"):
        step = macro.finish_recording(recording, status="success")

    assert step is not None and step.run_dir == str(run_dir)
    assert macro.current_macro().modules == ("mask",)
    run_copy = run_dir / macro.MACRO_FILENAME
    assert run_copy.is_file()
    compile(run_copy.read_text(), str(run_copy), "exec")
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not write the macro for chain" in text
               for text in messages), messages
    # The surviving copy is announced, so a user reading the log knows
    # which of the two paths actually holds the script.
    assert any(str(run_copy) in text for text in messages), messages


def test_when_neither_copy_can_be_written_the_recorder_announces_nothing(
        isolated_macros, monkeypatch, caplog):
    """A run with no journal folder and no macro folder writes no script.

    Both destinations are gone: the macro folder is a file, and a run
    launched without a journal (the CLI's ``--no-journal`` shape) has no
    run folder to hold the second copy. The recorder must not claim in the
    log that it wrote a script it did not write -- a user who greps for
    the "macro ... ->" line and follows the path would find nothing there.
    The run itself still finishes and the step is still in the chain, so
    ``current_macro()`` remains usable for the exporter.
    """
    blocker = isolated_macros / "not-a-folder"
    blocker.write_text("x\n")
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(blocker))

    recording = macro.begin_recording("mask", {"src": str(isolated_macros)},
                                      run_dir="")
    with caplog.at_level(logging.INFO, logger="spacr.macro"):
        step = macro.finish_recording(recording, status="success")

    assert step is not None and step.module == "mask"
    assert step.run_dir == ""
    assert macro.current_macro().modules == ("mask",)
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not write the macro for chain" in text
               for text in messages), messages
    assert not any(text.startswith("macro ") and "\u2192" in text
                   for text in messages), messages

    # Same recorder, working macro folder: now the success line *is*
    # emitted, which is what makes its absence above meaningful.
    caplog.clear()
    good = isolated_macros / "macros-ok"
    monkeypatch.setenv(macro.MACRO_DIR_ENV, str(good))
    macro.reset()
    with caplog.at_level(logging.INFO, logger="spacr.macro"):
        again = macro.begin_recording("mask", {"src": str(isolated_macros)})
        macro.finish_recording(again, status="success")
    later = [record.getMessage() for record in caplog.records]
    assert any(text.startswith("macro ") and "\u2192" in text
               for text in later), later
    assert len(list(good.glob("*.py"))) == 1


def test_a_run_folder_that_vanished_costs_only_its_copy(isolated_macros,
                                                        caplog):
    """A deleted run folder must not stop the stable copy being written.

    Journal folders get cleaned up while a long chain is still running.
    The stable copy under the macro folder is the one that survives that,
    so it must be written regardless, and no exception about the missing
    run folder may be logged -- there is nothing wrong to report.
    """
    recording = macro.begin_recording(
        "mask", {"src": str(isolated_macros)},
        run_dir=str(isolated_macros / "runs" / "already-deleted"))
    with caplog.at_level(logging.INFO, logger="spacr.macro"):
        step = macro.finish_recording(recording, status="success")

    assert step.run_dir.endswith("already-deleted")
    scripts = list(Path(os.environ[macro.MACRO_DIR_ENV]).glob("*.py"))
    assert len(scripts) == 1
    assert scripts[0].name == f"{macro.current_macro().macro_id}.py"
    messages = [record.getMessage() for record in caplog.records]
    assert not any("could not write the macro into" in text
                   for text in messages), messages
    # And when the folder does exist, the second copy is written -- so the
    # absence above is a missing folder, not a dead code path.
    other = isolated_macros / "runs" / "present"
    other.mkdir(parents=True)
    second = macro.begin_recording("measure", {"src": str(isolated_macros)},
                                   run_dir=str(other))
    macro.finish_recording(second, status="success")
    assert (other / macro.MACRO_FILENAME).is_file()


# ---------------------------------------------------------------------------
# Rendering a value Python cannot round-trip


class _Model:
    """Stands in for a fitted estimator that reached a settings dict."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __str__(self) -> str:
        return f"<fitted model {self.tag}>"


def test_a_key_holding_several_unrenderable_values_is_named_once(
        isolated_macros):
    """The lossy-conversion list must name a key once, not once per item.

    ``coerced`` is what tells a reader "this line is a string of the thing,
    not the thing" -- the methods exporter prints it verbatim. A list of
    two fitted models under one key is one lossy key; naming it twice
    would make the exported methods section claim two settings were
    approximated when only one was, and a consumer de-duplicating the list
    would be papering over a recorder bug.
    """
    chain = _demo_macro("classify",
                        src=str(isolated_macros),
                        models=[_Model("a"), _Model("b")],
                        threshold=0.5)

    source = macro.render(chain)
    step = chain.steps[0]

    assert step.coerced == ("models",)
    assert "<fitted model a>" in source
    assert "<fitted model b>" in source
    # A genuine literal in the same dict is rendered as itself and is not
    # named as coerced, which is what the distinction is for.
    assert "0.5" in source
    assert "threshold" not in step.coerced
    compile(source, "<macro>", "exec")


def test_an_unrenderable_value_under_no_key_is_still_rendered(
        isolated_macros):
    """Rendering a bare value must not depend on a key being supplied.

    ``_render_value`` is called recursively for dict *keys* as well as
    values, and for nested items where there is nothing sensible to add to
    the coerced list. If it required a key it would raise while rendering
    a settings dict keyed by anything exotic, and the whole script would be
    lost to a run that was otherwise fine.
    """
    threader = macro._Threader([])
    collected = []

    anonymous = macro._render_value(_Model("solo"), threader, "")
    named = macro._render_value(_Model("solo"), threader, "", collected,
                                "model")

    assert anonymous == repr("<fitted model solo>")
    assert named == anonymous
    assert collected == ["model"]
