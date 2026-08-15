"""The instruction index cannot go stale without CI saying so.

It was hand-written on 2026-08-05 and was nine days out of date by
2026-08-14: it listed work that had shipped and omitted work that had been
filed. An index that disagrees with the folder is worse than none, because it
is believed -- and this repository has already lost time to eight instruction
files that were wrong about their own state.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "build_instruction_index.py"
INSTRUCTIONS = REPO / "instructions"

pytestmark = pytest.mark.skipif(
    not TOOL.exists() or not INSTRUCTIONS.is_dir(),
    reason="run from a source checkout")


def _tool():
    spec = importlib.util.spec_from_file_location("_instr_index", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_instr_index"] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_index_matches_the_instruction_files():
    """If this fails, run tools/build_instruction_index.py."""
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"], cwd=str(REPO),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_open_instruction_appears():
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    for path in (INSTRUCTIONS / "open").glob("*.txt"):
        assert path.name in text, f"{path.name} is missing from the index"


def test_a_done_instruction_is_not_listed_as_open():
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    open_block = text.split("OPEN")[1].split("DONE")[0]
    for path in (INSTRUCTIONS / "done").glob("*.txt"):
        assert path.name not in open_block, (
            f"{path.name} is done but listed under OPEN")


def test_the_counts_are_the_real_counts():
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    n_open = len(list((INSTRUCTIONS / "open").glob("*.txt")))
    n_done = len(list((INSTRUCTIONS / "done").glob("*.txt")))
    assert f"{n_done} done / {n_open} open" in text


def test_codex_owned_open_files_are_marked_do_not_touch():
    """Two sessions editing one file is how work gets lost."""
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    open_numbers = {
        path.name.split("_", 1)[0]
        for path in (INSTRUCTIONS / "open").glob("*.txt")
    }
    # An owner whose instruction has since been DONE is the normal end state.
    # Asserting every owner is still open made the index fail for work being
    # finished, which is the opposite of what this guard is for.
    stale = {n for n in tool.OWNERS if n not in open_numbers}
    assert not stale, (
        f"OWNERS names {sorted(stale)}, which are no longer open; remove "
        f"them so the marking tracks the folder")
    for number in tool.OWNERS:
        paths = list((INSTRUCTIONS / "open").glob(f"{number}_*.txt"))
        assert len(paths) == 1, number
        block = [b for b in text.split("\n\n") if paths[0].name in b]
        assert block, f"instruction {number} is not in the index"
        assert "DO NOT TOUCH" in block[0], number


def test_the_date_alone_does_not_make_it_stale():
    """An index that failed CI because a day passed is one nobody keeps."""
    tool = _tool()
    fresh = tool.render(today="2099-01-01")
    other = tool.render(today="1999-01-01")
    body = lambda t: "\n".join(l for l in t.splitlines()
                               if not l.startswith("Regenerated "))
    assert body(fresh) == body(other)


def test_the_index_points_at_the_handoff():
    """A new session reads the index; the traps are in the handoff."""
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    assert "HANDOFF.md" in text
    assert (INSTRUCTIONS / "HANDOFF.md").exists()
