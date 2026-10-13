"""The unit file and the instructions beside it (item 453, and 3.001).

A unit that names the wrong script, or thresholds that disagree with the
watchdog's own defaults, fails silently -- the guard is simply not there
the next time it is needed. These read the files rather than the machine,
so they run anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "packaging" / "systemd" / "spacr-memory-watchdog.service"
README = ROOT / "packaging" / "systemd" / "README.md"
WATCHDOG = ROOT / "tools" / "memory_watchdog.py"


def test_the_unit_runs_the_watchdog_this_repository_ships():
    text = UNIT.read_text(encoding="utf-8")
    assert "spacr_memory_watchdog.py" in text
    assert "--act-gb 100" in text, "the maintainer's threshold"
    assert "--kill-gb 112" in text


def test_the_watchdog_cannot_be_the_thing_that_dies():
    """A watchdog chosen by the OOM killer was not there when it mattered."""
    text = UNIT.read_text(encoding="utf-8")
    assert re.search(r"^MemoryMax=\d+M", text, re.M), (
        "the unit has to cap the watchdog itself, or a leak in it becomes "
        "the next incident")
    assert "Restart=always" in text


def test_the_unit_thresholds_match_the_watchdogs_own_defaults():
    """Two numbers that can drift apart are two numbers that will."""
    unit = UNIT.read_text(encoding="utf-8")
    script = WATCHDOG.read_text(encoding="utf-8")
    for flag, name in (("--act-gb", "DEFAULT_ACT_GB"),
                       ("--kill-gb", "DEFAULT_KILL_GB"),
                       ("--floor-gb", "DEFAULT_FLOOR_GB")):
        in_unit = re.search(rf"{flag} (\d+)", unit)
        in_script = re.search(rf"{name}[^=]*= *([\d.]+)", script)
        assert in_unit and in_script, (flag, name)
        assert float(in_unit.group(1)) == float(in_script.group(1)), (
            f"{flag} in the unit and {name} in the script disagree")


def test_the_readme_says_to_copy_the_script_out_of_the_checkout():
    """A unit pointing into a git worktree stops working the moment the
    worktree moves, and the guard is then missing silently."""
    text = README.read_text(encoding="utf-8")
    assert "~/.local/bin/spacr_memory_watchdog.py" in text
    assert "install -Dm755" in text


def test_the_readme_gives_the_back_pressure_setting_and_says_it_needs_root():
    text = README.read_text(encoding="utf-8")
    assert "MemoryHigh=100G" in text
    assert "user@.service.d" in text
    assert "sudo" in text


def test_the_readme_says_what_none_of_it_replaces():
    """The failure mode of a safety net nobody distrusts is that people
    stop using the harness."""
    text = README.read_text(encoding="utf-8")
    assert "run_capped.sh" in text
    assert "every allocation" in text


def test_a_frozen_process_is_explained_rather_than_left_looking_hung():
    text = README.read_text(encoding="utf-8")
    assert "kill -CONT" in text
