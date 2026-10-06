"""The serial Qt journal keeps file and failure evidence through hard exits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="journal reads /proc")


def _run_two_files(tmp_path: Path, second_body: str, first_body: str | None = None):
    """Run two tiny tests in one process with an isolated journal."""
    first = tmp_path / "test_first.py"
    second = tmp_path / "test_second.py"
    first.write_text(
        first_body or "def test_first():\n    assert True\n", encoding="utf-8"
    )
    second.write_text(second_body, encoding="utf-8")
    journal = tmp_path / "file-rss.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["SPACR_QT_SERIAL_RSS_JOURNAL"] = str(journal)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "tools.pytest_plugins.qt_serial_rss_journal",
            str(first),
            str(second),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    records = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    return result, records


def test_a_normal_failure_records_both_completed_file_boundaries(tmp_path):
    """A red assertion still leaves a complete ordered RSS accounting."""
    result, records = _run_two_files(tmp_path, "def test_second():\n    assert False\n")
    assert result.returncode == 1, result.stderr
    assert [row["event"] for row in records] == [
        "session_start",
        "collection",
        "file_begin",
        "file_end",
        "file_begin",
        "test_failure",
        "file_end",
        "session_finish",
    ]
    assert records[1]["tests"] == 2 and records[1]["files"] == 2
    assert records[3]["file"] == "test_first.py"
    assert records[5]["nodeid"].endswith("test_second.py::test_second")
    assert records[5]["when"] == "call"
    assert "assert False" in records[5]["detail"]
    assert records[5]["detail_truncated"] is False
    assert records[6]["file"] == "test_second.py"
    assert records[-1]["exitstatus"] == 1
    assert records[-1]["completed_files"] == 2
    assert all(row["rss_bytes"] > 0 for row in records)


def test_a_hard_exit_preserves_the_previous_file_and_running_file(tmp_path):
    """The memory guard's os._exit leaves synced evidence of the last start."""
    result, records = _run_two_files(
        tmp_path, "import os\n\ndef test_second():\n    os._exit(3)\n"
    )
    assert result.returncode == 3, result.stderr
    assert [row["event"] for row in records] == [
        "session_start",
        "collection",
        "file_begin",
        "file_end",
        "file_begin",
    ]
    assert records[3]["file"] == "test_first.py"
    assert records[-1]["file"] == "test_second.py"
    assert records[3]["hwm_bytes"] >= records[3]["rss_bytes"] > 0


def test_a_prior_failure_is_synced_before_a_later_hard_exit(tmp_path):
    """A native exit cannot erase the assertions from earlier test reports."""
    result, records = _run_two_files(
        tmp_path,
        "import os\n\ndef test_second():\n    os._exit(3)\n",
        first_body="def test_first():\n    assert 2 == 3\n",
    )
    assert result.returncode == 3, result.stderr
    assert [row["event"] for row in records] == [
        "session_start",
        "collection",
        "file_begin",
        "test_failure",
        "file_end",
        "file_begin",
    ]
    assert records[3]["nodeid"].endswith("test_first.py::test_first")
    assert "assert 2 == 3" in records[3]["detail"]
    assert records[3]["detail_chars"] == len(records[3]["detail"])
    assert records[-1]["file"] == "test_second.py"
