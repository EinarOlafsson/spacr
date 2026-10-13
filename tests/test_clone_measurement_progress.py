"""Clone measurements distinguish completed transfers from progress updates."""
from pathlib import Path
import os
import subprocess


HELPER = Path(__file__).resolve().parents[1] / "packaging" / "measure_clone_forms.sh"


def _measure(tmp_path, form, progress):
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_git = binary / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "for destination do :; done\n"
        'mkdir -p "$destination/.git"\n'
        'cp "$SPACR_MEASUREMENT_PROGRESS" /dev/stderr\n'
    )
    fake_git.chmod(0o755)
    log = tmp_path / "progress"
    log.write_text(progress)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
           "SPACR_MEASUREMENT_PROGRESS": str(log)}
    result = subprocess.run(
        ["sh", str(HELPER), "--repo", "https://example.invalid/spacr.git",
         "--dir", str(tmp_path / "clones"), "--forms", form, "--keep"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()[1].split()


def test_multigigabyte_download_counts_completed_transfers_once(tmp_path):
    row = _measure(tmp_path, "full",
        "Receiving objects: 100% (152192/152192), 7.56 GiB | 40.52 MiB/s\r"
        "Receiving objects: 100% (152192/152192), 7.56 GiB | 41.15 MiB/s, done.\n"
        "Receiving objects: 100% (2/2), 512.00 KiB | 1.00 MiB/s, done.\n")
    assert row[2:4] == ["7741.9", "MB"]


def test_partial_clone_does_not_call_initial_tree_fetch_total_download(tmp_path):
    row = _measure(tmp_path, "depth1-filter",
        "Receiving objects: 100% (522/522), 497.05 KiB | 4.60 MiB/s, done.\n")
    assert row[2] == "n/a"
