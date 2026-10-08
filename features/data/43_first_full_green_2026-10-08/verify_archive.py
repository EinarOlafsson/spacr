"""Check the exact-source first-green GitHub Actions evidence archive."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _digest(data):
    """Return the SHA-256 of an archived byte string."""
    return hashlib.sha256(data).hexdigest()


def _payload(record):
    """Validate and decode one unchanged GitHub API response."""
    compressed = (HERE / record["file"]).read_bytes()
    assert _digest(compressed) == record["gzip_sha256"]
    raw = gzip.decompress(compressed)
    assert _digest(raw) == record["raw_sha256"]
    return json.loads(raw)


def main():
    """Verify source identity, every terminal job and the required gates."""
    receipt = json.loads((HERE / "receipt.json").read_text())
    sha = receipt["source_sha"]
    subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                   cwd=ROOT, check=True)
    expected = {"tests": 29, "docs": 4, "compat": 17}
    for name, count in expected.items():
        entry = receipt["runs"][name]
        run = _payload(entry["payloads"]["run"])
        jobs = _payload(entry["payloads"]["jobs"])
        assert run["id"] == entry["id"]
        assert run["head_sha"] == sha
        assert run["status"] == "completed" and run["conclusion"] == "success"
        assert run["created_at"] == entry["created_at"]
        assert run["updated_at"] == entry["updated_at"]
        assert jobs["total_count"] == count == len(jobs["jobs"])
        assert entry["job_count"] == entry["successful_jobs"] == count
        identifiers = {job["id"] for job in jobs["jobs"]}
        assert len(identifiers) == count
        assert all(job["run_id"] == run["id"] and job["head_sha"] == sha
                   and job["status"] == "completed"
                   and job["conclusion"] == "success" for job in jobs["jobs"])
        if name == "tests":
            names = [job["name"] for job in jobs["jobs"]]
            assert sum(n.startswith("Coverage shard ") for n in names) == 12
            assert sum(n.startswith("Qt (") for n in names) == 3
            assert sum(n.startswith("Fast /") for n in names) == 3
            assert sum(n.startswith("Minimum dependencies") for n in names) == 3
            assert "Coverage / every module at 90% and none loses coverage" in names
            assert "Release gate / zero unexpected failures" in names
    print("Exact-source green Actions archive verified: tests 29, docs 4, compat 17")


if __name__ == "__main__":
    main()
