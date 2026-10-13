"""Recount and verify terminal 5636 ordinary Qt jobs from original logs."""

from hashlib import sha256
from pathlib import Path
import gzip
import json
import re
import sys


SOURCE = "5636eac60938f5d6021eaf73546ff8a30ae17c32"
SOURCE_TREE = "0c4e4f01526f31796e2c87eb5c2c8f728dd81803"
RUN_ID = 37763414112
JOBS = ((0, 113289789217, "success"), (1, 113289789324, "success"),
        (2, 113289789703, "success"))
OUTCOME = re.compile(r"(\d+) (passed|skipped|xfailed|xpassed|failed|errors?)\b")
TIMING = re.compile(r"=+ .*\bin [\d.]+s")
BATCH = re.compile(r"pytest batch \d+/\d+:")
FINAL_FAILURE = re.compile(r"\bFAILED (tests/[^ ]+) - ")


def _sha(data: bytes) -> str:
    """Return the SHA-256 hex digest of archival bytes."""
    return sha256(data).hexdigest()


def build_manifest(directory: Path) -> dict:
    """Validate source identity and recount all original pytest batch summaries."""
    run_bytes = (directory / "run.json").read_bytes()
    run = json.loads(run_bytes)
    assert run["id"] == RUN_ID and run["head_sha"] == SOURCE
    records = []
    for shard, job_id, expected_conclusion in JOBS:
        metadata_name = f"job-{job_id}.json"
        metadata_bytes = (directory / metadata_name).read_bytes()
        job = json.loads(metadata_bytes)
        assert job["id"] == job_id and job["head_sha"] == SOURCE
        assert job["status"] == "completed" and job["conclusion"] == expected_conclusion
        assert job["run_id"] == RUN_ID
        compressed_name = f"qt{shard}-{job_id}.log.gz"
        compressed = (directory / compressed_name).read_bytes()
        raw = gzip.decompress(compressed)
        text = raw.decode("utf-8", errors="replace")
        assert SOURCE in text
        headings = len(BATCH.findall(text))
        assert headings == 243
        summaries = []
        for line in text.splitlines():
            if TIMING.search(line):
                outcomes = OUTCOME.findall(line)
                if outcomes:
                    summaries.append(outcomes)
        totals = {
            name: sum(int(count) for row in summaries for count, kind in row
                      if kind == name)
            for name in ("passed", "skipped", "xfailed", "failed", "error", "errors")
        }
        failed_nodes = sorted(set(FINAL_FAILURE.findall(text)))
        assert not failed_nodes
        assert totals["failed"] == totals["error"] == totals["errors"] == 0
        records.append({
            "shard": shard,
            "job_id": job_id,
            "conclusion": expected_conclusion,
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "job_metadata": metadata_name,
            "job_metadata_sha256": _sha(metadata_bytes),
            "log": compressed_name,
            "log_gzip_bytes": len(compressed),
            "log_gzip_sha256": _sha(compressed),
            "log_original_bytes": len(raw),
            "log_original_sha256": _sha(raw),
            "batch_headings": headings,
            "pytest_summary_rows": len(summaries),
            "passed": totals["passed"],
            "skipped": totals["skipped"],
            "expected_xfailed": totals["xfailed"],
            "failed": totals["failed"],
            "failed_nodes": failed_nodes,
        })
    return {
        "run_id": RUN_ID,
        "run_conclusion": run["conclusion"],
        "event": run["event"],
        "source_sha": SOURCE,
        "source_tree_sha": SOURCE_TREE,
        "run_metadata": "run.json",
        "run_metadata_sha256": _sha(run_bytes),
        "jobs": records,
        "total_passed": sum(item["passed"] for item in records),
        "total_skipped": sum(item["skipped"] for item in records),
        "total_expected_xfailed": sum(item["expected_xfailed"] for item in records),
        "total_failed": sum(item["failed"] for item in records),
        "scope": "All three 5636 ordinary Qt shards passed. Coverage0 had a separate native SIGSEGV; the required run and its gates remain failed.",
    }


def main() -> None:
    """Write a deterministic manifest once, then verify it on later invocations."""
    directory = Path(__file__).resolve().parent
    computed = build_manifest(directory)
    path = directory / "MANIFEST.json"
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        path.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n")
    else:
        assert len(sys.argv) == 1
        assert json.loads(path.read_text()) == computed
    print(json.dumps({"jobs": len(computed["jobs"]),
                      "passed": computed["total_passed"],
                      "skipped": computed["total_skipped"],
                      "xfailed": computed["total_expected_xfailed"],
                      "failed": computed["total_failed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
