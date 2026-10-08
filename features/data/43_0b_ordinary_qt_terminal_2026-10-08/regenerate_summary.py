"""Recount every pytest batch summary in source-bound Qt receipt archives."""

from hashlib import sha256
from pathlib import Path
import gzip
import json
import re
import sys


OUTCOME = re.compile(r"(\d+) (passed|skipped|xfailed|xpassed|failed|errors?)\b")
TIMING = re.compile(r"=+ .*\bin [\d.]+s")


def recount(directory: Path) -> None:
    """Rebuild outcome counts from original compressed logs after hash checks."""
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    run = json.loads((directory / manifest["run_metadata"]).read_text())
    assert run["head_sha"] == manifest["source_sha"]
    assert sha256((directory / manifest["run_metadata"]).read_bytes()).hexdigest() == manifest[
        "run_metadata_sha256"
    ]
    for record in manifest["jobs"]:
        compressed = (directory / record["log"]).read_bytes()
        raw = gzip.decompress(compressed)
        assert sha256(compressed).hexdigest() == record["log_gzip_sha256"]
        assert sha256(raw).hexdigest() == record["log_original_sha256"]
        assert len(raw) == record["log_original_bytes"]
        job = json.loads((directory / record["job_metadata"]).read_text())
        assert job["head_sha"] == manifest["source_sha"] and job["conclusion"] == "success"
        assert sha256((directory / record["job_metadata"]).read_bytes()).hexdigest() == record[
            "job_metadata_sha256"
        ]
        text = raw.decode("utf-8", errors="replace")
        assert manifest["source_sha"] in text
        assert len(re.findall(r"pytest batch \d+/\d+:", text)) == record["batch_headings"]
        summaries = []
        for line in text.splitlines():
            if not TIMING.search(line):
                continue
            outcomes = OUTCOME.findall(line)
            if outcomes:
                summaries.append(outcomes)
        assert not any(kind in ("failed", "error", "errors") for row in summaries
                       for _, kind in row)
        totals = {kind: sum(int(count) for row in summaries for count, name in row
                            if name == kind) for kind in ("passed", "skipped", "xfailed")}
        record["pytest_summary_rows"] = len(summaries)
        record["passed"] = totals["passed"]
        record["skipped"] = totals["skipped"]
        record["expected_xfailed"] = totals["xfailed"]
        print(directory.name, record["shard"], len(summaries), totals)
    for kind, key in (("passed", "total_passed"), ("skipped", "total_skipped"),
                      ("expected_xfailed", "total_expected_xfailed")):
        manifest[key] = sum(record[kind] for record in manifest["jobs"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        recount(Path(path))
