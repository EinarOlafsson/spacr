"""Build the source-bound exact-224 ordinary CI failure archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRATCH = Path("/mnt/wd4tb/scratch/ci-224-failures-20261007")
ROOT = Path("/mnt/firecuda2/Claude/repo/spacr")
INTEGRATED_LOG = Path(
    "/mnt/wd4tb/scratch/ci-final-repairs-20261006/sorting-retry-integrated.log"
)
FAILED_SOURCE = "22413aa4d072fe4eb06303cf82d0c6fbfe677d61"
REPAIR_SOURCE = "aee25140049788161edd047e59bc9cda6927e810"
FILES = (
    "spacr/qt/widgets/timelapse_preview.py",
    "tests/qt/test_timelapse_event_annotation_editor.py",
    "tests/qt/test_every_table_asks_for_the_sorting_contract.py",
    "tests/test_cov_utils_masks_misc.py",
    "spacr/qt/widgets/live_preview.py",
    "spacr/qt/widgets/ambient.py",
)

TRIAGE = {
    "source_sha": FAILED_SOURCE,
    "fixed_in_later_cpu_checkpoint": {
        "F567 HiDPI preview": [112526287755, 112526287973],
        "Make Masks empty-field mask hash": [112526287801, 112526287973],
        "advection test matrix population": [112526287801, 112526287973],
        "Python 3.9 fungal pairwise test import": [112526287581, 112526287731, 112526287784],
        "spaCR archive name spelling": [112526287726, 112526287784],
        "Make Masks toolbar actual-width layout": [112526287617, 112526288044],
        "F567 sorted event table identity and cell pairing": [112526287791, 112526288795],
        "model retry test captures unrelated worker sleeps": [112526287801],
    },
    "still_needs_independent_resolution": {
        "fungal young-scene visibility and motion": [112526287617, 112526287924],
    },
    "workstation_generated_assets_pending": {
        "reviewed runtime and UI translations": [112526287581, 112526287605, 112526287612, 112526287617, 112526287671, 112526287684, 112526287731, 112526287755, 112526287784, 112526287791, 112526287883, 112526287924, 112526288795],
        "API and private callable inventories and help index": [112526287612, 112526287680, 112526287731, 112526287784, 112526287791, 112526287801, 112526287883],
        "settings-flow and reviewed settings tables": [112526287612, 112526287684, 112526287731, 112526287801],
        "notebook and README source-current prose": [112526287581, 112526287680, 112526287784, 112526287791],
    },
    "downstream_prerequisite_failures": {
        "coverage combine refused failed shards, no numeric ratchet verdict": [112553274190],
        "release gate refused failed prerequisites after ordinary cancellation": [112567471589],
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True
    ).strip()


def build(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    before = json.loads((SCRATCH / "jobs-pre-cancel.json").read_text())
    after = json.loads((SCRATCH / "jobs-post-cancel.json").read_text())
    run = json.loads((SCRATCH / "run-pre-cancel.json").read_text())
    assert run["head_sha"] == FAILED_SOURCE
    failed = [job for job in before["jobs"] if job["conclusion"] == "failure"]
    assert len(failed) == 19 and before["total_count"] == 28
    assert after["total_count"] == 29
    assert sum(job["conclusion"] == "failure" for job in after["jobs"]) == 20
    assert (SCRATCH / "cancel-response.txt").read_text().startswith(
        "HTTP/2.0 202 Accepted"
    )

    names = [
        "run-pre-cancel.json", "jobs-pre-cancel.json",
        "run-post-cancel.json", "jobs-post-cancel.json",
    ]
    names += [f"{job['id']}.log.gz" for job in failed]
    names += ["112567471589.log.gz"]
    for name in names:
        shutil.copyfile(SCRATCH / name, dest / name)
    (dest / "cancel-response.txt.gz").write_bytes(
        gzip.compress(
            (SCRATCH / "cancel-response.txt").read_bytes(),
            compresslevel=9,
            mtime=0,
        )
    )
    (dest / "cancel-response.txt").unlink(missing_ok=True)
    (dest / "sorting-retry-integrated.log.gz").write_bytes(
        gzip.compress(INTEGRATED_LOG.read_bytes(), compresslevel=9, mtime=0)
    )
    (dest / "triage.json").write_text(
        json.dumps(TRIAGE, indent=2, sort_keys=True) + "\n"
    )
    node_receipt = json.loads((SCRATCH / "MANIFEST.json").read_text())
    assert sorted(job["id"] for job in node_receipt["failed_jobs"]) == sorted(
        job["id"] for job in failed
    )
    (dest / "failed_nodes.json").write_text(
        json.dumps({
            "source_sha": FAILED_SOURCE,
            "jobs": {
                str(job["id"]): job["failed_nodes"]
                for job in node_receipt["failed_jobs"]
            },
        }, indent=2, sort_keys=True) + "\n"
    )
    (dest / "README.md").write_text(
        "# Exact-224 ordinary CI failure receipt\n\n"
        "Ordinary run 37537358238 tested Git source `" + FAILED_SOURCE + "`. "
        "The pre-cancel snapshot has 19 failed jobs, seven successful jobs and "
        "two running jobs. Every one of those 19 full failed logs is retained. "
        "After the authorized cancellation, the release gate became the 20th "
        "failed job because its blocking jobs had failed; its full log is "
        "retained separately. The two remaining Fast jobs were cancelled, "
        "not counted as test passes or test failures. Only this superseded "
        "ordinary run was cancelled; protected serial runs were untouched.\n\n"
        "`triage.json` separates CPU repairs, ongoing fungal visibility, "
        "workstation-owned generated artifacts and downstream failures. "
        "A job can have more than one category. The coverage-combine job "
        "refused failed shard tests before evaluating the absolute-count "
        "ratchet, so this run supplies no numerical coverage verdict.\n\n"
        "The focused sorting/retry receipt is from integrated source `"
        + REPAIR_SOURCE + "`: 81 passed in 24.52 s under 4 GiB. "
        "`manifest.json` binds exact Git file blobs for both source versions "
        "and SHA-256 of every archived byte; `verify_archive.py` checks all "
        "payload hashes without network access.\n",
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__), dest / "build_archive.py")
    (dest / "verify_archive.py").write_text(
        "from __future__ import annotations\n"
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        "manifest = json.loads((root / 'manifest.json').read_text())\n"
        "for name, record in manifest['payloads'].items():\n"
        "    data = (root / name).read_bytes()\n"
        "    assert len(data) == record['bytes'], name\n"
        "    assert hashlib.sha256(data).hexdigest() == record['sha256'], name\n"
        "print(len(manifest['payloads']), 'source-bound archive payloads verified')\n",
        encoding="utf-8",
    )
    payloads = {}
    for path in sorted(dest.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        data = path.read_bytes()
        record = {"bytes": len(data), "sha256": sha(data)}
        if path.suffix == ".gz":
            raw = gzip.decompress(data)
            record.update(raw_bytes=len(raw), raw_sha256=sha(raw))
        payloads[path.name] = record
    manifest = {
        "schema": 1,
        "run_id": 37537358238,
        "failed_source_sha": FAILED_SOURCE,
        "integrated_repair_source_sha": REPAIR_SOURCE,
        "pre_cancel_failed_job_ids": sorted(job["id"] for job in failed),
        "post_cancel_downstream_release_gate_job_id": 112567471589,
        "source_blobs": {
            sha_id: {path: blob(sha_id, path) for path in FILES}
            for sha_id in (FAILED_SOURCE, REPAIR_SOURCE)
        },
        "payloads": payloads,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    build(Path(sys.argv[1]))
