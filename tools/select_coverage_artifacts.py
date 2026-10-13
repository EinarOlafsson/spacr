"""Select one coverage artifact per shard across attempts of one GitHub run.

GitHub re-runs use the original run's commit.  A failed-jobs re-run only
uploads new artifacts for the re-run shards, so the combine job must retain
successful shards from earlier attempts of that same run.  The run ID in each
artifact name is an exact identity boundary; a newer attempt wins per shard.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def select_artifacts(
    source: Path, destination: Path, *, run_id: int,
    run_attempt: int, shard_count: int, head_sha: str,
) -> dict[int, int]:
    """Copy the newest same-run artifact for each shard into a flat directory.

    Refuse malformed identities, missing shards, incomplete artifact payloads,
    and filename collisions before coverage.py or the numerical gate runs.
    """
    if run_id < 1 or run_attempt < 1 or shard_count < 1:
        raise ValueError("run ID, attempt, and shard count must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("head SHA must be a full lowercase Git commit ID")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("coverage artifact source must be a directory")
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("coverage artifact source and output must be separate")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("coverage output directory must start empty")

    name = re.compile(rf"spacr-coverage-data-{run_id}-([1-9][0-9]*)-([0-9]+)")
    selected: dict[int, tuple[int, Path]] = {}
    for artifact in source.iterdir():
        if artifact.is_symlink() or not artifact.is_dir():
            raise ValueError(f"invalid coverage artifact directory: {artifact.name}")
        match = name.fullmatch(artifact.name)
        if match is None:
            raise ValueError(f"coverage artifact is outside this run: {artifact.name}")
        attempt, shard = (int(part) for part in match.groups())
        if (attempt > run_attempt or shard >= shard_count
                or str(shard) != match.group(2)):
            raise ValueError(f"coverage artifact has impossible attempt or shard: {artifact.name}")
        current = selected.get(shard)
        if current is not None and attempt == current[0]:
            raise ValueError(f"duplicate coverage artifact for shard {shard} attempt {attempt}")
        if current is None or attempt > current[0]:
            selected[shard] = attempt, artifact
    missing = sorted(set(range(shard_count)) - selected.keys())
    if missing:
        raise ValueError(f"missing coverage shard artifacts: {missing}")

    files: list[tuple[Path, str]] = []
    names: set[str] = set()
    for shard, (attempt, artifact) in sorted(selected.items()):
        ledger = f"spacr-coverage-integrity.shard-{shard:02d}.json"
        identity = f"spacr-coverage-source.shard-{shard:02d}.json"
        coverage_prefix = f".coverage.shard-{shard:02d}.batch-"
        xdist_prefix = f"spacr-xdist-ledger.shard-{shard:02d}.batch-"
        found_ledger = False
        found_identity = False
        found_coverage = False
        for path in artifact.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"nonregular coverage input: {artifact.name}/{path.name}")
            if path.name == ledger:
                found_ledger = True
            elif path.name == identity:
                found_identity = True
                try:
                    recorded = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid coverage source identity: {artifact.name}") from exc
                expected = {
                    "schema": "spacr.coverage-source/v1",
                    "run_id": run_id,
                    "run_attempt": attempt,
                    "head_sha": head_sha,
                    "shard_index": shard,
                }
                if recorded != expected:
                    raise ValueError(f"coverage source identity mismatch: {artifact.name}")
            elif path.name.startswith(coverage_prefix):
                found_coverage = True
            elif path.name.startswith(xdist_prefix) and path.suffix == ".json":
                pass
            else:
                raise ValueError(f"unexpected coverage input: {artifact.name}/{path.name}")
            if path.name in names:
                raise ValueError(f"duplicate coverage filename: {path.name}")
            names.add(path.name)
            files.append((path, path.name))
        if not found_ledger or not found_identity or not found_coverage:
            raise ValueError(f"incomplete coverage artifact: {artifact.name}")

    destination.mkdir(parents=True, exist_ok=True)
    for path, basename in files:
        shutil.copy2(path, destination / basename)
    attempts = {shard: attempt for shard, (attempt, _path) in sorted(selected.items())}
    (destination / "spacr-coverage-artifact-selection.json").write_text(
        json.dumps({
            "run_id": run_id,
            "head_sha": head_sha,
            "current_attempt": run_attempt,
            "selected_attempts": attempts,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return attempts


def main() -> int:
    """Parse workflow identity and select the required shard inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    attempts = select_artifacts(
        args.source, args.destination, run_id=args.run_id,
        run_attempt=args.run_attempt, shard_count=args.shard_count,
        head_sha=args.head_sha,
    )
    print(f"selected coverage shard attempts: {attempts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
