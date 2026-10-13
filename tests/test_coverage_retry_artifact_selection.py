"""A failed-job retry must combine exactly one same-source artifact per shard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.select_coverage_artifacts import select_artifacts

RUN_ID = 37807749125
SHA = "a" * 40


def _artifact(
    folder: Path, shard: int, attempt: int, *, run_id: int = RUN_ID,
    head_sha: str = SHA,
) -> Path:
    """Write a minimal uniquely identifiable artifact from one shard run."""
    path = folder / f"spacr-coverage-data-{run_id}-{attempt}-{shard}"
    path.mkdir()
    (path / f".coverage.shard-{shard:02d}.batch-001").write_text(
        f"{shard}/{attempt}", encoding="ascii",
    )
    (path / f"spacr-coverage-integrity.shard-{shard:02d}.json").write_text(
        json.dumps({"shard_index": shard}), encoding="utf-8",
    )
    (path / f"spacr-xdist-ledger.shard-{shard:02d}.batch-001.json").write_text(
        json.dumps({"batch": 1}), encoding="utf-8",
    )
    (path / f"spacr-coverage-source.shard-{shard:02d}.json").write_text(
        json.dumps({
            "schema": "spacr.coverage-source/v1",
            "run_id": run_id,
            "run_attempt": attempt,
            "head_sha": head_sha,
            "shard_index": shard,
        }), encoding="utf-8",
    )
    return path


def _select(source: Path, destination: Path, *, attempts: int = 2) -> dict[int, int]:
    """Run the production selector for the twelve-shard workflow contract."""
    return select_artifacts(
        source, destination, run_id=RUN_ID, run_attempt=attempts,
        shard_count=12, head_sha=SHA,
    )


def test_failed_job_retry_uses_eleven_old_and_one_new_shard(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    for shard in range(12):
        _artifact(source, shard, 1)
    _artifact(source, 2, 2)

    selected = _select(source, tmp_path / "selected")

    assert selected == {shard: (2 if shard == 2 else 1) for shard in range(12)}
    files = list((tmp_path / "selected").glob(".coverage.*"))
    assert len(files) == 12
    assert (tmp_path / "selected/.coverage.shard-02.batch-001").read_text() == "2/2"
    assert (tmp_path / "selected/.coverage.shard-03.batch-001").read_text() == "3/1"
    assert (tmp_path / "selected/spacr-xdist-ledger.shard-02.batch-001.json").is_file()
    receipt = json.loads(
        (tmp_path / "selected/spacr-coverage-artifact-selection.json").read_text()
    )
    assert receipt["head_sha"] == SHA
    assert receipt["run_id"] == RUN_ID
    assert receipt["selected_attempts"]["2"] == 2


def test_full_rerun_uses_only_new_shards_without_double_counting(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    for attempt in (1, 2):
        for shard in range(12):
            _artifact(source, shard, attempt)

    selected = _select(source, tmp_path / "selected")

    assert selected == {shard: 2 for shard in range(12)}
    assert len(list((tmp_path / "selected").glob(".coverage.*"))) == 12
    assert all(
        (tmp_path / "selected" / f".coverage.shard-{shard:02d}.batch-001")
        .read_text() == f"{shard}/2"
        for shard in range(12)
    )


@pytest.mark.parametrize("damage, message", [
    ("missing", "missing coverage shard artifacts"),
    ("wrong_source", "coverage source identity mismatch"),
    ("wrong_run", "coverage artifact is outside this run"),
    ("future", "impossible attempt or shard"),
    ("missing_ledger", "incomplete coverage artifact"),
    ("missing_data", "incomplete coverage artifact"),
])
def test_selector_refuses_missing_or_foreign_measurements(tmp_path, damage, message):
    source = tmp_path / "artifacts"
    source.mkdir()
    artifacts = [_artifact(source, shard, 1) for shard in range(12)]
    if damage == "missing":
        for path in artifacts[11].iterdir():
            path.unlink()
        artifacts[11].rmdir()
    elif damage == "wrong_source":
        identity = artifacts[2] / "spacr-coverage-source.shard-02.json"
        value = json.loads(identity.read_text())
        value["head_sha"] = "b" * 40
        identity.write_text(json.dumps(value))
    elif damage == "wrong_run":
        _artifact(source, 2, 2, run_id=RUN_ID + 1)
    elif damage == "future":
        _artifact(source, 2, 3)
    elif damage == "missing_ledger":
        (artifacts[2] / "spacr-coverage-integrity.shard-02.json").unlink()
    elif damage == "missing_data":
        (artifacts[2] / ".coverage.shard-02.batch-001").unlink()

    with pytest.raises(ValueError, match=message):
        _select(source, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_selector_refuses_two_names_for_the_same_shard_attempt(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    for shard in range(12):
        _artifact(source, shard, 1)
    duplicate = source / f"spacr-coverage-data-{RUN_ID}-1-02"
    duplicate.mkdir()

    with pytest.raises(ValueError, match="impossible attempt or shard"):
        _select(source, tmp_path / "selected")


def test_selector_does_not_overwrite_an_existing_coverage_input(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    for shard in range(12):
        _artifact(source, shard, 1)
    destination = tmp_path / "selected"
    destination.mkdir()
    sentinel = destination / ".coverage.shard-00.batch-old"
    sentinel.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="must start empty"):
        _select(source, destination)
    assert sentinel.read_text() == "old"


def test_workflow_binds_each_upload_and_selects_only_this_run():
    """The source identity and selector must wrap the unchanged coverage gate."""
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    shards = workflow["jobs"]["coverage-shards"]
    combine = workflow["jobs"]["coverage-combine"]
    bind = next(step for step in shards["steps"] if step.get("name") ==
                "Bind coverage artifact to this run and source")
    select = next(step for step in combine["steps"] if step.get("name") ==
                  "Select the newest same-run artifact per shard")
    download = next(step for step in combine["steps"] if step.get("uses") ==
                    "actions/download-artifact@v8")

    assert bind["env"]["SPACR_RUN_ID"] == "${{ github.run_id }}"
    assert bind["env"]["SPACR_RUN_ATTEMPT"] == "${{ github.run_attempt }}"
    assert bind["env"]["SPACR_HEAD_SHA"] == "${{ github.sha }}"
    assert "spacr-coverage-source.shard-" in bind["run"]
    assert download["with"]["pattern"] == "spacr-coverage-data-${{ github.run_id }}-*"
    assert download["with"].get("merge-multiple") is not True
    assert "--head-sha \"$SPACR_HEAD_SHA\"" in select["run"]
    assert select["env"]["SPACR_COVERAGE_SHARD_COUNT"] == "12"
    assert combine["needs"] == "coverage-shards"
    assert "coverage-combine" in workflow["jobs"]["release-gate"]["needs"]
