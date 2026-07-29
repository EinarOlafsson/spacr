"""Static contracts for lint, typing, complexity, and the release gate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / ".github" / "quality-targets.txt"
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def _quality_targets():
    return [
        line.strip()
        for line in TARGETS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_strict_quality_target_manifest_names_real_python_files():
    targets = _quality_targets()

    assert targets
    assert len(targets) == len(set(targets))
    for relative in targets:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.suffix == ".py", relative


def test_quality_tools_are_contributor_dependencies():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")

    for requirement in (
        "'ruff>=0.9,<1'",
        "'mypy>=1.11,<2'",
        "'xenon>=0.9,<1'",
    ):
        assert requirement in setup


def test_quality_job_runs_all_four_enforced_checks():
    workflow = TESTS_WORKFLOW.read_text(encoding="utf-8")

    assert "\n  quality:" in workflow
    assert "ruff check spacr tests --select E9,F63,F7,F82" in workflow
    assert "ruff check \"${targets[@]}\"" in workflow
    assert "mypy \"${targets[@]}\"" in workflow
    assert "python -m xenon" in workflow
    assert "--max-absolute B" in workflow
    assert "--max-modules B" in workflow
    assert "--max-average A" in workflow


def test_release_gate_requires_every_blocking_job():
    workflow = TESTS_WORKFLOW.read_text(encoding="utf-8")
    expected = (
        "quality",
        "fast",
        "minimum-dependencies",
        "integration",
        "slow",
        "qt",
        "gpu",
        "network",
        "nas",
    )

    assert "\n  release-gate:" in workflow
    gate = workflow.split("\n  release-gate:", 1)[1]
    for job in expected:
        assert f"      - {job}" in gate
    assert 'details["result"] != "success"' in gate
