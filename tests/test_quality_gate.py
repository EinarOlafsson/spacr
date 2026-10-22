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
    import ast
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    extras = next(ast.literal_eval(keyword.value)
                  for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "id", "") == "setup"
                  for keyword in node.keywords
                  if keyword.arg == "extras_require")
    core = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", "") == "dependencies"
                        for target in node.targets))
    runtime = {Requirement(value).name for value in core}
    dev = [Requirement(value) for value in extras["dev"]]
    for name, bounds in (("ruff", ">=0.9,<1"),
                         ("mypy", ">=1.11,<3"),
                         ("xenon", ">=0.9,<1")):
        matches = [requirement for requirement in dev if requirement.name == name]
        assert len(matches) == 1, (name, matches)
        assert matches[0].specifier == SpecifierSet(bounds)
        assert matches[0].marker is None
        assert name not in runtime


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
