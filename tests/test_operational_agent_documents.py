"""Operational instructions live only in their canonical, unpackaged folder."""
from __future__ import annotations

import ast
import os
from pathlib import Path

from setuptools import Distribution
from setuptools._distutils.filelist import FileList
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "spacr" / "resources" / "AI"


def test_agent_instructions_exist_only_in_the_canonical_folder():
    assert (AI / "AGENTS.md").is_file()
    assert "spacr/resources/AI/HANDOFF.md" in (AI / "AGENTS.md").read_text()
    assert not (ROOT / "AGENTS.md").exists()
    assert not (ROOT / "CLAUDE.md").exists()
    assert not list((ROOT / "features").glob("HANDOFF*.md"))


def test_canonical_handoffs_retain_complete_documents():
    canonical = set(AI.glob("HANDOFF*.md"))
    assert len(canonical) >= 4
    for target in canonical:
        assert len(target.read_text()) > 1000, target
    assert "CURRENT HANDOFF FOR CODEX AND OTHER AGENTS" in (AI / "HANDOFF.md").read_text()


def _wheel_exclusions():
    tree = ast.parse((ROOT / "setup.py").read_text())
    setup = next(node for node in tree.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "setup").value
    return ast.literal_eval(next(arg.value for arg in setup.keywords
                                if arg.arg == "exclude_package_data"))


def test_wheel_data_selection_excludes_agent_documents_even_under_a_broad_include(tmp_path):
    package = tmp_path / "spacr"
    operational = {"resources/AI/AGENTS.md", "resources/AI/archive/handoff.md"}
    product = {"resources/model_cards/card.json", "resources/data/example.csv"}
    for name in operational | product:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture")
    distribution = Distribution({"packages": ["spacr"],
                                 "package_data": {"spacr": ["resources/**/*"]},
                                 "exclude_package_data": _wheel_exclusions()})
    command = build_py(distribution)
    command.ensure_finalized()
    command.manifest_files = {}
    selected = {Path(name).relative_to(package).as_posix()
                for name in command.find_data_files("spacr", str(package))}
    assert not operational & selected
    assert product <= selected


def test_source_manifest_removes_operational_files_from_vcs_candidates():
    operational = {"AGENTS.md", "CLAUDE.md", "features/HANDOFF.md",
                   "features/HANDOFF_2026-10-13_claude_home_to_codex.md",
                   "spacr/resources/AI/AGENTS.md", "spacr/resources/AI/archive/handoff.md"}
    product = {"CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "SECURITY.md",
               "spacr/resources/model_cards/card.json", "spacr/resources/data/example.csv"}
    files = FileList()
    files.set_allfiles(sorted(os.path.normpath(name) for name in operational | product))
    files.files = sorted(os.path.normpath(name) for name in operational | product)
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            files.process_template_line(line)
    selected = {Path(name).as_posix() for name in files.files}
    assert not operational & selected
    assert product <= selected
