"""A push run parses every committed PowerShell script in 5.1 and 7."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_the_compat_matrix_parses_powershell_with_both_shells():
    workflow = yaml.safe_load((ROOT / ".github/workflows/compat-matrix.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "nightly" in triggers["push"]["branches"]
    job = workflow["jobs"]["powershell-parse"]
    assert job["runs-on"].startswith("windows")
    assert not job.get("continue-on-error")
    steps = [step for step in job["steps"] if "run" in step]
    assert {step["shell"] for step in steps} == {"powershell", "pwsh"}
    for step in steps:
        assert "Parser]::ParseFile" in step["run"]
        assert "Get-ChildItem -Path packaging -Recurse -Filter *.ps1" in step["run"]
        assert "throw" in step["run"]
    assert list(ROOT.glob("packaging/**/*.ps1"))
