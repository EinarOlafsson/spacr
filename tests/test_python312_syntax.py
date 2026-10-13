"""Every shipped Python source compiles with Python 3.12 warning rules."""
from __future__ import annotations

import warnings
from pathlib import Path


def test_package_has_no_syntax_warnings():
    package = Path(__file__).resolve().parents[1] / "spacr"
    failures = []
    for path in package.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                compile(source, str(path), "exec")
        except (SyntaxError, SyntaxWarning) as exc:
            failures.append(f"{path.relative_to(package)}: {exc}")
    assert failures == []
