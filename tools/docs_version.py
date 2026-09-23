"""Read documentation version metadata from the checkout being published."""

import ast
from pathlib import Path


def _literal_version(path: Path, name: str) -> str:
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"Expected one nonempty literal {name} in {path}")
    return values[0]


def source_version(root: Path) -> str:
    """Require setup and package versions to agree without importing spaCR."""
    setup = _literal_version(root / "setup.py", "VERSION")
    package = _literal_version(root / "spacr/_version.py", "__version__")
    if setup != package:
        raise ValueError(f"Documentation source versions disagree: {setup} != {package}")
    return setup
