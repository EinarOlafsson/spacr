"""Bundle actual GUI navigation and published help for the hosted AI relay."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "services/spacr-first-responder/src/navigation_catalog.json"


def _catalog():
    """Compile shipped routes without reading or changing the user's preferences."""
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QSettings

    with tempfile.TemporaryDirectory(prefix="spacr-ai-navigation-") as directory:
        for scope in (QSettings.UserScope, QSettings.SystemScope):
            for format_ in (QSettings.NativeFormat, QSettings.IniFormat):
                QSettings.setPath(format_, scope, directory)
        from spacr.qt import preferences
        from spacr.qt.ai.prompts import _navigation_modules
        from spacr.qt.preview_registry import PREVIEWS

        preferences.set_show_alpha(True)
        preferences.set_show_beta(True)
        preferences._set_show_alpha_features(True)
        preferences._set_show_alpha_species(True)
        workflows = json.loads((ROOT / "spacr/resources/module_workflows.json").read_text())["modules"]
        modules = []
        for key, name, _description, api, tutorial in _navigation_modules():
            workflow = workflows.get(key, {})
            preview = PREVIEWS.get(key)
            modules.append({"key": key, "name": name, "api": api, "tutorial": tutorial,
                            "preview": preview.title if preview else "",
                            "inputs": workflow.get("inputs", []),
                            "outputs": workflow.get("outputs", [])})
    return {"schema": 1, "modules": modules}


def main():
    """Write the generated relay asset, or verify its canonical source binding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(_catalog(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        return 0 if TARGET.is_file() and TARGET.read_text() == text else 1
    TARGET.write_text(text)
    print(f"written: {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
