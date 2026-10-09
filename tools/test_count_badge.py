"""Collect hosted pytest outcomes and build a Shields passed/total endpoint."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import uuid

_RECORD = None
_SESSION = None
_ORDER = {"not_run": 0, "passed": 1, "skipped": 2, "failed": 3}


def pytest_configure(config):
    global _RECORD, _SESSION
    directory = os.environ.get("SPACR_TEST_COUNT_DIR")
    if not directory:
        return
    _SESSION = (os.environ["SPACR_TEST_COUNT_SESSION"]
                if hasattr(config, "workerinput") else uuid.uuid4().hex)
    os.environ["SPACR_TEST_COUNT_SESSION"] = _SESSION
    _RECORD = Path(directory) / f"{_SESSION}-{os.getpid()}.jsonl"
    _RECORD.parent.mkdir(parents=True, exist_ok=True)


def _emit(nodeid, outcome):
    if _RECORD is not None:
        with _RECORD.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"session": _SESSION, "nodeid": nodeid,
                                     "outcome": outcome}) + "\n")


def pytest_collection_finish(session):
    if _RECORD is not None:
        for item in session.items:
            _emit(item.nodeid, "not_run")


def pytest_runtest_logreport(report):
    if report.failed:
        _emit(report.nodeid, "failed")
    elif report.skipped:
        _emit(report.nodeid, "skipped")
    elif report.when == "call" and report.passed:
        _emit(report.nodeid, "passed")


def endpoint(directory, *, run_id="", source_sha=""):
    sessions = {}
    for path in sorted(Path(directory).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                outcome = row["outcome"]
                if outcome not in _ORDER:
                    raise ValueError(f"Unknown test outcome: {outcome}")
                key = (row["session"], row["nodeid"])
                prior = sessions.get(key, "not_run")
                sessions[key] = max((prior, outcome), key=_ORDER.__getitem__)
    outcomes = {}
    for (_, nodeid), outcome in sessions.items():
        outcomes.setdefault(nodeid, []).append(outcome)
    total = len(outcomes)
    passed = sum(all(value == "passed" for value in values)
                 for values in outcomes.values())
    color = ("lightgrey" if not total else
             "green" if passed * 100 > total * 90 else
             "yellow" if passed * 100 >= total * 80 else
             "orange" if passed * 100 >= total * 70 else "red")
    return {"schemaVersion": 1, "label": "tests",
            "message": f"{passed}/{total}" if total else "pending",
            "color": color, "passed": passed, "total": total,
            "run_id": str(run_id), "source_sha": source_sha}



def install_readme_badges(root):
    root = Path(root)
    reviewed = {}
    producer = root / "tools/build_documentation_i18n.py"
    if producer.exists():
        for node in ast.parse(producer.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and
                    target.id == "REVIEWED_README_BADGE_ALT_TEXT"
                    for target in node.targets):
                reviewed = ast.literal_eval(node.value)
                break
    url = ("https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com"
           "%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static"
           "%2Ftest-counts.json&cacheSeconds=300")
    block = (f".. |Test counts| image:: {url}\n"
             "   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml\n"
             "   :alt: tests passed/total\n")
    for path in [root / "README.rst", *sorted(
            (root / "docs/i18n/readme").glob("README.*.rst"))]:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^\.\. \|Tests\| image::[^\n]*\n(?:[ \t]+[^\n]*\n)*", "", text)
        text = text.replace("|Tests| |Test counts|", "|Test counts|")
        text = text.replace("|Tests|", "|Test counts|")
        if ".. |Test counts| image::" not in text:
            text = text.replace(".. |Qt| image::", block + ".. |Qt| image::", 1)
        locale = path.name[7:-4] if path.name.startswith("README.") else ""
        if locale in reviewed:
            text = text.replace("   :alt: tests passed/total",
                                f"   :alt: {reviewed[locale][4]}", 1)
        path.write_text(text, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("output")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--install-readme-badges", type=Path)
    args = parser.parse_args()
    if args.install_readme_badges:
        install_readme_badges(args.install_readme_badges)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(endpoint(args.directory, run_id=args.run_id,
                                         source_sha=args.source_sha), indent=2)
                      + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
