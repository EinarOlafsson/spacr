"""Bind and run one original hosted-worker context without editing its tests."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


def source_guard(context):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    assert actual == context["source_sha"], "source checkout SHA"
    tree = subprocess.check_output(["git", "rev-parse", "HEAD:spacr"]).decode().strip()
    assert tree == context["spacr_tree"], "application tree"
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "spacr", "tests", "tools", ".github", "setup.py", "pyproject.toml", "pytest.ini", ".coveragerc"], check=True)
    for name, expected_tree in context["trees"].items():
        assert subprocess.check_output(["git", "rev-parse", "HEAD:" + name]).decode().strip() == expected_tree, name
    for path, row in context["source_bindings"].items():
        raw = Path(path).read_bytes()
        assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], path
    assert len(context["nodes"]) == len(set(context["nodes"])) == 104
    assert len(context["modules"]) == 6
    assert list(dict.fromkeys(x.split("::", 1)[0] for x in context["nodes"])) == context["modules"]
    return {"source_sha": actual, "spacr_tree": tree, "source_binding_count": len(context["source_bindings"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bind", "run"))
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    binding = source_guard(context)
    args.evidence.mkdir(parents=True, exist_ok=True)
    if args.mode == "bind":
        from PySide6 import __version__ as pyside_version
        from PySide6.QtCore import qVersion

        expected = {"PySide6": "6.12.0", "shiboken6": "6.12.0", "pytest": "8.4.2",
                    "pytest-qt": "4.5.0", "pytest-xdist": "3.8.0", "pytest-timeout": "2.4.0",
                    "pluggy": "1.6.0", "hypothesis": "6.169.1", "anyio": "4.15.1",
                    "platformdirs": "4.13.0"}
        versions = {name: importlib.metadata.version(name) for name in expected}
        assert versions == expected and platform.python_version() == "3.12.15"
        assert pyside_version == qVersion() == "6.12.0"
        plugins = {entry.name: entry.value for entry in importlib.metadata.entry_points(group="pytest11")}
        expected_plugins = {"hypothesispytest": "_hypothesis_pytestplugin",
            "platformdirs": "platformdirs.pytest_plugin", "pytest-qt": "pytestqt.plugin",
            "anyio": "anyio.pytest_plugin", "xdist": "xdist.plugin",
            "xdist.looponfail": "xdist.looponfail", "timeout": "pytest_timeout"}
        assert plugins == expected_plugins, "unexpected pytest entrypoint set"
        receipt = {**binding, "python": platform.python_version(), "Qt_runtime": qVersion(),
                   "versions": versions, "pytest_entrypoints": plugins,
                   "scope": context["scope"], "diagnostic_only": True,
                   "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                   "workflow_sha": os.environ.get("GITHUB_SHA")}
        (args.evidence / "source-runtime-binding.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], check=True,
                       stdout=(args.evidence / "installed-packages.json").open("w"))
        return 0
    expected_binding = json.loads((args.evidence / "source-runtime-binding.json").read_text(encoding="utf-8"))
    assert expected_binding["source_sha"] == binding["source_sha"]
    command = [sys.executable, "-m", "pytest", *context["nodes"],
        "-m", "qt and not slow and not heavy and not gpu and not network and not nas and not gui",
        "-n", "1", "--dist", "loadfile", "--max-worker-restart=0", "--timeout", "1200",
        "--timeout-method", "thread", "-o", "faulthandler_timeout=900", "-v", "--tb=short",
        "-p", "tools.pytest_plugins.qt_serial_rss_journal", "-p", "qt_failure_images",
        "--junitxml=" + str(args.evidence / "original-context.xml")]
    (args.evidence / "launch.json").write_text(json.dumps({"command": command, **binding,
        "scope": context["scope"], "timeout_seconds": 2700, "retries": 0,
        "post_failure_capture_only": True, "original_concurrent_gw1_reproduced": False,
        "workflow_sha": expected_binding["workflow_sha"],
        "native_child_GITHUB_SHA_is_actual_test_checkout": context["source_sha"]}, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(Path.cwd()))
    from tools.run_pytest_batches import _run_bounded
    child_environment = os.environ.copy()
    child_environment["SPACR_DIAGNOSTIC_WORKFLOW_SHA"] = expected_binding["workflow_sha"]
    child_environment["GITHUB_SHA"] = context["source_sha"]
    original = _run_bounded(command, timeout=2700, env=child_environment)
    receipt = {**binding, "actual_original_pytest_exit": original.returncode,
               "diagnostic_only": True, "full_ci_acceptance": False,
               "retries": 0}
    junit = args.evidence / "original-context.xml"
    if junit.is_file():
        root = ET.parse(junit).getroot()
        suites = list(root.iter("testsuite"))
        receipt["junit"] = {name: sum(int(x.get(name, "0")) for x in suites)
                            for name in ("tests", "failures", "errors", "skipped")}
        receipt["original_104_all_executed"] = receipt["junit"]["tests"] == 104 and receipt["junit"]["skipped"] == 0
    count_files = list(Path(os.environ["SPACR_TEST_COUNT_DIR"]).glob("*.jsonl"))
    worker_sequences = []
    for path in count_files:
        rows = [json.loads(line) for line in path.open()]
        actual = [row["nodeid"] for row in rows if row["outcome"] != "not_run"]
        if actual == context["nodes"]:
            pid = int(path.stem.rsplit("-", 1)[1])
            identities = list(Path(os.environ["SPACR_QT_NATIVE_EVIDENCE_DIR"]).glob("process-" + str(pid) + "-*.jsonl"))
            for identity in identities:
                first = json.loads(identity.open().readline())
                if first.get("worker") == "gw0" and first.get("source_sha") == context["source_sha"]:
                    worker_sequences.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "pid": pid, "worker": "gw0", "source_sha": first["source_sha"],
                        "native_identity_path": str(identity), "native_identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest()})
    receipt["exact104_sequence_journals"] = worker_sequences
    receipt["exact104_sequence_confirmed"] = bool(worker_sequences)
    try:
        source_guard(context)
        receipt["source_guard_after_pass"] = True
    except Exception as error:
        receipt["source_guard_after_pass"] = False
        receipt["source_guard_error_type"] = type(error).__name__
    integrity = receipt.get("original_104_all_executed", False) and bool(worker_sequences) and receipt["source_guard_after_pass"]
    receipt["context_integrity_pass"] = integrity
    diagnostic_exit = original.returncode if original.returncode != 0 else 0 if integrity else 3
    receipt["actual_diagnostic_exit"] = diagnostic_exit
    (args.evidence / "terminal.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (args.evidence / "actual-original-pytest.exit").write_text(str(original.returncode) + "\n", encoding="utf-8")
    return diagnostic_exit


if __name__ == "__main__":
    raise SystemExit(main())
