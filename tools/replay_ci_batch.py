#!/usr/bin/env python3
"""Replay one CI batch locally, using the workflow and runners from that source.

List batches with --suite fast|qt|coverage --shard N --list. Execute a
1-based --batch N with --output NEW_DIRECTORY. --repo can point to an
isolated checkout of the failing CI SHA; --expect-sha refuses a wrong SHA.
Output streams immediately and is saved alongside the source/environment
manifest and result. This is a single-batch diagnostic, not a suite verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time


def _load_runner(root, coverage):
    name = "run_coverage_batches" if coverage else "run_pytest_batches"
    spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_options(job, script):
    matches = []
    for step in job.get("steps", []):
        command = step.get("run", "").replace("\\\n", " ")
        for line in command.splitlines():
            tokens = shlex.split(line) if line.strip().startswith(f"python tools/{script} ") else []
            if tokens:
                matches.append(tokens[2:])
    if len(matches) != 1:
        raise ValueError(f"expected one workflow command for {script}, found {len(matches)}")
    return matches[0]


def _configuration(root, suite, shard, output):
    import yaml

    workflow = yaml.safe_load((root / ".github/workflows/tests.yml").read_text())
    job = workflow["jobs"]["coverage-shards" if suite == "coverage" else suite]
    runner = _load_runner(root, suite == "coverage")
    replacements = {}
    if suite == "qt":
        reusable = yaml.safe_load((root / ".github/workflows/_pytest-suite.yml").read_text())
        inputs = reusable.get("on", reusable.get(True))["workflow_call"]["inputs"]
        values = {key: value.get("default") for key, value in inputs.items()}
        values.update(job["with"])
        replacements = {f"${{{{ inputs.{key} }}}}": str(value) for key, value in values.items()}
        replacements["$batch_workers"] = str(max(1, int(values["xdist_workers"])))
        options = _runner_options(reusable["jobs"]["pytest"], "run_pytest_batches.py")
        count = int(values["file_shard_count"])
        python_version = values["python_version"]
    else:
        script = "run_coverage_batches.py" if suite == "coverage" else "run_pytest_batches.py"
        options = _runner_options(job, script)
        replacements = {
            "$SPACR_COVERAGE_SHARD": str(shard),
            "$SPACR_COVERAGE_DATA_DIR": str(output / "coverage"),
        }
        count = None
        if suite == "fast":
            step = next(step for step in job["steps"] if step.get("name") == "Run fast tests")
            count = int(step["env"]["SPACR_PYTEST_FILE_SHARD_COUNT"])
        python_version = next(step["with"]["python-version"] for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
    for old, new in replacements.items():
        options = [token.replace(old, new) for token in options]
    if any("${{" in token or "$" in token for token in options):
        raise ValueError("workflow runner has an unresolved expression")
    args = runner.build_parser().parse_args(options)
    if suite == "coverage":
        count = args.shard_count
    if not 0 <= shard < count:
        raise ValueError(f"shard must be within [0, {count})")
    return runner, args, options, count, str(python_version)


def _partition(root, suite, runner, args, shard, count):
    if suite == "coverage":
        excluded = {Path(value).as_posix() for value in args.exclude_file}
        files = [path for path in runner._test_files(root, args.paths)
                 if path.relative_to(root).as_posix() not in excluded
                 and runner._shard(path, root, count) == shard]
        return [[path.relative_to(root).as_posix() for path in batch]
                for batch in runner._batches(files, args.batch_size)]
    ignored = [Path(path).resolve() for path in args.ignore]
    files = [path for path in runner._test_files(args.paths)
             if not any(Path(path).resolve() == excluded
                        or excluded in Path(path).resolve().parents
                        for excluded in ignored)]
    return runner._batches(files, args.batch_size)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(root, files):
    sources = [".github/workflows/tests.yml", ".github/workflows/_pytest-suite.yml",
               "tools/run_pytest_batches.py", "tools/run_coverage_batches.py",
               "tools/pytest_plugins/xdist_coverage_ledger.py", "tools/run_capped.sh",
               "tests/conftest.py", "pytest.ini", ".coveragerc"]
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    return {
        "sha": _git(root, "rev-parse", "HEAD"),
        "working_tree": _git(root, "status", "--porcelain", "--untracked-files=no"),
        "untracked_sources": {path: _hash(root / path) for path in untracked
                              if Path(path).suffix != ".pyc"
                              and not {"__pycache__", ".pytest_cache"}.intersection(Path(path).parts)},
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output([
            "git", "-C", str(root), "diff", "HEAD", "--", "spacr", "tests", "tools", ".github", ".coveragerc", "pytest.ini",
        ])).hexdigest(),
        "files": [{"path": path, "sha256": _hash(root / path)} for path in files],
        "configuration": {path: _hash(root / path) for path in sources if (root / path).is_file()},
    }


def _write(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _environment(root, output, suite, shard, count, disabled_plugins=()):
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen",
        "CI": "true", "GITHUB_ACTIONS": "true",
        "MPLBACKEND": "Agg", "PYTHONPATH": str(root), "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "SPACR_TEST_MEMORY_GB": "6",
        "SPACR_PYTEST_FILE_SHARD_INDEX": str(shard if suite != "coverage" else 0),
        "SPACR_PYTEST_FILE_SHARD_COUNT": str(count if suite != "coverage" else 1),
        "NUMBA_CACHE_DIR": str(output / "numba-cache"),
        "XDG_CACHE_HOME": str(output / "cache"),
        "XDG_CONFIG_HOME": str(output / "config"),
    })
    environment.pop("PYTEST_ADDOPTS", None)
    if disabled_plugins:
        environment["PYTEST_ADDOPTS"] = shlex.join([part for name in disabled_plugins for part in ("-p", f"no:{name}")])
    environment.pop("COVERAGE_FILE", None)
    environment.pop("SPACR_COVERAGE_LEDGER", None)
    environment.pop("SPACR_HF_E2E_STUB", None)
    if suite == "coverage":
        environment["SPACR_HF_E2E_STUB"] = "1"
    return environment


def _execute(cli, root, runner, args, options, files):
    manifest = json.loads((cli.output / "manifest.json").read_text())
    if _identity(root, files) != manifest["source"] or _hash(Path(__file__)) != manifest["helper_sha256"]:
        raise ValueError("source changed after planning the batch")
    if cli.suite == "coverage":
        args.data_dir.mkdir()
        entry = runner._run_batch(args, cli.batch, files)
        _write(cli.output / "coverage-batch.json", entry)
        status = entry["exit_code"]
        return (0 if status == 5 else status) or (1 if entry["unrecovered_files"] else 0)
    first_option = next(i for i, value in enumerate(options) if value.startswith("--"))
    return runner.main([*files, *options[first_option:]])


def _owned_scope(pid):
    try:
        child = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::")[-1]
        parent = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    except OSError:
        return None
    unit = Path(child).name
    if child != parent and re.fullmatch(r"run-[a-zA-Z0-9]+\.scope", unit):
        return unit, child
    return None


def _stop_child(process, scope):
    descendants = {process.pid}
    parents = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            match = re.search(r"^PPid:\s+(\d+)", status.read_text(), re.MULTILINE)
            if match:
                parents[int(status.parent.name)] = int(match.group(1))
        except OSError:
            continue
    while True:
        found = {pid for pid, parent in parents.items() if parent in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    scope = scope or _owned_scope(process.pid)
    if scope:
        try:
            actual = subprocess.check_output(["systemctl", "--user", "show", "--property=ControlGroup", "--value", scope[0]], text=True, timeout=5).strip()
            if actual != scope[1]:
                scope = None
        except (OSError, subprocess.SubprocessError):
            scope = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if scope:
            try:
                subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all", f"--signal={sig.name}", scope[0]], check=False, timeout=5, capture_output=True)
            except (OSError, subprocess.SubprocessError):
                pass
        for pid in sorted(descendants, reverse=True):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", choices=("fast", "qt", "coverage"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--batch", type=int, help="original CI batch number, starting at 1")
    parser.add_argument("--list", action="store_true", help="list batches/files without executing tests")
    parser.add_argument("--output", type=Path, help="new scratch directory for log, manifest and result")
    parser.add_argument("--expect-sha", help="refuse any other source commit (full SHA or unique prefix)")
    parser.add_argument("--allow-dirty", action="store_true", help="explicitly permit edited source with --expect-sha; changes are recorded")
    parser.add_argument("--disable-plugin", action="append", default=[], metavar="NAME", help="disable an extra local pytest plugin (e.g. randomly); recorded in the manifest")
    parser.add_argument("--_execute", action="store_true", help=argparse.SUPPRESS)
    cli = parser.parse_args(argv)
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None for name in cli.disable_plugin):
        parser.error("plugin exclusions must be pytest plugin names")
    root = cli.repo.resolve()
    output_supplied = cli.output is not None
    cli.output = (cli.output or Path("unused-local-replay-output")).resolve()
    os.chdir(root)
    sha = _git(root, "rev-parse", "HEAD")
    if cli.expect_sha:
        try:
            expected = _git(root, "rev-parse", f"{cli.expect_sha}^{{commit}}")
        except subprocess.CalledProcessError:
            parser.error(f"unknown source commit: {cli.expect_sha}")
        if expected != sha:
            parser.error(f"source SHA is {sha}, not {cli.expect_sha}")
        if not cli.allow_dirty:
            identity = _identity(root, [])
            if identity["working_tree"] or identity["untracked_sources"]:
                parser.error("--expect-sha requires clean source; use --allow-dirty for a recorded local repair")
    runner, args, options, count, python_version = _configuration(root, cli.suite, cli.shard, cli.output)
    batches = _partition(root, cli.suite, runner, args, cli.shard, count)
    if cli.batch is not None and not 1 <= cli.batch <= len(batches):
        parser.error(f"batch must be within [1, {len(batches)}]")
    print(f"{sha}: {cli.suite} shard {cli.shard}/{count}, {len(batches)} CI batches", flush=True)
    if cli.suite == "qt":
        print("Functional batches only; the separate Qt serial measurement tail is excluded.", flush=True)
    if cli.list:
        if cli.batch:
            print("\n".join(batches[cli.batch - 1]))
        else:
            for number, files in enumerate(batches, 1):
                print(f"batch {number}: {len(files)} files; {files[0]} ... {files[-1]}")
        return 0
    if cli.batch is None or not output_supplied:
        parser.error("execution requires --batch and --output; use --list to inspect")
    if cli.output == root or root in cli.output.parents:
        parser.error("--output must be outside the repository")
    files = batches[cli.batch - 1]
    if cli._execute:
        return _execute(cli, root, runner, args, options, files)
    if cli.suite == "coverage" and importlib.util.find_spec("coverage") is None:
        parser.error("coverage.py is required to execute a coverage replay; --list remains available")
    if args.per_test_timeout > 0 and importlib.util.find_spec("pytest_timeout") is None:
        parser.error("pytest-timeout is required to preserve the CI per-test time limit")
    cli.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("pytest", "pytest-xdist", "pytest-timeout", "pytest-cov", "coverage", "pytest-qt", "pytest-randomly", "PySide6", "shiboken6", "numpy", "matplotlib", "torch", "cellpose"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema": "spacr.local-ci-batch/v1", "scope": "one batch; not a full-suite or coverage-gate verdict",
        "suite": cli.suite, "shard": cli.shard, "shard_count": count,
        "batch": cli.batch, "batches_total": len(batches), "marker": args.marker,
        "runner_options": options, "source": _identity(root, files),
        "helper_sha256": _hash(Path(__file__)), "python": platform.python_version(),
        "python_executable": sys.executable, "workflow_python": python_version,
        "packages": versions, "environment_parity": "not established; compare versions with the hosted job",
        "memory_cap": "8G", "per_worker_memory_guard_gb": 6,
        "qt_serial_tail": "excluded" if cli.suite == "qt" else "not applicable",
        "disabled_plugins": cli.disable_plugin, "ci_environment": {"CI": "true", "GITHUB_ACTIONS": "true"},
        "pytest_plugin_entry_points": {entry.name: entry.value for entry in importlib.metadata.entry_points(group="pytest11")},
    }
    _write(cli.output / "manifest.json", manifest)
    (cli.output / "replay-helper.py").write_bytes(Path(__file__).read_bytes())
    command = ["bash", str(root / "tools/run_capped.sh"), "8G", sys.executable,
               str(Path(__file__).resolve()), "--repo", str(root), "--suite", cli.suite,
               "--shard", str(cli.shard), "--batch", str(cli.batch),
               "--output", str(cli.output), "--expect-sha", sha, "--_execute"]
    if not cli.expect_sha or cli.allow_dirty:
        command.append("--allow-dirty")
    for name in cli.disable_plugin:
        command.extend(["--disable-plugin", name])
    print(f"Replaying CI batch {cli.batch}; live log: {cli.output / 'pytest.log'}", flush=True)
    started = time.monotonic()
    failures = set()
    process = None
    scope = None
    error = None
    cleanup_complete = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        process = subprocess.Popen(command, cwd=root, env=_environment(root, cli.output, cli.suite, cli.shard, count, cli.disable_plugin),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        with (cli.output / "pytest.log").open("w") as log:
            for line in process.stdout:
                scope = scope or _owned_scope(process.pid)
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                match = re.search(r"\b(?:FAILED|ERROR)\s+(tests/\S+)", line)
                if match:
                    failures.add(match.group(1))
                    _write(cli.output / "failures.json", sorted(failures))
        status = process.wait()
    except BaseException as exc:
        error = repr(exc)
        status = 128 + (exc.args[0] if exc.args and isinstance(exc.args[0], int) else signal.SIGINT) if isinstance(exc, KeyboardInterrupt) else 1
        if process is not None:
            cleanup_complete = _stop_child(process, scope)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if process is not None and process.stdout is not None:
            process.stdout.close()
    try:
        helper_changed = _hash(Path(__file__)) != manifest["helper_sha256"]
        changed = _identity(root, files) != manifest["source"] or helper_changed
    except OSError as exc:
        helper_changed = not Path(__file__).is_file()
        changed = True
        error = error or repr(exc)
    status = (128 - status if status < 0 else status) or (2 if changed else 0)
    _write(cli.output / "result.json", {"exit_code": status, "elapsed_seconds": time.monotonic() - started,
                                       "failed_test_ids": sorted(failures), "source_changed": changed,
                                       "helper_changed": helper_changed, "interruption": error,
                                       "cleanup_complete": cleanup_complete,
                                       "scope": manifest["scope"]})
    print(f"Local batch exit {status}; evidence: {cli.output}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
