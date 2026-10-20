"""Check and publish explicitly prepared items without calling a model."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

WORKTREES = Path("/mnt/wd4tb/spacr-worktrees")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True,
                                   stderr=subprocess.STDOUT, timeout=120).rstrip("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(job):
    root = Path(job["root"]).resolve()
    if not root.is_relative_to(WORKTREES.resolve()) or root == WORKTREES.resolve():
        raise ValueError("Use an isolated worktree under the authorized worktree root")
    if job.get("completion_approved") is not True:
        raise ValueError("Full item scope must have been reviewed before enqueueing")
    if git(root, "rev-parse", "HEAD") != job["base"]:
        raise ValueError("Prepared source no longer matches HEAD")
    if git(root, "branch", "--show-current") in {"", "main", "nightly"}:
        raise ValueError("Use a private named worktree branch")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
        raise ValueError("Index already contains changes")
    for name, expected in job["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or Path(name).is_absolute() or ".git" in Path(name).parts:
            raise ValueError("File outside prepared source")
        if digest(path) != expected:
            raise ValueError("Prepared file changed: " + name)
    changed = set()
    for record in git(root, "status", "--porcelain", "-uall").splitlines():
        if " -> " in record:
            raise ValueError("Prepare renames separately")
        changed.add(record[3:].strip('"'))
    if not changed or changed != set(job["files"]):
        raise ValueError("Changes must match the exact prepared file allowlist")
    if job["item"] not in job["files"]:
        raise ValueError("Item must belong to the prepared file allowlist")
    source = (root / job["item"]).read_text()
    if not re.search(
            r"^Status:\s*(?:COMPLETE|DONE)\s+100%(?:\s|$)", source, re.I | re.M):
        raise ValueError("Prepared item must explicitly record full completion")
    if not job.get("tests"):
        raise ValueError("Explicit owning test cases are required")
    for node in job["tests"]:
        path = node.split("::", 1)[0]
        if not path.startswith("tests/") or not path.endswith(".py") or not (root / path).resolve().is_relative_to(root):
            raise ValueError("Select test files or nodes, never a full suite")
    if re.search(r"Co-Authored-By:|AI[- ](?:generated|trailer)", job["message"], re.I):
        raise ValueError("No authorship trailers")
    return root


def counts(path):
    suites = list(ET.parse(path).iter("testsuite"))
    result = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ["tests", "failures", "errors", "skipped"]}
    if not result["tests"] or any(result[key] for key in ["failures", "errors", "skipped"]):
        raise ValueError("Checks must actually run and pass, without skipped cases")
    return result


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process(manifest, output):
    result = output / (manifest.stem + ".json")
    if result.exists():
        prior = json.loads(result.read_text())
        if prior["status"] == "committed":
            try:
                root = Path(json.loads(manifest.read_text())["root"])
                if git(root, "rev-parse", "HEAD") != prior["commit"] or git(root, "status", "--porcelain"):
                    raise ValueError("Committed checkpoint changed; do not retry publication")
                git(root, "push", "origin", "HEAD:nightly")
                prior["status"] = "published"
            except Exception as exc:
                prior.update(status="blocked", error=str(exc))
            save(result, prior)
            return
        if prior["status"] != "running":
            return
        pid = prior.get("check_pid")
        if pid and Path(f"/proc/{pid}/cmdline").exists():
            return
    state = dict(job=str(manifest), status="running", started=time.time())
    save(result, state)
    try:
        job = json.loads(manifest.read_text())
        root = validate(job)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        junit = output / (manifest.stem + ".xml")
        command = ["bash", "tools/run_capped.sh", "4G", sys.executable,
                   "-m", "pytest", "-q", "--tb=short", "--timeout=90",
                   "--junitxml=" + str(junit), *job["tests"]]
        with (output / (manifest.stem + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state["check_pid"] = child.pid
            save(result, state)
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
                raise ValueError("Prepared checks exceeded 30 minutes")
        if code:
            raise ValueError("Prepared checks failed; no commit or push")
        state["counts"] = counts(junit)
        validate(job)
        git(root, "diff", "--check")
        git(root, "add", "--", *job["files"])
        git(root, "commit", "-m", job["message"])
        state.update(status="committed", commit=git(root, "rev-parse", "HEAD"))
        save(result, state)
        git(root, "push", "origin", "HEAD:nightly")
        state["status"] = "published"
    except Exception as exc:
        state.update(status="blocked", error=str(exc))
    state["finished"] = time.time()
    save(result, state)
    print(manifest.stem, state["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.queue.mkdir(parents=True, exist_ok=True)
    output = args.queue / "results"
    output.mkdir(exist_ok=True)
    with (args.queue / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for manifest in sorted(args.queue.glob("*.job.json")):
                process(manifest, output)
            if not args.watch:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
