"""Require existing or explicitly dispatched CI for the exact release commit."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote, urlencode


WORKFLOWS = ("tests.yml", "docs.yml", "compat-matrix.yml")


class GitHub:
    """Bounded GitHub requests; a failed dispatch is never retried."""

    def __init__(self, repository):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            raise ValueError("Expected an owner/repository")
        self.prefix = f"repos/{repository}"
        self.attempt_cache = {}

    def request(self, path, payload=None):
        command = ["gh", "api", self.prefix + path]
        data = None
        if payload is not None:
            command += ["--method", "POST", "--input", "-"]
            data = json.dumps(payload)
        result = subprocess.run(command, input=data, capture_output=True,
                                text=True, check=True, timeout=60)
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path, key, query=None):
        values = []
        for page in range(1, 11):
            parameters = {**(query or {}), "per_page": 100, "page": page}
            batch = self.request(path + "?" + urlencode(parameters))[key]
            values.extend(batch)
            if len(batch) < 100:
                return values
        raise RuntimeError(f"Refusing an incomplete GitHub listing: {path}")

    def runs(self, workflow, sha):
        return self.pages(f"/actions/workflows/{workflow}/runs", "workflow_runs",
                          {"head_sha": sha})

    def jobs(self, run):
        if not 1 <= run["run_attempt"] <= 20:
            raise RuntimeError("Refusing an unbounded workflow attempt history")
        effective = {}
        for attempt in range(1, run["run_attempt"] + 1):
            key = (run["id"], attempt)
            jobs = self.attempt_cache.get(key)
            if jobs is None:
                jobs = self.pages(f"/actions/runs/{run['id']}/attempts/"
                                  f"{attempt}/jobs", "jobs")
                if any(job["head_sha"] != run["head_sha"] for job in jobs):
                    raise RuntimeError("Workflow job source differs from its run")
                if attempt < run["run_attempt"] or run["status"] == "completed":
                    self.attempt_cache[key] = jobs
            effective.update((job["name"], job) for job in jobs)
        return list(effective.values())

    def branch_sha(self, branch):
        return self.request("/git/ref/heads/" + quote(branch, safe=""))["object"]["sha"]

    def dispatch(self, workflow, branch):
        payload = {"ref": branch}
        if workflow == "docs.yml":
            payload["inputs"] = {"mode": "publish"}
        self.request(f"/actions/workflows/{workflow}/dispatches", payload)


def compatibility_jobs(path):
    """Derive all cells and their existing blocking policy from the source."""
    import yaml

    expected = {}
    for key, job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].items():
        matrix = job.get("strategy", {}).get("matrix", {})
        if "exclude" in matrix:
            raise ValueError("Unsupported compatibility matrix exclusions")
        axes = {name: values for name, values in matrix.items() if name != "include"}
        cells = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
        cells = matrix.get("include", []) if not axes and "include" in matrix else cells
        if axes and "include" in matrix:
            raise ValueError("Unsupported combined compatibility matrix")
        for cell in cells:
            def render(match):
                return str(cell[match[1]])

            name = re.sub(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", render,
                          job.get("name", key))
            optional = job.get("continue-on-error", False)
            if isinstance(optional, str):
                match = re.fullmatch(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}", optional)
                if match is None:
                    raise ValueError("Unsupported compatibility blocking policy")
                optional = cell[match[1]]
            if not isinstance(optional, bool) or name in expected or "${{" in name:
                raise ValueError("Ambiguous compatibility job")
            expected[name] = not optional
    if not expected:
        raise ValueError("Compatibility workflow has no jobs")
    return expected


def select_run(api, workflow, sha, branch):
    runs = [run for run in api.runs(workflow, sha)
            if run["head_sha"] == sha and run["head_branch"] == branch
            and run["event"] in ("push", "workflow_dispatch")]
    for run in sorted(runs, key=lambda item: (item["run_number"], item["run_attempt"]),
                      reverse=True):
        jobs = api.jobs(run)
        if workflow == "docs.yml" and any(
                job["name"] == "Pin the documentation branch"
                and job.get("conclusion") == "skipped" for job in jobs):
            continue
        return run, jobs
    return None


def require_current_head(api, sha, branch):
    """Refuse publication or acceptance of a superseded main source."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or branch != "main":
        raise ValueError("Release CI requires a full lowercase SHA on main")
    if api.branch_sha(branch) != sha:
        raise RuntimeError("Release branch moved; refusing obsolete release source")


def require_source_ci(api, sha, branch, compatibility, *, timeout=19800,
                      interval=60, clock=time.monotonic, sleep=time.sleep):
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or branch != "main":
        raise ValueError("Release CI requires a full lowercase SHA on main")
    if not 0 < timeout <= 19800 or not 0 < interval <= 60:
        raise ValueError("Release CI wait exceeds its bounded job budget")
    deadline = clock() + timeout
    dispatched = {}
    reported = set()
    expected = {
        "tests.yml": {"Release gate / zero unexpected failures": True},
        "docs.yml": {"Build main documentation": True,
                     "Publish main documentation to GitHub Pages": True},
        "compat-matrix.yml": compatibility,
    }
    while clock() < deadline:
        require_current_head(api, sha, branch)
        accepted = set()
        for workflow in WORKFLOWS:
            candidate = select_run(api, workflow, sha, branch)
            if candidate is None:
                if workflow in dispatched:
                    if clock() - dispatched[workflow] >= 300:
                        raise RuntimeError(f"Dispatched {workflow} produced no exact-source run")
                    continue
                if api.branch_sha(branch) != sha:
                    raise RuntimeError("Release branch moved; refusing to dispatch another source")
                candidate = select_run(api, workflow, sha, branch)
                if candidate is None:
                    api.dispatch(workflow, branch)
                    dispatched[workflow] = clock()
                    print(f"Dispatched {workflow} for {sha}", flush=True)
                    continue
            run, jobs = candidate
            by_name = {job["name"]: job for job in jobs}
            # spaCR 1.5.1.6: the functional gate suffices while advisory
            # coverage is still running or has made the workflow red.
            functional_gate = by_name.get("Release gate / zero unexpected failures")
            tests_ready = (workflow == "tests.yml" and functional_gate is not None
                           and functional_gate["status"] == "completed")
            if run["status"] != "completed" and not tests_ready:
                continue
            if run["conclusion"] != "success" and not tests_ready:
                raise RuntimeError(f"{workflow} exact-source run {run['id']} "
                                   f"finished with {run['conclusion']}")
            for name, blocking in expected[workflow].items():
                job = by_name.get(name)
                if (job is None or job["status"] != "completed"
                        or blocking and job["conclusion"] != "success"):
                    raise RuntimeError(f"{workflow} run {run['id']} lacks completed "
                                       f"{'successful ' if blocking else ''}job {name}")
            accepted.add(workflow)
            identity = (workflow, run['id'], run['run_attempt'])
            if identity not in reported:
                reported.add(identity)
                print(f"Accepted {workflow} run {run['id']} attempt {run['run_attempt']} "
                      f"for {sha}", flush=True)
        if len(accepted) == len(WORKFLOWS):
            require_current_head(api, sha, branch)
            return
        sleep(min(interval, max(0, deadline - clock())))
    raise TimeoutError("Exact-source release CI did not finish before the deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--head-only", action="store_true")
    arguments = parser.parse_args()
    api = GitHub(os.environ["GITHUB_REPOSITORY"])
    if arguments.head_only:
        require_current_head(api, arguments.sha, arguments.branch)
        return
    root = Path(__file__).resolve().parents[1]
    require_source_ci(api, arguments.sha,
                      arguments.branch,
                      compatibility_jobs(root / ".github/workflows/compat-matrix.yml"))


if __name__ == "__main__":
    main()
