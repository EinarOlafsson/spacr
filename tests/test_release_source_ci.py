"""Publication waits for complete CI of the exact post-bump commit."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit
import zipfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_source_ci", ROOT / "tools/require_release_source_ci.py")
CI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CI)
SHA = "a" * 40
COMPATIBILITY = CI.compatibility_jobs(ROOT / ".github/workflows/compat-matrix.yml")
EXPECTED = {
    "tests.yml": {"Release gate / zero unexpected failures": True},
    "docs.yml": {"Pin the documentation branch": True,
                 "Build main documentation": True,
                 "Publish main documentation to GitHub Pages": True},
    "compat-matrix.yml": COMPATIBILITY,
}


class Clock:
    def __init__(self):
        self.now = 0
        self.after_sleep = lambda: None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.after_sleep()


class GitHub:
    def __init__(self):
        self.records = {workflow: [] for workflow in CI.WORKFLOWS}
        self.job_records = {}
        self.dispatches = []
        self.head = SHA
        self.next_id = 1
        self.on_dispatch = lambda workflow: None

    def add(self, workflow, *, status="completed", conclusion="success",
            sha=SHA, attempt=1, number=None):
        number = number or len(self.records[workflow]) + 1
        identifier = self.next_id
        self.next_id += 1
        run = {"id": identifier, "head_sha": sha, "head_branch": "main",
               "event": "push", "run_number": number, "run_attempt": attempt,
               "status": status, "conclusion": conclusion}
        self.records[workflow].append(run)
        jobs = [{"name": name, "status": "completed", "conclusion": "success"}
                for name in EXPECTED[workflow]]
        self.job_records[identifier, attempt] = jobs
        return run, jobs

    def runs(self, workflow, sha):
        return self.records[workflow]

    def jobs(self, run):
        return self.job_records[run["id"], run["run_attempt"]]

    def branch_sha(self, branch):
        assert branch == "main"
        return self.head

    def dispatch(self, workflow, branch):
        self.dispatches.append((workflow, branch))
        self.on_dispatch(workflow)


@pytest.fixture
def github():
    api = GitHub()
    for workflow in CI.WORKFLOWS:
        api.add(workflow)
    return api


def require(api, clock=None, timeout=600):
    clock = clock or Clock()
    return CI.require_source_ci(api, SHA, "main", COMPATIBILITY,
                                timeout=timeout, interval=60,
                                clock=clock, sleep=clock.sleep)


def test_green_exact_source_runs_are_reused_without_dispatch(github):
    require(github)
    assert github.dispatches == []
    assert len(COMPATIBILITY) == 18
    assert sum(COMPATIBILITY.values()) == 15


def test_running_source_run_is_awaited_without_duplicate(github):
    run = github.records["tests.yml"][0]
    run.update(status="in_progress", conclusion=None)
    clock = Clock()
    clock.after_sleep = lambda: run.update(status="completed", conclusion="success")
    require(github, clock)
    assert clock.now == 60
    assert github.dispatches == []


def test_missing_workflows_are_dispatched_once_and_must_finish():
    api, clock = GitHub(), Clock()
    pending = []

    def dispatch(workflow):
        run, _ = api.add(workflow, status="queued", conclusion=None)
        pending.append(run)

    api.on_dispatch = dispatch
    clock.after_sleep = lambda: [run.update(status="completed", conclusion="success")
                                 for run in pending]
    require(api, clock)
    assert api.dispatches == [(workflow, "main") for workflow in CI.WORKFLOWS]


def test_branch_moving_refuses_dispatch_of_another_source():
    api = GitHub()
    api.head = "b" * 40
    with pytest.raises(RuntimeError, match="branch moved"):
        require(api)
    assert api.dispatches == []


def test_automatic_run_appearing_before_dispatch_is_reused(github):
    run = github.records["tests.yml"].pop()

    def branch(branch):
        github.records["tests.yml"].append(run)
        return SHA

    github.branch_sha = branch
    require(github)
    assert github.dispatches == []


def test_dispatch_branch_race_cannot_accept_another_heads_green_run(github):
    github.records["tests.yml"] = []
    github.on_dispatch = lambda workflow: github.add(workflow, sha="b" * 40)
    with pytest.raises(RuntimeError, match="no exact-source run"):
        require(github)
    assert github.dispatches == [("tests.yml", "main")]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_newer_failed_run_blocks_an_older_success_without_retry(github, conclusion):
    github.add("tests.yml", conclusion=conclusion)
    with pytest.raises(RuntimeError, match=conclusion):
        require(github)
    assert github.dispatches == []


def test_latest_successful_attempt_supersedes_the_previous_failure(github):
    run = github.records["tests.yml"][0]
    old = dict(run, run_attempt=1, conclusion="failure")
    run["run_attempt"] = 2
    github.job_records[run["id"], 2] = github.job_records[run["id"], 1]
    github.records["tests.yml"].append(old)
    require(github)
    assert github.dispatches == []


def test_docs_access_success_is_not_publication_acceptance(github):
    _, jobs = github.add("docs.yml")
    jobs[:] = [{"name": "Pin the documentation branch", "status": "completed",
                "conclusion": "skipped"}]
    github.records["docs.yml"].pop(0)
    github.on_dispatch = lambda workflow: github.add(workflow)
    require(github)
    assert github.dispatches == [("docs.yml", "main")]


def test_unknown_live_docs_mode_waits_until_it_can_be_classified(github):
    run, jobs = github.add("docs.yml", status="queued", conclusion=None)
    jobs[:] = []
    clock = Clock()

    def advance():
        run.update(status="completed", conclusion="success")
        jobs[:] = [{"name": "Pin the documentation branch", "status": "completed",
                    "conclusion": "skipped"}]

    clock.after_sleep = advance
    require(github, clock)
    assert clock.now == 60
    assert github.dispatches == []


@pytest.mark.parametrize("change", ["missing", "skipped", "unfinished"])
def test_a_green_run_cannot_hide_a_missing_or_skipped_compatibility_cell(github, change):
    run = github.records["compat-matrix.yml"][0]
    jobs = github.jobs(run)
    job = next(job for job in jobs if job["name"] == "Linux ARM64 / py3.11")
    if change == "missing":
        jobs.remove(job)
    elif change == "skipped":
        job["conclusion"] = "skipped"
    else:
        job["status"] = "in_progress"
    with pytest.raises(RuntimeError, match="Linux ARM64"):
        require(github)


def test_existing_informational_failures_remain_informational(github, capsys):
    run = github.records["compat-matrix.yml"][0]
    jobs = github.jobs(run)
    optional = {name for name, blocking in COMPATIBILITY.items() if not blocking}
    assert len(optional) == 3
    for job in jobs:
        if job["name"] in optional:
            job["conclusion"] = "failure"
    states = {job["name"]: (job["status"], job["conclusion"]) for job in jobs}
    require(github)
    assert {job["name"]: (job["status"], job["conclusion"])
            for job in github.jobs(run)} == states
    assert github.dispatches == []
    assert (f"Accepted compat-matrix.yml run {run['id']} attempt {run['run_attempt']} "
            f"for {SHA}") in capsys.readouterr().out


def test_a_new_failure_invalidates_an_earlier_acceptance_while_other_ci_is_live(github):
    docs = github.records["docs.yml"][0]
    docs.update(status="in_progress", conclusion=None)
    clock = Clock()

    def advance():
        docs.update(status="completed", conclusion="success")
        github.add("tests.yml", conclusion="failure")

    clock.after_sleep = advance
    with pytest.raises(RuntimeError, match="tests.yml.*failure"):
        require(github, clock)


def test_missing_dispatched_run_times_out_without_redispatch():
    api, clock = GitHub(), Clock()
    with pytest.raises(TimeoutError):
        require(api, clock, timeout=120)
    assert clock.now == 120
    assert api.dispatches == [(workflow, "main") for workflow in CI.WORKFLOWS]


def test_dispatch_requests_docs_publication_not_its_other_modes(monkeypatch):
    api = CI.GitHub("EinarOlafsson/spacr")
    calls = []
    monkeypatch.setattr(api, "request", lambda path, payload: calls.append((path, payload)))
    api.dispatch("docs.yml", "main")
    assert calls == [("/actions/workflows/docs.yml/dispatches",
                      {"ref": "main", "inputs": {"mode": "publish"}})]


def test_partial_rerun_retains_untouched_successes_and_replaces_previous_failures(monkeypatch):
    api = CI.GitHub("EinarOlafsson/spacr")
    run = {"id": 41, "head_sha": SHA, "run_attempt": 2, "status": "completed"}
    attempts = {
        1: [{"name": "kept", "head_sha": SHA, "conclusion": "success"},
            {"name": "rerun", "head_sha": SHA, "conclusion": "failure"}],
        2: [{"name": "rerun", "head_sha": SHA, "conclusion": "success"}],
    }
    calls = []

    def pages(path, key):
        calls.append(path)
        assert key == "jobs"
        return attempts[int(path.split("/")[-2])]

    monkeypatch.setattr(api, "pages", pages)
    assert {job["name"]: job["conclusion"] for job in api.jobs(run)} == {
        "kept": "success", "rerun": "success"}
    api.jobs(run)
    assert calls == ["/actions/runs/41/attempts/1/jobs",
                     "/actions/runs/41/attempts/2/jobs"]


def test_job_payload_from_another_commit_is_rejected(monkeypatch):
    api = CI.GitHub("EinarOlafsson/spacr")
    monkeypatch.setattr(api, "pages", lambda *args: [{"head_sha": "b" * 40}])
    with pytest.raises(RuntimeError, match="source differs"):
        api.jobs({"id": 41, "head_sha": SHA, "run_attempt": 1, "status": "completed"})


def test_workflow_listing_paginates_and_filters_the_exact_sha(monkeypatch):
    api = CI.GitHub("EinarOlafsson/spacr")
    paths = []

    def request(path):
        paths.append(path)
        page = int(parse_qs(urlsplit(path).query)["page"][0])
        return {"workflow_runs": [{}] * (100 if page == 1 else 1)}

    monkeypatch.setattr(api, "request", request)
    assert len(api.runs("tests.yml", SHA)) == 101
    assert paths == [f"/actions/workflows/tests.yml/runs?head_sha={SHA}&per_page=100&page=1",
                     f"/actions/workflows/tests.yml/runs?head_sha={SHA}&per_page=100&page=2"]


def test_incomplete_api_listing_cannot_be_mistaken_for_no_existing_run(monkeypatch):
    api = CI.GitHub("EinarOlafsson/spacr")
    calls = []

    def request(path):
        calls.append(path)
        return {"workflow_runs": [{}] * 100}

    monkeypatch.setattr(api, "request", request)
    with pytest.raises(RuntimeError, match="incomplete GitHub listing"):
        api.runs("tests.yml", SHA)
    assert len(calls) == 10


def test_release_publication_edges_require_the_exact_source_gate():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    gate = jobs["source-ci"]
    assert gate["needs"] == "bump"
    assert gate["permissions"] == {"contents": "read", "actions": "write"}
    assert gate["timeout-minutes"] == 360
    assert gate["steps"][0]["with"]["ref"] == "${{ needs.bump.outputs.commit_sha }}"
    command = gate["steps"][-1]
    assert command["env"]["RELEASE_SHA"] == "${{ needs.bump.outputs.commit_sha }}"
    assert command["env"]["RELEASE_BRANCH"] == "${{ needs.bump.outputs.branch }}"
    assert '--sha "$RELEASE_SHA" --branch "$RELEASE_BRANCH"' in command["run"]
    for name in ("publish-pypi", "installers", "github-release"):
        assert "source-ci" in jobs[name]["needs"]
        assert "continue-on-error" not in jobs[name]
    for name in ("publish-pypi", "installers", "github-release"):
        assert "needs.source-ci.result == 'success'" in jobs[name]["if"]


def test_every_release_payload_and_tag_uses_the_ci_accepted_bump_commit():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    jobs = yaml.safe_load(workflow)["jobs"]
    source = "${{ needs.bump.outputs.commit_sha }}"
    assert "needs.installers.outputs.release_commit" not in workflow
    assert jobs["source-ci"]["steps"][0]["with"]["ref"] == source
    assert jobs["package"]["steps"][0]["with"]["ref"] == source
    for name in ("installers", "frozen-installers", "container-images"):
        assert jobs[name]["with"]["source_ref"] == source
    release = jobs["github-release"]["steps"]
    assert release[0]["with"]["ref"] == source
    expected = {"Verify and select exact native release payloads",
                "Tag exact release commit",
                "Publish release notes and downloadable installers"}
    steps = [step for step in release if step.get("name") in expected]
    assert len(steps) == len(expected)
    assert all(step["env"]["RELEASE_SHA"] == source for step in steps)
    assert '--source-commit "$RELEASE_SHA"' in steps[0]["run"]
    assert 'git tag -a "$RELEASE_TAG" "$RELEASE_SHA"' in steps[1]["run"]
    assert '--target "$RELEASE_SHA"' in steps[2]["run"]


def test_repository_installer_sync_is_post_release_and_reuses_the_same_artifacts():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    online = yaml.safe_load((ROOT / ".github/workflows/online-installers.yml").read_text())
    sync = jobs["installer-repository-sync"]
    assert jobs["installers"]["with"]["publish_to_repository"] is False
    assert sync["uses"] == jobs["installers"]["uses"]
    assert set(sync["needs"]) == {"bump", "github-release", "release-notes", "release-metadata"}
    assert "always()" in sync["if"]
    assert "needs.github-release.result == 'success'" in sync["if"]
    assert sync["with"] == {
        "source_ref": "${{ needs.bump.outputs.branch }}",
        "target_branch": "${{ needs.bump.outputs.branch }}",
        "publish_to_repository": True, "collect_only": True,
    }
    trigger = online.get("on", online.get(True))
    assert trigger["workflow_call"]["inputs"]["collect_only"]["default"] is False
    for name in ("linux", "windows", "macos"):
        condition = online["jobs"][name]["if"]
        assert "!inputs.collect_only && (" in condition
        assert f"inputs.platform == '{name}'" in condition
    collect = online["jobs"]["collect"]
    assert set(collect["needs"]) == {"linux", "windows", "macos"}
    assert "always() && !cancelled() && inputs.publish_to_repository" in collect["if"]
    assert "inputs.collect_only || (" in collect["if"]
    for name in ("linux", "windows", "macos"):
        assert f"needs.{name}.result == 'success'" in collect["if"]
    downloads = [step for step in collect["steps"]
                 if step.get("uses", "").startswith("actions/download-artifact@")]
    assert [step["with"] for step in downloads] == [{
        "pattern": "spacr-online-*", "path": "installers", "merge-multiple": True}]
    commands = "\n".join(step.get("run", "") for step in collect["steps"])
    assert "packaging/release.py collect" in commands
    assert "packaging/release.py index" in commands
    assert "--localized-readme-dir docs/i18n/readme" in commands
    assert "--current-installers spacr/application" in commands
    assert "build_linux_online" not in commands
    assert "build_windows_online" not in commands
    assert "build_macos_online" not in commands
    assert "git tag" not in commands


def test_bump_checks_and_stages_tracked_owner_outputs_before_finalizing_the_source():
    steps = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["bump"]["steps"]
    validate = next(step for step in steps if step.get("id") == "version")
    commit = next(step for step in steps if step.get("id") == "commit")
    finalize = next(step for step in steps if step.get("id") == "finalize")
    commands = validate["run"]
    install = next(line for line in commands.splitlines() if "pip install" in line)
    assert {"pillow", "fonttools", "brotli", "uharfbuzz", "pyyaml", "pypdf"} <= set(shlex.split(install))
    assert commands.index(install) < commands.index("python packaging/release.py bump")
    assert commands.index("python packaging/release.py bump") < commands.index("python tools/build_readme_images.py --check")
    assert 'Path("docs/source/_static/deck/slides.json")' in commands
    assert 'manifest.read_text(encoding="utf-8")' in commands
    assert 'raise SystemExit("The info deck does not match the release version")' in commands
    assert steps.index(validate) < steps.index(commit) < steps.index(finalize)
    stage = next(line for line in commit["run"].splitlines() if line.startswith("git add "))
    arguments = shlex.split(stage)
    assert arguments[:4] == ["git", "add", "-u", "--"]
    assert set(arguments[4:]) == {
        "setup.py", "CITATION.cff", "spacr/_version.py", "spacr/resources/release_notes.json", "README.rst",
        "packaging/readme_source.yaml", "docs/i18n/readme", "docs/readme_images",
        "docs/readme_text", "docs/readme_assets", "docs/source/_static/deck",
        "docs/source/_generated/hardware_table.rst", "docs/source/installer_guide.rst",
        "packaging/apptainer/README.md",
    }


@pytest.fixture
def pypi_release(tmp_path):
    spec = importlib.util.spec_from_file_location("pypi_source_release", ROOT / "packaging/release.py")
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    description = (
        "spaCR Å — immutable release\n\n"
        f".. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/{SHA}/docs/panel.svg\n"
        f"   :target: https://github.com/EinarOlafsson/spacr/blob/{SHA}/docs/panel.md\n\n"
        "`Installer <https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst>`_\n"
        "`Guide <https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst>`_\n"
        "`Guide <https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst>`_\n"
        f"`External <https://github.com/another/project/blob/{'b' * 40}/guide.md>`_\n"
        "Preserved trailing spaces.  \n"
    )
    wheel = tmp_path / "spacr-1.5.1.6-py3-none-any.whl"

    def write_wheel(body=description, version="1.5.1.6", name="spacr", entries=1):
        with zipfile.ZipFile(wheel, "w") as archive:
            for index in range(entries):
                archive.writestr(
                    f"spacr{index}-1.5.1.6.dist-info/METADATA",
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                    f"Description-Content-Type: text/x-rst\n\n{body}".encode("utf-8"))

    write_wheel()
    published = {"info": {"name": "spacr", "version": "1.5.1.6", "description": description}}
    return release, tmp_path, wheel, write_wheel, published


def test_same_source_pypi_resume_preserves_unicode_spaces_and_intentional_links(pypi_release):
    release, directory, _wheel, _write, published = pypi_release
    assert release.verify_pypi_source(directory, published, "1.5.1.6", SHA) == SHA


@pytest.mark.parametrize("defect", ["published-source", "local-source", "missing-pin",
                                   "mixed-pin", "wheel-version", "pypi-version",
                                   "package-name", "missing-info", "whitespace"])
def test_pypi_resume_refuses_another_or_unproven_immutable_source(pypi_release, defect):
    release, directory, _wheel, write, published = pypi_release
    body = published["info"]["description"]
    if defect == "published-source":
        published["info"]["description"] = body.replace(SHA, "b" * 40)
    elif defect == "local-source":
        write(body.replace(SHA, "b" * 40))
        published["info"]["description"] = body.replace(SHA, "b" * 40)
    elif defect == "missing-pin":
        write(body.replace(SHA, "main"))
        published["info"]["description"] = body.replace(SHA, "main")
    elif defect == "mixed-pin":
        body += f".. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/{'c' * 40}/other.svg\n"
        write(body)
        published["info"]["description"] = body
    elif defect == "wheel-version":
        write(version="1.5.1.5")
    elif defect == "pypi-version":
        published["info"]["version"] = "1.5.1.5"
    elif defect == "package-name":
        published["info"]["name"] = "another-project"
    elif defect == "missing-info":
        published.clear()
    else:
        published["info"]["description"] = body + "\n"
    with pytest.raises(ValueError):
        release.verify_pypi_source(directory, published, "1.5.1.6", SHA)


@pytest.mark.parametrize("defect", ["missing-wheel", "multiple-wheels", "missing-metadata", "multiple-metadata"])
def test_pypi_provenance_requires_one_unambiguous_validated_wheel(pypi_release, defect):
    release, directory, wheel, write, published = pypi_release
    if defect == "missing-wheel":
        wheel.unlink()
    elif defect == "multiple-wheels":
        (directory / "another.whl").write_bytes(wheel.read_bytes())
    else:
        write(entries=0 if defect == "missing-metadata" else 2)
    with pytest.raises(ValueError, match="exactly one"):
        release.verify_pypi_source(directory, published, "1.5.1.6", SHA)


def test_pypi_source_command_uses_the_recorded_response_without_network(pypi_release):
    _release, directory, _wheel, _write, published = pypi_release
    response = directory / "pypi.json"
    response.write_text(json.dumps(published, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(ROOT / "packaging/release.py"), "verify-pypi-source",
        "--wheel-dir", str(directory), "--pypi-json", str(response),
        "--version", "1.5.1.6", "--source-commit", SHA,
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == SHA + "\n"


def test_pypi_verification_checks_the_gated_sha_against_the_existing_package_artifact():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    verify = jobs["verify-pypi"]
    assert set(verify["needs"]) == {"bump", "package", "source-ci", "publish-pypi"}
    assert "needs.source-ci.result == 'success'" in verify["if"]
    assert "needs.publish-pypi.result == 'skipped'" in verify["if"]
    steps = verify["steps"]
    assert steps[0]["with"]["ref"] == "${{ needs.bump.outputs.commit_sha }}"
    artifact = next(step for step in steps if step.get("uses", "").startswith("actions/download-artifact@"))
    assert artifact["with"] == {"name": "pypi-distributions", "path": "dist"}
    check = steps[-1]
    assert check["env"] == {"VERSION": "${{ needs.bump.outputs.version }}",
                            "RELEASE_SHA": "${{ needs.bump.outputs.commit_sha }}"}
    assert '--output "$RUNNER_TEMP/pypi-release.json"' in check["run"]
    assert '--wheel-dir dist --pypi-json "$RUNNER_TEMP/pypi-release.json"' in check["run"]
    assert '--version "$VERSION" --source-commit "$RELEASE_SHA"' in check["run"]
    assert check["run"].index("verify-pypi-source") < check["run"].index("exit 0")


def test_actual_nightly_compatibility_cells_keep_their_blocking_policies():
    expected = {
        "Packaging metadata": True,
        "PowerShell scripts parse (5.1 and 7)": True,
        "Fractal extra constructs the default renderer": True,
        "Linux x86-64 / py3.12 (control)": True,
        "Linux x86-64 / py3.9 (floor)": True,
        "Linux x86-64 / py3.10": True,
        "Linux x86-64 / py3.13": True,
        "Linux x86-64 / py3.14": True,
        "Linux ARM64 / py3.11": True,
        "Windows x86-64 / py3.11": False,
        "macOS arm64 / py3.11": False,
        "Upstream wheel availability (informational)": False,
    }
    expected.update({f"{version} installs, from wheels only": True
                     for version in ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14")})
    assert COMPATIBILITY == expected


def test_actual_nightly_tests_and_docs_job_names_match_the_source_gate():
    tests = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())["jobs"]
    docs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())["jobs"]
    assert tests["release-gate"]["name"] == "Release gate / zero unexpected failures"
    assert docs["resolve"]["name"] == "Pin the documentation branch"
    assert docs["build"]["name"].replace(
        "${{ needs.resolve.outputs.branch }}", "main") == "Build main documentation"
    assert docs["deploy-pages"]["name"] == "Publish main documentation to GitHub Pages"


@pytest.mark.parametrize("matrix,policy,error", [
    ({"os": ["linux"], "include": [{"os": "windows"}]}, False,
     "combined compatibility matrix"),
    ({"os": ["linux"], "exclude": [{"os": "linux"}]}, False,
     "matrix exclusions"),
    ({"os": ["linux"]}, "${{ github.event_name == 'push' }}",
     "blocking policy"),
    ({"include": [{"experimental": "false"}]}, "${{ matrix.experimental }}",
     "Ambiguous compatibility job"),
])
def test_unsupported_compatibility_policy_cannot_silently_omit_cells(
        tmp_path, matrix, policy, error):
    source = tmp_path / "compat.yml"
    source.write_text(yaml.safe_dump({"jobs": {"check": {
        "name": "check", "continue-on-error": policy,
        "strategy": {"matrix": matrix}}}}), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        CI.compatibility_jobs(source)


def test_green_runs_for_a_superseded_head_cannot_qualify(github):
    github.head = "b" * 40
    with pytest.raises(RuntimeError, match="branch moved"):
        require(github)
    assert github.dispatches == []


def test_main_advancing_during_ci_wait_stops_acceptance(github):
    run = github.records["tests.yml"][0]
    run.update(status="in_progress", conclusion=None)
    clock = Clock()

    def advance():
        github.head = "b" * 40
        run.update(status="completed", conclusion="success")

    clock.after_sleep = advance
    with pytest.raises(RuntimeError, match="branch moved"):
        require(github, clock)
    assert clock.now == 60 and github.dispatches == []


def test_main_advancing_while_last_green_run_is_read_cannot_qualify(github):
    original = github.jobs

    def jobs(run):
        result = original(run)
        if run is github.records["compat-matrix.yml"][0]:
            github.head = "b" * 40
        return result

    github.jobs = jobs
    with pytest.raises(RuntimeError, match="branch moved"):
        require(github)
    assert github.dispatches == []


def test_publication_head_recheck_refuses_advance_after_green_ci(github, monkeypatch):
    require(github)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setattr(CI, "GitHub", lambda repository: github)
    monkeypatch.setattr(sys, "argv", ["source-ci", "--sha", SHA,
                                      "--branch", "main", "--head-only"])
    CI.main()
    github.head = "b" * 40
    with pytest.raises(RuntimeError, match="branch moved"):
        CI.main()
    assert github.dispatches == []


def test_pypi_and_tagged_installer_publication_recheck_the_exact_main_head():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    boundaries = {
        "publish-pypi": ["pypa/gh-action-pypi-publish@release/v1"],
        "github-release": ["Tag exact release commit",
                           "Publish release notes and downloadable installers"],
    }
    for job, names in boundaries.items():
        steps = jobs[job]["steps"]
        for name in names:
            index, = [i for i, step in enumerate(steps)
                       if step.get("uses", step.get("name")) == name]
            guard = steps[index - 1]
            assert guard["run"] == ('python tools/require_release_source_ci.py '
                                     '--sha "$RELEASE_SHA" --branch "$RELEASE_BRANCH" --head-only')
            assert guard["env"] == {
                "GH_TOKEN": "${{ github.token }}",
                "RELEASE_SHA": "${{ needs.bump.outputs.commit_sha }}",
                "RELEASE_BRANCH": "${{ needs.bump.outputs.branch }}",
            }
        assert jobs[job]["permissions"]["contents"] in ("read", "write")
        assert steps[0]["with"]["ref"] == "${{ needs.bump.outputs.commit_sha }}"
