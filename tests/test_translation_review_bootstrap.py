"""Pinned editorial inputs restore offline without overwriting local review work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("translation_bootstrap_under_test", ROOT / "tools/bootstrap_translation_reviews.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


_PADDED_ARCHIVE_WORKER = r"""
import importlib.util,json,subprocess,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("padded_archive_bootstrap",sys.argv[1])
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
real_popen=subprocess.Popen
producer='''import json,subprocess,sys
raw=subprocess.check_output(json.loads(sys.argv[1]))
sys.stdout.buffer.write(raw)
sys.stdout.buffer.write(bytes(2*1024*1024))
sys.stdout.buffer.flush()
sys.stderr.write("padded archive producer completed")
sys.stderr.flush()
raise SystemExit(int(sys.argv[2]))
'''
def archive_popen(command,*args,**kwargs):
    if command[0]=="git" and "archive" in command:
        command=[sys.executable,"-c",producer,json.dumps(command),sys.argv[4]]
    return real_popen(command,*args,**kwargs)
bootstrap.subprocess.Popen=archive_popen
status=int(sys.argv[4])
try:
    bootstrap.ensure_review_inputs(Path(sys.argv[2]),cache=Path(sys.argv[3]))
except bootstrap.ReviewInputsError as error:
    assert status==7 and str(error)=="padded archive producer completed",str(error)
else:
    assert status==0,"a failed archive producer was accepted"
print("PADDED_ARCHIVE_FINISHED",status)
"""


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], stderr=subprocess.PIPE)


@pytest.fixture
def pinned_reviews(tmp_path):
    bootstrap._READY.clear()
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-q")
    payloads = {
        "docs/i18n/reviewed/api/de/current.json": b'{"records":[{"retired":true,"retired_reason":"source changed","translation":"historical"}]}\n',
        "docs/i18n/reviewed/runtime/de/current.json": b'{"records":[{"source":"Ready","translation":"Bereit"}]}\n',
        "docs/i18n/reviewed/api/pt/README.md": b"Retain source-bound authoring instructions.\n",
        "docs/i18n/retired/api/old.json": b'{"records":[{"translation":"old evidence"}]}\n',
    }
    files = {}
    for path, raw in payloads.items():
        target = remote / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[path] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = json.dumps({"schema": 1, "files": files}, sort_keys=True).encode()
    (remote / "manifest.json").write_bytes(manifest)
    git(remote, "add", ".")
    git(remote, "-c", "user.name=Archive test", "-c", "user.email=archive-test@example.invalid", "commit", "-qm", "Fixture review inputs")
    revision = git(remote, "rev-parse", "HEAD").decode().strip()
    root = tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    pin = {"schema": 1, "repo": str(remote), "revision": revision,
           "manifest": "manifest.json", "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
           "file_count": len(files)}
    (root / bootstrap.PIN_NAME).write_text(json.dumps(pin))
    return root, tmp_path / "cache", remote, payloads, pin


def test_restore_preserves_all_review_and_retirement_bytes_without_cache_hardlinks(pinned_reviews):
    root, cache, remote, payloads, pin = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    for path, raw in payloads.items():
        assert (root / path).read_bytes() == raw
    path = next(iter(payloads))
    (root / path).write_bytes(b"local edit")
    cached_git = cache / "git" / hashlib.sha256(str(remote).encode()).hexdigest()
    assert git(cached_git, "show", pin["revision"] + ":" + path) == payloads[path]


def test_complete_cached_inputs_verify_without_network_or_git(pinned_reviews, monkeypatch):
    root, cache, _, _, _ = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    bootstrap._READY.clear()
    monkeypatch.setattr(bootstrap, "_git", lambda *args, **kwargs: pytest.fail("offline verification contacted Git"))
    bootstrap.ensure_review_inputs(root, cache=cache, offline=True, check=True)


def test_offline_restore_uses_existing_git_cache_and_preserves_other_files(pinned_reviews):
    root, cache, _, payloads, _ = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    path = next(iter(payloads))
    (root / path).unlink()
    bootstrap._READY.clear()
    bootstrap.ensure_review_inputs(root, cache=cache, offline=True)
    assert (root / path).read_bytes() == payloads[path]


def test_cold_offline_failure_is_actionable_and_does_not_create_review_files(pinned_reviews):
    root, cache, _, _, _ = pinned_reviews
    with pytest.raises(bootstrap.ReviewInputsError, match="unavailable offline"):
        bootstrap.ensure_review_inputs(root, cache=cache, offline=True)
    assert not (root / "docs/i18n/reviewed").exists()


def test_differing_review_is_never_overwritten(pinned_reviews):
    root, cache, _, payloads, _ = pinned_reviews
    path = root / next(iter(payloads))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unpublished reviewed translation")
    with pytest.raises(bootstrap.ReviewInputsError, match="refusing to overwrite"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert path.read_bytes() == b"unpublished reviewed translation"
    assert not (root / "docs/i18n/retired").exists()


def test_new_unpinned_review_is_preserved_and_rejected(pinned_reviews):
    root, cache, _, _, _ = pinned_reviews
    path = root / "docs/i18n/reviewed/api/de/new.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new review")
    with pytest.raises(bootstrap.ReviewInputsError, match="Unpinned local review"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert path.read_bytes() == b"new review"


def _repin(remote, root, pin, payloads):
    for path in list(remote.glob("docs/**/*")):
        if path.is_file() and path.relative_to(remote).as_posix() not in payloads:
            path.unlink()
    for path, raw in payloads.items():
        (remote / path).parent.mkdir(parents=True, exist_ok=True)
        (remote / path).write_bytes(raw)
    files = {path: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for path, raw in payloads.items()}
    manifest = json.dumps({"schema": 1, "files": files}, sort_keys=True).encode()
    (remote / "manifest.json").write_bytes(manifest)
    git(remote, "add", "-A")
    git(remote, "-c", "user.name=Archive test", "-c", "user.email=archive-test@example.invalid", "commit", "-qm", "Next review inputs")
    pin = dict(pin, revision=git(remote, "rev-parse", "HEAD").decode().strip(),
               manifest_sha256=hashlib.sha256(manifest).hexdigest(), file_count=len(files))
    (root / bootstrap.PIN_NAME).write_text(json.dumps(pin))
    bootstrap._READY.clear()


def test_a_new_pin_replaces_and_drops_what_the_previous_pin_hydrated(pinned_reviews):
    root, cache, remote, payloads, pin = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    changed, dropped = "docs/i18n/reviewed/runtime/de/current.json", "docs/i18n/retired/api/old.json"
    following = {path: raw for path, raw in payloads.items() if path != dropped}
    following[changed] = b'{"records":[{"source":"Ready","translation":"Fertig"}]}\n'
    _repin(remote, root, pin, following)
    bootstrap.ensure_review_inputs(root, cache=cache)
    for path, raw in following.items():
        assert (root / path).read_bytes() == raw
    assert not (root / dropped).exists()
    bootstrap._READY.clear()
    bootstrap.ensure_review_inputs(root, cache=cache, offline=True, check=True)


def test_a_new_pin_still_refuses_to_overwrite_a_local_edit(pinned_reviews):
    root, cache, remote, payloads, pin = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    edited, dropped = "docs/i18n/reviewed/runtime/de/current.json", "docs/i18n/retired/api/old.json"
    (root / edited).write_bytes(b"unpublished reviewed translation")
    (root / dropped).write_bytes(b"unpublished retirement")
    following = {path: raw for path, raw in payloads.items() if path != dropped}
    following[edited] = b'{"records":[{"source":"Ready","translation":"Fertig"}]}\n'
    _repin(remote, root, pin, following)
    with pytest.raises(bootstrap.ReviewInputsError, match="Unpinned local review"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    (root / dropped).unlink()
    with pytest.raises(bootstrap.ReviewInputsError, match="refusing to overwrite"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert (root / edited).read_bytes() == b"unpublished reviewed translation"


def test_manifest_corruption_is_a_hard_failure_before_hydration(pinned_reviews):
    root, cache, remote, _, pin = pinned_reviews
    pin["manifest_sha256"] = "0" * 64
    (root / bootstrap.PIN_NAME).write_text(json.dumps(pin))
    with pytest.raises(bootstrap.ReviewInputsError, match="manifest SHA-256"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert not (root / "docs/i18n/reviewed").exists()


@pytest.mark.parametrize("path", ["../escape.json", "docs/i18n/reviewed/../../escape.json", "docs/i18n/reviewed\\escape.json", "/docs/i18n/reviewed/escape.json", "spacr/runtime.py"])
def test_manifest_paths_cannot_escape_approved_input_roots(pinned_reviews, path):
    root, cache, remote, _, pin = pinned_reviews
    raw = json.dumps({"schema": 1, "files": {path: {"size": 1, "sha256": "0" * 64}}}).encode()
    (remote / "manifest.json").write_bytes(raw)
    git(remote, "add", "manifest.json")
    git(remote, "-c", "user.name=Archive test", "-c", "user.email=archive-test@example.invalid", "commit", "-qm", "Unsafe fixture manifest")
    pin.update(revision=git(remote, "rev-parse", "HEAD").decode().strip(), manifest_sha256=hashlib.sha256(raw).hexdigest(), file_count=1)
    (root / bootstrap.PIN_NAME).write_text(json.dumps(pin))
    with pytest.raises(bootstrap.ReviewInputsError, match="Unsafe or invalid"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert not (root / "escape.json").exists()


def test_symlinked_source_parent_cannot_redirect_restoration(pinned_reviews, tmp_path):
    root, cache, _, _, _ = pinned_reviews
    external = tmp_path / "external"
    external.mkdir()
    try:
        (root / "docs/i18n").symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"this host cannot create symlinks: {error}")
    with pytest.raises(bootstrap.ReviewInputsError, match="symlink"):
        bootstrap.ensure_review_inputs(root, cache=cache)
    assert list(external.iterdir()) == []


def test_review_validation_is_cached_once_per_process_and_pin(pinned_reviews, monkeypatch):
    root, cache, _, _, _ = pinned_reviews
    bootstrap.ensure_review_inputs(root, cache=cache)
    monkeypatch.setattr(bootstrap, "_existing", lambda *args: pytest.fail("repeated pin verification"))
    bootstrap.ensure_review_inputs(root, cache=cache)


def test_authoring_to_external_checkout_preserves_existing_retirement_history(tmp_path):
    spec = importlib.util.spec_from_file_location("review_writer_under_test", ROOT / "tools/write_reviewed_api_record.py")
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    reviews = tmp_path / "external/docs/i18n/reviewed"
    old = {"label": "old#1", "retired": True, "translation": "Original retired wording", "retired_reason": "source changed"}
    path = reviews / "api/de/review.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "language": "de", "reviewer": "retained", "records": [old]}))
    result = writer.write("de", "review", [{"label": "new#1", "translation": "Reviewed target"}], review_root=reviews)
    payload = json.loads(result.read_text())
    assert result == path
    assert payload["reviewer"] == "retained"
    assert old in payload["records"]
    assert len(payload["records"]) == 2


def test_manifest_generation_and_pin_update_leave_editable_review_bytes_untouched(pinned_reviews):
    root, cache, remote, payloads, _ = pinned_reviews
    manifest = bootstrap.write_manifest(remote, provenance_root=remote)
    assert json.loads(manifest.read_text())["files"] == {
        path: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for path, raw in payloads.items()}
    git(remote, "add", "manifest.json")
    git(remote, "-c", "user.name=Archive test", "-c", "user.email=archive-test@example.invalid", "commit", "-qm", "Regenerated source manifest")
    revision = git(remote, "rev-parse", "HEAD").decode().strip()
    target = bootstrap.pin_revision(revision, root=root, repo=str(remote), cache=cache)
    assert json.loads(target.read_text())["revision"] == revision
    assert not (root / "docs/i18n/reviewed").exists()
    for path, raw in payloads.items():
        assert (remote / path).read_bytes() == raw


def _child_code(body):
    return "import importlib.util, sys\nfrom pathlib import Path\nspec=importlib.util.spec_from_file_location('bootstrap_child',sys.argv[1])\nb=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(b)\n" + body


def test_kernel_lock_is_released_when_owning_process_is_killed(tmp_path):
    import sys
    lock = tmp_path / "lock"
    code = _child_code("with b._lock(Path(sys.argv[2])):\n print('locked',flush=True)\n sys.stdin.read()\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(ROOT / "tools/bootstrap_translation_reviews.py"), str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline().splitlines() == [b"locked"]
    finally:
        child.kill()
        child.wait(timeout=5)
    with bootstrap._lock(lock, timeout=1):
        assert lock.is_file()


def test_killed_partial_stage_never_becomes_a_hydrated_review(pinned_reviews):
    import sys
    root, cache, _, payloads, _ = pinned_reviews
    path = next(iter(payloads))
    target = root / path
    target.parent.mkdir(parents=True)
    code = _child_code("def interrupted(stream,raw):\n stream.write(raw[:3]);stream.flush()\n print('partial',flush=True)\n sys.stdin.read()\nb._write_stage=interrupted\nb._publish_no_clobber(Path(sys.argv[2]),b'complete target bytes')\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(ROOT / "tools/bootstrap_translation_reviews.py"), str(target)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline().splitlines() == [b"partial"]
        assert not target.exists()
    finally:
        child.kill()
        child.wait(timeout=5)
    bootstrap.ensure_review_inputs(root, cache=cache)
    assert target.read_bytes() == payloads[path]


def test_independent_file_created_after_staging_is_preserved(tmp_path, monkeypatch):
    target = tmp_path / "review.json"
    original = bootstrap._write_stage
    def staged_then_local_edit(stream, raw):
        original(stream, raw)
        target.write_bytes(b"independently authored review")
    monkeypatch.setattr(bootstrap, "_write_stage", staged_then_local_edit)
    with pytest.raises(bootstrap.ReviewInputsError, match="preserved existing file"):
        bootstrap._publish_no_clobber(target, b"pinned review")
    assert target.read_bytes() == b"independently authored review"


def test_concurrent_manifest_cache_writers_publish_only_complete_identical_bytes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    target = tmp_path / "manifest.json"
    raw = b'{"files":{}}' * 10000
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(bootstrap._publish_no_clobber, target, raw, allow_identical=True) for _ in range(2)]
        for result in results:
            result.result(timeout=5)
    assert target.read_bytes() == raw
    assert list(tmp_path.glob(".spacr-translation-review-*.tmp")) == []


def test_two_processes_can_bootstrap_the_same_cold_checkout(pinned_reviews):
    import sys
    root, cache, _, payloads, _ = pinned_reviews
    code = _child_code("b.ensure_review_inputs(Path(sys.argv[2]),cache=Path(sys.argv[3]))\nprint('verified',flush=True)\n")
    children = [subprocess.Popen([sys.executable, "-c", code, str(ROOT / "tools/bootstrap_translation_reviews.py"), str(root), str(cache)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    try:
        for child in children:
            output, errors = child.communicate(timeout=10)
            assert child.returncode == 0, errors.decode()
            assert output.splitlines() == [b"verified"]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    for path, raw in payloads.items():
        assert (root / path).read_bytes() == raw


@pytest.mark.parametrize("script", ["build_documentation_i18n.py", "build_i18n_catalogs.py"])
def test_catalog_builder_help_needs_no_translation_cache_or_network(tmp_path, script):
    import os
    import sys
    cache = tmp_path / "cold-cache"
    result = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--help"],
                            env={**os.environ, "SPACR_TRANSLATIONS_CACHE": str(cache)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    assert b"usage:" in result.stdout
    assert not cache.exists()


@pytest.mark.parametrize(("workflow", "job", "reader"), [
    ("tests.yml", "fast", "Run fast tests"),
    ("tests.yml", "coverage-shards", "Run coverage shard in bounded batches"),
    ("tests.yml", "minimum-dependencies", "Run tests at minimum direct versions"),
    ("_pytest-suite.yml", "pytest", "Run ${{ inputs.suite_name }} tests"),
    ("compat-matrix.yml", "install", "Packaging + smoke subset (decides this cell)"),
    ("qt-serial-acceptance.yml", "serial-qt", "Run one guarded, original-order Qt process"),
    ("docs.yml", "build", "Assemble the API catalogs from the pinned translation memory"),
])
def test_current_source_ci_restores_public_inputs_before_translation_readers(workflow, job, reader):
    import yaml
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    steps = document["jobs"][job]["steps"]
    restore = next(index for index, step in enumerate(steps) if step.get("name") == "Restore pinned translation review inputs")
    consume = next(index for index, step in enumerate(steps) if step.get("name") == reader)
    assert restore < consume
    assert steps[restore]["run"] == "python tools/bootstrap_translation_reviews.py fetch"
    assert steps[restore]["if"] == "hashFiles('docs/translation_reviews_revision.json') != ''"


@pytest.mark.parametrize(("job", "reader"), [
    ("fast", "Run fast tests"),
    ("coverage-shards", "Run coverage shard in bounded batches"),
    ("minimum-dependencies", "Run tests at minimum direct versions"),
])
def test_api_catalog_preparation_precedes_pytest_without_bypassing_readers(job, reader):
    import yaml

    document = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    steps = document["jobs"][job]["steps"]
    names = [step.get("name") for step in steps]
    preparation = "Assemble pinned API catalogs before pytest"
    assert names.count(preparation) == 1
    index = names.index(preparation)
    assert names.index("Restore pinned translation review inputs") < index < names.index(reader)
    assert steps[index] == {
        "name": preparation,
        "env": {"CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen"},
        "shell": "bash",
        "run": "\n".join((
            "set -euo pipefail",
            "sudo -n -E systemd-run --scope --quiet \\",
            "  -p MemoryMax=8G -p MemorySwapMax=0 \\",
            "  /usr/sbin/runuser -u \"$(id -un)\" -- /usr/bin/env \\",
            "  \"PATH=$PATH\" \"HOME=$HOME\" \"PYTHONPATH=$GITHUB_WORKSPACE\" \\",
            "  \"CUDA_VISIBLE_DEVICES=\" \"QT_QPA_PLATFORM=offscreen\" \\",
            "  \"OMP_NUM_THREADS=1\" \"MKL_NUM_THREADS=1\" \"OPENBLAS_NUM_THREADS=1\" \\",
            "  python tools/build_documentation_i18n.py --assemble",
        )) + "\n",
    }


def test_isolated_review_directories_bypass_canonical_bootstrap(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import build_documentation_i18n as documentation
    import build_i18n_catalogs as runtime
    import bootstrap_translation_reviews as canonical_bootstrap

    def forbidden(*args, **kwargs):
        pytest.fail("Temporary review inputs must never bootstrap canonical sources")

    monkeypatch.setattr(canonical_bootstrap, "ensure_review_inputs", forbidden)
    monkeypatch.setattr(documentation, "REVIEWED_API_DIR", tmp_path / "api")
    monkeypatch.setattr(documentation, "REVIEWED_README_DIR", tmp_path / "readme")
    monkeypatch.setattr(runtime, "REVIEWED_RUNTIME_DIR", tmp_path / "runtime")
    runtime.reviewed_runtime_translations.cache_clear()
    assert documentation.reviewed_api_block_translations({}, "de") == {}
    assert documentation._load_reviewed_readme_evidence() == {}
    assert runtime.reviewed_runtime_translations("de") == {}
    runtime.reviewed_runtime_translations.cache_clear()


def test_interrupted_git_cache_initialization_can_resume(pinned_reviews):
    root, cache, remote, payloads, pin = pinned_reviews
    repository = cache / "git" / hashlib.sha256(pin["repo"].encode()).hexdigest()
    repository.mkdir(parents=True)
    git(repository, "init", "-q")
    bootstrap.ensure_review_inputs(root, cache=cache)
    assert git(repository, "remote", "get-url", "origin").decode().strip() == pin["repo"]
    assert all((root / path).read_bytes() == raw for path, raw in payloads.items())


def test_cold_catalog_import_fails_offline_with_setup_guidance(tmp_path):
    import os
    import sys
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import bootstrap_translation_reviews as bootstrap
def forbidden(*args, **kwargs):
    raise AssertionError("Import must not invoke Git or network bootstrap")
bootstrap._git = forbidden
try:
    import build_documentation_i18n
except bootstrap.ReviewInputsError as error:
    assert "unavailable offline" in str(error), str(error)
    assert "fetch online once" in str(error), str(error)
else:
    raise AssertionError("Cold import must not silently omit pinned reviews")
"""
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools")],
                            env={**os.environ, "SPACR_TRANSLATIONS_CACHE": str(tmp_path / "cold")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "cold" / "git").exists()


@pytest.mark.parametrize("call", ["run", "Popen"])
def test_a_missing_git_is_named_with_what_to_do(monkeypatch, tmp_path, call):
    """On Windows without Git on PATH both git calls raised a bare WinError 2."""
    def absent(*_args, **_kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(bootstrap.subprocess, call, absent)
    with pytest.raises(bootstrap.ReviewInputsError) as raised:
        if call == "run":
            bootstrap._git(tmp_path, "status")
        else:
            monkeypatch.setattr(bootstrap, "_repository",
                                lambda pin, cache, offline: tmp_path)
            bootstrap._restore(tmp_path, {"revision": "0" * 40}, set(), set(),
                               tmp_path, True)
    message = str(raised.value)
    assert "git was not found on PATH" in message
    assert "https://git-scm.com/downloads" in message


def test_archive_logical_eof_drains_the_real_producer_and_preserves_exit_errors(
        pinned_reviews, tmp_path):
    """Legal padding beyond a pipe's capacity cannot strand stderr EOF."""
    import sys

    root, cache, _, payloads, pin = pinned_reviews
    for status in (0, 7):
        destination = root if status == 0 else tmp_path / 'failed-producer'
        if status:
            (destination / 'docs').mkdir(parents=True)
            (destination / bootstrap.PIN_NAME).write_text(json.dumps(pin))
        result = subprocess.run(
            [sys.executable, '-c', _PADDED_ARCHIVE_WORKER,
             str(ROOT / 'tools/bootstrap_translation_reviews.py'),
             str(destination), str(cache), str(status)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == f'PADDED_ARCHIVE_FINISHED {status}'.encode()
        for relative, raw in payloads.items():
            assert (destination / relative).read_bytes() == raw
