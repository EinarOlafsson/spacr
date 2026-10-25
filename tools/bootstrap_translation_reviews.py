#!/usr/bin/env python3
"""Restore source-bound translation reviews from their immutable public Git pin."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PIN_NAME = "docs/translation_reviews_revision.json"
PREFIXES = ("docs/i18n/reviewed/", "docs/i18n/retired/")
_READY = set()


class ReviewInputsError(RuntimeError):
    """Pinned review inputs cannot be obtained or differ from local edits."""


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


_NO_GIT = ("git was not found on PATH. The translation reviews are fetched with "
           "Git: install Git (https://git-scm.com/downloads; on Windows also tick "
           "'Git from the command line'), open a new terminal so PATH is "
           "refreshed, and run this command again.")


def _git(repository, *args, check=True):
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except FileNotFoundError as exc:
        raise ReviewInputsError(_NO_GIT) from exc
    if check and result.returncode:
        raise ReviewInputsError(result.stderr.decode("utf-8", "replace").strip())
    return result


@contextmanager
def _lock(path, timeout=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ReviewInputsError(f"Translation bootstrap lock is busy: {path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_stage(stream, raw):
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def _publish_no_clobber(target, raw, *, allow_identical=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".spacr-translation-review-", suffix=".tmp", dir=target.parent)
    stage = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _write_stage(stream, raw)
        try:
            os.link(stage, target)
        except FileExistsError:
            if not allow_identical or target.is_symlink() or target.read_bytes() != raw:
                raise ReviewInputsError(f"Translation target appeared during publication; preserved existing file: {target}")
    finally:
        stage.unlink(missing_ok=True)


def _cache_root(cache):
    if cache is not None:
        return Path(cache).expanduser()
    override = os.environ.get("SPACR_TRANSLATIONS_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "spacr" / "translation-reviews"


def _read_pin(root):
    try:
        raw = (root / PIN_NAME).read_bytes()
        pin = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ReviewInputsError(f"Cannot read translation review pin: {error}") from error
    if (pin.get("schema") != 1 or not isinstance(pin.get("repo"), str)
            or not pin["repo"] or pin["repo"].startswith("-")
            or not re.fullmatch(r"[0-9a-f]{40}", str(pin.get("revision", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(pin.get("manifest_sha256", "")))
            or not isinstance(pin.get("file_count"), int) or pin["file_count"] < 1
            or pin.get("manifest") != "manifest.json"):
        raise ReviewInputsError("Invalid immutable translation review pin")
    return pin, _sha256(raw)


def _repository(pin, cache, offline):
    repository = cache / "git" / _sha256(pin["repo"].encode())
    with _lock(cache / "git.lock"):
        if not (repository / ".git").is_dir():
            if offline:
                raise ReviewInputsError("Pinned translation Git cache is unavailable offline; run python tools/bootstrap_translation_reviews.py fetch online once")
            repository.mkdir(parents=True, exist_ok=True)
            _git(repository, "init", "-q")
        origin = _git(repository, "remote", "get-url", "origin", check=False)
        if origin.returncode:
            if offline:
                raise ReviewInputsError("Pinned translation Git cache initialization is incomplete; run translation bootstrap fetch online once")
            _git(repository, "remote", "add", "origin", pin["repo"])
        elif origin.stdout.decode().strip() != pin["repo"]:
            raise ReviewInputsError("Translation Git cache has a different origin; preserved existing cache")
        if _git(repository, "cat-file", "-e", pin["revision"] + "^{commit}", check=False).returncode:
            if offline:
                raise ReviewInputsError("Pinned translation revision is unavailable offline; run python tools/bootstrap_translation_reviews.py fetch online once")
            _git(repository, "fetch", "--no-tags", "--depth=1", "origin", pin["revision"])
        actual = _git(repository, "rev-parse", pin["revision"] + "^{commit}").stdout.decode().strip()
        if actual != pin["revision"]:
            raise ReviewInputsError("Translation Git cache resolved a different revision")
    return repository


def _manifest(pin, cache, offline):
    cached = cache / "manifests" / (pin["manifest_sha256"] + ".json")
    if cached.is_file():
        raw = cached.read_bytes()
    else:
        repository = _repository(pin, cache, offline)
        raw = _git(repository, "show", pin["revision"] + ":" + pin["manifest"]).stdout
    if _sha256(raw) != pin["manifest_sha256"]:
        raise ReviewInputsError("Translation manifest SHA-256 does not match its pin")
    try:
        manifest = json.loads(raw)
        files = manifest["files"]
    except (ValueError, KeyError, TypeError) as error:
        raise ReviewInputsError("Invalid translation manifest") from error
    if manifest.get("schema") != 1 or not isinstance(files, dict) or len(files) != pin["file_count"]:
        raise ReviewInputsError("Translation manifest schema/file count differs from its pin")
    for path, record in files.items():
        relative = PurePosixPath(path)
        if (not path.startswith(PREFIXES) or relative.is_absolute()
                or ".." in relative.parts or str(relative) != path or "\\" in path
                or not isinstance(record, dict)
                or not isinstance(record.get("size"), int) or record["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
            raise ReviewInputsError(f"Unsafe or invalid translation manifest entry: {path}")
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        _publish_no_clobber(cached, raw, allow_identical=True)
    return files


def _published(cache):
    """Every path/hash pair an earlier verified pin hydrated on this machine."""
    published = {}
    for cached in (cache / "manifests").glob("*.json"):
        raw = cached.read_bytes()
        if _sha256(raw) != cached.stem:
            continue
        try:
            records = json.loads(raw)["files"]
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(records, dict):
            continue
        for path, record in records.items():
            if isinstance(record, dict) and isinstance(record.get("sha256"), str):
                published.setdefault(path, set()).add(record["sha256"])
    return published


def _existing(root, files, published=None):
    """Return the files present and the earlier-pin hydrations the new pin supersedes."""
    published = published or {}
    present = set()
    for prefix in PREFIXES:
        folder = root / prefix
        if folder.resolve() != folder or folder.is_symlink():
            raise ReviewInputsError(f"Translation input directory is a symlink: {folder}")
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_symlink():
                    raise ReviewInputsError(f"Translation input is a symlink: {path}")
                if path.is_file():
                    if path.name.startswith(".spacr-translation-review-") and path.name.endswith(".tmp"):
                        continue
                    present.add(path.relative_to(root).as_posix())
    hashes = {path: _sha256((root / path).read_bytes()) for path in present}
    superseded = {path for path in present
                  if hashes[path] in published.get(path, ())
                  and (path not in files or hashes[path] != files[path]["sha256"])}
    unexpected = present - files.keys() - superseded
    if unexpected:
        raise ReviewInputsError("Unpinned local review files; commit/pin them before bootstrap: "
                                + ", ".join(sorted(unexpected)[:5]))
    for path in present - superseded:
        if (root / path).stat().st_size != files[path]["size"] or hashes[path] != files[path]["sha256"]:
            raise ReviewInputsError(f"Local review differs from immutable pin; refusing to overwrite: {path}")
    return present - superseded, superseded


def _restore(root, pin, files, missing, cache, offline):
    repository = _repository(pin, cache, offline)
    folders = [prefix.rstrip("/") for prefix in PREFIXES if any(path.startswith(prefix) for path in files)]
    try:
        process = subprocess.Popen(
            ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
             "-C", str(repository), "archive", pin["revision"],
             "--", *folders], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ReviewInputsError(_NO_GIT) from exc
    restored = set()
    try:
        try:
            archive = tarfile.open(fileobj=process.stdout, mode="r|")
        except tarfile.ReadError as exc:
            stderr = process.stderr.read().decode("utf-8", "replace").strip()
            raise ReviewInputsError(stderr or "git archive produced no translation inputs") from exc
        with archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in files:
                    raise ReviewInputsError(f"Unexpected translation archive member: {member.name}")
                if member.name not in missing:
                    continue
                record = files[member.name]
                if member.size != record["size"]:
                    raise ReviewInputsError(f"Translation archive size mismatch: {member.name}")
                raw = archive.extractfile(member).read()
                if _sha256(raw) != record["sha256"]:
                    raise ReviewInputsError(f"Translation file SHA-256 mismatch: {member.name}")
                target = root / member.name
                if target.resolve() != target:
                    raise ReviewInputsError(f"Translation target traverses a symlink: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                _publish_no_clobber(target, raw)
                restored.add(member.name)
        stderr = process.stderr.read().decode("utf-8", "replace")
        if process.wait():
            raise ReviewInputsError(stderr.strip() or "git archive failed")
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()
    if restored != missing:
        raise ReviewInputsError("Translation archive did not provide every missing input")


def ensure_review_inputs(root=ROOT, *, cache=None, offline=False, check=False):
    """Validate once per process/pin; restore missing files without overwriting edits."""
    root = Path(root).resolve()
    pin, identity = _read_pin(root)
    key = (str(root), identity)
    if key in _READY and not check:
        return
    cache = _cache_root(cache)
    with _lock(root / ".translation-reviews.lock"):
        files = _manifest(pin, cache, offline)
        present, superseded = _existing(root, files, _published(cache))
        missing = set(files) - present
        if missing or superseded:
            if check:
                raise ReviewInputsError(f"Missing {len(missing)} pinned review files; run translation bootstrap fetch")
            for path in superseded:
                (root / path).unlink()
            if missing:
                _restore(root, pin, files, missing, cache, offline)
    _READY.add(key)


def write_manifest(source, *, provenance_root=ROOT):
    """Regenerate the editable external checkout's source-relative input manifest."""
    source = Path(source).resolve()
    files = {}
    for prefix in PREFIXES:
        folder = source / prefix
        if folder.resolve() != folder:
            raise ReviewInputsError(f"Translation source traverses a symlink: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ReviewInputsError(f"Translation source is a symlink: {path}")
            if path.is_file():
                raw = path.read_bytes()
                files[path.relative_to(source).as_posix()] = {
                    "size": len(raw), "sha256": _sha256(raw)}
    if not files:
        raise ReviewInputsError("No reviewed/retired inputs found in the external checkout")
    payload = {
        "schema": 1,
        "source_repository": "https://github.com/EinarOlafsson/spacr",
        "source_revision": _git(provenance_root, "rev-parse", "HEAD").stdout.decode().strip(),
        "files": files,
    }
    path = source / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def pin_revision(revision, *, root=ROOT, repo="https://github.com/EinarOlafsson/spacr-translations.git", cache=None):
    """Pin an already published external commit; do not alter hydrated/local reviews."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not repo or repo.startswith("-"):
        raise ReviewInputsError("Pin a full immutable commit and an explicit Git repository")
    cache = _cache_root(cache)
    spec = {"repo": repo, "revision": revision}
    repository = _repository(spec, cache, False)
    raw = _git(repository, "show", revision + ":manifest.json").stdout
    payload = json.loads(raw)
    pin = {"schema": 1, "repo": repo, "revision": revision,
           "manifest": "manifest.json", "manifest_sha256": _sha256(raw),
           "file_count": len(payload["files"])}
    _manifest(pin, cache, False)
    target = Path(root) / PIN_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "check", "manifest", "pin"))
    parser.add_argument("--offline", action="store_true", help="Use existing verified Git/cache inputs only")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--source", type=Path, help="Editable external checkout, for manifest")
    parser.add_argument("--revision", help="Full published external commit, for pin")
    parser.add_argument("--repo", default="https://github.com/EinarOlafsson/spacr-translations.git")
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            if args.source is None:
                parser.error("manifest needs --source pointing to the editable external checkout")
            print(write_manifest(args.source))
        elif args.command == "pin":
            if not args.revision:
                parser.error("pin needs --revision with the full published external commit")
            print(pin_revision(args.revision, repo=args.repo, cache=args.cache))
        else:
            ensure_review_inputs(cache=args.cache, offline=args.offline or args.command == "check",
                                 check=args.command == "check")
            print("Pinned translation review inputs verified" + (" offline" if args.offline else ""))
    except (OSError, ValueError, KeyError, tarfile.TarError, ReviewInputsError) as error:
        parser.exit(1, f"Translation review bootstrap failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
