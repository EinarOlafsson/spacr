"""Explicit managed-startup provisioning; ordinary application launches never download."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "spacr/model_zoo.py", "spacr/settings.py", "tools/build_model_cards.py",
    "tools/build_score_cards.py", "tools/web_mask_generator/registry.py",
    "tools/web_mask_generator/provision.py", "tools/web_mask_generator/space_app.py",
)
MAX_MODEL_BYTES = 2 * 1024 ** 3
MAX_TOTAL_BYTES = 24 * 1024 ** 3
STARTUP_SECONDS = 1200
OWNER = {"schema": 1, "owner": "spaCR-managed-web-provisioner"}


class ProvisioningError(RuntimeError):
    """Carry a machine diagnostic for operator startup logs, never a product caption."""


def _refuse(code):
    raise ProvisioningError(json.dumps({"operation": "managed_model_provisioning", "code": code}, sort_keys=True))


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _refuse("duplicate_json_key")
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(1024 ** 2 + 1)
    if len(raw) > 1024 ** 2:
        _refuse("manifest_size_limit")
    return json.loads(raw, object_pairs_hook=_unique), raw


def _source_pins(root):
    return {name: _sha((Path(root) / name).read_bytes()) for name in SOURCE_FILES}


def _identity(family, item):
    url = urllib.parse.urlsplit(item["source_url"])
    if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.password or url.port not in (None, 443):
        _refuse("catalogue_source_url")
    parts = url.path.strip("/").split("/")
    repo_type = "dataset" if parts[0] == "datasets" else "model"
    if repo_type == "dataset":
        parts = parts[1:]
    if len(parts) < 5 or parts[2] != "resolve":
        _refuse("catalogue_source_url")
    result = {"family": family, "key": item["key"], "kind": item["kind"],
              "repo_type": repo_type, "repo_id": "/".join(parts[:2]),
              "filename": urllib.parse.unquote("/".join(parts[4:])), "sha256": item["sha256"]}
    if item.get("repo_type") != repo_type or item.get("repo_id") != result["repo_id"]:
        _refuse("catalogue_repository_identity")
    return result


def _catalogue():
    from .registry import model_families
    return sorted((_identity(family["family"], item) for family in model_families()
                   for item in family["versions"]), key=lambda row: row["key"])


def _row(row):
    expected = {"family", "key", "kind", "repo_type", "repo_id", "filename", "sha256", "revision", "bytes"}
    if not isinstance(row, dict) or set(row) != expected:
        _refuse("manifest_model_schema")
    if (not all(isinstance(row[key], str) for key in expected - {"bytes"})
            or row["repo_type"] not in ("model", "dataset")
            or row["kind"] not in ("cellpose", "detector")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["key"])
            or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", row["repo_id"])
            or not re.fullmatch(r"[0-9a-f]{40}", row["revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_MODEL_BYTES):
        _refuse("manifest_model_identity")
    filename = row["filename"]
    if (not filename or "\\" in filename or "\x00" in filename
            or PurePosixPath(filename).is_absolute()
            or any(part in ("", ".", "..") for part in filename.split("/"))):
        _refuse("manifest_filename")


def _filename(row):
    """Retain the actual backend format suffix behind a safe key/checksum prefix."""
    return row["key"].replace(".", "_") + "_" + row["sha256"] + PurePosixPath(row["filename"]).suffix


def prepare_manifest(snapshot, *, root=ROOT):
    """Build a deterministic source-bound manifest from reviewed public Hub metadata."""
    catalogue = _catalogue()
    source, _ = _read_json(snapshot)
    rows = source["models"]
    if not isinstance(rows, list) or len(rows) != len(catalogue):
        _refuse("incomplete_advertised_versions")
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != {row["key"] for row in catalogue}:
        _refuse("incomplete_advertised_versions")
    models = []
    for identity in catalogue:
        metadata = by_key[identity["key"]]
        file = metadata["file_metadata"]
        if (metadata.get("private") is not False or metadata.get("gated") not in (False, None)
                or metadata.get("catalog_sha_matches_lfs") is not True
                or metadata["uri_repo_type"] != identity["repo_type"]
                or metadata["uri_repo_id"] != identity["repo_id"]
                or metadata["filename"] != identity["filename"]
                or metadata["sha256"] != identity["sha256"]
                or file["lfs"]["sha256"] != identity["sha256"]):
            _refuse("unverified_public_metadata")
        model = identity | {"revision": metadata["revision"], "bytes": file["size"]}
        _row(model)
        models.append(model)
    result = {"schema": 1, "catalogue_sha256": _sha(_encoded(catalogue)),
              "source_sha256": _source_pins(root), "models": models}
    _validate(result, root=root)
    return result


def _validate(manifest, *, root):
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "catalogue_sha256", "source_sha256", "models"}
            or type(manifest["schema"]) is not int or manifest["schema"] != 1
            or manifest["source_sha256"] != _source_pins(root)):
        _refuse("manifest_source_binding")
    rows = manifest["models"]
    if not isinstance(rows, list) or not 0 < len(rows) <= 64:
        _refuse("manifest_model_count")
    for row in rows:
        _row(row)
    catalogue = _catalogue()
    identities = [{key: value for key, value in row.items() if key not in ("revision", "bytes")} for row in rows]
    if (identities != catalogue or len({row["key"] for row in rows}) != len(rows)
            or manifest["catalogue_sha256"] != _sha(_encoded(catalogue))):
        _refuse("incomplete_advertised_versions")
    if len({_filename(row) for row in rows}) != len(rows):
        _refuse("cache_filename_collision")
    if sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES:
        _refuse("manifest_total_byte_limit")


def _deadline(deadline):
    if time.monotonic() >= deadline:
        _refuse("startup_deadline")


def _private_directory(path):
    path = Path(path).absolute()
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            _refuse("cache_symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        _refuse("cache_not_private")
    return path


@contextmanager
def _cache_lease(cache):
    import fcntl
    if os.name != "posix":
        _refuse("managed_platform_unsupported")
    cache = Path(cache).absolute()
    if cache.exists() and not (cache / ".owner.json").exists() and any(cache.iterdir()):
        _refuse("cache_foreign_directory")
    cache = _private_directory(cache)
    owner = cache / ".owner.json"
    if owner.exists() and _regular_bytes(owner, 4096) != _encoded(OWNER):
        _refuse("cache_foreign_owner")
    descriptor = os.open(cache / ".lease", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            _refuse("cache_lease_identity")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _refuse("cache_in_use")
        if owner.exists():
            if _regular_bytes(owner, 4096) != _encoded(OWNER):
                _refuse("cache_foreign_owner")
        else:
            if any(path.name != ".lease" for path in cache.iterdir()):
                _refuse("cache_foreign_directory")
            with owner.open("xb") as handle:
                handle.write(_encoded(OWNER))
            owner.chmod(0o600)
        yield cache
    finally:
        os.close(descriptor)


def _regular_bytes(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > limit:
            _refuse("cache_file_identity")
        raw = handle.read(limit + 1)
        if len(raw) > limit:
            _refuse("cache_file_size")
        return raw


def _verify(path, row, deadline):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1
                or before.st_mode & 0o077 or before.st_size != row["bytes"]):
            _refuse("cache_file_identity")
        digest = hashlib.sha256()
        total = 0
        while True:
            _deadline(deadline)
            chunk = handle.read(min(1024 ** 2, row["bytes"] - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > row["bytes"]:
                _refuse("checkpoint_size")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        if (total != row["bytes"] or digest.hexdigest() != row["sha256"]
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or path.lstat().st_ino != after.st_ino):
            _refuse("checkpoint_integrity")


def _public_url(url):
    selected = urllib.parse.urlsplit(url)
    host = selected.hostname or ""
    if (selected.scheme != "https" or selected.username or selected.password or selected.port not in (None, 443)
            or not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"))):
        _refuse("download_redirect_identity")


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        _public_url(new_url)
        if any(key.lower() in ("authorization", "cookie") for key in request.headers):
            _refuse("download_credentials")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _transport(url, timeout):
    _public_url(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PublicRedirect())
    return opener.open(urllib.request.Request(url, headers={"User-Agent": "spaCR-managed-provisioner"}), timeout=timeout)


def _url(row):
    prefix = "datasets/" if row["repo_type"] == "dataset" else ""
    return "https://huggingface.co/" + prefix + row["repo_id"] + "/resolve/" + row["revision"] + "/" + urllib.parse.quote(row["filename"], safe="/")


def _download(target, row, deadline):
    _deadline(deadline)
    descriptor, temporary = tempfile.mkstemp(prefix=".provision-", dir=target.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with _transport(_url(row), min(60, max(0.1, deadline - time.monotonic()))) as response:
                _public_url(response.geturl())
                if response.status != 200:
                    _refuse("download_http_status")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != row["bytes"]:
                    _refuse("download_declared_size")
                reader = getattr(response, "read1", None)
                if not callable(reader):
                    _refuse("download_stream_contract")
                digest = hashlib.sha256()
                total = 0
                while True:
                    _deadline(deadline)
                    chunk = reader(min(1024 ** 2, row["bytes"] - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > row["bytes"]:
                        _refuse("download_byte_limit")
                    digest.update(chunk)
                    handle.write(chunk)
                if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                    _refuse("download_integrity")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, target)
    finally:
        pending.unlink(missing_ok=True)


def provision_models(manifest_path, cache, *, root=ROOT):
    """Provision every advertised identity before returning a source-bound paths map."""
    manifest, raw = _read_json(manifest_path)
    _validate(manifest, root=root)
    deadline = time.monotonic() + STARTUP_SECONDS
    digest = _sha(raw)
    with _cache_lease(cache) as controlled:
        namespace = _private_directory(controlled / digest)
        filenames = {_filename(row) for row in manifest["models"]}
        if any(path.name not in filenames | {"paths.json"} for path in namespace.iterdir()):
            _refuse("cache_foreign_entry")
        paths = {row["key"]: str(namespace / (_filename(row))) for row in manifest["models"]}
        receipt = _encoded({"schema": 1, "manifest_sha256": digest, "catalogue_sha256": manifest["catalogue_sha256"],
                            "source_sha256": manifest["source_sha256"], "models": paths})
        destination = namespace / "paths.json"
        if destination.exists() or destination.is_symlink():
            if _regular_bytes(destination, 1024 ** 2) != receipt:
                _refuse("cache_paths_receipt_identity")
        missing = []
        for row in manifest["models"]:
            path = namespace / (_filename(row))
            if path.exists() or path.is_symlink():
                _verify(path, row, deadline)
            else:
                missing.append(row)
        required = sum(row["bytes"] for row in missing) + max((row["bytes"] for row in missing), default=0)
        if shutil.disk_usage(namespace).free < required:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(namespace))
        for row in missing:
            _download(namespace / (_filename(row)), row, deadline)
        _validate(manifest, root=root)
        for row in manifest["models"]:
            _verify(Path(paths[row["key"]]), row, deadline)
        if destination.exists() or destination.is_symlink():
            if _regular_bytes(destination, 1024 ** 2) != receipt:
                _refuse("cache_paths_receipt_identity")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".provision-", dir=namespace)
            pending = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(receipt)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(pending, destination)
            finally:
                pending.unlink(missing_ok=True)
        return paths


def managed_startup():
    """Opt in only for explicit managed ZeroGPU startup, leaving local routes alone."""
    if os.environ.get("SPACR_WEB_ZERO_GPU") != "1" or os.environ.get("SPACR_WEB_PROVISION_MODELS") != "1":
        return
    if os.environ.get("SPACR_WEB_MODELS_JSON", "").strip() not in ("", "{}"):
        _refuse("conflicting_operator_model_paths")
    cache = Path(os.environ.get("SPACR_WEB_PROVISION_CACHE", str(Path.home() / ".cache/spacr/web-managed")))
    paths = provision_models(Path(__file__).parent / "space/models.json", cache)
    os.environ["SPACR_WEB_MODELS_JSON"] = json.dumps(paths, sort_keys=True)


def main():
    """Prepare a deterministic manifest without network acquisition or cache writes."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.write_bytes(_encoded(prepare_manifest(options.prepare_manifest)))


if __name__ == "__main__":
    main()
