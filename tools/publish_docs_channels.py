"""Publish main documentation to GitHub Pages and nightly to a Hugging Face Space.

Each input is a complete build of its own pinned branch. Main goes to GitHub
Pages with a /nightly/ redirect stub; nightly is a standalone site uploaded to
a static Space, so the two channels never share one size budget. Videos with
matching readback evidence use their immutable media-host revision; other
media bytes are stored by content hash. This operates only on disposable
build artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


MEDIA_ROOT = "https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/"
MAIN_URL = "https://einarolafsson.github.io/spacr/"
NIGHTLY_SPACE = "einarolafsson/spacr-docs-nightly"
NIGHTLY_URL = "https://einarolafsson-spacr-docs-nightly.static.hf.space/"
PAGES_LIMIT = 900 * 1000 * 1000
_HF_STATIC_INJECTION_BYTES = 101
_HF_STATIC_INJECTION = re.compile(
    rb'<script>window\.huggingface=\{variables:\{"SPACE_CREATOR_USER_ID":'
    rb'"[0-9a-f]{24}"\}\};</script>')


def _read_text(path: Path) -> str:
    """Read UTF-8 text keeping its line endings byte for byte on every OS."""
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text without the platform's newline translation."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def verified_video_hosts(build: Path, checkpoint: Path) -> dict:
    """Bind local renditions to their verified full-resolution hosted recordings."""
    manifest_path = checkpoint / "release-manifest.json"
    receipt_path = checkpoint / "publication-receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        return {}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    commit = receipt.get("commit", "")
    readback = receipt.get("readback", {})
    if (not re.fullmatch(r"[0-9a-f]{40}", commit)
            or receipt.get("media_root") != MEDIA_ROOT + commit
            or receipt.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or readback.get("passed") is not True or readback.get("commit") != commit
            or not readback.get("files_expected")
            or readback.get("downloaded_sha256_matched") != readback["files_expected"]
            or readback.get("metadata_matched") != readback["files_expected"]):
        return {}
    records = {row["path"]: row for row in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]}
    hosts = {}
    for path in (build / "tutorials/production").rglob("*.mp4"):
        relative = path.relative_to(build / "tutorials/production").as_posix()
        web = records.get("web/production/" + relative, {})
        hosted = records.get("media_host/" + relative, {})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (web.get("sha256") == digest and web.get("bytes") == path.stat().st_size
                and re.fullmatch(r"[0-9a-f]{64}", hosted.get("sha256", ""))
                and hosted.get("bytes", 0) > 0):
            hosts[relative] = {"sha256": digest, "hosted_sha256": hosted["sha256"],
                               "hosted_bytes": hosted["bytes"],
                               "url": MEDIA_ROOT + commit + "/" + relative}
    return hosts


def prepare(build: Path, report_path: Path, branch: str, commit: str,
            checkpoint: Path | None = None) -> None:
    """Remove build caches and incompatible locale payloads, retaining a report."""
    if branch not in {"main", "nightly"} or not (build / "index.html").is_file():
        raise ValueError("Expected a complete main or nightly HTML build")
    for cache in [build / ".doctrees", *build.glob(".doctrees-guide-*")]:
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_commit"] = commit
    for language, row in report["api"].items():
        if not row["source_compatible"]:
            path = build / "_static/i18n/api" / f"{language}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_text(path, json.dumps({"schema": 2, "language": language,
                                          "symbols": {}, "unavailable": True}) + "\n")
    _write_text(build / "translation-compatibility.json",
                json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_text(build / "publication.json", json.dumps({
        "schema": 1, "branch": branch, "commit": commit,
        "translation_policy": "report-only; incompatible locales fall back to English",
    }, indent=2) + "\n")
    if checkpoint is not None and (build / "tutorials").is_dir():
        _write_text(build / "tutorials/verified-video-hosts.json",
                    json.dumps(verified_video_hosts(build, checkpoint), indent=2) + "\n")


def share_tutorial_media(channel: Path, output: Path, branch: str) -> None:
    """Rewrite only the player's local video/poster lookup, never narration pins."""
    tutorial = channel / "tutorials"
    player = tutorial / "app_v2.js"
    production = tutorial / "production"
    if not player.exists() or not production.exists():
        return
    text = _read_text(player)
    lookups = ("`${PRODUCTION_ROOT}/${lesson.silent}`",
               "`${PRODUCTION_ROOT}/${activeLesson.poster}`")
    if any(text.count(lookup) != 1 for lookup in lookups):
        raise ValueError(f"{branch}: tutorial media lookups changed; review the publisher")
    manifest = {}
    proof_path = tutorial / "verified-video-hosts.json"
    verified = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.exists() else {}
    media_dir = output / "_media"
    media_dir.mkdir(exist_ok=True)
    prefix = "../"
    for path in sorted(production.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".png", ".webp"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(production).as_posix()
        if relative in verified:
            proof = verified[relative]
            expected_url = re.escape(MEDIA_ROOT) + r"[0-9a-f]{40}/" + re.escape(relative)
            if (path.suffix.lower() != ".mp4" or proof["sha256"] != digest
                    or not re.fullmatch(expected_url, proof["url"])):
                raise ValueError(f"{branch}: hosted video proof differs: {relative}")
            manifest[relative] = proof["url"]
            path.unlink()
            continue
        name = digest + path.suffix.lower()
        target = media_dir / name
        if not target.exists():
            shutil.copyfile(path, target)
        manifest[path.relative_to(production).as_posix()] = prefix + "_media/" + name
        path.unlink()
    helper = (
        "\nconst SPACR_PUBLISHED_MEDIA = " + json.dumps(manifest, sort_keys=True) + ";\n"
        "function publishedMedia(path) {\n"
        "  return SPACR_PUBLISHED_MEDIA[path] || `${PRODUCTION_ROOT}/${path}`;\n"
        "}\n"
    )
    text = text.replace(lookups[0], "publishedMedia(lesson.silent)")
    text = text.replace(lookups[1], "publishedMedia(activeLesson.poster)")
    text = text.replace('"use strict";', '"use strict";' + helper, 1)
    _write_text(player, text)
    index = tutorial / "index.html"
    version = hashlib.sha256(player.read_bytes()).hexdigest()
    html, matches = re.subn(
        r'''(<script\b[^>]*\bsrc=["'])app_v2\.js(?:\?[^"']*)?(["'])''',
        lambda match: match[1] + "app_v2.js?v=" + version + match[2],
        _read_text(index),
    )
    if matches != 1:
        raise ValueError(f"{branch}: expected one tutorial player script reference")
    _write_text(index, html)
    _write_text(tutorial / "published-media.json", json.dumps(manifest, indent=2) + "\n")


def channel_banner(branch: str, main_url: str, nightly_url: str) -> str:
    """Return the cross-channel switcher shown at the top of every page."""
    return (
        '<div class="spacr-publication-channel" style="padding:.5rem 1rem;'
        'background:#14243a;color:#fff;font:14px sans-serif">'
        f'spaCR {branch} · <a style="color:#bcdcff" href="{main_url}">Main documentation</a>'
        f' · <a style="color:#bcdcff" href="{nightly_url}">Nightly preview</a></div>'
    )


def add_banner(channel: Path, banner: str) -> None:
    """Insert the switcher after <body>, or above the tutorial player."""
    for path in channel.rglob("*.html"):
        text = _read_text(path)
        target = (r'(<main\b[^>]*id="lesson-content"[^>]*>)'
                  if path == channel / "tutorials/index.html" and 'id="lesson-content"' in text
                  else r"(<body\b[^>]*>)")
        text = re.sub(target, lambda match: match[0] + banner, text, count=1)
        _write_text(path, text)


def site_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def write_channel_receipt(output: Path, receipt: dict) -> dict:
    """Record the complete site size, including the receipt being published."""
    target = output / "channels.json"
    content_size = site_size(output) - (target.stat().st_size if target.exists() else 0)
    receipt["size_bytes"] = content_size
    while True:
        content = json.dumps(receipt, indent=2) + "\n"
        total = content_size + len(content.encode("utf-8"))
        if total == receipt["size_bytes"]:
            _write_text(target, content)
            return receipt
        receipt["size_bytes"] = total


def read_record(source: Path, branch: str) -> dict:
    record = json.loads((source / "publication.json").read_text(encoding="utf-8"))
    if record["branch"] != branch or not (source / "index.html").is_file():
        raise ValueError(f"Missing or mismatched {branch} build")
    return record


def nightly_redirect(base: str, nightly_url: str) -> str:
    """Script sending any old <base>/nightly/<path> URL to the same Space path.

    The Space serves files only, so a directory path gains its index.html.
    """
    prefix = json.dumps(base + "/nightly")
    return (
        "<script>(function(){var p=location.pathname,b=" + prefix + ";"
        "if(p===b||p.indexOf(b+'/')===0){var r=p.slice(b.length+1);"
        "if(r&&r.slice(-1)==='/'){r+='index.html';}"
        "location.replace(" + json.dumps(nightly_url) + "+r+location.search+location.hash);}})();</script>"
    )


def write_nightly_stub(output: Path, base: str, nightly_url: str) -> None:
    """Keep old /nightly/ links working: an index stub plus a 404 redirect."""
    script = nightly_redirect(base, nightly_url)
    stub = output / "nightly"
    stub.mkdir()
    _write_text(
        stub / "index.html",
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={nightly_url}">'
        f'<link rel="canonical" href="{nightly_url}">'
        f"<title>spaCR nightly documentation moved</title>{script}</head>"
        f'<body><p>The nightly preview moved to <a href="{nightly_url}">{nightly_url}</a>.</p>'
        "</body></html>\n")
    missing = output / "404.html"
    if missing.is_file():
        text = _read_text(missing)
        text, count = re.subn(r"(<head\b[^>]*>)", lambda match: match[0] + script, text, count=1)
        if count != 1:
            raise ValueError("main: 404.html has no <head>")
        _write_text(missing, text)
    else:
        _write_text(
            missing,
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>Page not found · spaCR documentation</title>{script}</head>"
            f'<body><p>Page not found. Open the <a href="{base}/">spaCR documentation</a>.</p>'
            "</body></html>\n")


def assemble_main(main: Path, output: Path, base_path: str = "/spacr",
                  nightly_url: str = NIGHTLY_URL, limit: int = PAGES_LIMIT) -> dict:
    """Build the GitHub Pages site: main only, plus a /nightly/ redirect stub.

    Nightly lives on its own Hugging Face Space, so the Pages budget holds
    one channel. ``limit`` fails the deploy before Pages would reject it.
    """
    if output.exists():
        raise ValueError("Output must be a new directory")
    record = read_record(main, "main")
    if (main / "nightly").exists():
        raise ValueError("main: build already has a nightly directory")
    shutil.copytree(main, output)
    base = "/" + base_path.strip("/") if base_path.strip("/") else ""
    share_tutorial_media(output, output, "main")
    add_banner(output, channel_banner("main", f"{base}/", nightly_url))
    write_nightly_stub(output, base, nightly_url)
    (output / ".nojekyll").touch()
    receipt = {"schema": 2, "channels": {"main": record}, "nightly_url": nightly_url,
               "limit_bytes": limit,
               "media_files": len(list((output / "_media").glob("*")))}
    write_channel_receipt(output, receipt)
    if receipt["size_bytes"] > limit:
        raise ValueError(f"Main Pages site is {receipt['size_bytes']:,} bytes; budget is {limit:,}")
    return receipt


def explicit_index_links(root: Path) -> int:
    """Point relative directory links at their index.html.

    A static Hugging Face Space serves files only: ``tutorials/`` is a 404,
    ``tutorials/index.html`` is not. Only links whose directory holds an
    index.html inside the site are changed. Returns the number rewritten.
    """
    base = root.resolve()
    pattern = re.compile(r'''(\b(?:href|src)=)(["\'])([^"\'<>]*)\2''')
    rewritten = 0
    for page in root.rglob("*.html"):
        text = _read_text(page)

        def fix(match):
            nonlocal rewritten
            value = match[3]
            path, sep, rest = re.match(r"([^?#]*)([?#]?)(.*)", value, re.S).groups()
            if not path.endswith("/") or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|//", path):
                return match[0]
            target = (base / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
            target = target.resolve()
            if not target.is_relative_to(base) or not (target / "index.html").is_file():
                return match[0]
            rewritten += 1
            return f"{match[1]}{match[2]}{path}index.html{sep}{rest}{match[2]}"

        updated = pattern.sub(fix, text)
        if updated != text:
            _write_text(page, updated)
    return rewritten


SPACE_README = """---
title: spaCR nightly documentation
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: bsd-3-clause
short_description: Nightly preview of the spaCR documentation
---

Built from the `nightly` branch of https://github.com/EinarOlafsson/spacr by
its docs workflow; every nightly push replaces this Space. Main documentation:
{main_url}
"""


def assemble_nightly(nightly: Path, output: Path, main_url: str = MAIN_URL,
                     nightly_url: str = NIGHTLY_URL) -> dict:
    """Build the standalone nightly site served from the Hugging Face Space."""
    if output.exists():
        raise ValueError("Output must be a new directory")
    record = read_record(nightly, "nightly")
    shutil.copytree(nightly, output)
    share_tutorial_media(output, output, "nightly")
    add_banner(output, channel_banner("nightly", main_url, nightly_url))
    index_links = explicit_index_links(output)
    _write_text(output / "README.md", SPACE_README.format(main_url=main_url))
    receipt = {"schema": 2, "channels": {"nightly": record}, "nightly_url": nightly_url,
               "index_links": index_links,
               "media_files": len(list((output / "_media").glob("*")))}
    return write_channel_receipt(output, receipt)


def upload_nightly(site: Path, space: str = NIGHTLY_SPACE) -> str:
    """Replace the Space's files in one commit; removed pages are deleted.

    The token comes from ``HF_TOKEN`` (CI secret) or the local login.
    ``upload_folder`` may split uploads into commits in newer Hub clients,
    leaving a mixed site if a later batch reaches the repository quota.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    record = json.loads((site / "channels.json").read_text(encoding="utf-8"))["channels"]["nightly"]
    if not (site / "index.html").is_file() or not (site / "README.md").is_file():
        raise ValueError("Expected an assembled nightly site")
    api = HfApi()
    parent = api.space_info(space).sha
    remote = set(api.list_repo_files(space, repo_type="space", revision=parent))
    local = {path.relative_to(site).as_posix(): path for path in site.rglob("*")
             if path.is_file() and not {".git", ".cache"}.intersection(path.relative_to(site).parts)}
    operations = [CommitOperationDelete(path_in_repo=path)
                  for path in sorted(remote - local.keys() - {".gitattributes"})]
    operations.extend(CommitOperationAdd(path_in_repo=path, path_or_fileobj=local[path])
                      for path in sorted(local))
    info = api.create_commit(
        repo_id=space, repo_type="space", operations=operations, parent_commit=parent,
        commit_message=f"nightly docs {record['commit'][:12]}")
    return info.oid


#: What a free Hugging Face Space may store, history included.
SPACE_LIMIT_BYTES = 1000 * 1000 * 1000
#: Squash before an upload that would take the Space past this share of it.
SQUASH_FRACTION = 0.6
#: ...or once this many commits have piled up, whatever their size.
SQUASH_EVERY = 20


def squash_reason(used_bytes: int, incoming_bytes: int, commits: int, *,
                  limit: int = SPACE_LIMIT_BYTES, fraction: float = SQUASH_FRACTION,
                  every: int = SQUASH_EVERY) -> str | None:
    """Whether the nightly Space's history should be squashed before an upload.

    Every nightly upload keeps the previous site in the Space's history, so
    the first Space filled its 1 GB in about five days of pushes and then
    refused every deploy. ``"storage"`` when what is stored plus the site
    about to go up passes ``fraction`` of ``limit``; ``"commits"`` once
    ``every`` commits have accumulated; otherwise None.
    """
    if used_bytes + incoming_bytes > fraction * limit:
        return "storage"
    if commits >= every:
        return "commits"
    return None


def squash_nightly_if_needed(site: Path, space: str = NIGHTLY_SPACE, api=None) -> dict:
    """Squash the Space to one commit and drop the files no longer served.

    Run before :func:`upload_nightly`. Squashing rewrites the history to the
    current site; the large files only earlier sites referenced still count
    against the quota until they are deleted, so the ones the current tree
    does not reference are deleted as well. The token comes from
    ``HF_TOKEN`` and is never printed.

    :returns: what was measured and done, for the job log.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.space_info(space, expand=["usedStorage"])
    used = int(getattr(info, "used_storage", None) or getattr(info, "usedStorage", 0) or 0)
    commits = sum(1 for _ in api.list_repo_commits(space, repo_type="space"))
    incoming = site_size(site)
    reason = squash_reason(used, incoming, commits)
    result = {"space": space, "reason": reason, "used_bytes": used,
              "incoming_bytes": incoming, "commits": commits, "squashed": False}
    if reason is None:
        return result
    api.super_squash_history(space, repo_type="space",
                             commit_message="Squash nightly documentation history")
    head = api.space_info(space).sha
    served = {entry.lfs.sha256 for entry in api.list_repo_tree(
        space, repo_type="space", revision=head, recursive=True)
        if getattr(entry, "lfs", None) is not None}
    orphans = [item for item in api.list_lfs_files(space, repo_type="space")
               if item.file_oid not in served]
    if orphans:
        api.permanently_delete_lfs_files(space, orphans, repo_type="space")
    result.update(squashed=True, deleted_files=len(orphans),
                  deleted_bytes=sum(int(item.size) for item in orphans))
    return result


def space_refusal_reason(error) -> str | None:
    """Classify a hosting refusal into a fixed, non-secret reason.

    The server text is only matched, never echoed, so a refusal that quotes a
    request URL or credential cannot leak it into a CI log.
    """
    response = None if isinstance(error, HTTPError) else getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None) or {}
    texts = [str(headers.get("X-Error-Message", "") or "")]
    if response is not None:
        try:
            texts.append(str(response.text)[:4096])
        except Exception:
            pass
    else:
        try:
            texts.append(error.read(4096).decode("utf-8", "replace"))
        except Exception:
            pass
    text = " ".join(texts).lower()
    if "storage limit" in text:
        return "storage-limit"
    if re.search(r"don't have the rights|not authori[sz]ed|permission|invalid (user )?token", text):
        return "not-authorized"
    return None


def verify_space_access(space: str) -> dict:
    """Prove token write access while retaining every current Space file byte.

    The authenticated commit endpoint receives only the existing pinned README.
    No files are removed, hardware requested or existing history rewritten.
    A no-content-change commit may be recorded by the hosting service.
    """
    if not re.fullmatch(r"einarolafsson/spacr-docs-[a-z0-9-]+", space):
        raise ValueError("Expected a spaCR documentation Space")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    from huggingface_hub.utils import build_hf_headers

    api = HfApi()

    def access_step(operation, call):
        try:
            return call()
        except Exception as error:
            error.spacr_space_access_operation = operation
            raise

    info = access_step("metadata", lambda: api.space_info(space))
    parent = info.sha
    if info.sdk != "static" or not re.fullmatch(r"[0-9a-f]{40}", parent or ""):
        raise ValueError("Expected an existing static Space with an immutable parent")

    def tree(revision):
        entries = {}
        for entry in api.list_repo_tree(space, repo_type="space", revision=revision,
                                        recursive=True):
            if isinstance(entry, RepoFolder):
                continue
            if not isinstance(entry, RepoFile) or entry.path in entries:
                raise ValueError("Unexpected or duplicate Space tree entry")
            lfs = None if entry.lfs is None else (entry.lfs.sha256, entry.lfs.size)
            entries[entry.path] = (entry.size, entry.blob_id, lfs)
        return entries

    before = access_step("tree", lambda: tree(parent))
    readme = before.get("README.md")
    if readme is None or readme[0] > 65536 or readme[2] is not None:
        raise ValueError("Expected a small existing Git README")
    path = access_step("readme", lambda: hf_hub_download(
        repo_id=space, repo_type="space", filename="README.md", revision=parent))
    raw = Path(path).read_bytes()
    git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if (len(raw), git_hash, None) != readme:
        raise ValueError("Pinned README content differs from the immutable tree")
    payload = [
        {"key": "header", "value": {
            "summary": "Verify documentation workflow write access without content changes",
            "description": "", "parentCommit": parent}},
        {"key": "file", "value": {"path": "README.md", "encoding": "base64",
                                   "content": base64.b64encode(raw).decode("ascii")}},
    ]
    headers = {**build_hf_headers(token=api.token), "Content-Type": "application/x-ndjson"}
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = Request(
        f"{api.endpoint}/api/spaces/{space}/commit/main", headers=headers,
        data=b"".join(json.dumps(row).encode() + b"\n" for row in payload), method="POST")
    def commit():
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise ValueError("Space commit endpoint did not accept the access check")
            return json.load(response)["commitOid"]

    revision = access_step("commit", commit)
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Expected an immutable write-access verification commit")
    if access_step("readback", lambda: tree(revision)) != before:
        raise ValueError("Space tree changed during write-access verification")
    return {"space": space, "parent_revision": parent, "revision": revision,
            "tree_unchanged": True, "files_verified": len(before),
            "readme_sha256": hashlib.sha256(raw).hexdigest()}


def verify_space_tree(site: Path, space: str, revision: str) -> dict:
    """Match every assembled file against one immutable uploaded Space revision.

    Git blobs and LFS objects use their respective content identities. This
    verifies the repository snapshot, separately from public-host deployment.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected an immutable full Space commit")
    record = json.loads((site / "channels.json").read_text(encoding="utf-8"))["channels"]["nightly"]
    if not (site / "index.html").is_file() or not (site / "README.md").is_file():
        raise ValueError("Expected an assembled nightly site")
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    local = {path.relative_to(site).as_posix(): path for path in site.rglob("*")
             if path.is_file() and not {".git", ".cache"}.intersection(path.relative_to(site).parts)}
    remote = {}
    for entry in HfApi().list_repo_tree(space, repo_type="space", revision=revision,
                                       recursive=True):
        if isinstance(entry, RepoFolder):
            continue
        if not isinstance(entry, RepoFile) or entry.path in remote:
            raise ValueError("Unexpected or duplicate Space tree entry")
        remote[entry.path] = entry
    missing = local.keys() - remote.keys()
    extra = remote.keys() - local.keys() - {".gitattributes"}
    if missing or extra:
        raise ValueError(f"Space tree paths differ: missing={sorted(missing)}, extra={sorted(extra)}")
    total = 0
    for name, path in local.items():
        before = path.stat()
        git_hash = hashlib.sha1(f"blob {before.st_size}\0".encode())
        content_hash = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                git_hash.update(chunk)
                content_hash.update(chunk)
                size += len(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino) or size != before.st_size:
            raise ValueError(f"Local file changed during Space verification: {name}")
        entry = remote[name]
        if entry.size != size:
            raise ValueError(f"Space file size differs: {name}")
        if entry.lfs is not None:
            matches = entry.lfs.size == size and entry.lfs.sha256 == content_hash.hexdigest()
        else:
            matches = entry.blob_id == git_hash.hexdigest()
        if not matches:
            raise ValueError(f"Space file content differs: {name}")
        total += size
    return {"space": space, "revision": revision, "source_commit": record["commit"],
            "files_verified": len(local), "bytes_verified": total}


def _public_index_matches(expected: bytes, received: bytes, url: str) -> bool:
    """Allow only the observed static-Space creator script around exact bytes."""
    if received == expected:
        return True
    address = urlsplit(url)
    if (address.scheme != "https"
            or not (address.hostname or "").endswith(".static.hf.space")
            or len(received) != len(expected) + _HF_STATIC_INJECTION_BYTES):
        return False
    head = re.search(rb"<head\b[^>]*>", expected)
    if head is None or received[:head.end()] != expected[:head.end()]:
        return False
    injection = _HF_STATIC_INJECTION.match(received, head.end())
    return (injection is not None
            and received[injection.end():] == expected[head.end():])


def verify_nightly(site: Path, url: str = NIGHTLY_URL, attempts: int = 30,
                   interval: float = 10) -> str:
    """Match public receipts and exact homepage bytes apart from HF's host script."""
    if attempts < 1 or interval < 0:
        raise ValueError("Expected at least one attempt and a nonnegative interval")
    expected = json.loads((site / "channels.json").read_text(encoding="utf-8"))
    expected_index = (site / "index.html").read_bytes()
    record = expected["channels"]["nightly"]
    last_error = "public receipt does not match the assembled site"
    for attempt in range(attempts):
        try:
            received = {}
            for name in ("publication.json", "channels.json"):
                request = Request(url.rstrip("/") + "/" + name + "?source=" + record["commit"],
                                  headers={"Cache-Control": "no-cache"})
                with urlopen(request, timeout=15) as response:
                    received[name] = json.load(response)
            if received["publication.json"] == record and received["channels.json"] == expected:
                name = "index.html"
                request = Request(url.rstrip("/") + "/" + name + "?source=" + record["commit"],
                                  headers={"Cache-Control": "no-cache"})
                with urlopen(request, timeout=15) as response:
                    received_index = response.read(
                        len(expected_index) + _HF_STATIC_INJECTION_BYTES + 1)
                    if _public_index_matches(expected_index, received_index, url):
                        return record["commit"]
                last_error = "public index.html does not match the assembled site"
            else:
                last_error = "public receipt does not match the assembled site"
        except (URLError, TimeoutError, ValueError) as error:
            last_error = f"{name}: {type(error).__name__}"
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise RuntimeError(f"Nightly deployment unverified for {record['commit']}: {last_error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    version = sub.add_parser("version")
    version.add_argument("--root", type=Path, default=Path.cwd())
    configure = sub.add_parser("configure")
    configure.add_argument("--root", type=Path, default=Path.cwd())
    prep = sub.add_parser("prepare")
    prep.add_argument("--build", type=Path, required=True)
    prep.add_argument("--report", type=Path, required=True)
    prep.add_argument("--branch", choices=("main", "nightly"), required=True)
    prep.add_argument("--commit", required=True)
    prep.add_argument("--checkpoint", type=Path, default=Path("tools/tutorials/release_candidate"))
    merge = sub.add_parser("assemble-main")
    merge.add_argument("--main", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--base-path", default="/spacr")
    merge.add_argument("--nightly-url", default=NIGHTLY_URL)
    merge.add_argument("--limit", type=int, default=PAGES_LIMIT)
    night = sub.add_parser("assemble-nightly")
    night.add_argument("--nightly", type=Path, required=True)
    night.add_argument("--output", type=Path, required=True)
    night.add_argument("--main-url", default=MAIN_URL)
    night.add_argument("--nightly-url", default=NIGHTLY_URL)
    squash = sub.add_parser("squash-nightly-if-needed")
    squash.add_argument("--site", type=Path, required=True)
    squash.add_argument("--space", default=NIGHTLY_SPACE)
    upload = sub.add_parser("upload-nightly")
    upload.add_argument("--site", type=Path, required=True)
    upload.add_argument("--space", default=NIGHTLY_SPACE)
    verify = sub.add_parser("verify-nightly")
    verify.add_argument("--site", type=Path, required=True)
    verify.add_argument("--url", default=NIGHTLY_URL)
    verify.add_argument("--attempts", type=int, default=30)
    verify.add_argument("--interval", type=float, default=10)
    tree = sub.add_parser("verify-space-tree")
    tree.add_argument("--site", type=Path, required=True)
    tree.add_argument("--space", required=True)
    tree.add_argument("--revision", required=True)
    access = sub.add_parser("verify-space-access")
    access.add_argument("--space", required=True)
    args = parser.parse_args(argv)
    if args.command == "configure":
        # Load only the compatibility extension from the publisher's tools.
        # Putting that directory on sys.path would shadow the pinned branch's
        # own tools (conf.py imports some lazily, e.g. build_module_workflows),
        # so an older branch would be checked by a newer generator.
        config = args.root / "docs/source/conf.py"
        compat = Path(__file__).resolve().parent / "docs_publication_compat.py"
        with config.open("a") as handle:
            handle.write("\nimport importlib.util as _publication_util\n"
                         "import sys as _publication_sys\n"
                         "_publication_spec = _publication_util.spec_from_file_location(\n"
                         f"    'docs_publication_compat', {str(compat)!r})\n"
                         "_publication_module = _publication_util.module_from_spec(_publication_spec)\n"
                         "_publication_sys.modules['docs_publication_compat'] = _publication_module\n"
                         "_publication_spec.loader.exec_module(_publication_module)\n"
                         "extensions = list(extensions) + ['docs_publication_compat']\n")
    elif args.command == "version":
        from docs_version import source_version
        print(source_version(args.root))
    elif args.command == "prepare":
        prepare(args.build, args.report, args.branch, args.commit, args.checkpoint)
    elif args.command == "assemble-main":
        print(json.dumps(assemble_main(args.main, args.output, args.base_path,
                                       args.nightly_url, args.limit)))
    elif args.command == "assemble-nightly":
        print(json.dumps(assemble_nightly(args.nightly, args.output, args.main_url,
                                          args.nightly_url)))
    elif args.command == "verify-nightly":
        print(verify_nightly(args.site, args.url, args.attempts, args.interval))
    elif args.command == "verify-space-tree":
        print(json.dumps(verify_space_tree(args.site, args.space, args.revision)))
    elif args.command == "squash-nightly-if-needed":
        print(json.dumps(squash_nightly_if_needed(args.site, args.space)))
    elif args.command == "verify-space-access":
        try:
            print(json.dumps(verify_space_access(args.space)))
        except Exception as error:
            response = None if isinstance(error, HTTPError) else getattr(error, "response", None)
            failure = {"space": args.space, "error": type(error).__name__,
                       "status": getattr(response, "status_code", getattr(error, "code", None)),
                       "operation": getattr(error, "spacr_space_access_operation", "validation")}
            reason = space_refusal_reason(error)
            if reason:
                failure["reason"] = reason
            print(json.dumps(failure), file=sys.stderr)
            return 1
    else:
        try:
            print(upload_nightly(args.site, args.space))
        except Exception as error:
            reason = space_refusal_reason(error)
            if reason == "storage-limit":
                print(f"::error::Hugging Face refused {args.space}: its storage limit is "
                      "reached. See features/new/689_* for the maintainer steps.", file=sys.stderr)
            elif reason == "not-authorized":
                print(f"::error::HF_DOCS_TOKEN cannot write {args.space}: give the "
                      "fine-grained token write access to this Space.", file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
