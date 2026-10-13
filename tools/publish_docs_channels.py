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
import hashlib
import json
from pathlib import Path
import re
import shutil


MEDIA_ROOT = "https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/"
MAIN_URL = "https://einarolafsson.github.io/spacr/"
NIGHTLY_SPACE = "einarolafsson/spacr-docs-nightly"
NIGHTLY_URL = "https://einarolafsson-spacr-docs-nightly.static.hf.space/"
PAGES_LIMIT = 900 * 1000 * 1000


def verified_video_hosts(build: Path, checkpoint: Path) -> dict:
    """Bind local renditions to their verified full-resolution hosted recordings."""
    manifest_path = checkpoint / "release-manifest.json"
    receipt_path = checkpoint / "publication-receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        return {}
    receipt = json.loads(receipt_path.read_text())
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
    records = {row["path"]: row for row in json.loads(manifest_path.read_text())["files"]}
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
    report = json.loads(report_path.read_text())
    report["source_commit"] = commit
    for language, row in report["api"].items():
        if not row["source_compatible"]:
            path = build / "_static/i18n/api" / f"{language}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": 2, "language": language,
                                       "symbols": {}, "unavailable": True}) + "\n")
    (build / "translation-compatibility.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (build / "publication.json").write_text(json.dumps({
        "schema": 1, "branch": branch, "commit": commit,
        "translation_policy": "report-only; incompatible locales fall back to English",
    }, indent=2) + "\n")
    if checkpoint is not None and (build / "tutorials").is_dir():
        (build / "tutorials/verified-video-hosts.json").write_text(
            json.dumps(verified_video_hosts(build, checkpoint), indent=2) + "\n")


def share_tutorial_media(channel: Path, output: Path, branch: str) -> None:
    """Rewrite only the player's local video/poster lookup, never narration pins."""
    tutorial = channel / "tutorials"
    player = tutorial / "app_v2.js"
    production = tutorial / "production"
    if not player.exists() or not production.exists():
        return
    text = player.read_text()
    lookups = ("`${PRODUCTION_ROOT}/${lesson.silent}`",
               "`${PRODUCTION_ROOT}/${activeLesson.poster}`")
    if any(text.count(lookup) != 1 for lookup in lookups):
        raise ValueError(f"{branch}: tutorial media lookups changed; review the publisher")
    manifest = {}
    proof_path = tutorial / "verified-video-hosts.json"
    verified = json.loads(proof_path.read_text()) if proof_path.exists() else {}
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
    player.write_text(text)
    index = tutorial / "index.html"
    version = hashlib.sha256(player.read_bytes()).hexdigest()
    html, matches = re.subn(
        r'''(<script\b[^>]*\bsrc=["'])app_v2\.js(?:\?[^"']*)?(["'])''',
        lambda match: match[1] + "app_v2.js?v=" + version + match[2],
        index.read_text(),
    )
    if matches != 1:
        raise ValueError(f"{branch}: expected one tutorial player script reference")
    index.write_text(html)
    (tutorial / "published-media.json").write_text(json.dumps(manifest, indent=2) + "\n")


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
        text = path.read_text()
        target = (r'(<main\b[^>]*id="lesson-content"[^>]*>)'
                  if path == channel / "tutorials/index.html" and 'id="lesson-content"' in text
                  else r"(<body\b[^>]*>)")
        text = re.sub(target, lambda match: match[0] + banner, text, count=1)
        path.write_text(text)


def site_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def read_record(source: Path, branch: str) -> dict:
    record = json.loads((source / "publication.json").read_text())
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
    (stub / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={nightly_url}">'
        f'<link rel="canonical" href="{nightly_url}">'
        f"<title>spaCR nightly documentation moved</title>{script}</head>"
        f'<body><p>The nightly preview moved to <a href="{nightly_url}">{nightly_url}</a>.</p>'
        "</body></html>\n")
    missing = output / "404.html"
    if missing.is_file():
        text = missing.read_text()
        text, count = re.subn(r"(<head\b[^>]*>)", lambda match: match[0] + script, text, count=1)
        if count != 1:
            raise ValueError("main: 404.html has no <head>")
        missing.write_text(text)
    else:
        missing.write_text(
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
    size = site_size(output)
    if size > limit:
        raise ValueError(f"Main Pages site is {size:,} bytes; budget is {limit:,}")
    receipt = {"schema": 2, "channels": {"main": record}, "nightly_url": nightly_url,
               "size_bytes": size, "limit_bytes": limit,
               "media_files": len(list((output / "_media").glob("*")))}
    (output / "channels.json").write_text(json.dumps(receipt, indent=2) + "\n")
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
        text = page.read_text()

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
            page.write_text(updated)
    return rewritten


SPACE_README = """---
title: spaCR nightly documentation
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
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
    (output / "README.md").write_text(SPACE_README.format(main_url=main_url))
    receipt = {"schema": 2, "channels": {"nightly": record}, "nightly_url": nightly_url,
               "size_bytes": site_size(output), "index_links": index_links,
               "media_files": len(list((output / "_media").glob("*")))}
    (output / "channels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def upload_nightly(site: Path, space: str = NIGHTLY_SPACE) -> str:
    """Replace the Space's files with ``site``; removed pages are deleted.

    The token comes from ``HF_TOKEN`` (CI secret) or the local login.
    """
    from huggingface_hub import HfApi
    record = json.loads((site / "channels.json").read_text())["channels"]["nightly"]
    if not (site / "index.html").is_file() or not (site / "README.md").is_file():
        raise ValueError("Expected an assembled nightly site")
    info = HfApi().upload_folder(
        repo_id=space, repo_type="space", folder_path=site, delete_patterns=["*"],
        commit_message=f"nightly docs {record['commit'][:12]}")
    return info.oid


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
    upload = sub.add_parser("upload-nightly")
    upload.add_argument("--site", type=Path, required=True)
    upload.add_argument("--space", default=NIGHTLY_SPACE)
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
    else:
        print(upload_nightly(args.site, args.space))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
