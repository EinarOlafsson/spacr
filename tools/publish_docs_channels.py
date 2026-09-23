"""Assemble main and nightly documentation without one overwriting the other.

Each input is a complete build of its own pinned branch. Tutorial video and
poster bytes are stored by content hash so identical recordings fit only once
on Pages. Catalogs, scripts, translated captions and external narration pins
remain branch-specific. This operates only on disposable build artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def prepare(build: Path, report_path: Path, branch: str, commit: str) -> None:
    """Remove build caches and incompatible locale payloads, retaining a report."""
    if branch not in {"main", "nightly"} or not (build / "index.html").is_file():
        raise ValueError("Expected a complete main or nightly HTML build")
    shutil.rmtree(build / ".doctrees", ignore_errors=True)
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
    media_dir = output / "_media"
    media_dir.mkdir(exist_ok=True)
    prefix = "../" if branch == "main" else "../../"
    for path in sorted(production.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".png", ".webp"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
    (tutorial / "published-media.json").write_text(json.dumps(manifest, indent=2) + "\n")


def assemble(main: Path, nightly: Path, output: Path, base_path: str = "/spacr",
             limit: int = 950 * 1024 * 1024) -> dict:
    """Require both complete inputs and reject artifacts above the Pages budget."""
    if output.exists():
        raise ValueError("Output must be a new directory")
    records = {}
    for branch, source in (("main", main), ("nightly", nightly)):
        record = json.loads((source / "publication.json").read_text())
        if record["branch"] != branch or not (source / "index.html").is_file():
            raise ValueError(f"Missing or mismatched {branch} build")
        records[branch] = record
    shutil.copytree(main, output)
    shutil.copytree(nightly, output / "nightly")
    base = "/" + base_path.strip("/") if base_path.strip("/") else ""
    for branch, channel in (("main", output), ("nightly", output / "nightly")):
        share_tutorial_media(channel, output, branch)
        banner = (
            '<div class="spacr-publication-channel" style="padding:.5rem 1rem;'
            'background:#14243a;color:#fff;font:14px sans-serif">'
            f'spaCR {branch} · <a style="color:#bcdcff" href="{base}/">Main documentation</a>'
            f' · <a style="color:#bcdcff" href="{base}/nightly/">Nightly preview</a></div>'
        )
        for path in channel.rglob("*.html"):
            if branch == "main" and "nightly" in path.relative_to(output).parts:
                continue
            text = path.read_text()
            text = re.sub(r"(<body\b[^>]*>)", lambda match: match[0] + banner, text, count=1)
            path.write_text(text)
    (output / ".nojekyll").touch()
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if size > limit:
        raise ValueError(f"Combined Pages site is {size:,} bytes; budget is {limit:,}")
    receipt = {"schema": 1, "channels": records, "size_bytes": size,
               "media_files": len(list((output / "_media").glob("*")))}
    (output / "channels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


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
    merge = sub.add_parser("assemble")
    merge.add_argument("--main", type=Path, required=True)
    merge.add_argument("--nightly", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--base-path", default="/spacr")
    args = parser.parse_args(argv)
    if args.command == "configure":
        config = args.root / "docs/source/conf.py"
        with config.open("a") as handle:
            handle.write("\nimport sys as _publication_sys\n"
                         f"_publication_sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
                         "extensions = list(extensions) + ['docs_publication_compat']\n")
    elif args.command == "version":
        from docs_version import source_version
        print(source_version(args.root))
    elif args.command == "prepare":
        prepare(args.build, args.report, args.branch, args.commit)
    else:
        print(json.dumps(assemble(args.main, args.nightly, args.output, args.base_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
