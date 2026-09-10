#!/usr/bin/env python3
"""Publish the tutorial library: docs tree, 1440p encode, and the media host.

One command, run after any lesson changes::

    python tools/publish_tutorials.py                    # everything, incremental
    python tools/publish_tutorials.py --lessons 07_mask  # just one
    python tools/publish_tutorials.py --skip-hf          # docs tree only

It is safe to re-run. Nothing is re-encoded unless its master actually
changed, and the Hugging Face upload hashes before it transfers, so a
no-op run costs seconds.

What goes where, and why
------------------------

The media is split across two homes because GitHub Pages has a hard cap:

``docs/source/_extra`` (GitHub Pages, 1 GB limit)
    The player, the posters, and the **1440p** video. This is the only media
    committed to git and remains subject to the documentation-site budget.

``HF_DATASET`` (Hugging Face dataset)
    All 50 curated narration voices, their timing sidecars, and the **4K** masters.
    The exact size grows with the lesson catalog. Narration cannot all live on
    Pages, so the complete voice catalog is served from the dataset.

``web/`` is the source of truth for the player. The copies under
``docs/source/_extra`` are derived, and this script overwrites them -- so
edit ``web/``, never the repo copy, or the next publish silently reverts you.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PRODUCTION = ROOT / "production"
REPO = Path(os.environ.get(
    "SPACR_REPO", "/mnt/firecuda2/codex/repo/spacr"))
DESTINATION = REPO / "docs/source/_extra/tutorials"

#: Must match tools/docs_media_budget.py NARRATION_HOST and the
#: data-audio-root / data-video4k-root attributes on index.html.
HF_DATASET = "einarolafsson/spacr-tutorials"

#: Records which master produced each published 1440p file, so a re-run only
#: re-encodes what changed. Lives in the repo so it is shared, not local
#: state -- but outside ``_extra``, because everything under there is copied
#: to the live site, and a build manifest is neither media nor a lesson. It
#: sat in ``production/`` briefly and was promptly counted as a 41st lesson.
ENCODE_MANIFEST = REPO / "tools" / "tutorial-encode-manifest.json"

WEB_FILES = [
    "index.html", "styles.css", "app_v2.js", "voice_catalog.js", "module_navigation.js",
    "lesson_catalog.js", "logo_spacr.png", "favicon.svg",
    "TUTORIAL_MEDIA_NOTICE.md",
]

# Scale only. No -r, no -t, no trimming: app_v2.js maps narration onto the
# video at a continuous playback rate, so a changed duration or a dropped
# frame desynchronises every voice in every language. The 4K masters are 30 fps
# CFR; -vsync 0 preserves every source frame and timestamp during downscaling.
ENCODE_ARGS = [
    "-vf", "scale='min(2560,iw)':-2", "-vsync", "0",
    "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
    "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _probe(path: Path, entries: str, stream: str | None = None) -> str:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return _run(cmd).stdout.strip()


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_manifest() -> dict:
    try:
        return json.loads(ENCODE_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def encode_1440p(master: Path, target: Path) -> tuple[bool, str]:
    """Re-encode one 4K master. Returns ``(ok, note)``.

    Frame-count equality is the gate: it proves no frame was dropped or
    duplicated. Container duration can move a fraction of a second on a VFR
    source because the muxer re-derives the last frame's display time, which
    is harmless -- the frames themselves keep their timestamps.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp4")
    result = _run(["ffmpeg", "-nostdin", "-y", "-v", "error",
                   "-i", str(master), *ENCODE_ARGS, str(tmp)])
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, f"ffmpeg failed: {result.stderr.strip()[:200]}"

    before = _probe(master, "stream=nb_frames", "v:0")
    after = _probe(tmp, "stream=nb_frames", "v:0")
    if before != after:
        tmp.unlink(missing_ok=True)
        return False, f"frame count changed {before} -> {after}"
    if tmp.stat().st_size < 50_000:
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        return False, f"suspiciously small output ({size} bytes)"

    tmp.replace(target)
    ratio = master.stat().st_size / max(target.stat().st_size, 1)
    return True, (f"{master.stat().st_size / 1048576:.1f}M -> "
                  f"{target.stat().st_size / 1048576:.1f}M ({ratio:.1f}x)")


def publish_web() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    # Removed source assets must not survive indefinitely in the derived
    # docs tree merely because publishing is incremental.
    (DESTINATION / "youtube_links.js").unlink(missing_ok=True)
    # Existing tracked audio is retained pending the maintainer's decision
    # (325, 2026-09-09). Updating navigation must never silently delete it.
    for filename in WEB_FILES:
        # The application icon is the canonical thinned spaCR logo. Keep the
        # tutorial publisher tied to that source instead of allowing an old
        # web-tree copy to overwrite it during every release.
        source = (
            REPO / "spacr" / "resources" / "icons" / "logo_spacr.png"
            if filename == "logo_spacr.png"
            else WEB / filename
        )
        if not source.is_file():
            print(f"  !! missing web asset: {filename}")
            continue
        shutil.copy2(source, DESTINATION / filename)
    published_index = DESTINATION / "index.html"
    published_index.write_text(
        published_index.read_text().replace(
            'data-production-root="../production"',
            'data-production-root="production"',
        )
    )
    for name in ("fonts", "catalog"):
        source, target = WEB / name, DESTINATION / name
        if not source.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file():
                out = target / path.relative_to(source)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)


def lessons(selected: set[str] | None) -> list[Path]:
    return [p for p in sorted(PRODUCTION.iterdir())
            if p.is_dir() and p.name[:2].isdigit()
            and (selected is None or p.name in selected)]


def upload_to_hf(selected: set[str] | None) -> int:
    """Push narration and the 4K masters. Incremental: hashes, then transfers."""
    suffixes = ["audio/*/*.m4a", "audio/*/*.json", "video/*_silent.mp4"]
    retired = [
        "audio/en/af_alloy.m4a",
        "audio/en/af_alloy.json",
        "audio/en/af_kore.m4a",
        "audio/en/af_kore.json",
        "audio/en/af_nicole.m4a",
        "audio/en/af_nicole.json",
        "audio/en/af_nova.m4a",
        "audio/en/af_nova.json",
    ]
    if selected:
        include = [f"{name}/{suffix}"
                   for name in sorted(selected) for suffix in suffixes]
        delete = [f"{name}/{suffix}"
                  for name in sorted(selected) for suffix in retired]
    else:
        include = [f"*/{suffix}" for suffix in suffixes]
        delete = [f"*/{suffix}" for suffix in retired]
    cmd = ["hf", "upload", HF_DATASET, ".", ".", "--repo-type", "dataset",
           "--include", *include, "--delete", *delete,
           "--commit-message", "Refresh tutorial narration and 4K masters"]
    print(f"  uploading to huggingface.co/datasets/{HF_DATASET} ...")
    return subprocess.run(cmd, cwd=PRODUCTION).returncode


def hf_upload_ready() -> bool:
    """Return whether the local CLI can write to the media host.

    Check before copying or encoding anything: discovering a missing login
    after an hour of local publication work is avoidable and leaves a derived
    docs tree that looks ready while its narration host is still stale.
    """
    if shutil.which("hf") is None:
        print("hf CLI is required unless --skip-hf is used", file=sys.stderr)
        return False
    result = _run(["hf", "auth", "whoami"])
    response = f"{result.stdout}\n{result.stderr}".lower()
    # Current huggingface_hub releases print "Not logged in" but return zero,
    # so the status code alone is not an authentication signal.
    if result.returncode != 0 or "not logged in" in response:
        print(
            "Hugging Face authentication is required before publishing. "
            "Run `hf auth login` with a write-scoped token for "
            f"{HF_DATASET}, then retry.",
            file=sys.stderr,
        )
        return False
    return True


def publish_navigation() -> None:
    """Copy only navigation assets; preserve every catalog and media byte."""
    names = ("index.html", "styles.css", "app_v2.js", "module_navigation.js")
    missing = [name for name in names if not (WEB / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing navigation assets: {missing}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = (WEB / name).read_text(encoding="utf-8")
        if name == "index.html":
            source = source.replace('data-production-root="../production"',
                                    'data-production-root="production"')
        temporary = DESTINATION / f"{name}.part"
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(DESTINATION / name)
    print("navigation only; all lesson catalogs, videos and audio preserved")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lessons",
                        help="comma-separated lesson ids; default is all")
    parser.add_argument("--skip-hf", action="store_true",
                        help="do not touch the media host")
    parser.add_argument("--force-encode", action="store_true",
                        help="re-encode even when the master is unchanged")
    parser.add_argument("--navigation-only", action="store_true",
                        help="copy navigation assets only; never encode, upload, or delete media")
    args = parser.parse_args()

    if args.navigation_only:
        publish_navigation()
        return 0

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg and ffprobe are required", file=sys.stderr)
        return 1
    if not args.skip_hf and not hf_upload_ready():
        return 1

    selected = None if not args.lessons else {
        value.strip() for value in args.lessons.split(",") if value.strip()}

    print("web assets ->", DESTINATION)
    publish_web()

    manifest = load_manifest()
    encoded = skipped = failed = 0
    print("video (4K master -> published 1440p)")
    for lesson in lessons(selected):
        master = lesson / "video" / f"{lesson.name}_silent.mp4"
        if not master.is_file():
            print(f"  skip {lesson.name}: no master")
            continue
        target = DESTINATION / "production" / lesson.name / "video" / master.name
        poster = lesson / "poster.jpg"
        if poster.is_file():
            target.parent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(poster, target.parent.parent / "poster.jpg")

        fingerprint = _fingerprint(master)
        if (manifest.get(lesson.name) == fingerprint and target.is_file()
                and not args.force_encode):
            skipped += 1
            continue
        ok, note = encode_1440p(master, target)
        print(f"  {'ok  ' if ok else 'FAIL'} {lesson.name:26s} {note}")
        if ok:
            manifest[lesson.name] = fingerprint
            encoded += 1
        else:
            failed += 1

    ENCODE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ENCODE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"  encoded {encoded}, unchanged {skipped}, failed {failed}")

    if not args.skip_hf:
        print("media host")
        if upload_to_hf(selected) != 0:
            print("  !! upload failed -- the docs tree is still updated, but "
                  "narration on the site will be stale until it succeeds")
            return 1

    print()
    if args.skip_hf:
        print("GitHub tree published locally; Hugging Face was intentionally "
              "skipped, so narration and 4K are not yet updated.")
    else:
        print("published. narration and 4K are live on Hugging Face;")
    print("the 1440p video and the player need a commit:")
    print(
        f"  cd {REPO} && git add -- docs/source/_extra/tutorials "
        "tools/tutorial-encode-manifest.json && \\"
    )
    print("    git commit -m 'tutorials: refresh lesson media' && git push")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
