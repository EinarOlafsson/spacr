"""What the published documentation site is allowed to weigh.

``docs/source/conf.py`` sets ``html_extra_path`` so that sphinx copies the
tutorial library into every build. The complete production library contains
one pre-rendered ``.m4a`` per lesson × language × **voice**. Copying it whole
would put the built site far past the
GitHub Pages limit of **1 GB**, and Pages does not fail loudly when that
happens -- it refuses the deployment and keeps serving the last build that
fitted, which reads exactly like a site that did not rebuild.

This module stages a *filtered* copy of ``_extra`` for the build, so the site
ships no duplicate narration. Staging itself deletes nothing. The publisher
removes obsolete narration copies from the derived git tree; the complete
library remains in the editable production workspace and on Hugging Face.
``SPACR_DOCS_FULL_AUDIO=1`` can still stage every locally available voice.

What is dropped, and what is not
--------------------------------

Dropped: every narration ``.m4a`` and voice timing ``.json`` under a lesson.

Kept, in full: every lesson, every silent ``.mp4``, every poster, every caption
catalog, and the player itself. Narration and timing sidecars come from the
configured Hugging Face dataset. Caption-only languages never had audio.

With :data:`NARRATION_HOST` configured, ``voice_catalog.js`` remains complete
because every listed voice is served there. It is filtered only for a
site-local audio build, where offering an unstaged voice would create a 404.

Why staging rather than pruning the output
------------------------------------------

Sphinx has no exclude mechanism for ``html_extra_path``, so the alternative is
to copy the complete changing library and delete most of it on
``build-finished``. Staging hardlinks instead keeps the build proportional to
the published subset.

Run it directly for the measurement::

    python tools/docs_media_budget.py --report
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

#: Where the player fetches narration. Empty means "from the site itself",
#: and the per-language rationing below applies. Set, it means the audio and
#: its timing sidecars are served from this host instead, so none of them are
#: published and the voice catalog is left listing every voice.
#:
#: This is the same string as ``data-audio-root`` on ``index.html``; the page
#: is what actually points the player at it, and this constant is what stops
#: the build shipping a second copy nothing would request.
NARRATION_HOST = (
    "https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/main")

#: :data:`VOICES_PER_LANGUAGE` sentinel: publish no narration at all.
NARRATION_EXTERNAL = -1

#: Voices published per language, in catalog order. 1 keeps the default voice
#: -- what a visitor hears without touching the picker; 0 keeps all of them;
#: :data:`NARRATION_EXTERNAL` keeps none.
#:
#: Narration is external. Publishing no audio is what lets the growing lesson
#: catalog keep every 1440p video on Pages while Hugging Face serves all 54
#: voices. If that host is unavailable, the player keeps the GitHub video and
#: reports that narration is temporarily unavailable.
VOICES_PER_LANGUAGE = NARRATION_EXTERNAL

#: Ceiling on the staged tutorial payload, in bytes. Not the Pages limit --
#: this is the tutorial library alone, and the rest of the site (autoapi HTML,
#: ``_modules``, ``_static``, ``resources``) is ~52 MiB on top.
#:
#: The payload consists of 1440p video, posters, player assets, and catalogs.
#: Exact measurements are reported in CI because the lesson catalog grows.
#:
#: The ceiling catches accidental 4K publication or narration duplication
#: before GitHub Pages silently refuses an oversized deployment.
PUBLISHED_MEDIA_CEILING = 700 * 1024 * 1024

#: Set to ``1`` to publish the entire library, ceiling and all.
FULL_AUDIO_ENV = "SPACR_DOCS_FULL_AUDIO"

#: Extensions the filter is allowed to drop. Anything else under ``audio/``
#: is kept whatever its voice, so a new sidecar format cannot go missing in
#: silence.
VOICE_ASSET_SUFFIXES = (".m4a", ".json")

_LANGUAGE_RE = re.compile(r'\bid:\s*"([^"]+)",\s*\n\s*label:\s*"')
_VOICE_RE = re.compile(r'\{\s*id:\s*"([^"]+)",\s*name:\s*"')


def repo_root() -> Path:
    """The checkout this file lives in."""
    return Path(__file__).resolve().parent.parent


def extra_root(root: Path | None = None) -> Path:
    """``docs/source/_extra`` -- what ``html_extra_path`` points at today."""
    return (root or repo_root()) / "docs" / "source" / "_extra"


def voice_catalog_path(extra: Path) -> Path:
    return extra / "tutorials" / "voice_catalog.js"


def parse_voice_catalog(text: str) -> Dict[str, List[str]]:
    """``{language_id: [voice_id, ...]}`` in the order the picker shows them.

    Read out of ``voice_catalog.js`` rather than off the directory tree,
    because "which voice is the default" is a property of the catalog's
    *order* and a directory listing has none. A language block is an ``id``
    immediately followed by a ``label``; a voice is an ``id`` immediately
    followed by a ``name``. Every voice belongs to the language block that
    opened most recently before it.
    """
    order: Dict[str, List[str]] = {}
    current: str | None = None
    events: List[Tuple[int, str, str]] = []
    for match in _LANGUAGE_RE.finditer(text):
        events.append((match.start(), "language", match.group(1)))
    for match in _VOICE_RE.finditer(text):
        events.append((match.start(), "voice", match.group(1)))
    for _, kind, value in sorted(events):
        if kind == "language":
            current = value
            order.setdefault(current, [])
        elif current is not None:
            order[current].append(value)
    return order


def published_voices(catalog: Dict[str, List[str]],
                     per_language: int = VOICES_PER_LANGUAGE
                     ) -> Dict[str, List[str]]:
    """The voices that go on the site, per language.

    ``per_language == 0`` means all of them, which is what
    :data:`FULL_AUDIO_ENV` selects. :data:`NARRATION_EXTERNAL` means none:
    narration is served from :data:`NARRATION_HOST`, so publishing a copy
    to Pages would cost the budget without anything ever requesting it.
    """
    if per_language < 0:
        return {lang: [] for lang in catalog}
    if per_language == 0:
        return {lang: list(voices) for lang, voices in catalog.items()}
    return {lang: list(voices)[:per_language]
            for lang, voices in catalog.items()}


def _voice_id_of(path: Path) -> str:
    return path.stem


def is_published(path: Path, extra: Path,
                 keep: Dict[str, List[str]]) -> bool:
    """Whether one file under ``_extra`` belongs on the published site.

    Everything that is not a voice asset is published. A voice asset is
    published when its language keeps its voice. A language the catalog does
    not name at all is published whole -- an unknown language is a catalog
    that has drifted from the tree, and dropping media on the strength of a
    parse that missed something is exactly the failure this file is trying to
    prevent elsewhere.
    """
    try:
        parts = path.relative_to(extra).parts
    except ValueError:
        return True
    if "audio" not in parts:
        return True
    index = parts.index("audio")
    # tutorials/production/<lesson>/audio/<language>/<voice>.<ext>
    if len(parts) < index + 3:
        return True
    if path.suffix.lower() not in VOICE_ASSET_SUFFIXES:
        return True
    language = parts[index + 1]
    if language not in keep:
        return True
    return _voice_id_of(path) in keep[language]


def plan(extra: Path | None = None,
         per_language: int = VOICES_PER_LANGUAGE
         ) -> Tuple[List[Path], List[Path], Dict[str, List[str]]]:
    """``(published, dropped, kept_voices)`` for one library.

    Both file lists are absolute and sorted, so a caller can diff two plans.
    """
    extra = extra or extra_root()
    catalog = parse_voice_catalog(voice_catalog_path(extra).read_text())
    keep = published_voices(catalog, per_language)
    published: List[Path] = []
    dropped: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(extra):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            (published if is_published(path, extra, keep)
             else dropped).append(path)
    return sorted(published), sorted(dropped), keep


def filter_voice_catalog(text: str, keep: Dict[str, List[str]]) -> str:
    """Drop the unpublished voices from ``voice_catalog.js``.

    Line-filtered rather than regenerated from the parse: a regenerated file
    would silently lose any field this module does not know about, and the
    catalog is edited by hand. The only structural repair is the trailing
    comma on the last surviving entry of a list.
    """
    language_of_line: Dict[int, str] = {}
    current: str | None = None
    lines = text.splitlines()
    for number, line in enumerate(lines):
        language = re.search(r'\bid:\s*"([^"]+)",\s*$', line)
        if language and "label" in (lines[number + 1] if
                                    number + 1 < len(lines) else ""):
            current = language.group(1)
        language_of_line[number] = current or ""

    out: List[str] = []
    for number, line in enumerate(lines):
        voice = _VOICE_RE.search(line)
        if voice:
            language = language_of_line[number]
            if language in keep and voice.group(1) not in keep[language]:
                continue
        if line.strip().startswith("]") and out:
            last = len(out) - 1
            while last >= 0 and not out[last].strip():
                last -= 1
            if last >= 0 and out[last].rstrip().endswith(","):
                out[last] = out[last].rstrip()[:-1]
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def stage(dest: Path, extra: Path | None = None,
          per_language: int = VOICES_PER_LANGUAGE) -> Path:
    """Build the tree sphinx should copy, and return it.

    Files are hardlinked where the filesystem allows it and copied where it
    does not, so staging a 90 MiB subset of a 712 MiB library costs no space
    and no time. ``dest`` is rebuilt from scratch on every call: a stale entry
    left behind by a previous policy would be published forever.
    """
    extra = extra or extra_root()
    published, _dropped, keep = plan(extra, per_language)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    catalog_source = voice_catalog_path(extra)
    for path in published:
        target = dest / path.relative_to(extra)
        target.parent.mkdir(parents=True, exist_ok=True)
        # The catalog is trimmed to what was staged only when the site is the
        # thing serving narration. With NARRATION_HOST set every voice is
        # reachable from there. The catalog must continue to list every hosted
        # voice even though no narration is duplicated in the staged tree.
        if path == catalog_source and not NARRATION_HOST:
            target.write_text(filter_voice_catalog(path.read_text(), keep))
            continue
        try:
            os.link(path, target)
        except OSError:
            # Different filesystem, or a filesystem with no hardlinks. Correct
            # output matters more than the copy this was meant to avoid.
            shutil.copy2(path, target)
    return dest


def _total(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def per_language_setting() -> int:
    """The policy in force, honouring :data:`FULL_AUDIO_ENV`.

    The environment variable still means "publish everything", which is now
    also the way to build a site that works with no access to
    :data:`NARRATION_HOST` -- an offline or air-gapped copy of the docs.
    """
    if os.environ.get(FULL_AUDIO_ENV, "").strip() in ("1", "true", "yes"):
        return 0
    return VOICES_PER_LANGUAGE


def report(extra: Path | None = None,
           per_language: int | None = None) -> str:
    """A human-readable before/after, for the build log and for a commit."""
    per_language = (per_language_setting() if per_language is None
                    else per_language)
    published, dropped, keep = plan(extra, per_language)
    before, after = _total(published) + _total(dropped), _total(published)
    mib = 1024 * 1024
    lines = [
        f"tutorial library: {before / mib:.1f} MiB in {len(published) + len(dropped)} files",
        f"published:        {after / mib:.1f} MiB in {len(published)} files"
        f"  ({after / before * 100:.1f}%)",
        f"dropped:          {(before - after) / mib:.1f} MiB in {len(dropped)} files",
        f"ceiling:          {PUBLISHED_MEDIA_CEILING / mib:.0f} MiB"
        f"  ({'ok' if after <= PUBLISHED_MEDIA_CEILING else 'OVER'})",
    ]
    if NARRATION_HOST:
        lines.append(f"narration:        served from {NARRATION_HOST}")
        lines.append("                  every hosted voice remains offered")
    if per_language < 0:
        lines.append("                  nothing published, no fallback")
    else:
        lines.append("voices published:")
        for language, voices in keep.items():
            lines.append(f"  {language:<6} {', '.join(voices) or '(none)'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true",
                        help="print the before/after and exit")
    parser.add_argument("--stage", metavar="DIR",
                        help="build the staged tree at DIR")
    parser.add_argument("--voices", type=int, default=None,
                        help=f"voices per language (default "
                             f"{VOICES_PER_LANGUAGE}, 0 for all)")
    args = parser.parse_args(argv)
    per_language = (per_language_setting() if args.voices is None
                    else args.voices)
    if args.stage:
        stage(Path(args.stage), per_language=per_language)
    if args.report or not args.stage:
        print(report(per_language=per_language))
    return 0


if __name__ == "__main__":
    sys.exit(main())
