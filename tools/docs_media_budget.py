"""What the published documentation site is allowed to weigh.

``docs/source/conf.py`` sets ``html_extra_path`` so that sphinx copies the
tutorial library into every build. That library is 712 MiB, of which 659 MiB
(93%) is pre-rendered narration: one ``.m4a`` per lesson x language x **voice**,
40 x 54 = 2,160 files. The built site is therefore ~876 MB against a GitHub
Pages limit of **1 GB** -- 88% of the ceiling, with a lesson batch and a ninth
language both queued behind it.

This module stages a *filtered* copy of ``_extra`` for the build, so the site
ships one voice per language instead of all 54. Nothing is deleted: the full
library stays on disk and in git exactly as recorded, and
``SPACR_DOCS_FULL_AUDIO=1`` publishes every voice again. The choice is a
publishing policy, not an edit to the media.

What is dropped, and what is not
--------------------------------

Dropped: the ``.m4a`` and its timing ``.json`` for every voice after the first
one the catalog lists for a language.

Kept, in full: every lesson, every silent ``.mp4``, every poster, every caption
catalog, the player itself, and **one voice in every narrated language** -- the
one ``app_v2.js`` selects by default (``language.voices[0]``), so a first visit
in any of the eight narrated languages sounds exactly as it does today. The six
caption-only languages are untouched because they never had audio.

``voice_catalog.js`` is rewritten in the staged copy to list only the voices
that were published. Without that the picker offers 27 English voices whose
audio 404s -- the player degrades to a "narration is unavailable" toast, which
is a worse answer than not offering the choice.

Why staging rather than pruning the output
------------------------------------------

Sphinx has no exclude mechanism for ``html_extra_path``, so the alternative is
to let it copy 712 MiB and delete most of it on ``build-finished``. Staging
hardlinks instead: the tree costs nothing to build, and sphinx copies ~90 MiB
rather than 712.

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

#: Voices published per language, in catalog order. 1 keeps the default voice
#: -- what a visitor hears without touching the picker. Raising it multiplies
#: the site by roughly this number; :data:`PUBLISHED_MEDIA_CEILING` and
#: ``tests/test_docs_media_budget.py`` are what stop that happening by accident.
VOICES_PER_LANGUAGE = 1

#: Ceiling on the staged tutorial payload, in bytes. Not the Pages limit --
#: this is the tutorial library alone, and the rest of the site (autoapi HTML,
#: ``_modules``, ``_static``, ``resources``) is ~52 MiB on top. Sized to leave
#: the whole site under a quarter of the 1 GB Pages limit so that the next
#: lesson batch does not need a conversation about it.
PUBLISHED_MEDIA_CEILING = 160 * 1024 * 1024

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

    ``per_language <= 0`` means all of them, which is what
    :data:`FULL_AUDIO_ENV` selects.
    """
    if per_language <= 0:
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
        if path == catalog_source:
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
    """The policy in force, honouring :data:`FULL_AUDIO_ENV`."""
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
        "voices published:",
    ]
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
