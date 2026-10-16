#!/usr/bin/env python3
"""Build spaCR's image README from its structured source.

The maintainer's decision of 2026-10-12: everything in the GitHub README that
is not a code block is an image set in Open Sans -- Regular (400) for titles
and headings, Light (300) for body text -- and code blocks stay real RST code
blocks so they can be copied. This tool is the only thing that writes that
README. Nothing in it is drawn by hand.

Source of truth
    ``packaging/readme_source.yaml`` holds the English README as an ordered
    list of blocks: headings, paragraphs, lists, code blocks, raw RST (the
    shields badges and the deck cover), and ``generated`` blocks owned by
    another generator (the module grid, the model zoo, the installer icons,
    the hardware table, the language picker). Paragraph text keeps RST
    inline markup -- ``**strong**``, ``*emphasis*``, ````literal````, and
    ```label <target>`_`` links -- so a translator edits text, not layout.
    ``docs/i18n/readme/source/<lang>.yaml`` holds each translation with the
    same block sequence; :func:`check_structure` refuses a translation whose
    blocks have drifted from the English ones. A generated block may include
    ``details`` paragraphs for its single readable destination; these do not
    add images or text to the README itself.

Rendering
    Consecutive prose blocks form one panel image, a dark rounded surface
    (the same surface the model-zoo score cards use) that reads on GitHub
    light, GitHub dark and PyPI alike. A new panel starts at every heading
    and wherever a second link would otherwise land in one panel. Each text
    panel opens its own readable Markdown page in ``docs/readme_text``;
    the page preserves the panel's words, structure and original detail links.
    A paragraph that is only links becomes a row of link buttons, one
    image per link. A list whose items each carry their own link becomes one
    linked image per row. A paragraph that still holds several links is
    followed by one button per link.

    Text is set by this file -- word-wrapped against the fonts' own advance
    widths -- and drawn by the browser from fonts EMBEDDED in each SVG as a
    base64 WOFF2 subset of exactly the characters that image draws; an image
    in an ``<img>`` cannot load a web font. Open Sans covers the Latin
    scripts. Devanagari, Hangul and Han are not in Open Sans, so Hindi,
    Korean and Simplified Chinese fall back per character to Noto Sans
    Devanagari, Noto Sans KR and Noto Sans SC at the same two weights, and
    arrows to Noto Sans Symbols. Those four come from google/fonts at a
    pinned commit with pinned SHA-256s (:data:`NOTO_FONTS`) and are cached
    under ``~/.cache/spacr/readme-fonts``.

    Line breaks are decided against widths HarfBuzz (``uharfbuzz``) shapes
    with the same fonts, so Devanagari conjuncts and Latin kerning measure
    what the browser draws. Drawing needs ``fontTools``, ``brotli`` and
    ``uharfbuzz``; ``--check`` needs none of them.

    Every image's ``:alt:`` carries the full text of its block, for screen
    readers and search, and every SVG repeats it in ``<desc>`` and records
    the SHA-256 of the block it was drawn from, so a check can tell a stale
    image from a current one without the fonts.

Run::

    python tools/build_readme_images.py            # write README + images
    python tools/build_readme_images.py --check    # exit 1 if anything is stale
    python tools/build_readme_images.py --refresh-generated  # regenerate the
        # blocks packaging/generate_readme_visuals.py owns, then build
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
README_DIR = ROOT / "docs" / "i18n" / "readme"
SOURCE_EN = ROOT / "packaging" / "readme_source.yaml"
SOURCE_DIR = README_DIR / "source"
IMAGE_DIR = ROOT / "docs" / "readme_images"
TEXT_DIR = ROOT / "docs" / "readme_text"
OPEN_SANS = ROOT / "spacr" / "resources" / "font" / "open_sans" / "static"

LANGUAGES = ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")

#: Bump when a change to this file changes what an image looks like, so the
#: currency check fails until the images are redrawn.
RENDERER_VERSION = "3"

#: Every block kind this README may contain. ``generated`` blocks are written
#: by ``packaging/generate_readme_visuals.py`` (and the installer row by
#: ``packaging/release.py``) between ``.. spacr-<name>-begin/end`` markers;
#: this tool rebuilds text-bearing blocks from their generated source sidecars.
GENERATED_BLOCKS = (
    "language-picker", "hardware", "installer-links", "workflow",
    "model-zoo",
)
HEADING_CHARS = {"=": 1, "~": 2, "-": 3, "^": 4}
HEADING_UNDERLINE = {level: char for char, level in HEADING_CHARS.items()}

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

#: One surface for every image, shared with the score cards in
#: ``docs/models/cards``: a neutral dark grey close to GitHub dark mode's own
#: surfaces, so an image reads on GitHub light, GitHub dark and PyPI without
#: a ``prefers-color-scheme`` switch (which follows the operating system, not
#: the GitHub theme, and would mismatch half the time). Teal is the accent;
#: nothing warm.
PALETTE = {
    "surface": "#161b22",
    "edge": "#30363d",
    "track": "#21262d",
    "ink_primary": "#e6edf3",
    "ink_secondary": "#c9d1d9",
    "ink_muted": "#8b949e",
    "accent": "#2ec4b6",
    "accent_blue": "#6cb6ff",
    "beta": "#8b8cf7",
    "shadow": "#1f2328",
}
#: The two layouts every image is drawn in, as the CSS pixel width of the
#: column it fills. The README is read on a phone (GitHub mobile and mobile
#: web: a 390 px viewport leaves a 358 px column) and on a desktop (GitHub's
#: repository page at 1280 px and wider: an 830 px column). Each image is
#: drawn once per layout at its true display size, one SVG unit to one CSS
#: pixel, and the README offers both through ``<picture>``: the phone
#: drawing is the ``<img>`` itself, so anything that ignores ``<source>``
#: shows it, and :data:`DESKTOP_MEDIA` selects the desktop drawing. No
#: image is scaled by the page, so every word is drawn at the size below on
#: both. Measured on 2026-10-13 from GitHub's rendered README (the API's
#: HTML) and github-markdown-css: body text 16 px, code blocks 85% of it.
PHONE = 358
DESKTOP = 830
LAYOUTS = {"phone": PHONE, "desktop": DESKTOP}
DESKTOP_MEDIA = "(min-width: 1280px)"
#: A row of images (buttons, cards) is this much narrower than the column,
#: so sub-pixel rounding can never push its last image onto a new line.
ROW_SLACK = 2
#: The gap between two buttons in a row, split across their two cells.
GUTTER = 8

MARGIN = 0
RADIUS = 12
#: Transparent space under every image. A block of images sits on a text
#: line, whose few pixels of descent add to this, so the gap between two
#: images matches the 16 px GitHub leaves under a code block.
GAP_BELOW = 11
#: The page's one text size: GitHub's code-block text (85% of its 16 px
#: body), so the words in the images and the commands between them are the
#: same size. Section titles (levels 1 and 2) are GitHub's body size,
#: slightly bigger; everything else is BODY.
BODY = 13.6
LINE = 20.4
TITLE = 16
TITLE_LINE = 24
HEADINGS = {1: (TITLE, TITLE_LINE), 2: (TITLE, TITLE_LINE), 3: (BODY, LINE),
            4: (BODY, LINE)}
LITERAL_SIZE = BODY
LITERAL_PAD = 3
BULLET_INDENT = 14
BLOCK_GAP = 10
CHIP_HEIGHT = 34
CHIP_TEXT = BODY


def pad_x(width: float) -> int:
    """The side padding of a panel ``width`` wide."""
    return 16 if width < 600 else 24


def pad_y(width: float) -> int:
    return 14 if width < 600 else 18


def pad_compact(width: float) -> int:
    return 10 if width < 600 else 12


FAMILY_STACK = "'spacr-os', 'spacr-script', 'spacr-symbols', 'Open Sans', sans-serif"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

NOTO_COMMIT = "bd8f81ddb5c74d5c8897b36ad88b440266245103"
#: ``key: (path in google/fonts, sha256)``.
NOTO_FONTS = {
    "sc": ("ofl/notosanssc/NotoSansSC[wght].ttf",
           "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"),
    "kr": ("ofl/notosanskr/NotoSansKR[wght].ttf",
           "194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252"),
    "deva": ("ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf",
             "14ec4af41f27482216d1c2229f417ff9b1425e1babb014e57d1d40d03229853e"),
    "symbols": ("ofl/notosanssymbols/NotoSansSymbols[wght].ttf",
                "f7e7e04b4a24b6c78893d50cbfd2b2f6cae49617ab047bfef668d252adb128f7"),
}
#: The script font each language falls back to after Open Sans.
SCRIPT_FONT = {"hi": "deva", "ko": "kr", "zh_CN": "sc"}
FONT_COVERAGE = {
    "en": "Open Sans", "sv": "Open Sans", "de": "Open Sans",
    "es": "Open Sans", "pt": "Open Sans", "is": "Open Sans",
    "fr": "Open Sans",
    "hi": "Open Sans + Noto Sans Devanagari (Devanagari)",
    "ko": "Open Sans + Noto Sans KR (Hangul)",
    "zh_CN": "Open Sans + Noto Sans SC (Han)",
}
CACHE = Path(os.environ.get("SPACR_README_FONT_CACHE",
                            Path.home() / ".cache" / "spacr" / "readme-fonts"))


def _noto_path(key: str) -> Path:
    """The pinned Noto font ``key``, downloaded once and verified."""
    relative, digest = NOTO_FONTS[key]
    path = CACHE / NOTO_COMMIT / Path(relative).name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://raw.githubusercontent.com/google/fonts/"
               f"{NOTO_COMMIT}/{relative.replace('[', '%5B').replace(']', '%5D')}")
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{path} has SHA-256 {actual}, expected {digest}")
    return path


def fonts_available() -> bool:
    """True when every font is present without a download."""
    return all((CACHE / NOTO_COMMIT / Path(relative).name).is_file()
               for relative, _digest in NOTO_FONTS.values())


@dataclass
class Face:
    """One font at one weight and style, with what layout needs from it."""
    family: str
    weight: int
    italic: bool
    path: Path
    variable: bool = False

    @property
    def cmap(self) -> dict:
        return _font_metrics(str(self.path), self.weight, self.variable)


@lru_cache(maxsize=None)
def _load_font(path: str, weight: int, variable: bool):
    from fontTools.ttLib import TTFont

    font = TTFont(path, recalcTimestamp=False)
    if variable:
        from fontTools.varLib import instancer

        axes = {"wght": weight}
        if "wdth" in [axis.axisTag for axis in font["fvar"].axes]:
            axes["wdth"] = 100
        cached = CACHE / "instances" / f"{Path(path).stem}-{weight}.ttf"
        if cached.is_file():
            return TTFont(str(cached), recalcTimestamp=False)
        font = instancer.instantiateVariableFont(font, axes)
        cached.parent.mkdir(parents=True, exist_ok=True)
        font.save(str(cached))
        return TTFont(str(cached), recalcTimestamp=False)
    return font


@lru_cache(maxsize=None)
def _font_metrics(path: str, weight: int, variable: bool) -> dict:
    """The face's character map, which decides the per-character fallback."""
    return _load_font(path, weight, variable).getBestCmap()


def _open_sans(weight: int, italic: bool = False) -> Face:
    name = {300: "Light", 400: "Regular"}[weight]
    file = f"OpenSans-{name}{'Italic' if italic else ''}.ttf"
    if italic and weight == 400:
        file = "OpenSans-Italic.ttf"
    return Face("spacr-os", weight, italic, OPEN_SANS / file)


def face_stack(language: str, weight: int, italic: bool = False) -> list[Face]:
    """Faces in fallback order: Open Sans, the script font, the symbols."""
    stack = [_open_sans(weight, italic)]
    script = SCRIPT_FONT.get(language)
    if script:
        stack.append(Face("spacr-script", weight, False, _noto_path(script),
                          variable=True))
    stack.append(Face("spacr-symbols", weight, False, _noto_path("symbols"),
                      variable=True))
    return stack


def _face_for(char: str, stack: Sequence[Face]) -> Optional[Face]:
    for face in stack:
        if ord(char) in face.cmap:
            return face
    return None


def _static_path(face: Face) -> Path:
    """The file HarfBuzz and the subsetter read: the instance if variable."""
    if not face.variable:
        return face.path
    _load_font(str(face.path), face.weight, face.variable)
    return CACHE / "instances" / f"{face.path.stem}-{face.weight}.ttf"


@lru_cache(maxsize=None)
def _harfbuzz(path: str):
    import uharfbuzz

    face = uharfbuzz.Face(uharfbuzz.Blob.from_file_path(path))
    return uharfbuzz.Font(face), face.upem


@lru_cache(maxsize=65536)
def _shaped_width(text: str, path: str, size: float) -> float:
    """Width of ``text`` as HarfBuzz shapes it -- conjuncts, kerning and all.

    Devanagari is the reason this is shaped rather than summed: its
    conjuncts and vowel signs take far less room than their characters'
    separate advances, and a sum left every Hindi line short and opened gaps
    between differently styled words.
    """
    import uharfbuzz

    font, upem = _harfbuzz(path)
    buffer = uharfbuzz.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    uharfbuzz.shape(font, buffer, {})
    return sum(position.x_advance for position in buffer.glyph_positions
               ) * size / upem


def measure(text: str, size: float, stack: Sequence[Face]) -> float:
    """Shaped width of ``text`` at ``size`` px, run by run with fallback.

    Characters are grouped into runs by the first face in ``stack`` that
    has them, as the browser does with the CSS font-family list; a mark
    stays with the run of the letter it sits on.
    """
    runs: list[tuple[Optional[Face], str]] = []
    for char in text:
        face = _face_for(char, stack)
        if face is None or (runs and unicodedata.category(char)
                            in ("Mn", "Mc", "Me", "Cf")):
            face = runs[-1][0] if runs else face
        if runs and runs[-1][0] is face:
            runs[-1] = (face, runs[-1][1] + char)
        else:
            runs.append((face, char))
    return sum(_shaped_width(chunk, str(_static_path(face)), size)
               for face, chunk in runs if face is not None)


def _subset_face(face: Face, chars: str) -> bytes:
    """``face`` cut down to ``chars`` as WOFF2 bytes, deterministically."""
    from fontTools import subset

    from fontTools.ttLib import TTFont

    font = TTFont(str(_static_path(face)), recalcTimestamp=False)
    options = subset.Options()
    options.flavor = "woff2"
    options.hinting = False
    options.desubroutinize = True
    options.layout_features = ["*"]
    options.name_IDs = [1, 2]
    options.notdef_outline = True
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=chars)
    subsetter.subset(font)
    font.flavor = "woff2"
    font.recalcTimestamp = False
    buffer = io.BytesIO()
    font.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# RST subset: parsing and inline markup
# ---------------------------------------------------------------------------

_MARKER = re.compile(r"^\.\. spacr-(?P<name>[a-z-]+)-begin\s*$")
_UNDERLINE = re.compile(r"^([=~\-^])\1+\s*$")
_BULLET = re.compile(r"^[-*] ")
_SUBSTITUTION_ROW = re.compile(r"^(\|[^|\s][^|]*\|(\\ |\s)*)+$")


def _indented_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Indented lines from ``start`` (skipping leading blanks), dedented."""
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    body = []
    while index < len(lines) and (lines[index].startswith((" ", "\t"))
                                  or (not lines[index].strip()
                                      and index + 1 < len(lines)
                                      and lines[index + 1].startswith(" "))):
        body.append(lines[index])
        index += 1
    indent = min((len(line) - len(line.lstrip()) for line in body
                  if line.strip()), default=0)
    return [line[indent:] for line in body], index


def parse_rst(text: str) -> list[dict]:
    """Split the README's RST subset into blocks.

    Understands exactly what the README uses: section headings, paragraphs,
    ``-``/``*`` bullet lists, ``.. code-block::`` and ``::`` literal blocks,
    generated-marker regions, ``.. list-table::`` (the hardware table), and
    everything else that starts with ``..`` or is a row of substitution
    references, which is kept verbatim as ``raw``.
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    index = 0

    def add_raw(chunk: str) -> None:
        if blocks and blocks[-1]["kind"] == "raw":
            blocks[-1]["text"] += "\n\n" + chunk
        else:
            blocks.append({"kind": "raw", "text": chunk})

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        marker = _MARKER.match(line)
        if marker:
            name = marker.group("name")
            end = f".. spacr-{name}-end"
            stop = index + 1
            while stop < len(lines) and lines[stop].strip() != end:
                stop += 1
            if stop >= len(lines):
                raise ValueError(f"{line.strip()} has no {end}")
            inner = "\n".join(lines[index + 1:stop]).strip("\n")
            blocks.append({"kind": "marker", "name": name, "content": inner})
            index = stop + 1
            continue
        if line.startswith(".. code-block::"):
            language = line.split("::", 1)[1].strip() or "bash"
            body, index = _indented_block(lines, index + 1)
            blocks.append({"kind": "code", "lang": language,
                           "text": "\n".join(body).rstrip()})
            continue
        if line.startswith(".. list-table::"):
            body, index = _indented_block(lines, index + 1)
            blocks.append(_parse_list_table(body))
            continue
        if line.startswith(".. ") or line == "..":
            chunk = [line]
            index += 1
            while index < len(lines) and (
                    lines[index].startswith((" ", "\t"))
                    or (not lines[index].strip() and index + 1 < len(lines)
                        and lines[index + 1].startswith("   "))):
                chunk.append(lines[index])
                index += 1
            add_raw("\n".join(chunk).rstrip())
            continue
        if (index + 1 < len(lines) and _UNDERLINE.match(lines[index + 1])
                and not line.startswith(" ")):
            level = HEADING_CHARS[lines[index + 1][0]]
            blocks.append({"kind": "heading", "level": level,
                           "text": line.strip()})
            index += 2
            continue
        if line.startswith("| ") or _SUBSTITUTION_ROW.match(line.strip()):
            chunk = []
            while index < len(lines) and lines[index].strip():
                chunk.append(lines[index])
                index += 1
            add_raw("\n".join(chunk))
            continue
        if _BULLET.match(line):
            items = []
            while index < len(lines):
                current = lines[index]
                if _BULLET.match(current):
                    items.append(current[2:].strip())
                elif current.startswith("  ") and current.strip() and items:
                    items[-1] += " " + current.strip()
                elif not current.strip():
                    following = lines[index + 1] if index + 1 < len(lines) else ""
                    if not _BULLET.match(following):
                        break
                else:
                    break
                index += 1
            blocks.append({"kind": "list", "items": items})
            continue
        chunk = []
        while index < len(lines) and lines[index].strip():
            chunk.append(lines[index].strip())
            index += 1
        paragraph = " ".join(chunk)
        if paragraph.endswith("::"):
            if paragraph.strip() != "::":
                blocks.append({"kind": "paragraph", "text": paragraph})
            body, index = _indented_block(lines, index)
            blocks.append({"kind": "code", "lang": "bash",
                           "text": "\n".join(body).rstrip()})
            continue
        blocks.append({"kind": "paragraph", "text": paragraph})
    return blocks


def _parse_list_table(body: list[str]) -> dict:
    """``.. list-table::`` body (options and rows) to a table block."""
    widths: list[int] = []
    header_rows = 0
    rows: list[list[str]] = []
    legend_lines: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith(":widths:"):
            widths = [int(value) for value in stripped.split()[1:]]
        elif stripped.startswith(":header-rows:"):
            header_rows = int(stripped.split()[1])
        elif line.startswith("* - "):
            rows.append([line[4:].strip()])
        elif line.startswith("  - ") and rows:
            rows[-1].append(line[4:].strip())
        elif stripped and not line.startswith(" "):
            legend_lines.append(stripped)
    return {"kind": "table", "widths": widths, "header_rows": header_rows,
            "rows": rows}


_INLINE = re.compile(
    r"``(?P<literal>.+?)``"
    r"|\*\*(?P<strong>.+?)\*\*"
    r"|`(?P<label>[^`<]*?)\s*<(?P<target>[^<>`]+)>`__?"
    r"|`(?P<ref>[^`]+)`_(?![_\w])"
    r"|(?<![\w*])\*(?P<em>[^*\s][^*]*?)\*(?![\w*])"
)


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda match: "" if match.group(1) == " "
                  else match.group(1), text)


def _without_literal_marker(text: str) -> str:
    """RST's ``text::`` (a literal block follows) as it reads: ``text:``."""
    if text.endswith(" ::"):
        return text[:-3].rstrip()
    if text.endswith("::"):
        return text[:-1]
    return text


def inline_runs(text: str) -> list[tuple[str, str, Optional[str]]]:
    """RST inline markup to ``[(text, style, link)]``.

    ``style`` is one of ``plain``, ``strong``, ``em``, ``literal``, ``link``.
    A reference to a section (```Citing spaCR`_``) is plain text: the
    headings are images now, so the anchor it pointed at no longer exists.
    """
    runs: list[tuple[str, str, Optional[str]]] = []
    position = 0
    text = _without_literal_marker(text)
    for match in _INLINE.finditer(text):
        if match.start() > position:
            runs.append((_unescape(text[position:match.start()]), "plain", None))
        if match.group("literal") is not None:
            runs.append((match.group("literal"), "literal", None))
        elif match.group("strong") is not None:
            runs.append((_unescape(match.group("strong")), "strong", None))
        elif match.group("target") is not None:
            label = _unescape(match.group("label") or match.group("target"))
            runs.append((label, "link", match.group("target")))
        elif match.group("ref") is not None:
            runs.append((_unescape(match.group("ref")), "plain", None))
        else:
            runs.append((_unescape(match.group("em")), "em", None))
        position = match.end()
    if position < len(text):
        runs.append((_unescape(text[position:]), "plain", None))
    return [run for run in runs if run[0]]


def plain(text: str) -> str:
    """The visible text of RST inline markup."""
    return re.sub(r"\s+", " ",
                  "".join(run for run, _style, _link in inline_runs(text))
                  ).strip()


def links_of(text: str) -> list[tuple[str, str]]:
    """``[(label, target)]`` for every link in ``text``, in order."""
    return [(run, link) for run, style, link in inline_runs(text)
            if style == "link"]


def only_links(text: str) -> bool:
    """A paragraph that is nothing but links and separators."""
    runs = inline_runs(text)
    return bool(runs) and all(
        style == "link" or not re.sub(r"[\s·,|/–—]", "", run)
        for run, style, _link in runs) and any(
        style == "link" for _run, style, _link in runs)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def source_path(language: str) -> Path:
    return SOURCE_EN if language == "en" else SOURCE_DIR / f"{language}.yaml"


def readme_path(language: str) -> Path:
    return README if language == "en" else README_DIR / f"README.{language}.rst"


def blocks_to_yaml(blocks: list[dict]) -> list[dict]:
    """Blocks to the compact YAML form of the source files."""
    out = []
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            out.append({f"h{block['level']}": block["text"]})
        elif kind == "paragraph":
            out.append({"p": block["text"]})
        elif kind == "list":
            item = {"list": list(block["items"])}
            if block.get("marker"):
                item["marker"] = block["marker"]
            out.append(item)
        elif kind == "code":
            out.append({"code": block["text"], "lang": block["lang"]})
        elif kind == "raw":
            out.append({"raw": block["text"]})
        elif kind == "generated":
            item = {"generated": block["name"]}
            if block.get("details"):
                item["details"] = list(block["details"])
            out.append(item)
        else:
            raise ValueError(f"no YAML form for {kind}")
    return out


def yaml_to_blocks(items: list[dict]) -> list[dict]:
    blocks = []
    for item in items:
        if "p" in item:
            blocks.append({"kind": "paragraph", "text": item["p"]})
        elif "list" in item:
            blocks.append({"kind": "list", "items": list(item["list"]),
                           "marker": item.get("marker")})
        elif "code" in item:
            blocks.append({"kind": "code", "text": item["code"].rstrip("\n"),
                           "lang": item.get("lang", "bash")})
        elif "raw" in item:
            blocks.append({"kind": "raw", "text": item["raw"].rstrip("\n")})
        elif "generated" in item:
            block = {"kind": "generated", "name": item["generated"]}
            if item.get("details"):
                block["details"] = list(item["details"])
            blocks.append(block)
        else:
            (key, value), = item.items()
            match = re.fullmatch(r"h([1-4])", key)
            if not match:
                raise ValueError(f"unknown README source block {item!r}")
            blocks.append({"kind": "heading", "level": int(match.group(1)),
                           "text": value})
    return blocks


class _SourceDumper(yaml.SafeDumper):
    pass


def _represent_text(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_SourceDumper.add_representer(str, _represent_text)


def dump_source(language: str, blocks: list[dict]) -> str:
    header = (
        f"# spaCR README source ({language}). Edit this file, then run\n"
        "#   python tools/build_readme_images.py\n"
        "# which redraws docs/readme_images and rewrites the README.\n"
        "# Paragraphs keep RST inline markup; relative link targets are\n"
        "# repository-relative. Block order must match the English source.\n")
    body = yaml.dump({"language": language, "blocks": blocks_to_yaml(blocks)},
                     Dumper=_SourceDumper, allow_unicode=True,
                     sort_keys=False, width=79)
    return header + body


@lru_cache(maxsize=None)
def _load_source_cached(path: str, stamp: float) -> tuple:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(json.dumps(block, sort_keys=True)
                 for block in yaml_to_blocks(data["blocks"]))


def load_source(language: str) -> list[dict]:
    path = source_path(language)
    return [json.loads(block) for block in
            _load_source_cached(str(path), path.stat().st_mtime)]


def source_rst(language: str = "en", *, markers: Optional[dict] = None) -> str:
    """The README source as plain reStructuredText prose.

    What README.rst said in words before it became images: headings with
    their underlines, one paragraph per line, bullet lists, code blocks, and
    each generated block in the words its images were drawn from (its
    ``generated-<block>.rst`` sidecar). Tools and tests
    that read the README's wording (the translation reviewers, the command
    reference check, the editorial word ceiling) read this instead of the
    image README, so they keep checking the words the reader sees.
    """
    blocks = load_source(language)
    markers = _current_markers(language) if markers is None else markers
    markers = {name: _generated_words(language, name, content)
               for name, content in markers.items()}
    parts = []
    for index, block in enumerate(blocks):
        kind = block["kind"]
        previous = blocks[index - 1] if index else {}
        literal = (previous.get("kind") == "paragraph"
                   and previous["text"].endswith("::"))
        if kind == "heading":
            text = block["text"]
            width = sum(2 if unicodedata.east_asian_width(char) in "WF"
                        else 1 for char in text)
            parts.append(f"{text}\n{HEADING_UNDERLINE[block['level']] * width}")
        elif kind == "paragraph":
            parts.append(block["text"])
        elif kind == "list":
            bullet = "*" if block.get("marker") else "-"
            body = "\n".join(f"{bullet} {item}" for item in block["items"])
            if block.get("marker"):
                body = (f".. spacr-{block['marker']}-begin\n\n{body}\n\n"
                        f".. spacr-{block['marker']}-end")
            parts.append(body)
        elif kind == "code":
            indent = "    " if literal else "   "
            code = "\n".join((indent + line) if line else ""
                             for line in block["text"].splitlines())
            parts.append(code if literal
                         else f".. code-block:: {block['lang']}\n\n{code}")
        elif kind == "raw":
            parts.append(block["text"])
        elif kind == "generated":
            content = markers.get(block["name"], "")
            parts.append(f".. spacr-{block['name']}-begin\n\n"
                         f"{content.strip()}\n\n.. spacr-{block['name']}-end")
    return "\n\n".join(part.rstrip() for part in parts) + "\n"


def _generated_words(language: str, name: str, content: str) -> str:
    """A generated block's words: the markup its images were drawn from."""
    sidecar = IMAGE_DIR / language / f"generated-{name}.rst"
    return sidecar.read_text(encoding="utf-8") if sidecar.is_file() \
        else content


def _signature(block: dict) -> tuple:
    kind = block["kind"]
    if kind == "heading":
        return (kind, block["level"])
    if kind == "generated":
        return (kind, block["name"], len(block.get("details", [])))
    if kind == "list":
        return (kind, len(block["items"]), block.get("marker"))
    return (kind,)


def check_structure(language: str) -> list[str]:
    """Differences between a translation's block sequence and English's."""
    english = [_signature(block) for block in load_source("en")]
    local = [_signature(block) for block in load_source(language)]
    problems = []
    if len(english) != len(local):
        problems.append(f"{language}: {len(local)} blocks, English has "
                        f"{len(english)}")
    for index, (left, right) in enumerate(zip(english, local)):
        if left != right:
            problems.append(f"{language}: block {index} is {right}, English "
                            f"has {left}")
            break
    return problems


# ---------------------------------------------------------------------------
# Grouping blocks into images
# ---------------------------------------------------------------------------

def _block_links(block: dict) -> list[str]:
    if block["kind"] == "paragraph":
        return [target for _label, target in links_of(block["text"])]
    if block["kind"] == "list":
        return [target for item in block["items"]
                for _label, target in links_of(item)]
    return []


def _one_link_per_item(block: dict) -> bool:
    items = block["items"]
    targets = [links_of(item) for item in items]
    return (len(items) > 1 and all(len({t for _l, t in found}) == 1
                                   for found in targets)
            and len({found[0][1] for found in targets}) == len(items))


def plan(blocks: list[dict]) -> list[dict]:
    """Turn source blocks into README units: panels, chip rows, code, raw.

    Every unit records ``span``, the indices of the source blocks it draws,
    so a translation is grouped exactly like English when its blocks align.
    """
    units: list[dict] = []
    current: Optional[dict] = None

    def flush():
        nonlocal current
        if current is not None:
            units.append(current)
            current = None

    for index, block in enumerate(blocks):
        kind = block["kind"]
        if kind in ("code", "raw", "generated"):
            flush()
            units.append({"kind": kind, "block": block, "span": [index]})
            continue
        if kind == "paragraph" and only_links(block["text"]):
            flush()
            units.append({"kind": "chips", "block": block, "span": [index],
                          "links": links_of(block["text"])})
            continue
        if kind == "list" and (block.get("marker") or _one_link_per_item(block)):
            flush()
            units.append({"kind": "rows", "block": block, "span": [index]})
            continue
        targets = list(dict.fromkeys(_block_links(block)))
        if kind == "heading" and block["level"] <= 3 and not (
                current and all(item["kind"] == "heading"
                                for item in current["blocks"])):
            flush()
        if current is not None:
            mine = set(current["targets"])
            if mine and targets and set(targets) != mine:
                flush()
        if current is None:
            current = {"kind": "panel", "blocks": [], "span": [],
                       "targets": []}
        current["blocks"].append(block)
        current["span"].append(index)
        current["targets"] = list(dict.fromkeys(current["targets"] + targets))
        if len(targets) > 1:
            flush()
    flush()
    return units


def alt_text(blocks: Iterable[dict]) -> str:
    """The full visible text of a panel, as its ``:alt:`` carries it."""
    parts = []
    for block in blocks:
        if block["kind"] == "heading":
            text = plain(block["text"])
            parts.append(text if text[-1:] in ".:!?。：" else text + ".")
        elif block["kind"] == "paragraph":
            parts.append(plain(block["text"]))
        elif block["kind"] == "list":
            parts.extend(plain(item) for item in block["items"])
        elif block["kind"] == "table":
            parts.append(table_text(block))
    return " ".join(part for part in parts if part)


def slug(text: str) -> str:
    value = unicodedata.normalize("NFKD", plain(text)).encode(
        "ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "section"


def image_ids(english_units: list[dict], english_blocks: list[dict]) -> list:
    """Stable file names for each unit, from the English headings."""
    ids: list = []
    section = "intro"
    seen: dict[str, int] = {}
    for unit in english_units:
        first = english_blocks[unit["span"][0]]
        if unit["kind"] == "panel" and first["kind"] == "heading":
            section = slug(first["text"])
        if unit["kind"] in ("panel", "chips", "rows"):
            seen[section] = seen.get(section, 0) + 1
            count = seen[section]
            ids.append(section if count == 1 else f"{section}-{count}")
        else:
            ids.append(None)
    return ids


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

#: Scripts that may break between any two characters. Not Hangul: Korean
#: is set with spaces between words and breaks only there.
_CJK = re.compile("[\\u2e80-\\u9fff\\uf900-\\ufaff\\uff00-\\uffef\\u3000-\\u303f]")
_NO_LINE_START = set("，。、；：！？）」』】》〉,.;:!?)]}%…·")
_NO_LINE_END = set("（「『【《〈([{")


@dataclass
class Style:
    size: float
    weight: int
    color: str
    italic: bool = False
    literal: bool = False


def _styles(base_size: float, base_weight: int, base_color: str) -> dict:
    return {
        "plain": Style(base_size, base_weight, base_color),
        "strong": Style(base_size, 400, PALETTE["ink_primary"]),
        "em": Style(base_size, base_weight, base_color, italic=True),
        "literal": Style(LITERAL_SIZE * base_size / BODY, 400,
                         PALETTE["accent"], literal=True),
        "link": Style(base_size, 400, PALETTE["accent_blue"]),
    }


@dataclass
class Piece:
    text: str
    style: Style
    width: float = 0.0


def _break_units(runs: list[tuple[str, Style]]) -> list[list[Piece]]:
    """Unbreakable units; a break is allowed after each unit."""
    chars = [(char, style) for text, style in runs for char in text]
    units: list[list[tuple[str, Style]]] = [[]]
    for position, (char, style) in enumerate(chars):
        units[-1].append((char, style))
        following = chars[position + 1][0] if position + 1 < len(chars) else ""
        if not following:
            continue
        breakable = False
        if char == " " and following != " ":
            breakable = True
        elif (_CJK.match(char) or _CJK.match(following)) and following != " ":
            breakable = (following not in _NO_LINE_START
                         and char not in _NO_LINE_END
                         and not (char.isascii() and following.isascii()))
        if breakable:
            units.append([])
    pieces = []
    for unit in units:
        merged: list[Piece] = []
        for char, style in unit:
            if merged and merged[-1].style is style:
                merged[-1].text += char
            else:
                merged.append(Piece(char, style))
        pieces.append(merged)
    return [unit for unit in pieces if unit]


def _piece_width(piece: Piece, language: str) -> float:
    stack = face_stack(language, piece.style.weight, piece.style.italic)
    return measure(piece.text, piece.style.size, stack)


def wrap(runs: list[tuple[str, Style]], width: float,
         language: str) -> list[list[Piece]]:
    """Greedy line filling over break units, against real advance widths."""
    lines: list[list[Piece]] = [[]]
    used = 0.0
    for unit in _break_units(runs):
        for piece in unit:
            piece.width = _piece_width(piece, language)
        unit_width = sum(piece.width for piece in unit)
        unit_width += sum(2 * LITERAL_PAD for piece in unit
                          if piece.style.literal)
        trailing = unit[-1].text[len(unit[-1].text.rstrip(" ")):]
        trailing_width = (_piece_width(Piece(trailing, unit[-1].style), language)
                          if trailing else 0.0)
        if lines[-1] and used + unit_width - trailing_width > width:
            lines.append([])
            used = 0.0
        lines[-1].extend(unit)
        used += unit_width
    return [_merge(line) for line in lines if line]


def _merge(line: list[Piece]) -> list[Piece]:
    merged: list[Piece] = []
    for piece in line:
        if merged and merged[-1].style is piece.style:
            merged[-1].text += piece.text
            merged[-1].width += piece.width
        else:
            merged.append(Piece(piece.text, piece.style, piece.width))
    if merged:
        last = merged[-1]
        stripped = last.text.rstrip(" ")
        if stripped != last.text:
            last.text = stripped
    return [piece for piece in merged if piece.text]


class Canvas:
    """Accumulates SVG elements and the characters each face must carry."""

    def __init__(self, language: str):
        self.language = language
        self.elements: list[str] = []
        self.chars: dict[tuple, set] = {}

    def use(self, text: str, style: Style) -> None:
        stack = face_stack(self.language, style.weight, style.italic)
        for char in text:
            face = _face_for(char, stack)
            if face is None:
                if unicodedata.category(char) in ("Mn", "Mc", "Me", "Cf"):
                    face = stack[1] if len(stack) > 2 else stack[0]
                elif char.strip():
                    raise ValueError(
                        f"no {self.language} font draws {char!r} "
                        f"(U+{ord(char):04X})")
                else:
                    continue
            self.chars.setdefault(
                (face.family, face.weight, face.italic, str(face.path),
                 face.variable), set()).add(char)

    def text(self, x: float, y: float, pieces: list[Piece]) -> None:
        """One line of pieces; each piece starts at its own measured x."""
        spans = []
        cursor = x
        for piece in pieces:
            style = piece.style
            if style.literal:
                height = style.size + 9
                self.elements.append(
                    f'<rect x="{cursor:.1f}" y="{y - style.size * 1.02:.1f}" '
                    f'width="{piece.width + 2 * LITERAL_PAD:.1f}" '
                    f'height="{height:.1f}" rx="5" fill="{PALETTE["accent"]}" '
                    f'fill-opacity=".12"/>')
                cursor += LITERAL_PAD
            self.use(piece.text, style)
            attributes = (f'x="{cursor:.1f}" font-size="{style.size:g}" '
                          f'font-weight="{style.weight}" fill="{style.color}"')
            if style.italic:
                attributes += ' font-style="italic"'
            spans.append(f'<tspan {attributes}>{escape(piece.text)}</tspan>')
            cursor += piece.width
            if style.literal:
                cursor += LITERAL_PAD
        self.elements.append(
            f'<text y="{y:.1f}" xml:space="preserve">{"".join(spans)}</text>')

    def font_faces(self) -> str:
        rules = []
        for (family, weight, italic, path, variable), chars in sorted(
                self.chars.items(), key=lambda item: item[0]):
            face = Face(family, weight, italic, Path(path), variable)
            data = base64.b64encode(
                _subset_face(face, "".join(sorted(chars)))).decode("ascii")
            style = "italic" if italic else "normal"
            rules.append(
                f"@font-face{{font-family:'{family}';font-weight:{weight};"
                f"font-style:{style};src:url(data:font/woff2;base64,{data}) "
                "format('woff2')}")
        return "".join(rules)


def _svg(canvas: Canvas, width: float, height: float, title: str,
         description: str, digest: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" '
        f'height="{height:g}" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-labelledby="t d" data-source-sha256="{digest}">\n'
        f'<title id="t">{escape(title)}</title>\n'
        f'<desc id="d">{escape(description)}</desc>\n'
        f"<style>{canvas.font_faces()}text{{font-family:{FAMILY_STACK};"
        "font-kerning:normal}</style>\n"
        f"{body}\n</svg>\n")


def _surface(x: float, y: float, width: float, height: float,
             radius: float, uid: str) -> str:
    """The card surface: fill, hairline edge and the cards' faint shadow."""
    return (
        f'<filter id="s{uid}" x="-5%" y="-20%" width="110%" height="140%">'
        f'<feDropShadow dx="0" dy="1" stdDeviation="1.2" '
        f'flood-color="{PALETTE["shadow"]}" flood-opacity=".22"/></filter>'
        f'<rect x="{x + .5:g}" y="{y + .5:g}" width="{width - 1:g}" '
        f'height="{height - 1:g}" rx="{radius:g}" fill="{PALETTE["surface"]}" '
        f'stroke="{PALETTE["edge"]}" filter="url(#s{uid})"/>')


def _panel_frame(panel_width: float, height: float, glow: bool,
                 uid: str) -> str:
    width = panel_width - 2 * MARGIN
    parts = [_surface(MARGIN, 1, width, height, RADIUS, uid)]
    if glow:
        parts.insert(0, (
            f'<defs><linearGradient id="g{uid}" x1="0" x2="1" y1="0" y2="0">'
            f'<stop offset="0" stop-color="{PALETTE["accent"]}"/>'
            f'<stop offset="1" stop-color="{PALETTE["accent_blue"]}"/>'
            f'</linearGradient><clipPath id="c{uid}"><rect x="{MARGIN}" y="1" '
            f'width="{width:g}" height="{height}" rx="{RADIUS}"/></clipPath>'
            "</defs>"))
        parts.append(
            f'<rect x="{MARGIN}" y="1" width="{width:g}" height="2" '
            f'fill="url(#g{uid})" fill-opacity=".7" clip-path="url(#c{uid})"/>')
    return "".join(parts)


def digest_of(payload) -> str:
    return hashlib.sha256(json.dumps(
        [RENDERER_VERSION, PALETTE, payload], sort_keys=True,
        ensure_ascii=False).encode("utf-8")).hexdigest()


def table_text(block: dict) -> str:
    """The hardware table as sentences, for alt text."""
    rows = block["rows"]
    header, body = rows[0], rows[1:]
    sentences = []
    for row in body:
        cells = ", ".join(f"{name} {_cell_text(cell)}"
                          for name, cell in zip(header[1:], row[1:]))
        sentences.append(f"{row[0]}: {cells}.")
    legend = block.get("legend")
    if legend:
        sentences.append(_cell_text(legend) + ".")
    return " ".join(sentences)


_MARKS = {"\U0001F7E2": "stable", "\U0001F7E3": "beta", "\U0001F534": "cpu"}


def _cell_text(cell: str) -> str:
    text = cell
    for mark in _MARKS:
        text = text.replace(mark, "")
    return re.sub(r"\s+", " ", text).strip()


def _cell_mark(cell: str) -> Optional[str]:
    for mark, name in _MARKS.items():
        if mark in cell:
            return name
    return None


def _dot(x: float, y: float, kind: str) -> str:
    if kind == "stable":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{PALETTE["accent"]}"/>'
    if kind == "beta":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{PALETTE["beta"]}"/>'
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.9" fill="none" '
            f'stroke="{PALETTE["ink_muted"]}" stroke-width="1.3"/>')


# ---------------------------------------------------------------------------
# Image renderers
# ---------------------------------------------------------------------------

def render_panel(blocks: list[dict], language: str, uid: str,
                 digest: str, *, linked: bool = False,
                 row: bool = False, width: float = PHONE) -> str:
    """One panel ``width`` CSS pixels wide: headings, paragraphs, lists and
    tables, top to bottom, every word at its true display size.

    ``linked`` draws the small outward arrow in the top right corner that
    says the whole panel is a link. ``row`` is one item of a list of links:
    compact, without a bullet, the arrow centred on the row.
    """
    canvas = Canvas(language)
    side = pad_x(width)
    content = width - 2 * MARGIN - 2 * side - (20 if linked else 0)
    left = MARGIN + side
    headings_only = all(block["kind"] == "heading" for block in blocks)
    compact = row or headings_only
    y = pad_compact(width) if compact else pad_y(width)
    glow = bool(blocks) and blocks[0]["kind"] == "heading" and \
        blocks[0]["level"] <= 3
    for position, block in enumerate(blocks):
        if position:
            y += BLOCK_GAP
        kind = block["kind"]
        if kind == "heading":
            size, line = HEADINGS[block["level"]]
            color = (PALETTE["ink_primary"] if block["level"] <= 3
                     else PALETTE["ink_secondary"])
            styles = _styles(size, 400, color)
            runs = [(text, styles[style if style != "link" else "plain"])
                    for text, style, _link in inline_runs(block["text"])]
            for pieces in wrap(runs, content, language):
                y += line
                canvas.text(left, _baseline(y, line, size), pieces)
            if block["level"] == 2:
                y += 6
                canvas.elements.append(
                    f'<rect x="{left}" y="{y:.1f}" width="24" height="2" '
                    f'rx="1" fill="{PALETTE["accent"]}"/>')
                y += 2
        elif kind == "paragraph":
            styles = _styles(BODY, 300, PALETTE["ink_secondary"])
            runs = [(text, styles[style])
                    for text, style, _link in inline_runs(block["text"])]
            for pieces in wrap(runs, content, language):
                y += LINE
                canvas.text(left, _baseline(y, LINE, BODY), pieces)
        elif kind == "list":
            styles = _styles(BODY, 300, PALETTE["ink_secondary"])
            for item_index, item in enumerate(block["items"]):
                if item_index:
                    y += 4
                runs = [(text, styles[style])
                        for text, style, _link in inline_runs(item)]
                indent = 0 if row else BULLET_INDENT
                lines = wrap(runs, content - indent, language)
                if not row:
                    canvas.elements.append(
                        f'<circle cx="{left + 4}" '
                        f'cy="{_baseline(y + LINE, LINE, BODY) - BODY * .33:.1f}" '
                        f'r="2.2" fill="{PALETTE["accent"]}"/>')
                for pieces in lines:
                    y += LINE
                    canvas.text(left + indent, _baseline(y, LINE, BODY),
                                pieces)
        elif kind == "table":
            y = _render_table(canvas, block, left, y, content)
    height = round(y + (pad_compact(width) if compact else pad_y(width)))
    frame = _panel_frame(width, height, glow, uid)
    if linked:
        arrow_y = height / 2 if row else pad_y(width) + 8
        canvas.elements.append(_out_arrow(width - MARGIN - side + 4, arrow_y,
                                          PALETTE["accent"]))
    title = plain(blocks[0]["text"]) if blocks and blocks[0]["kind"] == \
        "heading" else "spaCR"
    return _svg(canvas, width, height + GAP_BELOW, title, alt_text(blocks),
                digest, frame + "".join(canvas.elements))


def _baseline(bottom: float, line: float, size: float) -> float:
    """The baseline of a line box ending at ``bottom``: Open Sans' ascent
    and descent (1.069 and 0.293 em) centred in the line, as CSS does."""
    return bottom - (line - 1.362 * size) / 2 - 0.293 * size


def _cell_lines(text: str, style: Style, width: float,
                language: str) -> list[list[Piece]]:
    return wrap([(text, style)], width, language) or [[]]


def _render_table(canvas: Canvas, block: dict, left: float, y: float,
                  content: float) -> float:
    """The table, its cells word-wrapped to their columns, then its legend
    flowing onto as many lines as the width needs."""
    rows = block["rows"]
    widths = block.get("widths") or [1] * len(rows[0])
    total = sum(widths)
    header_style = _styles(BODY, 400, PALETTE["ink_muted"])["plain"]
    cell_style = _styles(BODY, 300, PALETTE["ink_secondary"])["plain"]
    name_style = _styles(BODY, 400, PALETTE["ink_primary"])["plain"]
    gap = 6
    mark_width = 14

    def words(cell: str) -> str:
        return re.sub(r" (?=\d+\b)", "\u00a0", _cell_text(cell))

    def style_of(row_index: int, column: int) -> Style:
        if row_index < block.get("header_rows", 1):
            return header_style
        return name_style if column == 0 else cell_style

    # Each column at least as wide as its widest unbreakable word, the
    # rest shared as the table's :widths: say.
    shares = [content * value / total for value in widths]
    least = [0.0] * len(shares)
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            for unit in _break_units([(words(cell),
                                       style_of(row_index, column))]):
                text = "".join(piece.text for piece in unit).rstrip(" ")
                width = measure(text, BODY, face_stack(
                    canvas.language, style_of(row_index, column).weight))
                least[column] = max(least[column], width + gap + (
                    mark_width if _cell_mark(cell) else 0))
    short = sum(max(0.0, need - have) for need, have in zip(least, shares))
    spare = sum(max(0.0, have - need) for need, have in zip(least, shares))
    if short and spare:
        shares = [max(need, have - (have - need) * short / spare)
                  if have > need else need
                  for need, have in zip(least, shares)]
    columns = []
    x = left
    for share in shares:
        columns.append((x, share))
        x += share
    y += 4
    for row_index, row in enumerate(rows):
        header = row_index < block.get("header_rows", 1)
        cells = []
        for column, cell in enumerate(row):
            start, span = columns[column]
            mark = _cell_mark(cell)
            style = header_style if header else (
                name_style if column == 0 else cell_style)
            room = span - gap - (mark_width if mark else 0)
            cells.append((start, mark, _cell_lines(
                words(cell), style, room, canvas.language)))
        count = max(len(lines) for _start, _mark, lines in cells)
        row_height = count * LINE + 10
        if not header and row_index % 2 == 0:
            canvas.elements.append(
                f'<rect x="{left - 8}" y="{y:.1f}" width="{content + 16:.1f}" '
                f'height="{row_height:.1f}" rx="6" fill="{PALETTE["track"]}" '
                'fill-opacity=".55"/>')
        for start, mark, lines in cells:
            x = start
            first = _baseline(y + 5 + LINE, LINE, BODY)
            if mark:
                canvas.elements.append(_dot(x + 4.5, first - BODY * .35, mark))
                x += mark_width
            for index, pieces in enumerate(lines):
                if pieces:
                    canvas.text(x, first + index * LINE, pieces)
        y += row_height
        if header:
            canvas.elements.append(
                f'<rect x="{left}" y="{y - 1:.1f}" width="{content:.1f}" '
                f'height="1" fill="{PALETTE["edge"]}"/>')
    legend = block.get("legend")
    if legend:
        y += 8
        style = _styles(BODY, 300, PALETTE["ink_muted"])["plain"]
        x = left
        line_top = y
        for part in re.split("\\s*\u2003\\s*", legend):
            mark = _cell_mark(part)
            piece = Piece(_cell_text(part), style)
            piece.width = _piece_width(piece, canvas.language)
            needed = piece.width + (mark_width if mark else 0)
            if x > left and x + needed > left + content:
                x = left
                line_top += LINE
            baseline = _baseline(line_top + LINE, LINE, BODY)
            if mark:
                canvas.elements.append(_dot(x + 4.5, baseline - BODY * .35,
                                            mark))
                x += mark_width
            canvas.text(x, baseline, [piece])
            x += piece.width + 18
        y = line_top + LINE
    return y


def _globe(x: float, y: float, color: str) -> str:
    return (
        f'<g fill="none" stroke="{color}" stroke-width="1.4">'
        f'<circle cx="{x}" cy="{y}" r="6.5"/>'
        f'<ellipse cx="{x}" cy="{y}" rx="2.9" ry="6.5"/>'
        f'<path d="M{x - 6.5} {y}h13M{x - 5.6} {y - 3.25}h11.2M{x - 5.6} {y + 3.25}h11.2"/>'
        "</g>")


def _caret(x: float, y: float, color: str) -> str:
    return (f'<path d="M{x - 5} {y - 2.5}l5 5 5-5" fill="none" '
            f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>')


def _out_arrow(x: float, y: float, color: str) -> str:
    return (f'<path d="M{x - 4} {y + 4}l8-8M{x - 2.5} {y - 4}h6.5v6.5" '
            f'fill="none" stroke="{color}" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round"/>')


#: The side padding inside a button and the room its arrow takes.
CHIP_PAD = 14
CHIP_ICON = 18
CHIP_PILL = 30


def chip_text_room(cell: float, icon: Optional[str] = "arrow") -> float:
    """The width a label may take inside a button filling ``cell``."""
    return cell - GUTTER - 2 * CHIP_PAD - (CHIP_ICON if icon else 0)


def render_chip(label: str, language: str, width: float, digest: str,
                description: str, *, align: str = "left",
                icon: str = "arrow", weight: int = 400,
                first: bool = True, last: bool = True) -> str:
    """One link button: a rounded chip filling a fixed-width grid cell.

    ``width`` is the cell in CSS pixels, so the label is drawn at the
    page's one text size. The row's outer buttons run to the cell's outer
    edge, so a row starts and ends where the panels do; between two
    buttons each gives up half of :data:`GUTTER`.
    """
    canvas = Canvas(language)
    inset = 0 if first else GUTTER / 2
    right = 0 if last else GUTTER / 2
    chip_width = width - inset - right
    style = Style(CHIP_TEXT, weight, PALETTE["ink_primary"])
    space = chip_width - 2 * CHIP_PAD - (CHIP_ICON if icon else 0)
    size = CHIP_TEXT
    text_width = measure(label, size, face_stack(language, weight))
    while text_width > space and size > 10:
        size -= 0.2
        text_width = measure(label, size, face_stack(language, weight))
    style.size = round(size, 1)
    piece = Piece(label, style, text_width)
    top, height = 1, CHIP_PILL
    middle = top + height / 2
    elements = [_surface(inset, top, chip_width, height, height / 2,
                         "chip")]
    if icon == "globe":
        elements.append(_globe(inset + CHIP_PAD + 7, middle,
                               PALETTE["accent"]))
        text_x = inset + CHIP_PAD + 20
        elements.append(_caret(text_x + text_width + 12, middle,
                               PALETTE["ink_muted"]))
    elif align == "right":
        text_x = inset + chip_width - CHIP_PAD - text_width
    elif align == "center":
        text_x = inset + (chip_width - text_width) / 2
    else:
        text_x = inset + CHIP_PAD
        if icon == "arrow":
            elements.append(_out_arrow(inset + chip_width - CHIP_PAD - 4,
                                       middle, PALETTE["accent"]))
    canvas.text(text_x, middle + size * 0.36, [piece])
    body = "".join(elements) + "".join(canvas.elements)
    return _svg(canvas, width, top + height + GAP_BELOW, label, description,
                digest, body)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@dataclass
class Output:
    """What one build of one README produces."""
    language: str
    text: str = ""
    images: dict = field(default_factory=dict)
    pages: dict = field(default_factory=dict)


def _readable_target(language: str, name: str) -> str:
    return f"docs/readme_text/{language}/{name}.md"


def _readable_inline(text: str, language: str, *, generated: bool = False) -> str:
    """Preserve the image's inline words and rebase its useful detail links."""
    parts = []
    for value, style, target in inline_runs(text):
        label = re.sub(r"([\\`*_[\]<>|])", r"\\\1", value)
        if style == "literal":
            fence = "`" * (max((len(run) for run in re.findall(r"`+", value)),
                              default=0) + 1)
            parts.append(f"{fence} {value} {fence}")
        elif style == "link":
            if generated:
                target = _to_repository(target, language)
            if not re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) and not \
                    target.startswith("//"):
                path, separator, fragment = target.partition("#")
                source = ROOT / path if path else readme_path(language)
                target = os.path.relpath(source, TEXT_DIR / language).replace(
                    os.sep, "/") + (separator + fragment if separator else "")
            parts.append(f"[{label}](<{target}>)")
        elif style == "strong":
            parts.append(f"<strong>{escape(value)}</strong>")
        elif style == "em":
            parts.append(f"<em>{escape(value)}</em>")
        else:
            parts.append(label)
    return "".join(parts)


def readable_page(blocks: list[dict], language: str, *,
                  generated: bool = False) -> str:
    """Copyable structured text from exactly the blocks drawn in a panel.

    Each panel owns a separate page, including shared and external detail
    destinations. No navigation label precedes the panel's substantive text.
    """
    def inline(value):
        return _readable_inline(value, language, generated=generated)

    parts = []
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            parts.append("#" * block["level"] + " " + inline(block["text"]))
        elif kind == "paragraph":
            parts.append(inline(block["text"]))
        elif kind == "list":
            parts.append("\n".join("- " + inline(item) for item in block["items"]))
        elif kind == "table":
            rows = block["rows"]
            if rows:
                def row(cells):
                    return "| " + " | ".join(inline(cell) for cell in cells) + " |"
                head = rows[0] if block.get("header_rows") else [""] * len(rows[0])
                body = rows[1:] if block.get("header_rows") else rows
                parts.append("\n".join([row(head), row(["---"] * len(head)),
                                         *(row(cells) for cells in body)]))
            if block.get("legend"):
                parts.append(inline(block["legend"]))
        else:
            raise ValueError(f"Cannot publish a readable {kind} panel")
    return "\n\n".join(parts) + "\n"


def _relative(target: str, language: str) -> str:
    """A repository-relative target as the README at ``language`` sees it."""
    if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(
            ("#", "//")):
        return target
    if language == "en":
        return target
    path, _, anchor = target.partition("#")
    relative = os.path.relpath(ROOT / path, README_DIR).replace(os.sep, "/")
    return relative + (f"#{anchor}" if anchor else "")


def _to_repository(target: str, language: str) -> str:
    """A target relative to the README at ``language``, made repo-relative."""
    if language == "en" or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) \
            or target.startswith(("#", "//")):
        return target
    path, _, anchor = target.partition("#")
    relative = os.path.relpath(os.path.normpath(README_DIR / path),
                               ROOT).replace(os.sep, "/")
    return relative + (f"#{anchor}" if anchor else "")


def _image_reference(language: str, name: str) -> str:
    path = f"docs/readme_images/{language}/{name}.svg"
    return _relative(path, language)


def _alt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _html(value: str) -> str:
    return escape(value, quote=True)


def picture_html(phone: str, desktop: str, alt: str,
                 target: Optional[str] = None) -> str:
    """One image in both layouts: the phone drawing is the ``<img>``, the
    desktop drawing a ``<source>`` chosen by :data:`DESKTOP_MEDIA`.

    No ``width`` attribute: each drawing is shown at its own size, one SVG
    unit to one CSS pixel, which is what keeps every word the same size on
    every page. GitHub keeps ``<picture>``, ``<source media srcset>`` and
    the ``<img>`` (checked against its Markdown renderer on 2026-10-13).
    """
    image = (f'<picture><source media="{DESKTOP_MEDIA}" '
             f'srcset="{_html(desktop)}"><img src="{_html(phone)}" '
             f'alt="{_html(_alt(alt))}"></picture>')
    return f'<a href="{_html(target)}">{image}</a>' if target else image


def raw_html(*rows: str) -> str:
    """A ``raw:: html`` block, one ``<div>`` per row of images, written
    without whitespace between images so a row never gains a space."""
    body = "\n".join(f"   <div>{row}</div>" for row in rows)
    return f".. raw:: html\n\n{body}"


def _image_directive(language: str, name: str, alt: str,
                     target: Optional[str]) -> str:
    """A full-width image as a phone/desktop picture."""
    return raw_html(picture_html(
        _image_reference(language, name),
        _image_reference(language, f"{name}.desktop"), alt,
        _relative(target, language) if target else None))


def estimate_width(text: str, size: float = BODY, weight: int = 400) -> float:
    """A label's width without the fonts, for choosing columns in
    ``--check`` (which needs no fonts): Open Sans' mean advance, generous
    for capitals and wide scripts."""
    total = 0.0
    for char in text:
        if unicodedata.east_asian_width(char) in "WF":
            total += 1.0
        elif char.isupper() or char in "mwMW@%":
            total += 0.72
        elif char in "ilj.,:;'|!I ":
            total += 0.3
        else:
            total += 0.57 + (0.01 if weight == 400 else 0)
    return total * size


def row_cell(layout_width: int, columns: int) -> int:
    """The width of one of ``columns`` equal cells across a row."""
    return (layout_width - ROW_SLACK) // columns


def fitting_columns(labels: Sequence[str], wanted: int, layout_width: int,
                    icon: Optional[str] = "arrow") -> int:
    """The most columns, up to ``wanted``, in which every label fits at the
    page's text size."""
    columns = wanted
    while columns > 1 and any(
            estimate_width(label) > chip_text_room(
                row_cell(layout_width, columns), icon)
            for label in labels):
        columns -= 1
    return columns


def _chip_grid(language: str, prefix: str, links: list[tuple[str, str]],
               images: dict, render: bool, *, columns: int = 0,
               nav: bool = False) -> str:
    """Link buttons in a grid; one image, one link, per button.

    Each layout picks its own column count (the most, up to ``columns``,
    in which every label fits at the page's text size), so one row of
    cells fills the column exactly and the browser wraps the next button
    onto the next row.
    """
    columns = columns or (2 if len(links) <= 2 else 3)
    labels = [label for label, _target in links]
    icon = None if nav else "arrow"
    cells = []
    for position, (label, target) in enumerate(links):
        name = f"{prefix}-{position + 1}"
        align = "left"
        if nav:
            align = "left" if position == 0 else "right"
        for layout, layout_width in LAYOUTS.items():
            count = fitting_columns(labels, columns, layout_width, icon)
            cell = row_cell(layout_width, count)
            column = position % count
            first, last = column == 0, column == count - 1
            key = name if layout == "phone" else f"{name}.desktop"
            digest = digest_of(["chip", language, label, target, cell, align,
                                first, last])
            if render:
                images[key] = render_chip(label, language, cell, digest,
                                          label, align=align, icon=icon,
                                          first=first, last=last)
            else:
                images[key] = digest
        cells.append(picture_html(
            _image_reference(language, name),
            _image_reference(language, f"{name}.desktop"), label,
            _relative(target, language)))
    return raw_html("".join(cells))


def _panel_images(language: str, name: str, blocks: list[dict],
                  images: dict, render: bool, *, linked: bool,
                  row: bool = False, kind: str = "panel") -> None:
    for layout, layout_width in LAYOUTS.items():
        key = name if layout == "phone" else f"{name}.desktop"
        digest = digest_of([kind, language, blocks, layout_width])
        if render:
            images[key] = render_panel(blocks, language, name, digest,
                                       linked=linked, row=row,
                                       width=layout_width)
        else:
            images[key] = digest


def _panel_markup(language: str, name: str, blocks: list[dict],
                  images: dict, render: bool, *, pages: Optional[dict] = None,
                  generated: bool = False) -> str:
    targets = list(dict.fromkeys(
        target for block in blocks for target in _block_links(block)))
    _panel_images(language, name, blocks, images, render,
                  linked=len(targets) == 1)
    if pages is not None:
        pages[name] = readable_page(blocks, language, generated=generated)
    markup = _image_directive(language, name, alt_text(blocks),
                              _readable_target(language, name))
    if len(targets) > 1:
        links = []
        for block in blocks:
            for text in ([block["text"]] if block["kind"] in
                         ("paragraph", "heading") else block.get("items", [])):
                links.extend(links_of(text))
        unique = list(dict.fromkeys(links))
        markup += "\n\n" + _chip_grid(language, f"{name}-link", unique,
                                      images, render)
    return markup


def _rows_markup(language: str, name: str, block: dict, images: dict,
                 render: bool, *, pages: Optional[dict] = None) -> list[str]:
    """A list where every row is its own linked image."""
    parts = []
    for position, item in enumerate(block["items"]):
        row = {"kind": "list", "items": [item]}
        found = links_of(item)
        target = found[0][1] if len(found) == 1 else None
        row_name = f"{name}-{position + 1}"
        _panel_images(language, row_name, [row], images, render,
                      linked=target is not None, row=True, kind="row")
        if pages is not None:
            pages[row_name] = readable_page([row], language)
        parts.append(_image_directive(language, row_name, alt_text([row]),
                                      _readable_target(language, row_name)))
        if len(found) > 1:
            parts.append(_chip_grid(language, f"{row_name}-link", found,
                                    images, render))
    return parts


# ---------------------------------------------------------------------------
# Icon rows and the cover, shared by every language
# ---------------------------------------------------------------------------

#: Language-independent images: the icon cells and the deck cover.
SHARED_DIR = ROOT / "docs" / "readme_assets"
#: An icon's width as a share of the column, the same in every row and
#: both layouts: 41 px on a 358 px phone column, 95 px on an 830 px desktop
#: one. Seven of them (the widest module band) fit one phone row.
ICON_SHARE = 0.115
#: The share of the column a row of icons spans; the rest is slack so the
#: last cell never wraps.
ICON_ROW_SHARE = 0.99
#: The resolution the cells are drawn at: a column this many pixels wide.
ICON_REFERENCE = 1600
#: Transparent space above and below an icon, as shares of the column:
#: below it matches the gap under a panel (11 px on a phone).
ICON_MARGIN_TOP = 0.008
ICON_MARGIN_BOTTOM = 0.028


def icon_cell(source: str, count: int) -> tuple[str, str]:
    """The cell of a row of ``count`` equal icons that holds ``source``.

    A cell is a transparent PNG one ``count``-th of the row wide with the
    icon centred in it, so every icon in every row is the same size
    (:data:`ICON_SHARE` of the column), the gaps in a row are equal, and
    the row spans the column at any width. Returns the cell's
    repository-relative path and its ``width`` attribute.
    """
    inside = Path(source).as_posix().split("icons/", 1)[-1]
    path = SHARED_DIR / "cells" / str(count) / Path(inside).with_suffix(".png")
    share = ICON_ROW_SHARE / count
    return path.relative_to(ROOT).as_posix(), f"{100 * share:.2f}%"


def draw_icon_cell(source: str, count: int) -> Path:
    """Write the cell :func:`icon_cell` names, if it is missing."""
    from PIL import Image

    relative, _width = icon_cell(source, count)
    path = ROOT / relative
    icon_px = round(ICON_REFERENCE * ICON_SHARE)
    top = round(ICON_REFERENCE * ICON_MARGIN_TOP)
    bottom = round(ICON_REFERENCE * ICON_MARGIN_BOTTOM)
    cell_w = round(ICON_REFERENCE * ICON_ROW_SHARE / count)
    with Image.open(ROOT / source) as image:
        art = image.convert("RGBA")
    size = (cell_w, round(icon_px * art.height / art.width) + top + bottom)
    if path.is_file():
        with Image.open(path) as existing:
            if existing.size == size:
                return path
    art = art.resize((icon_px, round(icon_px * art.height / art.width)),
                     Image.Resampling.LANCZOS)
    cell = Image.new("RGBA", (cell_w, art.height + top + bottom),
                     (0, 0, 0, 0))
    cell.alpha_composite(art, ((cell_w - art.width) // 2, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    cell.save(path, optimize=True)
    return path


def icon_html(target: str, cell: str, width: str, alt: str) -> str:
    """One linked icon cell as raw HTML; GitHub keeps ``width`` and
    ``align``."""
    return (f'<a href="{_html(target)}"><img src="{_html(cell)}" '
            f'width="{width}" align="middle" alt="{_html(alt)}"></a>')


_ICON_DEFINITION = re.compile(
    r"(?m)^\.\. \|(?P<name>(?:Data|Installer)[A-Za-z]+)\| image:: "
    r"(?P<src>\S+)\n(?P<options>(?:   :[a-z]+:[^\n]*(?:\n|$))*)")
_ICON_ROW = re.compile(
    r"(?m)^(?P<row>\|(?:Data|Installer)[A-Za-z]+\|(?: \|(?:Data|Installer)"
    r"[A-Za-z]+\|)+)[ \t]*$")


def icon_rows(text: str, language: str, render: bool) -> str:
    """The data and installer icon rows as rows of equal cells.

    The source writes them as a paragraph of image substitutions; here
    each becomes a line block of raw-HTML cells joined without spaces (an
    image substitution with a percentage width gains a space inside its
    link on GitHub, which would wrap a full row).
    """
    rows = [match.group("row") for match in _ICON_ROW.finditer(text)]
    counts = {}
    for row in rows:
        names = re.findall(r"\|([^|]+)\|", row)
        for name in names:
            counts[name] = len(names)

    def definition(match) -> str:
        name = match.group("name")
        if name not in counts:
            return match.group(0)
        options = dict(re.findall(r"   :([a-z]+): ?([^\n]*)",
                                  match.group("options")))
        source = _to_repository(re.sub(
            r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/", "",
            match.group("src")), language)
        cell, width = icon_cell(source, counts[name])
        if render:
            draw_icon_cell(source, counts[name])
        html = icon_html(options.get("target", ""), _relative(cell, language),
                         width, options.get("alt", ""))
        return f".. |{name}| raw:: html\n\n   {html}\n"

    text = _ICON_DEFINITION.sub(definition, text)
    return _ICON_ROW.sub(
        lambda match: "| " + "\\ ".join(
            f"|{name}|" for name in re.findall(r"\|([^|]+)\|",
                                               match.group("row"))), text)


COVER_SOURCE = "docs/source/_static/deck/slides/slide_01.jpg"
#: The cover's width as a share of each layout's column, centred: a little
#: under the full phone column, and a moderate block on a desktop.
COVER_SHARE = {"phone": 0.85, "desktop": 0.6}
_COVER = re.compile(
    r"(?m)^\.\. image:: (?P<src>\S*slide_01\.jpg)\n"
    r"(?P<options>(?:   :[a-z]+:[^\n]*(?:\n|$))*)")


def cover_name(layout: str) -> str:
    return "cover" if layout == "phone" else "cover.desktop"


def render_cover(layout: str) -> str:
    """The deck cover centred in a transparent strip of one layout's width,
    as an SVG holding the slide as JPEG at twice its display size."""
    from PIL import Image

    width = LAYOUTS[layout]
    shown = round(width * COVER_SHARE[layout])
    with Image.open(ROOT / COVER_SOURCE) as image:
        slide = image.convert("RGB")
        height = round(shown * slide.height / slide.width)
        slide = slide.resize((2 * shown, 2 * height),
                             Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    slide.save(buffer, "JPEG", quality=86, optimize=True, progressive=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    left = (width - shown) / 2
    digest = hashlib.sha256(json.dumps([RENDERER_VERSION, layout, width, shown])
                            .encode() + (ROOT / COVER_SOURCE).read_bytes()
                            ).hexdigest()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" '
        f'height="{height + GAP_BELOW}" viewBox="0 0 {width} '
        f'{height + GAP_BELOW}" role="img" aria-label="spaCR" '
        f'data-source-sha256="{digest}">\n'
        f'<image x="{left:g}" y="0" width="{shown}" height="{height}" '
        f'href="data:image/jpeg;base64,{data}"/>\n</svg>\n')


def cover_digest(layout: str) -> str:
    width = LAYOUTS[layout]
    shown = round(width * COVER_SHARE[layout])
    return hashlib.sha256(json.dumps([RENDERER_VERSION, layout, width, shown])
                          .encode() + (ROOT / COVER_SOURCE).read_bytes()
                          ).hexdigest()


def cover_markup(text: str, language: str, render: bool) -> str:
    """The deck cover directive as a centred, moderately sized picture."""
    def replace(match) -> str:
        options = dict(re.findall(r"   :([a-z]+): ?([^\n]*)",
                                  match.group("options")))
        if render:
            SHARED_DIR.mkdir(parents=True, exist_ok=True)
            for layout in LAYOUTS:
                path = SHARED_DIR / f"{cover_name(layout)}.svg"
                current = svg_digest(path) if path.is_file() else None
                if current != cover_digest(layout):
                    path.write_text(render_cover(layout), encoding="utf-8")
        shared = SHARED_DIR.relative_to(ROOT).as_posix()
        return raw_html(picture_html(
            _relative(f"{shared}/{cover_name('phone')}.svg", language),
            _relative(f"{shared}/{cover_name('desktop')}.svg", language),
            options.get("alt", "spaCR"), options.get("target"))) + "\n"

    return _COVER.sub(replace, text)


def responsive_raw(text: str, language: str, render: bool) -> str:
    """A raw block with its cover and icon rows in the two-layout form."""
    return icon_rows(cover_markup(text, language, render), language, render)


def _current_markers(language: str) -> dict[str, str]:
    path = readme_path(language)
    if not path.is_file():
        return {}
    return {block["name"]: block["content"]
            for block in parse_rst(path.read_text(encoding="utf-8"))
            if block["kind"] == "marker"}


def compose(language: str, *, render: bool = True,
            markers: Optional[dict] = None) -> Output:
    """The README, images and readable panel pages for one language.

    ``render=False`` skips drawing (and the fonts): each image maps to the
    digest it would carry, which is what the currency check compares.
    """
    english = load_source("en")
    blocks = english if language == "en" else load_source(language)
    problems = [] if language == "en" else check_structure(language)
    if problems:
        raise ValueError("; ".join(problems))
    units = plan(english)
    ids = image_ids(units, english)
    markers = _current_markers(language) if markers is None else markers
    output = Output(language)
    parts: list[str] = []
    for unit, name in zip(units, ids):
        local = [blocks[index] for index in unit["span"]]
        kind = unit["kind"]
        if kind == "raw":
            parts.append(responsive_raw(local[0]["text"], language, render))
        elif kind == "code":
            code = "\n".join(("   " + line) if line else ""
                             for line in local[0]["text"].splitlines())
            parts.append(f".. code-block:: {local[0]['lang']}\n\n{code}")
        elif kind == "generated":
            marker = local[0]["name"]
            content = markers.get(marker)
            if content is None:
                raise ValueError(
                    f"{readme_path(language).name} has no {marker} block; "
                    "run packaging/generate_readme_visuals.py")
            if marker == "installer-links":
                content = responsive_raw(content, language, render)
            else:
                content = convert_generated(
                    _generated_words(language, marker, content), language,
                    marker, render=render, pages=output.pages,
                    image_output=output.images)
            parts.append(f".. spacr-{marker}-begin\n\n{content.strip()}\n\n"
                         f".. spacr-{marker}-end")
        elif kind == "chips":
            found = links_of(local[0]["text"])
            pager = any(arrow in label for label, _target in found
                        for arrow in "\u2190\u2192")
            parts.append(_chip_grid(language, name, found, output.images,
                                    render, columns=2, nav=pager))
        elif kind == "rows":
            block = local[0]
            rows = _rows_markup(language, name, block, output.images, render,
                                pages=output.pages)
            if block.get("marker"):
                marker = block["marker"]
                parts.append(f".. spacr-{marker}-begin\n\n"
                             + "\n\n".join(rows) + f"\n\n.. spacr-{marker}-end")
            else:
                parts.extend(rows)
        else:
            parts.append(_panel_markup(language, name, local, output.images,
                                       render, pages=output.pages))
    output.text = "\n\n".join(part.rstrip() for part in parts) + "\n"
    return output


# ---------------------------------------------------------------------------
# Converting generated markup (used by packaging/generate_readme_visuals.py)
# ---------------------------------------------------------------------------

def convert_generated(markup: str, language: str, block_name: str, *,
                      render: bool = True, pages: Optional[dict] = None,
                      image_output: Optional[dict] = None) -> str:
    """Generated RST with every text block turned into images.

    ``packaging/generate_readme_visuals.py`` still produces its blocks as RST
    (headings, notes, the hardware list-table, the picker link, the model
    zoo's "Older models" line); this turns their text into images in the
    README's one look and leaves image substitutions and line blocks alone.
    The images are written to ``docs/readme_images/<lang>/`` as a side
    effect when ``render`` is true, with the markup they were drawn from
    beside them as ``generated-<block>.rst``: the block's words, which
    :func:`source_rst` reads back for tools and tests about the wording.
    Readable Markdown destinations are generated from the same blocks, with
    any source-owned ``details`` appended to the sole destination.
    """
    blocks = parse_rst(markup)
    images: dict = {}
    readable: dict = {}
    parts: list[str] = []
    counter = 0
    pending: list[dict] = []
    base = f"generated-{block_name}"

    def flush_panel():
        nonlocal counter, pending
        if pending:
            counter += 1
            parts.append(_panel_markup(language, f"{base}-{counter}",
                                       pending, images, render, pages=readable,
                                       generated=True))
            pending = []

    for block in blocks:
        kind = block["kind"]
        if kind == "raw":
            flush_panel()
            parts.append(block["text"])
        elif kind == "table":
            flush_panel()
            pending = [block]
        elif kind == "paragraph" and pending and pending[-1]["kind"] == \
                "table" and not links_of(block["text"]):
            pending[-1]["legend"] = block["text"]
            flush_panel()
        elif kind == "paragraph" and block_name == "language-picker":
            flush_panel()
            label, target = links_of(block["text"])[0]
            prefix = plain(block["text"]).split(":", 1)[0]
            native = re.sub(r"[\U0001F310▾]", "", label).strip()
            name = base
            for layout, layout_width in LAYOUTS.items():
                cell = row_cell(layout_width, 2 if layout == "phone" else 3)
                key = name if layout == "phone" else f"{name}.desktop"
                digest = digest_of(["picker", language, native, target, cell])
                if render:
                    images[key] = render_chip(
                        native, language, cell, digest,
                        f"{prefix}: {native}", icon="globe")
                else:
                    images[key] = digest
            parts.append(raw_html(picture_html(
                _image_reference(language, name),
                _image_reference(language, f"{name}.desktop"),
                f"{prefix}: {native}", target)))
        elif kind == "paragraph" and links_of(block["text"]) and \
                plain(block["text"]).find(":") > 0 and len(
                    links_of(block["text"])) > 1:
            flush_panel()
            label = plain(block["text"]).split(":", 1)[0]
            counter += 1
            name = f"{base}-{counter}"
            parts.append(_panel_markup(
                language, name, [{"kind": "heading", "level": 4,
                                  "text": label}], images, render,
                pages=readable, generated=True))
            found = [(text, _to_repository(link, language))
                     for text, link in links_of(block["text"])]
            tail = re.findall(r">`__?\s*\(([^)]+)\)", block["text"])
            if tail:
                found = [(text + (f" ({tail[0]})" if i == len(found) - 1
                                  else ""), link)
                         for i, (text, link) in enumerate(found)]
            parts.append(_chip_grid(language, f"{name}-link", found, images,
                                    render, columns=3))
        elif kind == "heading" and block["level"] <= 3:
            flush_panel()
            pending = [block]
        else:
            if kind == "heading":
                flush_panel()
            pending.append(block)
    flush_panel()
    details = next((block.get("details", []) for block in load_source(language)
                    if block["kind"] == "generated" and
                    block["name"] == block_name), [])
    if details:
        if len(readable) != 1:
            raise ValueError(f"{block_name} details need one readable destination")
        destination = next(iter(readable))
        readable[destination] = readable[destination].rstrip() + "\n\n" + readable_page(
            [{"kind": "paragraph", "text": text} for text in details], language)
    if pages is not None:
        overlap = pages.keys() & readable.keys()
        if overlap:
            raise ValueError(f"Duplicate readable panel names: {sorted(overlap)}")
        pages.update(readable)
    if image_output is not None:
        image_output.update(images)
    if render:
        _write_images(language, images, prefix=base, keep=False)
        _write_pages(language, readable, prefix=base)
        sidecar = IMAGE_DIR / language / f"{base}.rst"
        if not sidecar.is_file() or sidecar.read_text(
                encoding="utf-8") != markup.strip() + "\n":
            sidecar.write_text(markup.strip() + "\n", encoding="utf-8")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------

def _write_pages(language: str, pages: dict, *, prefix: str = "") -> None:
    directory = TEXT_DIR / language
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        path = directory / f"{name}.md"
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    for path in directory.glob("*.md"):
        owned = not prefix or path.stem == prefix or path.stem.startswith(prefix + "-")
        if owned and path.stem not in pages:
            path.unlink()

def _write_images(language: str, images: dict, *, prefix: str = "",
                  keep: bool = True) -> None:
    """Write ``images``; drop stale files that share ``prefix`` unless kept."""
    directory = IMAGE_DIR / language
    directory.mkdir(parents=True, exist_ok=True)
    for name, svg in images.items():
        path = directory / f"{name}.svg"
        if not path.is_file() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8")
    if not keep and prefix:
        for path in directory.glob(f"{prefix}*.svg"):
            owned = path.stem == prefix or path.stem.startswith(prefix + "-")
            if owned and path.stem not in images:
                path.unlink()


def _generated_names(language: str) -> set[str]:
    """Image names the generated blocks reference in the current README."""
    names = set()
    for content in _current_markers(language).values():
        for match in re.finditer(r"readme_images/[^/\s]+/([^/\s]+)\.svg",
                                 content):
            names.add(match.group(1))
    return names


def build(languages: Sequence[str] = LANGUAGES) -> list[Path]:
    written = []
    for language in languages:
        output = compose(language)
        _write_images(language, output.images)
        _write_pages(language, output.pages)
        keep = set(output.images) | _generated_names(language)
        for path in (IMAGE_DIR / language).glob("*.svg"):
            if path.stem not in keep:
                path.unlink()
        path = readme_path(language)
        if path.read_text(encoding="utf-8") != output.text:
            path.write_text(output.text, encoding="utf-8")
        written.append(path)
    return written


def svg_digest(path: Path) -> Optional[str]:
    match = re.search(r'data-source-sha256="([0-9a-f]{64})"',
                      path.read_text(encoding="utf-8")[:2000])
    return match.group(1) if match else None


def check(languages: Sequence[str] = LANGUAGES) -> list[str]:
    """Stale README, images and text pages, without drawing or needing fonts."""
    problems = []
    for language in languages:
        if language != "en":
            structure = check_structure(language)
            problems += structure
            if structure:
                continue
        output = compose(language, render=False)
        for name, expected in output.pages.items():
            page = TEXT_DIR / language / f"{name}.md"
            if not page.is_file():
                problems.append(f"{page.relative_to(ROOT)} is missing")
            elif page.read_text(encoding="utf-8") != expected:
                problems.append(f"{page.relative_to(ROOT)} is stale")
        for page in (TEXT_DIR / language).glob("*.md"):
            if page.stem not in output.pages:
                problems.append(f"{page.relative_to(ROOT)} is obsolete")
        path = readme_path(language)
        if path.read_text(encoding="utf-8") != output.text:
            problems.append(f"{path.relative_to(ROOT)} is not what "
                            f"{source_path(language).relative_to(ROOT)} "
                            "builds")
        for shared in sorted(set(re.findall(
                r"readme_assets/[^\s\"]+\.(?:svg|png)", output.text))):
            asset = SHARED_DIR / shared.split("readme_assets/", 1)[1]
            if not asset.is_file():
                problems.append(f"{asset.relative_to(ROOT)} is missing")
            elif asset.stem.startswith("cover") and svg_digest(asset) != \
                    cover_digest("desktop" if "desktop" in asset.stem
                                 else "phone"):
                problems.append(f"{asset.relative_to(ROOT)} is stale")
        for name, digest in output.images.items():
            image = IMAGE_DIR / language / f"{name}.svg"
            if not image.is_file():
                problems.append(f"{image.relative_to(ROOT)} is missing")
            elif svg_digest(image) != digest:
                problems.append(f"{image.relative_to(ROOT)} is stale")
    return problems


def refresh_generated(languages: Sequence[str] = LANGUAGES) -> None:
    """Have ``generate_readme_visuals.py`` rewrite its README blocks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "spacr_generate_readme_visuals",
        ROOT / "packaging" / "generate_readme_visuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HARDWARE_TABLE.write_text(module._hardware_table(), encoding="utf-8")
    for language in languages:
        module.write_readme_blocks(readme_path(language))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="report stale README, images or text pages; write "
                             "nothing")
    parser.add_argument("--refresh-generated", action="store_true",
                        help="first regenerate the blocks owned by "
                             "packaging/generate_readme_visuals.py")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES,
                        default=list(LANGUAGES))
    options = parser.parse_args(argv)
    if options.check:
        problems = check(options.languages)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    if options.refresh_generated:
        refresh_generated(options.languages)
    for path in build(options.languages):
        print(path.relative_to(ROOT))
    total = sum(path.stat().st_size for path in IMAGE_DIR.rglob("*.svg"))
    count = len(list(IMAGE_DIR.rglob("*.svg")))
    print(f"{count} images, {total / 1024 / 1024:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
