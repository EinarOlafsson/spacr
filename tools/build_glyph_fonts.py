#!/usr/bin/env python3
"""Build the small fonts that carry the symbols Open Sans does not have.

Open Sans has no arrows, triangles, check marks or ballot boxes, and spaCR's
interface draws all of them: the ``▾`` of a card title, the ``↓`` of the
console's jump button, ``→`` in a hundred captions. For every one of those
glyphs Qt asked fontconfig which installed family could draw it, at every
new pixel size: about 40 ms a glyph a size on a machine with 2,464 families,
and the 50 ms of a column's Ctrl + wheel restyle that made lesson 05's
``14_resize_panels`` clip miss its frame-rate gate.

This tool writes exactly those glyphs into five subset families under
``spacr/resources/font/glyphs``, which :func:`spacr.qt.app._load_bundled_fonts`
registers and names as Open Sans's substitutes, so Qt finds every symbol in a
font it already has open instead of searching the system.

Each glyph comes from the font that drew it before, so nothing changes on
screen: arrows and set symbols from Noto Sans Math, phonetic letters and
sub- and superscripts from Noto Sans (four weights), other symbols from
DejaVu Sans (Book and Bold), full-width signs from Noto Sans CJK JP (Regular
and Bold) and what none of those has from Noto Sans Symbols (Regular,
SemiBold and Bold). Measured on Ubuntu 24.04 with those fonts installed,
every glyph draws pixel for pixel as before except ``⌗`` at light and
regular weight, which fontconfig took from GNU FreeFont's FreeSerif (GPL,
not shipped). The sources are downloaded once from pinned commits and an
Ubuntu package, checked against pinned SHA-256s (:data:`SOURCES`), and
cached under ``~/.cache/spacr/glyph-fonts``; the DejaVu package needs the
``zstandard`` module. The subsets are renamed, as the Bitstream Vera licence
asks of a modified DejaVu and as is customary under the OFL; both licences
are written next to the fonts.

Which glyphs: every character outside Open Sans that a string literal in
``spacr/qt`` draws, docstrings and the translation catalogs excepted
(:func:`ui_glyphs`), plus :data:`EXTRA`. Characters of the scripts the
locales bring (Devanagari, Hangul, Han, kana and their punctuation) are left
to the system's fonts, as they always were.

Run::

    python tools/build_glyph_fonts.py            # write the fonts
    python tools/build_glyph_fonts.py --check    # exit 1 if a glyph is missing

``tests/qt/test_a_splitter_drag_and_the_column_zoom_stay_smooth.py`` fails
when a new string draws a glyph no shipped font has; run this tool then.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import os
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OPEN_SANS = REPO / "spacr" / "resources" / "font" / "open_sans" / "static"
OUT = REPO / "spacr" / "resources" / "font" / "glyphs"
SCAN = REPO / "spacr" / "qt"
CACHE = Path(os.environ.get("SPACR_GLYPH_FONT_CACHE",
                            Path.home() / ".cache" / "spacr" / "glyph-fonts"))

NOTO = ("https://raw.githubusercontent.com/notofonts/noto-fonts/"
        "ffebf8c1ee449e544955a7e813c54f9b73848eac/")
NOTO_2020 = ("https://raw.githubusercontent.com/notofonts/noto-fonts/"
             "c58682b28b18fd5d3755f4371635f862870c1531/hinted/ttf/NotoSans/")
NOTO_CJK = ("https://raw.githubusercontent.com/notofonts/noto-cjk/"
            "f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/")
DEJAVU = ("http://archive.ubuntu.com/ubuntu/pool/main/f/fonts-dejavu/"
          "fonts-dejavu-core_2.37-8_all.deb")

#: ``name: (url, sha256)``. Each is the very file a stock Ubuntu 24.04
#: draws these glyphs from (fonts-dejavu-core 2.37-8, fonts-noto-core
#: 20201225-2, fonts-noto-cjk 20230817), or one whose outlines and metrics
#: for them are the same, so the subsets render pixel for pixel as before.
SOURCES = {
    "fonts-dejavu-core.deb": (
        DEJAVU,
        "40049660c194f3b8a2541fc7369efebb10e9f94bdac836a2f38fafedd10fa73a"),
    "NotoSansMath-Regular.ttf": (
        NOTO + "hinted/ttf/NotoSansMath/NotoSansMath-Regular.ttf",
        "80b61fd613d3519197e64fff6f7e71fdc7f3e6526440ea4115b554ef7fd59af7"),
    "NotoSansSymbols-Regular.ttf": (
        NOTO + "hinted/ttf/NotoSansSymbols/NotoSansSymbols-Regular.ttf",
        "8f02f31959bbdf6061547a188248e13f84dc5fdd940326ec494675f453f072bb"),
    "NotoSansSymbols-SemiBold.ttf": (
        NOTO + "hinted/ttf/NotoSansSymbols/NotoSansSymbols-SemiBold.ttf",
        "de0b525a71555976f50cc0a46418b0f0c450e03e864fef6a2d31f8b101a1c84a"),
    "NotoSansSymbols-Bold.ttf": (
        NOTO + "hinted/ttf/NotoSansSymbols/NotoSansSymbols-Bold.ttf",
        "8a962e49d186810e201ca591f42d5c4a826fcee63bf13db117f5fa6da7404152"),
    "NotoSans-Light.ttf": (
        NOTO_2020 + "NotoSans-Light.ttf",
        "cdba47164863da01cdff6368331cab9186d96678a475542d772666d9593d82ea"),
    "NotoSans-Regular.ttf": (
        NOTO_2020 + "NotoSans-Regular.ttf",
        "89c3c497f618fdaa0b2d1e98fef93582f28c71debd2c4a8cdf41f190ced2909d"),
    "NotoSans-SemiBold.ttf": (
        NOTO_2020 + "NotoSans-SemiBold.ttf",
        "68543590a7c3185d69ad16664c46df28832beeb72621d2470e86b04099eb4efb"),
    "NotoSans-Bold.ttf": (
        NOTO_2020 + "NotoSans-Bold.ttf",
        "e83493c945848ecd4a9ad0f6d19164541a0d3e23a9c952304a00a46e00272ac5"),
    "NotoSansCJK-Regular.ttc": (
        NOTO_CJK + "OTC/NotoSansCJK-Regular.ttc",
        "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a"),
    "NotoSansCJK-Bold.ttc": (
        NOTO_CJK + "OTC/NotoSansCJK-Bold.ttc",
        "faa5f3656a78b2e2d450d27fe8382c778bc2b6bb5ea29c986664a6a435056ceb"),
    "OFL-Noto.txt": (
        NOTO + "LICENSE",
        "0dab92d0544f7b233403f14b84a663bdbfa746982eda629e7f4f9ffe1b036feb"),
    "OFL-Noto-CJK.txt": (
        NOTO_CJK + "LICENSE",
        "6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2"),
}

#: The DejaVu files inside the Ubuntu package.
DEJAVU_MEMBERS = {
    "DejaVuSans.ttf": "./usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf":
        "./usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "LICENSE-DejaVu.txt": "./usr/share/doc/fonts-dejavu-core/copyright",
}

#: ``(family, [(style, source file, output file)], takes)``, asked in this
#: order; ``takes`` is the character ranges the family is preferred for
#: (empty: anything it has that no earlier family took).
FAMILIES = (
    ("spaCR Glyphs Math",
     [("Regular", "NotoSansMath-Regular.ttf", "spacr-glyphs-math.ttf")],
     ((0x2190, 0x21FF), (0x2200, 0x22FF), (0x27F0, 0x27FF),
      (0x2900, 0x297F), (0x2980, 0x29FF), (0x2A00, 0x2AFF))),
    ("spaCR Glyphs Latin",
     [("Regular", "NotoSans-Regular.ttf", "spacr-glyphs-latin.ttf"),
      ("Light", "NotoSans-Light.ttf", "spacr-glyphs-latin-light.ttf"),
      ("SemiBold", "NotoSans-SemiBold.ttf", "spacr-glyphs-latin-semibold.ttf"),
      ("Bold", "NotoSans-Bold.ttf", "spacr-glyphs-latin-bold.ttf")],
     ((0x1D00, 0x1DBF), (0x2070, 0x209F))),
    ("spaCR Glyphs Sans",
     [("Regular", "DejaVuSans.ttf", "spacr-glyphs-sans.ttf"),
      ("Bold", "DejaVuSans-Bold.ttf", "spacr-glyphs-sans-bold.ttf")],
     ()),
    ("spaCR Glyphs Symbols",
     [("Regular", "NotoSansSymbols-Regular.ttf", "spacr-glyphs-symbols.ttf"),
      ("SemiBold", "NotoSansSymbols-SemiBold.ttf",
       "spacr-glyphs-symbols-semibold.ttf"),
      ("Bold", "NotoSansSymbols-Bold.ttf", "spacr-glyphs-symbols-bold.ttf")],
     ()),
    ("spaCR Glyphs Noto Sans CJK JP",
     [("Regular", "NotoSansCJK-Regular.ttc", "spacr-glyphs-cjk.otf"),
      ("Bold", "NotoSansCJK-Bold.ttc", "spacr-glyphs-cjk-bold.otf")],
     ()),
)

#: FreeType's auto-hinter, which draws these fonts at fontconfig's
#: ``hintslight``, sets each script's blue zones from reference characters
#: of that script in the same font (``afblue.dat``). A subset without them
#: snaps its glyphs to other heights than the whole font did, so the family
#: whose glyphs belong to a script carries that script's references. The
#: CJK family needs none: FreeType hints CFF outlines with their own hints.
#: Its name keeps "Noto Sans CJK JP" because fontconfig configurations match
#: rendering rules on it (Ubuntu's Japanese language support turns hinting
#: off at 18 px and below for any family containing it).
HINTING_REFERENCES = {
    "spaCR Glyphs Latin": (
        "THEZOCQSHEZLOCUSfijkdbhxzroescpqgjy"
        "₀₃₅₇₈₀₁₂₃₈ᵢⱼₕₖₗₐₑₒₓₙₛᵥᵤᵣᵦᵧᵨᵩₚ"
        "⁰³⁵⁷ᵀᴴᴱᴼ⁰¹²³ᴱᴸᴼᵁᵇᵈᵏʰʲᶠⁱᵉᵒʳˢˣᶜᵖʸᵍ"),
}

#: Glyphs shipped although no literal draws them today: the other halves of
#: pairs the interface uses one side of.
EXTRA = "▴▾▸◂▲▼◀▶↑↓←→↔✓✗✕●○■□"

#: Scripts the locales bring with their own fonts; their letters and
#: punctuation are not this tool's to ship.
LOCALE_SCRIPTS = ("DEVANAGARI", "HANGUL", "CJK", "HIRAGANA", "KATAKANA",
                  "IDEOGRAPHIC", "FULLWIDTH COMMA", "FULLWIDTH COLON",
                  "FULLWIDTH SEMICOLON", "FULLWIDTH QUESTION",
                  "FULLWIDTH EXCLAMATION", "FULLWIDTH LEFT PARENTHESIS",
                  "FULLWIDTH RIGHT PARENTHESIS")

LICENSES = ("LICENSE-DejaVu.txt", "OFL-Noto.txt", "OFL-Noto-CJK.txt")


def _cmap(path: Path) -> set:
    """The code points ``path`` maps."""
    from fontTools.ttLib import TTFont

    return set(TTFont(str(path), lazy=True, fontNumber=0).getBestCmap())


def open_sans_cmap() -> set:
    """Every code point some shipped Open Sans weight draws."""
    points = set()
    for path in sorted(OPEN_SANS.glob("*.ttf")):
        points |= _cmap(path)
    return points


def _docstrings(tree) -> set:
    """``id`` of every string constant that is a statement of its own."""
    return {id(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)}


def _locale_text(text: str) -> bool:
    """Whether ``text`` is written in a script a locale brings."""
    for char in text:
        name = unicodedata.name(char, "")
        if any(name.startswith(script) for script in LOCALE_SCRIPTS):
            return True
    return False


def _skipped(path: Path) -> bool:
    """Translation catalogs: their text is the locales' to draw."""
    return ("i18n" in path.name or "catalog" in path.name
            or "i18n_catalogs" in path.parts)


def ui_glyphs(root: Path = SCAN, have: set | None = None) -> dict:
    """``{char: [where, ...]}`` for what the interface draws beyond ``have``.

    :param root: the package directory to read.
    :param have: code points already drawn; Open Sans by default.
    """
    have = open_sans_cmap() if have is None else have
    found: dict = {}
    for path in sorted(root.rglob("*.py")):
        if _skipped(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            if (not isinstance(node, ast.Constant)
                    or not isinstance(node.value, str) or id(node) in skip):
                continue
            if _locale_text(node.value):
                continue
            for char in node.value:
                if (ord(char) > 0x7F and ord(char) not in have
                        and unicodedata.category(char)[0] not in "CZ"):
                    found.setdefault(char, []).append(
                        f"{path.relative_to(REPO)}:{node.lineno}")
    return found


def _download(name: str) -> Path:
    """The pinned source ``name``, fetched once and verified."""
    url, digest = SOURCES[name]
    path = CACHE / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            path.write_bytes(response.read())
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{path} has SHA-256 {actual}, expected {digest}")
    return path


def _deb_member(deb: Path, member: str) -> bytes:
    """One file of a ``.deb`` (an ``ar`` archive holding ``data.tar.zst``)."""
    import zstandard

    blob = deb.read_bytes()
    offset = 8
    while offset < len(blob):
        header = blob[offset:offset + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        body = blob[offset + 60:offset + 60 + size]
        offset += 60 + size + (size & 1)
        if name.startswith("data.tar"):
            data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body))
            with tarfile.open(fileobj=data, mode="r|") as tar:
                for entry in tar:
                    if entry.name == member:
                        return tar.extractfile(entry).read()
    raise RuntimeError(f"{member} is not in {deb}")


def source(name: str) -> Path:
    """A source font or licence by file name, unpacked if it is DejaVu's."""
    if name in DEJAVU_MEMBERS:
        path = CACHE / name
        if not path.is_file():
            deb = _download("fonts-dejavu-core.deb")
            path.write_bytes(_deb_member(deb, DEJAVU_MEMBERS[name]))
        return path
    return _download(name)


def assign(glyphs) -> dict:
    """``{family: chars}``: which family carries each glyph.

    :raises SystemExit: when no source has a glyph.
    """
    taken: dict = {family: [] for family, _faces, _takes in FAMILIES}
    regular = {family: _cmap(source(faces[0][1]))
               for family, faces, _takes in FAMILIES}
    missing = []
    for char in sorted(set(glyphs)):
        point = ord(char)
        preferred = [family for family, _faces, takes in FAMILIES
                     if any(low <= point <= high for low, high in takes)]
        order = preferred + [family for family, _faces, takes in FAMILIES
                             if not takes]
        for family in order:
            if point in regular[family]:
                taken[family].append(char)
                break
        else:
            missing.append(char)
    if missing:
        raise SystemExit("no source font draws "
                         + " ".join(f"{c} U+{ord(c):04X}" for c in missing))
    return {family: "".join(chars) for family, chars in taken.items()}


def _rename(font, family: str, style: str) -> None:
    """Give the subset its own names (required of a modified DejaVu)."""
    postscript = (family + "-" + style).replace(" ", "")
    full = f"{family} {style}" if style != "Regular" else family
    table = font["name"]
    keep = {0, 7, 8, 9, 11, 12, 13, 14}
    table.names = [record for record in table.names if record.nameID in keep]
    for name_id, value in ((1, family), (2, style), (3, postscript),
                           (4, full), (6, postscript)):
        table.setName(value, name_id, 3, 1, 0x409)
        table.setName(value, name_id, 1, 0, 0)
    if "CFF " in font:
        cff = font["CFF "].cff
        cff.fontNames = [postscript]
        top = cff.topDictIndex[0]
        for key, value in (("FamilyName", family), ("FullName", full)):
            if hasattr(top, key):
                setattr(top, key, value)


def subset(src: Path, chars: str, out: Path, family: str,
           style: str) -> None:
    """Write ``chars`` of ``src``, renamed, to ``out``."""
    from fontTools import subset as fsubset
    from fontTools.ttLib import TTFont

    options = fsubset.Options()
    options.hinting = True
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.notdef_outline = True
    options.recalc_timestamp = False
    options.layout_features = ["*"]
    font = TTFont(str(src), recalcTimestamp=False, fontNumber=0)
    subsetter = fsubset.Subsetter(options=options)
    subsetter.populate(unicodes=[ord(c) for c in chars])
    subsetter.subset(font)
    _rename(font, family, style)
    font["head"].modified = font["head"].created
    buffer = io.BytesIO()
    font.save(buffer)
    out.write_bytes(buffer.getvalue())


def build(out: Path = OUT) -> dict:
    """Write every font and licence; returns ``{family: chars}``."""
    glyphs = set(ui_glyphs()) | {c for c in EXTRA
                                 if ord(c) not in open_sans_cmap()}
    plan = assign(glyphs)
    out.mkdir(parents=True, exist_ok=True)
    for family, faces, _takes in FAMILIES:
        for style, name, target in faces:
            subset(source(name),
                   plan[family] + HINTING_REFERENCES.get(family, ""),
                   out / target, family, style)
    for name in LICENSES:
        (out / name).write_bytes(source(name).read_bytes())
    return plan


def shipped_cmap(out: Path = OUT) -> set:
    """Every code point Open Sans and the glyph fonts draw together."""
    points = open_sans_cmap()
    for path in sorted(out.glob("*.[ot]tf")):
        points |= _cmap(path)
    return points


def main(argv=None) -> int:
    """Build, or with ``--check`` list what no shipped font draws."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        missing = ui_glyphs(have=shipped_cmap())
        for char, where in sorted(missing.items()):
            print(f"{char} U+{ord(char):04X} {where[0]}")
        return 1 if missing else 0
    for family, chars in build().items():
        print(f"{family}: {chars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
