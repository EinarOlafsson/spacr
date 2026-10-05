#!/usr/bin/env python
"""Publish a slide deck as the flip-through spaCR info deck.

The maintainer, 2026-09-21: "it would be cool if we could add it to the spacr
readme and let people flip through the slides as a kind of info deck."

A GitHub README cannot run a script, so it cannot hold a slide viewer. What
it can do is show a picture and link, and GitHub itself shows a PDF one page
at a time with next/previous. So the deck is published three ways from one
.pptx, all under ``docs/source/_static/deck/``:

    slides/slide_NN.jpg     every slide, 3200 px wide    (the README's cover,
                                                           and the viewer)
    thumbs/slide_NN.jpg     every slide, 320 px wide     (the viewer's strip)
    anim/slide_NN_*.gif     the deck's animated GIFs     (played by the viewer)
    spacr_deck.pdf          the deck as one PDF, text    (flip through on GitHub)
                            and shapes as vectors
    slides.json             count and a title per slide  (the viewer)
    pages/NN.md             one GitHub page per slide: the slide, then
                            "← Back" and "Next →" to the pages either side
                            (the last wraps to the first), so the deck
                            pages through on GitHub itself; the slide
                            opens the viewer at that slide
    index.html              the viewer: arrow keys, swipe, thumbnails,
                            a link per slide (#12), full screen; the slide
                            list is written into the page, so it opens from
                            disk as well as from the site

The docs site copies ``_static`` as it is, so the viewer is served at
``https://einarolafsson.github.io/spacr/_static/deck/``.

AN ANIMATED GIF ON A SLIDE (the Make Masks magnifier) exports as its first
frame, in the PDF and in the pictures alike. So the GIFs are read out of the
.pptx itself with where each sits on its slide, copied to ``anim/``, and the
viewer plays each one over its slide at that place.

THE TITLE SLIDE'S VERSION FOLLOWS THE RELEASE (the maintainer, 2026-09-21:
"the title page should autoupdate with version bumps"). The deck lives
outside the repository, so a version bump cannot re-render it. Instead the
title slide is rendered WITHOUT its ``spaCR <version> · ...`` line, that
picture is kept as ``title_base.jpg``, and the line is drawn onto it here --
in spaCR's own bundled Open Sans, at the place, size and colour the deck gave
it -- from ``VERSION`` in setup.py. ``--stamp`` redraws it without the deck;
``packaging/release.py`` calls that on every version bump.

HIGH RESOLUTION (item 589, 2026-09-26: "all slides need to be higher
resolution"). The pictures are 3200 px wide (they were 1600), rendered from
a lossless PDF so no image is compressed twice, and saved without chroma
subsampling. The published PDF is LibreOffice's own, with vector text; its
title page is the stamped picture (``replace_first_page``).

Run it again with a new deck to replace the old one; slides the new deck no
longer has are removed::

    python tools/build_readme_deck.py path/to/deck.pptx

For a bounded UI refresh, render the complete updated source privately, then
retain the accepted deck's unrelated pages and animations::

    python tools/build_readme_deck.py updated.pptx --out /private/rendered
    python tools/build_readme_deck.py --refresh-pages 6,12,32,34 \
        --rendered /private/rendered --out /private/checked-deck

The second command also regenerates the viewer and all GitHub navigation
pages. Its output folder must be fresh. Source-native captures can be inserted
with refresh_deck_captures.py before rendering; that tool verifies capture
hashes and changes a separate presentation without editing captured pixels.

Needs LibreOffice (``soffice``) and poppler (``pdftoppm``, ``pdftotext``).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "source" / "_static" / "deck"
VIEWER = Path(__file__).resolve().parent / "readme_deck_viewer.html"


#: LibreOffice's PDF export options for the two PDFs a build makes. The one
#: the pictures are rendered from keeps every image as the deck holds it
#: (no downsampling, lossless); the published one keeps text and shapes as
#: vectors and stores images as JPEG at up to 300 dpi, a fraction of the size.
PDF_LOSSLESS = {"ReduceImageResolution": {"type": "boolean", "value": "false"},
                "UseLosslessCompression": {"type": "boolean", "value": "true"}}
PDF_PUBLISHED = {"ReduceImageResolution": {"type": "boolean", "value": "true"},
                 "MaxImageResolution": {"type": "long", "value": "300"},
                 "Quality": {"type": "long", "value": "90"}}


def to_pdf(pptx: Path, work: Path, options: Optional[dict] = None) -> Path:
    """The deck as LibreOffice renders it to PDF.

    :param pptx: the deck.
    :param work: a scratch folder; the PDF is written into it.
    :param options: LibreOffice PDF export options (``PDF_LOSSLESS``,
        ``PDF_PUBLISHED``); LibreOffice's defaults when None.
    :returns: the PDF.
    """
    profile = work.parent / "lo_profile"
    target = "pdf"
    if options:
        target = "pdf:impress_pdf_Export:" + json.dumps(options)
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["soffice", f"-env:UserInstallation=file://{profile}", "--headless",
         "--convert-to", target, "--outdir", str(work), str(pptx)],
        check=True, capture_output=True, timeout=1800)
    pdf = work / f"{pptx.stem}.pdf"
    if not pdf.is_file():
        raise SystemExit(f"LibreOffice wrote no PDF for {pptx}")
    return pdf


def render(pdf: Path, folder: Path, width: int, quality: int) -> List[Path]:
    """Every page of ``pdf`` as a JPEG ``width`` pixels wide.

    poppler draws each page losslessly and Pillow encodes it with full-
    resolution colour (4:4:4): the usual 4:2:0 halves the colour resolution,
    which fringes the deck's coloured text on its dark background.

    :param pdf: the deck.
    :param folder: where the pictures go; emptied first.
    :param width: the width in pixels.
    :param quality: JPEG quality.
    :returns: the pictures, in slide order.
    """
    from PIL import Image

    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-png", "-scale-to-x", str(width), "-scale-to-y", "-1",
         str(pdf), str(folder / "slide")],
        check=True, capture_output=True, timeout=3600)
    pages = sorted(folder.glob("slide-*.png"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    named = []
    for number, page in enumerate(pages, 1):
        target = folder / f"slide_{number:02d}.jpg"
        with Image.open(page) as drawn:
            drawn.convert("RGB").save(target, quality=quality, optimize=True,
                                      subsampling=0, progressive=True)
        page.unlink()
        named.append(target)
    return named


def titles(pdf: Path, count: int) -> List[str]:
    """A title per slide: the first line of its text that reads like one.

    A slide's small capitals kicker ("INTRODUCTION") is passed over for the
    heading under it.

    :param pdf: the deck.
    :param count: how many slides.
    :returns: one title per slide, ``Slide N`` where none is found.
    """
    out = []
    for number in range(1, count + 1):
        text = subprocess.run(
            ["pdftotext", "-f", str(number), "-l", str(number), "-layout",
             str(pdf), "-"], capture_output=True, text=True).stdout
        lines = [" ".join(line.split()) for line in text.splitlines()]
        lines = [line for line in lines if len(line) > 2]
        heading = next((line for line in lines if not line.isupper()),
                       lines[0] if lines else "")
        out.append(heading[:90] or f"Slide {number}")
    return out


SETUP = ROOT / "setup.py"
FONT = (ROOT / "spacr" / "resources" / "font" / "open_sans" / "static"
        / "OpenSans-Regular.ttf")
_VERSIONED = __import__("re").compile(r"spaCR \d+(?:\.\d+)+")

EMU_NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def animations(pptx: Path, folder: Path) -> dict:
    """Every animated GIF on every slide, and where on the slide it sits.

    :param pptx: the deck.
    :param folder: where the GIFs are copied; emptied first.
    :returns: ``slide number -> [{src, x, y, w, h}]``, the box as fractions
        of the slide's width and height.
    """
    import posixpath
    import re
    import zipfile
    from xml.etree import ElementTree as ET

    if folder.exists():
        shutil.rmtree(folder)
    out: dict = {}
    with zipfile.ZipFile(pptx) as deck:
        names = set(deck.namelist())
        size = ET.fromstring(deck.read("ppt/presentation.xml")).find(
            "p:sldSz", EMU_NS)
        width, height = float(size.get("cx")), float(size.get("cy"))
        order = ET.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target")
                   for rel in order.findall("rel:Relationship", EMU_NS)}
        slide_ids = ET.fromstring(deck.read("ppt/presentation.xml")).findall(
            "p:sldIdLst/p:sldId", EMU_NS)
        for number, sid in enumerate(slide_ids, 1):
            slide = posixpath.normpath(posixpath.join(
                "ppt", targets[sid.get(f"{{{EMU_NS['r']}}}id")]))
            rels_name = posixpath.join(posixpath.dirname(slide), "_rels",
                                       posixpath.basename(slide) + ".rels")
            if rels_name not in names:
                continue
            media = {}
            for rel in ET.fromstring(deck.read(rels_name)).findall(
                    "rel:Relationship", EMU_NS):
                target = posixpath.normpath(posixpath.join(
                    posixpath.dirname(slide), rel.get("Target")))
                if target.lower().endswith(".gif") and target in names:
                    media[rel.get("Id")] = target
            if not media:
                continue
            tree = ET.fromstring(deck.read(slide))
            for pic in tree.iter(f"{{{EMU_NS['p']}}}pic"):
                blip = pic.find(".//a:blip", EMU_NS)
                box = pic.find("p:spPr/a:xfrm", EMU_NS)
                if blip is None or box is None:
                    continue
                target = media.get(blip.get(f"{{{EMU_NS['r']}}}embed"))
                if target is None:
                    continue
                offset, extent = box.find("a:off", EMU_NS), box.find("a:ext", EMU_NS)
                folder.mkdir(parents=True, exist_ok=True)
                name = f"slide_{number:02d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', posixpath.basename(target))}"
                (folder / name).write_bytes(deck.read(target))
                out.setdefault(number, []).append({
                    "src": f"{folder.name}/{name}",
                    "x": round(float(offset.get("x")) / width, 5),
                    "y": round(float(offset.get("y")) / height, 5),
                    "w": round(float(extent.get("cx")) / width, 5),
                    "h": round(float(extent.get("cy")) / height, 5)})
    return out


def deck_titles(pptx: Path) -> List[str]:
    """A title per slide from the deck itself: its topmost large text.

    Read from the slide XML rather than from the PDF's text, whose line order
    follows the layout: a slide whose heading sits beside a table read as
    "INTRODUCTION tool lacks it". Of the text at least 60% the size of the
    slide's largest, the topmost is the heading. Text that is only numbers
    (a statistic, a section number) and a capitals kicker are passed over.

    :param pptx: the deck.
    :returns: one title per slide, empty where a slide has no text.
    """
    import posixpath
    import zipfile
    from xml.etree import ElementTree as ET

    out = []
    with zipfile.ZipFile(pptx) as deck:
        root = ET.fromstring(deck.read("ppt/presentation.xml"))
        rels = ET.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target")
                   for rel in rels.findall("rel:Relationship", EMU_NS)}
        for sid in root.findall("p:sldIdLst/p:sldId", EMU_NS):
            slide = posixpath.normpath(posixpath.join(
                "ppt", targets[sid.get(f"{{{EMU_NS['r']}}}id")]))
            tree = ET.fromstring(deck.read(slide))
            shapes = []
            for shape in tree.iter(f"{{{EMU_NS['p']}}}sp"):
                runs = shape.findall(".//a:r", EMU_NS)
                text = " ".join("".join(t.text or "" for t in run.findall("a:t", EMU_NS))
                                for run in runs).strip()
                text = " ".join(text.split())
                sizes = [int(rp.get("sz")) for rp in shape.findall(".//a:rPr", EMU_NS)
                         if rp.get("sz")]
                offset = shape.find("p:spPr/a:xfrm/a:off", EMU_NS)
                top = int(offset.get("y")) if offset is not None else 0
                if text and sizes:
                    shapes.append((max(sizes), top, text))
            shapes = [(size, top, text) for size, top, text in shapes
                      if not (text.isupper() and len(text) > 3)
                      and any(c.isalpha() for c in text)]
            if not shapes:
                out.append("")
                continue
            biggest = max(size for size, _t, _x in shapes)
            large = [(top, text) for size, top, text in shapes
                     if size >= 0.6 * biggest]
            out.append(min(large)[1][:90])
    return out


def current_version(setup: Path = SETUP) -> str:
    """``VERSION`` from setup.py."""
    import re

    found = re.search(r'^VERSION\s*=\s*["\']([^"\']+)', setup.read_text(
        encoding="utf-8"), re.M)
    if not found:
        raise SystemExit(f"no VERSION in {setup}")
    return found.group(1)


def blank_version_line(pptx: Path) -> Optional[dict]:
    """Empty the title slide's ``spaCR <version>`` line, and say where it was.

    :param pptx: a COPY of the deck; rewritten in place.
    :returns: the line's box (fractions of the slide), size (fraction of the
        slide's height), colour and text with ``{version}`` in place of the
        version; None when the title slide has no such line.
    """
    import os
    import posixpath
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(pptx) as deck:
        entries = {name: deck.read(name) for name in deck.namelist()}
    root = ET.fromstring(entries["ppt/presentation.xml"])
    size = root.find("p:sldSz", EMU_NS)
    width, height = float(size.get("cx")), float(size.get("cy"))
    rels = ET.fromstring(entries["ppt/_rels/presentation.xml.rels"])
    targets = {rel.get("Id"): rel.get("Target")
               for rel in rels.findall("rel:Relationship", EMU_NS)}
    first = root.find("p:sldIdLst/p:sldId", EMU_NS)
    slide = posixpath.normpath(posixpath.join(
        "ppt", targets[first.get(f"{{{EMU_NS['r']}}}id")]))
    xml = entries[slide].decode("utf-8")
    tree = ET.fromstring(entries[slide])
    spec = None
    for shape in tree.iter(f"{{{EMU_NS['p']}}}sp"):
        for run in shape.findall(".//a:r", EMU_NS):
            text_node = run.find("a:t", EMU_NS)
            if text_node is None or not _VERSIONED.search(text_node.text or ""):
                continue
            box = shape.find("p:spPr/a:xfrm", EMU_NS)
            offset, extent = box.find("a:off", EMU_NS), box.find("a:ext", EMU_NS)
            props = run.find("a:rPr", EMU_NS)
            colour = props.find(".//a:srgbClr", EMU_NS) if props is not None else None
            spec = {
                "x": float(offset.get("x")) / width,
                "y": float(offset.get("y")) / height,
                "w": float(extent.get("cx")) / width,
                "h": float(extent.get("cy")) / height,
                "size": (int(props.get("sz", "1400")) / 100 * 12700) / height,
                "colour": "#" + (colour.get("val") if colour is not None else "6B7785"),
                "text": _VERSIONED.sub("spaCR {version}", text_node.text),
            }
            xml = xml.replace(f"<a:t>{text_node.text}</a:t>", "<a:t></a:t>", 1)
            break
        if spec:
            break
    if spec is None:
        return None
    entries[slide] = xml.encode("utf-8")
    temporary = pptx.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as deck:
        for name, data in entries.items():
            deck.writestr(name, data)
    os.replace(temporary, pptx)
    return spec


def stamp_title(folder: Path, version: Optional[str] = None) -> Optional[str]:
    """Draw the current version onto the title slide, and rebuild the PDF.

    :param folder: the published deck folder.
    :param version: the version to draw; setup.py's when None.
    :returns: the text drawn, or None when the deck carries no title line.
    """
    import json

    from PIL import Image, ImageDraw, ImageFont

    manifest_path = folder / "slides.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("title_line")
    base = folder / "title_base.jpg"
    if not spec or not base.is_file():
        return None
    version = version or current_version()
    text = spec["text"].format(version=version)
    picture = Image.open(base).convert("RGB")
    width, height = picture.size
    font = ImageFont.truetype(str(FONT), max(8, round(spec["size"] * height)))
    ImageDraw.Draw(picture).text(
        (spec["x"] * width, (spec["y"] + spec["h"] / 2) * height), text,
        fill=spec["colour"], font=font, anchor="lm")
    first = manifest["slides"][0]
    picture.save(folder / first["image"], quality=86, optimize=True)
    thumb = picture.resize((320, round(320 * height / width)), Image.LANCZOS)
    thumb.save(folder / first["thumb"], quality=75, optimize=True)
    pdf = folder / "spacr_deck.pdf"
    if not replace_first_page(pdf, picture):
        if pdf.is_file():
            print("info deck: pypdf is not installed here, so the PDF's title "
                  "page keeps its old version line; the pictures are stamped.")
        else:
            slides = [folder / s["image"] for s in manifest["slides"]]
            pdf_from(slides, pdf)
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False)
                             + "\n", encoding="utf-8")
    return text


def replace_first_page(pdf: Path, picture) -> bool:
    """Put ``picture`` in place of the first page of ``pdf``, at its size.

    The published PDF is LibreOffice's, with text and shapes as vectors; only
    the title page, whose version line is drawn by this tool, is a picture.

    :param pdf: the published deck; rewritten in place.
    :param picture: the stamped title slide (a PIL image).
    :returns: False when there is no PDF or pypdf is not installed.
    """
    import io
    import os

    if not pdf.is_file():
        return False
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return False
    reader = PdfReader(str(pdf))
    if not reader.pages:
        return False
    width = float(reader.pages[0].mediabox.width)
    page_pdf = io.BytesIO()
    picture.convert("RGB").save(page_pdf, "PDF", quality=90,
                                resolution=picture.width * 72.0 / width)
    page_pdf.seek(0)
    writer = PdfWriter()
    writer.add_page(PdfReader(page_pdf).pages[0])
    for page in reader.pages[1:]:
        writer.add_page(page)
    writer.add_metadata({"/Title": "spaCR"})
    temporary = pdf.with_suffix(".tmp.pdf")
    with open(temporary, "wb") as handle:
        writer.write(handle)
    os.replace(temporary, pdf)
    return True


def pdf_from(slides: Sequence[Path], target: Path) -> None:
    """One PDF of the slide pictures, for GitHub's page-by-page view.

    Used only where no vector PDF exists (a folder stamped before it was
    built); a build publishes LibreOffice's PDF, whose text stays sharp at any
    zoom (``PDF_PUBLISHED``).

    :param slides: the pictures, in order.
    :param target: the PDF to write.
    """
    from PIL import Image

    pages = [Image.open(path).convert("RGB") for path in slides]
    pages[0].save(target, "PDF", resolution=150.0, save_all=True,
                  append_images=pages[1:])
    for page in pages:
        page.close()


VIEWER_URL = "https://einarolafsson.github.io/spacr/_static/deck/"


def github_pages(folder: Path, titles_: Sequence[str]) -> List[Path]:
    """One Markdown page per slide, each linked to the ones either side.

    GitHub runs no script in a README, so the deck cannot be swiped there;
    what it does render is a Markdown file with an image and two links. The
    README shows the first slide with ``← Back`` to the last page and
    ``Next →`` to the second, and each page carries on from there.

    :param folder: the deck folder; pages go in ``pages/``.
    :param titles_: one title per slide, for the alternative text.
    :returns: the pages written.
    """
    pages = folder / "pages"
    if pages.exists():
        shutil.rmtree(pages)
    pages.mkdir(parents=True)
    count = len(titles_)
    written = []
    for number, title in enumerate(titles_, 1):
        back = (number - 2) % count + 1
        ahead = number % count + 1
        page = pages / f"{number:02d}.md"
        page.write_text(
            f'<a href="{VIEWER_URL}#{number}"><img src="../slides/'
            f'slide_{number:02d}.jpg" alt="{title}" width="100%"></a>\n\n'
            f'<p align="center"><a href="{back:02d}.md">← Back</a>'
            f' &nbsp;&nbsp; {number} / {count} &nbsp;&nbsp; '
            f'<a href="{ahead:02d}.md">Next →</a></p>\n',
            encoding="utf-8")
        written.append(page)
    return written


def _aspect(picture: Path) -> float:
    """Width over height of a slide picture."""
    from PIL import Image

    with Image.open(picture) as opened:
        return round(opened.width / opened.height, 5)


def refresh_pages(baseline: Path, rendered: Path, output: Path, numbers: Sequence[int]) -> None:
    """Refresh selected pages from a normal build, preserving unrelated assets."""
    from pypdf import PdfReader, PdfWriter

    if output.exists():
        raise ValueError("A fresh private output folder is required")
    old = json.loads((baseline / "slides.json").read_text(encoding="utf-8"))
    new = json.loads((rendered / "slides.json").read_text(encoding="utf-8"))
    selected = set(numbers)
    if (not selected or len(selected) != len(numbers)
            or not selected <= set(range(1, old["count"] + 1))
            or old["count"] != new["count"]
            or len(old["slides"]) != old["count"]
            or len(new["slides"]) != new["count"]
            or old["aspect"] != new["aspect"]):
        raise ValueError("Selected pages and complete deck geometry must agree")
    for number, (before, after) in enumerate(zip(old["slides"], new["slides"]), 1):
        if (before["animations"] != after["animations"]
                or any(before[key] != after[key] for key in ("image", "thumb"))
                or (number not in selected and before["title"] != after["title"])):
            raise ValueError("Unselected titles, numbering or animation placement changed")
        for key in ("image", "thumb"):
            relative = Path(after[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Slide assets must stay inside the deck")
            if not (rendered / relative).is_file():
                raise ValueError("Complete rendered slide assets are required")
    original = PdfReader(io.BytesIO((baseline / "spacr_deck.pdf").read_bytes()))
    replacement = PdfReader(io.BytesIO((rendered / "spacr_deck.pdf").read_bytes()))
    if len(original.pages) != old["count"] or len(replacement.pages) != old["count"]:
        raise ValueError("Both vector PDFs must contain the complete deck")
    writer = PdfWriter()
    for number, (before, after) in enumerate(zip(original.pages, replacement.pages), 1):
        if tuple(before.mediabox) != tuple(after.mediabox):
            raise ValueError("Rendered PDF page geometry changed")
        writer.add_page(after if number in selected else before)
    if original.metadata:
        writer.add_metadata({str(key): str(value) for key, value in original.metadata.items()})
    shutil.copytree(baseline, output)
    writer.write(output / "spacr_deck.pdf")
    for number in selected:
        item = new["slides"][number - 1]
        for key in ("image", "thumb"):
            shutil.copyfile(rendered / item[key], output / item[key])
        old["slides"][number - 1] = item
    (output / "slides.json").write_text(json.dumps(old, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    github_pages(output, [item["title"] for item in old["slides"]])
    (output / "index.html").write_text(
        VIEWER.read_text(encoding="utf-8").replace("__SLIDES__", json.dumps(old, ensure_ascii=False)),
        encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pptx", type=Path, nargs="?")
    parser.add_argument("--stamp", action="store_true",
                        help="only redraw the title slide's version line")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--quality", type=int, default=86)
    parser.add_argument("--refresh-pages", help="comma-separated pages from an already rendered complete deck")
    parser.add_argument("--rendered", type=Path, help="complete private output of the normal deck builder")
    parser.add_argument("--baseline", type=Path, default=OUT, help="accepted deck preserved by --refresh-pages")
    args = parser.parse_args(argv)
    if args.refresh_pages:
        if args.rendered is None or args.stamp or args.pptx is not None:
            parser.error("--refresh-pages requires --rendered, without a PPTX or --stamp")
        refresh_pages(args.baseline, args.rendered, args.out,
                      [int(number) for number in args.refresh_pages.split(",")])
        print(f"Refreshed pages {args.refresh_pages} -> {args.out}")
        return 0
    if args.stamp:
        drawn = stamp_title(args.out)
        print(f"title slide: {drawn}" if drawn else "no title line to stamp")
        return 0
    if args.pptx is None or not args.pptx.is_file():
        raise SystemExit(f"no deck at {args.pptx}")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacr_deck_") as scratch:
        work = Path(scratch)
        copy = work / "deck.pptx"
        shutil.copyfile(args.pptx, copy)
        try:
            title_line = blank_version_line(copy)
        except Exception:
            title_line = None
        pdf = to_pdf(copy, work / "lossless", PDF_LOSSLESS)
        slides = render(pdf, args.out / "slides", args.width, args.quality)
        render(pdf, args.out / "thumbs", 320, 75)
        names = titles(pdf, len(slides))
        published = to_pdf(copy, work / "published", PDF_PUBLISHED)
        shutil.copyfile(published, args.out / "spacr_deck.pdf")
    try:
        from_deck = deck_titles(args.pptx)
    except Exception:
        from_deck = []
    if len(from_deck) == len(names):
        names = [own or found for own, found in zip(from_deck, names)]
    moving = animations(args.pptx, args.out / "anim")
    if title_line:
        shutil.copyfile(slides[0], args.out / "title_base.jpg")
    elif (args.out / "title_base.jpg").exists():
        (args.out / "title_base.jpg").unlink()
    manifest = {"count": len(slides), "source": args.pptx.name,
                "title_line": title_line,
                "aspect": _aspect(slides[0]),
                "slides": [{"image": f"slides/{p.name}",
                            "thumb": f"thumbs/{p.name}", "title": title,
                            "animations": moving.get(number, [])}
                           for number, (p, title)
                           in enumerate(zip(slides, names), 1)]}
    github_pages(args.out, names)
    (args.out / "slides.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    stamp_title(args.out)
    manifest = json.loads((args.out / "slides.json").read_text(encoding="utf-8"))
    page = VIEWER.read_text(encoding="utf-8").replace(
        "__SLIDES__", json.dumps(manifest, ensure_ascii=False))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    weight = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{len(slides)} slides -> {args.out} ({weight / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
