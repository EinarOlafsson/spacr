#!/usr/bin/env python
"""Publish a slide deck as the flip-through spaCR info deck.

The maintainer, 2026-09-21: "it would be cool if we could add it to the spacr
readme and let people flip through the slides as a kind of info deck."

A GitHub README cannot run a script, so it cannot hold a slide viewer. What
it can do is show a picture and link, and GitHub itself shows a PDF one page
at a time with next/previous. So the deck is published three ways from one
.pptx, all under ``docs/source/_static/deck/``:

    slides/slide_NN.jpg     every slide, 1600 px wide    (the README's cover,
                                                           and the viewer)
    thumbs/slide_NN.jpg     every slide, 320 px wide     (the viewer's strip)
    anim/slide_NN_*.gif     the deck's animated GIFs     (played by the viewer)
    spacr_deck.pdf          the deck as one PDF          (flip through on GitHub)
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

Run it again with a new deck to replace the old one; slides the new deck no
longer has are removed::

    python tools/build_readme_deck.py path/to/deck.pptx

Needs LibreOffice (``soffice``) and poppler (``pdftoppm``, ``pdftotext``).
"""
from __future__ import annotations

import argparse
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


def to_pdf(pptx: Path, work: Path) -> Path:
    """The deck as LibreOffice renders it to PDF.

    :param pptx: the deck.
    :param work: a scratch folder.
    :returns: the PDF.
    """
    profile = work / "lo_profile"
    subprocess.run(
        ["soffice", f"-env:UserInstallation=file://{profile}", "--headless",
         "--convert-to", "pdf", "--outdir", str(work), str(pptx)],
        check=True, capture_output=True, timeout=900)
    pdf = work / f"{pptx.stem}.pdf"
    if not pdf.is_file():
        raise SystemExit(f"LibreOffice wrote no PDF for {pptx}")
    return pdf


def render(pdf: Path, folder: Path, width: int, quality: int) -> List[Path]:
    """Every page of ``pdf`` as a JPEG ``width`` pixels wide.

    :param pdf: the deck.
    :param folder: where the pictures go; emptied first.
    :param width: the width in pixels.
    :param quality: JPEG quality.
    :returns: the pictures, in slide order.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-jpeg", "-jpegopt", f"quality={quality},optimize=y",
         "-scale-to-x", str(width), "-scale-to-y", "-1", str(pdf),
         str(folder / "slide")],
        check=True, capture_output=True, timeout=900)
    pages = sorted(folder.glob("slide-*.jpg"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    named = []
    for number, page in enumerate(pages, 1):
        target = folder / f"slide_{number:02d}.jpg"
        page.rename(target)
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
    slides = [folder / s["image"] for s in manifest["slides"]]
    pdf_from(slides, folder / "spacr_deck.pdf")
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False)
                             + "\n", encoding="utf-8")
    return text


def pdf_from(slides: Sequence[Path], target: Path) -> None:
    """One PDF of the slide pictures, for GitHub's page-by-page view.

    Built from the pictures rather than copied from LibreOffice's PDF, whose
    embedded full-size images make it several times larger for a page GitHub
    shows at the same size either way.

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pptx", type=Path, nargs="?")
    parser.add_argument("--stamp", action="store_true",
                        help="only redraw the title slide's version line")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args(argv)
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
        pdf = to_pdf(copy, work)
        slides = render(pdf, args.out / "slides", args.width, args.quality)
        render(pdf, args.out / "thumbs", 320, 75)
        names = titles(pdf, len(slides))
    try:
        from_deck = deck_titles(args.pptx)
    except Exception:
        from_deck = []
    if len(from_deck) == len(names):
        names = [own or found for own, found in zip(from_deck, names)]
    moving = animations(args.pptx, args.out / "anim")
    pdf_from(slides, args.out / "spacr_deck.pdf")
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
