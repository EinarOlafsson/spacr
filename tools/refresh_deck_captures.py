#!/usr/bin/env python3
"""Refresh an isolated PPTX source with hash-checked native captures.

The JSON specification pins the original presentation, exact old text and
capture inventories. This changes the source presentation only. Publish it
with build_readme_deck.py, then refresh_alpha_deck.py. Original presentations,
captures, unrelated ZIP members and image pixels remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from PIL import Image

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def refresh(source, specification, output):
    """Write a new source deck, refusing stale text, captures and shared media."""
    source, output = Path(source), Path(output)
    if output.exists() or output.resolve() == source.resolve():
        raise ValueError("A fresh, separate output presentation is required")
    original = source.read_bytes()
    if digest(original) != specification["source_sha256"]:
        raise ValueError("Source presentation differs from its specification")
    with zipfile.ZipFile(source) as deck:
        members = {info.filename: deck.read(info) for info in deck.infolist()}
        presentation = ET.fromstring(members["ppt/presentation.xml"])
        if len(presentation.findall("p:sldIdLst/p:sldId", NS)) != specification["slide_count"]:
            raise ValueError("Presentation slide count changed")
        replacements, captures = {}, []
        for change in specification["slides"]:
            number = int(change["number"])
            name = f"ppt/slides/slide{number}.xml"
            slide = ET.fromstring(members[name])
            for old, new in change.get("text", {}).items():
                matches = [node for node in slide.findall(".//a:t", NS) if node.text == old]
                if len(matches) != 1:
                    raise ValueError(f"Slide {number}: old text must match exactly once: {old}")
                matches[0].text = new
            for picture in change.get("pictures", []):
                capture = Path(picture["capture"])
                provenance = json.loads((capture / "provenance.json").read_text())
                if not provenance.get("completed_capture") or provenance.get("app_source_modified"):
                    raise ValueError("A completed, unmodified native capture is required")
                inventory = json.loads((capture / "frames.json").read_text())
                frame = inventory[picture["frame"]]
                image = (capture / frame["image"]).resolve()
                if not image.is_relative_to(capture.resolve()):
                    raise ValueError("Capture image leaves its recorded directory")
                payload = image.read_bytes()
                if digest(payload) != frame["sha256"]:
                    raise ValueError("Native capture pixels changed")
                with Image.open(image) as pixels:
                    if pixels.format != "PNG" or pixels.size != (3840, 2160):
                        raise ValueError("A native 4K PNG capture is required")
                    width, height = pixels.size
                matches = [node for node in slide.findall(".//p:pic", NS)
                           if node.find("p:nvPicPr/p:cNvPr", NS).get("descr", "").endswith(picture["description_suffix"])]
                if len(matches) != 1:
                    raise ValueError("Source picture description must match exactly once")
                pic = matches[0]
                rid = pic.find("p:blipFill/a:blip", NS).get(f"{{{NS['r']}}}embed")
                rels = ET.fromstring(members[f"ppt/slides/_rels/slide{number}.xml.rels"])
                targets = [rel.get("Target") for rel in rels if rel.get("Id") == rid]
                if len(targets) != 1:
                    raise ValueError("Picture relationship is ambiguous")
                target = posixpath.normpath(posixpath.join("ppt/slides", targets[0]))
                if not target.startswith("ppt/media/") or not target.endswith(".png"):
                    raise ValueError("Expected an embedded PNG")
                references = [key for key, value in members.items()
                              if key.startswith("ppt/slides/_rels/") and key.endswith(".rels")
                              and any(posixpath.normpath(posixpath.join("ppt/slides", rel.get("Target", ""))) == target
                                      for rel in ET.fromstring(value))]
                if references != [f"ppt/slides/_rels/slide{number}.xml.rels"] or target in replacements:
                    raise ValueError("Cannot replace shared or repeated presentation media")
                transform = pic.find("p:spPr/a:xfrm", NS)
                offset, extent = transform.find("a:off", NS), transform.find("a:ext", NS)
                x, y = int(offset.get("x")), int(offset.get("y"))
                box_width, box_height = int(extent.get("cx")), int(extent.get("cy"))
                scale = min(box_width / width, box_height / height)
                fitted_width, fitted_height = round(width * scale), round(height * scale)
                offset.set("x", str(x + (box_width - fitted_width) // 2))
                offset.set("y", str(y + (box_height - fitted_height) // 2))
                extent.set("cx", str(fitted_width))
                extent.set("cy", str(fitted_height))
                pic.find("p:nvPicPr/p:cNvPr", NS).set("descr", picture["label"])
                replacements[target] = payload
                captures.append({"slide": number, "frame": picture["frame"],
                                 "sha256": digest(payload), "member": target,
                                 "fit_preserves_aspect": True})
            if name in replacements:
                raise ValueError("Duplicate slide selection")
            replacements[name] = ET.tostring(slide, encoding="utf-8", xml_declaration=True)
        if source.read_bytes() != original:
            raise ValueError("Original presentation changed during refresh")
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as target:
            for info in deck.infolist():
                target.writestr(info, replacements.get(info.filename, members[info.filename]))
    with zipfile.ZipFile(output) as target:
        assert set(target.namelist()) == set(members)
        assert all(target.read(key) == replacements.get(key, value) for key, value in members.items())
    receipt = {"source_sha256": digest(original), "output_sha256": digest(output.read_bytes()),
               "slide_count": specification["slide_count"], "changed_members": sorted(replacements),
               "untouched_members_verified": len(members) - len(replacements), "captures": captures,
               "original_unchanged": True, "capture_pixels_unchanged": True, "published": False}
    output.with_suffix(".refresh.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = refresh(args.source, json.loads(args.specification.read_text()), args.output)
    print(f"Refreshed {len(result['captures'])} native captures; preserved {result['untouched_members_verified']} unrelated members")


if __name__ == "__main__":
    main()
