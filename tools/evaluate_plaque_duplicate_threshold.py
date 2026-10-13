"""Measure the plaque reader's duplicate flags on hash-pinned published figures.

The transfer manifest supplies arms/figures with key, pmcid, pixels and sha256.
Images must already exist under --images/PMCID/filename; no download occurs.
Every source is checked before scoring. Known same-source controls are lossless
PNG/BMP copies, JPEG at quality 95/75/50, half/double size with Lanczos, and a
half-size JPEG at quality 75. These controls are not human duplicate labels.

Distinct source figures provide an unreviewed collision audit, NOT a false
positive estimate: two different figure files can contain reused images.
The threshold sweep records the current directional aspect-ratio gate. Actual
production matcher decisions are also recorded at the installed threshold.
No threshold is selected automatically and no source image is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from spacr import plaque_papers as papers


def controls(image):
    """Yield encoded same-source variants, decoding each through Pillow."""
    for name, scale, fmt, options in (
        ("png", 1, "PNG", {}), ("bmp", 1, "BMP", {}),
        ("jpeg95", 1, "JPEG", {"quality": 95}),
        ("jpeg75", 1, "JPEG", {"quality": 75}),
        ("jpeg50", 1, "JPEG", {"quality": 50}),
        ("half", 0.5, "PNG", {}), ("double", 2, "PNG", {}),
        ("half_jpeg75", 0.5, "JPEG", {"quality": 75}),
    ):
        variant = image
        if scale != 1:
            variant = image.resize(tuple(max(1, round(n * scale)) for n in image.size),
                                   Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        variant.save(stream, format=fmt, **options)
        encoded = stream.getvalue()
        with Image.open(io.BytesIO(encoded)) as decoded:
            yield name, hashlib.sha256(encoded).hexdigest(), np.asarray(decoded.convert("RGB"))


def source_rows(manifest, images):
    """Resolve unique manifest keys without allowing paths outside the image root."""
    rows = {}
    root = images.resolve()
    for arm in manifest["arms"].values():
        for row in arm["figures"]:
            key = row["key"]
            pmcid, name = key.split("__", 1)
            if pmcid != row["pmcid"] or Path(name).name != name:
                raise ValueError(f"invalid figure key: {key}")
            path = (root / pmcid / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"image lies outside source root: {key}")
            if key in rows and rows[key]["sha256"] != row["sha256"]:
                raise ValueError(f"conflicting hashes for {key}")
            rows[key] = dict(row, path=path)
    if not rows:
        raise ValueError("manifest has no figures")
    return list(rows.values())


def match(connection, row, pixel_hash, dhash, width, height, file_hash, name):
    """Call the production matcher with the reference's real paper identity."""
    paper = papers.Paper(row.get("doi") or row["pmcid"], "europepmc")
    figure = papers.Figure(Path(name), sha256=file_hash)
    return papers._find_duplicate(connection, paper, figure, pixel_hash, dhash,
                                  width=width, height=height)


def evaluate(rows):
    """Verify every input, then measure controls and all ordered source pairs."""
    for row in rows:
        if hashlib.sha256(row["path"].read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"image hash mismatch: {row['key']}")
        with Image.open(row["path"]) as image:
            if list(image.size) != row["pixels"]:
                raise ValueError(f"image dimensions mismatch: {row['key']}")
    references, variants = [], []
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE figures (figure_sha256 TEXT, paper_key TEXT, "
                           "path TEXT, pixel_sha256 TEXT, dhash TEXT, width INT, height INT)")
        for row in rows:
            with Image.open(row["path"]) as source:
                image = source.convert("RGB")
            pixel_hash, dhash = papers._image_fingerprint(np.asarray(image))
            reference = {key: row.get(key) for key in
                         ("key", "pmcid", "doi", "licence", "sha256", "pixels")}
            reference.update(pixel_sha256=pixel_hash, dhash=dhash)
            references.append(reference)
            connection.execute("DELETE FROM figures")
            connection.execute("INSERT INTO figures VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (row["sha256"], row.get("doi") or row["pmcid"],
                                row["path"].name, pixel_hash, dhash, *image.size))
            for name, file_hash, array in controls(image):
                transformed_pixel, transformed_dhash = papers._image_fingerprint(array)
                height, width = array.shape[:2]
                found = match(connection, row, transformed_pixel, transformed_dhash,
                              width, height, file_hash, f"control-{name}-{row['path'].name}")
                variants.append({"key": row["key"], "control": name,
                                 "sha256": file_hash, "pixels": [width, height],
                                 "distance": papers._hamming(dhash, transformed_dhash),
                                 "pixel_identical": transformed_pixel == pixel_hash,
                                 "match": found})
        connection.execute("DELETE FROM figures")
        for row in references:
            connection.execute("INSERT INTO figures VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (row["sha256"], row.get("doi") or row["pmcid"],
                                row["key"].split("__", 1)[1], row["pixel_sha256"],
                                row["dhash"], *row["pixels"]))
        nearest = []
        for row in references:
            found = match(connection, row, row["pixel_sha256"], row["dhash"],
                          *row["pixels"], row["sha256"], row["key"].split("__", 1)[1])
            if found is not None:
                nearest.append({"key": row["key"], "match": found})
    histogram = Counter()
    close_pairs = []
    eligible = 0
    for query in references:
        aspect = query["pixels"][0] / query["pixels"][1]
        for candidate in references:
            if query["key"] == candidate["key"]:
                continue
            candidate_aspect = candidate["pixels"][0] / candidate["pixels"][1]
            if abs(candidate_aspect - aspect) > 0.03 * aspect:
                continue
            eligible += 1
            distance = papers._hamming(query["dhash"], candidate["dhash"])
            histogram[distance] += 1
            if distance <= 8:
                close_pairs.append({"query": query["key"], "candidate": candidate["key"],
                                    "distance": distance})
    summaries = {}
    for name in sorted({row["control"] for row in variants}):
        cohort = [row for row in variants if row["control"] == name]
        summaries[name] = {"total": len(cohort),
                           "matched": sum(row["match"] is not None for row in cohort),
                           "skipped_exact": sum(bool(row["match"] and row["match"]["skip"])
                                                for row in cohort),
                           "distance_histogram": dict(sorted(Counter(
                               row["distance"] for row in cohort).items()))}
    sweep = [{"bits": bits, "same_source_within_distance": sum(
        row["distance"] <= bits for row in variants),
        "distinct_source_ordered_pairs_within_distance_and_aspect": sum(
            count for distance, count in histogram.items() if distance <= bits)}
        for bits in range(17)]
    return {"references": references, "controls": variants, "by_control": summaries,
            "current_threshold": papers.SIMILAR_BITS, "sweep": sweep,
            "distinct_source_ordered_pairs": len(references) * (len(references) - 1),
            "aspect_eligible_ordered_pairs": eligible,
            "aspect_eligible_distance_histogram": dict(sorted(histogram.items())),
            "close_source_pairs_at_most_8_bits": close_pairs,
            "current_nearest_source_matches": nearest}


def main():
    """Write the measured result, refusing source overwrite or unverified inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = source_rows(json.loads(args.manifest.read_text()), args.images)
        protected = {args.manifest.resolve(), *(row["path"] for row in rows)}
        if args.output.resolve() in protected or (args.output.exists() and any(
                path.exists() and args.output.samefile(path) for path in protected)):
            raise ValueError("output must differ from manifest and every source image")
        started = time.monotonic()
        result = evaluate(rows)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    result.update(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  runtime_sha256=hashlib.sha256(Path(papers.__file__).read_bytes()).hexdigest(),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pillow_version=pillow_version, seconds=time.monotonic() - started,
                  limitations=["Same-source transformed controls, not human duplicate truth",
                               "Distinct source pairs are unreviewed, not confirmed negatives",
                               "No cropping, panel reuse, rotation or text-overlay controls",
                               "Similar hashes only flag; exact bytes/pixels alone skip",
                               "Threshold sweep does not automatically select a threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"figures": len(rows), "controls": len(result["controls"]),
                      "by_control": result["by_control"],
                      "nearest_source_matches": len(result["current_nearest_source_matches"]),
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
