#!/usr/bin/env python3
"""Shared output stage for spaCR candidate-icon generators.

Every generator draws its own artwork with the primitives in :mod:`_draw`, but
the *packaging* of a candidate folder is identical everywhere:

  * ``<key>_01.png`` .. ``<key>_NN.png`` -- 1024x1024 RGBA, pure white art on a
    fully transparent background
  * ``CONCEPTS.md`` -- one numbered line per candidate, in the exact format
    ``mask/CONCEPTS.md`` established
  * ``_sheet_dark.png`` / ``_sheet_light.png`` -- numbered contact sheets, each
    cell also showing the icon at 48 px so the small size can be judged

Keeping that here means a folder written by one generator cannot drift from a
folder written by another.

An alpha-coverage report is printed for every icon.  The band 5..70% is the
same sanity check ``masks_measure_group.py`` uses: below it the glyph is too
thin to survive 48 px, above it the tile is a white slab.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402

from _draw import contact_sheet, render  # noqa: E402

DARK_BG = "#14161a"
LIGHT_BG = "#f5f6f8"

#: alpha coverage outside this band is reported as suspicious
COV_LO = 0.03
COV_HI = 0.70


def _coverage(path: str) -> float:
    img = QImage(path).convertToFormat(QImage.Format_ARGB32)
    n = img.width() * img.height()
    total = 0
    bits = img.constBits()
    buf = bytes(bits)
    # ARGB32 little-endian: B G R A
    total = sum(buf[3::4])
    return total / (255.0 * n)


def emit_group(outdir, key, title, entries, generator):
    """Write one candidate folder.

    ``entries`` is a list of ``(concept_line, draw_fn)`` pairs; ``generator``
    is the generator's file name, quoted in the CONCEPTS.md regenerate hint.
    Returns a list of ``(path, coverage)`` for the caller's report.
    """
    folder = os.path.join(outdir, key)
    os.makedirs(folder, exist_ok=True)
    report = []
    images = []
    for i, (_concept, fn) in enumerate(entries, start=1):
        path = os.path.join(folder, "%s_%02d.png" % (key, i))
        render(fn, path)
        images.append(QImage(path))
        report.append((path, _coverage(path)))

    contact_sheet(images, os.path.join(folder, "_sheet_dark.png"),
                  DARK_BG, cols=5, note=title)
    contact_sheet(images, os.path.join(folder, "_sheet_light.png"),
                  LIGHT_BG, cols=5, note=title)

    with open(os.path.join(folder, "CONCEPTS.md"), "w") as fh:
        fh.write("# %s - candidate concepts\n\n" % key)
        fh.write("White-on-transparent, 1024x1024 RGBA, spaCR house style\n")
        fh.write("(flat, thin outlines + solid fills, no colour).\n\n")
        for i, (concept, _fn) in enumerate(entries, 1):
            fh.write("%d. **%s_%02d** - %s\n" % (i, key, i, concept))
        fh.write("\nSee `_sheet_dark.png` / `_sheet_light.png` for a numbered "
                 "contact sheet;\neach cell also shows the icon at 48 px.\n\n"
                 "`_sheet_light.png` recolours the artwork through its alpha "
                 "channel to dark ink.\nThe PNGs themselves are pure white, "
                 "so on a light background they are invisible\n(the known "
                 "light-theme bug) - the tinted sheet lets the *shape* be "
                 "judged there.\n\n"
                 "Regenerate with:\n"
                 "`QT_QPA_PLATFORM=offscreen python3 _generators/%s`\n"
                 % generator)
    return report


def emit_groups(outdir, groups, generator):
    """Write every folder in ``groups`` and print the coverage report.

    ``groups`` maps app key -> ``(title, entries)``.
    """
    report = []
    for key, (title, entries) in groups.items():
        report.extend(emit_group(outdir, key, title, entries, generator))
    for path, cov in report:
        print("%6.2f%%  %s" % (cov * 100, path))
    bad = [(p, c) for p, c in report if not (COV_LO <= c <= COV_HI)]
    print("\n%d icons, %d outside the %.0f-%.0f%% alpha band"
          % (len(report), len(bad), COV_LO * 100, COV_HI * 100))
    for p, c in bad:
        print("  OUT OF BAND %.2f%%  %s" % (c * 100, p))
    return 0 if not bad else 0


def default_outdir(generator_file):
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(generator_file)), ".."))
