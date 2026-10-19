"""Every README icon row has the same gutter, read off the generated cells.

Item 366, the rule adopted 2026-10-19: in every icon row the horizontal gap
between neighbouring icons' visible artwork is G = 0.026 of the column width,
within a pixel, and the gap between consecutive rows' artwork is G too. G is
derived here from the cell PNGs the README actually references, not from the
generator's constants.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
CELL = re.compile(r'<img src="(?P<src>[^"]*readme_assets/cells/[^"]+\.png)" '
                  r'width="(?P<width>[0-9.]+)%"')
#: The two columns the rule is measured at: github.com's README column at a
#: 1440 px window and at a 390 px phone.
COLUMNS = (830, 358)


def _cells():
    text = (ROOT / "README.rst").read_text(encoding="utf-8")
    found = [(m.group("src"), float(m.group("width"))) for m in CELL.finditer(text)]
    assert found, "the README references no icon cells"
    return found


def _measure(src):
    with PIL.open(ROOT / src) as image:
        alpha = image.convert("RGBA").getchannel("A")
        box = alpha.getbbox()
        return image.size, box


def test_every_cell_has_one_width_in_the_readme():
    widths = {width for _src, width in _cells()}
    assert len(widths) == 1, widths


def test_the_horizontal_gutter_is_the_same_in_every_row():
    share = _cells()[0][1] / 100
    gutters = set()
    for src, _width in _cells():
        (cell_w, _h), box = _measure(src)
        ink_w = box[2] - box[0]
        left, right = box[0], cell_w - box[2]
        assert abs(left - right) <= 1, (src, left, right)
        for column in COLUMNS:
            scale = column * share / cell_w
            gutters.add((column, round((cell_w - ink_w) * scale, 1)))
    for column in COLUMNS:
        values = [g for c, g in gutters if c == column]
        assert max(values) - min(values) <= 1, (column, sorted(values))
        assert abs(min(values) - 0.026 * column) <= 1, (column, sorted(values))


def test_rows_sit_as_far_apart_as_icons():
    share = _cells()[0][1] / 100
    for src, _width in _cells():
        (cell_w, cell_h), box = _measure(src)
        above_and_below = box[1] + (cell_h - box[3])
        for column in COLUMNS:
            scale = column * share / cell_w
            horizontal = (cell_w - (box[2] - box[0])) * scale
            assert abs(above_and_below * scale - horizontal) <= 1, (
                src, column, above_and_below * scale, horizontal)


def test_no_whitespace_between_neighbouring_cells():
    text = (ROOT / "README.rst").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| |")]
    assert rows
    for row in rows:
        assert " |" not in row[2:].replace("\\ |", ""), row
