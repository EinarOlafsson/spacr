"""Instruction 350: a container capped in device pixels does not scale.

THE SWEEP CANNOT SEE THIS ONE, and that is why it is a separate file. The
text-fits sweep asks "is any painted caption cut off", and a QComboBox
answers no however narrow it is: Qt's style elides the closed text and the
popup opens wider than the box, so the full choice stays reachable. Both
behaviours are decisions the sweep deliberately forgives.

What it forgives with them is a settings column that cannot grow. Measured
on Control Charts, the column's own sizeHint against its cap:

    font scale 1.00    wants  586 px    cap 330
    font scale 1.25    wants  707 px    cap 330
    font scale 2.00    wants 1107 px    cap 330

The glyphs double and the box does not -- the same defect this instruction
already fixed on `UsageBar`, which set `setFixedWidth(48)` on a caption
column that needed 66 px of text at 200 %.

So this asserts the RULE rather than any one screen: a hard pixel cap on a
settings column goes through `scaled_px`, so it tracks the font the user
actually reads at.
"""
from __future__ import annotations

import pathlib
import re

import pytest

pytest.importorskip("PySide6")


#: Every screen that caps its settings column, and the base width it caps at.
#: A new one belongs here in the same change that adds it -- the point of the
#: table is that the list is written down rather than discovered.
CAPPED_COLUMNS = {
    "control_chart.py": 330,
    "pca.py": 320,
    "feature_explorer.py": 340,
    "outliers.py": 360,
    "trellis.py": 360,
    "graph_builder.py": 320,
    "tabulate.py": 320,
}

_SCREENS = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt" / "screens"

#: A cap wide enough to be a COLUMN rather than a field. Below this a number
#: is a unit box or a progress bar -- `dose_response.py`'s 70 px unit field,
#: `data_manager.py`'s 140 px progress -- and those are sized to their own
#: content, not to a body of text that grows with the font.
COLUMN_PX = 300

_BARE_CAP = re.compile(r"setMaximumWidth\(\s*(\d+)\s*\)")


def _bare_caps(path: pathlib.Path):
    """``(line number, px)`` for every unscaled pixel cap in ``path``."""
    found = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _BARE_CAP.search(line)
        if match:
            found.append((number, int(match.group(1))))
    return found


@pytest.mark.parametrize("name", sorted(CAPPED_COLUMNS))
def test_the_column_cap_is_scaled(name):
    """The cap this screen puts on its settings column tracks the font."""
    path = _SCREENS / name
    text = path.read_text()
    base = CAPPED_COLUMNS[name]
    assert f"setMaximumWidth(scaled_px({base}))" in text, (
        f"{name} no longer caps its settings column at scaled_px({base}); "
        "a device-pixel cap does not grow when the glyphs inside it do")


def test_no_screen_caps_a_settings_column_in_device_pixels():
    """The rule, swept over every screen rather than the seven known ones.

    A number this large is a COLUMN OF TEXT, and text is what changes size
    when the user changes the font scale. Smaller caps are left alone: they
    size a field to its own content -- a unit box, a percentage, a progress
    bar -- and those do not carry prose.
    """
    offenders = []
    for path in sorted(_SCREENS.glob("*.py")):
        for number, px in _bare_caps(path):
            if px >= COLUMN_PX:
                offenders.append(f"{path.name}:{number} setMaximumWidth({px})")
    assert not offenders, (
        "a settings column is capped in device pixels, so it cannot grow "
        "with the font:\n  " + "\n  ".join(offenders) +
        "\nWrap the number in scaled_px(), and add the screen to "
        "CAPPED_COLUMNS in this file.")


def test_the_table_matches_what_the_screens_actually_do():
    """A table that drifts from the code is worse than none, because it is
    believed. Every entry must name a screen that really caps a column."""
    missing = [name for name in CAPPED_COLUMNS
               if not (_SCREENS / name).exists()]
    assert not missing, f"CAPPED_COLUMNS names screens that are gone: {missing}"

    unlisted = []
    for path in sorted(_SCREENS.glob("*.py")):
        if path.name in CAPPED_COLUMNS:
            continue
        if re.search(r"setMaximumWidth\(\s*scaled_px\(\s*\d\d\d\s*\)\s*\)",
                     path.read_text()):
            unlisted.append(path.name)
    assert not unlisted, (
        f"these screens cap a settings column and are not in CAPPED_COLUMNS: "
        f"{unlisted}")
