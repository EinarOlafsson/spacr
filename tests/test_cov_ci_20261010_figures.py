"""Recipe drawing branches of the figure bundle, plates and sheet modules."""
from __future__ import annotations

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures.plates import build_plates  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_plate_recipe_skips_keys_whose_well_cannot_be_read():
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r2_c2", "p1_r3_cX"],
                          "value": [1.0, 2.0, 3.0]})
    figure, panel = build_plates(frame, "value", plates=["p1"], target="print")
    assert panel.drawn
    (recipe,) = figure._spacr_spec["panels"]
    assert recipe["plate"]["coordinates"] == {"p1_r1_c1": [1, 1],
                                              "p1_r2_c2": [2, 2]}


def test_a_sheet_with_no_drawable_panel_registers_a_plain_sheet():
    from spacr.figures import sheet as sheet_module

    sheet = sheet_module.build_sheet(pd.DataFrame({"unrelated": [1, 2]}),
                                     target="print")
    assert sheet.panels == []
    assert sheet.skipped
    assert all(not panel.drawn for panel in sheet.skipped)
    assert sheet.figure._spacr_spec["kind"] == "sheet"
    assert "panels" not in sheet.figure._spacr_spec
    assert not any(axes.axison or axes.has_data() for axes in sheet.figure.axes)
