import pytest
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
# importorskip, not a bare import. A module-scope `from pypdf import ...`
# here made a missing optional dependency a COLLECTION error, which aborts the
# whole run: `pytest tests/` exited 2 having executed nothing at all, and the
# only clue was one ERROR line for this file. A skip costs this file and
# nothing else.
#
# pypdf is a declared dependency, so in a correct
# environment this skips nothing -- it is here so that an incorrect one fails
# narrowly.
PdfReader = pytest.importorskip("pypdf").PdfReader

from spacr.regression_panels import (
    PanelNarrative,
    PanelStyle,
    apply_primary_call,
    guide_control_threshold,
    shared_limits,
    write_panel_package,
)


def _guide_results():
    return pd.DataFrame(
        {
            "grna": ["000000_1", "000000_2", "225160_2", "239740_3"],
            "effect": [0.10, 0.20, 0.70, 0.80],
            "bh": [False, False, True, True],
            "lopit": [
                "Non-targeting", "Non-targeting", "dense granules", "dense granules"
            ],
            "plot_y": [0.1, 0.2, 3.0, 4.0],
            "gene_label": [
                "Non-targeting control",
                "Non-targeting control",
                "EAF1 (TGME49_225160)",
                "GRA14 (TGME49_239740)",
            ],
            "gene_url": [
                "https://toxodb.org/toxo/app",
                "https://toxodb.org/toxo/app",
                "https://toxodb.org/toxo/app/record/gene/TGME49_225160",
                "https://toxodb.org/toxo/app/record/gene/TGME49_239740",
            ],
        }
    )


def test_gene_call_borrows_the_guide_control_threshold():
    guide = _guide_results()
    threshold, audit = guide_control_threshold(guide, effect_column="effect")
    expected = 0.15 + 3 * np.std([0.10, 0.20], ddof=1)
    assert np.isclose(threshold, expected)
    assert audit["control_grnas"] == 2
    assert audit["effect_multiplier"] == 3.0

    gene = pd.DataFrame(
        {"effect": [threshold - 0.01, threshold + 0.01], "bh": [True, True]}
    )
    called = apply_primary_call(
        gene,
        effect_column="effect",
        bh_column="bh",
        effect_threshold=threshold,
    )
    assert called["primary_call"].tolist() == [False, True]
    assert called["plot_effect_threshold"].nunique() == 1


def test_matched_panels_get_identical_finite_limits():
    first = pd.DataFrame({"x": [0.1, 0.8], "y": [0.0, 4.0]})
    second = pd.DataFrame({"x": [-0.2, 1.1], "y": [0.2, 7.0]})
    x_limits, y_limits = shared_limits(
        [first, second], x_column="x", y_column="y"
    )
    assert x_limits[0] <= -0.2
    assert x_limits[1] >= 1.1
    assert y_limits[0] == 0.0
    assert y_limits[1] >= 7.0


def test_panel_package_has_exactly_the_four_contract_files(tmp_path):
    frame = _guide_results()
    destination = tmp_path / "Figure_5" / "Panel_B"
    paths = write_panel_package(
        frame,
        destination,
        panel_id="Figure_5B",
        x_column="effect",
        y_column="plot_y",
        lopit_column="lopit",
        x_label="Effect",
        y_label="-log10(P)",
        x_limits=(0.0, 1.0),
        y_limits=(0.0, 5.0),
        horizontal_threshold=1.3,
        horizontal_threshold_label="BH boundary",
        effect_threshold=0.5,
        effect_threshold_label="gRNA NT mean + 3 SD",
        narrative=PanelNarrative(
            legend="Each point is a tested gRNA.",
            purpose="Rank gRNAs.",
            shows="Two effects pass both lines.",
            implications="The panel prioritizes candidates.",
        ),
        gene_label_column="gene_label",
        gene_url_column="gene_url",
        statistics={"bh_significant": 2, "primary_calls": 2},
        style=PanelStyle(point_size=104, point_alpha=0.60, png_dpi=100),
    )
    assert {path.name for path in destination.iterdir()} == {
        "Figure_5B.pdf",
        "Figure_5B.png",
        "Figure_5B_stats.csv",
        "Figure_5B_data.csv",
    }
    data = pd.read_csv(paths["data"])
    assert len(data) == len(frame)
    assert data["point_size"].eq(104).all()
    assert data["point_alpha"].eq(0.60).all()
    assert data["marker_edge_color"].eq("none").all()
    assert data["line_color"].eq("#000000").all()
    assert data["line_width_points"].eq(0.50).all()
    stats = pd.read_csv(paths["stats"])
    values = dict(zip(stats["metric"], stats["value"]))
    assert data["label_above_threshold"].sum() == 2
    assert float(values["point_size"]) == 104
    assert float(values["point_alpha"]) == 0.60
    assert float(values["line_width_points"]) == 0.50
    assert values["marker_edge_color"] == "none"
    assert int(values["linked_points"]) == len(frame)
    assert int(values["labeled_points"]) == 2
    assert values["legend_title"] == "LOPIT/TAGM"
    annotations = PdfReader(paths["pdf"]).pages[0]["/Annots"]
    assert len(annotations) == len(frame)
    pdf_text = PdfReader(paths["pdf"]).pages[0].extract_text()
    assert "LOPIT/TAGM" in pdf_text
    assert "EAF1" in pdf_text
    assert "GRA14" in pdf_text
