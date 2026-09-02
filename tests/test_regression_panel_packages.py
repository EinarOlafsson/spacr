import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import pytest

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
_pypdf = pytest.importorskip("pypdf")
PdfReader = _pypdf.PdfReader
PdfWriter = _pypdf.PdfWriter

from spacr import regression_panels as panels  # noqa: E402
from spacr.regression_panels import (  # noqa: E402
    PanelNarrative,
    PanelStyle,
    apply_primary_call,
    build_manifest_packages,
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


def _package_kwargs(frame, destination, **overrides):
    kwargs = {
        "results": frame,
        "destination": destination,
        "panel_id": "Figure_5B",
        "x_column": "effect",
        "y_column": "plot_y",
        "lopit_column": "lopit",
        "x_label": "Effect",
        "y_label": "-log10(P)",
        "x_limits": (0.0, 1.0),
        "y_limits": (0.0, 5.0),
        "horizontal_threshold": 1.3,
        "horizontal_threshold_label": "BH boundary",
        "effect_threshold": 0.5,
        "effect_threshold_label": "gRNA NT mean + 3 SD",
        "narrative": PanelNarrative(
            legend="Each point is a tested gRNA.",
            purpose="Rank gRNAs.",
            shows="Two effects pass both lines.",
            implications="The panel prioritizes candidates.",
        ),
        "style": PanelStyle(point_size=104, point_alpha=0.60, png_dpi=100),
    }
    kwargs.update(overrides)
    return kwargs


def _blank_pdf(path, pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with Path(path).open("wb") as handle:
        writer.write(handle)


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


def test_prefixed_control_guides_use_the_same_threshold():
    guide = _guide_results()
    guide["grna"] = "TGME49_" + guide["grna"]

    threshold, audit = guide_control_threshold(guide, effect_column="effect")

    expected = 0.15 + 3 * np.std([0.10, 0.20], ddof=1)
    assert np.isclose(threshold, expected)
    assert audit["control_grnas"] == 2


def test_the_control_threshold_refuses_an_unusable_guide_table():
    with pytest.raises(ValueError, match="lacks 'grna'"):
        guide_control_threshold(
            pd.DataFrame({"effect": [0.1, 0.2]}), effect_column="effect"
        )
    with pytest.raises(ValueError, match="At least two"):
        guide_control_threshold(
            pd.DataFrame({"grna": ["000000_1"], "effect": [0.1]}),
            effect_column="effect",
        )


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


def test_shared_limits_refuse_no_panels_and_nonfinite_values():
    with pytest.raises(ValueError, match="At least one frame"):
        shared_limits([], x_column="x", y_column="y")
    with pytest.raises(ValueError, match="must be finite"):
        shared_limits(
            [pd.DataFrame({"x": [0.1, np.inf], "y": [0.2, 0.4]})],
            x_column="x",
            y_column="y",
        )


def test_the_bundled_font_has_a_named_fallback(monkeypatch):
    original_exists = Path.exists

    def without_open_sans(path):
        if path.name in {"OpenSans-Regular.ttf", "OpenSans-Bold.ttf"}:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", without_open_sans)
    module_name = "spacr._regression_panels_font_fallback"
    specification = importlib.util.spec_from_file_location(module_name, panels.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, module_name, module)
    specification.loader.exec_module(module)

    assert module.OPEN_SANS_FAMILY == "Open Sans"


def test_pdf_link_validation_refuses_ambiguous_inputs(tmp_path):
    two_pages = tmp_path / "two-pages.pdf"
    one_page = tmp_path / "one-page.pdf"
    _blank_pdf(two_pages, pages=2)
    _blank_pdf(one_page)
    empty = pd.DataFrame(columns=["x", "y", "label", "url"])
    kwargs = {
        "data": empty,
        "x_limits": (0.0, 1.0),
        "y_limits": (0.0, 1.0),
        "x_column": "x",
        "y_column": "y",
        "gene_label_column": "label",
        "gene_url_column": "url",
        "style": PanelStyle(),
    }

    with pytest.raises(ValueError, match="one-page PDF"):
        panels._add_pdf_point_links(two_pages, **kwargs)
    with pytest.raises(ValueError, match="increasing axis limits"):
        panels._add_pdf_point_links(
            one_page, **{**kwargs, "x_limits": (1.0, 1.0)}
        )

    missing_link = pd.DataFrame(
        {"x": [0.5], "y": [0.5], "label": ["gene"], "url": [""]}
    )
    with pytest.raises(ValueError, match="gene label and URL"):
        panels._add_pdf_point_links(
            one_page, **{**kwargs, "data": missing_link}
        )


@pytest.mark.parametrize(
    "frame,style,message",
    [
        (
            _guide_results().assign(lopit="not in the palette"),
            PanelStyle(),
            "lack colours",
        ),
        (_guide_results(), PanelStyle(point_alpha=0.0), "point_alpha"),
        (_guide_results(), PanelStyle(point_size=0.0), "point_size"),
    ],
)
def test_panel_packages_refuse_invalid_visual_contracts(
    tmp_path, frame, style, message
):
    with pytest.raises(ValueError, match=message):
        write_panel_package(
            **_package_kwargs(frame, tmp_path / message, style=style)
        )


def test_link_columns_are_supplied_as_a_pair(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        write_panel_package(
            **_package_kwargs(
                _guide_results(),
                tmp_path / "unpaired",
                gene_label_column="gene_label",
            )
        )


def test_an_unlinked_package_draws_no_point_labels(tmp_path):
    paths = write_panel_package(
        **_package_kwargs(_guide_results(), tmp_path / "unlinked")
    )

    stats = pd.read_csv(paths["stats"])
    values = dict(zip(stats["metric"], stats["value"]))
    assert int(values["linked_points"]) == 0
    assert int(values["labeled_points"]) == 0


def test_a_panel_folder_refuses_an_unmanifested_fifth_file(
    tmp_path, monkeypatch
):
    from matplotlib import pyplot as plt

    from spacr import plot, tabular

    def quick_save(figure, path, **_kwargs):
        Path(path).write_bytes(b"figure")
        plt.close(figure)
        return str(path)

    write_table = tabular.write_table

    def write_with_extra(frame, path, **kwargs):
        written = write_table(frame, path, **kwargs)
        Path(path).parent.joinpath("unexpected.txt").write_text(
            "not in the manifest", encoding="utf-8"
        )
        return written

    monkeypatch.setattr(plot, "save_figure", quick_save)
    monkeypatch.setattr(tabular, "write_table", write_with_extra)

    with pytest.raises(RuntimeError, match="exactly four files"):
        write_panel_package(
            **_package_kwargs(_guide_results(), tmp_path / "extra")
        )


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
    assert data["line_color"].eq(panels.INK_PRINT).all()
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


def _manifest_frame(phenotype, level, control_effects):
    if level == "grna":
        names = ["000000_1", "000000_2", "225160_2", "239740_3"]
        effects = [*control_effects, 0.90, 1.10]
    else:
        names = ["225160", "239740", "245530", "250000"]
        effects = [0.25, 0.70, 0.95, 1.20]
    return pd.DataFrame(
        {
            level: names,
            "effect": effects,
            "bh": [False, False, True, True],
            "plot_y": [0.2, 0.5, 3.0, 4.0],
            "lopit": [
                "Non-targeting", "cytosol", "dense granules", "micronemes"
            ],
            "gene_label": [f"{phenotype}-{name}" for name in names],
            "gene_url": [f"https://example.org/{phenotype}/{name}" for name in names],
            "group": ["reference", "reference", "candidate", "candidate"],
            "box_value": effects,
        }
    )


def _publication_artifacts():
    return {
        "xgboost_grna": {
            "phenotype": "xgboost", "level": "grna",
            "data": _manifest_frame("xgboost", "grna", (0.10, 0.20)),
        },
        "xgboost_gene": {
            "phenotype": "xgboost", "level": "gene",
            "data": _manifest_frame("xgboost", "gene", (0.10, 0.20)),
        },
        "maxvit_grna": {
            "phenotype": "maxvit", "level": "grna",
            "data": _manifest_frame("maxvit", "grna", (0.30, 0.50)),
        },
        "maxvit_gene": {
            "phenotype": "maxvit", "level": "gene",
            "data": _manifest_frame("maxvit", "gene", (0.30, 0.50)),
        },
    }


def _narrative(subject):
    return {
        "legend": f"Each mark is one {subject}.",
        "purpose": f"Compare {subject} effects.",
        "shows": f"The {subject} results and their uncertainty.",
        "implications": f"Prioritize reproducible {subject} effects.",
    }


def _scatter(panel_id, source, phenotype, level, limit_group):
    return {
        "panel_id": panel_id,
        "source": source,
        "phenotype": phenotype,
        "level": level,
        "kind": "scatter",
        "effect_column": "effect",
        "bh_column": "bh",
        "y_column": "plot_y",
        "lopit_column": "lopit",
        "gene_label_column": "gene_label",
        "gene_url_column": "gene_url",
        "x_label": "Effect",
        "y_label": "-log10(q)",
        "horizontal_threshold": 1.3,
        "horizontal_threshold_label": "BH boundary",
        "limit_group": limit_group,
        "narrative": _narrative(level),
    }


def _publication_manifest():
    return {
        "figure_id": "Figure_7",
        "columns": 2,
        "panels": [
            _scatter("Figure_7A", "xgboost_grna", "xgboost", "grna", "guides"),
            _scatter("Figure_7B", "xgboost_gene", "xgboost", "gene", "genes"),
            _scatter("Figure_7C", "maxvit_grna", "maxvit", "grna", "guides"),
            {
                "panel_id": "Figure_7D",
                "source": "maxvit_gene",
                "phenotype": "maxvit",
                "level": "gene",
                "kind": "box_jitter",
                "effect_column": "effect",
                "bh_column": "bh",
                "category_column": "group",
                "value_column": "box_value",
                "category_order": ["reference", "candidate"],
                "gene_label_column": "gene_label",
                "gene_url_column": "gene_url",
                "x_label": "Group",
                "y_label": "Effect",
                "narrative": _narrative("gene group"),
            },
        ],
    }


def _stats_values(path):
    table = pd.read_csv(path)
    return dict(zip(table["metric"], table["value"]))


def test_manifest_builds_four_traceable_packages_and_one_vector_figure(tmp_path):
    manifest_path = tmp_path / "panels.json"
    manifest_path.write_text(json.dumps(_publication_manifest()), encoding="utf-8")
    destination = tmp_path / "publication"

    written = build_manifest_packages(
        manifest_path,
        _publication_artifacts(),
        destination,
        style=PanelStyle(png_dpi=100),
    )

    assert set(written["panels"]) == {
        "Figure_7A", "Figure_7B", "Figure_7C", "Figure_7D"
    }
    for panel_id, paths in written["panels"].items():
        folder = destination / panel_id
        assert set(path.name for path in folder.iterdir()) == {
            f"{panel_id}.pdf", f"{panel_id}.png",
            f"{panel_id}_stats.csv", f"{panel_id}_data.csv",
        }
        assert set(paths) == {"pdf", "png", "stats", "data"}

    xgb_guide = _stats_values(written["panels"]["Figure_7A"]["stats"])
    xgb_gene = _stats_values(written["panels"]["Figure_7B"]["stats"])
    maxvit_guide = _stats_values(written["panels"]["Figure_7C"]["stats"])
    maxvit_gene = _stats_values(written["panels"]["Figure_7D"]["stats"])
    plotted_thresholds = {}
    for panel_id, paths in written["panels"].items():
        values = pd.read_csv(paths["data"])["plot_effect_threshold"].unique()
        assert len(values) == 1
        plotted_thresholds[panel_id] = float(values[0])
    assert plotted_thresholds["Figure_7A"] == plotted_thresholds["Figure_7B"]
    assert plotted_thresholds["Figure_7C"] == plotted_thresholds["Figure_7D"]
    assert float(xgb_guide["effect_threshold"]) == float(
        xgb_gene["effect_threshold"]
    )
    assert float(maxvit_guide["effect_threshold"]) == float(
        maxvit_gene["effect_threshold"]
    )
    assert not np.isclose(
        float(xgb_guide["effect_threshold"]),
        float(maxvit_guide["effect_threshold"]),
    )
    for key in ("x_min", "x_max", "y_min", "y_max"):
        assert float(xgb_guide[key]) == float(maxvit_guide[key])

    box_data = pd.read_csv(written["panels"]["Figure_7D"]["data"])
    assert box_data["box_alpha"].eq(0.60).all()
    assert box_data["point_alpha"].eq(0.60).all()
    assert np.allclose(box_data["plot_x"], [-0.18, 0.18, 0.82, 1.18])
    assert np.allclose(box_data["plot_y"], box_data["box_value"])

    final_pdf = written["figure_pdf"]
    assert final_pdf == destination / "Figure_7.pdf"
    reader = PdfReader(final_pdf)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    xobjects = page.get("/Resources", {}).get("/XObject", {})
    assert all(
        item.get_object().get("/Subtype") != "/Image"
        for item in xobjects.values()
    )
    assert len(page.get("/Annots", [])) == 16
    page_midpoint = (
        float(page.mediabox.width) / 2,
        float(page.mediabox.height) / 2,
    )
    occupied_quadrants = set()
    for annotation in page["/Annots"]:
        rect = [float(value) for value in annotation.get_object()["/Rect"]]
        assert rect[2] > rect[0] and rect[3] > rect[1]
        occupied_quadrants.add(
            (
                (rect[0] + rect[2]) / 2 > page_midpoint[0],
                (rect[1] + rect[3]) / 2 > page_midpoint[1],
            )
        )
    assert occupied_quadrants == {
        (False, False), (False, True), (True, False), (True, True)
    }


def test_manifest_validation_finishes_before_any_package_is_written(tmp_path):
    manifest = _publication_manifest()
    manifest["panels"][-1]["source"] = "missing_run_artifact"
    destination = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="missing_run_artifact"):
        build_manifest_packages(manifest, _publication_artifacts(), destination)

    assert not destination.exists()


def test_manifest_refuses_source_level_or_phenotype_guessing(tmp_path):
    artifacts = _publication_artifacts()
    artifacts["xgboost_gene"]["phenotype"] = "maxvit"

    with pytest.raises(ValueError, match="declares phenotype"):
        build_manifest_packages(
            _publication_manifest(), artifacts, tmp_path / "not-written"
        )
