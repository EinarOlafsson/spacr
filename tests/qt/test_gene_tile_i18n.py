"""Localization contracts for structured gene-result tiles."""
from __future__ import annotations

import html

from spacr.gene_tile import GeneCandidate, GeneTile, gene_tile
from spacr.qt.widgets import gene_tile as widget_module
from spacr.qt.widgets.gene_tile import GeneTilePanel


def _fake_translation(source, language=None, **values):
    rendered = str(source).format(**values)
    return f"[{language or 'xx'}] {rendered}"


def test_structured_tile_translates_chrome_but_not_gene_data():
    candidate = GeneCandidate(
        gene="239740",
        accession="TGGT1_239740",
        product="Dense granule protein",
    )
    tile = GeneTile(
        feature="fraction:grna[TGGT1_239740_3]",
        kind="guide",
        candidates=(candidate,),
        effect=1.25,
        p_value=0.004,
    )

    rendered = tile.to_html(_fake_translation)

    assert "[xx] gene id" in rendered
    assert "[xx] effect (coefficient)" in rendered
    assert "TGGT1_239740" in rendered
    assert "Dense granule protein" in rendered
    assert "[xx] TGGT1_239740" not in rendered


def test_dynamic_resolution_message_retains_values_when_translated():
    tile = gene_tile("Intercept", barcodes=None, metadata=None,
                     localisation=None)

    rendered = tile.to_html(_fake_translation)

    plain = html.unescape(rendered)
    assert "[xx] 'Intercept' is a model covariate" in plain
    assert plain.count("Intercept") >= 2


def test_panel_rerenders_idle_text_after_a_language_change(qtbot, monkeypatch):
    monkeypatch.setattr(widget_module, "tr", _fake_translation)
    panel = GeneTilePanel()
    qtbot.addWidget(panel)

    panel.retranslate_dynamic_content("de")

    assert panel._view.toPlainText().startswith("[de] Click a point")
