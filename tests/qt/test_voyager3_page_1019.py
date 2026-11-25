"""Item 1019: Voyager3 has its own button and page beside OPS (alpha).

The Voyager3 switch sits left of AI on Mask Generation and opens a page
beside OPS. The page runs Voyager3 with or without AI through
:func:`spacr.voyager3.run_voyager3`, the code the OPS step and the OPS live
tab run, and shows the knowledge map: search, zoom, a click tile and
filters. Nothing here reaches UniProt or an AI: the runs are fakes and the
map is drawn from a fixture provenance record.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.qt


def _provenance(method="enrichment"):
    """A two-cluster run with its control, as voyager3_provenance.json records it."""
    def record(cluster, kind, genes, role, supporting, candidates, terms):
        parsed = {"status": "ok", "reasons": [], "biological_role": role,
                  "confidence_model": "", "confidence_rule": "high" if supporting else "none",
                  "confidence_agrees": None,
                  "support_fraction": len(supporting) / len(genes),
                  "supporting": [{"gene": g, "evidence": role} for g in supporting],
                  "misfit": [], "candidates": [{"gene": g, "suggested_role": role,
                                                "priority": None, "rationale": "fixture"}
                                               for g in candidates],
                  "unknown_genes": [], "summary": f"Top terms: {role}"}
        return {"cluster": cluster, "kind": kind, "genes": genes, "parsed": parsed,
                "terms": terms, "attempts": []}
    term = {"source": "GO", "term_id": "GO:0005840", "term_name": "ribosome", "k": 3,
            "n": 4, "K": 5, "N": 40, "fold": 6.0, "p": 1e-4, "fdr": 1e-3,
            "genes": ["RPL3", "RPL4", "RPS6"]}
    clusters = [
        record("0", "real", ["RPL3", "RPL4", "RPS6", "MYC"], "GO: ribosome",
               ["RPL3", "RPL4", "RPS6"], ["MYC"], [term]),
        record("1", "real", ["PSMA1", "PSMB2"], "no enriched term", [], [], []),
        record("0", "shuffled", ["RPL3", "PSMA1", "MYC", "PSMB2"], "no enriched term",
               [], [], []),
    ]
    facts = {g: {"gene": g, "found": True, "accession": f"P{n:05d}", "protein_name": f"{g} protein",
                 "function": f"{g} does things", "location": "cytoplasm",
                 "annotation_score": 5}
             for n, g in enumerate(["RPL3", "RPL4", "RPS6", "MYC", "PSMA1", "PSMB2"])}
    return {"voyager3": "voyager3-1", "method": method, "provider": "" if method == "enrichment"
            else "spacr", "provider_label": "spaCR AI", "facts": facts, "clusters": clusters}


def test_the_knowledge_map_holds_clusters_genes_and_terms_and_filters_them():
    from spacr.qt.voyager3_panel import _knowledge_map

    full = _knowledge_map(_provenance())
    kinds = [n["kind"] for n in full["nodes"]]
    assert kinds.count("cluster") == 2 and kinds.count("gene") == 6 and kinds.count("term") == 1
    assert len(full["edges"]) == 7
    roles = {n["label"]: n["info"]["role"] for n in full["nodes"] if n["kind"] == "gene"}
    assert roles["RPL3"] == "supporting" and roles["MYC"] == "candidate"
    assert sum(n["kind"] == "cluster" for n in _knowledge_map(
        _provenance(), shuffled=True)["nodes"]) == 3, "the shuffled control on request"
    only_high = _knowledge_map(_provenance(), genes=False, terms=False, confidences={"high"})
    assert [n["info"]["cluster"] for n in only_high["nodes"]] == ["0"]


@pytest.fixture
def page(qtbot):
    from spacr.qt.voyager3_panel import _Voyager3Page

    built = _Voyager3Page()
    qtbot.addWidget(built)
    built.resize(900, 600)
    return built


def test_clicking_a_gene_fills_its_tile_and_search_finds_it(qtbot, page):
    assert page.load(_provenance())
    assert "2 clusters named by the enrichment test" in page.status.text()
    gene = next(p for p in page.scatter.points() if str(p.data()).endswith(":RPS6"))
    page.scatter.sigClicked.emit(page.scatter, [gene], None)
    text = page.tile.toPlainText()
    assert "RPS6" in text and "supporting" in text and "RPS6 protein" in text
    whole = [high - low for low, high in page.plot.getViewBox().viewRange()]
    found = page.find("myc")
    assert found and found[0].endswith(":MYC")
    assert "candidate" in page.tile.toPlainText()
    spans = [high - low for low, high in page.plot.getViewBox().viewRange()]
    assert min(spans) < min(whole), "the map zooms onto what was found"
    page.filters["genes"].setChecked(False)
    assert not [n for n in page.map["nodes"] if n["kind"] == "gene"]
    page.find("ribosome")
    assert page.shown_node["kind"] in ("cluster", "term")


@pytest.mark.parametrize("method", ["enrichment", "ai"])
def test_both_versions_run_through_the_one_voyager3(qtbot, page, tmp_path, monkeypatch, method):
    seen = {}

    def fake_run(settings, report=None, out_dir=None, provider=None):
        seen.update(settings, out_dir=out_dir)
        report("Progress: 1/2, operation_type: Voyager3")
        path = tmp_path / "voyager3_provenance.json"
        path.write_text(json.dumps(_provenance(settings["ops_voyager3_method"])))
        return {"provenance": str(path), "outputs": {}, "clusters": 2, "folder": str(tmp_path)}

    monkeypatch.setattr("spacr.voyager3.run_voyager3", fake_run)
    table = tmp_path / "clusters.csv"
    table.write_text("gene,cluster\nRPL3,0\nRPL4,0\n")
    page.source.setText(str(table))
    page.method.setCurrentIndex(page.method.findData(method))
    assert page.run()
    qtbot.waitUntil(lambda: not page.is_running(), timeout=10000)
    assert seen["ops_voyager3_method"] == method
    assert seen["ops_voyager3_clusters"] == str(table)
    assert seen["out_dir"].endswith("ops_voyager3")
    assert any(n["kind"] == "cluster" for n in page.map["nodes"])
    named = "the enrichment test" if method == "enrichment" else "spaCR AI"
    assert named in page.status.text()


def test_an_ops_run_folder_is_read_through_its_dst_root(page, tmp_path):
    page.source.setText(str(tmp_path))
    settings, out_dir = page.run_settings()
    assert settings["dst_root"] == str(tmp_path) and out_dir is None
    assert settings["ops_voyager3_clusters"] is None


def test_the_voyager3_switch_sits_left_of_ai_and_opens_the_page(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.map_barcodes import page_fold
    from spacr.qt.voyager3_panel import _Voyager3Page

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    switch = screen._voyager3_switch
    assert switch is not None and switch.objectName() == "MaskVoyager3Switch"
    assert switch.text() == "Voyager3"
    row = switch.parentWidget().layout()
    assert row.indexOf(switch) + 1 == row.indexOf(screen._ai_switch)
    assert row.indexOf(screen._ops_switch) < row.indexOf(switch)
    fold = page_fold(screen, "voyager3")
    fold.set_shown(True)
    assert isinstance(fold.page, _Voyager3Page)
    fold.set_shown(False)
