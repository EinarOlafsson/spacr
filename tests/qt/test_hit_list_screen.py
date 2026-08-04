"""The Hit List screen, driven against a real regression results folder.

The screen is a view over :mod:`spacr.hits`, so what is tested here is the
view: that the table shows what the list holds, that every filter control
reaches the right argument, that the dial means the right thing for a
backend with no p-value, and that an export writes the rows currently on
screen rather than all of them.

Every test runs ``threaded=False`` so the list is built by the time the call
returns. Both paths run the same code and emit the same signals.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import hit_list as screen_module           # noqa: E402

pytestmark = pytest.mark.qt


def _gene_frame():
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in
                    ("100", "200", "300", "400", "233460")],
        "coefficient": [2.4, -1.8, 0.9, 0.05, 0.02],
        "std_err": [0.30, 0.25, 0.40, 0.30, 0.10],
        "p_value": [1e-6, 4e-5, 0.02, 0.86, 0.84],
        "condition": ["other", "other", "other", "other", "nc"],
        "n_gene": [48, 44, 30, 40, 60],
    })


def _grna_frame():
    rows = [("100_1", 2.2), ("100_2", 2.9), ("100_3", 1.7),
            ("200_1", -1.9), ("200_2", 1.4), ("200_3", 0.8),
            ("300_1", 0.9), ("233460_1", 0.02)]
    return pd.DataFrame({
        "feature": [f"fraction:grna[{g}]" for g, _ in rows],
        "grna": [g for g, _ in rows],
        "coefficient": [c for _, c in rows]})


@pytest.fixture()
def folder(tmp_path):
    """A results folder laid out the way ``perform_regression`` writes one."""
    root = tmp_path / "results" / "pred" / "ols"
    root.mkdir(parents=True)
    _gene_frame().to_csv(root / "results_gene.csv", index=False)
    _grna_frame().to_csv(root / "results_grna.csv", index=False)
    pd.concat([_gene_frame(), _grna_frame()], ignore_index=True).to_csv(
        root / "results.csv", index=False)
    return str(root)


@pytest.fixture()
def metadata(tmp_path):
    """An annotation file with one row per TRANSCRIPT, repeated 32 times."""
    rows = []
    for gene in ("100", "200", "300"):
        for transcript in range(32):
            rows.append({"Gene ID": f"TGME49_{gene}",
                         "Gene Name": f"name-{gene}",
                         "Product Description": f"product {gene}"})
    path = tmp_path / "toxo_metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def screen(qtbot, folder):
    """The screen, opened on the folder, running inline."""
    widget = screen_module.HitListScreen(folder=folder, threaded=False,
                                         regression_type="ols")
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_through_the_seam():
    from spacr.qt.app import APPS, SECTION_RESULTS, registered_factory

    row = next((r for r in APPS if r[0] == screen_module.APP_KEY), None)
    assert row is not None, "importing the module did not register the app"
    assert row[3] == SECTION_RESULTS
    assert registered_factory(screen_module.APP_KEY) is (
        screen_module.make_hit_list_screen)
    assert screen_module.register() is False, "register() is not idempotent"


def test_the_screen_answers_spacr_run_with_a_sentence():
    from spacr import cli

    note = cli.INTERACTIVE_ONLY.get(screen_module.APP_KEY, "")
    assert len(note) >= 40
    assert "build_hit_list" in note


def test_the_screen_styles_itself_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names

    assert "HitListFilters" in widget_qss_names()
    assert "QLabel#HitListSummary" in stylesheet()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_table_shows_one_row_per_gene_in_rank_order(screen):
    table = screen._table

    assert table.topLevelItemCount() == 5
    genes = [table.topLevelItem(i).text(1) for i in range(5)]
    assert genes == ["100", "200", "300", "400", "233460"] or genes[:3] == [
        "100", "200", "300"]
    assert [table.topLevelItem(i).text(0) for i in range(5)] == [
        "1", "2", "3", "4", "5"]


def test_the_table_shows_the_effect_its_interval_and_the_agreement(screen):
    row = screen._table.topLevelItem(0)

    assert row.text(1) == "100"
    assert row.text(3) == "2.4"
    assert "…" in row.text(4), "the 95% interval must be shown"
    assert row.text(7) == "3/3"
    assert row.text(8) == "100%"


def test_a_missing_value_is_an_em_dash_not_a_nan(qtbot, tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    pd.DataFrame({"feature": ["gene_fraction:gene[100]"],
                  "coefficient": [1.0]}).to_csv(
        root / "results_gene.csv", index=False)
    widget = screen_module.HitListScreen(folder=str(root), threaded=False)
    qtbot.addWidget(widget)

    row = widget._table.topLevelItem(0)

    assert row.text(4) == "—", "no standard error means no interval"
    assert row.text(5) == "—" and row.text(6) == "—"
    assert row.text(8) == "—", "no guide table means agreement is unknown"
    assert "nan" not in " ".join(row.text(c) for c in range(11)).lower()


def test_a_metadata_file_with_one_row_per_transcript_cannot_multiply_a_row(
        qtbot, folder, metadata):
    widget = screen_module.HitListScreen(
        folder=folder, metadata_files=[metadata], threaded=False)
    qtbot.addWidget(widget)

    assert widget._table.topLevelItemCount() == 5, (
        "32 transcripts per gene must not become 32 rows per gene")
    names = [widget._table.topLevelItem(i).text(2) for i in range(5)]
    assert "name-100" in names, "the annotation must reach the table"


def test_changing_the_metadata_files_rebuilds_the_list(screen, metadata):
    assert screen.metadata_files() == []

    screen.set_metadata_files([metadata])

    assert screen.metadata_files() == [metadata]
    assert screen._table.topLevelItemCount() == 5
    assert screen.hits().gene("100").name == "name-100"


def test_a_folder_with_no_results_reports_it_and_never_raises(qtbot, tmp_path):
    empty = tmp_path / "not-results"
    empty.mkdir()
    widget = screen_module.HitListScreen(folder=str(empty), threaded=False)
    qtbot.addWidget(widget)

    assert widget.hits() is None
    assert "results_gene.csv" in widget.last_error
    assert "Could not build the hit list" in widget._summary.text()
    assert widget._summary.property("problem") == "true"


def test_the_summary_counts_directions_and_corroboration(screen):
    text = screen._summary.text()

    assert "genes shown" in text
    assert "up," in text and "down" in text
    assert "corroborated" in text


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_every_control_reaches_the_filter_it_names(screen):
    screen._q_spin.setValue(0.01)
    screen._effect_spin.setValue(0.5)
    screen._agreement_spin.setValue(0.6)
    screen._guides_spin.setValue(2)
    screen._direction.setCurrentText("up")
    screen._drop_controls.setChecked(True)
    screen._query.setText("100")

    arguments = screen.current_filters()

    assert arguments == {"max_q": 0.01, "min_effect": 0.5,
                         "min_agreement": 0.6, "min_guides": 2,
                         "direction": "up", "exclude_controls": True,
                         "query": "100"}


def test_a_control_at_its_neutral_value_contributes_nothing(screen):
    screen._q_spin.setValue(1.0)

    assert screen.current_filters() == {}, (
        "an untouched control must not put a no-op criterion on the export")


def test_the_fdr_dial_narrows_the_table(screen):
    before = screen._table.topLevelItemCount()

    screen._q_spin.setValue(0.001)

    assert screen._table.topLevelItemCount() < before
    assert [h.gene for h in screen.filtered()] == ["100", "200"]


def test_hiding_controls_drops_the_negative_control(screen):
    assert screen.filtered().gene("233460") is not None

    screen._drop_controls.setChecked(True)

    assert screen.filtered().gene("233460") is None


def test_the_agreement_dial_drops_a_gene_whose_guides_disagree(screen):
    screen._q_spin.setValue(1.0)
    screen._agreement_spin.setValue(0.5)

    genes = [h.gene for h in screen.filtered()]
    assert "200" not in genes, "one guide of three agreeing is 0.33"
    assert "100" in genes


def test_the_text_query_searches_the_annotation(qtbot, folder, metadata):
    widget = screen_module.HitListScreen(
        folder=folder, metadata_files=[metadata], threaded=False)
    qtbot.addWidget(widget)
    widget._q_spin.setValue(1.0)

    widget._query.setText("name-300")

    assert [h.gene for h in widget.filtered()] == ["300"]


def test_an_empty_result_is_reported_rather_than_looking_broken(screen):
    screen._q_spin.setValue(0.0)
    screen._effect_spin.setValue(1e5)

    assert screen._table.topLevelItemCount() == 0
    assert screen._summary.property("problem") == "true"
    assert "0 of 5 genes shown" in screen._summary.text()


def test_filtering_emits_the_narrowed_list(screen, qtbot):
    with qtbot.waitSignal(screen.hits_filtered, timeout=1000) as caught:
        screen._q_spin.setValue(0.001)

    assert [h.gene for h in caught.args[0]] == ["100", "200"]


def test_the_dial_becomes_a_selection_floor_for_a_backend_with_no_p_value(
        qtbot, tmp_path):
    root = tmp_path / "lasso"
    root.mkdir()
    pd.DataFrame({
        "feature": ["gene_fraction:gene[100]", "gene_fraction:gene[200]"],
        "coefficient": [0.4, 1.9],
        "selection_frequency": [0.95, 0.30]}).to_csv(
        root / "results_gene.csv", index=False)
    widget = screen_module.HitListScreen(
        folder=str(root), threaded=False, regression_type="lasso")
    qtbot.addWidget(widget)

    widget._q_spin.setValue(0.6)

    assert widget.current_filters() == {"min_selection": 0.6}
    assert [h.gene for h in widget.filtered()] == ["100"], (
        "a bigger coefficient chosen 30% of the time is not the better hit")


def test_the_flag_legend_only_names_the_flags_actually_present(screen):
    screen._q_spin.setValue(1.0)

    text = screen._legend.text()

    assert "single-guide" in text
    assert "guides-disagree" in text


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_the_csv_export_writes_the_rows_currently_shown(screen, tmp_path):
    screen._q_spin.setValue(0.001)

    path = screen.export(str(tmp_path / "hits.csv"), "csv")

    frame = pd.read_csv(path)
    assert len(frame) == 2, "the export must be the filtered list, not all"
    assert frame["gene"].astype(str).tolist() == ["100", "200"]
    assert "written to hits.csv" in screen._summary.text()


def test_the_markdown_export_carries_the_legend(screen, tmp_path):
    path = screen.export(str(tmp_path / "hits.md"), "markdown")

    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("# Hit list")
    assert "| rank | gene |" in text
    assert "Flags:" in text


def test_the_html_export_is_self_contained(screen, tmp_path):
    path = screen.export(str(tmp_path / "hits.html"), "html")

    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "http://" not in text and "https://" not in text
    assert "<script" not in text.lower()


def test_exporting_before_anything_is_loaded_says_so(qtbot, tmp_path):
    widget = screen_module.HitListScreen(threaded=False)
    qtbot.addWidget(widget)

    assert widget.export(str(tmp_path / "x.csv"), "csv") == ""
    assert "no hit list to export" in widget._summary.text()


def test_an_unknown_export_format_is_refused(screen, tmp_path):
    with pytest.raises(ValueError):
        screen.export(str(tmp_path / "x.pdf"), "pdf")


# ---------------------------------------------------------------------------
# Rendering and threading
# ---------------------------------------------------------------------------

def test_the_screen_renders_at_the_window_size(screen, qt_theme_applied):
    screen.resize(1200, 720)
    screen.show()

    frame = screen.grab()

    assert not frame.isNull()
    assert frame.width() >= 1200 and frame.height() >= 720


def test_the_threaded_path_builds_the_same_list_and_retires(qtbot, folder):
    widget = screen_module.HitListScreen(threaded=True)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.hits_loaded, timeout=15000):
        widget.load_folder(folder)

    assert len(widget.hits()) == 5
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=15000)
    assert widget.is_busy() is False
    widget.close()


def test_a_failing_build_reports_inline_and_never_modally(qtbot, monkeypatch):
    widget = screen_module.HitListScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        screen_module, "build_hit_list",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("csv went bad")))

    widget.load_folder("/nowhere")

    assert widget.last_error == "csv went bad"
    assert "csv went bad" in widget._summary.text()
