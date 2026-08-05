"""The Methods & Results screen.

What is tested is the guarantee the screen makes to the user: that the
sections on display are made of the run, and that the strip under the tabs
says so honestly. That means the number check has to reach the panel — a
refused draft must not appear as prose with a quiet warning somewhere — and
the digest has to travel with the export, because a methods section whose
provenance lives in another file is a methods section with no provenance.

Every test runs ``threaded=False``.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.methods_export import render_methods                  # noqa: E402
from spacr.qt.screens import methods_export as screen_module     # noqa: E402

pytestmark = pytest.mark.qt

PLANTED = 48.3179


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    from spacr import artifacts

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


@pytest.fixture()
def results_folder(tmp_path):
    """A regression results folder carrying the planted effect."""
    folder = tmp_path / "results" / "pred" / "ols"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "feature": ["gene_fraction:gene[233460]", "gene_fraction:gene[239740]"],
        "coefficient": [PLANTED, -2.9013],
        "p_value": [1.2e-6, 0.31]}).to_csv(folder / "results_gene.csv",
                                           index=False)
    return str(folder)


@pytest.fixture()
def screen(qtbot, results_folder):
    """The screen, pointed at the results folder, running inline."""
    widget = screen_module.MethodsExportScreen(
        results_folder=results_folder, threaded=False)
    qtbot.addWidget(widget)
    return widget


class _Draft:
    """A stand-in :class:`~spacr.qt.ai.manuscript.ManuscriptDraft`."""

    def __init__(self, ok, methods="M", results="R", problems=(),
                 rejected="", checked=7):
        self.ok = ok
        self.methods = methods
        self.results = results
        self.problems = list(problems)
        self.rejected = rejected
        self.provider = "stub"
        self.source = "model" if ok else "digest"
        self.methods_check = type("V", (), {"checked": checked})()
        self.results_check = type("V", (), {"checked": checked})()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_through_the_seam():
    from spacr.qt.app import APPS, SECTION_RESULTS, registered_factory

    row = next((r for r in APPS if r[0] == screen_module.APP_KEY), None)
    assert row is not None, "importing the module did not register the app"
    assert row[3] == SECTION_RESULTS
    assert registered_factory(screen_module.APP_KEY) is (
        screen_module.make_methods_export_screen)
    assert screen_module.register() is False, "register() is not idempotent"


def test_the_screen_answers_spacr_run_with_a_sentence():
    from spacr import cli

    note = cli.INTERACTIVE_ONLY.get(screen_module.APP_KEY, "")
    assert len(note) >= 40
    assert "build_digest" in note
    assert "no AI provider" in note, (
        "the headless path must say it needs no model")


def test_the_screen_styles_itself_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names

    assert "MethodsExportSources" in widget_qss_names()
    assert "QLabel#MethodsExportProvenance" in stylesheet()


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------

def test_building_the_digest_fills_all_four_tabs(screen):
    screen.build()

    assert screen.digest() is not None
    assert screen._methods_view.toPlainText().startswith("## Methods")
    assert screen._results_view.toPlainText().startswith("## Results")
    assert screen._caveats_view.toPlainText()
    payload = json.loads(screen._digest_view.toPlainText())
    assert payload["statistics"]["n_genes_tested"] == 2


def test_the_planted_number_reaches_the_results_pane(screen):
    screen.build()

    assert str(PLANTED) in screen._results_view.toPlainText(), (
        "a number in the run must appear in the prose the user reads")


def test_the_digest_is_built_from_whichever_sources_are_named(screen,
                                                              tmp_path):
    project = tmp_path / "plate9"
    project.mkdir()
    screen._fields["project"].setText(str(project))

    screen.build()

    assert screen.sources()["project"] == str(project)
    assert screen.digest()["project"] == str(project)
    assert screen.digest()["title"] == "plate9"


def test_the_provenance_strip_says_who_wrote_the_sections(screen):
    screen.build()

    text = screen._provenance.text()

    assert "written by spaCR" in text
    assert "every number in them is from the run" in text
    assert screen._provenance.property("problem") == "false"


def test_a_source_that_cannot_be_read_is_reported_not_swallowed(screen,
                                                                tmp_path):
    screen._fields["model"].setText(str(tmp_path / "no_such_model.pth"))

    screen.build()

    assert screen.digest()["notes"], "the failure must reach the digest"
    assert "could not be read" in screen._provenance.text()
    assert screen._provenance.property("problem") == "true"


def test_the_caveats_tab_lists_every_caveat(screen):
    screen.build()

    text = screen._caveats_view.toPlainText()

    for caveat in screen.digest()["caveats"]:
        assert caveat in text


def test_the_prompt_is_available_for_inspection(screen):
    assert screen.prompt() == "", "there is no prompt before a digest"

    screen.build()

    assert str(PLANTED) in screen.prompt()
    assert "RUN DIGEST" in screen.prompt()


def test_drafting_before_a_digest_says_so(screen):
    screen.generate()

    assert "Build the digest first" in screen._provenance.text()
    assert screen._generate_button.isEnabled() is False


# ---------------------------------------------------------------------------
# The draft and its number check
# ---------------------------------------------------------------------------

def test_an_accepted_draft_replaces_the_sections_and_states_the_check(
        screen, monkeypatch):
    screen.build()
    monkeypatch.setattr(screen_module, "_generate",
                        lambda _digest: _Draft(True, "AI methods",
                                               "AI results", checked=9))

    screen.generate()

    assert screen._methods_view.toPlainText() == "AI methods"
    assert screen._results_view.toPlainText() == "AI results"
    text = screen._provenance.text()
    assert "Drafted by stub" in text
    assert "18 number(s)" in text
    assert "every one of them came from it" in text
    assert screen._provenance.property("problem") == "false"
    assert screen._tabs.isTabVisible(4) is False


def test_a_refused_draft_shows_the_problem_and_never_the_invented_prose(
        screen, monkeypatch):
    screen.build()
    monkeypatch.setattr(
        screen_module, "_generate",
        lambda digest: _Draft(
            False, render_methods(digest), "spaCR results",
            problems=["The generated draft was rejected…",
                      "  • Results: 1 number(s) in the draft are not in the "
                      "run digest: 9999"],
            rejected="## Results\n\nWe found 9999 hits."))

    screen.generate()

    assert "9999" not in screen._results_view.toPlainText(), (
        "the invented figure must not be displayed as the result")
    assert "9999" in screen._provenance.text()
    assert screen._provenance.property("problem") == "true"
    assert screen._tabs.isTabVisible(4) is True
    assert "9999" in screen._rejected_view.toPlainText(), (
        "but a human must be able to read what the model claimed")


def test_the_provenance_message_is_the_whole_guarantee(screen):
    accepted = screen.provenance_message(_Draft(True, checked=4))
    refused = screen.provenance_message(
        _Draft(False, problems=["a", "b"]))

    assert "8 number(s)" in accepted and "checked against the run" in accepted
    assert refused == "a\nb"


def test_a_new_digest_clears_the_previous_draft(screen, monkeypatch):
    screen.build()
    monkeypatch.setattr(screen_module, "_generate",
                        lambda _d: _Draft(True, "AI methods", "AI results"))
    screen.generate()
    assert screen.draft() is not None

    screen.build()

    assert screen.draft() is None
    assert screen._methods_view.toPlainText().startswith("## Methods")
    assert screen._tabs.isTabVisible(4) is False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_the_export_carries_the_digest_as_an_appendix(screen, tmp_path):
    screen.build()

    path = screen.export(str(tmp_path / "paper.md"))

    text = Path(path).read_text(encoding="utf-8")
    assert "## Methods" in text and "## Results" in text
    assert "## Appendix: run digest" in text
    body = text.split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(body)["digest_version"] == 1, (
        "a methods section whose provenance lives elsewhere has none")
    assert "written to paper.md".casefold() in (
        screen._provenance.text().casefold())


def test_exporting_before_anything_is_built_says_so(screen, tmp_path):
    assert screen.export(str(tmp_path / "x.md")) == ""
    assert "nothing to export" in screen._provenance.text()


def test_copying_puts_both_sections_on_the_clipboard(screen, qapp):
    screen.build()

    screen._on_copy()

    text = qapp.clipboard().text()
    assert "## Methods" in text and "## Results" in text
    assert "Both sections copied" in screen._provenance.text()


def test_copying_before_anything_is_built_says_so(screen, qapp):
    screen._on_copy()

    assert "nothing to copy" in screen._provenance.text()


# ---------------------------------------------------------------------------
# Rendering and threading
# ---------------------------------------------------------------------------

def test_the_screen_renders_at_the_window_size(screen, qt_theme_applied):
    screen.build()
    screen.resize(1200, 720)
    screen.show()

    frame = screen.grab()

    assert not frame.isNull()
    assert frame.width() >= 1200 and frame.height() >= 720


def test_the_threaded_path_builds_the_same_digest_and_retires(qtbot,
                                                              results_folder):
    widget = screen_module.MethodsExportScreen(
        results_folder=results_folder, threaded=True)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.digest_built, timeout=20000):
        widget.build()

    assert widget.digest()["statistics"]["n_genes_tested"] == 2
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=20000)
    assert widget.is_busy() is False
    widget.close()


def test_a_failing_build_reports_inline_and_never_modally(qtbot, monkeypatch):
    widget = screen_module.MethodsExportScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        screen_module, "build_digest",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("journal on fire")))

    widget.build()

    assert widget.last_error == "journal on fire"
    assert "journal on fire" in widget._provenance.text()


def test_the_real_ai_path_degrades_without_a_provider(screen, monkeypatch):
    """End to end through the real generator, with no provider configured."""
    from spacr.qt.ai import manuscript

    monkeypatch.setattr(manuscript, "configured_providers", lambda: [])
    screen.build()

    screen.generate()

    draft = screen.draft()
    assert draft is not None and draft.ok is False
    assert draft.source == "digest"
    assert "No AI provider is configured" in screen._provenance.text()
    assert screen._methods_view.toPlainText().startswith("## Methods"), (
        "a user with no AI still gets their methods section")
    assert str(PLANTED) in screen._results_view.toPlainText()
