"""What the Feature Dictionary panel says, and which dialog it hands out.

The detail pane composes sentences about one concrete column -- its channel
pair, whether it is metadata, which table it really lives in -- and must not
invent a definition, a row with nothing in it, or a column name it was not
given (a name is text: HTML in it is shown, never rendered).  The pane is
pinned to a column after a lookup and must drop that pin the moment the user
asks a different question.  ``register()`` reports whether the app row exists
whatever the theme does, and the shared dialog cache never hands out a dead
dialog or forgets the live one.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.  Offscreen, offline, no database; the context-menu runner
is a recorder, so no ``QMenu.exec`` can reach a headless run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

from spacr.feature_dict import CHANNEL_PAIR, FeatureDoc, parse_column  # noqa: E402
from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus instead of opening one, and leave nothing behind."""
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


@pytest.fixture
def qss_sandbox():
    """Take the panel's QSS block back out, whatever the test did to it."""
    from spacr.qt import theme as theme_mod

    had = fd.OBJECT_NAME in theme_mod.widget_qss_names()
    register = theme_mod.register_widget_qss
    theme_mod.unregister_widget_qss(fd.OBJECT_NAME)
    try:
        yield theme_mod
    finally:
        theme_mod.unregister_widget_qss(fd.OBJECT_NAME)
        if had:
            register(fd.OBJECT_NAME, fd._panel_qss)


def _doc(**overrides) -> FeatureDoc:
    fields = dict(
        key="cell_area", title="Area", kind="measurement", family="shape",
        concepts=(), description="How many pixels the object covers.",
        unit="px²", computed_by="regionprops", module="spacr.measure",
        object_types=("cell",), channel_scope="none", written_when=None,
        notes=None, examples=(),
    )
    fields.update(overrides)
    return FeatureDoc(**fields)


def _key(panel):
    doc = panel.current_doc()
    return doc.key if doc is not None else None


def _panel(qtbot, **kwargs):
    widget = fd.FeatureDictionaryPanel(**kwargs)
    qtbot.addWidget(widget)
    return widget


def test_doc_html_states_what_it_knows_and_drops_what_it_does_not(
        qapp, monkeypatch, subtests):
    """A missing definition is said, an empty row is dropped, a pair is named."""
    pair = parse_column("pathogen_channel_0_channel_2_M1_correlation_85")
    assert (pair.channel, pair.channel_2) == (0, 2)
    cases = [
        # id, doc overrides, entry, blank channel sentence, present, absent
        ("no curated definition", dict(description=None), None, False,
         ["No definition", "see the note below"], []),
        ("a definition is shown", {}, None, False,
         ["How many pixels the object covers."], ["No definition"]),
        ("an empty row is not rendered", {}, None, True,
         ["Unit", "Module"], ["Channel"]),
        ("a colocalisation pair names its channels",
         dict(key=pair.key, channel_scope=CHANNEL_PAIR), pair, False,
         ["channels 0 and 2"], []),
    ]
    for case_id, overrides, entry, blank, present, absent in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                if blank:
                    patch.setattr(fd, "_channel_sentence", lambda scope: "")
                html = fd._doc_html(_doc(**overrides), entry)
            for text in present:
                assert text in html
            for text in absent:
                assert text not in html


def test_the_panel_answers_the_column_it_was_asked_about(qtbot, subtests):
    """Each column shape gets its own sentences, the same from either route in."""
    cases = [
        # column, resolved key, sentences present, sentences absent
        ("cell_channel_0_channel_2_M1_correlation_85", "M1_correlation_<t>",
         ["This column is channels 0 and 2.", "One column per channel PAIR",
          "percentile 85", "fraction in [0, 1]"], []),
        ("object_label", "object_label",
         ["Not a per-object measurement.",
          "never feed these to a model as features", "none (an identifier)"],
         ["This column"]),
        ("cell_skeleton_length", "skeleton_length",
         ["Not written for any object type by a standard run"], []),
        ("organelle_summary_organelle_count",
         "organelle_summary_organelle_count",
         ["NOT in the organelle table", "organelle_summary tables"],
         ["This column"]),
        ("cell_before_filtration", "before_filtration",
         ["pivoted_counts table only"], ["This column"]),
        ("cell_<b>evil</b>_area", None, ["cell_<b>evil</b>_area"], []),
        ("nucleus_channel_2_percentile_25", "percentile_<p>",
         ["nucleus_channel_2_percentile_25", "This column is channel 2."], []),
        ("cell_area", "area", ["cell table"], []),
    ]
    for column, key, present, absent in cases:
        with subtests.test(msg=column):
            asked = _panel(qtbot)
            asked.show_column(column)
            opened = _panel(qtbot, column=column)
            text = asked.detail_text()
            assert _key(asked) == key
            assert _key(opened) == key
            assert opened.detail_text() == text
            for sentence in present:
                assert sentence in text
            for sentence in absent:
                assert sentence not in text


def test_the_detail_pane_drops_the_pin_when_the_question_changes(
        qtbot, monkeypatch, subtests):
    """Another row, a re-typed search or no row at all each move the pin right."""
    with subtests.test(msg="picking another feature unpins the column"):
        panel = _panel(qtbot)
        panel.show_column("nucleus_channel_2_percentile_25")
        assert "This column is channel 2." in panel.detail_text()
        panel._list.setCurrentRow(panel.result_keys().index("std_intensity"))
        text = panel.detail_text()
        assert _key(panel) == "std_intensity"
        assert panel._column is None
        for gone in ("nucleus_channel_2_percentile_25",
                     "This column is channel 2.", "the nucleus table"):
            assert gone not in text
        # Qt clears the current row on repopulation; the pane keeps its answer.
        panel._list.setCurrentRow(-1)
        assert _key(panel) == "std_intensity"
        assert panel.detail_text() == text

    with subtests.test(msg="a row that is the pinned column keeps the pin"):
        panel = _panel(qtbot, column="cell_channel_1_mean_intensity")
        assert panel._column == "cell_channel_1_mean_intensity"
        assert "channel 1" in panel.detail_text()
        other = next(i for i, hit in enumerate(panel._hits)
                     if hit.doc.key != "mean_intensity")
        panel._on_row_changed(other)
        assert panel._column is None
        assert "channel 1" not in panel.detail_text()
        panel.show_column("cell_channel_1_mean_intensity")
        same = next(i for i, hit in enumerate(panel._hits)
                    if hit.doc.key == "mean_intensity")
        panel._on_row_changed(same)
        assert panel._column == "cell_channel_1_mean_intensity"
        assert "channel 1" in panel.detail_text()

    with subtests.test(msg="a row change to no row changes nothing"):
        panel = _panel(qtbot, column="cell_area")
        before = panel.detail_text()
        panel._on_row_changed(-1)
        panel._on_row_changed(len(panel._hits))
        assert _key(panel) == "area"
        assert panel.detail_text() == before

    with subtests.test(msg="searching the pinned column's text is a search"):
        panel = _panel(qtbot)
        panel.show_column("cell_channel_1_percentile_75")
        assert panel._column == "cell_channel_1_percentile_75"
        panel.set_query("cell_channel_1_percentile_75")
        assert panel._column is None
        assert panel._search.text() == "cell_channel_1_percentile_75"
        assert _key(panel) == panel.result_keys()[0]
        assert panel.detail_text().strip()

    with subtests.test(msg="a listed key below the top hit is selected"):
        panel = _panel(qtbot)
        panel.set_query("texture")
        keys = panel.result_keys()
        assert len(keys) >= 3
        panel._select_key(keys[2])
        assert panel._list.currentRow() == 2
        assert _key(panel) == keys[2]

    with subtests.test(msg="a key the search never surfaced is still shown"):
        panel = _panel(qtbot)
        panel.set_query("no feature has this in its name at all")
        assert panel._hits == []
        assert _key(panel) is None
        panel._select_key("area")
        assert _key(panel) == "area"
        before = panel.detail_text()
        panel._select_key("this_is_not_a_spacr_column_at_all_%%%")
        assert _key(panel) == "area"
        assert panel.detail_text() == before

    with subtests.test(msg="a broken search backend still answers the column"):
        panel = _panel(qtbot)

        def boom(*args, **kwargs):
            raise RuntimeError("index unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(fd, "search_features", boom)
            panel.show_column("cell_area")
        assert panel.result_keys() == []
        assert _key(panel) == "area"
        assert "Area" in panel.detail_text()


def test_register_reports_the_app_row_whatever_the_theme_does(
        monkeypatch, qss_sandbox, subtests):
    """``register()`` is True exactly when the app row is there; QSS is extra."""
    from spacr.qt import app as app_mod

    def boom(*args, **kwargs):
        raise RuntimeError("refused")

    cases = [
        # id, which seam refuses, returned, app row present, QSS registered
        ("the registry refuses the app", "app", False, False, True),
        ("the theme refuses the stylesheet", "theme", True, True, False),
    ]
    for case_id, refusing, returned, row, qss in cases:
        with subtests.test(msg=case_id):
            app_mod.unregister_app(fd.APP_KEY)
            qss_sandbox.unregister_widget_qss(fd.OBJECT_NAME)
            with monkeypatch.context() as patch:
                if refusing == "app":
                    patch.setattr(app_mod, "register_app", boom)
                else:
                    patch.setattr(qss_sandbox, "register_widget_qss", boom)
                try:
                    assert fd.register() is returned
                    assert any(r[0] == fd.APP_KEY for r in app_mod.APPS) is row
                    assert (fd.OBJECT_NAME
                            in qss_sandbox.widget_qss_names()) is qss
                finally:
                    app_mod.unregister_app(fd.APP_KEY)


def test_open_feature_dictionary_hands_out_only_the_live_dialog(
        qtbot, qapp, subtests):
    """One dictionary window: a dead dialog is forgotten, a live one never."""
    with subtests.test(msg="the dialog dies with its parent window"):
        window = QMainWindow()
        qtbot.addWidget(window)
        dialog = fd.open_feature_dictionary(window, "cell_area")
        assert dialog.panel.current_doc().key == "area"
        window.close()
        window.deleteLater()
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        assert fd._DIALOG is None
        fresh = fd.open_feature_dictionary(None, "nucleus_channel_0_blur")
        assert fresh is not dialog
        assert fresh.panel.current_doc().key == "blur"
        fd.close_feature_dictionary()

    with subtests.test(msg="an old dialog dying keeps the live one"):
        first = fd.open_feature_dictionary(None, "cell_area")
        fd.close_feature_dictionary()            # posts deleteLater on first
        second = fd.open_feature_dictionary(None, "cell_area")
        assert second is not first
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        try:
            assert fd._DIALOG is second
            assert fd.open_feature_dictionary(None, "cell_area") is second
        finally:
            fd.close_feature_dictionary()

    with subtests.test(msg="forgetting a stale dialog keeps the cached one"):
        dialog = fd.open_feature_dictionary(None, "cell_area")
        stale = fd.FeatureDictionaryDialog()
        qtbot.addWidget(stale)
        fd._forget_dialog(stale)
        assert fd.open_feature_dictionary() is dialog
        fd._forget_dialog(dialog)
        assert fd._DIALOG is None
        assert fd.open_feature_dictionary() is not dialog
