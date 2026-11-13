"""What the module screen does when the form it reasons about is not there.

The dimension switches, the empty-state card, the settings advisor, the console
fold and the example-data buttons all ask the settings FORM a question (which
rows exist, which category owns them, what ``src`` holds) that can come back
empty: the form is built after the screen, replaced when a category is
re-rendered, and absent on screens that never grew one. The answer must be the
conservative one -- do not hide the row, do not claim the card, do not lose
the advice -- because the visible failure is a control the user cannot reach.

One regression screen is shared per test; every case patches through its own
``monkeypatch.context()`` so nothing it changes reaches the next case.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPixmap                                # noqa: E402
from PySide6.QtWidgets import (                                  # noqa: E402
    QLabel, QLineEdit, QPushButton, QSplitter, QWidget,
)

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _walk(subtests, monkeypatch, cases, env):
    completed = []
    for name, case in cases.items():
        if getattr(env, "caplog", None) is not None:
            env.caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            env.mp = mp
            case(env)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- dimension switches and whether a row is on screen -----------------------

def _dim_unknown_dimension_not_switched(env):
    env.screen.set_dimension("hyperspectral", True)
    assert "hyperspectral" not in (getattr(env.screen, "_dimension_on", None)
                                   or {})


def _dim_file_names_unknown_dimension(env):
    env.mp.setattr(env.screen, "_dimension_on", {}, raising=False)
    assert env.screen._sync_dimension_switches({"z_stack": True}) == ()


def _dim_prose_category_owns_no_rows(env):
    gated = AppScreen("mask")
    env.qtbot.addWidget(gated)
    assert gated._dimension_rows(), "mask has no dimensional rows to gate"
    prose = QWidget()
    env.qtbot.addWidget(prose)
    env.mp.setattr(gated, "_settings_sections", [prose], raising=False)
    assert gated._dimension_rows() == []
    assert gated._dimension_hidden_sections() == set()


def _dim_form_went_away(env):
    stale = types.SimpleNamespace(_form=None)
    env.mp.setattr(env.screen, "_dimension_rows",
                   lambda: [(stale, "z_stack", object())])
    assert env.screen._dimension_hidden_sections() == set()


def _dim_no_row_helper_leaves_form_alone(env):
    import builtins

    real_import = builtins.__import__

    def refuse_the_helper(name, *args, **kwargs):
        if name.endswith("settings_search"):
            raise ImportError("no row-visibility helper")
        return real_import(name, *args, **kwargs)

    walked = []
    env.mp.setattr(env.screen, "_dimension_rows",
                   lambda: walked.append(1) or [])
    env.mp.setattr(builtins, "__import__", refuse_the_helper)
    env.screen._apply_dimension_visibility()
    assert walked == [], "the form was walked anyway"


def _row_unknown_key_not_visible(env):
    assert env.screen.setting_row_is_visible("no_such_setting") is False


def _row_unmounted_field_not_visible(env):
    orphan = QLineEdit()
    env.qtbot.addWidget(orphan)
    env.mp.setitem(env.screen._settings_model._widgets, "verbose", orphan)
    assert env.screen.setting_row_is_visible("verbose") is False


def _row_prose_category_skipped(env):
    mounted = next(iter(env.screen._settings_model._widgets))
    prose = QWidget()
    env.qtbot.addWidget(prose)
    env.mp.setattr(env.screen, "_settings_sections", [prose], raising=False)
    assert env.screen.setting_row_is_visible(mounted) is False


def test_dimension_gates_and_row_visibility_never_hide_what_they_cannot_see(
        subtests, monkeypatch, qtbot, screen):
    """An unknown dimension moves no switch, a prose card or a re-rendered
    category owns no gated rows, a missing row helper leaves the form untouched,
    and a key that is unknown, unmounted or only in a prose card is reported
    as not visible."""
    cases = {
        "unknown dimension": _dim_unknown_dimension_not_switched,
        "file names unknown dimension": _dim_file_names_unknown_dimension,
        "prose category (mask)": _dim_prose_category_owns_no_rows,
        "form went away": _dim_form_went_away,
        "no row helper": _dim_no_row_helper_leaves_form_alone,
        "row: unknown key": _row_unknown_key_not_visible,
        "row: unmounted field": _row_unmounted_field_not_visible,
        "row: prose category": _row_prose_category_skipped,
    }
    _walk(subtests, monkeypatch, cases,
          types.SimpleNamespace(screen=screen, qtbot=qtbot))


# -- the empty-state card and the console fold -------------------------------

class _OnePaneSplitter:
    """A splitter holding only the console, recording the sizes asked for."""

    def __init__(self, height):
        self._sizes = [height]
        self.asked_for = []

    def indexOf(self, _widget):
        return 0

    def sizes(self):
        return list(self._sizes)

    def setSizes(self, sizes):
        self._sizes = list(sizes)
        self.asked_for.append(list(sizes))


def _card_on_screen(env, src):
    card = QWidget()
    env.qtbot.addWidget(card)
    env.mp.setattr(env.screen, "_empty_state_card", card, raising=False)
    env.mp.setattr(env.screen, "_settings_src_path", lambda: src)
    env.screen._refresh_empty_state()
    return card


def _empty_unreadable_src(env):
    env.mp.setitem(env.screen._settings_model._widgets, "src",
                   types.SimpleNamespace(get_value=_boom,
                                         findChildren=lambda *_args: []))
    assert env.screen._settings_src_path() == ""


def _empty_no_card(env):
    env.mp.setattr(env.screen, "_empty_state_card", None, raising=False)
    env.screen._refresh_empty_state()
    assert env.screen._empty_state_card is None


def _empty_card_back_with_no_src(env):
    card = _card_on_screen(env, "")
    assert card.isVisibleTo(env.screen) or not card.isHidden()


def _empty_placeholder_is_not_data(env):
    card = _card_on_screen(env, "/path/to/src")
    assert not card.isHidden()


def _fold_no_wrapper(env):
    env.mp.setattr(env.screen, "_console_wrap", None, raising=False)
    env.screen._console_folded(True)
    assert env.screen._console_wrap is None


def _fold_in_no_splitter(env):
    wrap = QWidget()
    env.qtbot.addWidget(wrap)
    env.mp.setattr(env.screen, "_console_wrap", wrap, raising=False)
    env.mp.setattr(env.screen, "_console_splitter", None, raising=False)
    env.screen._console_folded(True)
    assert wrap.minimumHeight() == 0


def _fold_left_its_splitter(env):
    splitter = QSplitter()
    env.qtbot.addWidget(splitter)
    wrap = QWidget()
    env.qtbot.addWidget(wrap)
    env.mp.setattr(env.screen, "_console_wrap", wrap, raising=False)
    env.mp.setattr(env.screen, "_console_splitter", splitter, raising=False)
    env.screen._console_folded(False)
    assert wrap.minimumHeight() == 180


def _one_pane(env):
    wrap = QLabel("spaCR output")
    env.qtbot.addWidget(wrap)
    splitter = _OnePaneSplitter(400)
    env.mp.setattr(env.screen, "_console_wrap", wrap, raising=False)
    env.mp.setattr(env.screen, "_console_splitter", splitter, raising=False)
    env.mp.setattr(env.screen, "_console_height", 0, raising=False)
    return splitter, wrap


def _fold_nothing_above(env):
    splitter, wrap = _one_pane(env)
    env.screen._console_folded(True)
    assert env.screen._console_height == 400
    assert wrap.minimumHeight() == 0
    assert splitter.asked_for == [[wrap.sizeHint().height()]]


def _unfold_without_recorded_height(env):
    splitter, wrap = _one_pane(env)
    env.screen._console_folded(False)
    assert splitter.asked_for == [[400]]
    assert wrap.minimumHeight() == 180
    env.screen._console_height = 260
    env.screen._console_folded(False)
    assert splitter.asked_for[-1] == [260]


def test_empty_state_card_and_console_fold_take_the_conservative_answer(
        subtests, monkeypatch, qtbot, screen):
    """An unreadable or placeholder ``src`` is no data (so the card shows), a
    screen without a card or console wrapper does nothing, and the console fold
    always sets its minimum height, keeps the freed height when there is no pane
    above, and gives height back only when a fold recorded one."""
    cases = {
        "unreadable src": _empty_unreadable_src,
        "no card": _empty_no_card,
        "card back with no src": _empty_card_back_with_no_src,
        "placeholder src": _empty_placeholder_is_not_data,
        "fold: no wrapper": _fold_no_wrapper,
        "fold: no splitter": _fold_in_no_splitter,
        "unfold: left its splitter": _fold_left_its_splitter,
        "fold: nothing above": _fold_nothing_above,
        "unfold: no recorded height": _unfold_without_recorded_height,
    }
    _walk(subtests, monkeypatch, cases,
          types.SimpleNamespace(screen=screen, qtbot=qtbot))


# -- the settings advisor ----------------------------------------------------

def _advisor_uncollectable_panel(env):
    env.mp.setattr(env.screen._settings_model, "collect", _boom)
    assert ("No count or score table is attached yet"
            in env.screen.the_advisor_can_run())


def _reading_no_run(env):
    reading = object()
    env.mp.setattr(env.screen, "_last_run_folder", "", raising=False)
    env.mp.setattr(env.screen, "_results_panel", None)
    assert env.screen._reading_with_the_last_run(reading, {}) is reading


def _reading_unreadable_run(env):
    from spacr import settings_advisor

    reading = object()
    env.mp.setattr(env.screen, "_last_run_folder", str(env.tmp_path),
                   raising=False)
    env.mp.setattr(settings_advisor, "read_the_last_run", _boom)
    assert env.screen._reading_with_the_last_run(reading, {}) is reading


def _reading_only_known_fields_folded(env):
    from spacr import settings_advisor
    from spacr.settings_advisor import Reading

    folder = str(env.tmp_path)
    env.mp.setattr(env.screen, "_last_run_folder", folder, raising=False)
    env.mp.setattr(settings_advisor, "read_the_last_run",
                   lambda f, values: {"run_folder": folder,
                                      "a_field_from_a_later_version": 42})
    answer = env.screen._reading_with_the_last_run(Reading(), {})
    assert answer.run_folder == folder
    assert not hasattr(answer, "a_field_from_a_later_version")


def _reading_run_adds_nothing(env):
    from spacr import settings_advisor
    from spacr.settings_advisor import Reading

    reading = Reading()
    env.mp.setattr(env.screen, "_last_run_folder", str(env.tmp_path),
                   raising=False)
    env.mp.setattr(settings_advisor, "read_the_last_run", lambda f, v: {})
    assert env.screen._reading_with_the_last_run(reading, {}) is reading


def _advise(env, reading):
    from spacr import settings_advisor

    said = []
    env.mp.setattr(env.screen, "_console",
                   types.SimpleNamespace(append_stdout=said.append))
    env.mp.setattr(env.screen, "the_advisor_can_run", lambda: "")
    env.mp.setattr(env.screen._settings_model, "collect", lambda: {
        "paired_data": [{"count": "/c.csv", "score": "/s.csv"}],
        "dependent_variable": "pred"})
    env.mp.setattr(settings_advisor, "read_the_screen", lambda *a, **k: reading)
    env.mp.setattr(env.screen, "_reading_with_the_last_run",
                   lambda reading_, values: reading_)
    env.mp.setattr(settings_advisor, "advise_that_runs",
                   lambda *a, **k: types.SimpleNamespace(as_settings=lambda: {}))
    env.screen.settings_for_my_data(answers={})
    return said


def _advice_prints_every_reservation(env):
    from spacr.settings_advisor import Reading

    said = _advise(env, Reading(trouble=["only two plates",
                                         "one guide per gene"]))
    assert any("only two plates" in line for line in said)
    assert any("one guide per gene" in line for line in said)


def _advice_names_why_a_run_was_unread(env):
    from spacr.settings_advisor import Reading

    said = _advise(env, Reading(run_folder="/data/run",
                                run_note="its summary was not written"))
    assert any("its summary was not written" in line for line in said)


def _advice_names_the_run_it_read(env):
    from spacr.settings_advisor import Reading

    said = _advise(env, Reading(run_folder="/data/run"))
    assert any("Also reading the diagnostics of /data/run" in line
               for line in said)


def _maturity_hides_alpha_without_notice(env):
    from spacr.qt import preferences

    visibilities, dimension_passes = [], []
    alpha = types.SimpleNamespace(maturity=lambda: "alpha",
                                  setVisible=visibilities.append)
    env.mp.setattr(env.screen, "rendered_settings_sections", lambda: (alpha,))
    env.mp.setattr(env.screen, "_dimension_hidden_sections", lambda: set())
    env.mp.setattr(env.screen, "_apply_dimension_visibility",
                   lambda: dimension_passes.append(True))
    env.mp.setattr(preferences, "maturity_is_visible",
                   lambda stage: stage != "alpha")
    env.mp.setattr(env.screen, "_maturity_notice", None, raising=False)
    env.screen.refresh_maturity_visibility()
    assert visibilities == [False]
    assert dimension_passes == [True]


def test_the_advisor_explains_rather_than_raises_or_goes_quiet(
        subtests, monkeypatch, qtbot, tmp_path, screen):
    """An uncollectable panel reads as "no tables yet", a last run that is
    absent, unreadable or empty leaves the reading unchanged (and only fields
    the Reading knows are folded in), the advice prints every reservation and
    names the run it read or why it could not; alpha sections hide even with
    no maturity notice to explain it."""
    cases = {
        "uncollectable panel": _advisor_uncollectable_panel,
        "reading: no run": _reading_no_run,
        "reading: unreadable run": _reading_unreadable_run,
        "reading: unknown fields": _reading_only_known_fields_folded,
        "reading: run adds nothing": _reading_run_adds_nothing,
        "advice: reservations": _advice_prints_every_reservation,
        "advice: unread run": _advice_names_why_a_run_was_unread,
        "advice: read run": _advice_names_the_run_it_read,
        "alpha hidden without notice": _maturity_hides_alpha_without_notice,
    }
    _walk(subtests, monkeypatch, cases,
          types.SimpleNamespace(screen=screen, qtbot=qtbot, tmp_path=tmp_path))


# -- the example-data buttons and the gene-tile photograph -------------------

class _Fetched:
    def __init__(self, dataset_path, settings_path):
        self.dataset_path = dataset_path
        self.settings_path = settings_path


def _gene_tile(env, to_pixmap):
    panel = types.SimpleNamespace(gene=types.SimpleNamespace(
        feature=lambda: "cell_area", to_pixmap=to_pixmap))
    assert env.screen._gene_tile_entry(panel) == []


def _gene_tile_null_pixmap(env):
    _gene_tile(env, lambda: QPixmap())


def _gene_tile_refuses(env):
    _gene_tile(env, _boom)


def _progress_button(env):
    button = QPushButton("Load the example screen…")
    env.qtbot.addWidget(button)
    env.mp.setattr(env.screen, "_example_data_button", button, raising=False)
    return button


def _progress_lands_on_button(env):
    button = _progress_button(env)
    env.screen._say_the_download_is_moving("counts.csv", 25, 100)
    assert button.text() == "counts.csv — 25%"


def _progress_unknown_size(env):
    button = _progress_button(env)
    env.screen._say_the_download_is_moving("counts.csv", 25, 0)
    assert button.text() == "Load the example screen…"


def _progress_no_button(env):
    env.mp.setattr(env.screen, "_example_data_button", None, raising=False)
    env.screen._say_the_download_is_moving("counts.csv", 25, 100)
    assert env.screen._example_data_button is None


def _images_dest(env):
    folder = env.tmp_path / env.name.replace(" ", "_").replace(":", "")
    folder.mkdir()
    env.mp.setattr(env.screen, "example_images_destination", lambda: folder)
    env.mp.setattr(env.screen, "_example_images_button", None, raising=False)
    return folder


def _images_already_on_disk(env):
    folder = _images_dest(env)
    (folder / "field.tif").write_bytes(b"")
    asked = []
    placed = env.screen.load_the_example_images(ask=lambda *a: asked.append(a))
    assert asked == []
    assert placed["src"] == str(folder)
    assert placed["settings"] == ""


def _images_download_fails(env):
    _images_dest(env)
    button = QPushButton("Load the example images…")
    env.qtbot.addWidget(button)
    env.mp.setattr(env.screen, "_example_images_button", button, raising=False)

    def download(_screen, _destination, done):
        assert not button.isEnabled()
        assert "Fetching" in button.text()
        done(None, "no route to huggingface.co")

    assert env.screen.load_the_example_images(ask=download) == {}
    assert button.isEnabled()
    # Restored to the screen's own caption, whatever it is today.
    assert "Fetching" not in button.text()
    assert button.text().strip()
    assert "no route to huggingface.co" in _console_text(env.screen._console)


def _images_download_fills_src(env):
    folder = _images_dest(env)
    images = folder / "toxo_mito"
    images.mkdir()
    settings = folder / "toxo_mito.csv"
    settings.write_text("k,v\n", encoding="utf-8")
    src_field = env.screen._settings_model._widgets["src"]
    before = src_field.text()

    def download(_screen, _destination, done):
        done(_Fetched(images, settings), None)

    try:
        placed = env.screen.load_the_example_images(ask=download)
        assert placed == {"src": str(images), "settings": str(settings)}
        assert src_field.text() == str(images)
        text = _console_text(env.screen._console)
        assert f"Source directory (src): {images}" in text
        assert f"Compatible example settings: {settings}" in text
    finally:
        src_field.setText(before)


def _images_real_downloader_reached(env):
    from spacr.qt import hf_download

    folder = _images_dest(env)
    called = []
    env.mp.setattr(
        hf_download, "download_toxo_mito_demo",
        lambda screen_, destination, done, worker_factory=None:
            called.append((destination, worker_factory)))
    assert env.screen.load_the_example_images() == {}
    assert len(called) == 1
    destination, worker = called[0]
    assert destination == str(folder)
    assert worker is hf_download._MaskTarWorker, (
        "the mask demo must ask for the archive, not fetch 212 files")


def _images_no_src_control(env):
    folder = _images_dest(env)
    env.mp.setattr(env.screen, "_settings_model",
                   types.SimpleNamespace(_widgets={}))
    placed = env.screen._put_the_example_images_in_place(folder, None)
    assert placed == {"src": str(folder), "settings": ""}
    assert f"Source directory (src): {folder}" in _console_text(
        env.screen._console)


def _example_screen_without_button(env):
    from spacr import example_data

    got = types.SimpleNamespace(counts=["p1_counts.csv"], scores=["p1_dv.csv"],
                                folder=str(env.tmp_path), note=lambda: "Cached.")
    env.mp.setattr(example_data, "missing", lambda folder=None, kind=None: [])
    env.mp.setattr(example_data, "fetch", lambda **_kwargs: got)
    env.mp.setattr(env.screen, "_example_data_button", None, raising=False)
    added = []
    env.mp.setitem(env.screen._settings_model._widgets, "paired_data",
                   types.SimpleNamespace(
                       findChildren=lambda *_args: [],
                       add_paths_for_side=lambda paths, side: (
                           added.append((tuple(paths), side)) or len(paths))))
    answer = env.screen.load_the_example_screen(download=False)
    assert answer["applied"] == 2
    assert added == [(("p1_dv.csv",), "score"), (("p1_counts.csv",), "count")]


def test_example_data_and_tile_photographs_report_what_happened(
        subtests, monkeypatch, qtbot, tmp_path, screen, console_info_enabled):
    """A gene tile that photographs to nothing is not offered; download progress
    lands on the button only for a known total; example images already on disk
    are reused, a failed download restores the button and says why, a finished
    one fills ``src`` and names the settings, the button reaches the real TAR
    downloader, and a screen without ``src`` or a fetch button still reports
    and pairs the tables."""
    cases = {
        "gene tile null pixmap": _gene_tile_null_pixmap,
        "gene tile refuses": _gene_tile_refuses,
        "progress lands on button": _progress_lands_on_button,
        "progress unknown size": _progress_unknown_size,
        "progress no button": _progress_no_button,
        "images already on disk": _images_already_on_disk,
        "images download fails": _images_download_fails,
        "images download fills src": _images_download_fills_src,
        "images real downloader": _images_real_downloader_reached,
        "images no src control": _images_no_src_control,
        "example screen without button": _example_screen_without_button,
    }
    completed = []
    for name, case in cases.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(types.SimpleNamespace(screen=screen, qtbot=qtbot, mp=mp,
                                       tmp_path=tmp_path, name=name))
            completed.append(name)
    assert completed == list(cases), "a case did not complete"
