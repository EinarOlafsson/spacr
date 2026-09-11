"""A module screen's build translates each widget once, not three times over.

380 measured one Measure screen being assembled: thirteen deferred passes,
23,454 widget visits, of which THREE near-root passes were 22,750. Each is
triggered when another large container is parented in, an event turn apart,
so the tree is walked roughly three times over and every widget in it had
already been translated once as it was constructed.

The item also named the risk, which is why the guard is here rather than
only the change: "the failure mode of getting it wrong is a widget left in
English in one of nine locales, which is invisible on this machine and to
this test suite". So the first four tests hold the OUTPUT -- the strings the
build renders are the ones a full pass renders -- and only the last one
holds the cost.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt import i18n
from spacr.qt.screens.app_screen import AppScreen


#: Small enough to build several times in one test, real enough to have the
#: nested containers that caused the overlapping passes.
APP = "analyze_plaques"


def _strings(root):
    """Every visible string in the tree, keyed by where the widget sits.

    Keyed by structural path rather than by object, because the comparison
    has to survive a wrapper being collected and recreated between the two
    readings.
    """
    out = {}
    for widget in [root] + root.findChildren(QWidget):
        parts, node = [], widget
        while node is not None:
            parent = node.parent()
            if parent is None:
                parts.append(type(node).__name__)
                break
            try:
                index = list(parent.children()).index(node)
            except ValueError:
                index = -1
            parts.append(f"{type(node).__name__}#{index}")
            node = parent
        path = "/".join(reversed(parts))
        # Live readings, not captions. `UsageBar` is the machine's
        # CPU/RAM/disk and `ActivitySpinner` carries a caption only while it
        # is spinning; both move between two readings for reasons that have
        # nothing to do with language, and comparing them would make this
        # test flap.
        if "UsageBar" in path or "ActivitySpinner" in path:
            continue
        record = {}
        for name in ("text", "toolTip", "windowTitle", "placeholderText",
                     "title", "accessibleName", "accessibleDescription"):
            getter = getattr(widget, name, None)
            if not callable(getter):
                continue
            try:
                value = str(getter() or "")
            except (RuntimeError, TypeError):
                continue
            if value:
                record[name] = value
        if record:
            out[path] = record
    return out


@pytest.fixture()
def visits(monkeypatch):
    """Count one per widget a pass actually translated.

    ``windowTitle`` is the first field of the per-widget body and the only
    one that is never conditional, so counting it counts VISITS rather than
    fields -- a widget that opts out of the tooltip arms would otherwise
    weigh less than one that does not.
    """
    counted = []
    real = i18n._translate_qt_text

    def counting(obj, getter_name, *args, **kwargs):
        if getter_name == "windowTitle":
            counted.append(obj)
        return real(obj, getter_name, *args, **kwargs)

    monkeypatch.setattr(i18n, "_translate_qt_text", counting)
    return counted


def _built(qtbot, qapp, language):
    """A screen assembled exactly as ``MainWindow`` assembles one."""
    screen = AppScreen(APP)
    qtbot.addWidget(screen)
    screen.resize(1200, 800)
    # `MainWindow._on_nav_selected` runs one pass over a screen it has just
    # built; `_LateCaptionTranslator` runs the rest as subtrees arrive.
    i18n.retranslate_widget_tree(screen, language)
    for _ in range(60):
        qapp.processEvents()
    return screen


def test_the_build_renders_what_a_full_pass_renders(qtbot, qapp, monkeypatch):
    """THE ONE THAT MATTERS. Nothing is left in English by the skipping.

    Built with the skip in place, then walked again with it off. A widget
    the deferred passes skipped wrongly shows up here as a string that
    changes on the second walk.
    """
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = _built(qtbot, qapp, "sv")
    after_build = _strings(screen)
    assert after_build, "the screen rendered no strings at all"

    i18n.retranslate_widget_tree(screen, "sv")           # only_new off
    after_full_pass = _strings(screen)

    assert after_build == after_full_pass


def test_a_language_change_after_the_build_reaches_every_widget(
    qtbot, qapp, monkeypatch,
):
    """The stamp carries the language, so a new one cannot match it.

    This is the guard against the cheapest wrong version of the change --
    a boolean "already translated" flag, which would pin the first language
    a process ever rendered.
    """
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = _built(qtbot, qapp, "sv")
    swedish = _strings(screen)

    i18n.retranslate_widget_tree(screen, "ko", only_new=True)
    korean = _strings(screen)

    moved = [path for path in swedish if swedish[path] != korean.get(path)]
    assert moved, "a whole screen rendered identically in Swedish and Korean"


def test_a_row_catalogued_after_a_pass_is_not_missed(qtbot, qapp):
    """`add_translation` moves the generation, which invalidates every stamp.

    Apps register themselves and hand their display name to the catalogs as
    they do it, which happens while screens already exist. A stamp that did
    not move with the catalog would leave that name in English for the life
    of the process.
    """
    widget = QWidget()
    qtbot.addWidget(widget)
    label = QLabel("Analyze plaques", widget)
    i18n.retranslate_widget_tree(widget, "sv")
    stamp = label.property(i18n._PASS_STAMP)
    assert stamp == i18n._pass_stamp("sv")

    source = "a caption catalogued while this screen existed"
    assert i18n.add_translation(source, ["x"] * 9) is True
    try:
        assert i18n._pass_stamp("sv") != stamp
        # And the widget is therefore no longer "already done".
        assert label.property(i18n._PASS_STAMP) != i18n._pass_stamp("sv")
    finally:
        i18n._ROWS.pop(source, None)
        for code in i18n._TRANSLATED_CODES:
            i18n.CATALOGS[code].pop(source, None)


def test_a_strip_that_gains_a_tab_is_not_already_done(qtbot):
    """THE ONE THE FIRST VERSION OF THIS CHANGE GOT WRONG.

    A `QTabWidget` renders its TABS, and a tab can arrive without the strip
    itself changing at all. `_LateCaptionTranslator._pass_root` correctly
    starts the pass at the strip rather than at the page -- "translating the
    page alone leaves that tab in English over a translated page" -- so a
    stamped strip means the second fold a user opens gets an English tab
    over a Swedish page.

    Caught by `test_activation_folds_into_classify::
    test_the_second_fold_opened_is_translated_as_well` and pinned here in
    isolation, because that test is about folds and this is about the rule:
    a widget whose captions are a COLLECTION is never already done. Combo
    items and table and tree headers are the same shape and are exempt with
    it.
    """
    from PySide6.QtWidgets import QTabWidget

    strip = QTabWidget()
    qtbot.addWidget(strip)
    strip.addTab(QWidget(), "Settings")
    i18n.retranslate_widget_tree(strip, "sv", only_new=True)
    first = strip.tabText(0)
    assert first != "Settings", "the catalog has no Swedish for 'Settings'"

    strip.addTab(QWidget(), "Results")
    i18n.retranslate_widget_tree(strip, "sv", only_new=True)
    assert strip.tabText(0) == first
    assert strip.tabText(1) != "Results", (
        "a tab added after the strip was stamped stayed in English")


def test_only_new_is_off_by_default(qtbot):
    """A caller that does not ask for it gets the full pass.

    `_translate_qt_text` detects an outside setter by comparing the value it
    last rendered, and a skipped widget is not compared -- so every caller
    that has just replaced a caption itself has to keep the whole walk.
    """
    widget = QWidget()
    qtbot.addWidget(widget)
    label = QLabel("Settings", widget)
    i18n.retranslate_widget_tree(widget, "sv")

    seen = []
    real = i18n._translate_qt_text
    try:
        i18n._translate_qt_text = lambda o, *a, **k: (
            seen.append(o), real(o, *a, **k))[1]
        i18n.retranslate_widget_tree(widget, "sv")
    finally:
        i18n._translate_qt_text = real
    assert label in seen


def test_the_tree_is_not_walked_three_times_over(qtbot, qapp, visits,
                                                 monkeypatch):
    """THE CEILING. Measured 5.7 visits per widget; this allows 2.

    A ratio rather than a count, because the count moves with the screen and
    the ratio is the defect: 380 measured three near-root passes over the
    same tree, an event turn apart, on top of the one `MainWindow` runs.

    Two rather than one, deliberately. A widget that genuinely arrives after
    a pass is translated by the next one and SHOULD be visited twice; the
    headroom is for that, not for another full walk.
    """
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = _built(qtbot, qapp, "sv")
    widgets = 1 + len(screen.findChildren(QWidget))
    assert widgets > 100, "this screen is too small to say anything"
    assert len(visits) <= widgets * 2, (
        f"{len(visits)} widget visits to translate {widgets} widgets "
        f"({len(visits) / widgets:.1f} passes over the tree)"
    )
