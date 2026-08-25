"""The dictionary's honest answers, and its seatbelts around a dying window.

Two groups. The renderer must not invent a definition it does not have and
must not print a row with nothing in it -- an empty labelled row reads as
"this feature has no unit", which is a claim rather than a silence. And the
application-wide event filter must survive being handed an object that is
already being destroyed: it runs for every event in the process, so an
exception there is not a lost menu, it is a lost window.
"""
from __future__ import annotations

import builtins
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QTimer  # noqa: E402
from PySide6.QtWidgets import QMenu  # noqa: E402

from spacr.feature_dict import FeatureDoc  # noqa: E402
from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus instead of opening one, and leave nothing behind.

    Autouse for the same reason the Qt suite's copy is: the default runner is
    ``QMenu.exec``, which in a headless run has nobody to click it. The one
    test that drives the real runner installs its own way out first.
    """
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


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


# ---------------------------------------------------------------------------
# what the detail pane refuses to say
# ---------------------------------------------------------------------------

def test_a_feature_with_no_curated_definition_says_so(qapp):
    """No description must read as "not written down", not as a blank pane.

    A blank space under the heading is indistinguishable from a rendering
    failure, and the note further down is where the missing definition is
    actually explained.
    """
    html = fd._doc_html(_doc(description=None))

    assert "No definition" in html
    assert "see the note below" in html


def test_a_feature_that_has_a_definition_shows_it_instead(qapp):
    """The other half of the same branch, so neither can swallow the other."""
    html = fd._doc_html(_doc(description="How many pixels the object covers."))

    assert "How many pixels the object covers." in html
    assert "No definition" not in html


def test_a_row_with_nothing_to_say_is_not_rendered_at_all(qapp, monkeypatch):
    """An empty value must drop its whole row, label included.

    A ``<b>Channel</b>`` beside an empty cell asserts that the answer is
    "nothing", which for a channel scope means something quite specific. A row
    that is simply absent asserts nothing.
    """
    monkeypatch.setattr(fd, "_channel_sentence", lambda scope: "")

    html = fd._doc_html(_doc())

    assert "Channel" not in html
    # The rows that do have content are unaffected.
    assert "Unit" in html and "Module" in html


# ---------------------------------------------------------------------------
# re-asking the question already in the box
# ---------------------------------------------------------------------------

def test_searching_for_the_pinned_column_again_unpins_the_detail_pane(qtbot):
    """Typing the column name back in is a search, not the column lookup.

    ``show_column`` pins the pane to one concrete column -- its object type,
    its channel, its resolved unit. Asking the same text as a SEARCH is a
    different question, and leaving the pin in place would answer it with the
    column while the result list underneath had been replaced.
    """
    panel = fd.FeatureDictionaryPanel()
    qtbot.addWidget(panel)
    panel.show_column("cell_channel_1_percentile_75")
    assert panel._column == "cell_channel_1_percentile_75"

    # The same text the box already holds, so nothing signals a change.
    panel.set_query("cell_channel_1_percentile_75")

    assert panel._column is None
    assert panel._search.text() == "cell_channel_1_percentile_75"
    assert panel._detail.toPlainText().strip()


# ---------------------------------------------------------------------------
# is this a measurements table at all
# ---------------------------------------------------------------------------

def test_a_view_with_no_model_is_not_a_measurements_table(qapp):
    """A view whose model is gone must answer no, not raise.

    The filter reaches this while deciding whether an unrecognised column is
    worth a menu item; the view it was handed can be mid-teardown.
    """
    assert fd._table_looks_measured(None) is False


# ---------------------------------------------------------------------------
# the default menu runner
# ---------------------------------------------------------------------------

def test_the_default_runner_opens_a_live_menu_that_can_be_clicked(qapp):
    """The shipped runner has to show a real, interactive menu.

    Every other test in the suite replaces it with a recorder, so nothing
    otherwise proves the production path opens a menu whose action can fire.
    The scheduled trigger stands in for the user's click.
    """
    menu = QMenu()
    action = menu.addAction(fd.CONTEXT_ACTION_TEXT)
    fired: list = []
    action.triggered.connect(lambda checked=False: fired.append(True))

    def click_it():
        action.trigger()
        menu.close()

    QTimer.singleShot(0, click_it)
    fd._default_menu_runner(menu, QPoint(10, 10))

    assert fired == [True]
    assert menu.isVisible() is False


# ---------------------------------------------------------------------------
# liveness checks around a wrapper that is going away
# ---------------------------------------------------------------------------

def test_without_shiboken_every_object_is_treated_as_live(monkeypatch):
    """No shiboken means no way to ask, and refusing everything loses the feature.

    ``_still_alive`` gates the whole context-menu filter. Answering "dead" for
    a perfectly good widget would silently switch the feature off on any
    install where the import is unavailable.
    """
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "shiboken6" or name.startswith("shiboken6."):
            raise ImportError("No module named 'shiboken6'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert fd._still_alive(object()) is True
    # None is still nothing, with or without shiboken.
    assert fd._still_alive(None) is False


def test_a_liveness_check_that_itself_fails_treats_the_object_as_live(monkeypatch):
    """``isValid`` raising is not evidence the object is dead.

    Same rule as the missing import: the check is a seatbelt, and a seatbelt
    that fastens itself shut is worse than none.
    """
    import shiboken6

    def explode(_wrapped):
        raise RuntimeError("shiboken is unhappy")

    monkeypatch.setattr(shiboken6, "isValid", explode)

    assert fd._still_alive(object()) is True


def test_an_event_that_dies_between_the_two_checks_is_ignored(qapp):
    """The wrapper passed the liveness check and died before it was read.

    This is the race the filter's seatbelt exists for: it runs for every event
    in the process, including ones whose C++ half is being freed as it looks.
    Reading ``type()`` on that is the segfault; catching it is a no-op.
    """
    class _DyingEvent:
        def type(self):
            raise RuntimeError("Internal C++ object already deleted.")

    filt = fd.FeatureHelpFilter()

    assert filt.eventFilter(object(), _DyingEvent()) is False


def test_a_dead_object_is_refused_before_its_event_is_read(qapp):
    """``None`` for either half is refused without touching the other."""
    filt = fd.FeatureHelpFilter()

    assert filt.eventFilter(object(), None) is False
    assert filt.eventFilter(None, QEvent(QEvent.Type.ContextMenu)) is False


# ---------------------------------------------------------------------------
# installing the filter before there is an application
# ---------------------------------------------------------------------------

def test_installing_the_filter_with_no_application_installs_nothing(monkeypatch):
    """Called before ``QApplication`` exists, it must report that, not crash.

    The hooks are installed from module-level wiring that can run at import
    time, so "there is no app yet" is a real state and returning ``None`` is
    how the caller finds out nothing was installed.
    """
    monkeypatch.setattr(fd, "QApplication",
                        types.SimpleNamespace(instance=lambda: None))
    monkeypatch.setattr(fd, "_FILTER", None)

    assert fd.install_context_menu_filter() is None
    assert fd._FILTER is None
