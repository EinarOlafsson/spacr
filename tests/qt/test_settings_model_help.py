"""Table-driven tests of the settings panel's help, labels and small widgets.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the clickable API tooltip filter, refreshing and
installing setting help (no dots are drawn, dead widgets are stepped over),
finding a field's label and its owning layout, the regression backend
selector's event filter, and the record table that edits lists of records.
Offscreen Qt, no network, no modal dialogs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout,  # noqa: E402
                               QLabel, QLineEdit, QSpinBox, QVBoxLayout,
                               QWidget)

from spacr.qt.screens import settings_model as SM  # noqa: E402
from spacr.qt.screens.settings_model import _RecordColumn, _RecordTable  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# _ApiTooltipFilter
# ---------------------------------------------------------------------------

def test_the_api_tooltip_filter_replaces_qts_tooltip_with_the_clickable_one(
        subtests, qtbot, monkeypatch):
    """Enter shows, Leave hides, ToolTip is swallowed -- only where help exists."""
    from spacr.qt.widgets import hover_tooltip

    calls = []

    class _Stub:
        def show_for(self, target, html):
            calls.append(("show", target, html))

        def start_hide(self):
            calls.append(("hide",))

    monkeypatch.setattr(hover_tooltip.HoverTooltip, "instance",
                        classmethod(lambda cls: _Stub()))
    cases = [
        # (id, help html, event, returned, calls)
        ("no help: Enter is left to Qt", None, QEvent.Enter, False, []),
        ("no help: ToolTip is left to Qt", None, QEvent.ToolTip, False, []),
        ("Enter shows the clickable tooltip", "<b>docs</b>", QEvent.Enter, False,
         [("show", "<b>docs</b>")]),
        ("Leave starts hiding it", "<b>docs</b>", QEvent.Leave, False, [("hide",)]),
        ("the native tooltip is suppressed", "<b>docs</b>", QEvent.ToolTip, True, []),
        ("other events pass through", "<b>docs</b>", QEvent.Show, False, []),
    ]
    for case_id, html, kind, returned, expected in cases:
        with subtests.test(msg=case_id):
            widget = QWidget()
            qtbot.addWidget(widget)
            if html is not None:
                widget.setProperty("apiTooltipHtml", html)
            calls.clear()
            assert SM._ApiTooltipFilter(widget).eventFilter(
                widget, QEvent(kind)) is returned
            assert [tuple(c for c in call if c is not widget) for call in calls] \
                == expected


# ---------------------------------------------------------------------------
# refresh_api_tooltips / install_api_tooltips
# ---------------------------------------------------------------------------

class _UrlSink(QWidget):
    """A dot that owns its destination, as the real API dot does."""

    def __init__(self):
        super().__init__()
        self.url = None

    def set_url(self, url):
        self.url = url


def test_refresh_api_tooltips_captions_dots_and_steps_over_dead_widgets(
        subtests, qtbot):
    """Dots get the API caption (and URL); fields get help; dead trees are a no-op."""

    def tagged(widget, owner):
        widget.setParent(owner)
        widget.setProperty("settingsAppKey", "measure")
        widget.setProperty("settingKey", "src")
        return widget

    caption = SM._api_reference_tooltip("src", "en", "measure")
    with subtests.test(msg="a dot is captioned and re-pointed, a field described"):
        owner = QWidget()
        qtbot.addWidget(owner)
        dot = tagged(_UrlSink(), owner)
        dot.setProperty("apiTooltipDisplayRole", "api-link")
        field = tagged(QLineEdit(), owner)
        SM.refresh_api_tooltips(owner, "en")
        assert caption == "API"
        assert dot.toolTip() == dot.accessibleName() == caption
        assert dot.accessibleDescription() == "API"
        assert dot.url == SM.api_docs_url("measure", "src", "en")
        assert field.toolTip() == field.property("apiTooltipHtml") != caption
        assert field.toolTip()

    with subtests.test(msg="a dot with no destination still gets its caption"):
        owner = QWidget()
        qtbot.addWidget(owner)
        dot = tagged(QWidget(), owner)
        dot.setProperty("apiTooltipDisplayRole", "api-link")
        SM.refresh_api_tooltips(owner, "en")
        assert dot.toolTip() == caption
        assert dot.accessibleDescription() == "API"

    with subtests.test(msg="a metadata field stays quiet"):
        owner = QWidget()
        qtbot.addWidget(owner)
        field = tagged(QLineEdit(), owner)
        field.setProperty("apiTooltipDisplayRole", "metadata")
        field.setToolTip("stale")
        SM.refresh_api_tooltips(owner, "en")
        assert field.toolTip() == ""
        assert field.property("apiTooltipHtml")

    class _Gone:
        def findChildren(self, *_args, **_kwargs):
            raise RuntimeError("Internal C++ object already deleted")

    class _DeadChild:
        def property(self, _name):
            raise RuntimeError("Internal C++ object already deleted")

    class _Root:
        def findChildren(self, *_args, **_kwargs):
            return [_DeadChild()]

        def property(self, _name):
            return None

    for case_id, root in (("nothing is a no-op", None),
                          ("a destroyed tree is a no-op", _Gone()),
                          ("a child that died mid-walk is skipped", _Root())):
        with subtests.test(msg=case_id):
            assert SM.refresh_api_tooltips(root) is None


def test_install_api_tooltips_puts_help_on_labels_and_draws_no_dots(
        subtests, qtbot):
    """The label (or a self-labelling box) carries the help; fields go quiet; no dot."""

    with subtests.test(msg="a label beside its field and a check box get the help"):
        owner = QWidget()
        qtbot.addWidget(owner)
        column = QVBoxLayout(owner)
        form_host = QWidget(owner)
        form = QFormLayout(form_host)
        field = QSpinBox(form_host)
        field.setProperty("settingKey", "cell_diameter")
        label = QLabel("Cell diameter", form_host)
        form.addRow(label, field)
        column.addWidget(form_host)
        box = QCheckBox("Verbose", owner)
        box.setProperty("settingKey", "verbose")
        column.addWidget(box)
        SM.install_api_tooltips(owner, "mask")
        assert "href=" in str(label.property("apiTooltipHtml"))
        assert label.toolTip() == label.property("apiTooltipHtml")
        assert (label.property("settingKey"), label.property("settingsAppKey"),
                label.property("settingHelpLabel")) == ("cell_diameter", "mask", True)
        assert "href=" in str(box.property("apiTooltipHtml"))
        assert (field.property("apiTooltipDisplayRole"), field.toolTip()) == (
            "metadata", "")
        assert [w for w in owner.findChildren(QWidget)
                if w.property("apiTooltipDisplayRole") == "api-link"] == []

    with subtests.test(msg="a field with no label is left quiet"):
        owner = QWidget()
        qtbot.addWidget(owner)
        layout = QVBoxLayout(owner)
        field = QWidget(owner)
        field.setProperty("settingKey", "nucleus_channel")
        field.setToolTip("something stale")
        layout.addWidget(field)
        SM.install_api_tooltips(owner, "mask")
        assert (field.toolTip(), field.property("apiTooltipHtml"),
                field.property("apiTooltipDisplayRole")) == ("", "", "metadata")

    with subtests.test(msg="the sweep walks past the dots it made"):
        owner = QWidget()
        qtbot.addWidget(owner)
        QVBoxLayout(owner)
        field = QLineEdit(owner)
        field.setProperty("settingKey", "src")
        dot = QLabel("i", owner)
        dot.setProperty("settingKey", "src")
        dot.setProperty("apiTooltipDisplayRole", "api-link")
        SM.install_api_tooltips(owner, "measure")
        assert field.property("settingsAppKey") == "measure"
        assert dot.property("settingsAppKey") is None
        assert dot.property("apiTooltipDisplayRole") == "api-link"


# ---------------------------------------------------------------------------
# _unwrap_setting_label / _setting_label_for_field / _owning_layout
# ---------------------------------------------------------------------------

def test_a_fields_label_is_found_in_this_window_or_not_at_all(subtests, qtbot):
    """Wrappers are unwrapped; a remembered label is reused only when alive and local."""

    def host(*labels, marked=None):
        wrapper = QWidget()
        qtbot.addWidget(wrapper)
        wrapper.setObjectName("SettingLabelWithInfo")
        QVBoxLayout(wrapper)
        made = [QLabel(text, wrapper) for text in labels]
        if marked is not None:
            made[marked].setProperty("settingHelpLabel", True)
        return wrapper, made

    with subtests.test(msg="unwrapping"):
        marked_host, (spacer, marked) = host("   ", "Source", marked=1)
        caption_host, (blank, caption) = host("   ", "Source")
        empty_host, _ = host()
        plain = QLabel("Source")
        qtbot.addWidget(plain)
        assert SM._unwrap_setting_label(marked_host) is marked
        assert SM._unwrap_setting_label(caption_host) is caption
        assert SM._unwrap_setting_label(empty_host) is empty_host
        assert SM._unwrap_setting_label(plain) is plain
        assert SM._unwrap_setting_label(None) is None

    def owner_with_field():
        owner = QWidget()
        qtbot.addWidget(owner)
        return owner, QLineEdit(owner)

    with subtests.test(msg="a form label is found, an orphan has none"):
        owner = QWidget()
        qtbot.addWidget(owner)
        form = QFormLayout(owner)
        label, field = QLabel("Source"), QLineEdit()
        form.addRow(label, field)
        orphan = QLineEdit()
        qtbot.addWidget(orphan)
        assert SM._setting_label_for_field(owner, field) is label
        assert SM._setting_label_for_field(owner, orphan) is None

    with subtests.test(msg="a remembered label in this window is reused"):
        owner, field = owner_with_field()
        label = QLabel("Source", owner)
        field._spacr_setting_label = label
        assert SM._setting_label_for_field(owner, field) is label

    with subtests.test(msg="a remembered label from another window is not reused"):
        owner, field = owner_with_field()
        other = QWidget()
        qtbot.addWidget(other)
        field._spacr_setting_label = QLabel("Source", other)
        assert SM._setting_label_for_field(owner, field) is None

    for attribute in ("_spacr_setting_label", "settingLabelWidget"):
        with subtests.test(msg=f"a destroyed remembered label ({attribute})"):
            owner, field = owner_with_field()
            dead = QLabel(owner)
            if attribute.startswith("_"):
                setattr(field, attribute, dead)
            else:
                field.setProperty(attribute, dead)
            shiboken6.delete(dead)
            assert SM._setting_label_for_field(owner, field) is None


def test_owning_layout_finds_the_innermost_layout_and_index(subtests, qtbot):
    """Nested layouts are walked; stretches and spacers are stepped over."""
    keep = []  # qtbot holds widgets weakly; the hosts must outlive the case


    def flat():
        host = QWidget()
        qtbot.addWidget(host)
        keep.append(host)
        layout = QHBoxLayout(host)
        layout.addStretch(1)
        field = QLineEdit(host)
        layout.addWidget(field)
        return layout, field, (layout, 1)

    def nested(spacing):
        host = QWidget()
        qtbot.addWidget(host)
        keep.append(host)
        outer = QVBoxLayout(host)
        inner = QHBoxLayout()
        outer.addLayout(inner)
        if spacing:
            inner.addSpacing(8)
        field = QLineEdit(host)
        inner.addWidget(field)
        return outer, field, (inner, 1 if spacing else 0)

    def absent():
        host = QWidget()
        qtbot.addWidget(host)
        keep.append(host)
        layout = QHBoxLayout(host)
        stranger = QLineEdit()
        qtbot.addWidget(stranger)
        keep.append(stranger)
        return layout, stranger, (None, -1)

    cases = [
        ("a stretch before the field is stepped over", flat),
        ("a nested layout is walked", lambda: nested(False)),
        ("a spacer in the nested layout is stepped over", lambda: nested(True)),
        ("a field outside the layout is not found", absent),
    ]
    for case_id, build in cases:
        with subtests.test(msg=case_id):
            root, field, (layout, index) = build()
            found, at = SM._owning_layout(root, field)
            assert found is layout and at == index


# ---------------------------------------------------------------------------
# _RegressionBackendField.eventFilter
# ---------------------------------------------------------------------------

def test_the_backend_field_routes_only_its_own_hover_events(subtests, qtbot):
    """Hover rules fire for its combo and popup; everything else is handed on."""

    def field_with_spies():
        field = SM._RegressionBackendField(regression_type="ols")
        qtbot.addWidget(field)
        spies = {"closed": [], "row": [], "released": []}
        field._hover_closed_combo = lambda *a: spies["closed"].append(True)
        field._hover_popup_row = lambda *a: spies["row"].append(True)
        field._release_popup = lambda *a: spies["released"].append(True)
        return field, spies

    def stranger(field):
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget

    move = QMouseEvent(QEvent.MouseMove, QPointF(-50.0, -50.0),
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    cases = [
        # (id, target, event, expected spy counts)
        ("an event from elsewhere reaches no hover rule", stranger,
         QEvent(QEvent.Enter), {"closed": 0, "row": 0, "released": 0}),
        ("entering the closed combo hovers it", lambda f: f.combo,
         QEvent(QEvent.Enter), {"closed": 1, "row": 0, "released": 0}),
        ("other popup events neither hover nor release",
         lambda f: f.combo.view().viewport(), QEvent(QEvent.Enter),
         {"closed": 0, "row": 0, "released": 0}),
        ("a move over the popup hovers a row",
         lambda f: f.combo.view().viewport(), move,
         {"closed": 0, "row": 1, "released": 0}),
        ("leaving the popup releases it", lambda f: f.combo.view().viewport(),
         QEvent(QEvent.Leave), {"closed": 0, "row": 0, "released": 1}),
    ]
    for case_id, target, event, counts in cases:
        with subtests.test(msg=case_id):
            field, spies = field_with_spies()
            assert field.eventFilter(target(field), event) is False
            assert {k: len(v) for k, v in spies.items()} == counts

    with subtests.test(msg="a pointer over no popup row opens no panel"):
        field = SM._RegressionBackendField(regression_type="ols")
        qtbot.addWidget(field)
        assert field.eventFilter(field.combo.view().viewport(), move) is False

    with subtests.test(msg="a field whose combo is gone hands every event on"):
        field = SM._RegressionBackendField(regression_type="ols")
        qtbot.addWidget(field)
        del field.combo
        assert field.eventFilter(field, QEvent(QEvent.Type.ToolTip)) is False


# ---------------------------------------------------------------------------
# _RecordTable
# ---------------------------------------------------------------------------

COLUMNS = (_RecordColumn("name", "Name"), _RecordColumn("size", "Size", "int"))


def test_the_record_table_reads_every_stored_shape(subtests, qtbot):
    """Rows, read-back value and problem notes for each stored value shape."""

    def make(default=None, mapping_key=None):
        table = _RecordTable(COLUMNS, default, mapping_key=mapping_key)
        qtbot.addWidget(table)
        return table

    cases = [
        # (id, stored, mapping key, rows, value read back, problem fragments)
        ("None shows nothing", None, None, 0, None, []),
        ("an empty string shows nothing", "", None, 0, None, []),
        ("whitespace shows nothing", "   ", None, 0, None, []),
        ("the text None shows nothing", "None", None, 0, None, []),
        ("text that is not a literal is a problem", "[{unclosed", None, 0, None,
         ["is not a list of records"]),
        ("text holding records becomes rows", "[{'name': 'a', 'size': 3}]", None,
         1, [{"name": "a", "size": 3}], []),
        ("a single record is one row", {"name": "a", "size": 1}, None, 1,
         [{"name": "a", "size": 1}], []),
        ("a mapping without a mapping key is one record", {"x": {"size": 2}},
         None, 1, [{"x": {"size": 2}}], []),
        ("a mapping of records is one row per name",
         {"one": {"size": 1}, "two": {"size": 2}}, "name", 2,
         {"one": {"size": 1}, "two": {"size": 2}}, []),
        ("entries that are not records are left out with a note",
         [{"name": "a"}, "oops", 7], None, 1, [{"name": "a"}],
         ["Entry 2 is not a record", "Entry 3 is not a record"]),
        ("a single non-record value is one entry and a note", 5, None, 0, None,
         ["Entry 1 is not a record"]),
    ]
    for case_id, stored, mapping_key, rows, value, problems in cases:
        with subtests.test(msg=case_id):
            table = make(stored, mapping_key)
            assert table.table.rowCount() == rows
            assert table.get_value() == value
            notes = table.problems()
            assert len(notes) == len(problems)
            assert all(fragment in note for fragment, note in zip(problems, notes))
            assert table.status.isHidden() is (not problems)

    with subtests.test(msg="a row without a name is left out of a stored mapping"):
        table = make({"one": {"size": 1}}, mapping_key="name")
        table._on_add()
        table.table.item(1, 1).setText("5")
        assert table.get_value() == {"one": {"size": 1}}

    with subtests.test(msg="removing the current row without a selection"):
        table = make([{"name": "a"}, {"name": "b"}])
        table.table.clearSelection()
        table.table.setCurrentCell(1, 0)
        changes = []
        table.value_changed.connect(lambda: changes.append(True))
        table._on_remove()
        assert table.table.rowCount() == 1
        assert table.get_value() == [{"name": "a"}]
        assert changes == [True]
