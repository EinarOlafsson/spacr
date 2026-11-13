"""Table-driven tests of the settings panel model ``SettingsWidgets``.

One general test per behaviour of the panel, each walking a case table with
pytest's ``subtests`` fixture: which control each default gets, writing a
value into each kind of control, reporting modified settings, which organelle
slots a saved file keeps, hiding the headings of absent slots, building the
sections (and breathing between widgets while doing it), the greying rules
that must never grey on a guess, and the object-visibility pass.  Offscreen
Qt, no network, no modal dialogs.
"""
from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEventLoop  # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QLineEdit, QSpinBox, QWidget)

from spacr import plugins  # noqa: E402
from spacr.qt import screens  # noqa: E402
from spacr.qt.screens import settings_model as SM  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    """Build a ``SettingsWidgets`` owned by a widget qtbot cleans up."""
    def build(app_key):
        owner = QWidget()
        qtbot.addWidget(owner)
        widgets = SM.SettingsWidgets(app_key, parent=owner)
        widgets._owner_under_test = owner
        return widgets
    return build


# ---------------------------------------------------------------------------
# _widget_for
# ---------------------------------------------------------------------------

def test_widget_for_builds_the_control_each_default_needs(subtests, qtbot,
                                                         monkeypatch):
    """Each default shape gets its control holding that default, or nothing."""

    def items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    cases = [
        # (id, kind, options, default, key, check(widget))
        ("a bool default gets a toggle already set", "entry", None, True,
         "_probe_bool", lambda w: isinstance(w, QCheckBox) and w.isChecked()),
        ("a false bool default gets a toggle left clear", "entry", None, False,
         "_probe_bool", lambda w: isinstance(w, QCheckBox) and not w.isChecked()),
        ("a list default gets a list editor holding it", "entry", None,
         ["a", "b"], "_probe_list", lambda w: w.get_value() == ["a", "b"]),
        ("a list default for a scalar setting is still a list", "entry", None,
         ["a", "b"], "src", lambda w: w.get_value() == ["a", "b"]),
        ("a combo offers an absent default first", "combo", ["a", "b"], "c",
         "_probe_named", lambda w: items(w) == ["c", "a", "b"]
         and w.currentText() == "c"),
        ("a combo with no default offers no empty choice", "combo", ["a", "b"],
         None, "_probe_blank", lambda w: items(w) == ["a", "b"]),
        ("a kind nobody declared builds nothing", "a_kind_that_does_not_exist",
         None, 1, "x", lambda w: w is None),
    ]
    for case_id, kind, options, default, key, check in cases:
        with subtests.test(msg=case_id):
            widgets = SM.SettingsWidgets("measure")
            if key == "src":
                widgets._defaults["src"] = default
            widget = widgets._widget_for(kind, options, default, key)
            if widget is not None:
                qtbot.addWidget(widget)
            assert check(widget)

    with subtests.test(msg="the class editor is handed the live preview frame"):
        widgets = SM.SettingsWidgets("classify_merged")
        seen = []
        widgets._preview_frame = object()

        class _Editor:
            def __init__(self, **_kwargs):
                pass

            def set_frame(self, frame):
                seen.append(frame)

        # Patched in the globals the method was defined against, which is
        # what it reads even if the module was re-imported in a long run.
        with monkeypatch.context() as mp:
            mp.setitem(type(widgets)._widget_for.__globals__,
                       "ClassEditorWidget", _Editor)
            assert isinstance(widgets._widget_for("entry", None, [], "classes"),
                              _Editor)
        assert seen == [widgets._preview_frame]


# ---------------------------------------------------------------------------
# set_value_for_key
# ---------------------------------------------------------------------------

def test_set_value_for_key_writes_what_each_control_takes(subtests, qtbot):
    """True and the control updated, or False and the control untouched."""

    def combo(editable=False):
        c = QComboBox()
        c.addItem("Alpha", "alpha")
        c.addItem("Beta", "beta")
        c.setEditable(editable)
        return c

    def auto_box():
        box = QDoubleSpinBox()
        box.setRange(0.0, 10.0)
        box.setSpecialValueText(SM.AUTO_TEXT)
        box.setValue(4.0)
        return box

    cases = [
        # (id, app, control, value, returned, read back)
        ("no control is bound", "measure", None, 1, False, None),
        ("a widget type nobody handles", "measure", QWidget, 1, False, None),
        ("a spin box refuses text", "measure", QSpinBox, "not an integer", False,
         lambda w: w.value() == 0),
        ("a spin box takes an int", "measure", QSpinBox, "7", True,
         lambda w: w.value() == 7),
        ("a check box takes truthiness", "measure", QCheckBox, 1, True,
         lambda w: w.isChecked()),
        ("a combo takes item data", "measure", combo, "beta", True,
         lambda w: w.currentText() == "Beta"),
        ("a combo takes visible text", "measure", combo, "Beta", True,
         lambda w: w.currentText() == "Beta"),
        ("an editable combo keeps an unheard-of value", "measure",
         lambda: combo(True), "something else entirely", True,
         lambda w: w.currentText() == "something else entirely"),
        ("a line edit takes None as empty text", "measure",
         lambda: QLineEdit("old"), None, True, lambda w: w.text() == ""),
        ("a line edit takes text", "measure", QLineEdit, 12, True,
         lambda w: w.text() == "12"),
        ("an auto-capable box takes auto as its minimum", "regression", auto_box,
         "auto", True, lambda w: w.value() == w.minimum() == 0.0),
    ]
    for case_id, app, make, value, returned, read_back in cases:
        with subtests.test(msg=case_id):
            widgets = SM.SettingsWidgets(app)
            key = "nms_threshold" if app == "regression" else "_probe"
            if make is not None:
                control = make()
                qtbot.addWidget(control)
                widgets._widgets[key] = control
            assert widgets.set_value_for_key(key, value) is returned
            if read_back is not None:
                assert read_back(control)


# ---------------------------------------------------------------------------
# modified_keys
# ---------------------------------------------------------------------------

def test_modified_keys_reports_readable_controls_that_differ_from_a_default(
        subtests, qtbot):
    """Controls without a default, or that cannot be read, are never reported."""

    class _Unreadable(QLineEdit):
        def text(self):
            raise RuntimeError("Internal C++ object already deleted")

    cases = [
        # (id, {key: (default or MISSING, control)}, expected)
        ("a control without a default is never modified",
         {"_not_a_default": (KeyError, QLineEdit("x"))}, []),
        ("a control that raises on read is skipped",
         {"_probe": ("x", _Unreadable())}, []),
        ("a control read as nothing differs from its default",
         {"_probe": ("x", QWidget())}, ["_probe"]),
        ("an edited control is reported and an untouched one is not",
         {"_edited": ("x", QLineEdit("y")), "_same": ("x", QLineEdit("x"))},
         ["_edited"]),
    ]
    for case_id, controls, expected in cases:
        with subtests.test(msg=case_id):
            widgets = SM.SettingsWidgets("measure")
            for key, (default, control) in controls.items():
                qtbot.addWidget(control)
                widgets._widgets[key] = control
                if default is not KeyError:
                    widgets._defaults[key] = default
            assert widgets.modified_keys() == expected


# ---------------------------------------------------------------------------
# _organelle_slots_worth_keeping
# ---------------------------------------------------------------------------

class _Unequal:
    """A value that cannot be compared, as a NumPy array cannot."""

    def __eq__(self, other):
        raise ValueError("truth value is ambiguous")

    __hash__ = None


def test_organelle_slots_above_the_count_are_kept_only_when_they_say_something(
        subtests, monkeypatch):
    """Slots the panel invented are dropped; anything the run said is kept, contiguously."""
    from spacr.qt import settings_diff

    base = {"number_of_organelles": 1, "organelle_channel": 0}
    unequal = _Unequal()
    cases = [
        # (id, invented slots, settings, expected kept settings)
        ("an invented value above the count is dropped",
         {"organelleb_channel": None}, dict(base, organelleb_channel=None), base),
        ("slot three's own answer keeps slots two and three",
         {"organelleb_channel": None, "organellec_channel": None},
         dict(base, organelleb_channel=None, organellec_channel=2),
         dict(base, organelleb_channel=None, organellec_channel=2)),
        ("a slot the panel never invented is written back",
         {"organelleb_channel": None},
         dict(base, organelleb_channel=None, organellec_channel=3),
         dict(base, organelleb_channel=None, organellec_channel=3)),
        ("a value that cannot be compared is kept",
         {"organelleb_channel": _Unequal()}, dict(base, organelleb_channel=unequal),
         dict(base, organelleb_channel=unequal)),
        ("slots within the count are kept whatever they hold",
         {"organelleb_channel": None},
         dict(base, number_of_organelles=2, organelleb_channel=None),
         dict(base, number_of_organelles=2, organelleb_channel=None)),
        ("nothing invented returns the settings as they are", {},
         dict(base, organellec_channel=None),
         dict(base, organellec_channel=None)),
    ]
    for case_id, invented, settings, expected in cases:
        with subtests.test(msg=case_id):
            widgets = SM.SettingsWidgets("measure")
            widgets._slots_the_panel_added = invented
            assert widgets._organelle_slots_worth_keeping(settings) == expected

    with subtests.test(msg="a comparison that blows up keeps the slot"):
        def explode(_a, _b):
            raise ValueError("the truth value of an array is ambiguous")

        widgets = SM.SettingsWidgets("measure")
        widgets._slots_the_panel_added = {"organelleb_channel": 1}
        with monkeypatch.context() as mp:
            mp.setattr(settings_diff, "_values_equal", explode)
            kept = widgets._organelle_slots_worth_keeping(
                dict(base, organelleb_channel=1))
        assert kept == dict(base, organelleb_channel=1)


# ---------------------------------------------------------------------------
# _hide_the_headings_of_slots_the_run_lacks
# ---------------------------------------------------------------------------

class _Heading(QWidget):
    """The part of ``Section`` the slot-heading rule reads."""

    def __init__(self, parent, stage="stable"):
        super().__init__(parent)
        self._stage = stage

    def maturity(self):                                  # noqa: D102
        return self._stage


def test_the_headings_of_absent_slots_follow_the_organelle_count(
        subtests, qtbot, monkeypatch):
    """A slot's heading hides when the count drops below it and returns when it rises."""
    from spacr.qt import preferences

    def model(stage="stable"):
        screen = QWidget()
        qtbot.addWidget(screen)
        section = _Heading(screen, stage)
        widgets = SM.SettingsWidgets("measure", parent=screen)
        widgets._screen_under_test = screen
        widgets.remember_section_rows(section, ("organelleb_channel",), False)
        return widgets, section

    with subtests.test(msg="hidden once, recorded once, shown again"):
        widgets, section = model()
        assert not section.isHidden()
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 1})
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 1})
        assert section.isHidden()
        assert list(widgets._headings_of_absent_slots) == [id(section)]
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 2})
        assert not section.isHidden()
        assert widgets._headings_of_absent_slots == {}

    with subtests.test(msg="an alpha heading stays hidden when alpha is off"):
        widgets, section = model(stage="alpha")
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 1})
        assert section.isHidden()
        with monkeypatch.context() as mp:
            mp.setattr(preferences, "get_show_alpha", lambda: False)
            widgets._hide_the_headings_of_slots_the_run_lacks(
                {"number_of_organelles": 2})
        assert section.isHidden()
        assert widgets._headings_of_absent_slots == {}

    with subtests.test(msg="a heading destroyed with its screen is forgotten"):
        widgets, section = model()
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 1})
        ident = id(section)
        assert list(widgets._headings_of_absent_slots) == [ident]
        shiboken6.delete(section)
        widgets._hide_the_headings_of_slots_the_run_lacks({"number_of_organelles": 1})
        assert widgets._headings_of_absent_slots == {}


# ---------------------------------------------------------------------------
# build_sections
# ---------------------------------------------------------------------------

def test_build_sections_lays_out_every_setting_it_built(subtests, panel,
                                                       monkeypatch):
    """Every built control is on a section; declined ones are left out; nothing raises."""

    def rendered(sections):
        return {id(widget) for _title, rows in sections for _label, widget in rows}

    with subtests.test(msg="a setting the factory declines is left out"):
        widgets = panel("measure")
        real = widgets._widget_for
        declined = []

        def decline(kind, options, default, key):
            if key == "src":
                declined.append(key)
                return None
            return real(kind, options, default, key)

        widgets._widget_for = decline
        sections = widgets.build_sections()
        assert declined == ["src"]
        assert "src" not in widgets._widgets
        assert widgets._widgets, "everything else was still built"
        assert all(id(w) in rendered(sections) for w in widgets._widgets.values())

    with subtests.test(msg="a plugin registry that raises still builds the panel"):
        with monkeypatch.context() as mp:
            mp.setattr(plugins, "get_app", lambda _key: None)
            expected = [title for title, _rows in panel("measure").build_sections()]

            def boom(_key):
                raise RuntimeError("plugin registry is on fire")

            mp.setattr(plugins, "get_app", boom)
            got = [title for title, _rows in panel("measure").build_sections()]
        assert got == expected and got

    with subtests.test(msg="classify_merged: family first, nothing lost, no stutter"):
        widgets = panel("classify_merged")
        sections = widgets.build_sections()
        titles = [str(title) for title, _rows in sections]
        assert str(getattr(sections[0], "title", "")) == titles[0] == "Classifier"
        shown = rendered(sections)
        assert sorted(k for k, w in widgets._widgets.items() if id(w) not in shown) == []
        assert [t for t in titles
                if t.startswith("Machine Learning") and "ML Classifier" in t] == []


def test_build_sections_breathes_between_widgets_on_a_budget(
        subtests, panel, monkeypatch):
    """The event loop runs every ~40 ms of widget work, never during a fast build."""

    def build(widgets, defaults, clock, built, pumps, *, opening, last_breath,
              cost, pump_cost):
        widgets._defaults = defaults
        original = widgets._widget_for

        def widget_for(kind, options, default, key):
            built.append(key)
            clock[0] += cost
            return original(kind, options, default, key)

        def process_events(mode):
            pumps.append((len(built), clock[0], mode))
            clock[0] += pump_cost

        with monkeypatch.context() as mp:
            mp.setattr(widgets, "_widget_for", widget_for)
            mp.setattr(time, "perf_counter", lambda: clock[0] / 1000)
            mp.setattr(QCoreApplication, "processEvents", process_events)
            mp.setattr(screens, "_a_window_is_opening_a_screen", opening)
            mp.setattr(screens, "_last_breath_at", last_breath)
            widgets.build_sections()
            return screens._last_breath_at

    with subtests.test(msg="slow widgets leave real work between breaths"):
        widgets = panel("measure")
        defaults = {f"owned_breath_{index}": str(index) for index in range(64)}
        built, pumps = [], []
        build(widgets, defaults, [0], built, pumps, opening=False,
              last_breath=0.0, cost=1, pump_cost=40)
        assert built == list(defaults) == list(widgets._widgets)
        counts = [count for count, _t, _mode in pumps]
        assert len(counts) == 2, pumps
        assert counts[0] >= 25 and counts[1] - counts[0] >= 25, pumps
        assert {mode for _c, _t, mode in pumps} == {QEventLoop.ExcludeUserInputEvents}

    with subtests.test(msg="inside a window open a fast panel breathes once"):
        widgets = panel("measure")
        defaults = {f"fast_breath_{index}": str(index) for index in range(10)}
        built, pumps = [], []
        breathed_at = build(widgets, defaults, [1000], built, pumps, opening=True,
                            last_breath=0.960, cost=1, pump_cost=0)
        assert list(widgets._widgets) == list(defaults)
        assert [(c, t) for c, t, _m in pumps] == [(0, 1000)]
        assert breathed_at == 1.0

    with subtests.test(msg="outside a window open a fast panel never breathes"):
        widgets = panel("measure")
        defaults = {f"fast_breath_{index}": str(index) for index in range(10)}
        built, pumps = [], []
        build(widgets, defaults, [1000], built, pumps, opening=False,
              last_breath=0.0, cost=1, pump_cost=0)
        assert pumps == []


# ---------------------------------------------------------------------------
# the greying rules
# ---------------------------------------------------------------------------

def test_greying_rules_never_grey_a_control_on_a_guess(subtests, panel, qtbot,
                                                      monkeypatch):
    """A rule that cannot decide leaves controls editable and raises nothing."""
    import spacr.classify as families
    import spacr.settings as settings

    def fail(message):
        def _raise(*_a, **_k):
            raise ValueError(message)
        return _raise

    with subtests.test(msg="dependency rules are empty when settings refuses"):
        widgets = panel("measure")
        widgets.build_sections()
        with monkeypatch.context() as mp:
            mp.setattr(settings, "get_setting_dependencies", fail("no rules today"))
            assert widgets._rules_for_this_panel() == {}

    with subtests.test(msg="a predicate that raises leaves its control enabled"):
        widgets = panel("measure")
        widgets.build_sections()
        key = next(iter(widgets._widgets))
        control = widgets._widgets[key]
        control.setEnabled(False)
        with monkeypatch.context() as mp:
            mp.setattr(widgets, "_rules_for_this_panel", lambda: {
                key: {"predicate": fail("x"), "sources": ()}})
            widgets._refresh_setting_dependencies()
        assert control.isEnabled() is True

    with subtests.test(msg="an unknown reducer method greys nothing"):
        widgets = panel("umap")
        widgets.build_sections()
        selector = widgets._widgets["reduction_method"]
        try:
            before = {k: w.isEnabled() for k, w in widgets._widgets.items()}
            selector.setEditable(True)
            selector.setEditText("a_method_that_does_not_exist")
            assert widgets._refresh_umap_reducer_enablement() is None
            assert {k: w.isEnabled() for k, w in widgets._widgets.items()} == before
        finally:
            editor = widgets._widgets.get("exclude_rows")
            if editor is not None:
                editor.shutdown()

    with subtests.test(msg="an unresolvable classifier family greys nothing"):
        widgets = panel("classify_merged")
        widgets.build_sections()
        assert "classifier_family" in widgets._widgets
        for control in widgets._widgets.values():
            control.setEnabled(True)
        with monkeypatch.context() as mp:
            mp.setattr(families, "resolve_family", fail("unknown family"))
            assert widgets._refresh_classifier_family_enablement() is None
        assert all(c.isEnabled() for c in widgets._widgets.values())

    with subtests.test(msg="changing the classifier family re-runs the greying"):
        widgets = panel("classify_merged")
        widgets.build_sections()
        calls = []
        with monkeypatch.context() as mp:
            mp.setattr(widgets, "_refresh_classifier_family_enablement",
                       lambda: calls.append(True))
            widgets._on_classifier_family_changed()
        assert calls == [True]

    def unit_panel():
        widgets = SM.SettingsWidgets("regression")
        unit = QComboBox()
        qtbot.addWidget(unit)
        unit.addItem("cell")
        widgets._widgets["analysis_unit"] = unit
        return widgets, unit

    with subtests.test(msg="no constraint registry leaves the unit editable"):
        widgets, unit = unit_panel()
        with monkeypatch.context() as mp:
            mp.setitem(sys.modules, "spacr.settings_advisor", None)
            widgets._refresh_analysis_unit_lock()
        assert unit.isEnabled() is True
        assert not hasattr(widgets, "_unit_locked") or widgets._unit_locked == set()

    with subtests.test(msg="a registry without the whole table locks only this unit"):
        import spacr.settings_advisor as advisor

        widgets, unit = unit_panel()
        with monkeypatch.context() as mp:
            mp.setattr(advisor, "requirements_for_unit", lambda _unit: {})
            mp.delattr(advisor, "UNIT_REQUIREMENTS")
            widgets._refresh_analysis_unit_lock()
        assert unit.isEnabled() is True
        assert widgets._unit_locked == set()

    with subtests.test(msg="a failing dependency pass does not undo the unit lock"):
        widgets, unit = unit_panel()
        widgets._refresh_setting_dependencies = fail("a dependency rule is broken")
        widgets._refresh_analysis_unit_lock()
        assert unit.isEnabled() is True
        assert widgets._unit_locked == set()


# ---------------------------------------------------------------------------
# the object-visibility pass
# ---------------------------------------------------------------------------

def test_object_visibility_follows_committed_values_and_survives_teardown(
        subtests, panel, qtbot, monkeypatch):
    """Diameter rules fire on commit; reads fall back to defaults; passes stop at close."""

    class _Refuses:
        def connect(self, _handler):
            raise RuntimeError("Signal source has been deleted")

    class _Records:
        def __init__(self):
            self.handlers = []

        def connect(self, handler):
            self.handlers.append(handler)

    class _Diameter:
        """A diameter control offering both signals, one of them broken."""

        def __init__(self):
            self.editingFinished = _Refuses()
            self.textChanged = _Records()

    def connected(control):
        widgets = SM.SettingsWidgets("measure")
        widgets._widgets = {"organelle_diameter": control}
        seen = []
        widgets._on_organelle_diameter_changed = lambda *args: seen.append(args)
        widgets._connect_object_visibility_signals()
        return seen

    with subtests.test(msg="a line edit is followed only once committed"):
        edit = QLineEdit("30")
        qtbot.addWidget(edit)
        seen = connected(edit)
        edit.setText("40")
        assert seen == []
        edit.editingFinished.emit()
        assert seen == [("organelle",)]

    with subtests.test(msg="a combo has no commit signal and follows its value"):
        combo = QComboBox()
        qtbot.addWidget(combo)
        combo.addItem("10", 10)
        combo.addItem("20", 20)
        seen = connected(combo)
        combo.setCurrentIndex(1)
        assert [args[0] for args in seen] == ["organelle"]

    with subtests.test(msg="a refused commit signal falls back to the value"):
        control = _Diameter()
        seen = connected(control)
        assert len(control.textChanged.handlers) == 1
        control.textChanged.handlers[0]("41")
        assert seen == [("organelle", "41")]

    with subtests.test(msg="a destroyed control reads as its default"):
        widgets = panel("mask")
        widgets.build_sections()
        key = next(name for name in widgets._object_visibility_keys()
                   if name in widgets._widgets)
        shiboken6.delete(widgets._widgets[key])
        state = widgets._object_visibility_settings()
        assert state[key] == widgets._defaults.get(key)

    with subtests.test(msg="a key this app does not have is not an error"):
        widgets = panel("mask")
        widgets.build_sections()
        before = dict(widgets._widgets)
        widgets._set_row_visible("no_such_setting_anywhere", False)
        assert widgets._widgets == before

    with subtests.test(msg="the first pass runs on the next event-loop turn"):
        widgets = panel("mask")
        calls = []
        monkeypatch.setattr(widgets, "refresh_object_visibility",
                            lambda: calls.append(True))
        widgets.build_sections()
        before = len(calls)
        qtbot.waitUntil(lambda: len(calls) > before)
        assert len(calls) == before + 1

    with subtests.test(msg="closing the screen cancels the pending pass"):
        parent = QWidget()
        widgets = SM.SettingsWidgets("mask", parent=parent)
        calls = []
        monkeypatch.setattr(widgets, "refresh_object_visibility",
                            lambda: calls.append(True))
        widgets.build_sections()
        before = len(calls)
        shiboken6.delete(parent)
        qtbot.wait(10)
        assert len(calls) == before


# ---------------------------------------------------------------------------
# _refresh_mask_gpu_enablement
# ---------------------------------------------------------------------------


def test_parallel_mask_controls_follow_the_gpu_count(subtests, panel,
                                                     monkeypatch):
    """Fewer than two GPUs greys both controls with the count in the reason;
    two or more frees mask_parallel, and the index list follows it."""
    from spacr import _mask_workers
    monkeypatch.setattr(_mask_workers, "_watch_mask_gpu_probe", lambda: None)
    few = ("Needs two or more compatible CUDA or ROCm GPUs; {n} found on "
           "this computer. Cluster Distribution runs use the GPUs allocated "
           "to the job.")
    cases = [  # (gpu count, mask_parallel, parallel on?, indices on?, index note)
        (0, False, False, False, few.format(n=0)),
        (1, True, False, False, few.format(n=1)),
        (2, True, True, True, None),
        (3, False, True, False,
         "Used only when mask_parallel is on. 3 GPUs found."),
    ]
    for count, on, want_parallel, want_indices, note in cases:
        with subtests.test(msg=f"{count} gpus, parallel={on}"):
            monkeypatch.setattr(_mask_workers, "_mask_gpu_count_for_controls",
                                lambda c=count: c)
            widgets = panel("mask")
            widgets.build_sections()
            parallel = widgets._built_control("mask_parallel")
            indices = widgets._built_control("mask_gpu_indices")
            widgets.set_value_for_key("mask_parallel", on)
            widgets._refresh_mask_gpu_enablement()
            assert (parallel.isEnabled(), indices.isEnabled()) == (
                want_parallel, want_indices)
            assert indices.property("_spacr_greyed_reason") == note
            if count < 2:
                assert parallel.property("_spacr_greyed_reason") == note
