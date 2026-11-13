"""The screen's small helpers when a collaborator is gone, refuses, or is plain.

Each case drives one helper on a stand-in ``self`` (or on a tiny real
layout) holding only what it reads: the screen's name, the console glyph
warm-up, layout indices, ambient and fractal teardown, owed-part stepping,
the settings advisor, the pickers, the usage sampler and the RAM dialogs.
The invariant they share is that a missing or raising collaborator yields
the documented fallback and nothing else is touched.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _refuse(*_args, **_kwargs):
    raise RuntimeError("gone")


class _Console:
    def __init__(self):
        self.notices = []
        self.stdout = []

    def append_notice(self, text, **kw):
        self.notices.append(text.format(**kw) if kw else text)

    def append_stdout(self, text, **_kw):
        self.stdout.append(text)


def test_a_helper_whose_collaborator_is_gone_falls_back_quietly(
        monkeypatch, subtests):
    """A missing or raising collaborator gives the helper's fallback.

    Invariant: an unreadable app table names the screen by its title; a
    console without the warm-up, one that raises, or a deleted Qt console is
    not warmed (a live one is, once); a gone caption host sweeps nothing; a
    late caption pass on a deleted widget is silent and any other failure
    is logged; the hint watcher passes the screen only its own widgets'
    events; a
    viewport with no split syncs nothing; an unreadable animation catalogue
    means no animation; the fractal pause counts only backdrops that paused;
    an ambient whose shutdown raises is still removed; a step that raises
    is dropped; headings whose sections cannot be listed count as shown.
    """
    import shiboken6
    from PySide6.QtCore import QObject
    from spacr.qt.widgets import section

    with subtests.test(case="screen name"), monkeypatch.context() as m:
        bogus = types.SimpleNamespace(APPS=[(), ("mask", "Mask")])
        m.setitem(sys.modules, "spacr.qt.app", bogus)
        assert AppScreen._screen_name(
            types.SimpleNamespace(app_key="mask")) == "Mask"
        bogus.APPS = [()]
        assert AppScreen._screen_name(types.SimpleNamespace(
            app_key="unlisted_key")) == aps.APP_TITLES.get(
            "unlisted_key", "unlisted_key")

    with subtests.test(case="glyph warm-up: no console or no warm-up"):
        ns = types.SimpleNamespace(_console=None)
        assert AppScreen._warm_console_glyphs(ns) is None
        assert vars(ns) == {"_console": None}
        plain = types.SimpleNamespace(_console=types.SimpleNamespace())
        assert AppScreen._warm_console_glyphs(plain) is None
        assert vars(plain._console) == {}

    with subtests.test(case="glyph warm-up: console raises"):
        asked = []
        assert AppScreen._warm_console_glyphs(types.SimpleNamespace(
            _console=types.SimpleNamespace(
                warm_notice_glyphs=lambda: (asked.append(True), _refuse()))
        )) is None
        assert asked == [True]

    with subtests.test(case="glyph warm-up: a deleted Qt console"):
        warmed = []

        class Console(QObject):
            def warm_notice_glyphs(self):
                warmed.append(True)

        console = Console()
        AppScreen._warm_console_glyphs(types.SimpleNamespace(_console=console))
        assert warmed == [True]
        shiboken6.delete(console)
        AppScreen._warm_console_glyphs(types.SimpleNamespace(_console=console))
        assert warmed == [True]

    with subtests.test(case="a gone caption host"):
        asked, arrived = [], []

        class Gone:
            def children(self):
                asked.append(True)
                raise RuntimeError("deleted")

        translator = types.SimpleNamespace(_on_arrival=arrived.append)
        assert aps._LateCaptionTranslator._on_arrivals_in(
            translator, Gone()) is None
        assert asked == [True] and arrived == []

    with subtests.test(case="hint watcher passes only its own widgets"):
        own, other = object(), object()
        heard = []
        watcher = types.SimpleNamespace(
            _widgets={own}, parent=lambda: types.SimpleNamespace(
                eventFilter=lambda w, e: heard.append((w, e)) or True))
        assert aps._HintWatcher.eventFilter(watcher, own, "hover") is True
        assert aps._HintWatcher.eventFilter(watcher, other, "hover") is False
        assert heard == [(own, "hover")]

    for name, error, logged in [
            ("late caption on a deleted widget",
             RuntimeError("Internal C++ object already deleted."), False),
            ("late caption pass that fails", ValueError("bad"), True)]:
        with subtests.test(case=name), monkeypatch.context() as m:
            from spacr.qt import i18n

            def refuse(widget, only_new=False, error=error):
                raise error

            logged_lines = []
            m.setattr(i18n, "retranslate_widget_tree", refuse)
            m.setattr(aps.LOG, "exception",
                      lambda msg, *a, **k: logged_lines.append(msg))
            assert aps._LateCaptionTranslator._translate(object()) is None
            assert bool(logged_lines) is logged

    with subtests.test(case="viewport with no split"):
        ns = types.SimpleNamespace(_split=None)
        assert aps._RuntimeViewport._sync_floor(ns) is None
        assert vars(ns) == {"_split": None}

    with subtests.test(case="animation catalogue unreadable"), \
            monkeypatch.context() as m:
        m.setitem(sys.modules, "spacr.qt.setting_animations",
                  types.ModuleType("spacr.qt.setting_animations"))
        assert aps._setting_has_an_animation("anything") is False
        assert aps._setting_has_an_animation("") is False

    with subtests.test(case="fractal paused backdrop by backdrop"), \
            monkeypatch.context() as m:
        good = types.SimpleNamespace(pause=lambda: True)
        bad = types.SimpleNamespace(pause=_refuse)
        m.setattr(aps, "_each_fractal_backdrop",
                  lambda screen: iter([good, bad, good]))
        assert aps._pause_the_fractal(object()) == 2

    with subtests.test(case="ambient shutdown raises"):
        widget = types.SimpleNamespace(
            shutdown=_refuse, set_animating=lambda on: None,
            deleteLater=lambda: None, setParent=lambda p: None,
            hide=lambda: None)
        ns = types.SimpleNamespace(
            _ambient=widget, _ambient_applied=1,
            _sync_page_palette=lambda: None, update=lambda: None)
        AppScreen._remove_ambient(ns)
        assert ns._ambient is None and ns._ambient_applied is None

    with subtests.test(case="an owed part's step raises"):
        def steps():
            raise ValueError("a step broke")
            yield

        ns = types.SimpleNamespace()
        ns.__dict__["_parts_stepping"] = {"p": steps()}
        AppScreen._step_owed_part(ns, "p")
        assert "p" not in ns._parts_stepping

    with subtests.test(case="sections cannot be listed"), \
            monkeypatch.context() as m:
        m.setattr(section, "_sections_below", _refuse)
        assert AppScreen._every_heading_below_is_hidden(object(), set()) \
            is False

    with subtests.test(case="no paired-data widget"):
        assert AppScreen._attached_database_rows(types.SimpleNamespace(
            _settings_model=types.SimpleNamespace(_widgets={}))) == []


def test_layout_indices_skip_spacers_and_fall_back_to_the_stretch(
        qtbot, subtests):
    """New rows go before the stretch, or the object grid before its heading.

    Invariant: with no stretch the index is the end; with a stretch it is
    the stretch's place; the object grid goes before the heading whose
    category source matches, and before the stretch when none does.
    """
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    hosts = []          # the layout dies with its host; keep every host

    def layout_of(*parts):
        host = QWidget()
        qtbot.addWidget(host)
        hosts.append(host)
        layout = QVBoxLayout(host)
        for part in parts:
            if part == "stretch":
                layout.addStretch(1)
            else:
                label = QLabel(part)
                if part == "heading":
                    label.setProperty("settingsCategorySource",
                                      "Cell Segmentation")
                layout.addWidget(label)
        return layout

    for parts, index in [(("a",), 1), (("a", "stretch"), 1)]:
        with subtests.test(case=f"stretch index of {parts}"):
            assert AppScreen._index_before_the_stretch(layout_of(*parts)) \
                == index

    ns = types.SimpleNamespace(
        OBJECT_GRID_BEFORE="Cell Segmentation",
        _index_before_the_stretch=AppScreen._index_before_the_stretch)
    for parts, index in [(("stretch", "plain"), 0),
                         (("heading", "stretch", "plain"), 0)]:
        with subtests.test(case=f"object grid in {parts}"):
            assert AppScreen._index_for_the_object_grid(
                ns, layout_of(*parts)) == index


def test_choosers_change_only_what_the_user_accepts(monkeypatch, subtests):
    """An advisor, a picker or a chooser changes the form only on consent.

    Invariant: an advisor refusal is said and nothing changes; given
    answers, the advice is applied, the trouble and the last run's note (or
    its folder when there is no note) are said and the written count
    reported; a declined proposal changes nothing and says so; a closed
    plate map returns no wells; the test-data route decides which example
    is fetched (none, measure or annotate) and is remembered when given.
    """
    from spacr import settings_advisor
    from spacr.qt.widgets import plate_map_picker, settings_advisor_dialog

    def advisor_screen(why_not=""):
        return types.SimpleNamespace(
            _console=_Console(),
            the_advisor_can_run=lambda: why_not,
            _settings_model=types.SimpleNamespace(
                collect=lambda: {"dependent_variable": "y"}),
            _tables_for_the_advisor=lambda values: (["c.csv"], ["s.csv"]),
            _reading_with_the_last_run=lambda reading, values: reading,
            apply_settings_dict=lambda chosen: len(chosen))

    def advice(m, run_note="", run_folder="", chosen=None):
        reading = types.SimpleNamespace(
            trouble=["thin table"], run_note=run_note, run_folder=run_folder)
        m.setattr(settings_advisor, "read_the_screen",
                  lambda counts, scores, dep: reading)
        m.setattr(settings_advisor, "advise_that_runs",
                  lambda r, answers: types.SimpleNamespace(
                      as_settings=lambda: chosen or {}))

    with subtests.test(case="refused"):
        ns = advisor_screen("No tables attached.")
        assert AppScreen.settings_for_my_data(ns) == {}
        assert ns._console.stdout == ["No tables attached.\n"]

    # (case, run note, run folder, chosen, returns, words said)
    cases = [
        ("applied and counted", "Last run was thin.", "", {"penalty": 1},
         {"penalty": 1},
         ["thin table", "Last run was thin.", "1 setting(s) written"]),
        ("run folder named without a note", "", "/runs/a", {}, {},
         ["diagnostics of /runs/a", "never shown"]),
    ]
    for name, note, folder, chosen, returns, words in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            advice(m, run_note=note, run_folder=folder, chosen=chosen)
            ns = advisor_screen()
            assert AppScreen.settings_for_my_data(
                ns, answers={"a": 1}) == returns
            text = "".join(ns._console.stdout)
            for word in words:
                assert word in text

    with subtests.test(case="proposal declined"), monkeypatch.context() as m:
        advice(m)

        class Declined:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                return 0

        m.setattr(settings_advisor_dialog, "SettingsAdvisorDialog", Declined)
        ns = advisor_screen()
        assert AppScreen.settings_for_my_data(ns) == {}
        assert "Nothing was changed" in ns._console.stdout[-1]

    with subtests.test(case="plate map closed"), monkeypatch.context() as m:
        class Closed:
            def __init__(self, before, parent=None):
                pass

            def exec(self):
                return 0

        m.setattr(plate_map_picker, "PlateMapPicker", Closed)
        assert AppScreen.pick_wells_for(
            types.SimpleNamespace(),
            types.SimpleNamespace(text=lambda: "A1")) == ""

    for route, called, out in [("", [], {}), ("stream", ["measure"], {"m": 1}),
                               ("crops", ["annotate"], {"a": 1})]:
        with subtests.test(case=f"route {route!r}"):
            seen = []
            ns = types.SimpleNamespace(
                load_the_measure_example=lambda ask=None:
                seen.append("measure") or {"m": 1},
                load_the_annotate_example=lambda ask=None:
                seen.append("annotate") or {"a": 1})
            chooser = types.SimpleNamespace(chosen=route, exec=lambda: 0)
            assert AppScreen.choose_the_test_data(ns, chooser=chooser) == out
            assert seen == called
            if route:
                assert ns._test_data_route == route


def test_usage_and_ram_dialogs_answer_without_real_hardware(
        monkeypatch, subtests, qtbot):
    """GPU readings and RAM dialogs work from stand-ins, never the machine.

    Invariant: with NVML silent, GPUtil's first GPU gives the load and VRAM
    percentages, no GPU reads zero, and a raising driver leaves the GPU out;
    the RAM-guard dialog returns the user's choice; the free-RAM dialog
    offers no boxes when nothing can be closed.
    """
    from spacr import resource_log

    fake = types.ModuleType("GPUtil")
    monkeypatch.setitem(sys.modules, "GPUtil", fake)
    monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: True)
    monkeypatch.setattr(aps, "_nvml_gpu_usage", lambda: None)
    gpu = types.SimpleNamespace(load=0.5, memoryUtil=0.25)
    for name, gpus, reading in [
            ("one GPU", lambda: [gpu], {"gpu": 50.0, "vram": 25.0}),
            ("no GPU", lambda: [], {"gpu": 0, "vram": 0}),
            ("driver raises", _refuse, None)]:
        with subtests.test(case=f"GPU sample: {name}"):
            fake.getGPUs = gpus
            sample = aps._sample_usage(False)
            if reading is None:
                assert "gpu" not in sample
            else:
                assert {k: sample[k] for k in reading} == reading

    with subtests.test(case="RAM guard returns the choice"):
        class Fake(aps._RamGuardDialog):
            def __init__(self, parent, plan):
                self.choice = "use"

            def exec(self):
                return 0

        assert Fake.ask(None, {}) == "use"

    with subtests.test(case="free RAM with nothing to close"), \
            monkeypatch.context() as m:
        m.setattr(resource_log, "_closable_processes",
                  lambda psutil_module=None: [])
        dialog = aps._FreeRamDialog()
        qtbot.addWidget(dialog)
        assert dialog.boxes == []
