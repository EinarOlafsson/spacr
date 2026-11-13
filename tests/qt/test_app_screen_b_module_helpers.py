"""Module-level helpers of the app screen, on the inputs that break them.

None of these needs a built screen. They decide what a settings heading is
tinted, which keys from an old settings file still reach a control, what the
resource strip may claim about a machine with no GPU, what a late translation
pass does when its panel has closed, which tables the advisor reads and where
the fractal backdrop is found. Each sits on a failure path a normal session
never takes, where a wrong answer still looks correct on screen.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the collaborator has gone")


# -- maturity of a settings heading and keys of an old settings file ---------

def _maturity_fold_table_unreadable(mp):
    from spacr.qt.widgets import fold_strip
    from spacr.qt.app import app_stage

    mp.setattr(fold_strip, "folded_fallback", _boom)
    assert aps.module_maturity("timelapse") == app_stage("timelapse")


def _maturity_heading_takes_the_cautious_stage(mp):
    assert aps.settings_section_maturity("mask", "Alpha") == "alpha"
    assert aps.settings_section_maturity(
        "mask", "Object filtration (Alpha)") == "alpha"
    assert aps.settings_section_maturity("mask", "Confluency α") == "alpha"
    assert aps.settings_section_maturity(
        "mask", "Object filtration (Beta)") == "beta"


def _legacy_retired_key_is_renamed(mp):
    translated = aps._translate_legacy_setting_keys(
        {"png_dims": [0, 1, 2], "src": "/data"})
    assert translated["png_channel_mapping"] == [0, 1, 2]
    assert "png_dims" not in translated
    assert translated["src"] == "/data"


def _legacy_new_key_wins(mp):
    translated = aps._translate_legacy_setting_keys(
        {"png_dims": [0, 1, 2], "png_channel_mapping": [2, 1, 0]})
    assert translated["png_channel_mapping"] == [2, 1, 0]


def _legacy_caller_dict_untouched(mp):
    original = {"png_dims": [0, 1, 2]}
    aps._translate_legacy_setting_keys(original)
    assert original == {"png_dims": [0, 1, 2]}


def test_settings_maturity_and_legacy_keys_take_the_safe_answer(
        subtests, monkeypatch):
    """A heading's stage never reads more mature than its module or label says,
    an unreadable fold table falls back to the registry, and a renamed key in an
    old settings file reaches its control under the new name without the new
    spelling being overwritten or the caller's dict being mutated."""
    cases = {
        "fold table unreadable": _maturity_fold_table_unreadable,
        "experimental heading": _maturity_heading_takes_the_cautious_stage,
        "retired key renamed": _legacy_retired_key_is_renamed,
        "new key wins": _legacy_new_key_wins,
        "caller dict untouched": _legacy_caller_dict_untouched,
    }
    completed = []
    for name, case in cases.items():
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(mp)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- what the resource strip may say about the machine -----------------------

def _fake_gputil(mp, cards):
    fake = types.ModuleType("GPUtil")
    fake.getGPUs = lambda: list(cards)
    mp.setitem(sys.modules, "GPUtil", fake)
    mp.setattr(aps, "_nvidia_smi_available", lambda: True)
    mp.setattr(aps, "_nvml_gpu_usage", lambda: None)


def _sample_without_psutil(mp, tmp_path):
    mp.setitem(sys.modules, "psutil", None)
    mp.setattr(aps, "_nvidia_smi_available", lambda: False)
    sample = aps._sample_usage(per_core=True)
    assert "ram" not in sample and "cpu" not in sample
    assert sample["gpu"] == 0 and sample["vram"] == 0


def _sample_cpu_only_host(mp, tmp_path):
    mp.setattr(aps, "_nvidia_smi_available", lambda: False)
    sample = aps._sample_usage(per_core=False)
    assert sample["gpu"] == 0
    assert sample["vram"] == 0
    assert "per_core" not in sample


def _sample_first_card(mp, tmp_path):
    _fake_gputil(mp, [types.SimpleNamespace(load=0.42, memoryUtil=0.75)])
    sample = aps._sample_usage(per_core=False)
    assert sample["gpu"] == pytest.approx(42.0)
    assert sample["vram"] == pytest.approx(75.0)


def _sample_driver_lists_no_card(mp, tmp_path):
    _fake_gputil(mp, [])
    sample = aps._sample_usage(per_core=False)
    assert sample["gpu"] == 0
    assert sample["vram"] == 0


def _smi_on_path(mp, tmp_path):
    mp.setattr(aps.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert aps._nvidia_smi_available() is True


def _smi_missing_on_linux(mp, tmp_path):
    mp.setattr(aps.shutil, "which", lambda name: None)
    mp.setattr(aps.sys, "platform", "linux")
    assert aps._nvidia_smi_available() is False


def _smi_in_the_windows_driver_folder(mp, tmp_path):
    installed = tmp_path / "Program Files" / "NVIDIA Corporation" / "NVSMI"
    installed.mkdir(parents=True)
    (installed / "nvidia-smi.exe").write_text("", encoding="utf-8")
    mp.setattr(aps.shutil, "which", lambda name: None)
    mp.setattr(aps.sys, "platform", "win32")
    mp.setenv("SystemDrive", str(tmp_path))
    assert aps._nvidia_smi_available() is True


def _smi_windows_without_driver(mp, tmp_path):
    mp.setattr(aps.shutil, "which", lambda name: None)
    mp.setattr(aps.sys, "platform", "win32")
    mp.setenv("SystemDrive", str(tmp_path))
    assert aps._nvidia_smi_available() is False


def test_machine_sampling_reports_numbers_never_invents_a_gpu(
        subtests, monkeypatch, tmp_path):
    """The usage sample always carries gpu/vram numbers (zero when there is no
    card or no driver, the first card's load when there is), leaves psutil keys
    out instead of failing, and GPU telemetry is only attempted when nvidia-smi
    is actually installed (on the PATH, or in the Windows driver folder)."""
    cases = {
        "no psutil": _sample_without_psutil,
        "cpu-only host": _sample_cpu_only_host,
        "first card": _sample_first_card,
        "driver lists no card": _sample_driver_lists_no_card,
        "smi on PATH": _smi_on_path,
        "linux without smi": _smi_missing_on_linux,
        "windows driver folder": _smi_in_the_windows_driver_folder,
        "windows without driver": _smi_windows_without_driver,
    }
    completed = []
    for name, case in cases.items():
        folder = tmp_path / name.replace(" ", "_")
        folder.mkdir()
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(mp, folder)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- helpers that must survive a widget or module that is not there ----------

def _discard_a_widget_that_refuses(env):
    attempts = []

    class Awkward:
        def setParent(self, parent):
            attempts.append(parent)
            raise RuntimeError("this widget is already gone")

    aps._discard_widget(Awkward())
    aps._discard_widget(None)
    assert attempts == [None]


def _late_translation_of_a_closed_panel(env):
    import spacr.qt.i18n as i18n

    # The stub takes what the real one takes (`only_new=True` is passed).
    def already_gone(_widget, _language=None, *, only_new=False):
        raise RuntimeError("Internal C++ object already deleted.")

    env.mp.setattr(i18n, "retranslate_widget_tree", already_gone)
    with env.caplog.at_level("ERROR"):
        aps._LateCaptionTranslator._translate(object())
    assert not env.caplog.records


def _late_translation_fault_is_reported(env):
    import spacr.qt.i18n as i18n

    def broken(_widget, _language=None, *, only_new=False):
        raise ValueError("the catalog is malformed")

    env.mp.setattr(i18n, "retranslate_widget_tree", broken)
    with env.caplog.at_level("ERROR"):
        aps._LateCaptionTranslator._translate(object())
    assert any("could not translate" in record.message
               for record in env.caplog.records)


def _orphaned_explainer_browser_renders_nothing(env):
    rendered = []

    class Owner:
        def refresh(self, language):
            rendered.append(language)

    owner = Owner()
    browser = aps._ExplainerBrowser(owner.refresh)
    env.qtbot.addWidget(browser)
    browser.retranslate_dynamic_content("is")
    assert rendered == ["is"]
    del owner
    gc.collect()
    browser.retranslate_dynamic_content("de")
    assert rendered == ["is"]


def _refuse_imports(mp, *needles):
    import builtins

    real = builtins.__import__

    def refuse(name, g=None, l=None, fromlist=(), level=0):
        if any(n in name or n in (fromlist or ()) for n in needles):
            raise ImportError(f"{name} is not installed")
        return real(name, g, l, fromlist, level)

    mp.setattr(builtins, "__import__", refuse)


def _no_sweep_module(env):
    _refuse_imports(env.mp, "parameter_sweep", "sweepable")
    assert aps._sweepable("mask") is False


def _no_hyperparam_module(env):
    _refuse_imports(env.mp, "hyperparam", "searchable")
    assert aps._hyperparam_searchable("mask") is False


def _probes_answer_with_modules_present(env):
    assert isinstance(aps._sweepable("mask"), bool)
    assert isinstance(aps._hyperparam_searchable("mask"), bool)


def _caption_index_walk_builds_once(env):
    built = []

    def caption_the_rest():
        built.append(1)
        index["late_key"] = "late help"

    index = aps._CaptionsBuiltWhenTheyAreAskedFor(caption_the_rest)
    index["early_key"] = "early help"
    assert set(index.keys()) == {"early_key", "late_key"}
    assert sorted(index.values()) == ["early help", "late help"]
    assert built == [1]


def _caption_index_lookup_builds_nothing(env):
    built = []
    index = aps._CaptionsBuiltWhenTheyAreAskedFor(lambda: built.append(1))
    assert index.get("anything") is None
    assert built == []
    list(index.keys())
    assert len(built) == 1


def _row_list_reads_build_the_rest(env):
    late = ("late label", object())

    def build_the_rest():
        rows.append(late)

    rows = aps._RowsBuiltWhenTheyAreAskedFor(build_the_rest)
    rows.append(("early label", object()))
    assert len(rows) == 2
    assert late in rows
    assert "late label" in repr(rows)


def test_helpers_survive_absent_collaborators_and_build_lazily(
        subtests, monkeypatch, qtbot, caplog):
    """Teardown helpers let go of widgets that refuse, a late translation pass
    swallows a closed panel but logs a real catalog fault, an orphaned explainer
    renders nothing, a missing optional module means "not available", and the
    lazy caption/row containers build their remainder exactly once, only when a
    reader walks them (a single lookup builds nothing)."""
    cases = {
        "discard refusing widget": _discard_a_widget_that_refuses,
        "late pass on closed panel": _late_translation_of_a_closed_panel,
        "late pass catalog fault": _late_translation_fault_is_reported,
        "orphaned explainer": _orphaned_explainer_browser_renders_nothing,
        "no sweep module": _no_sweep_module,
        "no hyperparam module": _no_hyperparam_module,
        "probes with modules": _probes_answer_with_modules_present,
        "caption index walk": _caption_index_walk_builds_once,
        "caption index lookup": _caption_index_lookup_builds_nothing,
        "row list reads": _row_list_reads_build_the_rest,
    }
    completed = []
    for name, case in cases.items():
        caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(types.SimpleNamespace(mp=mp, qtbot=qtbot, caplog=caplog))
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- the fractal backdrop, found and wound down for a run --------------------

def _backdrop(**answers):
    return types.SimpleNamespace(backend_name="numba", **answers)


def _screen_showing(*backdrops):
    window = types.SimpleNamespace(findChildren=lambda _cls: list(backdrops))
    return types.SimpleNamespace(window=lambda: window)


def _gone_window():
    raise RuntimeError("wrapped C/C++ object has been deleted")


def _fractal_found_only_with_a_window(caplog):
    one = _backdrop(pause=lambda: True)
    assert list(aps._each_fractal_backdrop(_screen_showing(one))) == [one]
    unparented = types.SimpleNamespace(window=lambda: None)
    assert list(aps._each_fractal_backdrop(unparented)) == []


def _fractal_pause_unwalkable_window(caplog):
    assert aps._pause_the_fractal(types.SimpleNamespace(window=_gone_window)) == 0
    assert "could not reach the fractal" in caplog.text


def _fractal_resume_unwalkable_window(caplog):
    assert aps._resume_the_fractal(
        types.SimpleNamespace(window=_gone_window)) == 0
    assert "could not reach the fractal" in caplog.text


def _fractal_refusing_resume_spares_the_rest(caplog):
    def refusing():
        raise RuntimeError("the GPU context is gone")

    resumed = aps._resume_the_fractal(_screen_showing(
        _backdrop(pause=lambda: True, resume=lambda: False),
        _backdrop(pause=lambda: True, resume=refusing),
        _backdrop(pause=lambda: True, resume=lambda: True)))
    assert resumed == 1
    assert "could not resume the fractal" in caplog.text


def test_fractal_backdrops_are_found_paused_and_resumed_defensively(
        subtests, caplog):
    """Backdrops are found only through a screen's window, a window that cannot
    be walked pauses/resumes nothing and says so, and one backdrop refusing to
    resume neither stops the others nor counts as resumed."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    cases = {
        "found via window": _fractal_found_only_with_a_window,
        "pause unwalkable": _fractal_pause_unwalkable_window,
        "resume unwalkable": _fractal_resume_unwalkable_window,
        "one refuses resume": _fractal_refusing_resume_spares_the_rest,
    }
    completed = []
    for name, case in cases.items():
        caplog.clear()
        with subtests.test(case=name):
            case(caplog)
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


# -- the paths a settings mapping names --------------------------------------

ADVISOR_TABLE_CASES = {
    "junk rows skipped": (
        {"paired_data": ["not a row", None,
                         {"count": "/c.csv", "score": "/s.csv"}]},
        ["/c.csv"], ["/s.csv"]),
    "legacy single strings": (
        {"count_data": "/counts.csv", "score_data": "/scores.csv"},
        ["/counts.csv"], ["/scores.csv"]),
    "path named twice counted once": (
        {"paired_data": [{"count": "/c.csv"}],
         "count_data": ["/c.csv", "/other.csv"]},
        ["/c.csv", "/other.csv"], None),
    "score-only row": (
        {"paired_data": [{"score": "plate1_dv.csv"},
                         {"count": "plate2_counts.csv",
                          "score": "plate2_dv.csv"},
                         "not a row at all"]},
        ["plate2_counts.csv"], ["plate1_dv.csv", "plate2_dv.csv"]),
    "legacy string and duplicate list": (
        {"count_data": "old_counts.csv",
         "score_data": ["old_scores.csv", "old_scores.csv"]},
        ["old_counts.csv"], ["old_scores.csv"]),
}

FIRST_COUNT_CASES = {
    "empty count defers to next row": (
        {"paired_data": [{"count": "  ", "score": "p1_dv.csv"},
                         {"count": "p2_counts.csv"}]}, "p2_counts.csv"),
    "legacy list when no paired row names one": (
        {"paired_data": [{"count": ""}], "count_data": ["legacy_counts.csv"]},
        "legacy_counts.csv"),
}

SCREEN_FOLDER_CASES = {
    "relative results folder": (os.path.join("results", "run_1"), ["results"]),
    "absolute screen above results": (
        os.path.join("/data", "screen", "results", "run_1"),
        [os.path.normpath(os.path.join("/data", "screen", "results")),
         os.path.normpath(os.path.join("/data", "screen"))]),
}


def test_settings_path_readers_accept_every_spelling(subtests):
    """The advisor's table reader, the first-count lookup and the screen-folder
    walk accept both the paired-row and the legacy spellings, skip junk and
    half-filled rows, never count a path twice, and never invent a folder above
    a relative ``results`` directory."""
    for name, (values, counts, scores) in ADVISOR_TABLE_CASES.items():
        with subtests.test(case=f"advisor tables: {name}"):
            got_counts, got_scores = AppScreen._tables_for_the_advisor(values)
            assert got_counts == counts
            if scores is not None:
                assert got_scores == scores
    for name, (values, expected) in FIRST_COUNT_CASES.items():
        with subtests.test(case=f"first count: {name}"):
            assert aps._first_count_file(values) == expected
    for name, (folder, expected) in SCREEN_FOLDER_CASES.items():
        with subtests.test(case=f"screen folders: {name}"):
            assert AppScreen._screen_folders_above(folder) == expected


# -- the plugin apps a screen picks up while it is being imported -------------

def _run_the_module_body_again(path: Path):
    """Execute ``app_screen``'s module body into a throwaway module object.

    The plugin absorption runs once at import; re-running the body under the
    real filename is the only way to drive it without touching the imported
    module every other test holds.
    """
    module = types.ModuleType("spacr.qt.screens.app_screen")
    module.__file__ = str(path)
    module.__package__ = "spacr.qt.screens"
    module.__loader__ = aps.__loader__
    module.__spec__ = aps.__spec__
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
         module.__dict__)
    return module


@pytest.fixture
def _keep_the_qss_registry():
    """The module body re-registers the settings-column QSS block."""
    from spacr.qt import theme

    saved = theme._WIDGET_QSS.get(aps.SETTINGS_PANEL_NAME)
    yield
    if saved is not None:
        theme._WIDGET_QSS[aps.SETTINGS_PANEL_NAME] = saved


def test_plugin_apps_get_a_header_unless_the_plugin_table_explodes(
        subtests, monkeypatch, _keep_the_qss_registry):
    """A registered plugin names itself in APP_TITLES and APP_INTROS, and a
    plugin table that raises at import leaves it out while the screen class
    still imports."""
    import spacr.plugins

    contribution = types.SimpleNamespace(
        key="ghost_module", name="Ghost Module",
        description="Counts what is not there.")
    with subtests.test(case="registered plugin"), monkeypatch.context() as mp:
        mp.setattr(spacr.plugins, "plugin_apps", lambda: (contribution,))
        module = _run_the_module_body_again(Path(aps.__file__))
        assert module.APP_TITLES["ghost_module"] == "Ghost Module"
        assert module.APP_INTROS["ghost_module"] == "Counts what is not there."
    with subtests.test(case="exploding table"), monkeypatch.context() as mp:
        mp.setattr(spacr.plugins, "plugin_apps", _boom)
        broken = _run_the_module_body_again(Path(aps.__file__))
        assert "ghost_module" not in broken.APP_TITLES
        assert issubclass(broken.AppScreen, QWidget)


# -- opening the umbrella above a heading ------------------------------------

def test_opening_a_sub_heading_opens_only_a_closed_umbrella(subtests, qtbot):
    """Expanding a sub-heading expands a collapsed parent, collapsing one leaves
    the parent open, an already-open parent is not re-asserted, and a parent
    that is not a heading (or has gone) is not an error."""
    from spacr.qt.widgets.section import Section

    for start in (False, True):
        with subtests.test(case=f"real section, start expanded={start}"):
            parent = Section("Advanced settings")
            qtbot.addWidget(parent)
            parent.set_expanded(start)
            AppScreen._open_the_headings_above(parent, start is False)
            assert parent.is_expanded() is True

    with subtests.test(case="already open is left alone"):
        opened = []
        already = types.SimpleNamespace(is_expanded=lambda: True,
                                        set_expanded=opened.append)
        shut = types.SimpleNamespace(is_expanded=lambda: False,
                                     set_expanded=opened.append)
        AppScreen._open_the_headings_above(already, True)
        assert opened == []
        AppScreen._open_the_headings_above(shut, True)
        assert opened == [True]

    with subtests.test(case="vanished or non-heading parent"):
        checks = []

        def vanished_heading():
            checks.append(True)
            raise RuntimeError("the heading has gone")

        AppScreen._open_the_headings_above(object(), True)
        AppScreen._open_the_headings_above(
            types.SimpleNamespace(is_expanded=vanished_heading), True)
        assert checks == [True]
