"""Module-level helpers of the app screen, on the inputs that break them.

None of these draws anything. They decide what a settings heading is tinted,
which keys from an old settings file still reach a control, what the resource
strip is allowed to claim about a machine with no GPU, and what a late
translation pass does when the panel it was queued for has already closed.

Each one sits on a failure path that the screen never sees in a normal
session, which is exactly why the behaviour has to be stated: a maturity that
silently reads "stable", a renamed key dropped on the floor, or a GPU reading
invented on a CPU-only host all look correct on screen.
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

pytestmark = pytest.mark.qt


# -- how mature a folded module's settings are -------------------------------

class TestModuleMaturity:

    def test_a_fold_table_that_cannot_be_read_falls_back_to_the_registry(
            self, monkeypatch):
        """A broken fold table must not stop a settings panel being tinted.

        The lookup exists so a module whose row was deleted keeps the stage
        its tile carried. When the tables themselves cannot be read there is
        nothing better than what the registry says, and raising here would
        take the whole settings panel down with it.
        """
        from spacr.qt.widgets import fold_strip
        from spacr.qt.app import app_stage

        def refusing(_key):
            raise RuntimeError("the fold tables are unreadable")

        monkeypatch.setattr(fold_strip, "folded_fallback", refusing)

        assert aps.module_maturity("timelapse") == app_stage("timelapse")

    def test_an_explicitly_experimental_heading_takes_the_cautious_stage(self):
        """A stable module may still hold an (Alpha) category.

        The heading is the more cautious of the two, because the settings
        under it are the experimental part of an otherwise finished module.
        """
        assert aps.settings_section_maturity("mask", "Alpha") == "alpha"
        assert aps.settings_section_maturity(
            "mask", "Object filtration (Alpha)") == "alpha"
        assert aps.settings_section_maturity(
            "mask", "Object filtration (Beta)") == "beta"


# -- a settings file written before a key was renamed -------------------------

class TestLegacySettingKeys:

    def test_a_retired_key_reaches_the_control_under_its_new_name(self):
        """Otherwise the value is dropped and the run uses the default.

        Silently: no widget answers to the old name, so nothing on the panel
        shows that the file said anything at all.
        """
        translated = aps._translate_legacy_setting_keys(
            {"png_dims": [0, 1, 2], "src": "/data"})

        assert translated["png_channel_mapping"] == [0, 1, 2]
        assert "png_dims" not in translated
        assert translated["src"] == "/data"

    def test_the_new_key_wins_when_the_file_holds_both(self):
        """Someone who said outright which channel is red keeps their answer."""
        translated = aps._translate_legacy_setting_keys(
            {"png_dims": [0, 1, 2], "png_channel_mapping": [2, 1, 0]})

        assert translated["png_channel_mapping"] == [2, 1, 0]

    def test_the_caller_s_dict_is_left_alone(self):
        original = {"png_dims": [0, 1, 2]}

        aps._translate_legacy_setting_keys(original)

        assert original == {"png_dims": [0, 1, 2]}


# -- getting rid of a half-built widget --------------------------------------

def test_a_widget_that_refuses_to_be_unparented_is_still_let_go():
    """The failure paths use this on widgets in every state.

    A backdrop that raises on ``setParent`` is exactly the half-installed
    case this exists for; letting the exception out would abandon the screen
    mid-construction instead of the one widget.
    """
    attempts = []

    class Awkward:
        def setParent(self, parent):
            attempts.append(parent)
            raise RuntimeError("this widget is already gone")

    aps._discard_widget(Awkward())
    aps._discard_widget(None)

    assert attempts == [None]


# -- the late translation pass ------------------------------------------------

class TestTranslatingAPanelThatArrivedLate:

    def test_a_panel_closed_before_the_pass_ran_is_not_a_failure(
            self, monkeypatch, caplog):
        """The pass is queued; the user can shut the panel before it runs."""
        import spacr.qt.i18n as i18n

        def already_gone(_widget):
            raise RuntimeError("Internal C++ object already deleted.")

        monkeypatch.setattr(i18n, "retranslate_widget_tree", already_gone)

        with caplog.at_level("ERROR"):
            aps._LateCaptionTranslator._translate(object())

        assert not caplog.records

    def test_a_translation_that_goes_wrong_is_reported_not_swallowed(
            self, monkeypatch, caplog):
        """A catalog fault is a defect; a closed panel is not.

        Collapsing the two would leave every translation bug invisible,
        because the pass runs on a timer nobody is watching.
        """
        import spacr.qt.i18n as i18n

        def broken(_widget):
            raise ValueError("the catalog is malformed")

        monkeypatch.setattr(i18n, "retranslate_widget_tree", broken)

        with caplog.at_level("ERROR"):
            aps._LateCaptionTranslator._translate(object())

        assert any("could not translate" in record.message
                   for record in caplog.records)


# -- what the resource strip is allowed to say -------------------------------

class TestSamplingTheMachine:

    def test_a_host_with_no_psutil_still_returns_a_sample(self, monkeypatch):
        """Missing psutil leaves those keys out rather than failing the poll.

        The poll runs every couple of seconds on a worker thread; an
        exception there would be a repeating traceback rather than a strip
        that simply says less.
        """
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: False)

        sample = aps._sample_usage(per_core=True)

        assert "ram" not in sample and "cpu" not in sample
        assert sample["gpu"] == 0 and sample["vram"] == 0

    def test_a_cpu_only_host_reports_zero_rather_than_nothing(self,
                                                              monkeypatch):
        """"No GPU" and "GPU idle" both read as zero; "unknown" reads as a gap.

        The strip has a GPU bar either way, so leaving the key out would draw
        a bar with no number in it on every CPU-only machine.
        """
        monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: False)

        sample = aps._sample_usage(per_core=False)

        assert sample["gpu"] == 0
        assert sample["vram"] == 0
        assert "per_core" not in sample

    def test_a_machine_with_a_card_reports_the_first_card(self, monkeypatch):
        """The strip is one machine's load, not a per-card table."""
        card = types.SimpleNamespace(load=0.42, memoryUtil=0.75)
        fake = types.ModuleType("GPUtil")
        fake.getGPUs = lambda: [card]
        monkeypatch.setitem(sys.modules, "GPUtil", fake)
        monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: True)

        sample = aps._sample_usage(per_core=False)

        assert sample["gpu"] == pytest.approx(42.0)
        assert sample["vram"] == pytest.approx(75.0)

    def test_a_driver_that_lists_no_cards_reads_as_idle_not_as_missing(
            self, monkeypatch):
        """nvidia-smi present and no card enumerated is still a number."""
        fake = types.ModuleType("GPUtil")
        fake.getGPUs = lambda: []
        monkeypatch.setitem(sys.modules, "GPUtil", fake)
        monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: True)

        sample = aps._sample_usage(per_core=False)

        assert sample["gpu"] == 0
        assert sample["vram"] == 0


class TestWhetherGpuTelemetryCanRun:

    def test_an_nvidia_smi_on_the_path_is_enough(self, monkeypatch):
        monkeypatch.setattr(aps.shutil, "which",
                            lambda name: "/usr/bin/nvidia-smi")

        assert aps._nvidia_smi_available() is True

    def test_a_linux_host_without_it_says_no(self, monkeypatch):
        """The check keeps CPU-only hosts out of GPUtil's subprocess boundary.

        Calling nvidia-smi when it is not installed once crashed a Qt process
        in CPython's subprocess boundary after hundreds of short-lived worker
        threads, so "probably not there" is not good enough.
        """
        monkeypatch.setattr(aps.shutil, "which", lambda name: None)
        monkeypatch.setattr(aps.sys, "platform", "linux")

        assert aps._nvidia_smi_available() is False

    def test_windows_looks_where_the_driver_puts_it(self, monkeypatch,
                                                     tmp_path):
        """On Windows nvidia-smi is installed off the PATH."""
        drive = tmp_path
        installed = (drive / "Program Files" / "NVIDIA Corporation" / "NVSMI")
        installed.mkdir(parents=True)
        (installed / "nvidia-smi.exe").write_text("", encoding="utf-8")

        monkeypatch.setattr(aps.shutil, "which", lambda name: None)
        monkeypatch.setattr(aps.sys, "platform", "win32")
        monkeypatch.setenv("SystemDrive", str(drive))

        assert aps._nvidia_smi_available() is True

    def test_windows_without_the_driver_directory_says_no(self, monkeypatch,
                                                           tmp_path):
        monkeypatch.setattr(aps.shutil, "which", lambda name: None)
        monkeypatch.setattr(aps.sys, "platform", "win32")
        monkeypatch.setenv("SystemDrive", str(tmp_path))

        assert aps._nvidia_smi_available() is False
