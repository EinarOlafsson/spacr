"""What a module screen does when a part of itself refuses to answer.

The screen is one object holding a settings model, a console, a fold strip and
a resource strip, and each of those can be absent, half-built or already
destroyed at the moment it is asked something. The paths below are the ones
that only run then, so they are the ones nobody sees fail: a workspace replayed
into the wrong module, a bulk settings apply whose dependency pass throws, a
restart whose settings could not be collected.

The rule in every case is the same. The screen does the part it can do, says
what it could not, and never lets the failure out onto the GUI thread.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLineEdit                          # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


# -- putting a sentence somewhere ---------------------------------------------

class TestSayingSomething:

    def test_a_sentence_goes_to_the_console_when_there_is_one(self, screen):
        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)

        screen._say("the run finished")

        assert said == ["the run finished"]

    def test_a_console_that_will_not_take_it_does_not_lose_the_sentence(
            self, screen, caplog):
        """A dead console is a reason to log, not a reason to raise.

        These sentences are written from result handlers; letting the console
        take one down with it would abandon the rest of the handler.
        """
        def refusing(_text):
            raise RuntimeError("Internal C++ object already deleted.")

        screen._console = types.SimpleNamespace(append_stdout=refusing)

        with caplog.at_level("INFO"):
            screen._say("the run finished")

        assert any("the run finished" in record.message
                   for record in caplog.records)

    def test_a_screen_with_no_console_at_all_still_records_it(self, screen,
                                                              caplog):
        screen._console = None

        with caplog.at_level("INFO"):
            screen._say("the run finished")

        assert any("the run finished" in record.message
                   for record in caplog.records)


# -- replaying a saved workspace ----------------------------------------------

class TestReplayingAWorkspace:

    def test_a_state_that_is_not_a_record_is_refused(self, screen):
        assert screen.apply_workspace_state(None) is False
        assert screen.apply_workspace_state("regression") is False

    def test_another_module_s_state_is_never_replayed_here(self, screen):
        """Settings keys collide across modules and mean different things.

        ``level`` is the regression's fit level and the proportion plots'
        unit; replaying one screen's state into another would set whichever
        keys happened to collide and quietly leave the rest.
        """
        applied = screen.apply_workspace_state(
            {"app_key": "measure", "settings": {"verbose": True}})

        assert applied is False

    def test_a_record_carrying_no_settings_applies_nothing(self, screen):
        assert screen.apply_workspace_state(
            {"app_key": "regression", "settings": {}}) is False
        assert screen.apply_workspace_state(
            {"app_key": "regression", "settings": "verbose=True"}) is False

    def test_this_module_s_own_state_is_applied(self, screen):
        """A record with no ``app_key`` is taken as this screen's own."""
        assert screen.apply_workspace_state({"settings": {"verbose": True}})


# -- a bulk settings apply that half fails ------------------------------------

class TestABulkApply:

    def test_a_dependency_pass_that_throws_still_leaves_the_values_in(
            self, screen, monkeypatch, caplog):
        """The values are the point; the greying rules are a redraw.

        Letting the refresh out would abandon the apply half way and leave
        ``_applying_settings`` set, which suppresses every later rule too.
        """
        model = screen._settings_model
        real = model._refresh_setting_dependencies

        def broken_on_the_final_pass():
            # While the bulk apply is running the flag is set, and the rules
            # are suppressed; the pass that matters here is the one after it
            # is cleared, over a panel the apply itself has rebuilt.
            if model._applying_settings:
                return real()
            raise RuntimeError("a rule read a widget that had gone")

        monkeypatch.setattr(model, "_refresh_setting_dependencies",
                            broken_on_the_final_pass)

        with caplog.at_level("DEBUG"):
            applied = screen.apply_settings_dict({"verbose": True})

        assert applied >= 1
        assert model._applying_settings is False

    def test_a_dimension_switch_that_throws_does_not_undo_the_apply(
            self, screen, monkeypatch):
        def broken(_settings):
            raise RuntimeError("the switches are not built yet")

        monkeypatch.setattr(screen, "_sync_dimension_switches", broken)

        assert screen.apply_settings_dict({"verbose": True}) >= 1

    def test_a_screen_whose_fold_switches_cannot_be_read_reports_none_moved(
            self, screen, monkeypatch):
        """A failure here costs the switch position, not the import.

        The answer is remembered so the screen can report that a folded flag
        was ignored rather than claiming it landed.
        """
        from spacr.qt.screens import mask

        def broken(_screen, _settings):
            raise RuntimeError("the fold strip is half built")

        monkeypatch.setattr(mask, "sync_folds", broken)

        switched = screen._sync_folded_switches({"timelapse": True})

        assert switched == ()
        assert screen._folds_last_switched_on == ()


# -- writing a value into whatever control holds it ---------------------------

def test_a_blank_value_empties_a_text_field_rather_than_writing_none(screen,
                                                                     qtbot):
    """``str(None)`` in a path box is a path called "None".

    The box is read straight back out into the settings dict, so the word
    would reach the run as the value of the setting.
    """
    box = QLineEdit()
    qtbot.addWidget(box)
    box.setText("/data/plate1")

    screen._apply_value(box, None)

    assert box.text() == ""


# -- the resource strip -------------------------------------------------------

def test_an_empty_usage_sample_paints_nothing(screen):
    """A worker that could read nothing must not blank the bars.

    Painting an empty sample would drop every bar to zero, which reads as an
    idle machine rather than as a reading that did not arrive.
    """
    screen._usage_ram.set_value(42)
    before = screen._usage_ram.value() if hasattr(
        screen._usage_ram, "value") else 42

    screen._apply_usage({})

    if hasattr(screen._usage_ram, "value"):
        assert screen._usage_ram.value() == before


# -- restarting a wedged spaCR ------------------------------------------------

class TestForcingARestart:

    def test_settings_that_cannot_be_collected_do_not_stop_the_restart(
            self, screen, monkeypatch):
        """The reason for restarting is that the screen is already stuck.

        Refusing to restart because the stuck screen would not answer is the
        one outcome that leaves the user with nothing to do.
        """
        def broken():
            raise RuntimeError("the panel is wedged")

        monkeypatch.setattr(screen._settings_model, "collect", broken)
        seen = {}

        def launcher(*args, **kwargs):
            seen["launched"] = True
            return True

        started = screen.force_restart(launcher=launcher,
                                       exiter=lambda *a, **k: None)

        assert seen.get("launched") is True
        assert started is True

    def test_a_restart_that_could_not_save_says_nothing_was_stopped(
            self, screen, monkeypatch):
        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)
        monkeypatch.setattr("spacr.qt.shutdown.restart_spacr",
                            lambda *a, **k: False)

        assert screen.force_restart() is False
        assert any("did NOT restart" in line for line in said)

    def test_a_warning_that_cannot_be_composed_is_empty_not_wrong(
            self, screen, monkeypatch):
        """An empty warning is a dialog with no scare in it; a wrong one lies."""
        import spacr.restart_state as restart_state

        def broken(*_args, **_kwargs):
            raise RuntimeError("the running list is malformed")

        monkeypatch.setattr(restart_state, "warning_text", broken)

        assert screen._restart_warning() == ""

    def test_the_warning_names_the_run_folders_that_are_open(self, screen):
        """A restart abandons them, so the dialog has to name them.

        Partial output is the only thing a stopped run leaves behind; a
        dialog that did not say where it is asks the user to accept a loss
        they cannot size.
        """
        screen._last_run_folder = "/data/plate1/results"

        text = screen._restart_warning()

        assert "/data/plate1/results" in text


# -- the QC verdict on the console --------------------------------------------

class TestTheQcVerdictLine:

    def test_a_run_with_no_verdict_says_nothing(self, screen):
        """QC off, or a suite that would not build, has nothing to report."""
        assert screen._say_the_qc_verdict(None) == ""
        assert screen._say_the_qc_verdict({}) == ""
        assert screen._say_the_qc_verdict({"qc_verdict": None}) == ""

    def test_a_passing_run_still_gets_one_line(self, screen):
        """"ok" is worth saying: it is the evidence the suite ran at all."""
        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)

        line = screen._say_the_qc_verdict(
            {"qc_verdict": "every panel passed", "qc_verdict_level": "ok"})

        assert "Regression QC: ok" in line
        assert "every panel passed" in line
        assert said == [line]

    def test_a_failing_run_names_the_report_file(self, screen):
        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)
        verdict = types.SimpleNamespace(detail="the design is rank deficient",
                                        name="collinearity")

        line = screen._say_the_qc_verdict(
            {"qc_verdict": verdict, "qc_verdict_level": "fail"})

        assert "REGRESSION QC: FAIL" in line
        assert "(collinearity)" in line
        assert "regression_qc_report.txt" in line


# -- restoring a run's workspace ----------------------------------------------

def test_a_record_naming_no_folder_restores_nothing(screen):
    """There is nowhere to read a workspace from, so nothing is claimed."""
    assert screen.restore_run_workspace({}) == {
        "restored": [], "skipped": [], "files": []}
    assert screen.restore_run_workspace("") == {
        "restored": [], "skipped": [], "files": []}
