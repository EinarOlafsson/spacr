"""The upgrade dialog answers the one failure it can (instruction 01).

The desktop installers build their environment with ``uv venv``, which
does not seed pip, so an install whose updater still runs ``python -m pip``
fails before it starts. It is a BOOTSTRAP TRAP: the fix cannot arrive by
the route it fixes, so a user who has hit it cannot be updated out of it.
Instruction 01's third step asks the dialog to say the escape itself --
"is the one case where the app knows the exact command that would work".

These test the decision, not the QMessageBox: `the_missing_pip_escape` is
a module-level function precisely so the wording and the guard can be
asserted without a modal dialog nobody is there to answer.
"""

import os
import sys

import pytest

from spacr.qt.app import the_missing_pip_escape


class TestItOnlyAnswersTheFailureItRecognises:

    @pytest.mark.parametrize("output", [
        "ERROR: could not build wheels for numpy",
        "Connection refused",
        "",
        None,
    ])
    def test_every_other_failure_gets_no_guess(self, output):
        """A remedy for a failure this cannot recognise is a wrong remedy."""
        assert the_missing_pip_escape(output) is None

    @pytest.mark.parametrize("said", [
        "No module named pip",
        "no module named pip",
        "/usr/bin/python: No module named pip\n",
        "Traceback...\nModuleNotFoundError: No module named pip",
    ])
    def test_the_missing_pip_failure_is_recognised_however_it_is_spelled(
            self, said):
        assert the_missing_pip_escape(said) is not None


class TestTheCommandItNames:

    def test_it_is_the_uv_form_and_names_this_interpreter(self):
        escape = the_missing_pip_escape("No module named pip")
        assert "pip install --upgrade" in escape
        assert "--python" in escape
        assert sys.executable in escape
        assert escape.rstrip().endswith("spacr")
        assert " -m pip " not in escape, (
            "the escape reaches for pip again -- which is the failure")

    def test_it_prefers_the_uv_the_installer_bootstrapped(self, monkeypatch):
        monkeypatch.setattr("spacr.updater.find_uv",
                            lambda: "/opt/spacr/bootstrap/uv")
        escape = the_missing_pip_escape("No module named pip")
        assert escape.startswith("/opt/spacr/bootstrap/uv ")

    def test_without_one_it_still_names_the_path_the_installers_write(
            self, monkeypatch):
        """A name the user can check beats silence.

        On the affected machine `find_uv` is exactly what the installed
        build is too old to have, so returning nothing here would leave
        the one user who needs this with an exit code.
        """
        monkeypatch.setattr("spacr.updater.find_uv", lambda: None)
        escape = the_missing_pip_escape("No module named pip")
        expected = "uv.exe" if os.name == "nt" else "uv"
        assert "bootstrap" in escape
        assert expected in escape

    def test_a_broken_updater_import_does_not_lose_the_answer(self,
                                                              monkeypatch):
        def _explode():
            raise RuntimeError("no updater here")

        monkeypatch.setattr("spacr.updater.find_uv", _explode)
        escape = the_missing_pip_escape("No module named pip")
        assert escape and "bootstrap" in escape

    def test_a_path_with_a_space_survives_being_pasted(self, monkeypatch):
        monkeypatch.setattr("spacr.updater.find_uv",
                            lambda: "/opt/spa cr/bootstrap/uv")
        escape = the_missing_pip_escape("No module named pip")
        # Quoted one way or the other, but never handed over bare -- a
        # command line the user cannot paste is not an escape.
        assert "'/opt/spa cr/bootstrap/uv'" in escape or \
               '"/opt/spa cr/bootstrap/uv"' in escape
