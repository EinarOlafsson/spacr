"""`pip install --upgrade spacr` must never replace a working tree.

REPORTED 2026-08-18: an OLS run's console carried

    Uninstalling spacr-1.5.0.4:
      Successfully uninstalled spacr-1.5.0.4
    Successfully installed spacr-1.5.0.4

spaCR reinstalled ITSELF mid-run. The update check runs at startup
(`app.py` connects `update_check_requested`), and `run_pip_upgrade` shells out
to `pip install --upgrade spacr` -- which uninstalls whatever is present and
installs from the index, INCLUDING when what is present is an editable install
pointing at the developer's checkout. The source stops being what runs, nothing
says so, and every change made afterwards has no visible effect.
"""

import os
import sys

import spacr



def test_this_checkout_is_recognised_as_editable():
    from spacr.updater import editable_install_location

    where = editable_install_location()
    assert where, "a checkout must be recognised, or the guard cannot fire"
    assert os.path.isdir(where)
    assert os.path.samefile(
        where, os.path.dirname(os.path.dirname(os.path.abspath(spacr.__file__))))


def test_it_is_found_from_inside_the_checkout_too(monkeypatch, tmp_path):
    """THE CASE THAT BROKE THE FIRST ATTEMPT.

    With the current directory inside the checkout, `importlib.metadata`
    resolves `spacr` to the SOURCE TREE's own metadata, which carries no
    `direct_url.json` -- so a guard reading only that returned None in exactly
    the situation it exists for. A developer is in that directory most of the
    time.
    """
    from spacr.updater import editable_install_location

    monkeypatch.chdir(os.path.dirname(os.path.dirname(
        os.path.abspath(spacr.__file__))))
    assert editable_install_location()


def test_the_upgrade_is_refused_and_says_why(monkeypatch):
    """It must not shell out at all -- refusing after running pip is not
    refusing."""
    from spacr import updater

    ran = []
    monkeypatch.setattr(updater, "run_install_command",
                        lambda *a, **k: ran.append(a) or (0, ""))
    code, message = updater.run_pip_upgrade()
    assert ran == [], "pip was invoked over an editable install"
    assert code == 0
    assert "editable" in message
    assert "git pull" in message, "the refusal must name the real remedy"


def test_a_normal_install_still_upgrades(monkeypatch):
    """The guard must not disable the updater for everybody else."""
    from spacr import updater

    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    ran = []
    monkeypatch.setattr(updater, "run_install_command",
                        lambda args, **k: ran.append(args) or (0, "ok"))
    code, message = updater.run_pip_upgrade()
    assert ran, "an ordinary install must still be upgradable"
    assert code == 0 and message == "ok"
