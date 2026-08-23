"""A settings file loaded into a screen used to run with its typos intact.

``spacr.validate.validate_settings`` knows that ``n_job`` is not a setting
and that ``n_jobs`` is, and it says so with the near-miss spelled out. The
CLI asks it and the batch runner asks it. The Qt bridge did not: it resolved
the app key to a pipeline function and handed the settings dict over
unchanged, so a key the panel could never have produced -- one that can only
come from a file the user chose to load -- reached the pipeline, matched
nothing, and did nothing.

The maintainer's own ``crop_measure_settings.csv`` asks for ``n_job``. A
measure run started thirty workers while the file said four, and said
nothing about either number.

The bridge now reports through the run console and runs anyway. Refusing is
the CLI's job: the panel cannot produce an unknown key, so declining to
start would fail a run over a key that is already being ignored safely.
"""
from __future__ import annotations

import pytest

from spacr.qt.bridge import APP_KEY_ATTR, resolve_pipeline_entry


@pytest.fixture
def measure_entry(monkeypatch):
    """The measure entry point, with the real pipeline stubbed out."""
    import spacr.measure

    seen = []
    monkeypatch.setattr(spacr.measure, "measure_crop", seen.append)
    return resolve_pipeline_entry("measure"), seen


def test_an_unknown_setting_is_named_with_its_near_miss(measure_entry, capsys):
    entry, _seen = measure_entry
    entry({"src": "/tmp", "n_job": 4})

    printed = capsys.readouterr().out
    assert "n_job" in printed
    assert "n_jobs" in printed, "the correction is the whole value of the message"


def test_the_run_still_starts(measure_entry, capsys):
    """Reporting may not become refusing: the value is ignored, not fatal."""
    entry, seen = measure_entry
    entry({"src": "/tmp", "n_job": 4, "channels": [0, 1]})

    assert len(seen) == 1
    assert seen[0]["n_job"] == 4, "the settings dict is passed through unedited"
    assert seen[0]["channels"] == [0, 1]


def test_clean_settings_say_nothing_about_unknown_keys(measure_entry, capsys):
    """A correct file must not be nagged at."""
    entry, seen = measure_entry
    entry({"src": "/tmp", "n_jobs": 4})

    printed = capsys.readouterr().out
    assert "is not a spaCR setting" not in printed
    assert len(seen) == 1


def test_the_wrapper_keeps_the_app_key_the_run_registry_reads(measure_entry):
    """_tag still has to see the function, or the registry cannot name the run."""
    entry, _seen = measure_entry
    assert getattr(entry, APP_KEY_ATTR, None) == "measure"


def test_a_broken_validator_does_not_stop_the_run(measure_entry, monkeypatch,
                                                  capsys):
    """Advice that cannot be given is not a reason to refuse to work."""
    import spacr.validate

    def explode(*_args, **_kwargs):
        raise RuntimeError("validator is broken")

    monkeypatch.setattr(spacr.validate, "validate_settings", explode)
    entry, seen = measure_entry
    entry({"src": "/tmp", "n_job": 4})

    assert len(seen) == 1
    assert "validator is broken" not in capsys.readouterr().out


@pytest.mark.parametrize("app_key", ["mask", "measure", "classify_merged",
                                     "map_barcodes", "regression"])
def test_every_core_module_gets_the_same_check(app_key):
    """The wrapper is applied where the apps are resolved, so all of them have it.

    Named individually rather than looped over a registry because these five
    are the modules the check exists for.
    """
    entry = resolve_pipeline_entry(app_key)
    assert entry is not None, f"{app_key} resolved to no pipeline at all"
    assert getattr(entry, APP_KEY_ATTR, None) == app_key
