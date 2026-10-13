"""A downloaded example configures the module, not just its path.

The point of shipping settings beside data: a user who has to work out which
column holds the labels, what the mask dimensions are and which channels were
measured has done most of the work the example was meant to save. With them
applied, Run is the next action.

Reported 2026-09-01: "whenever a test dataset is downloaded and loaded the
settings that come with the dataset should be implemented so the user can just
press run".
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spacr.qt import settings_pack
from spacr.qt.screens.app_screen import AppScreen


@pytest.fixture(autouse=True)
def _small_form_schema(monkeypatch):
    """Keep file-discovery tests independent of full pipeline defaults."""
    from spacr.qt.screens import settings_model

    monkeypatch.setattr(settings_model, "resolve_default_settings",
                        lambda app_key: {"src": "", "verbose": False})


def _settings_file(folder, name, rows):
    where = Path(folder) / "settings"
    where.mkdir(parents=True, exist_ok=True)
    path = where / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return path


class _Screen:
    """Only what the applier touches."""

    def __init__(self, app_key):
        self.app_key = app_key
        self.applied_with = None
        self._console = type("C", (), {
            "append_notice": lambda *a, **k: None,
            "append_stdout": lambda *a, **k: None,
        })()

    _EXAMPLE_SETTINGS_FILES = AppScreen._EXAMPLE_SETTINGS_FILES
    apply_settings_that_came_with = AppScreen.apply_settings_that_came_with
    # The applier re-homes the publisher's paths before applying them, so a
    # stand-in that is "only what the applier touches" has to carry this too.
    reanchor_example_paths = staticmethod(AppScreen.reanchor_example_paths)

    def apply_settings_dict(self, values):
        self.applied_with = values
        return len(values)


@pytest.mark.parametrize("app, name", [
    ("mask", "gen_mask_settings.csv"),
    ("measure", "measure_crop_settings.csv"),
    ("classify", "classify_settings.csv"),
])
def test_each_module_finds_its_own_file(tmp_path, app, name):
    _settings_file(tmp_path, name, [("src", "/somewhere"), ("verbose", "True")])
    screen = _Screen(app)

    assert screen.apply_settings_that_came_with(tmp_path) == 2
    assert screen.applied_with == {"src": "/somewhere", "verbose": True}


def test_an_older_archives_spelling_still_works(tmp_path):
    """The published sets were written by different runs at different times:
    a mask run saves gen_mask_settings.csv and the older pack shipped
    gen_masks_settings.csv. Trying each is what keeps an older archive
    working rather than silently filling in nothing."""
    _settings_file(tmp_path, "gen_masks_settings.csv", [("verbose", "True")])
    screen = _Screen("mask")
    assert screen.apply_settings_that_came_with(tmp_path) == 1
    assert screen.applied_with == {"verbose": True}


def test_the_preferred_spelling_wins(tmp_path):
    _settings_file(tmp_path, "gen_mask_settings.csv", [("verbose", "False")])
    _settings_file(tmp_path, "gen_masks_settings.csv", [("verbose", "True")])
    screen = _Screen("mask")
    assert screen.apply_settings_that_came_with(tmp_path) == 1
    assert screen.applied_with == {"verbose": False}


def test_a_module_takes_only_its_own_file(tmp_path):
    """The annotate archive ships BOTH an annotate and a classify file; the
    Mask screen must not pick one of them up."""
    _settings_file(tmp_path, "classify_settings.csv", [("verbose", "True")])
    screen = _Screen("mask")
    assert screen.apply_settings_that_came_with(tmp_path) == 0
    assert screen.applied_with is None


def test_no_settings_file_is_not_an_error(tmp_path):
    assert _Screen("measure").apply_settings_that_came_with(tmp_path) == 0


def test_an_unreadable_file_does_not_raise(tmp_path, monkeypatch):
    """A dataset with a broken settings file must still leave the data
    usable -- the download is the expensive part."""
    _settings_file(tmp_path, "classify_settings.csv", [("verbose", "True")])
    screen = _Screen("classify")

    def _explode(*args, **kwargs):
        raise ValueError("that is not a settings file")

    monkeypatch.setattr(settings_pack, "settings_from_pack", _explode)
    assert screen.apply_settings_that_came_with(tmp_path) == 0
    assert screen.applied_with is None


def test_it_goes_through_the_shared_migrating_reader(tmp_path, monkeypatch):
    """Example callers share the reader that explains migration and losses."""
    _settings_file(tmp_path, "gen_masks_settings.csv", [("verbose", "True")])
    original = settings_pack.settings_from_pack
    calls = []

    def record(app_key, pack_dir, **kwargs):
        result = original(app_key, pack_dir, **kwargs)
        calls.append((app_key, Path(pack_dir), result[1].source))
        return result

    monkeypatch.setattr(settings_pack, "settings_from_pack", record)
    screen = _Screen("mask")

    assert screen.apply_settings_that_came_with(tmp_path) == 1
    assert calls == [("mask", tmp_path / "settings", "gen_masks_settings.csv")]
    assert screen.applied_with == {"verbose": True}


@pytest.mark.parametrize("where", [
    "_put_the_example_images_in_place",
    "_put_the_measure_example_in_place",
    "_apply_the_example_settings",
])
def test_every_example_path_calls_it(where):
    """A source check: the helper is inert wherever it is not called, and
    each of these is a different button the user can press."""
    import inspect

    source = inspect.getsource(getattr(AppScreen, where))
    assert "apply_settings_that_came_with" in source, where


def test_the_screen_picker_calls_it_too():
    import inspect

    source = inspect.getsource(AppScreen.load_the_screen_data)
    assert "apply_settings_that_came_with" in source
