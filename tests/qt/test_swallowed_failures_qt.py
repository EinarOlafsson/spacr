"""A plate dropped on the queue that is silently not queued.

Companion to ``tests/test_swallowed_failures.py`` for the one Qt swallow
site that costs the user work rather than polish. ``xfail(strict=True)``:
the product code is unchanged, so this fails today and starts passing the
moment the site stops swallowing.

No QApplication is needed — ``PlateQueueDropHandler.apply`` takes a
duck-typed screen.
"""
from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.qt


class _Screen:
    """The two things PlateQueueDropHandler.apply touches on a screen."""

    def __init__(self):
        self.items = []
        self.console_lines = []
        self._console = self

    def add_item(self, module, settings):
        self.items.append((module, settings))

    def append_stdout(self, text):
        self.console_lines.append(str(text))


def _snapshot(folder, name):
    path = folder / name
    path.write_text("Key,Value\nsrc,/data/plate1\nnucleus_channel,0\n")
    return path


@pytest.mark.xfail(strict=True, reason=(
    "spacr/qt/dnd_handlers.py:1213 swallows the second load_settings attempt "
    "and `continue`s, so a plate whose settings snapshot will not parse is "
    "dropped from the queue and the drop still reports success as long as one "
    "other snapshot parsed. Fix: log the skipped snapshot at warning and say "
    "'queued N of M' in the console."))
def test_plate_queue_drop_reports_a_snapshot_it_could_not_read(
        tmp_path, monkeypatch, caplog):
    """Queuing 1 of 2 dropped plates in silence is work the user loses."""
    from spacr import utils
    from spacr.qt.dnd_handlers import PlateQueueDropHandler

    plate = tmp_path / "plate1"
    settings_dir = plate / "settings"
    settings_dir.mkdir(parents=True)
    good = _snapshot(settings_dir, "gen_mask_settings.csv")
    bad = _snapshot(settings_dir, "measure_crop_settings.csv")

    real_load = utils.load_settings

    def refuse(path, *args, **kwargs):
        if str(path) == str(bad):
            raise ValueError("could not parse the settings snapshot")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(utils, "load_settings", refuse)

    screen = _Screen()
    with caplog.at_level(logging.WARNING, logger="spacr.qt.dnd_handlers"):
        PlateQueueDropHandler().apply(plate, screen)

    assert len(screen.items) == 1          # only the readable snapshot queued
    told = caplog.text + "\n".join(screen.console_lines)
    assert bad.name in told, (
        "one of the two dropped snapshots was skipped and neither the log nor "
        "the console said which")
