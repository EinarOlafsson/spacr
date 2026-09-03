"""Every value the Measure preview propagates must reach the run.

The preview exists to be tuned and then pressed into the run. A control that
writes a key the run does not read is worse than one that does nothing: the
user watches the preview change, propagates it, and the run quietly uses the
default. Nothing reports it, because a settings dict carrying an extra key is
not an error anywhere.

This has now happened twice in this one dictionary. `png_dims` was discarded
because Measure sets `png_channel_mapping` by default and the resolver
prefers it. `organelle_min_size` was discarded because the name was retired
in favour of `organelle_min_area` and the substitution that renamed it did
not reach a literal inside a dict.

So this is a ratchet over the whole dictionary rather than a test per
control: whatever the panel propagates, Measure has to be able to read it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def panel(qtbot, qt_theme_applied):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    widget = MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_every_propagated_key_is_one_measure_ships(panel):
    """The keys the panel writes are the keys the factory defines."""
    from spacr.settings import get_measure_crop_settings

    # AS MANY ORGANELLES AS THE PANEL OFFERS. Measure ships a slot's keys
    # when the run declares the slot, so comparing against a one-organelle
    # factory would call the panel's own organelleb controls unread when the
    # only thing missing is the count.
    known = set(get_measure_crop_settings(
        {"number_of_organelles": len(panel._mask_dims)}))
    propagated = set(panel.settings_for_propagation())

    unread = sorted(propagated - known)
    assert not unread, (
        "the preview propagates keys Measure does not ship, so tuning them "
        f"and pressing run changes nothing: {unread}")


def test_the_organelle_size_floor_is_the_surviving_name(panel):
    """The specific key that was discarded, named so a rename cannot lose it."""
    propagated = panel.settings_for_propagation()

    assert "organelle_min_area" in propagated
    assert "organelle_min_size" not in propagated


def test_the_size_floors_seed_from_the_same_keys_they_write(panel):
    """Read and write must agree, or the panel forgets what it propagated."""
    written = panel.settings_for_propagation()
    floors = {name: 37 + index
              for index, name in enumerate(panel._min_sizes)}
    seed = {panel._size_floor_key(name): value
            for name, value in floors.items()}

    panel.apply_settings(seed)
    again = panel.settings_for_propagation()

    for name, value in floors.items():
        key = panel._size_floor_key(name)
        assert key in written, f"{key} is not propagated at all"
        assert again[key] == value, f"{key} did not survive the round trip"
