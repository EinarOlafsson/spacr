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
control: whatever a panel propagates, some module that reaches it has to be
able to read it. Both previews are covered -- Measure's, where the two known
faults were, and Mask's live preview, which is clean and stays that way.
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


# ---------------------------------------------------------------------------
# The Mask live preview, which serves several modules at once
# ---------------------------------------------------------------------------

#: The modules that reach the live preview, directly or through
#: :mod:`spacr.qt.preview_registry`. A key it propagates has to be read by one
#: of them; `model_name` is read only by `cellpose_masks`, which is why the
#: union rather than Mask alone is the right comparison.
_LIVE_PREVIEW_MODULES = (
    "get_timelapse_settings",
    "get_identify_masks_finetune_default_settings",
    "get_analyze_plaque_settings",
)


def test_the_live_preview_propagates_nothing_unread(qtbot, qt_theme_applied):
    """Every tuned value has a module that reads it back."""
    import spacr.settings as settings_module
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    known = set()
    for name in _LIVE_PREVIEW_MODULES:
        try:
            known |= set(getattr(settings_module, name)(
                {"number_of_organelles": 4}))
        except Exception:                                # noqa: BLE001
            continue

    propagated = set(screen._live_preview.settings_for_propagation())

    unread = sorted(propagated - known)
    assert not unread, (
        "the live preview propagates keys no module it serves reads, so "
        f"tuning them changes nothing: {unread}")
