"""The classifier checkpoint control describes what the saver really does."""

from __future__ import annotations


def test_intermediate_checkpoint_tooltip_describes_every_supported_mode():
    """Users can disable archives without mistaking best/last for archives."""
    from spacr.settings import tooltips

    text = tooltips["intermedeate_save"]
    lowered = text.casefold()

    assert "no effect" not in lowered
    assert "regardless of this flag" not in lowered
    assert "true or none" in lowered
    assert "false disables" in lowered
    assert "custom thresholds" in lowered
    assert "best-model" in lowered and "last-model" in lowered
    for threshold in ("0.99", "0.98", "0.95", "0.94"):
        assert threshold in text
