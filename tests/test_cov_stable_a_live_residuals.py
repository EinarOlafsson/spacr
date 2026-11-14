"""Driven residual paths for instruction 288 stable group A."""

from __future__ import annotations

import sys


def test_checkout_version_fallback_succeeds_and_can_be_unavailable(monkeypatch):
    """Version provenance is useful when present and harmless when absent."""
    import spacr._version as checkout
    import spacr.version as installed
    from spacr import ome_zarr

    monkeypatch.setattr(installed, "__version__", "unknown")
    monkeypatch.setattr(checkout, "__version__", "7.6.5-checkout")
    assert ome_zarr._spacr_version() == "7.6.5-checkout"

    monkeypatch.setitem(sys.modules, "spacr._version", None)
    assert ome_zarr._spacr_version() == "unknown"
