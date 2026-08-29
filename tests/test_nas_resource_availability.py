"""Bounded availability contract for the optional real-data NAS lane."""
from __future__ import annotations

import pytest

from tests.resource_capabilities import paths_available

pytestmark = pytest.mark.nas

_PLATE1_REQUIREMENTS = (
    ("/nas_mnt/data/sequencing/plate1/orig/orig", "dir"),
    (
        "/nas_mnt/data/sequencing/settings/"
        "preprocess_generate_masks_settings.csv",
        "file",
    ),
    (
        "/nas_mnt/data/sequencing/settings/measure_crop_settings.csv",
        "file",
    ),
)


def test_the_declared_nas_inputs_are_reachable():
    """Pass on the NAS host and skip within five seconds everywhere else."""
    available = paths_available(_PLATE1_REQUIREMENTS, timeout=5.0)
    if not available:
        pytest.skip("NAS plate1 dataset unavailable")
    assert available is True
