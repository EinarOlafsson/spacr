"""Item 288: a Mask run without preprocessing checks the PSF record first.

``preprocess_generate_masks`` with ``preprocess=False`` reuses the archives a
previous run left in ``masks/``. When PSF or enhancement processing is asked
for now -- or a previous run left a record of having applied it -- the reuse
must be proven to match (:func:`spacr.psf_pipeline.validate_psf_resume`)
before a single object is segmented: archives made under a different PSF
would otherwise be segmented as though they were this run's.

What is pinned: the check is called with the source channels in role order
(nucleus, cell, pathogen, then organelles), each once, and the field stems
the archives carry; and a record that does not match stops the run before
segmentation. Segmentation itself is stubbed -- it is a Cellpose forward
pass, not this module's decision.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _settings(src, **over):
    s = {
        "src": str(src), "metadata_type": "cellvoyager",
        "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": 1,
        "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "batch_size": 10,
        "save": True, "custom_regex": None, "randomize": True,
        "examples_to_plot": 0, "illumination_correction": False,
    }
    s.update(over)
    return s


@pytest.fixture
def segmented(monkeypatch):
    """Stub the segmenters and the disk-heavy collaborators; record calls."""
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    calls = []
    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam",
                        lambda *a, **k: calls.append(a[2]))
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam",
                        lambda *a, **k: calls.append(a[2]))
    for mod, name in ((sio, "_load_and_concatenate_arrays"),
                      (splot, "plot_image_mask_overlay"),
                      (splot, "plot_arrays"), (su, "adjust_cell_masks"),
                      (su, "_pivot_counts_table")):
        monkeypatch.setattr(mod, name, lambda *a, **k: None)
    monkeypatch.setattr(su, "cleanup_pipeline_folders", lambda *a, **k: [])
    return calls


@pytest.fixture
def plate(tmp_path):
    """A v1 plate: two stack fields, their archive, and a PSF record."""
    src = tmp_path / "plate1"
    (src / "stack").mkdir(parents=True)
    (src / "masks").mkdir()
    for name in ("f0", "f1"):
        np.save(src / "stack" / f"{name}.npy", np.zeros((3, 8, 8), np.uint16))
    np.savez(src / "masks" / "batch_0.npz",
             data=np.zeros((2, 8, 8, 3), np.float32),
             filenames=np.array(["f0.npy", "f1.npy"]))
    (src / "psf").mkdir()
    (src / "psf" / "segmentation_application.json").write_text(
        json.dumps({"configuration": {"version": 0}, "complete": True,
                    "fields": [], "archives": {}}))
    return src


def test_the_check_gets_each_channel_once_in_role_order(plate, segmented,
                                                        monkeypatch):
    import spacr.psf_pipeline as psf
    from spacr.core import preprocess_generate_masks

    asked = []

    def check(settings, root, channels, *, expected_fields):
        asked.append((root, channels, sorted(expected_fields)))

    monkeypatch.setattr(psf, "validate_psf_resume", check)
    preprocess_generate_masks(_settings(plate))
    assert len(asked) == 1
    root, channels, fields = asked[0]
    assert os.fspath(root) == str(plate)
    assert channels == [0, 1], "nucleus first, cell next, the repeat dropped"
    assert fields == ["f0", "f1"]
    assert segmented, "a matching record lets segmentation go ahead"


def test_a_record_that_does_not_match_stops_the_run(plate, segmented):
    from spacr.core import preprocess_generate_masks

    with pytest.raises(ValueError) as excinfo:
        preprocess_generate_masks(_settings(plate))
    assert "does not match the saved Mask inputs" in str(excinfo.value)
    assert segmented == []
