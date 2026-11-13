"""What a search refuses before it takes the machine.

Every search here can run for hours, so each one checks its own arguments
first and says which one is wrong. The alternative -- a step of ``"fast"``
reaching the fit function -- fails somewhere inside UMAP, after the features
have been loaded, with a message about the wrong thing entirely.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.hyperparam import umap_walk_axes

_START = {"n_neighbors": 15, "min_dist": 0.1}


# ---------------------------------------------------------------------------
# One axis of a Walk
# ---------------------------------------------------------------------------


def test_a_step_a_caller_chose_reaches_the_axis_it_names():
    """The panel lets a user set a step per parameter. A step that did not
    reach the axis would leave the walk moving at the shipped default while
    the dialog said otherwise."""
    axes = umap_walk_axes(dict(_START), parameters=("n_neighbors",),
                          steps={"n_neighbors": 4})

    assert [axis.name for axis in axes] == ["n_neighbors"]
    assert axes[0].step == 4


# ---------------------------------------------------------------------------
# What an Activation sweep can be pointed at
# ---------------------------------------------------------------------------

def test_merged_arrays_without_a_mask_plane_fall_back_to_the_crop_tar(
        tmp_path):
    """A merged array from a run that segmented nothing carries no mask plane.

    The pointing game needs one, so those files are skipped -- and the note
    has to say the tar was used instead, because the criterion the user
    selected is quietly unavailable on that source.
    """
    torch = pytest.importorskip("torch")
    import tarfile

    model_path = tmp_path / "model.pth"
    torch.save(torch.nn.Sequential(torch.nn.Flatten(),
                                   torch.nn.Linear(4, 2)), model_path)
    merged = tmp_path / "src" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "field1.npy", np.zeros((8, 8, 2), dtype=np.uint16))
    empty_tar = tmp_path / "crops.tar"
    with tarfile.open(empty_tar, "w"):
        pass

    with pytest.raises(ValueError) as excinfo:
        from spacr.hyperparam import load_activation_data
        load_activation_data({
            "model_path": str(model_path), "src": str(tmp_path / "src"),
            "dataset": str(empty_tar), "channels": [0, 1],
            "image_size": 8, "input_statistics": "symmetric"})

    assert "yielded no images" in str(excinfo.value)


def test_crop_tar_loading_applies_exactly_the_requested_normalization(tmp_path):
    """``none`` keeps raw values while ``symmetric`` transforms the same crop."""
    import tarfile
    from io import BytesIO

    torch = pytest.importorskip("torch")
    from PIL import Image

    model_path = tmp_path / "model.pth"
    torch.save(torch.nn.Sequential(torch.nn.Flatten(),
                                   torch.nn.Linear(3 * 8 * 8, 2)), model_path)

    pixels = np.empty((8, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 64
    pixels[..., 1] = 128
    pixels[..., 2] = 255
    payload = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(payload, format="PNG")
    image_bytes = payload.getvalue()

    dataset = tmp_path / "crops.tar"
    with tarfile.open(dataset, "w") as archive:
        member = tarfile.TarInfo("plate_A1_1.png")
        member.size = len(image_bytes)
        archive.addfile(member, BytesIO(image_bytes))

    from spacr.hyperparam import load_activation_data

    def _load(statistics):
        data = load_activation_data({
            "model_path": str(model_path), "dataset": str(dataset),
            "src": str(tmp_path / "no-merged"), "channels": [1, 2, 3],
            "image_size": 8, "normalize_input": True,
            "input_statistics": statistics,
        }, n_images=1)
        return data.images[0][:, 0, 0].detach().cpu().numpy()

    raw = np.array([64, 128, 255]) / 255.0
    unnormalized = _load("none")
    symmetric = _load("symmetric")

    assert unnormalized == pytest.approx(raw)
    assert symmetric == pytest.approx(2.0 * raw - 1.0, abs=1e-7)
    assert not np.allclose(unnormalized, symmetric)
