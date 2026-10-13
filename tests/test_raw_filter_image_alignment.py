"""Original filter planes must match the field manifest and model canvas."""

from __future__ import annotations

import numpy as np
import pytest

from spacr.object import _raw_filter_images


@pytest.fixture
def field(tmp_path):
    """A smaller original field with a distinct value in every channel."""
    raw = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    raw_dir = tmp_path / "stack"
    raw_dir.mkdir()
    path = raw_dir / "plate1_A01_1.npy"
    np.save(path, raw)
    model_input = np.full((6, 7, 2), 0.25, dtype=np.float32)
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:3, 1:3] = 7
    arguments = {
        "src": str(tmp_path / "masks"),
        "filenames": [path.name],
        "model_inputs": [model_input],
        "masks": [mask],
        "channel": 2,
    }
    return arguments, raw, path


def _assert_valid(arguments, raw):
    """Check placement, channel, dtype and source immutability together."""
    raw_before = raw.copy()
    input_before = arguments["model_inputs"][0].copy()
    mask_before = arguments["masks"][0].copy()
    plane, = _raw_filter_images(**arguments)
    selected = raw if raw.ndim == 2 else raw[..., arguments["channel"]]
    expected = np.zeros(mask_before.shape, dtype=raw.dtype)
    expected[:selected.shape[0], :selected.shape[1]] = selected
    assert plane.dtype == raw.dtype
    np.testing.assert_array_equal(plane, expected)
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(arguments["model_inputs"][0], input_before)
    np.testing.assert_array_equal(arguments["masks"][0], mask_before)


@pytest.mark.parametrize("component", ["filenames", "model_inputs", "masks"])
def test_every_manifest_component_must_have_the_same_length(field, component):
    arguments, raw, path = field
    _assert_valid(arguments, raw)
    before = path.read_bytes()
    invalid = dict(arguments)
    invalid[component] = []

    with pytest.raises(ValueError, match="fields, model inputs and masks must align"):
        _raw_filter_images(**invalid)

    assert path.read_bytes() == before


def test_a_missing_original_is_not_replaced_by_the_normalized_model_input(field):
    arguments, raw, path = field
    _assert_valid(arguments, raw)
    missing = "plate1_A01_missing.npy"

    with pytest.raises(FileNotFoundError) as error:
        _raw_filter_images(**dict(arguments, filenames=[missing]))

    assert missing in str(error.value)
    np.testing.assert_array_equal(np.load(path), raw)


def test_an_own_channel_is_required_even_when_the_raw_file_is_readable(field):
    arguments, raw, _ = field
    _assert_valid(arguments, raw)

    with pytest.raises(ValueError, match="explicit own-channel index"):
        _raw_filter_images(**dict(arguments, channel=None))


@pytest.mark.parametrize("channel", [-1, 3])
def test_an_out_of_range_channel_is_not_wrapped_or_guessed(field, channel):
    arguments, raw, _ = field
    _assert_valid(arguments, raw)

    with pytest.raises(ValueError, match="shape/channel mismatch"):
        _raw_filter_images(**dict(arguments, channel=channel))


@pytest.mark.parametrize("spelling", ["parent", "relative", "absolute"])
def test_raw_filenames_must_be_basenames_even_when_the_path_resolves(field, spelling):
    arguments, raw, path = field
    _assert_valid(arguments, raw)
    filenames = {
        "parent": f"../stack/{path.name}",
        "relative": f"./{path.name}",
        "absolute": str(path),
    }
    before = path.read_bytes()

    with pytest.raises(ValueError, match="field basenames"):
        _raw_filter_images(**dict(arguments, filenames=[filenames[spelling]]))

    assert path.read_bytes() == before


def test_a_bare_original_plane_accepts_only_channel_zero(field):
    arguments, raw, path = field
    bare = raw[..., 2]
    np.save(path, bare)
    _assert_valid(dict(arguments, channel=0), bare)

    with pytest.raises(ValueError, match="shape/channel mismatch"):
        _raw_filter_images(**dict(arguments, channel=1))

    np.testing.assert_array_equal(np.load(path), bare)


@pytest.mark.parametrize("shape", [(20,), (1, 4, 5, 3)])
def test_raw_rank_must_match_the_canvas_with_at_most_one_channel_axis(field, shape):
    arguments, raw, path = field
    _assert_valid(arguments, raw)
    np.save(path, np.ones(shape, dtype=np.uint16))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="shape/channel mismatch"):
        _raw_filter_images(**arguments)

    assert path.read_bytes() == before


@pytest.mark.parametrize("shape", [(7, 5, 3), (4, 8, 3)])
def test_raw_spatial_dimensions_cannot_exceed_the_model_canvas(field, shape):
    arguments, _, path = field
    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    np.save(path, raw)
    height = max(6, shape[0])
    width = max(7, shape[1])
    fitting = dict(arguments,
                   model_inputs=[np.zeros((height, width, 2), dtype=np.float32)],
                   masks=[np.zeros((height, width), dtype=np.uint16)])
    _assert_valid(fitting, raw)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="exceeds segmentation canvas"):
        _raw_filter_images(**arguments)

    assert path.read_bytes() == before


@pytest.mark.parametrize("mask_shape", [(5, 7), (1, 6, 7)])
def test_a_mask_must_match_the_padded_plane_in_shape_and_rank(field, mask_shape):
    arguments, raw, _ = field
    _assert_valid(arguments, raw)

    with pytest.raises(ValueError, match="same shape as mask"):
        _raw_filter_images(**dict(
            arguments, masks=[np.zeros(mask_shape, dtype=np.uint16)]))


def test_raw_fields_follow_manifest_order_and_each_retained_canvas(field):
    arguments, first, first_path = field
    second = np.full((3, 4, 3), 900, dtype=np.uint16)
    second[..., 2] = 123
    second_path = first_path.with_name("plate1_A01_2.npy")
    np.save(second_path, second)
    before = {path.name: path.read_bytes() for path in (first_path, second_path)}

    planes = _raw_filter_images(
        arguments["src"], [second_path.name, first_path.name],
        [np.zeros((5, 6, 2)), arguments["model_inputs"][0]],
        [np.zeros((5, 6), dtype=np.uint16), arguments["masks"][0]], 2,
    )

    expected_second = np.zeros((5, 6), dtype=np.uint16)
    expected_second[:3, :4] = 123
    expected_first = np.zeros((6, 7), dtype=np.uint16)
    expected_first[:4, :5] = first[..., 2]
    assert len(planes) == 2
    np.testing.assert_array_equal(planes[0], expected_second)
    np.testing.assert_array_equal(planes[1], expected_first)
    assert {path.name: path.read_bytes() for path in (first_path, second_path)} == before
