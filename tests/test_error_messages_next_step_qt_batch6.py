"""Mask input and volume-editing errors explain how to continue without changing data."""
from types import SimpleNamespace

import numpy as np
import pytest

from spacr.qt import mask_engine as engine

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("image,mask,advice", [
    (np.ones((2, 3)), np.ones((3, 2)), "same height and width"),
    (np.full((2, 3), np.nan), np.ones((2, 3)), "without NaN or infinite values"),
    (np.ones((2, 3)), np.full((2, 3), 0.5), "zero for background and positive whole numbers"),
])
def test_puncta_input_errors_explain_compatible_data(image, mask, advice):
    """Invalid puncta inputs remain unchanged while the error explains valid input."""
    before_image, before_mask = image.copy(), mask.copy()
    with pytest.raises(ValueError) as caught:
        engine._center_puncta_candidates(image, mask, SimpleNamespace())
    assert advice in str(caught.value)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(mask, before_mask)


def test_puncta_setting_error_lists_valid_ranges():
    """Invalid centre size still fails before detection and states the allowed range."""
    params = SimpleNamespace(puncta_sigmas=(1,), puncta_k=3, puncta_min_corrected=0,
                             puncta_edge_margin=0, puncta_min_distance=1,
                             puncta_center_pixels=82)
    with pytest.raises(ValueError) as caught:
        engine._center_puncta_candidates(np.ones((2, 3)), np.ones((2, 3)), params)
    assert "1 to 81 centre pixels" in str(caught.value)
    assert "integer spacing of at least 1 pixel" in str(caught.value)


@pytest.mark.parametrize("spacing", [(), (0, 1, 1), (1, np.nan, 1)])
def test_volume_spacing_error_explains_stored_axis_order(spacing):
    """Invalid spacing remains rejected before the session owns any mutable mask."""
    image = np.zeros((2, 3, 4))
    mask = np.zeros_like(image, dtype=np.uint16)
    with pytest.raises(ValueError) as caught:
        engine.VolumeMaskSession(image, mask, spacing=spacing)
    assert "one positive spacing value for each spatial axis" in str(caught.value)
    assert "stored axis order" in str(caught.value)
    assert not mask.any()


def test_volume_mask_error_explains_shape_and_label_range():
    """A mismatched mask is rejected with the shape and label requirements."""
    with pytest.raises(ValueError) as caught:
        engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 3), np.uint16))
    assert "same shape as the volume" in str(caught.value)
    assert "integer labels from 0 to 65535" in str(caught.value)


def test_volume_section_error_points_to_the_displayed_range():
    """An out-of-range section remains a read-only error with an actionable range hint."""
    session = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), np.uint16))
    with pytest.raises(ValueError) as caught:
        session._section(session.mask, "Z", 2)
    assert "Select a section within the displayed range" in str(caught.value)
    assert not session.mask.any()


def test_volume_memory_refusal_preserves_mask_and_undo_history(monkeypatch):
    """An edit above the history budget still changes neither labels nor undo state."""
    session = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), np.uint16))
    monkeypatch.setattr(session, "MAX_HISTORY_BYTES", 1)
    with pytest.raises(ValueError) as caught:
        session._commit(np.ones_like(session.mask))
    assert "Apply the change in smaller steps" in str(caught.value)
    assert not session.mask.any()
    assert session._undo == [] and session._redo == []
