"""get_measure_crop_settings must default the voxel-geometry keys.

measure.resolve_measurement_spacing reads voxel_size_z_um / voxel_size_xy_um /
anisotropy directly off the settings dict. Before these defaults existed a
measure run assembled by get_measure_crop_settings raised KeyError on a 3-D
mask -- the one case the keys are for.
"""
import pytest


def test_measure_crop_settings_default_the_voxel_geometry_keys():
    from spacr.settings import get_measure_crop_settings
    out = get_measure_crop_settings({'src': '/tmp/x'})
    for k in ('voxel_size_z_um', 'voxel_size_xy_um', 'anisotropy'):
        assert k in out, f"{k} missing -- a 3-D measure run would KeyError"
        assert out[k] is None, f"{k} must default to None, not a guess"


def test_measure_crop_settings_do_not_override_a_supplied_voxel_size():
    from spacr.settings import get_measure_crop_settings
    out = get_measure_crop_settings(
        {'src': '/tmp/x', 'voxel_size_z_um': 0.5, 'voxel_size_xy_um': 0.1,
         'anisotropy': 5.0})
    assert out['voxel_size_z_um'] == 0.5
    assert out['voxel_size_xy_um'] == 0.1
    assert out['anisotropy'] == 5.0


@pytest.mark.parametrize("key", ["voxel_size_z_um", "voxel_size_xy_um", "anisotropy"])
def test_voxel_tooltips_mention_measure(key):
    """The tooltip must say these now reach Measure, not only segmentation."""
    from spacr.settings import tooltips
    text = tooltips[key].lower()
    assert "measure" in text, f"{key} tooltip does not say it affects Measure"
