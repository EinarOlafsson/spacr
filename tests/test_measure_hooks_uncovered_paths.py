"""What a measurement hook's context says about itself when it is printed.

A hook that misbehaves is debugged from the object it was handed, and both
context objects are ``__slots__`` classes with no attribute dictionary: a
default ``<... object at 0x7f...>`` tells the person reading the traceback
neither which field failed nor which object type was being filtered. The
tests below pin what the two contexts print, both when built directly and
when built by the pipeline entry points that hand them to a hook.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import measure_hooks as mh


@pytest.fixture(autouse=True)
def _pristine_registries(monkeypatch):
    """Both registries start and end empty.

    They are module-level process state; a hook leaking out of one test would
    silently change every measurement made by the next one.
    """
    monkeypatch.delenv(mh.HOOKS_ENV_VAR, raising=False)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()


def test_a_preprocessing_context_prints_the_field_channels_and_dimensionality():
    """The three things that identify which call is being looked at."""
    context = mh.PreprocessingContext(
        file_name='plate1_A01_F001', channels=[0, 2],
        settings={'channels': [0, 2]}, volumetric=False)

    assert repr(context) == (
        "PreprocessingContext(file_name='plate1_A01_F001', "
        "channels=(0, 2), volumetric=False)")


def test_a_volumetric_preprocessing_context_says_so_when_printed():
    """A z-stack and a 2-D field must not print identically.

    The hook contract differs between them -- ``(Z, Y, X, C)`` against
    ``(Y, X, C)`` -- so a shape mismatch reported for the wrong one sends the
    reader looking in the wrong place.
    """
    flat = mh.PreprocessingContext(file_name='plate1_A01_F001', channels=[1],
                                   settings={}, volumetric=False)
    stack = mh.PreprocessingContext(file_name='plate1_A01_F001', channels=[1],
                                    settings={}, volumetric=True,
                                    spacing=(2.0, 0.65, 0.65))

    assert 'volumetric=True' in repr(stack)
    assert 'volumetric=False' in repr(flat)
    assert repr(stack) != repr(flat)


def test_the_context_a_preprocessing_hook_receives_prints_its_own_field():
    """Driven through :func:`apply_preprocessing_hooks`, not built by hand.

    The hook prints the object the pipeline actually constructed, which is
    what a hook author sees when they log it.
    """
    seen = []

    def record(arrays, context):
        seen.append(repr(context))
        return arrays

    mh.register_preprocessing_hook(record, name='record')
    arrays = np.zeros((4, 4, 2), dtype=np.uint16)
    context = mh.PreprocessingContext(
        file_name='plate2_B03_F007', channels=(1, 3),
        settings={'channels': [1, 3]})

    out = mh.apply_preprocessing_hooks(arrays, context)

    assert out is arrays
    assert seen == ["PreprocessingContext(file_name='plate2_B03_F007', "
                    "channels=(1, 3), volumetric=False)"]


def test_a_region_context_prints_the_object_type_field_and_mask_shape():
    """A filter is consulted once per object type; the print says which."""
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[0, 0] = 1
    context = mh.RegionContext(object_type='nucleus',
                               file_name='plate1_A01_F001', mask=mask,
                               settings={})

    assert repr(context) == ("RegionContext(object_type='nucleus', "
                             "file_name='plate1_A01_F001', shape=(4, 4))")


def test_the_context_a_region_filter_receives_prints_its_object_type_and_shape():
    """One repr per object type, each naming that type and the 3-D shape.

    :func:`apply_region_filter_hooks` is called once per object type by
    ``_measure_crop_core``; a filter that logs its context must be able to
    tell the cell call from the pathogen call.
    """
    seen = []

    def record(context):
        seen.append(repr(context))
        return np.ones(len(context.labels), dtype=bool)

    mh.register_region_filter_hook(record, name='record')
    mask = np.zeros((2, 4, 4), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[1, 2, 2] = 2

    for object_type in ('cell', 'pathogen'):
        filtered, dropped = mh.apply_region_filter_hooks(
            mask, object_type=object_type, file_name='plate3_C11_F002',
            settings={}, spacing=(2.0, 1.0, 1.0))
        assert filtered is mask
        assert dropped == ()

    assert seen == [
        "RegionContext(object_type='cell', "
        "file_name='plate3_C11_F002', shape=(2, 4, 4))",
        "RegionContext(object_type='pathogen', "
        "file_name='plate3_C11_F002', shape=(2, 4, 4))",
    ]
