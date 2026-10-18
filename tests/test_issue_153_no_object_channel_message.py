"""GitHub #153: the no-object-channel refusal says exactly what to change.

The reporter ran Mask with every object channel left at None. The refusal
said "every registered object channel is None" and told them to "set at
least one *_channel setting". It now names the channel settings with their
current values and lists the acquired channel indices from ``channels``.
"""
from __future__ import annotations

import pytest

PRIMARY = ("cell_channel", "nucleus_channel", "pathogen_channel", "organelle_channel")


def _channel_problem(settings):
    from spacr.validate import validate_settings

    found = [p for p in validate_settings(settings, "mask")
             if p.setting == "cell_channel" and "no segmentation channel" in p.message]
    assert len(found) == 1
    return found[0]


def test_the_refusal_names_each_setting_and_the_acquired_indices():
    settings = {key: None for key in PRIMARY}
    settings["channels"] = [0, 1, 2, 3]
    problem = _channel_problem(settings)
    for key in PRIMARY:
        assert f"{key}=None" in problem.message
    assert "*_channel" not in problem.fix
    assert "0, 1, 2, 3 (channels)" in problem.fix
    assert "cell_channel=0" in problem.fix


@pytest.mark.parametrize("channels", [None, []])
def test_without_channels_the_fix_still_gives_a_concrete_example(channels):
    settings = {key: None for key in PRIMARY}
    settings["channels"] = channels
    problem = _channel_problem(settings)
    assert "zero-based acquisition-channel index" in problem.fix
    assert "cell_channel=0" in problem.fix


def test_the_example_follows_the_first_acquired_channel():
    settings = {key: None for key in PRIMARY}
    settings["channels"] = [2, 3]
    assert "cell_channel=2" in _channel_problem(settings).fix


def test_one_set_channel_clears_the_refusal():
    from spacr.validate import validate_settings

    settings = {key: None for key in PRIMARY}
    settings.update(channels=[0, 1, 2, 3], nucleus_channel=0)
    assert not [p for p in validate_settings(settings, "mask")
                if "no segmentation channel" in p.message]
