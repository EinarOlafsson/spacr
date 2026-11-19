"""Items 983/984: any-channel crop rendering with image and outline roles, and presets."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.crop_store import (composite_crop, format_channel_roles,
                              load_channel_presets, parse_channel_roles,
                              save_channel_preset)


def _five_channels():
    pixels = np.zeros((5, 20, 20), np.uint16)
    for c in range(5):
        pixels[c, :, c * 4:(c + 1) * 4] = 1000 * (c + 1)
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    return pixels, mask


def test_five_channels_blend_additively_in_their_colours():
    pixels, _mask = _five_channels()
    roles = [{"channel": 0, "colour": "red", "range": (0, 1000)},
             {"channel": 1, "colour": "green", "range": (0, 2000)},
             {"channel": 2, "colour": "blue", "range": (0, 3000)},
             {"channel": 3, "colour": "magenta", "range": (0, 4000)},
             {"channel": 4, "colour": "#808080", "range": (0, 5000), "gamma": 1.0}]
    rgb = composite_crop(pixels, roles)
    assert rgb.dtype == np.uint8 and rgb.shape == (20, 20, 3)
    assert tuple(rgb[0, 1]) == (255, 0, 0)
    assert tuple(rgb[0, 5]) == (0, 255, 0)
    assert tuple(rgb[0, 9]) == (0, 0, 255)
    assert tuple(rgb[0, 13]) == (255, 0, 255)
    assert tuple(rgb[0, 17]) == (128, 128, 128)
    overlap = composite_crop(pixels, [{"channel": 0, "colour": "red", "range": (0, 1000)},
                                      {"channel": 0, "colour": "green", "range": (0, 1000)}])
    assert tuple(overlap[0, 1]) == (255, 255, 0)


def test_gamma_and_display_range_change_only_the_view():
    pixels, _mask = _five_channels()
    pixels[1, :, 4:8] = 500
    linear = composite_crop(pixels, [{"channel": 1, "colour": "grey", "range": (0, 2000)}])
    bright = composite_crop(pixels, [{"channel": 1, "colour": "grey", "range": (0, 2000),
                                      "gamma": 2.0}])
    assert linear[0, 5, 0] == round(0.25 * 255)
    assert bright[0, 5, 0] == round(0.5 * 255)
    assert pixels[1, 0, 5] == 500


def test_outlines_from_the_mask_a_label_channel_and_a_threshold():
    pixels, mask = _five_channels()
    base = [{"channel": 0, "colour": "grey", "range": (0, 1000)}]
    drawn = composite_crop(pixels, base + [{"channel": "mask", "role": "outline",
                                            "colour": "yellow", "width": 1}], mask=mask)
    edge = mask & ~np.pad(mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                          & mask[1:-1, :-2] & mask[1:-1, 2:], 1)
    assert (drawn[edge] == (255, 255, 0)).all()
    assert not (drawn[~edge] == (255, 255, 0)).all(axis=-1).any()
    by_threshold = composite_crop(pixels, [{"channel": 2, "role": "outline",
                                            "colour": "cyan", "threshold": 2500}])
    assert (by_threshold[:, 8] == (0, 255, 255)).all()
    assert (by_threshold[1:-1, 10] == 0).all()
    assert (by_threshold[[0, -1], 10] == (0, 255, 255)).all()
    with pytest.raises(ValueError, match="needs the crop's mask"):
        composite_crop(pixels, [{"channel": "mask", "role": "outline"}])
    with pytest.raises(ValueError, match="'image' or 'outline'"):
        composite_crop(pixels, [{"channel": 0, "role": "glow"}])


def test_the_one_line_form_round_trips_and_refuses_nonsense():
    text = ("0:image:blue; 1:image:green:range=100-4000:gamma=1.5; "
            "mask:outline:yellow:width=2; 3:outline:magenta:threshold=900")
    roles = parse_channel_roles(text)
    assert roles[1] == {"channel": 1, "role": "image", "colour": "green",
                        "range": (100.0, 4000.0), "gamma": 1.5}
    assert roles[2] == {"channel": "mask", "role": "outline", "colour": "yellow",
                        "width": 2}
    assert parse_channel_roles(format_channel_roles(roles)) == roles
    assert parse_channel_roles("  ") == []
    for bad, message in (("x:image", "must be a number"), ("0:image:mauve", "unknown colour"),
                         ("0:image:red:blur=2", "unknown channel-role option")):
        with pytest.raises(ValueError, match=message):
            parse_channel_roles(bad)


def test_presets_are_saved_per_project(tmp_path):
    roles = parse_channel_roles("0:image:blue; mask:outline:yellow")
    assert load_channel_presets(tmp_path) == {}
    path = save_channel_preset(tmp_path, "nuclei", roles)
    assert path == tmp_path / "settings" / "annotate_channel_presets.json"
    save_channel_preset(tmp_path, "two", roles[:1])
    assert load_channel_presets(tmp_path) == {"nuclei": roles, "two": roles[:1]}
    assert load_channel_presets(tmp_path / "other") == {}
    with pytest.raises(ValueError):
        save_channel_preset(tmp_path, "bad", [{"channel": 0, "role": "glow"}])
