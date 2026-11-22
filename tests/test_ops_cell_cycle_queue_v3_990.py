"""Known validated v3 geometry uses fixed context, nucleus dilation and thin final outlines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr import ops_cell_cycle as cc
from spacr.tabular import read_table


class _ContextSource:
    """Explicit synthetic unmasked context with an own nucleus and a measured toy pixel scale."""

    context = True
    pixel_size_um = 1.0

    def __init__(self, images, masks):
        """Keep test arrays by label; no image or mask data is read from disk."""
        self.images, self.masks = images, masks

    def get_raw(self, row):
        """Return the full synthetic field and only the selected nucleus mask."""
        return self.images[row['label']], self.masks[row['label']]


def _arrays(centre=64):
    """A small nucleus with visible neighbouring signal inside and outside radius30."""
    image = np.full((2, 128, 128), 1000, np.uint16)
    image[1] = 2000
    mask = np.zeros((128, 128), bool)
    mask[centre-2:centre+2, centre-2:centre+2] = True
    image[:, centre, min(127, centre+20)] = (12000, 3000)
    image[:, centre, min(127, centre+40)] = (12000, 3000)
    return image, mask


def _queue(labels=(301,)):
    """Return exact synthetic identities for the geometry checks."""
    return pd.DataFrame({'plate': 'P_secret', 'well': 'A1', 'site': 1124,
                         'label': labels, 'model_stage': 'metaphase', 'rank': range(len(labels))})


def _write(tmp_path, source, *, labels=(301,), name='queue', **kwargs):
    """Run the production writer and return its tile pixels, mapping and queue metadata."""
    folder = tmp_path / name
    cc.write_label_queue(_queue(labels), '', str(folder), labels=str(tmp_path/'labels.csv'),
                         crop_source=source, **kwargs)
    mapping = read_table(str(folder/'queue.csv'), canonicalise=False, report=None)
    pixels = {}
    for row in mapping.itertuples():
        with Image.open(row.png_path) as image:
            pixels[row.label] = np.asarray(image)
    return pixels, mapping, json.loads((folder/'queue.json').read_text())


@pytest.mark.parametrize('radius', [0, 7, 30, 43])
def test_fixed96_window_is_independent_of_dilation_radius(tmp_path, radius):
    """Radius changes visible context, while the three panels retain the exact872by288 geometry."""
    image, mask = _arrays()
    pixels, _, meta = _write(tmp_path, _ContextSource({301:image}, {301:mask}), radius=radius)
    tile = pixels[301]
    assert tile.shape == (288, 872, 3)
    assert (tile[:, 288:292] == 90).all() and (tile[:, 580:584] == 90).all()
    assert meta['window'] == 96 and meta['radius'] == radius and meta['scale'] == 3
    assert meta['ranges'] == [[550, 23039], [1084, 5002]]
    assert (tile[278:282, 834:865] == 255).all()


def test_nucleus_disk_keeps_neighbours_inside_and_blacks_outside(tmp_path):
    """The dilation follows the nucleus boundary rather than a circle centred on its centroid."""
    image, mask = _arrays()
    pixels, _, _ = _write(tmp_path, _ContextSource({301:image}, {301:mask}))
    tile = pixels[301]
    y = 48*3+1
    assert tile[y, 68*3+1, 0] > tile[y, 70*3+1, 0]
    assert (tile[y, 88*3+1] == 0).all()
    assert (tile[16*3+1, 16*3+1] == 0).all()
    assert tile[48*3+1, 16*3+1, 0] > 0


def test_outline_is_one_final_pixel_in_all_panels(tmp_path):
    """Upscaling the mask before the inner rim prevents three-pixel-wide yellow outlines."""
    image, mask = _arrays()
    pixels, _, _ = _write(tmp_path, _ContextSource({301:image}, {301:mask}))
    tile = pixels[301]
    for left in (0, 292, 584):
        assert tile[138, left+138].tolist() == [255,255,0]
        assert tile[139, left+139].tolist() != [255,255,0]
        assert tile[149, left+145].tolist() == [255,255,0]
        assert tile[148, left+145].tolist() != [255,255,0]
    composite = tile[:270, 584:]
    yellow = np.all(composite == [255,255,0], axis=2)
    assert not composite[...,0][~yellow].any()


def test_edge_windows_are_excluded_with_exact_included_mapping(tmp_path):
    """An edge-crossing window produces no image and is recorded outside the included mapping."""
    first, first_mask = _arrays()
    edge, edge_mask = _arrays(centre=16)
    pixels, mapping, meta = _write(tmp_path, _ContextSource({301:first,902:edge},
                                   {301:first_mask,902:edge_mask}), labels=(902,301))
    assert list(pixels) == [301] and mapping.label.tolist() == [301]
    assert meta['cells'] == 1 and meta['excluded'] == [{'plate':'P_secret','well':'A1',
        'site':1124,'label':902,'reason':'window_crosses_context_edge'}]


def test_fixed_ranges_preserve_pixels_across_different_queues(tmp_path):
    """Adding another cell cannot change the display of the original nucleus and neighbours."""
    image, mask = _arrays()
    source = _ContextSource({301:image,902:image*3}, {301:mask,902:mask})
    alone, _, _ = _write(tmp_path, source, name='alone')
    together, _, _ = _write(tmp_path, source, labels=(902,301), name='together')
    assert np.array_equal(alone[301], together[301])


def test_masked_context_and_unknown_physical_scale_are_refused(tmp_path):
    """Missing context or calibration must not be replaced by invented pixels or physical scales."""
    image, mask = _arrays()
    source = _ContextSource({301:image}, {301:mask})
    source.context = False
    with pytest.raises(ValueError, match='unmasked'):
        _write(tmp_path, source)
    source.context = True
    source.pixel_size_um = None
    with pytest.raises(ValueError, match='measured pixel_size_um'):
        _write(tmp_path, source)


@pytest.mark.parametrize('centroid', [63.5, 64.5])
def test_half_tie_centroids_round_before_subtracting_odd_crop_origin(tmp_path, centroid):
    """An offset context crop must show the same field window at both banker-rounding half ties."""
    yy, xx = np.indices((128, 128))
    image = np.stack([yy * 127 + xx, xx * 127 + yy]).astype(np.uint16)
    mask = np.zeros((128, 128), bool)
    mask[62:66, 62:66] = True
    queue = _queue()
    queue['cy'] = centroid
    queue['cx'] = centroid
    outputs = []
    for origin in (0, 1):
        rows = queue.assign(bbox_y0=origin, bbox_x0=origin)
        source = _ContextSource({301:image[:, origin:, origin:]},
                                {301:mask[origin:, origin:]})
        folder = tmp_path / str(origin)
        cc.write_label_queue(rows, '', str(folder), labels=str(tmp_path/'unused.csv'),
                             crop_source=source, ranges=[(0, 16384), (0, 16384)])
        mapping = read_table(str(folder/'queue.csv'), canonicalise=False, report=None)
        with Image.open(mapping.iloc[0].png_path) as tile:
            outputs.append(np.array(tile))
    np.testing.assert_array_equal(outputs[0], outputs[1])
