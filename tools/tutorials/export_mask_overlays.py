"""Explicit per-array plotting workaround for the recorded four-channel example.

The native plotting loop can select its JSON layout file as an image. This
helper calls the actual spaCR plotting API for each .npy file, leaves every
input and previous failed run intact, and writes new PNGs to a new directory.
It does not repair that loop or certify segmentation accuracy.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


LAYOUT = dict(version=1, intensity_channels=[0, 1, 2, 3],
              mask_plane_order=['cell', 'nucleus', 'pathogen'],
              mask_dims=dict(cell=4, nucleus=5, pathogen=6))


def source_files(source):
    """Accept only the exact documented layout and regular numeric arrays."""
    source = Path(source).resolve()
    marker = source / '.spacr_plane_layout.json'
    if marker.is_symlink() or json.loads(marker.read_text()) != LAYOUT:
        raise ValueError('This helper requires the documented four-channel, three-mask layout')
    files = sorted(source.glob('*.npy'))
    if not files or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Expected regular merged .npy files, not metadata or symbolic links')
    return marker, files


def check_figure(stack, fig):
    """Verify channel intensities/contours and combined-label coverage, not biology."""
    import cv2
    if stack.ndim != 3 or stack.shape[2] != 7 or stack.dtype != np.uint16:
        raise ValueError('Expected seven uint16 planes')
    if len(fig.axes) != 5 or any(len(ax.images) != 1 for ax in fig.axes):
        raise ValueError('Expected four actual channel panels and one combined panel')
    specifications = {0: (5, (0., 0., 1.), 'nucleus'),
                      1: (4, (1., 0., 0.), 'cell'),
                      2: (6, (0., .5019607843137255, 0.), 'pathogen')}
    checked = 0
    for channel, ax in enumerate(fig.axes[:4]):
        raw = stack[..., channel].astype(np.float32)
        lo, hi = np.percentile(raw, (1, 99))
        normalized = np.clip((raw - lo) / (hi - lo + 1e-5), 0, 1)
        expected = np.repeat(normalized[..., None], 3, axis=2)
        title = f'channel {channel}'
        if channel in specifications:
            plane, colour, name = specifications[channel]
            mask = stack[..., plane]
            for label in np.unique(mask[mask > 0]):
                contours, _ = cv2.findContours((mask == label).astype(np.uint8),
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(expected, contours, -1, colour, 3)
            title = f'{name} (channel {channel})'
        actual = np.asarray(ax.images[0].get_array())
        if ax.get_title() != title or actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise ValueError(f'Panel {channel} does not match its source channel and declared mask')
        checked += int(actual.size)
    combined = np.asarray(fig.axes[4].images[0].get_array())
    union = np.any(stack[..., 4:] > 0, axis=2)
    if (fig.axes[4].get_title() != 'combined objects' or combined.shape != (*union.shape, 3)
            or not np.isfinite(combined).all() or np.any(combined < 0) or np.any(combined > 1)
            or not np.array_equal(np.any(combined != 0, axis=2), union)):
        raise ValueError('Combined panel foreground does not match the union of saved object masks')
    return dict(channel_rgb_values_checked=checked, combined_foreground_pixels_checked=int(union.size),
                counts={name: int(np.count_nonzero(np.unique(stack[..., plane])))
                        for name, plane in LAYOUT['mask_dims'].items()},
                segmentation_accuracy_certified=False)


def export(source, destination):
    """Plot every numeric input to a fresh directory and verify source preservation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from spacr.plot import plot_image_mask_overlay
    api_source = Path(inspect.getsourcefile(plot_image_mask_overlay)).resolve()
    api_provenance = dict(path=str(api_source), sha256=hashlib.sha256(api_source.read_bytes()).hexdigest())
    source, destination = Path(source).resolve(), Path(destination).resolve()
    marker, files = source_files(source)
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Keep the output outside the source merged directory')
    if destination.exists():
        raise FileExistsError('Use a new output folder; previous work is never overwritten')
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [marker, *files]}
    destination.mkdir(parents=True)
    reports = []
    try:
        for path in files:
            stack = np.load(path, allow_pickle=False)
            if stack.ndim != 3 or stack.shape[2] != 7 or stack.dtype != np.uint16:
                raise ValueError('Expected seven uint16 planes')
            fig = plot_image_mask_overlay(str(path), channels=[0, 1, 2, 3], cell_channel=1,
                nucleus_channel=0, pathogen_channel=2, organelle_channel=None, figuresize=10,
                percentiles=(1, 99), thickness=3, save_pdf=False, outline_palette='default')
            try:
                check = check_figure(stack, fig)
                output = destination / (path.stem + '.png')
                # A transparent plot inherits the viewer's checkerboard and
                # can make the real white panel titles illegible. Specify the
                # export background, not a bitmap edit or a change to data.
                fig.savefig(output, dpi=100, transparent=False, facecolor='black')
                with Image.open(output) as image:
                    image.load()
                    if min(image.size) < 500:
                        raise ValueError('Saved plot is unexpectedly small')
                    if image.convert('RGBA').getchannel('A').getextrema() != (255, 255):
                        raise ValueError('The saved figure background must be opaque')
                reports.append(dict(source=str(path), output=str(output),
                    png_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), **check))
            finally:
                plt.close(fig)
    finally:
        if any(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in original.items()):
            raise ValueError('An original merged array or its layout changed')
    receipt = dict(accepted=True, scope='Explicit per-array PNG workaround, not the native plotting loop',
                   plotting_api_source=api_provenance,
                   source_hashes=original, reports=reports, original_inputs_unchanged=True,
                   app_source_modified=False, native_plotting_loop_fixed=False,
                   segmentation_accuracy_certified=False, published=False)
    (destination / 'overlay_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args.source, args.destination)
    print(json.dumps(dict(accepted=receipt['accepted'], plots=len(receipt['reports']),
                         native_plotting_loop_fixed=False, segmentation_accuracy_certified=False), indent=2))
