"""Scientific plotting and statistical-annotation helpers."""

from __future__ import annotations
import os, random, cv2, glob, math, torch, itertools

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import scipy.ndimage as ndi
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import imageio.v2 as imageio
from IPython.display import display
from skimage import measure
from skimage.measure import find_contours, label, regionprops
from skimage.transform import resize as sk_resize
import scikit_posthocs as sp
from scipy.stats import chi2_contingency
import tifffile as tiff
from scipy.stats import normaltest, ttest_ind, mannwhitneyu, f_oneway, kruskal, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

from ipywidgets import IntSlider, interact
from IPython.display import Image as ipyimage
from matplotlib_venn import venn2

# Fail-loud accounting: a missing annotation column silently pools every
# condition together, which is far worse than a plot that refuses to render.
from .errors import RunLedger, raise_if_strict
from .image_colors import read_image_rgb, write_image_rgb
from .tiff_io import write_tiff

#: Shipped answers, used when the preference store cannot be reached — a
#: pipeline run from the CLI, a notebook, a machine with no Qt installed.
#: They are the same values the Preferences dialog defaults to, so a headless
#: run and a GUI run that never touched the setting produce the same files.
DEFAULT_FIGURE_FORMAT = "pdf"
DEFAULT_FIGURE_DPI = 300

#: Formats the figure-format preference can hold.
FIGURE_FORMATS = ("png", "pdf")

#: matplotlib's Agg backend refuses a canvas above 2**16 px on either axis and
#: is unusable well before that. `spacrGraph._standerdize_figure_format` forces
#: a **10 inch minimum** square canvas and grows it with the group count, so
#: the DPI the preference offers is not always deliverable: a 50-inch grouped
#: graph at 1200 DPI is 60 000 px square, four gigapixels. `deliverable_dpi`
#: says so instead of letting matplotlib raise -- or worse, letting the file
#: appear at a resolution nobody asked for.
MAX_FIGURE_PX = 2 ** 16 - 1

#: Above this the raster is not a figure any more. 200 megapixels is a
#: 14 000 px square, already past any display or printer.
MAX_FIGURE_MEGAPIXELS = 200


def figure_output_preferences():
    """Return ``(format, dpi)`` from the user's preferences.

    Degrades to :data:`DEFAULT_FIGURE_FORMAT` / :data:`DEFAULT_FIGURE_DPI`
    rather than raising: the preference store is Qt's, and the pipelines that
    call this run headless from the CLI and from notebooks, where importing
    PySide6 to decide a file extension would be absurd.
    """
    try:
        from .qt.preferences import get_figure_format, get_figure_png_dpi
        fmt = str(get_figure_format()).strip().lower()
        dpi = int(get_figure_png_dpi())
    except Exception:
        return DEFAULT_FIGURE_FORMAT, DEFAULT_FIGURE_DPI
    if fmt not in FIGURE_FORMATS:
        fmt = DEFAULT_FIGURE_FORMAT
    if dpi <= 0:
        dpi = DEFAULT_FIGURE_DPI
    return fmt, dpi


def deliverable_dpi(fig, dpi, path=None):
    """The DPI this figure can actually be written at, and a word if it is not
    the one that was asked for.

    A resolution preference is a request, not a guarantee. ``spacrGraph`` pins
    its canvas to at least 10 inches square and grows it with the number of
    groups, so 600 and 1200 DPI are simply not available for a large grouped
    figure -- the raster would be tens of thousands of pixels on a side.

    The old behaviour was to hand the number to matplotlib and find out. This
    returns the DPI that will be used and says, by name, when that is not the
    DPI that was requested. Appearing to accept a setting and then quietly
    delivering another one is the failure this avoids.

    :param fig: the figure about to be written.
    :param dpi: the requested dots per inch.
    :param path: destination, named in the message when there is one.
    :returns: the DPI to pass to ``savefig``.
    """
    try:
        width_in, height_in = (float(v) for v in fig.get_size_inches())
    except Exception:
        return int(dpi)
    longest = max(width_in, height_in, 0.01)
    area = max(width_in * height_in, 0.0001)

    by_edge = MAX_FIGURE_PX / longest
    by_area = ((MAX_FIGURE_MEGAPIXELS * 1_000_000) / area) ** 0.5
    ceiling = int(min(by_edge, by_area))
    if ceiling >= int(dpi):
        return int(dpi)

    ceiling = max(72, ceiling)
    where = f" for {path}" if path else ""
    print(
        f"Figure DPI: {int(dpi)} was requested but this figure is "
        f"{width_in:.0f}x{height_in:.0f} inches, so {int(dpi)} DPI would be "
        f"{int(width_in * dpi)}x{int(height_in * dpi)} pixels. Writing at "
        f"{ceiling} DPI instead{where}. Grouped graphs (spacrGraph) pin the "
        "canvas to at least 10 inches and grow it with the group count, so "
        "the highest DPI settings cannot be delivered for them.")
    return ceiling


def _with_extension(path, fmt):
    """``path`` with its extension replaced by ``fmt``.

    Only a *known figure extension* is replaced. ``os.path.splitext`` on its
    own would turn ``plate_2.5_umap`` into ``plate_2.pdf`` -- and
    ``plate_2.6_umap`` into the same name, which is two figures overwriting
    one file.
    """
    text = str(path)
    stem, extension = os.path.splitext(text)
    known = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff", ".eps"}
    if extension.lower() in known:
        return f"{stem}.{fmt}"
    return f"{text}.{fmt}"


def save_figure(fig, path, *, fmt=None, dpi=None, close=False, **kwargs):
    """Write ``fig`` to ``path``, honouring the figure preferences.

    The single place a spaCR figure the user keeps gets written. Before this
    existed there were sixty-odd ``savefig`` calls, each with its own
    hard-coded format and DPI, and the "Figure format" and "Resolution"
    preferences reached exactly two of them -- both writing to a temp
    directory. Everything a pipeline saved into its results folder ignored
    both settings entirely.

    Three things are decided here rather than left to the caller or to
    matplotlib.

    **The format follows the preference**, and the file NAME follows the
    format: a PNG written to ``figure.pdf`` is a file no viewer opens. An
    explicit ``fmt=`` still wins, for the few callers that genuinely need one
    particular format.

    **Fonts are embedded as TrueType** (``pdf.fonttype = 42``) for the length
    of the save. matplotlib's default is Type 3, which draws every glyph as
    its own content stream: the file is still vector, but Illustrator and
    Inkscape open the text as unselectable outlines, and the preference that
    selects this path is labelled "PDF (vector, editable)". Scoped with
    ``rc_context`` so a caller that has deliberately chosen otherwise is not
    changed underneath it.

    **The DPI is passed**, always. A PDF page is resolution-independent, but
    spaCR figures are full of ``imshow`` panels -- cell montages, mask
    overlays, plate heatmaps -- and those are rasterised at the figure's own
    100 DPI unless told otherwise. Without it, 100, 300 and 600 produced
    byte-identical files. What is passed is :func:`deliverable_dpi`, which
    says so out loud when the requested number is not achievable here.

    :param fig: a matplotlib ``Figure``.
    :param path: destination; its extension is corrected to the format.
    :param fmt: force a format, bypassing the preference.
    :param dpi: force a DPI, bypassing the preference.
    :param close: close the figure once written.
    :param kwargs: passed through to ``savefig`` (``bbox_inches`` etc.).
    :returns: the path actually written, as a ``str``.
    """
    preferred_fmt, preferred_dpi = figure_output_preferences()
    chosen_fmt = str(fmt or preferred_fmt).strip().lower().lstrip(".")
    if chosen_fmt not in FIGURE_FORMATS:
        chosen_fmt = preferred_fmt
    chosen_dpi = int(dpi) if dpi else preferred_dpi

    destination = _with_extension(path, chosen_fmt)
    directory = os.path.dirname(str(destination))
    if directory:
        os.makedirs(directory, exist_ok=True)

    kwargs.pop("format", None)
    kwargs.pop("dpi", None)
    from matplotlib import rc_context
    with rc_context({"pdf.fonttype": 42}):
        fig.savefig(destination, format=chosen_fmt,
                    dpi=deliverable_dpi(fig, chosen_dpi, destination),
                    **kwargs)
    if close:
        plt.close(fig)
    return destination


def plot_image_mask_overlay(
    file,
    channels,
    cell_channel,
    nucleus_channel,
    pathogen_channel,
    organelle_channel=None,
    figuresize=10,
    percentiles=(2, 98),
    thickness=3,
    save_pdf=True,
    mode='outlines',
    export_tiffs=False,
    all_on_all=False,
    all_outlines=False,
    filter_dict=None
):
    """Plot image and mask overlays.

    Loads the merged ``.npy`` stack, draws one panel per requested channel
    with the object masks applied as contours or filled labels, and closes
    with a panel showing every object combined.

    :param file: Path to the merged ``.npy`` stack for one field of view.
    :param channels: Indices of the image channels to draw, one panel each.
    :param cell_channel: Intensity channel the cell mask belongs to, or
        ``None`` when there is no cell mask.
    :param nucleus_channel: Intensity channel the nucleus mask belongs to,
        or ``None``.
    :param pathogen_channel: Intensity channel the pathogen mask belongs to,
        or ``None``.
    :param organelle_channel: Intensity channel the organelle mask belongs
        to, or ``None``. Default ``None``.
    :param figuresize: Figure height in inches; the figure is drawn four
        times as wide. Default ``10``.
    :param percentiles: Two-element percentile pair used to normalise each
        channel. Default ``(2, 98)``.
    :param thickness: Contour line width in pixels. Default ``3``.
    :param save_pdf: If True, save the figure into ``results/overlay/``
        two directories above ``file``, in the configured figure format
        rather than always as PDF. Default ``True``.
    :param mode: ``'outlines'`` draws mask contours; any other value
        overlays filled, randomly coloured labels. Default ``'outlines'``.
    :param export_tiffs: If True, also write every stack plane as a
        grayscale TIFF into ``results/<stem>/tiff/`` alongside it. Default
        ``False``.
    :param all_on_all: If True, draw every mask on every channel. Default
        ``False``.
    :param all_outlines: If True, draw every mask on the channels that own
        no mask themselves. Default ``False``.
    :param filter_dict: Optional per-object limits keyed by ``'cell'``,
        ``'nucleus'``, ``'pathogen'`` or ``'organelle'``, each holding
        ``((min_area, max_area), (min_intensity, max_intensity))``; objects
        outside the limits are dropped before plotting.
    :returns: The generated matplotlib ``Figure``.
    """

    def random_color_cmap(n_labels, seed=None):
        """Generate a random-looking but deterministic colormap with a unique seed.

        :param n_labels: How many object colours to draw. Index 0 of the
            returned colormap is forced to black for background, so the map
            holds ``n_labels + 1`` entries; callers here pass
            ``int(outline.max() + 1)`` per object type, or
            ``int(combined_mask.max() + 1)`` for the merged panel. A value
            ``<= 0`` short-circuits to a black-only colormap rather than
            raising.
        :param seed: Seed for a local ``default_rng``; the same seed always
            produces the same hue assignment, which is why each object type
            is given its own fixed seed and so keeps its colours across
            panels. ``None`` draws fresh entropy and colours change per call.
        :returns: A ``ListedColormap`` of vivid, well-separated hues.
        """
        if n_labels <= 0:
            return ListedColormap(np.array([[0, 0, 0]]))

        rng = np.random.default_rng(seed)

        # Spread colors across hue space, then shuffle so different seeds give different maps
        hues = np.linspace(0, 1, n_labels, endpoint=False)
        rng.shuffle(hues)

        # Keep colors vivid and bright so different objects are visually distinct
        sats = rng.uniform(0.70, 1.00, size=n_labels)
        vals = rng.uniform(0.85, 1.00, size=n_labels)

        rand_colors = mpl.colors.hsv_to_rgb(np.column_stack([hues, sats, vals]))
        rand_colors = np.vstack([[0, 0, 0], rand_colors])  # background = black
        return ListedColormap(rand_colors)

    def _plot_merged_plot(
        image,
        outlines,
        outline_colors,
        figuresize,
        thickness,
        percentiles,
        mode='outlines',
        all_on_all=False,
        all_outlines=False,
        channels=None,
        channel_to_outline=None,
        channel_to_label=None,
        save_pdf=True
    ):
        """Plot the merged plot with overlay, image channels, and masks."""

        def _generate_colored_mask(mask, cmap):
            """Generate a colored mask using the given colormap."""
            mask_norm = mask / (mask.max() + 1e-5)
            colored_mask = cmap(mask_norm)
            colored_mask[..., 3] = np.where(mask > 0, 1, 0)
            return colored_mask

        def _overlay_mask(image, mask):
            """Overlay the colored mask onto the original image."""
            combined = np.clip(image * (1 - mask[..., 3:]) + mask[..., :3] * mask[..., 3:], 0, 1)
            return combined

        def _normalize_image(image, percentiles):
            """Normalize the image based on given percentiles."""
            v_min, v_max = np.percentile(image, percentiles)
            image_normalized = np.clip((image - v_min) / (v_max - v_min + 1e-5), 0, 1)
            return image_normalized

        def _generate_contours(mask):
            """Generate contours from the mask using OpenCV."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            return contours

        def _apply_contours(image, mask, color, thickness):
            """Apply contours to the image."""
            unique_labels = np.unique(mask)
            for label in unique_labels:
                if label == 0:
                    continue
                label_mask = (mask == label).astype(np.uint8)
                contours = _generate_contours(label_mask)
                cv2.drawContours(
                    image, contours, -1, mpl.colors.to_rgb(color), thickness
                )
            return image

        num_channels = image.shape[-1]
        fig, ax = plt.subplots(1, num_channels + 1, figsize=(4 * figuresize, figuresize))
        # The grid is always num_channels + 1 wide, so a single channel still
        # yields a 2-axes array -- the old `if num_channels == 1: ax = [ax]`
        # wrapped that array in a list and made ax[0] the array, not an Axes.
        ax = np.atleast_1d(ax).ravel()

        channels_with_outlines = set(channel_to_outline.keys()) if channel_to_outline is not None else set()

        for v in range(num_channels):
            channel_image = image[..., v]
            channel_image_normalized = _normalize_image(channel_image, percentiles)
            channel_image_rgb = np.dstack([channel_image_normalized] * 3)

            current_channel = channels[v]

            if all_on_all:
                for idx, (outline, color) in enumerate(zip(outlines, outline_colors)):
                    if mode == 'outlines':
                        channel_image_rgb = _apply_contours(
                            channel_image_rgb, outline, color, thickness
                        )
                    else:
                        cmap = random_color_cmap(
                            int(outline.max() + 1),
                            seed=1000 + idx
                        )
                        mask = _generate_colored_mask(outline, cmap)
                        channel_image_rgb = _overlay_mask(channel_image_rgb, mask)

            elif current_channel in channels_with_outlines:
                outline_info = channel_to_outline.get(current_channel, None)

                if outline_info is not None:
                    outline = outline_info['mask']
                    color = outline_info['color']
                    cmap_seed = outline_info.get('cmap_seed', current_channel + 1000)

                    if outline is not None:
                        if mode == 'outlines':
                            channel_image_rgb = _apply_contours(
                                channel_image_rgb, outline, color, thickness
                            )
                        else:
                            cmap = random_color_cmap(
                                int(outline.max() + 1),
                                seed=cmap_seed
                            )
                            mask = _generate_colored_mask(outline, cmap)
                            channel_image_rgb = _overlay_mask(channel_image_rgb, mask)

            else:
                if all_outlines:
                    for idx, (outline, color) in enumerate(zip(outlines, outline_colors)):
                        if mode == 'outlines':
                            channel_image_rgb = _apply_contours(
                                channel_image_rgb, outline, color, thickness
                            )
                        else:
                            cmap = random_color_cmap(
                                int(outline.max() + 1),
                                seed=1000 + idx
                            )
                            mask = _generate_colored_mask(outline, cmap)
                            channel_image_rgb = _overlay_mask(channel_image_rgb, mask)

            title = channel_to_label.get(current_channel, f'channel {current_channel}')
            ax[v].imshow(channel_image_rgb)
            ax[v].set_title(title)
            ax[v].axis('off')

        if len(outlines) > 0:
            # Priority order is the order in which outlines were added:
            # cell < nucleus < pathogen < organelle
            # Later objects overwrite earlier ones in overlapping pixels.
            combined_mask = np.zeros_like(outlines[0], dtype=np.int64)
            label_offset = 0

            for outline in outlines:
                outline_int = outline.astype(np.int64)
                object_pixels = outline_int > 0

                if np.any(object_pixels):
                    combined_mask[object_pixels] = outline_int[object_pixels] + label_offset
                    label_offset += int(outline_int.max())

            cmap = random_color_cmap(int(combined_mask.max() + 1), seed=9999)
            mask = _generate_colored_mask(combined_mask, cmap)
            blank_image = np.zeros((*combined_mask.shape, 3))
            filled_image = _overlay_mask(blank_image, mask)

            ax[-1].imshow(filled_image)
            ax[-1].set_title('combined objects')
            ax[-1].axis('off')
        else:
            ax[-1].imshow(np.zeros((*image.shape[:2], 3)))
            ax[-1].set_title('no objects')
            ax[-1].axis('off')

        plt.tight_layout()

        if save_pdf:
            pdf_dir = os.path.join(
                os.path.dirname(os.path.dirname(file)), 'results', 'overlay'
            )
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(
                pdf_dir, os.path.basename(file).replace('.npy', '.pdf')
            )
            pdf_path = save_figure(fig, pdf_path)

        plt.show()
        return fig

    def _save_channels_as_tiff(stack, save_dir, filename):
        """Save each channel in the stack as a grayscale TIFF."""
        os.makedirs(save_dir, exist_ok=True)
        for i in range(stack.shape[-1]):
            channel = stack[..., i]
            tiff_path = os.path.join(save_dir, f"{filename}_channel_{i}.tiff")
            write_tiff(tiff_path, channel.astype(np.uint16))
            print(f"Saved {tiff_path}")

    def _filter_object(mask, intensity_image, min_max_area=(0, 10000000), min_max_intensity=(0, 65000), type_='object'):
        """
        Filter objects in a mask based on their area (size) and mean intensity.

        Args:
            mask (ndarray): The input mask.
            intensity_image (ndarray): The corresponding intensity image.
            min_max_area (tuple): A tuple (min_area, max_area) specifying the minimum and maximum area thresholds.
            min_max_intensity (tuple): A tuple (min_intensity, max_intensity) specifying the minimum and maximum intensity thresholds.

        Returns:
            ndarray: The filtered mask.
        """
        original_dtype = mask.dtype
        mask_int = mask.astype(np.int64)
        intensity_image = intensity_image.astype(np.float64)

        unique_labels = np.unique(mask_int)
        unique_labels = unique_labels[unique_labels != 0]
        num_objects_before = len(unique_labels)

        areas = []
        mean_intensities = []
        labels_to_keep = []

        for label in unique_labels:
            label_mask = (mask_int == label)
            area = np.sum(label_mask)
            mean_intensity = np.mean(intensity_image[label_mask])

            areas.append(area)
            mean_intensities.append(mean_intensity)

            if (min_max_area[0] <= area <= min_max_area[1]) and (min_max_intensity[0] <= mean_intensity <= min_max_intensity[1]):
                labels_to_keep.append(label)

        areas = np.array(areas)
        mean_intensities = np.array(mean_intensities)
        num_objects_after = len(labels_to_keep)

        avg_area_before = areas.mean() if num_objects_before > 0 else 0
        avg_intensity_before = mean_intensities.mean() if num_objects_before > 0 else 0
        areas_after = areas[np.isin(unique_labels, labels_to_keep)]
        mean_intensities_after = mean_intensities[np.isin(unique_labels, labels_to_keep)]
        avg_area_after = areas_after.mean() if num_objects_after > 0 else 0
        avg_intensity_after = mean_intensities_after.mean() if num_objects_after > 0 else 0

        print(f"Before filtering {type_}: {num_objects_before} objects")
        print(f"Average area {type_}: {avg_area_before:.2f} pixels, Average intensity: {avg_intensity_before:.2f}")
        print(f"After filtering {type_}: {num_objects_after} objects")
        print(f"Average area {type_}: {avg_area_after:.2f} pixels, Average intensity: {avg_intensity_after:.2f}")

        mask_filtered = np.zeros_like(mask_int)
        for label in labels_to_keep:
            mask_filtered[mask_int == label] = label
        mask_filtered = mask_filtered.astype(original_dtype)
        return mask_filtered

    stack = np.load(file)

    if export_tiffs:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(file)),
            'results',
            os.path.splitext(os.path.basename(file))[0],
            'tiff'
        )
        filename = os.path.splitext(os.path.basename(file))[0]
        _save_channels_as_tiff(stack, save_dir, filename)

    if stack.dtype in (np.uint16, np.uint8):
        stack = stack.astype(np.float32)

    image = stack[..., channels]
    outlines = []
    outline_colors = []

    object_specs = [
        ('cell', cell_channel, 'red'),
        ('nucleus', nucleus_channel, 'blue'),
        ('pathogen', pathogen_channel, 'green'),
        ('organelle', organelle_channel, 'yellow'),
    ]

    present_objects = [(name, channel, color) for name, channel, color in object_specs if channel is not None]
    n_masks = len(present_objects)
    base_image_planes = stack.shape[2] - n_masks

    channel_to_outline = {}
    channel_to_label = {}

    for mask_offset, (name, channel, color) in enumerate(present_objects):
        mask_dim = base_image_planes + mask_offset
        outline = np.take(stack, mask_dim, axis=2)

        if filter_dict is not None and name in filter_dict:
            intensity = np.take(stack, channel, axis=2)
            outline = _filter_object(
                outline,
                intensity,
                filter_dict[name][0],
                filter_dict[name][1],
                type_=name
            )

        outlines.append(outline)
        outline_colors.append(color)

        channel_to_outline[channel] = {
            'mask': outline,
            'color': color,
            'cmap_seed': 1000 + mask_offset
        }
        channel_to_label[channel] = f'{name} (channel {channel})'

    for ch in channels:
        if ch not in channel_to_label:
            channel_to_label[ch] = f'channel {ch}'

    fig = _plot_merged_plot(
        image=image,
        outlines=outlines,
        outline_colors=outline_colors,
        figuresize=figuresize,
        thickness=thickness,
        percentiles=percentiles,
        mode=mode,
        all_on_all=all_on_all,
        all_outlines=all_outlines,
        channels=channels,
        channel_to_outline=channel_to_outline,
        channel_to_label=channel_to_label,
        save_pdf=save_pdf
    )

    return fig


def plot_image_mask_overlay_magenta_outlines(
    file,
    channels,
    cell_channel,
    nucleus_channel,
    pathogen_channel,
    figuresize=10,
    percentiles=(2, 98),
    thickness=3,
    save_pdf=True,
    mode='outlines',
    export_tiffs=False,
    all_on_all=False,
    all_outlines=False,
    filter_dict=None
):
    """Plot image and mask overlays, outlining each channel's own mask in magenta.

    Variant of :func:`plot_image_mask_overlay` with no ``organelle_channel``:
    when ``mode`` is ``'outlines'`` and ``all_on_all`` is False, the mask
    belonging to a channel is outlined in magenta rather than in that
    object's colour. In every other mode it falls back to filled, randomly
    coloured labels as that function does, but seeded per call rather than
    per object, so the colours differ between runs and between panels.

    :param file: Path to the merged ``.npy`` stack for one field of view.
    :param channels: Indices of the image channels to draw, one panel each.
    :param cell_channel: Intensity channel the cell mask belongs to, or
        ``None`` when there is no cell mask.
    :param nucleus_channel: Intensity channel the nucleus mask belongs to,
        or ``None``.
    :param pathogen_channel: Intensity channel the pathogen mask belongs to,
        or ``None``.
    :param figuresize: Figure height in inches; the figure is drawn four
        times as wide. Default ``10``.
    :param percentiles: Two-element percentile pair used to normalise each
        channel. Default ``(2, 98)``.
    :param thickness: Contour line width in pixels. Default ``3``.
    :param save_pdf: If True, save the figure into ``results/overlay/``
        two directories above ``file``, in the configured figure format
        rather than always as PDF. Default ``True``.
    :param mode: ``'outlines'`` draws mask contours; any other value
        overlays filled, randomly coloured labels. Default ``'outlines'``.
    :param export_tiffs: If True, also write every stack plane as a
        grayscale TIFF into ``results/<stem>/tiff/`` alongside it. Default
        ``False``.
    :param all_on_all: If True, draw every mask on every channel in its own
        colour. Default ``False``.
    :param all_outlines: If True, draw every mask on the channels that own
        no mask themselves. Default ``False``.
    :param filter_dict: Optional per-object limits with a ``'cell'``,
        ``'nucleus'`` and ``'pathogen'`` entry, each holding
        ``((min_area, max_area), (min_intensity, max_intensity))``; objects
        outside the limits are dropped before plotting.
    :returns: The generated matplotlib ``Figure``.
    """

    def random_color_cmap(n_labels, seed=None):
        """Generates a random color map for a given number of labels.

        :param n_labels: How many object colours to draw. Index 0 is
            prepended as black for background, so the map holds
            ``n_labels + 1`` entries; callers here pass
            ``int(outline.max() + 1)`` per object type, or
            ``int(combined_mask.max() + 1)`` for the merged panel. Colours
            are drawn as uniform RGB, so unlike the
            HSV variant in :func:`plot_image_mask_overlay` some come out
            dark and low-contrast against the image.
        :param seed: Seeds the *global* ``numpy.random`` state, not a local
            generator, so passing it also shifts every later ``np.random``
            draw in the process. Callers here pass a fresh
            ``random.randint(0, 100)`` per panel, which is why the same
            object gets a different colour in each panel and each run.
            ``None`` leaves the global state alone.
        :returns: A ``ListedColormap``.
        """
        if seed is not None:
            np.random.seed(seed)
        rand_colors = np.random.rand(n_labels, 3)
        rand_colors = np.vstack([[0, 0, 0], rand_colors])  # Ensure background is black
        cmap = ListedColormap(rand_colors)
        return cmap

    def _plot_merged_plot(
        image,
        outlines,
        outline_colors,
        figuresize,
        thickness,
        percentiles,
        mode='outlines',
        all_on_all=False,
        all_outlines=False,
        channels=None,
        cell_channel=None,
        nucleus_channel=None,
        pathogen_channel=None,
        cell_outlines=None,
        nucleus_outlines=None,
        pathogen_outlines=None,
        save_pdf=True
    ):
        """Plot the merged plot with overlay, image channels, and masks."""

        def _generate_colored_mask(mask, cmap):
            """Generate a colored mask using the given colormap."""
            mask_norm = mask / (mask.max() + 1e-5)  # Normalize mask
            colored_mask = cmap(mask_norm)
            colored_mask[..., 3] = np.where(mask > 0, 1, 0)  # Alpha channel
            return colored_mask

        def _overlay_mask(image, mask):
            """Overlay the colored mask onto the original image."""
            combined = np.clip(image * (1 - mask[..., 3:]) + mask[..., :3] * mask[..., 3:], 0, 1)
            return combined

        def _normalize_image(image, percentiles):
            """Normalize the image based on given percentiles."""
            v_min, v_max = np.percentile(image, percentiles)
            image_normalized = np.clip((image - v_min) / (v_max - v_min + 1e-5), 0, 1)
            return image_normalized

        def _generate_contours(mask):
            """Generate contours from the mask using OpenCV."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            return contours

        def _apply_contours(image, mask, color, thickness):
            """Apply contours to the image."""
            unique_labels = np.unique(mask)
            for label in unique_labels:
                if label == 0:
                    continue  # Skip background
                label_mask = (mask == label).astype(np.uint8)
                contours = _generate_contours(label_mask)
                cv2.drawContours(
                    image, contours, -1, mpl.colors.to_rgb(color), thickness
                )
            return image

        num_channels = image.shape[-1]
        fig, ax = plt.subplots(1, num_channels + 1, figsize=(4 * figuresize, figuresize))

        # Identify channels without associated outlines
        channels_with_outlines = []
        if cell_channel is not None:
            channels_with_outlines.append(cell_channel)
        if nucleus_channel is not None:
            channels_with_outlines.append(nucleus_channel)
        if pathogen_channel is not None:
            channels_with_outlines.append(pathogen_channel)

        for v in range(num_channels):
            channel_image = image[..., v]
            channel_image_normalized = _normalize_image(channel_image, percentiles)
            channel_image_rgb = np.dstack([channel_image_normalized] * 3)

            current_channel = channels[v]

            if all_on_all:
                # Apply all outlines to all channels
                for outline, color in zip(outlines, outline_colors):
                    if mode == 'outlines':
                        channel_image_rgb = _apply_contours(
                            channel_image_rgb, outline, color, thickness
                        )
                    else:
                        cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                        mask = _generate_colored_mask(outline, cmap)
                        channel_image_rgb = _overlay_mask(channel_image_rgb, mask)
            elif current_channel in channels_with_outlines:
                # Apply only the relevant outline to each channel
                outline = None
                color = None

                if current_channel == cell_channel and cell_outlines is not None:
                    outline = cell_outlines
                elif current_channel == nucleus_channel and nucleus_outlines is not None:
                    outline = nucleus_outlines
                elif current_channel == pathogen_channel and pathogen_outlines is not None:
                    outline = pathogen_outlines

                if outline is not None:
                    if mode == 'outlines':
                        # Use magenta color when all_on_all=False
                        channel_image_rgb = _apply_contours(
                            channel_image_rgb, outline, '#FF00FF', thickness
                        )
                    else:
                        cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                        mask = _generate_colored_mask(outline, cmap)
                        channel_image_rgb = _overlay_mask(channel_image_rgb, mask)
            else:
                # Channel without associated outlines
                if all_outlines:
                    # Apply all outlines with specified colors. The colours must
                    # come from outline_colors (as the all_on_all branch above
                    # does); a hard-coded list mislabels the objects and silently
                    # truncates the zip when a fourth object type is present.
                    for outline, color in zip(outlines, outline_colors):
                        if mode == 'outlines':
                            channel_image_rgb = _apply_contours(
                                channel_image_rgb, outline, color, thickness
                            )
                        else:
                            cmap = random_color_cmap(int(outline.max() + 1), random.randint(0, 100))
                            mask = _generate_colored_mask(outline, cmap)
                            channel_image_rgb = _overlay_mask(channel_image_rgb, mask)

            ax[v].imshow(channel_image_rgb)
            ax[v].set_title(f'Image - Channel {current_channel}')

        # Create an image combining all objects filled with colors.
        # outlines is empty when no object channel was supplied, and outlines[0]
        # then raises IndexError instead of drawing an empty panel.
        if len(outlines) > 0:
            combined_mask = np.zeros_like(outlines[0])
            for outline in outlines:
                combined_mask = np.maximum(combined_mask, outline)

            cmap = random_color_cmap(int(combined_mask.max() + 1), random.randint(0, 100))
            mask = _generate_colored_mask(combined_mask, cmap)
            blank_image = np.zeros((*combined_mask.shape, 3))
            filled_image = _overlay_mask(blank_image, mask)

            ax[-1].imshow(filled_image)
            ax[-1].set_title('Combined Objects Image')
        else:
            ax[-1].imshow(np.zeros((*image.shape[:2], 3)))
            ax[-1].set_title('no objects')

        plt.tight_layout()

        # Save the figure as a PDF
        if save_pdf:
            pdf_dir = os.path.join(
                os.path.dirname(os.path.dirname(file)), 'results', 'overlay'
            )
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(
                pdf_dir, os.path.basename(file).replace('.npy', '.pdf')
            )
            pdf_path = save_figure(fig, pdf_path)

        plt.show()
        return fig

    def _save_channels_as_tiff(stack, save_dir, filename):
        """Save each channel in the stack as a grayscale TIFF."""
        os.makedirs(save_dir, exist_ok=True)
        for i in range(stack.shape[-1]):
            channel = stack[..., i]
            tiff_path = os.path.join(save_dir, f"{filename}_channel_{i}.tiff")
            write_tiff(tiff_path, channel.astype(np.uint16))
            print(f"Saved {tiff_path}")

    def _filter_object(mask, intensity_image, min_max_area=(0, 10000000), min_max_intensity=(0, 65000), type_='object'):
        """
        Filter objects in a mask based on their area (size) and mean intensity.

        Args:
            mask (ndarray): The input mask.
            intensity_image (ndarray): The corresponding intensity image.
            min_max_area (tuple): A tuple (min_area, max_area) specifying the minimum and maximum area thresholds.
            min_max_intensity (tuple): A tuple (min_intensity, max_intensity) specifying the minimum and maximum intensity thresholds.

        Returns:
            ndarray: The filtered mask.
        """
        original_dtype = mask.dtype
        mask_int = mask.astype(np.int64)
        intensity_image = intensity_image.astype(np.float64)
        # Compute properties for each labeled object
        unique_labels = np.unique(mask_int)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        num_objects_before = len(unique_labels)

        # Initialize lists to store area and intensity for each object
        areas = []
        mean_intensities = []
        labels_to_keep = []

        for label in unique_labels:
            label_mask = (mask_int == label)
            area = np.sum(label_mask)
            mean_intensity = np.mean(intensity_image[label_mask])

            areas.append(area)
            mean_intensities.append(mean_intensity)

            # Check if the object meets both area and intensity criteria
            if (min_max_area[0] <= area <= min_max_area[1]) and (min_max_intensity[0] <= mean_intensity <= min_max_intensity[1]):
                labels_to_keep.append(label)

        # Convert lists to numpy arrays for easier computation
        areas = np.array(areas)
        mean_intensities = np.array(mean_intensities)
        num_objects_after = len(labels_to_keep)
        # Compute average area and intensity before and after filtering
        avg_area_before = areas.mean() if num_objects_before > 0 else 0
        avg_intensity_before = mean_intensities.mean() if num_objects_before > 0 else 0
        areas_after = areas[np.isin(unique_labels, labels_to_keep)]
        mean_intensities_after = mean_intensities[np.isin(unique_labels, labels_to_keep)]
        avg_area_after = areas_after.mean() if num_objects_after > 0 else 0
        avg_intensity_after = mean_intensities_after.mean() if num_objects_after > 0 else 0
        print(f"Before filtering {type_}: {num_objects_before} objects")
        print(f"Average area {type_}: {avg_area_before:.2f} pixels, Average intensity: {avg_intensity_before:.2f}")
        print(f"After filtering {type_}: {num_objects_after} objects")
        print(f"Average area {type_}: {avg_area_after:.2f} pixels, Average intensity: {avg_intensity_after:.2f}")
        mask_filtered = np.zeros_like(mask_int)
        for label in labels_to_keep:
            mask_filtered[mask_int == label] = label
        mask_filtered = mask_filtered.astype(original_dtype)
        return mask_filtered

    stack = np.load(file)

    if export_tiffs:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(file)),
            'results',
            os.path.splitext(os.path.basename(file))[0],
            'tiff'
        )
        filename = os.path.splitext(os.path.basename(file))[0]
        _save_channels_as_tiff(stack, save_dir, filename)

    # Convert to float for normalization and ensure correct handling of arrays
    if stack.dtype in (np.uint16, np.uint8):
        stack = stack.astype(np.float32)

    image = stack[..., channels]
    outlines = []
    outline_colors = []

    # Define variables to hold individual outlines
    cell_outlines = None
    nucleus_outlines = None
    pathogen_outlines = None

    if pathogen_channel is not None:
        pathogen_mask_dim = -1 
        pathogen_outlines = np.take(stack, pathogen_mask_dim, axis=2)
        if not filter_dict is None:
            pathogen_intensity = np.take(stack, pathogen_channel, axis=2)
            pathogen_outlines = _filter_object(pathogen_outlines, pathogen_intensity, filter_dict['pathogen'][0], filter_dict['pathogen'][1], type_='pathogen')
        
        outlines.append(pathogen_outlines)
        outline_colors.append('green')  

    if nucleus_channel is not None:
        nucleus_mask_dim = -2 if pathogen_channel is not None else -1
        nucleus_outlines = np.take(stack, nucleus_mask_dim, axis=2)
        if not filter_dict is None:
            nucleus_intensity = np.take(stack, nucleus_channel, axis=2)
            nucleus_outlines = _filter_object(nucleus_outlines, nucleus_intensity, filter_dict['nucleus'][0], filter_dict['nucleus'][1], type_='nucleus')
        outlines.append(nucleus_outlines)
        outline_colors.append('blue')  

    if cell_channel is not None:
        if nucleus_channel is not None and pathogen_channel is not None:
            cell_mask_dim = -3
        elif nucleus_channel is not None or pathogen_channel is not None:
            cell_mask_dim = -2
        else:
            cell_mask_dim = -1
        cell_outlines = np.take(stack, cell_mask_dim, axis=2)
        if not filter_dict is None:
            cell_intensity = np.take(stack, cell_channel, axis=2)
            cell_outlines = _filter_object(cell_outlines, cell_intensity, filter_dict['cell'][0], filter_dict['cell'][1], type_='cell')
        outlines.append(cell_outlines)
        outline_colors.append('red')

    fig = _plot_merged_plot(
        image=image,
        outlines=outlines,
        outline_colors=outline_colors,
        figuresize=figuresize,
        thickness=thickness,
        percentiles=percentiles,  # Pass percentiles to the plotting function
        mode=mode,
        all_on_all=all_on_all,
        all_outlines=all_outlines,
        channels=channels,
        cell_channel=cell_channel,
        nucleus_channel=nucleus_channel,
        pathogen_channel=pathogen_channel,
        cell_outlines=cell_outlines,
        nucleus_outlines=nucleus_outlines,
        pathogen_outlines=pathogen_outlines,
        save_pdf=save_pdf
    )

    return fig

def plot_cellpose4_output(batch, masks, flows, cmap='inferno', figuresize=10, nr=1, print_object_number=True):
    """Display per-channel images, label mask and flow field for Cellpose v4 outputs.

    :param batch: Image batch of shape ``(N, H, W, C)``.
    :param masks: Label masks, one per image.
    :param flows: Flow arrays, one per image.
    :param cmap: Colormap for image channels. Default ``'inferno'``.
    :param figuresize: Base figure size. Default ``10``.
    :param nr: Maximum number of images to plot. Default ``1``.
    :param print_object_number: If True, annotate each object with its
        label ID. Default ``True``.
    :returns: None
    """
    
    from .utils import _generate_mask_random_cmap
    
    font = figuresize/2
    index = 0
    
    for image, mask, flow in zip(batch, masks, flows):
        #if print_object_number:
        #    num_objects = mask_object_count(mask)
        #    print(f'Number of objects: {num_objects}')
        random_cmap = _generate_mask_random_cmap(mask)
        
        if index < nr:
            index += 1
            chans = image.shape[-1]
            fig, ax = plt.subplots(1, image.shape[-1] + 2, figsize=(4 * figuresize, figuresize))
            for v in range(0, image.shape[-1]):
                ax[v].imshow(image[..., v], cmap=cmap, interpolation='nearest')
                ax[v].set_title('Image - Channel'+str(v))
            ax[chans].imshow(mask, cmap=random_cmap, interpolation='nearest')
            ax[chans].set_title('Mask')
            if print_object_number:
                # Drop the background label explicitly: [1:] assumes 0 sorts
                # first, so a mask with no background pixel loses a real object.
                unique_objects = np.unique(mask)
                unique_objects = unique_objects[unique_objects != 0]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[chans].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')
            ax[chans+1].imshow(flow, cmap='viridis', interpolation='nearest')
            ax[chans+1].set_title('Flow')
            plt.show()
    return

def plot_organelle_output(img_batch, masks, settings, cmap='inferno', figuresize=10, nr=1, print_object_number=True):
    """Plot organelle segmentation results: raw channel, label mask, morphology-specific diagnostic.

    :param img_batch: Single-channel image batch of shape ``(N, H, W)``.
    :param masks: Label masks, one per image.
    :param settings: Organelle settings dict; ``organelle_morphology``
        and ``organelle_method`` drive the diagnostic panel.
    :param cmap: Colormap for the raw channel. Default ``'inferno'``.
    :param figuresize: Base figure size. Default ``10``.
    :param nr: Maximum number of images to plot. Default ``1``.
    :param print_object_number: If True, annotate each object with its
        label ID. Default ``True``.
    :returns: None
    """
    from .utils import _generate_mask_random_cmap, _organelle_diagnostic
    
    morphology = settings.get('organelle_morphology', 'spots')
    method = settings.get('organelle_method', 'otsu')
    font = figuresize / 2

    for idx in range(min(len(masks), nr, img_batch.shape[0])):
        img = img_batch[idx]
        mask = masks[idx]
        random_cmap = _generate_mask_random_cmap(mask)
        num_objects = len(np.unique(mask)) - (1 if 0 in mask else 0)

        # Generate diagnostic image based on morphology/method
        diag_img, diag_title = _organelle_diagnostic(img, morphology, method, settings)

        n_panels = 3
        fig, ax = plt.subplots(1, n_panels, figsize=(n_panels * figuresize, figuresize))

        # Panel 1: Raw image
        ax[0].imshow(img, cmap=cmap, interpolation='nearest')
        ax[0].set_title(f'Organelle channel ({morphology}/{method})')

        # Panel 2: Label mask
        ax[1].imshow(mask, cmap=random_cmap, interpolation='nearest')
        ax[1].set_title(f'Mask ({num_objects} objects)')
        if print_object_number:
            unique_objects = np.unique(mask)
            unique_objects = unique_objects[unique_objects != 0]
            for obj in unique_objects:
                cy, cx = ndi.center_of_mass(mask == obj)
                ax[1].text(cx, cy, str(obj), color='white', fontsize=font,
                           ha='center', va='center')

        # Panel 3: Diagnostic
        ax[2].imshow(diag_img, cmap='viridis', interpolation='nearest')
        ax[2].set_title(diag_title)

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()

    return

def plot_masks(batch, masks, flows, cmap='inferno', figuresize=10, nr=1, file_type='.npz', print_object_number=True):
    """Display per-channel images, label masks and flow fields for a batch.

    :param batch: Image batch — shape ``(N, H, W, C)`` or a single image
        of shape ``(H, W, C)``.
    :param masks: Label masks, one per image (list or ndarray).
    :param flows: Flow arrays, one per image.
    :param cmap: Colormap for image channels. Default ``'inferno'``.
    :param figuresize: Base figure size. Default ``10``.
    :param nr: Maximum number of images to plot. Default ``1``.
    :param file_type: Source file type of ``flows`` — ``'png'`` selects
        the first element of each flow entry. Default ``'.npz'``.
    :param print_object_number: If True, annotate each object with its
        label ID. Default ``True``.
    :returns: None
    """
    if len(batch.shape) == 3:
        batch = np.expand_dims(batch, axis=0)
    if not isinstance(masks, list):
        # `batch` takes either one image or a stack, so `masks` has to as well
        # (the docstring promises "list or ndarray"). Blindly wrapping made an
        # (N, H, W) stack a single "mask" and imshow died with
        # "Invalid shape (N, H, W) for image data"; a swallowed pytest.skip in
        # tests/test_all_plotting_functions.py hid that for the whole batch path.
        masks = np.asarray(masks)
        masks = [masks] if masks.ndim == 2 else list(masks)
    if not isinstance(flows, list):
        flows = [flows]
    else:
        flows = flows[0]
    if file_type == 'png':
        flows = [f[0] for f in flows]  # assuming this is what you want to do when file_type is 'png'
    font = figuresize/2
    index = 0
    for image, mask, flow in zip(batch, masks, flows):
        unique_labels = np.unique(mask)
        
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        
        if index < nr:
            index += 1
            chans = image.shape[-1]
            fig, ax = plt.subplots(1, image.shape[-1] + 2, figsize=(4 * figuresize, figuresize))
            for v in range(0, image.shape[-1]):
                ax[v].imshow(image[..., v], cmap=cmap) #_imshow
                ax[v].set_title('Image - Channel'+str(v))
            ax[chans].imshow(mask, cmap=random_cmap) #_imshow
            ax[chans].set_title('Mask')
            if print_object_number:
                # Drop the background label explicitly: [1:] assumes 0 sorts
                # first, so a mask with no background pixel loses a real object.
                unique_objects = np.unique(mask)
                unique_objects = unique_objects[unique_objects != 0]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[chans].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')
            ax[chans+1].imshow(flow, cmap='viridis') #_imshow
            ax[chans+1].set_title('Flow')
            plt.show()
    return

def _plot_4D_arrays(src, figuresize=10, cmap='inferno', nr_npz=1, nr=1):
    """
    Plot 4D arrays from .npz files.

    Args:
        src (str): The directory path where the .npz files are located.
        figuresize (int, optional): The size of the figure. Defaults to 10.
        cmap (str, optional): The colormap to use for image visualization. Defaults to 'inferno'.
        nr_npz (int, optional): The number of .npz files to plot. Defaults to 1.
        nr (int, optional): The number of images to plot from each .npz file. Defaults to 1.
    """
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    paths = random.sample(paths, min(nr_npz, len(paths)))

    for path in paths:
        with np.load(path) as data:
            stack = data['data']
        num_images = stack.shape[0]
        num_channels = stack.shape[3]

        for i in range(min(nr, num_images)):
            img = stack[i]

            # Create subplots
            if num_channels == 1:
                fig, axs = plt.subplots(1, 1, figsize=(figuresize, figuresize))
                axs = [axs]  # Make axs a list to use axs[c] later
            else:
                fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * figuresize, figuresize))

            for c in range(num_channels):
                axs[c].imshow(img[:, :, c], cmap=cmap) #_imshow
                axs[c].set_title(f'Channel {c}', size=24)
                axs[c].axis('off')

            fig.tight_layout()
            plt.show()
    return

def generate_mask_random_cmap(mask):
    """Return a random ``ListedColormap`` sized to the labels in ``mask``.

    :param mask: Label mask array (0 = background).
    :returns: Random colormap where index 0 is black and remaining
        entries are random opaque RGBA colours.
    """
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap
    
def random_cmap(num_objects=100):
    """Return a random ``ListedColormap`` with ``num_objects + 1`` colours.

    :param num_objects: Number of foreground colours to generate.
        Default ``100``.
    :returns: Colormap with index 0 = black and remaining indices random
        opaque RGBA colours.
    """
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def _generate_mask_random_cmap(mask):
    """
    Generate a random colormap based on the unique labels in the given mask.

    Parameters:
    mask (ndarray): The mask array containing unique labels.

    Returns:
    ListedColormap: A random colormap generated based on the unique labels in the mask.
    """
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def _get_colours_merged(outline_color):
    
    """
    Get the merged outline colors based on the specified outline color format.

    Parameters:
    outline_color (str): The outline color format. Can be one of 'rgb', 'bgr', 'gbr', or 'rbg'.

    Returns:
    list: A list of merged outline colors based on the specified format.
    """

    if outline_color == 'rgb':
        outline_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rgb
    elif outline_color == 'bgr':
        outline_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # bgr
    elif outline_color == 'gbr':
        outline_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # gbr
    elif outline_color == 'rbg':
        outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
    else:
        outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
    return outline_colors

def plot_images_and_arrays(folders, lower_percentile=1, upper_percentile=99, threshold=1000, extensions=None, overlay=False, max_nr=None, randomize=True):
    """Show side-by-side images and arrays found across multiple folders.

    Each image is either percentile-normalised (values below
    ``threshold``) or shown as a label mask. Optionally overlays object
    outlines from a matching mask file.

    :param folders: Folders to scan for image/array files.
    :param lower_percentile: Lower percentile clip. Default ``1``.
    :param upper_percentile: Upper percentile clip. Default ``99``.
    :param threshold: Values <= threshold are treated as label data
        instead of intensity. Default ``1000``.
    :param extensions: File extensions to include.
        Default ``['.npy', '.tif', '.tiff', '.png']``.
    :param overlay: If True, overlay object outlines. Default ``False``.
    :param max_nr: Maximum number of key groups to plot.
    :param randomize: If True, shuffle key order before plotting.
        Default ``True``.
    :returns: None
    """

    if extensions is None:
        extensions = ['.npy', '.tif', '.tiff', '.png']
    def normalize_image(image, lower=1, upper=99):
        """Percentile-clip and rescale ``image`` to ``[0, 1]``.

        :param image: Any numeric array; normalisation is over the whole
            array at once, so a multi-channel stack is scaled by a single
            pair of percentiles rather than per channel.
        :param lower: Lower percentile, in 0-100. Default ``1``.
        :param upper: Upper percentile, in 0-100, and must be strictly
            greater than ``lower``: when the two percentiles evaluate equal
            (a flat image) the rescale divides by zero and returns ``nan``
            rather than a blank frame, and swapping the two inverts the
            image instead of raising. Default ``99``.
        :returns: A float array clipped to ``[0, 1]``.
        """
        p2, p98 = np.percentile(image, (lower, upper))
        return np.clip((image - p2) / (p98 - p2), 0, 1)

    def find_files(folders, extensions=None):
        """Return a dict keyed by base filename mapping to files with the requested extensions.

        :param folders: Folder paths, each walked recursively. Grouping is
            by basename without extension, and only names found in *every*
            folder survive the final filter — one missing file drops that
            name from the result entirely, and two files with the same
            basename under one folder keep only the last one walked.
        :param extensions: Extensions to accept, matched with
            ``str.endswith`` so they must include the dot and match case.
            ``None`` means ``['.npy', '.tif', '.tiff', '.png']``.
        :returns: ``{basename: {folder: path}}`` for complete groups only.
        """
        if extensions is None:
            extensions = ['.npy', '.tif', '.tiff', '.png']
        file_dict = {}

        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        file_name_wo_ext = os.path.splitext(file)[0]
                        file_path = os.path.join(root, file)
                        if file_name_wo_ext not in file_dict:
                            file_dict[file_name_wo_ext] = {}
                        file_dict[file_name_wo_ext][folder] = file_path

        # Filter out files that don't have paths in all folders
        filtered_dict = {k: v for k, v in file_dict.items() if len(v) == len(folders)}
        return filtered_dict

    def plot_from_file_dict(file_dict, threshold=1000, lower_percentile=1, upper_percentile=99, overlay=False, save=False):
        """Show image/mask pairs collected in ``file_dict`` side-by-side.

        :param file_dict: ``{filename: {folder: path}}`` produced by
            ``find_files``.
        :param threshold: Values above this unique-count are treated as
            intensity images; otherwise as label masks. Default ``1000``.
        :param lower_percentile: Lower percentile clip. Default ``1``.
        :param upper_percentile: Upper percentile clip. Default ``99``.
        :param overlay: If True, overlay mask outlines on the image.
            Default ``False``.
        :param save: If True, save each figure alongside the source
            file. Default ``False``.
        :returns: None
        """

        for filename, folder_paths in file_dict.items():
            image_data = None
            mask_data = None

            for folder, path in folder_paths.items():
                if path.endswith('.npy'):
                    data = np.load(path)
                elif path.endswith('.tif') or path.endswith('.tiff'):
                    data = imageio.imread(path)
                else:
                    continue

                unique_values = np.unique(data)

                if len(unique_values) > threshold:
                    image_data = normalize_image(data, lower_percentile, upper_percentile)
                else:
                    mask_data = data

            if image_data is not None and mask_data is not None:
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                
                # Display the mask with random colormap
                cmap = random_cmap(num_objects=len(np.unique(mask_data)))
                axes[0].imshow(mask_data, cmap=cmap)
                axes[0].set_title(f"{filename} - Mask")
                axes[0].axis('off')

                # Display the normalized image
                axes[1].imshow(image_data, cmap='gray')
                if overlay:
                    labeled_mask = label(mask_data)
                    for region in regionprops(labeled_mask):
                        if region.image.shape[0] >= 2 and region.image.shape[1] >= 2:
                            contours = find_contours(region.image, 0.75)
                            for contour in contours:
                                # Adjust contour coordinates relative to the full image
                                contour[:, 0] += region.bbox[0]
                                contour[:, 1] += region.bbox[1]
                                axes[1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='magenta')

                axes[1].set_title(f"{filename} - Normalized Image")
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()

                if save:
                    save_path = os.path.join(folder, f"{filename}.png")
                    save_path = save_figure(plt.gcf(), save_path)

    if overlay:
        print(f'Overlay will only work on the first two folders in the list')

    file_dict = find_files(folders, extensions)
    items = list(file_dict.items())
    if randomize:
        random.shuffle(items)
    if isinstance(max_nr, (int, float)):
        items = items[:int(max_nr)]
    file_dict = dict(items)

    plot_from_file_dict(file_dict, threshold, lower_percentile, upper_percentile, overlay, save=False)
    return

def _filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, nuclei_limit, pathogen_limit):
    """
    Filters objects in a plot based on various criteria.

    Args:
        stack (numpy.ndarray): The input stack of masks.
        cell_mask_dim (int): The dimension index of the cell mask.
        nucleus_mask_dim (int): The dimension index of the nucleus mask.
        pathogen_mask_dim (int): The dimension index of the pathogen mask.
        mask_dims (list): A list of dimension indices for additional masks.
        filter_min_max (list): A list of minimum and maximum area values for each mask.
        nuclei_limit (bool): Whether to include multinucleated cells.
        pathogen_limit (bool): Whether to include multiinfected cells.

    Returns:
        numpy.ndarray: The filtered stack of masks.
    """
    from .utils import _remove_outside_objects, _remove_multiobject_cells
    
    stack = _remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)

    # filter_min_max is in ROLE order -- [cell, nucleus, pathogen] -- while
    # mask_dims is the COMPACTED list of the planes that exist. Indexing one
    # by the other's position only agrees when every role is enabled.
    #
    # With cell_mask_dim=4, nucleus_mask_dim=None, pathogen_mask_dim=6,
    # mask_dims is [4, 6]: i=0 gave the cell its own range, and i=1 gave the
    # PATHOGEN the nucleus's range. So on any run with a disabled object, one
    # object type was size-filtered by another's limits -- objects removed
    # from the figure that the settings never asked to remove, and objects
    # kept that they did.
    _role_index = {}
    for _position, _dim in enumerate((cell_mask_dim, nucleus_mask_dim,
                                      pathogen_mask_dim)):
        if _dim is not None and _dim not in _role_index:
            _role_index[_dim] = _position

    for mask_dim in mask_dims:
        if filter_min_max is None:
            min_max = [0, 100000000]
        else:
            _position = _role_index.get(mask_dim)
            if _position is None or _position >= len(filter_min_max):
                # A plane that is not one of the three named roles has no
                # declared range. Unfiltered beats borrowing a neighbour's.
                min_max = [0, 100000000]
            else:
                min_max = filter_min_max[_position]

        mask = np.take(stack, mask_dim, axis=2)
        props = measure.regionprops_table(mask, properties=['label', 'area'])
        #props = measure.regionprops_table(mask, intensity_image=intensity_image, properties=['label', 'area', 'mean_intensity'])
        avg_size_before = np.mean(props['area'])
        total_count_before = len(props['label'])

        if not filter_min_max is None:
            valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  
            stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  

        props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
        avg_size_after = np.mean(props_after['area'])
        total_count_after = len(props_after['label'])

        if mask_dim == cell_mask_dim:
            # object_dim must be the dim each flag is named after: the two were
            # swapped, so nuclei_limit=False dropped multi-infected cells and
            # pathogen_limit=False dropped multinucleated ones. The inversion is
            # invisible when both flags are False, which is why it survived.
            if nuclei_limit is False and nucleus_mask_dim is not None:
                stack = _remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
            if pathogen_limit is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
                stack = _remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
            cell_area_before = avg_size_before
            cell_count_before = total_count_before
            cell_area_after = avg_size_after
            cell_count_after = total_count_after
        if mask_dim == nucleus_mask_dim:
            nucleus_area_before = avg_size_before
            nucleus_count_before = total_count_before
            nucleus_area_after = avg_size_after
            nucleus_count_after = total_count_after
        if mask_dim == pathogen_mask_dim:
            pathogen_area_before = avg_size_before
            pathogen_count_before = total_count_before
            pathogen_area_after = avg_size_after
            pathogen_count_after = total_count_after

    if cell_mask_dim is not None:
        print(f'removed {cell_count_before-cell_count_after} cells, cell size from {cell_area_before} to {cell_area_after}')
    if nucleus_mask_dim is not None:
        print(f'removed {nucleus_count_before-nucleus_count_after} nucleus, nucleus size from {nucleus_area_before} to {nucleus_area_after}')
    if pathogen_mask_dim is not None:
        print(f'removed {pathogen_count_before-pathogen_count_after} pathogens, pathogen size from {pathogen_area_before} to {pathogen_area_after}')

    return stack


def plot_arrays(src, figuresize=10, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    """Plot random ``.npy`` / ``.npz`` arrays from ``src``, one channel per subplot.

    :param src: Directory or single ``.npy``/``.npz`` path.
    :param figuresize: Base figure size. Default ``10``.
    :param cmap: Matplotlib colormap. Default ``'inferno'``.
    :param nr: Maximum number of arrays to plot. Default ``1``.
    :param normalize: If True, percentile-normalise before display.
        Default ``True``.
    :param q1: Lower percentile for normalisation. Default ``1``.
    :param q2: Upper percentile for normalisation. Default ``99``.
    :returns: None
    """
    from .utils import normalize_to_dtype

    paths = []

    if src.endswith('.npz') or src.endswith('.npy'):
        paths = [src]
    else:
        paths = [os.path.join(src, f) for f in os.listdir(src) if f.endswith(('.npy', '.npz'))]
        paths = random.sample(paths, min(nr, len(paths)))

    for path in paths:
        print(f'Image path: {path}')
        if path.endswith('.npz'):
            with np.load(path) as data:
                key = list(data.keys())[0]  # assume first key
                img = data[key][0]          # get first image in batch
        else:
            img = np.load(path)

        if normalize:
            if img.ndim == 2:
                # normalize_to_dtype indexes array.shape[2], so a single-plane
                # array raises IndexError; promote it for the call and drop the
                # axis again so the 2-D display path below is unchanged.
                img = normalize_to_dtype(array=img[:, :, np.newaxis], p1=q1, p2=q2)[:, :, 0]
            else:
                img = normalize_to_dtype(array=img, p1=q1, p2=q2)

        if img.ndim == 3:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr, figsize=(figuresize, figuresize))
            if array_nr == 1:
                axs = [axs]  # ensure iterable
            for channel in range(array_nr):
                i = img[:, :, channel]
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
                axs[channel].set_title(f'Channel {channel}', size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap))
            ax.set_title('Channel 0', size=24)
            ax.axis('off')

        fig.tight_layout()
        plt.show()

def _normalize_and_outline(image, remove_background, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
    """
    Normalize and outline an image.

    Args:
        image (ndarray): The input image.
        remove_background (bool): Flag indicating whether to remove the background.
        backgrounds (list): List of background values for each channel.
        normalize (bool): Flag indicating whether to normalize the image.
        normalization_percentiles (list): List of percentiles for normalization.
        overlay (bool): Flag indicating whether to overlay outlines onto the image.
        overlay_chans (list): List of channel indices to overlay.
        mask_dims (list): List of dimensions to use for masking.
        outline_colors (list): List of colors for the outlines.
        outline_thickness (int): Thickness of the outlines.

    Returns:
        tuple: A tuple containing the overlayed image, the original image, and a list of outlines.
    """
    from .utils import normalize_to_dtype, _outline_and_overlay, _gen_rgb_image

    # `image` is the caller's stack and the remove_background branch mutates it
    # in place, so copy the label planes rather than aliasing them.
    raw_masks = {d: image[:, :, d].copy() for d in mask_dims}

    if remove_background:
        backgrounds = np.percentile(image, 1, axis=(0, 1))
        backgrounds = backgrounds[:, np.newaxis, np.newaxis]
        mask = np.zeros_like(image, dtype=bool)
        for chan_index in range(image.shape[-1]):
            if chan_index not in mask_dims:
                mask[:, :, chan_index] = image[:, :, chan_index] < backgrounds[chan_index]
        image[mask] = 0

    if normalize:
        image = normalize_to_dtype(array=image, p1=normalization_percentiles[0], p2=normalization_percentiles[1])
    else:
        image = normalize_to_dtype(array=image, p1=0, p2=100)

    rgb_image = _gen_rgb_image(image, channels=overlay_chans)

    # Label values are categorical, not intensities. Percentile-rescaling them
    # clips the background up to the lowest label (so that object merges into
    # the background) and collapses a single-object mask to a constant image,
    # from which no contour can be found. Restore the raw labels after the RGB
    # build so the overlay image itself is unchanged.
    for d, raw in raw_masks.items():
        image[:, :, d] = raw

    if overlay:
        overlayed_image, outlines, image = _outline_and_overlay(image, rgb_image, mask_dims, outline_colors, outline_thickness)

        return overlayed_image, image, outlines
    else:
        # Remove mask_dims from image
        channels_to_keep = [i for i in range(image.shape[-1]) if i not in mask_dims]
        image = np.take(image, channels_to_keep, axis=-1)
        return [], image, []


def _plot_merged_plot(overlay, image, stack, mask_dims, figuresize, overlayed_image, outlines, cmap, outline_colors, print_object_number, mask_names=None):
    
    """
    Plot the merged plot with overlay, image channels, and masks.

    Args:
        overlay (bool): Flag indicating whether to overlay the image with outlines.
        image (ndarray): Input image array.
        stack (ndarray): Stack of masks.
        mask_dims (list): List of mask dimensions.
        figuresize (float): Size of the figure.
        overlayed_image (ndarray): Overlayed image array.
        outlines (list): List of outlines.
        cmap (str): Colormap for the masks.
        outline_colors (list): List of outline colors.
        print_object_number (bool): Flag indicating whether to print object numbers on the masks.

    Returns:
        fig (Figure): The generated matplotlib figure.
    """
    
    if overlay:
        fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims) + 1, figsize=(4 * figuresize, figuresize))
        ax[0].imshow(overlayed_image) #_imshow
        ax[0].set_title('Overlayed Image')
        ax_index = 1
    else:
        fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims), figsize=(4 * figuresize, figuresize))
        ax_index = 0

    # Normalize and plot each channel with outlines
    for v in range(0, image.shape[-1]):
        channel_image = image[..., v]
        channel_image_normalized = channel_image.astype(float)
        channel_image_normalized -= channel_image_normalized.min()
        channel_image_normalized /= channel_image_normalized.max()
        channel_image_rgb = np.dstack((channel_image_normalized, channel_image_normalized, channel_image_normalized))

        # Apply the outlines onto the RGB image
        for outline, color in zip(outlines, outline_colors):
            for j in np.unique(outline)[1:]:
                channel_image_rgb[outline == j] = mpl.colors.to_rgb(color)

        ax[v + ax_index].imshow(channel_image_rgb)
        # 1-based, human-friendly channel label.
        ax[v + ax_index].set_title(f'Channel {v + 1}')

    for i, mask_dim in enumerate(mask_dims):
        mask = np.take(stack, mask_dim, axis=2)
        random_cmap = _generate_mask_random_cmap(mask)
        ax[i + image.shape[-1] + ax_index].imshow(mask, cmap=random_cmap)
        # Name the mask by its object class + live object count, e.g.
        # "Cell Mask - 200 objects".
        n_obj = int(len(np.unique(mask)) - 1)   # exclude background 0
        cls = (mask_names[i] if mask_names and i < len(mask_names)
               else f'Mask {i + 1}')
        ax[i + image.shape[-1] + ax_index].set_title(
            f'{cls} - {n_obj} object' + ('' if n_obj == 1 else 's'))
        if print_object_number:
            unique_objects = np.unique(mask)[1:]
            for obj in unique_objects:
                cy, cx = ndi.center_of_mass(mask == obj)
                ax[i + image.shape[-1] + ax_index].text(cx, cy, str(obj), color='white', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.show()
    return fig

def plot_merged(src, settings):
    """Show multi-channel image stacks with per-object outlines overlaid.

    :param src: Folder containing ``.npy`` merged stacks.
    :param settings: Plot settings dict — includes channel/mask dims,
        overlay colours, normalisation, filter and object-count keys.
    :returns: The last generated ``Figure`` when ``settings['nr']`` is
        exceeded; otherwise ``None``.
    """
    from .utils import _remove_noninfected

    
    
    outline_colors = _get_colours_merged(settings['outline_color'])
    index = 0
        
    _mask_dim_pairs = [('Cell Mask', settings['cell_mask_dim']),
                       ('Nucleus Mask', settings['nucleus_mask_dim']),
                       ('Pathogen Mask', settings['pathogen_mask_dim'])]
    mask_dims = [dim for _name, dim in _mask_dim_pairs if dim is not None]
    mask_names = [name for name, dim in _mask_dim_pairs if dim is not None]

    if settings['verbose']:
        display(settings)
        
    if settings['pathogen_mask_dim'] is None:
        settings['pathogen_limit'] = True

    # nr=0 takes the else-branch on the very first file, so `fig` has to exist
    # before the loop or `return fig` raises UnboundLocalError.
    fig = None

    for file in os.listdir(src):
        path = os.path.join(src, file)
        stack = np.load(path)
        print(f'Loaded: {path}')
        if settings['pathogen_limit'] > 0:
            if settings['pathogen_mask_dim'] is not None and settings['cell_mask_dim'] is not None:
                stack = _remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['pathogen_limit'] is not True or settings['nuclei_limit'] is not True or settings['filter_min_max'] is not None:
            stack = _filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['nuclei_limit'], settings['pathogen_limit'])

        overlayed_image, image, outlines = _normalize_and_outline(image=stack, 
                                                                  remove_background=settings['remove_background'],
                                                                  normalize=settings['normalize'],
                                                                  normalization_percentiles=settings['normalization_percentiles'],
                                                                  overlay=settings['overlay'],
                                                                  overlay_chans=settings['overlay_chans'],
                                                                  mask_dims=mask_dims,
                                                                  outline_colors=outline_colors,
                                                                  outline_thickness=settings['outline_thickness'])
        if index < settings['nr']:
            index += 1
            fig = _plot_merged_plot(overlay=settings['overlay'],
                                    image=image,
                                    stack=stack,
                                    mask_dims=mask_dims,
                                    figuresize=settings['figuresize'],
                                    overlayed_image=overlayed_image,
                                    outlines=outlines,
                                    cmap=settings['cmap'],
                                    outline_colors=outline_colors,
                                    print_object_number=settings['print_object_number'],
                                    mask_names=mask_names)
        else:
            return fig

def _plot_images_on_grid(image_files, channel_indices, um_per_pixel, scale_bar_length_um=5, fontsize=8, show_filename=True, channel_names=None, plot=False):
    """
    Plots a grid of images with optional scale bar and channel names.

    Args:
        image_files (list): List of image file paths.
        channel_indices (list): List of channel indices to select from the images.
        um_per_pixel (float): Micrometers per pixel.
        scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 5.
        fontsize (int, optional): Font size for the image titles. Defaults to 8.
        show_filename (bool, optional): Whether to show the image file names as titles. Defaults to True.
        channel_names (list, optional): Names for the legend, **in FILE
            order** — entry 0 is the file's red plane, 1 green, 2 blue.
            Defaults to None.
        plot (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The generated figure object.

    .. note::

       ``channel_names`` is in FILE order, not source-channel order, and the
       distinction is the one that caused the crop-colour episode
       (INVARIANTS 13). The legend colours entry *i* with
       ``['red', 'green', 'blue'][i]``, which is right because the image is
       read as RGB — but a caller passing names in SOURCE order
       (``['DAPI', 'GFP', 'RFP']`` for channels 0, 1, 2) would get DAPI
       labelled red while it is rendered blue, and the figure would look
       entirely reasonable.

       To go from the source order a user thinks in to the file order this
       wants, ask ``spacr.crops.resolve_png_channel_mapping(settings)`` and
       read off ``r``, ``g``, ``b``. No live caller passes this argument
       today (the one call site passes ``channel_names=None``), so this is a
       contract being written down before it is relied on rather than a bug
       being fixed.
    """
    print(f'scale bar represents {scale_bar_length_um} um')
    nr_of_images = len(image_files)
    cols = int(np.ceil(np.sqrt(nr_of_images)))
    rows = np.ceil(nr_of_images / cols)
    # squeeze=False keeps the return a 2-D array: a single image gives a 1x1
    # grid, which matplotlib otherwise collapses to a bare Axes with no
    # .flatten().
    fig, axes = plt.subplots(int(rows), int(cols), figsize=(20, 20), facecolor='black', squeeze=False)
    fig.patch.set_facecolor('black')
    axes = axes.flatten()
    # Calculate the scale bar length in pixels
    scale_bar_length_px = int(scale_bar_length_um / um_per_pixel)  # Convert to pixels

    channel_colors = ['red','green','blue']
    for i, image_file in enumerate(image_files):
        img_array = read_image_rgb(image_file, cv2.IMREAD_UNCHANGED)
        # Handle different channel selections
        if channel_indices is not None:
            if len(channel_indices) == 1:  # Single channel (grayscale)
                img_array = img_array[:, :, channel_indices[0]]
                cmap = 'gray'
            elif len(channel_indices) == 2:  # Dual channels
                img_array = np.mean(img_array[:, :, channel_indices], axis=2)
                cmap = 'gray'
            else:  # RGB or more channels
                img_array = img_array[:, :, channel_indices]
                cmap = None
        else:
            cmap = None if img_array.ndim == 3 else 'gray'
        # Normalize based on dtype
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        ax = axes[i]
        ax.imshow(img_array, cmap=cmap)
        ax.axis('off')
        if show_filename:
            ax.set_title(os.path.basename(image_file), color='white', fontsize=fontsize, pad=20)
        # Add scale bar
        ax.plot([10, 10 + scale_bar_length_px], [img_array.shape[0] - 10] * 2, lw=2, color='white')
    # Add channel names at the top if specified
    initial_offset = 0.02  # Starting offset from the left side of the figure
    increment = 0.05  # Fixed increment for each subsequent channel name, adjust based on figure width
    if channel_names:
        current_offset = initial_offset
        for ci, channel_name in enumerate(channel_names):
            color = channel_colors[ci] if ci < len(channel_colors) else 'white'
            fig.text(current_offset, 0.99, channel_name, color=color, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='black', edgecolor='none', pad=3))
            current_offset += increment

    # Pad from the image count, not from a leaked loop variable: the
    # channel_names loop above used to rebind `i`, so the unused cells were
    # blanked starting at the wrong index whenever channel_names was given.
    for j in range(nr_of_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=3)
    if plot:
        plt.show()
    return fig

def _save_scimg_plot(src, nr_imgs=16, channel_indices=None, um_per_pixel=0.1, scale_bar_length_um=10, standardize=True, fontsize=8, show_filename=True, channel_names=None, dpi=300, plot=False, i=1, all_folders=1):

    """
    Save and visualize single-cell images.

    Args:
        src (str): The source directory path.
        nr_imgs (int, optional): The number of images to visualize. Defaults to 16.
        channel_indices (list, optional): List of channel indices to visualize. Defaults to [0,1,2].
        um_per_pixel (float, optional): Micrometers per pixel. Defaults to 0.1.
        scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 10.
        standardize (bool, optional): Whether to standardize the image sizes. Defaults to True.
        fontsize (int, optional): Font size for the filename. Defaults to 8.
        show_filename (bool, optional): Whether to show the filename on the image. Defaults to True.
        channel_names (list, optional): List of channel names. Defaults to None.
        dpi (int, optional): Dots per inch for the saved image. Defaults to 300.
        plot (bool, optional): Whether to plot the images. Defaults to False.

    Returns:
        None
    """
    if channel_indices is None:
        channel_indices = [0,1,2]
    from .io import _save_figure
    
    def _visualize_scimgs(src, channel_indices=None, um_per_pixel=0.1, scale_bar_length_um=10, show_filename=True, standardize=True, nr_imgs=None, fontsize=8, channel_names=None, plot=False):
        """
        Visualize single-cell images.

        Args:
            src (str): The source directory path.
            channel_indices (list, optional): List of channel indices to visualize. Defaults to None.
            um_per_pixel (float, optional): Micrometers per pixel. Defaults to 0.1.
            scale_bar_length_um (float, optional): Length of the scale bar in micrometers. Defaults to 10.
            show_filename (bool, optional): Whether to show the filename on the image. Defaults to True.
            standardize (bool, optional): Whether to standardize the image sizes. Defaults to True.
            nr_imgs (int, optional): The number of images to visualize. Defaults to None.
            fontsize (int, optional): Font size for the filename. Defaults to 8.
            channel_names (list, optional): List of channel names. Defaults to None.
            plot (bool, optional): Whether to plot the images. Defaults to False.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plotted images.
        """
        from .utils import _find_similar_sized_images
        def _generate_filelist(src):
            """
            Generate a list of image files in the specified directory.

            Args:
                src (str): The source directory path.

            Returns:
                list: A list of image file paths.

            """
            files = glob.glob(os.path.join(src, '*'))
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'))]
            return image_files

        def _random_sample(file_list, nr_imgs=None):
            """
            Randomly selects a subset of files from the given file list.

            Args:
                file_list (list): A list of file names.
                nr_imgs (int, optional): The number of files to select. If None, all files are selected. Defaults to None.

            Returns:
                list: A list of randomly selected file names.
            """
            if nr_imgs is not None and nr_imgs < len(file_list):
                random.seed(42)
                file_list = random.sample(file_list, nr_imgs)
            return file_list

        image_files = _generate_filelist(src)

        if standardize:
            image_files = _find_similar_sized_images(image_files)

        if nr_imgs is not None:
            image_files = _random_sample(image_files, nr_imgs)

        fig = _plot_images_on_grid(image_files, channel_indices, um_per_pixel, scale_bar_length_um, fontsize, show_filename, channel_names, plot)

        return fig

    fig = _visualize_scimgs(src, channel_indices, um_per_pixel, scale_bar_length_um, show_filename, standardize, nr_imgs, fontsize, channel_names, plot)
    _save_figure(fig, src, text='all_channels')

    for channel in channel_indices:
        channel_indices=[channel]
        fig = _visualize_scimgs(src, channel_indices, um_per_pixel, scale_bar_length_um, show_filename, standardize, nr_imgs, fontsize, channel_names=None, plot=plot)
        _save_figure(fig, src, text=f'channel_{channel}')

    return

def _plot_cropped_arrays(stack, filename, figuresize=10, cmap='inferno', threshold=500):
    """
    Plot cropped arrays.

    Args:
        stack (ndarray): The array to be plotted, 2D (one panel) or 3D with
            the channels last (one panel per ``stack.shape[2]``). A 1D array
            matches neither branch and raises ``UnboundLocalError`` on the
            return rather than being rejected up front.
        filename (str): Accepted and ignored -- the only reference to it is a
            commented-out print, so it never reaches the figure or a file.
        figuresize (int, optional): Both width and height of the figure, in
            inches; the multi-channel case does not widen it per channel, so
            panels get thinner as channels are added. Defaults to 10.
        cmap (str, optional): Name resolved with ``plt.get_cmap`` and used
            only for the planes treated as intensity images. Defaults to
            'inferno'.
        threshold (int, optional): A plane with this many distinct values or
            fewer is drawn as a label mask with a random colormap and an
            object count in its title. Defaults to 500.

    Returns:
        Figure: The figure that was drawn. The 2D case also calls
        ``plt.show()`` before returning; the multi-channel case does not.
    """
    #start = time.time()
    dim = stack.shape
    
    def plot_single_array(array, ax, title, chosen_cmap):
        """Render one channel from ``stack`` onto ``ax``. No colorbar is drawn.

        Args:
            array (ndarray): One 2D plane of ``stack``. Its count of distinct
                values, not its dtype, is what decides whether it is treated
                as an intensity image or as a label mask -- so a uint8 plane,
                which can hold at most 256 distinct values, is always taken
                for a mask under the default ``threshold`` of 500.
            ax (matplotlib.axes.Axes): Axes drawn on in place; its frame and
                ticks are switched off, the title is fixed at size 18, and
                nothing is returned.
            title (str): Panel title. When the plane is treated as a mask,
                the object count is appended as ``", N (obj.)"``. That count
                is the number of distinct non-zero values, so the background
                value 0 is never counted, but a negative value is counted as
                an object.
            chosen_cmap (Colormap): Colormap for the intensity case only.
                It is discarded when the plane has no more than
                ``threshold`` unique values (the enclosing function's
                argument, default ``500``), because a random colormap --
                black at index 0, one random opaque colour per non-zero
                label -- is generated instead so neighbouring objects stay
                distinguishable.
        """
        unique_values = np.unique(array)
        num_unique_values = len(unique_values)

        if num_unique_values <= threshold:
            # The number of distinct values decides mask vs intensity, but the
            # object count in the title must exclude the background label 0,
            # otherwise a 3-object mask is annotated "4 (obj.)".
            num_objects = int(np.count_nonzero(unique_values))
            chosen_cmap = _generate_mask_random_cmap(array)
            title = f'{title}, {num_objects} (obj.)'

        ax.imshow(array, cmap=chosen_cmap)
        ax.set_title(title, size=18)
        ax.axis('off')

    if len(dim) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(figuresize, figuresize))
        plot_single_array(stack, ax, 'Channel one', plt.get_cmap(cmap))
        fig.tight_layout()
        plt.show()
    elif len(dim) > 2:
        num_channels = dim[2]
        fig, axs = plt.subplots(1, num_channels, figsize=(figuresize, figuresize))
        # A single channel makes plt.subplots return a bare Axes, not an array,
        # so axs[channel] below would raise TypeError.
        axs = np.atleast_1d(axs)
        for channel in range(num_channels):
            plot_single_array(stack[:, :, channel], axs[channel], f'C. {channel}', plt.get_cmap(cmap))
        fig.tight_layout()    
    #print(f'{filename}')
    return fig
    
def _visualize_and_save_timelapse_stack_with_tracks(masks, tracks_df, save, src, name, plot, filenames, object_type, mode='btrack', interactive=False):
    """
    Visualizes and saves a timelapse stack with tracks.

    Args:
        masks (list): List of binary masks representing each frame of the timelapse stack.
        tracks_df (pandas.DataFrame): DataFrame containing track information.
        save (bool): Flag indicating whether to save the timelapse stack.
        src (str): Source file path.
        name (str): Name of the timelapse stack.
        plot (bool): Flag indicating whether to plot the timelapse stack.
        filenames (list): List of filenames corresponding to each frame of the timelapse stack.
        object_type (str): Type of object being tracked.
        mode (str, optional): Tracking mode. Defaults to 'btrack'.
        interactive (bool, optional): Flag indicating whether to display the timelapse stack interactively. Defaults to False.
    """
    
    from .io import _save_mask_timelapse_as_gif
    
    highest_label = max(np.max(mask) for mask in masks)
    # Generate random colors for each label, including the background
    random_colors = np.random.rand(highest_label + 1, 4)
    random_colors[:, 3] = 1  # Full opacity
    random_colors[0] = [0, 0, 0, 1]  # Background color
    cmap = plt.cm.colors.ListedColormap(random_colors)
    # Ensure the normalization range covers all labels
    norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)

    # Function to plot a frame and overlay tracks
    def _view_frame_with_tracks(frame=0):
        """
        Display the frame with tracks overlaid.

        Parameters:
        frame (int): The frame number to display.

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(50, 50))
        current_mask = masks[frame]
        ax.imshow(current_mask, cmap=cmap, norm=norm)  # Apply both colormap and normalization
        ax.set_title(f'Frame: {frame}')

        # Directly annotate each object with its label number from the mask
        for label_value in np.unique(current_mask):
            if label_value == 0: continue  # Skip background
            y, x = np.mean(np.where(current_mask == label_value), axis=1)
            ax.text(x, y, str(label_value), color='white', fontsize=24, ha='center', va='center')

        # Overlay tracks
        for track in tracks_df['track_id'].unique():
            _track = tracks_df[tracks_df['track_id'] == track]
            ax.plot(_track['x'], _track['y'], '-k', linewidth=1)

        ax.axis('off')
        plt.show()

    if plot:
        if interactive:
            interact(_view_frame_with_tracks, frame=IntSlider(min=0, max=len(masks)-1, step=1, value=0))

    if save:
        # Save as gif
        gif_path = os.path.join(os.path.dirname(src), 'movies', 'gif')
        os.makedirs(gif_path, exist_ok=True)
        save_path_gif = os.path.join(gif_path, f'timelapse_masks_{object_type}_{name}.gif')
        _save_mask_timelapse_as_gif(masks, tracks_df, save_path_gif, cmap, norm, filenames)
        if plot:
            if not interactive:
                _display_gif(save_path_gif)
                
def _display_gif(path):
    """
    Display a GIF image from the given path.

    Parameters:
    path (str): The path to the GIF image file.

    Returns:
    None
    """
    # `format='gif'` is stated rather than sniffed. IPython only learned to
    # recognise the GIF87a/GIF89a magic bytes in 9.0.0; before that, raw bytes
    # with no format fall through to 'png' and the animation is emitted with
    # an `image/png` mime type. IPython 9 needs Python 3.11, so on the 3.9 and
    # 3.10 ends of the range spaCR claims there is no version of IPython that
    # would guess right -- setup.py's `IPython>=8.18.1` resolves to exactly
    # 8.18.1 on 3.9. Saying what the file is costs nothing and is correct on
    # every version.
    with open(path, 'rb') as file:
        display(ipyimage(file.read(), format='gif'))


def _plot_recruitment(df, df_type, channel_of_interest, columns=None, figuresize=10):
    """
    Plot recruitment data for different conditions and pathogens.

    Args:
        df (DataFrame): The input DataFrame containing the recruitment data.
        df_type (str): The type of DataFrame (e.g., 'train', 'test').
        channel_of_interest (str): The channel of interest for plotting.
        target (str): The target variable for plotting.
        columns (list, optional): Additional columns to plot. Defaults to an empty list.
        figuresize (int, optional): The size of the figure. Defaults to 50.

    Returns:
        None
    """

    if columns is None:
        columns = []
    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]

    sns.set_palette(sns.color_palette(color_list))
    font = figuresize/2
    width=figuresize
    height=figuresize/4

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(width, height))
    sns.barplot(ax=axes[0], data=df, x='condition', y=f'cell_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, errorbar='sd', dodge=False)
    axes[0].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[0].set_ylabel(f'cell_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[1], data=df, x='condition', y=f'nucleus_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, errorbar='sd', dodge=False)
    axes[1].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[1].set_ylabel(f'nucleus_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[2], data=df, x='condition', y=f'cytoplasm_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, errorbar='sd', dodge=False)
    axes[2].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[2].set_ylabel(f'cytoplasm_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    sns.barplot(ax=axes[3], data=df, x='condition', y=f'pathogen_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, errorbar='sd', dodge=False)
    axes[3].set_xlabel(f'pathogen {df_type}', fontsize=font)
    axes[3].set_ylabel(f'pathogen_channel_{channel_of_interest}_mean_intensity', fontsize=font)

    #axes[0].legend_.remove()
    #axes[1].legend_.remove()
    #axes[2].legend_.remove()
    #axes[3].legend_.remove()
        
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    for i in [0,1,2,3]:
        axes[i].tick_params(axis='both', which='major', labelsize=font)
        plt.setp(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
    #columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

    width = figuresize*2
    columns_per_row = math.ceil(len(columns) / 2)
    height = (figuresize*2)/columns_per_row

    fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
    axes = axes.flatten()

    print(f'{columns}')

    for i, col in enumerate(columns):

        ax = axes[i]
        sns.barplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', capsize=.1, errorbar='sd', dodge=False)
        ax.set_xlabel(f'pathogen {df_type}', fontsize=font)
        ax.set_ylabel(f'{col}', fontsize=int(font*2))
        if ax.get_legend() is not None:
            ax.legend_.remove()
        ax.tick_params(axis='both', which='major', labelsize=font)
        plt.setp(ax.get_xticklabels(), rotation=45)
        if i <= 5:
            ax.set_ylim(1, None)

    for i in range(len(columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
def _plot_controls(df, mask_chans, channel_of_interest, figuresize=5):
    """
    Plot controls for different channels and conditions.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        mask_chans (list): The list of channels to include in the plot.
        channel_of_interest (int): The channel of interest.
        figuresize (int, optional): The size of the figure. Defaults to 5.

    Returns:
        None
    """
    mask_chans.append(channel_of_interest)
    if len(mask_chans) == 4:
        mask_chans = [0,1,2,3]
    if len(mask_chans) == 3:
        mask_chans = [0,1,2]
    if len(mask_chans) == 2:
        mask_chans = [0,1]
    if len(mask_chans) == 1:
        mask_chans = [0]
    controls_cols = []
    for chan in mask_chans:

        controls_cols_c = []
        controls_cols_c.append(f'cell_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'nucleus_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'pathogen_channel_{chan}_mean_intensity')
        controls_cols_c.append(f'cytoplasm_channel_{chan}_mean_intensity')
        controls_cols.append(controls_cols_c)

    unique_conditions = df['condition'].unique().tolist()

    if len(unique_conditions) ==1:
        unique_conditions=unique_conditions+unique_conditions

    fig, axes = plt.subplots(len(unique_conditions), len(mask_chans)+1, figsize=(figuresize*len(mask_chans), figuresize*len(unique_conditions)))

    # Define RGB color tuples (scaled to 0-1 range)
    color_list = [(55/255, 155/255, 155/255), 
                  (155/255, 55/255, 155/255), 
                  (55/255, 155/255, 255/255), 
                  (255/255, 55/255, 155/255)]

    for idx_condition, condition in enumerate(unique_conditions):
        df_temp = df[df['condition'] == condition]
        for idx_channel, control_cols_c in enumerate(controls_cols):
            # Build labels and colours alongside the data. The bar call used to
            # pass all four component names unconditionally while the guard
            # above skips missing columns, so any absent component (or a whole
            # absent channel) made x and height different lengths and raised.
            names = []
            data = []
            std_dev = []
            colors = []
            for color, control_col in zip(color_list, control_cols_c):
                if control_col in df_temp.columns:
                    mean_intensity = df_temp[control_col].mean()
                    mean_intensity = 0 if np.isnan(mean_intensity) else mean_intensity
                    names.append(control_col.split('_channel_')[0])
                    data.append(mean_intensity)
                    std_dev.append(df_temp[control_col].std())
                    colors.append(color)

            current_axis = axes[idx_condition][idx_channel]
            current_axis.bar(names, data, yerr=std_dev,
                             capsize=4, color=colors)
            current_axis.set_xlabel('Component')
            current_axis.set_ylabel('Mean Intensity')
            current_axis.set_title(f'Condition: {condition} - Channel {idx_channel}')
    plt.tight_layout()
    plt.show()

def _imshow(img, labels, nrow=20, color='white', fontsize=12):
    """
    Display multiple images in a grid with corresponding labels.

    Args:
        img (list): List of images to display.
        labels (list): List of labels corresponding to each image.
        nrow (int, optional): Number of images per row in the grid. Defaults to 20.
        color (str, optional): Color of the label text. Defaults to 'white'.
        fontsize (int, optional): Font size of the label text. Defaults to 12.
    """
    n_images = len(labels)
    n_col = nrow
    n_row = int(np.ceil(n_images / n_col))
    img_height = img[0].shape[1]
    img_width = img[0].shape[2]
    canvas = np.zeros((img_height * n_row, img_width * n_col, 3))
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n_images:
                canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = np.transpose(img[idx], (1, 2, 0))        
    fig = plt.figure(figsize=(50, 50))
    plt.imshow(canvas)
    plt.axis("off")
    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')
    return fig

def _imshow_gpu(img, labels, nrow=20, color='white', fontsize=12):
    """
    Display multiple images in a grid with corresponding labels.

    Args:
        img (torch.Tensor): A batch of images as a tensor.
        labels (list): List of labels corresponding to each image.
        nrow (int, optional): Number of images per row in the grid. Defaults to 20.
        color (str, optional): Color of the label text. Defaults to 'white'.
        fontsize (int, optional): Font size of the label text. Defaults to 12.
    """
    if img.is_cuda:
        img = img.cpu()  # Move to CPU if the tensor is on GPU

    n_images = len(labels)
    n_col = nrow
    n_row = int(np.ceil(n_images / n_col))

    img_height = img.shape[2]  # Height of the image
    img_width = img.shape[3]   # Width of the image

    # Prepare the canvas on CPU
    canvas = torch.zeros((img_height * n_row, img_width * n_col, 3))

    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n_images:
                # Place the image on the canvas
                canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = img[idx].permute(1, 2, 0)

    canvas = canvas.numpy()  # Convert to NumPy for plotting

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(canvas)
    plt.axis("off")

    for i, label in enumerate(labels):
        row = i // n_col
        col = i % n_col
        x = col * img_width + 2
        y = row * img_height + 15
        plt.text(x, y, label, color=color, fontsize=fontsize, fontweight='bold')

    return fig
    
def _plot_histograms_and_stats(df):
    conditions = df['condition'].unique()
    
    for condition in conditions:
        subset = df[df['condition'] == condition]
        
        # Calculate the statistics
        mean_pred = subset['pred'].mean()
        over_0_5 = sum(subset['pred'] > 0.5)
        under_0_5 = sum(subset['pred'] <= 0.5)

        # Print the statistics
        print(f"Condition: {condition}")
        print(f"Number of rows: {len(subset)}")
        print(f"Mean of pred: {mean_pred}")
        print(f"Count of pred values over 0.5: {over_0_5}")
        print(f"Count of pred values under 0.5: {under_0_5}")
        print(f"Percent positive: {(over_0_5/(over_0_5+under_0_5))*100}")
        print(f"Percent negative: {(under_0_5/(over_0_5+under_0_5))*100}")
        print('-'*40)
        
        # Plot the histogram
        plt.figure(figsize=(10,10))
        plt.hist(subset['pred'], bins=30, edgecolor='black')
        plt.axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f"Mean = {mean_pred:.2f}")
        plt.title(f'Histogram for pred - Condition: {condition}')
        plt.xlabel('Pred Value')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

def _show_residules(model):

    # Get the residuals
    residuals = model.resid

    # Histogram of residuals
    plt.hist(residuals, bins=30)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()

    # QQ plot
    sm.qqplot(residuals, fit=True, line='45')
    plt.title('QQ Plot')
    plt.show()

    # Residuals vs. Fitted values
    plt.scatter(model.fittedvalues, residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    plt.axhline(y=0, color='red')
    plt.show()

    # Shapiro-Wilk test for normality
    W, p_value = stats.shapiro(residuals)
    print(f'Shapiro-Wilk Test W-statistic: {W}, p-value: {p_value}')
    
def _reg_v_plot(df, grouping=None, variable=None, plate_number=None):
    # grouping/variable/plate_number are unused by the body but kept for
    # call-site compatibility; they default so utils.MLR's `_reg_v_plot(df)`
    # call works instead of raising TypeError.
    df['-log10(p)'] = -np.log10(df['p'])

    # Create the volcano plot
    plt.figure(figsize=(40, 30))
    plt.scatter(df['effect'], df['-log10(p)'], c=np.sign(df['effect']), cmap='coolwarm')
    plt.title('Volcano Plot', fontsize=12)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('-log10(P-value)', fontsize=12)

    # Add text for specified points
    for idx, row in df.iterrows():
        if row['p'] < 0.05:# and abs(row['effect']) > 0.1:
            plt.text(row['effect'], -np.log10(row['p']), idx, fontsize=12, ha='center', va='bottom', color='black')

    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--')  # line for p=0.05
    plt.show()

def _well_axis_labels(tokens, parse, render):
    """Map raw row (or column) tokens onto ``(index, canonical label)``.

    Two rules, and both of them matter:

    * **The letter walk is borrowed, never rewritten.** ``parse`` is
      :func:`spacr.plate_qc.parse_row_label` / ``parse_column_label`` and
      ``render`` is :func:`spacr.schema.row_id` / ``column_id``, so
      ``'AA'``, ``'r27'`` and ``'row27'`` are one row here, in QC and in
      the database. A hand-rolled ``chr(ord('A') + n)`` produces ``'['``
      for row 27, which is how this class of bug started.
    * **Each distinct token is parsed once.** A measurement frame is a
      million object rows over a few hundred wells; parsing per row would
      run the regexes a million times for a few hundred answers.

    :param tokens: sequence of raw tokens as they appear in ``prc``.
    :param parse: label reader returning a 1-based index or ``None``.
    :param render: id builder turning that index into ``'r<N>'``/``'c<N>'``.
    :returns: ``(indices, labels)`` object arrays. An unreadable token has
        index ``None`` and keeps its raw text as its label, so a caller can
        name it in a report instead of dropping it namelessly.
    """
    cache = {}
    for token in set(tokens):
        index = parse(token)
        cache[token] = (index, token if index is None else render(int(index)))
    indices = np.array([cache[t][0] for t in tokens], dtype=object)
    labels = np.array([cache[t][1] for t in tokens], dtype=object)
    return indices, labels


def generate_plate_heatmap(df, plate_number, variable, grouping, min_max, min_count):
    """Aggregate a well-level DataFrame into a plate-shaped heatmap.

    The grid is **read off the data**. It used to be pinned to ``r1..r16``
    by ``c1..c27``, so every well of a 1536 plate past row P or past column
    27 fell outside the ``Categorical``, became NaN, and was dropped by the
    groupby — measured, in the database, and simply absent from the figure
    with nothing said. Rows and columns now go through
    :func:`spacr.plate_qc.parse_row_label` / ``parse_column_label`` (which
    is :mod:`spacr.schema`'s letter walk, so ``AA``…``AF`` and beyond are
    real rows), and the axes span exactly the wells present: a 96 plate is
    still 8x12 and a 384 still 16x24, because nothing is padded out to the
    largest format that exists.

    A well that genuinely cannot be placed — a ``prc`` with too few parts,
    or a row/column token holding no position — is reported through
    :func:`spacr.errors.raise_if_strict` (an ``ERROR`` on
    ``spacr.errors``, or a raise under ``SPACR_STRICT_ERRORS``) naming the
    identifiers concerned. Replacing a silent drop with a quieter silent
    drop would fix nothing.

    :param df: Long-format DataFrame with a ``prc`` (plate_row_column)
        identifier and the requested ``variable`` column.
    :param plate_number: Plate ID selecting the subset to display.
    :param variable: Column to aggregate. Ignored when
        ``grouping='count'``.
    :param grouping: Aggregation — ``'count'``, ``'mean'`` or ``'sum'``.
    :param min_max: Colour scale spec — ``'all'``, ``'allq'``, or a
        two-element list ``[vmin, vmax]`` (floats treated as quantiles).
    :param min_count: Drop wells with fewer than this many rows.
    :returns: ``(plate_map, (vmin, vmax))`` — the pivoted matrix, indexed
        ``'r<N>'`` by ``'c<N>'``, and the colour-limit tuple.
    :raises ValueError: if ``grouping`` is not one of the accepted values.
    :raises KeyError: if ``variable`` is missing and required.
    """
    from . import plate_qc as _plate_qc
    from . import schema as _schema

    if not isinstance(min_count, (int, float)):
        min_count = 0

    # -- read the well out of prc -----------------------------------------
    # prc is <plate>_<row>_<column>, read right to left: the last two tokens
    # are the position and whatever precedes them is the plate. Left-to-right
    # unpacking put the *row* in the plate slot for any identifier carrying
    # an experiment prefix, and only ``prc.iloc[0]`` was ever probed for its
    # length, so a frame mixing 3- and 4-token identifiers misaligned every
    # row of the minority shape.
    prc_text = df['prc'].astype(str)
    parts = [text.split(_schema.KEY_SEPARATOR) for text in prc_text]
    # A longer identifier carries an experiment prefix; the plate the caller
    # asked for is then the authority on which plate it is, which is what the
    # old 4-part rebuild did. Too short and there is no position at all — the
    # rows are kept here precisely so they can be reported below.
    plate_token = np.array(
        [p[0] if len(p) == 3 else str(plate_number) for p in parts], dtype=object)
    row_token = np.array([p[-2] if len(p) >= 3 else '' for p in parts], dtype=object)
    col_token = np.array([p[-1] if len(p) >= 3 else '' for p in parts], dtype=object)

    if not all(len(p) == 3 for p in parts):
        # A rebuilt identifier must not be written back onto the caller's
        # frame. The plain 3-token path always has done, and is pinned.
        df = df.copy()

    # Derive plateID,rowID,columnID from prc if not already present
    if 'column_name' not in df.columns:
        if 'column' in df.columns:
            df['columnID'] = df['column']
        elif 'column_name' in df.columns:
            df['columnID'] = df['column_name']

    if 'plateID' not in df.columns:
        if 'plate' in df.columns:
            df['plateID'] = df['plate']
        elif 'plate_name' in df.columns:
            df['plateID'] = df['plate_name']
        else:
            df['plateID'] = 'p1'

    row_index, row_label = _well_axis_labels(
        row_token, _plate_qc.parse_row_label, _schema.row_id)
    col_index, col_label = _well_axis_labels(
        col_token, _plate_qc.parse_column_label, _schema.column_id)

    df['plateID'], df['rowID'], df['columnID'] = plate_token, row_label, col_label

    # -- filter one plate, and say what could not be drawn ------------------
    # dtype=bool explicitly: an empty frame gives an empty float array, and
    # `~` on a float array is a TypeError rather than "nothing to report".
    on_plate = np.asarray(plate_token == str(plate_number), dtype=bool)
    placeable = np.array([r is not None and c is not None
                          for r, c in zip(row_index, col_index)], dtype=bool)
    lost = on_plate & ~placeable
    if lost.any():
        names = sorted(set(prc_text.to_numpy()[lost]))
        shown = ', '.join(names[:12]) + (' …' if len(names) > 12 else '')
        raise_if_strict(
            f"plate {plate_number!r}: {int(lost.sum())} row(s) covering "
            f"{len(names)} identifier(s) hold no well position and are "
            f"missing from the heatmap: {shown}. A prc must be "
            f"<plate>_<row>_<column> with a readable row ('r3', 'C', 'AA') "
            f"and column ('c7', '7'); a well drawn nowhere is "
            f"indistinguishable from a well that was never measured.")

    keep = on_plate & placeable
    df = df[keep].copy()
    # Group on the integer position, not on the label: 'c10' sorts before
    # 'c2' as text, and a Categorical of hard-coded labels was what silently
    # deleted rows past P in the first place.
    df['_row_index'] = row_index[keep].astype(int)
    df['_col_index'] = col_index[keep].astype(int)
    keys = ['_row_index', '_col_index']

    # Optional min_count filter on true per-well counts
    df['_well_count'] = df.groupby(
        keys, observed=False)['_row_index'].transform('count')
    if min_count > 0:
        df = df[df['_well_count'] >= min_count]

    grouped = df.groupby(keys, observed=False)

    # --- Aggregation ---
    if grouping == 'count':
        plate = grouped.size().reset_index(name='value')               # per-well row counts
    elif grouping in ('mean', 'sum'):
        if variable not in df.columns:
            raise KeyError(f"variable '{variable}' not in df")
        vals = pd.to_numeric(df[variable], errors='coerce')            # ensure numeric
        tmp  = df.assign(__val__=vals)
        if grouping == 'mean':
            plate = tmp.groupby(
                keys, observed=False)['__val__'].mean().reset_index(name='value')
        else:  # sum
            plate = tmp.groupby(
                keys, observed=False)['__val__'].sum().reset_index(name='value')
    else:
        raise ValueError("grouping must be 'count', 'sum', or 'mean'")

    plate_map = pd.pivot_table(plate, values='value', index='_row_index',
                               columns='_col_index').fillna(0)
    # Back to the ids the rest of spaCR speaks, in numeric order.
    plate_map.index = pd.Index([_schema.row_id(int(i)) for i in plate_map.index],
                               name='rowID')
    plate_map.columns = pd.Index([_schema.column_id(int(i)) for i in plate_map.columns],
                                 name='columnID')

    # vmin/vmax selection. Guard against an empty pivot (e.g. a tiny plate
    # where every well was filtered out): np.quantile / np.nanmin on a
    # zero-size array raises, so fall back to a neutral [0, 1] range.
    if plate_map.values.size == 0:
        return plate_map, (0.0, 1.0)
    if min_max == 'all':
        vmin, vmax = float(np.nanmin(plate_map.values)), float(np.nanmax(plate_map.values))
    elif min_max == 'allq':
        vmin, vmax = np.quantile(plate_map.values, [0.02, 0.98])
    elif isinstance(min_max, (list, tuple)) and len(min_max) == 2:
        if all(isinstance(x, float) for x in min_max):
            vmin, vmax = np.quantile(plate_map.values, [min_max[0], min_max[1]])
        else:
            vmin, vmax = float(min_max[0]), float(min_max[1])
    else:
        vmin, vmax = float(np.nanmin(plate_map.values)), float(np.nanmax(plate_map.values))

    # avoid degenerate colormap
    if vmin == vmax:
        vmax = vmin + 1e-6

    return plate_map, (vmin, vmax)


def plot_plates(df, variable, grouping, min_max, cmap, min_count=0, verbose=True, dst=None):
    """Render one heatmap per plate on a common grid so per-well phenotypes can be eye-balled across a full screen.

    Splits ``df`` by the plate token in the ``prc`` identifier
    (``plateID_rowID_columnID``), aggregates ``variable`` per well
    according to ``grouping``, and lays the plates out four-per-row on
    a single figure. Optionally writes the figure to
    ``<dst>/plate_heatmap_<n>.pdf``.

    Each panel's own grid comes from :func:`generate_plate_heatmap`, which
    reads it off the wells present — 8x12, 16x24, 32x48 or whatever the
    data says — so a 1536 plate draws all 32 of its rows and a 96 plate is
    not padded out to match it.

    :param df: Long-format DataFrame with a ``prc`` column of the form
        ``plateID_rowID_columnID`` and the column named by ``variable``.
    :param variable: Column to aggregate (see
        :func:`generate_plate_heatmap`).
    :param grouping: Aggregation mode — ``'count'``, ``'mean'`` or
        ``'sum'``.
    :param min_max: Color-scale spec forwarded to
        :func:`generate_plate_heatmap` (``'all'``, ``'allq'``,
        ``[vmin, vmax]``).
    :param cmap: Matplotlib colormap name or object.
    :param min_count: Drop wells with fewer than this many rows before
        plotting. Default ``0``.
    :param verbose: If True, call ``plt.show()`` after building the
        figure. Default ``True``.
    :param dst: If given, save the figure as ``plate_heatmap_<n>.pdf``
        under this folder (auto-numbered).
    :returns: The generated matplotlib ``Figure``.

    Example:
        .. code-block:: python

            from spacr.plot import plot_plates
            fig = plot_plates(
                df, variable='recruitment', grouping='mean',
                min_max='allq', cmap='viridis', min_count=20,
            )

    See Also:
        :func:`spacr.ml.generate_ml_scores` — produces score dataframes
        typically fed to this plotter.
    """
    plates = df['prc'].str.split('_', expand=True)[0].unique()
    n_rows, n_cols = (len(plates) + 3) // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))
    ax = ax.flatten()

    for index, plate in enumerate(plates):
        plate_map, (vmin, vmax) = generate_plate_heatmap(df, plate, variable, grouping, min_max, min_count)
        sns.heatmap(plate_map, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax[index])
        ax[index].set_title(plate)

    # remove unused axes
    for i in range(len(plates), n_rows * n_cols):
        fig.delaxes(ax[i])

    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    if dst is not None:
        for i in range(0, 1000):
            filename = os.path.join(dst, f'plate_heatmap_{i}.pdf')
            if not os.path.exists(filename):
                filename = save_figure(fig, filename)
                print(f'Saved heatmap to {filename}')
                break

    if verbose:
        plt.show()
    return fig

def print_mask_and_flows(stack, mask, flows, overlay=True, max_size=1000, thickness=2):
    """Show a single image, its label mask (optionally outlined) and flow image.

    :param stack: Original 2D image or ``(H, W, C)`` stack.
    :param mask: Label mask matching ``stack`` spatially.
    :param flows: Optional list of flow arrays; skipped when ``None``.
    :param overlay: If True, draw mask contours over the image instead
        of showing the mask alone. Default ``True``.
    :param max_size: Downsample any dimension exceeding this size.
        Default ``1000``.
    :param thickness: Contour line thickness in pixels. Default ``2``.
    :returns: None
    """

    def resize_if_needed(image, max_size):
        """Resize image if any dimension exceeds max_size while maintaining aspect ratio.

        :param image: 2D or ``(H, W, C)`` array. The channel axis is left
            untouched, and the result is cast back to the input dtype, so a
            label mask keeps integer labels — but the interpolation is
            anti-aliased, which can invent label values that belong to no
            object along object borders.
        :param max_size: Cap on the larger of height and width, in pixels.
            The image is returned unchanged when it already fits, so no
            upscaling ever happens; a non-positive value drives the scale
            factor to zero, so pass a real pixel budget.
        :returns: The resized array, or ``image`` itself when it fits.
        """
        if max(image.shape[:2]) > max_size:
            scale = max_size / max(image.shape[:2])
            new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
            if image.ndim == 3:
                new_shape += (image.shape[2],)
            return sk_resize(image, new_shape, preserve_range=True, anti_aliasing=True).astype(image.dtype)
        return image

    def generate_contours(mask):
        """Generate contours for each object in the mask using OpenCV.

        :param mask: Label mask, cast to ``uint8`` before tracing — labels
            above 255 wrap around, and because only external contours are
            retrieved, touching objects trace as one outline and holes
            inside an object are not outlined.
        :returns: The OpenCV contour list, ready for ``cv2.drawContours``.
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def apply_contours_on_image(image, mask, color=(255, 0, 0), thickness=2):
        """Draw the contours on the original image.

        :param image: Base image. A 2D array is normalised to ``uint8`` and
            promoted to RGB first, which assumes it is already scaled to
            ``[0, 1]``; anything already 3D is copied and drawn on as-is, so
            the caller owns its dtype and value range.
        :param mask: Label mask the outlines come from, traced with
            ``generate_contours``. It must line up pixel-for-pixel with
            ``image``, so resize both with the same ``max_size``.
        :param color: Contour colour as a BGR/RGB triple in 0-255, matching
            however the image channels are ordered. Default ``(255, 0, 0)``.
        :param thickness: Line width in pixels; a negative value fills each
            contour solid instead of outlining it. Default ``2``.
        :returns: A new RGB array; the input ``image`` is not modified.
        """
        # Ensure the image is in RGB format
        if image.ndim == 2:  # Grayscale to RGB
            image = normalize_to_uint8(image)  # Convert to uint8 if needed
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        # Generate and draw contours
        contours = generate_contours(mask)
        cv2.drawContours(image_rgb, contours, -1, color, thickness)

        return image_rgb

    def normalize_to_uint8(image):
        """Normalize and convert image to uint8.

        :param image: Array whose values are assumed to be scaled to
            ``[0, 1]`` already — the function only clips and multiplies by
            255, it does not rescale. Raw 16-bit camera data therefore
            saturates to solid white apart from its zero pixels, and
            negative values clip to black.
        :returns: A ``uint8`` array of the same shape.
        """
        image = np.clip(image, 0, 1)  # Ensure values are between 0 and 1
        return (image * 255).astype(np.uint8)  # Convert to uint8
    
    
    # Resize if necessary
    stack = resize_if_needed(stack, max_size)
    mask = resize_if_needed(mask, max_size)
    if flows != None:
        flows = [resize_if_needed(flow, max_size) for flow in flows]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if stack.shape[-1] == 1:
        stack = np.squeeze(stack)

    # Display original image
    if stack.ndim == 2:
        original_image = stack
    elif stack.ndim == 3:
        original_image = stack[..., 0]  # Use the first channel as the base
    else:
        raise ValueError("Unexpected stack dimensionality.")

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Overlay mask outlines on original image if overlay is True
    if overlay:
        outlined_image = apply_contours_on_image(original_image, mask, color=(255, 0, 0), thickness=thickness)
        axs[1].imshow(outlined_image)
    else:
        axs[1].imshow(mask, cmap='gray')

    axs[1].set_title('Mask with Overlay' if overlay else 'Mask')
    axs[1].axis('off')

    if flows != None:

        # Display flow image or its first channel
        if flows and isinstance(flows, list) and flows[0].ndim in [2, 3]:
            flow_image = flows[0]
            if flow_image.ndim == 3:
                flow_image = flow_image[:, :, 0]  # Use first channel for 3D
            axs[2].imshow(flow_image, cmap='jet')
        else:
            raise ValueError("Unexpected flow dimensionality or structure.")

        axs[2].set_title('Flows')
        axs[2].axis('off')

    fig.tight_layout()
    plt.show()
    
def plot_resize(images, resized_images, labels, resized_labels):
    """Show original vs. resized image/label pairs in a 2x2 grid.

    :param images: Sequence of original images (first element shown).
    :param resized_images: Sequence of resized images.
    :param labels: Sequence of original label arrays.
    :param resized_labels: Sequence of resized label arrays.
    :returns: None
    """
    def prepare_image(img):
        """Return ``(display_array, cmap)`` handling 2D/3D input shapes.

        :param img: A 2D array, or a 3D array in channels-last order. One
            channel is squeezed to 2D and three or four are passed through
            as RGB/RGBA with a ``None`` colormap; any other channel count
            (a 5-channel spaCR stack, or a channels-first array read
            straight off disk) falls back to the mean across the last axis,
            which is a legal but usually misleading picture.
        :returns: ``(array, cmap)`` to hand straight to ``imshow``, where
            ``cmap`` is ``None`` for true-colour data.
        :raises ValueError: if ``img`` is neither 2D nor 3D.
        """
        if img.ndim == 2:
            return img, 'gray'
        elif img.ndim == 3:
            if img.shape[-1] == 1:
                return np.squeeze(img, axis=-1), 'gray'
            elif img.shape[-1] == 3:
                return img, None  # RGB
            elif img.shape[-1] == 4:
                return img, None  # RGBA
            else:
                # fallback: average across channels to show as grayscale
                return np.mean(img, axis=-1), 'gray'
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    # Original Image
    img, cmap = prepare_image(images[0])
    ax[0, 0].imshow(img, cmap=cmap)
    ax[0, 0].set_title('Original Image')

    # Resized Image
    img, cmap = prepare_image(resized_images[0])
    ax[0, 1].imshow(img, cmap=cmap)
    ax[0, 1].set_title('Resized Image')

    # Labels (assumed grayscale or single-channel)
    lbl, cmap = prepare_image(labels[0])
    ax[1, 0].imshow(lbl, cmap=cmap)
    ax[1, 0].set_title('Original Label')

    lbl, cmap = prepare_image(resized_labels[0])
    ax[1, 1].imshow(lbl, cmap=cmap)
    ax[1, 1].set_title('Resized Label')

    plt.tight_layout()
    plt.show()
    
def normalize_and_visualize(image, normalized_image, title=""):
    """Show the original and the normalised image side by side in grayscale.

    Multi-channel inputs are averaged over their channels for display.

    :param image: Original image, 2D or ``(H, W, C)``.
    :param normalized_image: Normalised counterpart to compare against.
    :param title: Suffix appended to both panel titles. Default ``""``.
    :returns: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if image.ndim == 3:  # Multi-channel image
        ax[0].imshow(np.mean(image, axis=-1), cmap='gray')  # Display the average over channels for visualization
    else:  # Grayscale image
        ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original " + title)
    ax[0].axis('off')

    if normalized_image.ndim == 3:
        ax[1].imshow(np.mean(normalized_image, axis=-1), cmap='gray')  # Similarly, display the average over channels
    else:
        ax[1].imshow(normalized_image, cmap='gray')
    ax[1].set_title("Normalized " + title)
    ax[1].axis('off')
    
    plt.show()
    
def visualize_masks(mask1, mask2, mask3, title="Masks Comparison"):
    """Show three masks side by side with random colormaps.

    :param mask1: First label mask.
    :param mask2: Second label mask.
    :param mask3: Third label mask.
    :param title: Figure suptitle. Default ``"Masks Comparison"``.
    :returns: None
    """
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    # The loop variable must not be named `title`: it shadowed the parameter,
    # so the suptitle below always read 'Mask 3' instead of the caller's title.
    for ax, mask, panel_title in zip(axs, [mask1, mask2, mask3], ['Mask 1', 'Mask 2', 'Mask 3']):
        cmap = generate_mask_random_cmap(mask)
        # If the mask is binary, we can skip normalization
        if np.isin(mask, [0, 1]).all():
            ax.imshow(mask, cmap=cmap)
        else:
            # Normalize the image for displaying purposes
            norm = plt.Normalize(vmin=0, vmax=mask.max())
            ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title(panel_title)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def visualize_cellpose_masks(masks, titles=None, filename=None, save=False, src=None):
    """Display several Cellpose-style label masks side by side for a quick visual QC.

    Handy for sanity-checking the masks produced by
    :func:`spacr.core.preprocess_generate_masks` (e.g. compare the cell,
    nucleus and pathogen masks of the same field, or two runs against
    each other). Each mask is rendered with a random-color palette so
    neighbouring objects stay distinguishable.

    :param masks: Sequence of 2D label mask arrays.
    :param titles: Titles paired positionally with ``masks``. Falls back
        to ``'Mask 1'``, ``'Mask 2'``, ...
    :param filename: Used in the figure suptitle and, when ``save``, as
        the output PDF filename.
    :param save: If True, save the figure under
        ``<src>/results/<filename>.pdf``. Default ``False``.
    :param src: Root folder for saving. Defaults to the current working
        directory.
    :returns: None. Displays (and optionally writes) the figure.
    :raises AssertionError: if ``titles`` and ``masks`` have different
        lengths.

    Example:
        .. code-block:: python

            from spacr.plot import visualize_cellpose_masks
            visualize_cellpose_masks(
                [cell_mask, nucleus_mask, pathogen_mask],
                titles=['cell','nucleus','pathogen'],
                filename='field_001', save=True, src='/data/plate01',
            )

    See Also:
        :func:`spacr.core.preprocess_generate_masks` — produces the
        masks visualized here.
    """
    
    comparison_title=f"Masks Comparison for {filename}"
    
    if titles is None:
        titles = [f'Mask {i+1}' for i in range(len(masks))]
    
    # Ensure the length of titles matches the number of masks
    assert len(titles) == len(masks), "Number of titles and masks must match"
    
    num_masks = len(masks)
    fig, axs = plt.subplots(1, num_masks, figsize=(10 * num_masks, 10))  # Adjusting figure size dynamically
    # A single mask makes plt.subplots return a bare Axes, which zip() below
    # cannot iterate.
    axs = np.atleast_1d(axs)

    for ax, mask, title in zip(axs, masks, titles):
        cmap = generate_mask_random_cmap(mask)
        # Normalize and display the mask
        norm = plt.Normalize(vmin=0, vmax=mask.max())
        ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle(comparison_title)
    plt.show()
    
    if save:
        if src is None:
            src = os.getcwd()
        results_dir = os.path.join(src, 'results')
        os.makedirs(results_dir, exist_ok=True)
        fig_path = os.path.join(results_dir, f'{filename}.pdf')
        fig_path = save_figure(fig, fig_path)
        print(f'Saved figure to {fig_path}')
    return

    
def plot_comparison_results(comparison_results):
    """Plot Jaccard, Dice, boundary-F1 and average-precision distributions per comparison.

    :param comparison_results: Iterable of dicts with per-file metrics
        (each key ending in ``jaccard``/``dice``/``boundary_f1``/
        ``average_precision``).
    :returns: The generated ``Figure``.
    """
    df = pd.DataFrame(comparison_results)
    df_melted = pd.melt(df, id_vars=['filename'], var_name='metric', value_name='value')
    df_jaccard = df_melted[df_melted['metric'].str.contains('jaccard')]
    df_dice = df_melted[df_melted['metric'].str.contains('dice')]
    df_boundary_f1 = df_melted[df_melted['metric'].str.contains('boundary_f1')]
    df_ap = df_melted[df_melted['metric'].str.contains('average_precision')]
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    
    # Jaccard Index Plot
    sns.boxplot(data=df_jaccard, x='metric', y='value', ax=axs[0], color='lightgrey')
    sns.stripplot(data=df_jaccard, x='metric', y='value', ax=axs[0], jitter=True, alpha=0.6)
    axs[0].set_title('Jaccard Index by Comparison')
    plt.setp(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[0].set_xlabel('Comparison')
    axs[0].set_ylabel('Jaccard Index')
    # Dice Coefficient Plot
    sns.boxplot(data=df_dice, x='metric', y='value', ax=axs[1], color='lightgrey')
    sns.stripplot(data=df_dice, x='metric', y='value', ax=axs[1], jitter=True, alpha=0.6)
    axs[1].set_title('Dice Coefficient by Comparison')
    plt.setp(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[1].set_xlabel('Comparison')
    axs[1].set_ylabel('Dice Coefficient')
    # Border F1 scores
    sns.boxplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], color='lightgrey')
    sns.stripplot(data=df_boundary_f1, x='metric', y='value', ax=axs[2], jitter=True, alpha=0.6)
    axs[2].set_title('Boundary F1 Score by Comparison')
    plt.setp(axs[2].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[2].set_xlabel('Comparison')
    axs[2].set_ylabel('Boundary F1 Score')
    # AP scores plot
    sns.boxplot(data=df_ap, x='metric', y='value', ax=axs[3], color='lightgrey')
    sns.stripplot(data=df_ap, x='metric', y='value', ax=axs[3], jitter=True, alpha=0.6)
    axs[3].set_title('Average Precision by Comparison')
    plt.setp(axs[3].get_xticklabels(), rotation=45, horizontalalignment='right')
    axs[3].set_xlabel('Comparison')
    axs[3].set_ylabel('Average Precision')
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_object_outlines(src, objects=None, channels=None, max_nr=10):
    """Overlay mask outlines on the matching channel image for each object type.

    :param src: Experiment root; ``masks/<object>_mask_stack`` and
        channel folders live directly under it.
    :param objects: Object types to plot. Default
        ``['nucleus', 'cell', 'pathogen']``.
    :param channels: Channel indices paired with ``objects`` (channel
        folders are named ``<channel + 1>``). Default ``[0, 1, 2]``.
    :param max_nr: Maximum number of images to plot per object.
        Default ``10``.
    :returns: None
    """
    if objects is None:
        objects = ['nucleus','cell','pathogen']
    if channels is None:
        channels = [0,1,2]
    for object_, channel in zip(objects, channels):
        folders = [os.path.join(src, 'masks', f'{object_}_mask_stack'),
                   os.path.join(src,f'{channel+1}')]
        print(folders)
        plot_images_and_arrays(folders,
                               lower_percentile=2,
                               upper_percentile=99.5,
                               threshold=1000,
                               extensions=['.npy', '.tif', '.tiff', '.png'],
                               overlay=True,
                               # Forward the caller's cap; the literal 10 made
                               # the documented max_nr parameter dead.
                               max_nr=max_nr,
                               randomize=True)
                

def plot_histogram(df, column, dst=None):
    """Plot a histogram of ``df[column]`` and optionally save it as PDF.

    :param df: DataFrame containing ``column``.
    :param column: Column to plot.
    :param dst: If set, save under ``<dst>/<column>_histogram.pdf``.
    :returns: None
    """
    # Plot histogram of the dependent variable
    bar_color = (0/255, 155/255, 155/255)
    plt.figure(figsize=(10, 10))
    sns.histplot(df[column], kde=False, color=bar_color, edgecolor=None, alpha=0.6)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    if not dst is None:
        filename = os.path.join(dst, f'{column}_histogram.pdf')
        filename = save_figure(plt.gcf(), filename)
        print(f'Saved histogram to {filename}')

    plt.show()

def plot_lorenz_curves(csv_files, name_column='grna_name', value_column='count',
                       remove_keys=None,
                       x_lim=None, y_lim=None, remove_outliers=False, save=True):
    """Overlay Lorenz curves from multiple gRNA count CSVs with per-plate Gini coefficients.

    :param csv_files: Paths to per-plate CSVs, each with columns
        ``name_column`` and ``value_column``.
    :param name_column: Identifier column used for outlier filtering.
        Default ``'grna_name'``.
    :param value_column: Column whose distribution is analysed.
        Default ``'count'``.
    :param remove_keys: Names to exclude before analysis. Default ``[]``
        (exclude nothing).
    :param x_lim: X-axis limits ``[lo, hi]``. Default ``[0.0, 1]``.
    :param y_lim: Y-axis limits ``[lo, hi]``. Default ``[0, 1]``.
    :param remove_outliers: If True, drop names whose per-well count
        falls outside a 1.5*IQR window. Default ``False``.
    :param save: If True, save the figure alongside the first CSV under
        ``results/lorenz_curve_with_gini.pdf``. Default ``True``.
    :returns: None
    """
    # remove_keys got the same mutable-default -> None treatment as x_lim/y_lim
    # but never got the matching guard, so the documented default call died on
    # `for remove in None`.
    if remove_keys is None:
        remove_keys = []
    if x_lim is None:
        x_lim = [0.0, 1]
    if y_lim is None:
        y_lim = [0, 1]
    def lorenz_curve(data):
        """Calculate Lorenz curve.

        :param data: 1D array of non-negative counts; it is sorted here, so
            the caller's order does not matter. The curve is normalised by
            the running total's last element, so an all-zero input divides
            by zero and an input mixing signs is not a Lorenz curve at all.
            Must be non-empty — an empty array indexes past the end.
        :returns: ``len(data) + 1`` cumulative shares rising from 0 to 1,
            one longer than the input because the origin is prepended.
        """
        sorted_data = np.sort(data)
        cumulative_data = np.cumsum(sorted_data)
        lorenz_curve = cumulative_data / cumulative_data[-1]
        lorenz_curve = np.insert(lorenz_curve, 0, 0)
        return lorenz_curve
    
    def gini_coefficient(data):
        """Calculate Gini coefficient from data.

        :param data: 1D array of non-negative counts, sorted internally.
            It is normalised by ``np.sum(data)``, so an all-zero input
            yields ``nan``. Unlike ``lorenz_curve``, an empty array does
            not raise here — it silently returns ``1.0``, the value for
            maximum inequality — so filter empty plates out upstream.
        :returns: The Gini coefficient as a float, from ``0.0`` for a
            perfectly even distribution up towards ``1.0`` as the counts
            concentrate on a few gRNAs. The area is taken with the
            trapezoid rule, so an even distribution reports exactly ``0.0``
            rather than ``1/n``.
        """
        sorted_data = np.sort(data)
        n = len(data)
        cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
        cumulative_data = np.insert(cumulative_data, 0, 0)
        # Trapezoid rule, not a left-Riemann sum: taking only the left endpoint
        # under-counts the area by exactly 1/n, so a perfectly equal
        # distribution reported 1/n instead of 0.
        gini = 1 - np.sum((cumulative_data[:-1] + cumulative_data[1:]) * np.diff(np.linspace(0, 1, n + 1)))
        return gini

    def remove_outliers_by_wells(data, name_col, wells_col):
        """Remove outliers based on 95% confidence interval for well counts.

        :param data: DataFrame with one row per well-and-name observation.
            Whole names are kept or dropped together, never individual
            rows, so the surviving frame still has every well of every
            name it keeps.
        :param name_col: Column identifying the gRNA (or other name). Rows
            are grouped on it and the group *size* — the number of wells a
            name appears in — is what the fence is applied to, so the count
            column's values play no part in this filter.
        :param wells_col: Accepted so the call reads symmetrically with the
            enclosing function's ``value_column``, but never read: the well
            count is derived from the group sizes above. Passing a wrong or
            missing column name changes nothing.
        :returns: ``data`` restricted to the names inside the fence. The
            fence is ``1.5 *`` the 5th-to-95th-percentile spread, not the
            interquartile range, so it is far wider than a textbook IQR
            rule and its lower edge is usually negative — in practice only
            unusually widespread names are removed.
        """
        well_counts = data.groupby(name_col, observed=False).size()
        q1 = well_counts.quantile(0.05)
        q3 = well_counts.quantile(0.95)
        iqr_range = q3 - q1
        lower_bound = q1 - 1.5 * iqr_range
        upper_bound = q3 + 1.5 * iqr_range
        valid_names = well_counts[(well_counts >= lower_bound) & (well_counts <= upper_bound)].index
        return data[data[name_col].isin(valid_names)]
    
    combined_data = []
    gini_values = {}

    plt.figure(figsize=(10, 10))

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        # Remove specified keys
        for remove in remove_keys:
            df = df[df[name_column] != remove]
        
        # Remove outliers
        if remove_outliers:
            df = remove_outliers_by_wells(df, name_column, value_column)
        
        values = df[value_column].values
        combined_data.extend(values)
        
        # Calculate Lorenz curve and Gini coefficient
        lorenz = lorenz_curve(values)
        gini = gini_coefficient(values)
        gini_values[f"plate {idx+1}"] = gini
        
        name = f"plate {idx+1} (Gini: {gini:.4f})"
        plt.plot(np.linspace(0, 1, len(lorenz)), lorenz, label=name)

    # Plot combined Lorenz curve
    combined_lorenz = lorenz_curve(np.array(combined_data))
    combined_gini = gini_coefficient(np.array(combined_data))
    gini_values["Combined"] = combined_gini
    
    plt.plot(np.linspace(0, 1, len(combined_lorenz)), combined_lorenz, label=f"Combined (Gini: {combined_gini:.4f})", linestyle='--', color='black')
    
    if x_lim is not None:
        plt.xlim(x_lim)
    
    if y_lim is not None:
        plt.ylim(y_lim)
        
    plt.title('Lorenz Curves')
    plt.xlabel('Cumulative Share of Individuals')
    plt.ylabel('Cumulative Share of Value')
    plt.legend()
    plt.grid(False)
    
    if save:
        save_path = os.path.join(os.path.dirname(csv_files[0]), 'results')
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'lorenz_curve_with_gini.pdf')
        save_file_path = save_figure(plt.gcf(), save_file_path,
                                     bbox_inches='tight')
        print(f"Saved Lorenz Curve: {save_file_path}")
    
    plt.show()

    # Print Gini coefficients
    for plate, gini in gini_values.items():
        print(f"{plate}: Gini Coefficient = {gini:.4f}")

def plot_permutation(permutation_df):
    """Plot a horizontal bar chart of permutation feature importances with error bars.

    :param permutation_df: DataFrame with columns ``feature``,
        ``importance_mean`` and ``importance_std``.
    :returns: The generated ``Figure``.
    """
    num_features = len(permutation_df)
    fig_height = max(8, num_features * 0.3)  # Set a minimum height of 8 and adjust height based on number of features
    fig_width = 10  # Width can be fixed or adjusted similarly
    font_size = max(10, 12 - num_features * 0.2)  # Adjust font size dynamically

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.barh(permutation_df['feature'], permutation_df['importance_mean'], xerr=permutation_df['importance_std'], color="teal", align="center", alpha=0.6)
    ax.set_xlabel('Permutation Importance', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance_df):
    """Plot a horizontal bar chart of raw feature importances.

    :param feature_importance_df: DataFrame with columns ``feature`` and
        ``importance``.
    :returns: The generated ``Figure``.
    """
    num_features = len(feature_importance_df)
    fig_height = max(8, num_features * 0.3)  # Set a minimum height of 8 and adjust height based on number of features
    fig_width = 10  # Width can be fixed or adjusted similarly
    font_size = max(10, 12 - num_features * 0.2)  # Adjust font size dynamically

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color="blue", align="center", alpha=0.6)
    ax.set_xlabel('Feature Importance', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    return fig

def read_and_plot__vision_results(base_dir, y_axis='accuracy', name_split='_time', y_lim=None):
    """Aggregate vision-model test CSVs under ``base_dir`` and plot mean score per model.

    :param base_dir: Root directory containing ``*_test_result.csv``
        files nested per epoch.
    :param y_axis: Metric column to average. Default ``'accuracy'``.
    :param name_split: Substring that splits filename into model name
        and epoch info. Default ``'_time'``.
    :param y_lim: Y-axis limits ``[lo, hi]``. Default ``[0.8, 0.9]``.
    :returns: None
    """
    # List to store data from all CSV files
    if y_lim is None:
        y_lim = [0.8, 0.9]
    data_frames = []

    dst = os.path.join(base_dir, 'result')
    # os.mkdir has no `exists` kwarg (nor `exist_ok`); the old call raised
    # TypeError on every invocation, before any file was ever read.
    os.makedirs(dst, exist_ok=True)

    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_test_result.csv"):
                file_path = os.path.join(root, file)
                # Extract model information from the file name
                file_name = os.path.basename(file_path)
                model = file_name.split(f'{name_split}')[0]

                # The epoch comes from the directory name below; the dropped
                # `file_name.split('_time')[1]` hard-coded the separator instead
                # of using name_split and its result was never read, so it only
                # ever raised IndexError on non-default naming.
                base_folder = os.path.dirname(file_path)
                epoch = os.path.basename(base_folder)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                df['model'] = model
                df['epoch'] = epoch
                
                # Append the data frame to the list
                data_frames.append(df)
    
    # Concatenate all data frames
    if data_frames:
        result_df = pd.concat(data_frames, ignore_index=True)
        
        # Calculate average y_axis per model
        avg_metric = result_df.groupby(
            'model', observed=False)[y_axis].mean().reset_index()
        avg_metric = avg_metric.sort_values(by=y_axis)
        print(avg_metric)
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.bar(avg_metric['model'], avg_metric[y_axis])
        plt.xlabel('Model')
        plt.ylabel(f'{y_axis}')
        plt.title(f'Average {y_axis.capitalize()} per Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.show()
    else:
        print("No CSV files found in the specified directory.")

def jitterplot_by_annotation(src, x_column, y_column, plot_title='Jitter Plot', output_path=None, filter_column=None, filter_values=None):
    """Read measurements + annotation from a spacr DB and plot a class-balanced jitter plot.

    :param src: Path to a spacr experiment directory containing
        ``measurements/measurements.db``.
    :param x_column: Column used as grouping variable (x-axis).
    :param y_column: Numeric column plotted on the y-axis.
    :param plot_title: Title for the plot. Default ``'Jitter Plot'``.
    :param output_path: If set, save the figure to this path; otherwise
        show it.
    :param filter_column: Optional column (or list of columns) to filter
        rows on before plotting.
    :param filter_values: Values (or list of value lists) accepted per
        ``filter_column``.
    :returns: Balanced ``DataFrame`` used for the plot.
    :raises KeyError: if required plate/row/col columns are missing.
    """

    def join_measurments_and_annotation(src, tables = None):
        """Join per-object measurement tables with the ``png_list`` annotation table.

        :param src: spaCR experiment directory; the database is read from
            ``<src>/measurements/measurements.db`` and no other layout is
            supported — pass the experiment folder, not the ``.db`` file.
        :param tables: Object tables to merge, joined on ``prcfo``.
            ``None`` means ``['cell', 'nucleus', 'pathogen', 'cytoplasm']``,
            and every name listed must exist in the database. Listing fewer
            tables is how you plot a run that measured fewer object types.
        :returns: One row per object, with the ``png_list`` crop path
            attached by a left join.
        :raises pandas.errors.MergeError: if ``png_list`` holds more than
            one crop per ``prcfo``; the join is validated ``one_to_one``
            precisely so duplicated crops cannot silently multiply the
            measurement rows and inflate the jitter plot.
        """
        if tables is None:
            tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
        from .io import _read_and_merge_data, _read_db
        db_loc = [src+'/measurements/measurements.db']
        loc = src+'/measurements/measurements.db'
        df, _ = _read_and_merge_data(db_loc, 
                                    tables, 
                                    verbose=True, 
                                    nuclei_limit=True, 
                                    pathogen_limit=True)
        
        paths_df = _read_db(loc, tables=['png_list'])
        # one_to_one: _read_and_merge_data returns one row per object keyed on
        # 'prcfo', and png_list carries at most one crop per that key. A
        # duplicated 'prcfo' in png_list (a crop step that ran twice, or two
        # crop_modes whose object labels collide) would multiply the
        # measurement rows, and the jitter plot would then draw the same cell
        # two or four times as if they were independent observations.
        merged_df = pd.merge(df, paths_df[0], on='prcfo', how='left',
                             validate='one_to_one')
        return merged_df

    # Read the CSV file into a DataFrame
    df = join_measurments_and_annotation(src, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'])

    # Print column names for debugging
    print(f"Generated dataframe with: {df.shape[1]} columns and {df.shape[0]} rows")
    #print("Columns in DataFrame:", df.columns.tolist())

    # Replace NaN values with a specific label in x_column
    df[x_column] = df[x_column].fillna('NaN')

    # Filter the DataFrame if filter_column and filter_values are provided
    if not filter_column is None:
        if isinstance(filter_column, str):
            df = df[df[filter_column].isin(filter_values)]
        if isinstance(filter_column, list):
            for i,val in enumerate(filter_column):
                print(f'hello {len(df)}')
                df = df[df[val].isin(filter_values[i])]

    # Resolve the well-identifier columns instead of hard-coding plate_x/row_x/
    # col_x: spacr.io emits plateID/rowID/columnID, so those literals never
    # match a current database and every call raised KeyError. The merge on
    # 'prcfo' collides on all three, hence the _x/_y suffixes; the bare and _y
    # forms are tried too so the lookup survives a non-colliding merge, and the
    # pre-rename names stay accepted for older frames.
    def _resolve_well_column(frame, *bases):
        for base in bases:
            for candidate in (f'{base}_x', base, f'{base}_y'):
                if candidate in frame.columns:
                    return candidate
        return None

    required_columns = [_resolve_well_column(df, 'plateID', 'plate'),
                        _resolve_well_column(df, 'rowID', 'row'),
                        _resolve_well_column(df, 'columnID', 'col')]
    if any(column is None for column in required_columns):
        raise KeyError("DataFrame does not contain the necessary columns: ['plateID', 'rowID', 'columnID']")

    # Filter to retain rows with non-NaN values in x_column and with matching plate, row, col values
    non_nan_df = df[df[x_column] != 'NaN']
    retained_rows = df[df[required_columns].apply(tuple, axis=1).isin(non_nan_df[required_columns].apply(tuple, axis=1))]

    # Determine the minimum count of examples across all groups in x_column
    min_count = retained_rows[x_column].value_counts().min()
    print(f'Found {min_count} annotated images')

    # Randomly sample min_count examples from each group in x_column
    balanced_df = retained_rows.groupby(
        x_column, observed=False, group_keys=False
    ).sample(n=min_count, random_state=42).reset_index(drop=True)

    # Create the jitter plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=balanced_df, x=x_column, y=y_column, hue=x_column, jitter=True, palette='viridis', dodge=False)
    plt.title(plot_title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    # Customize the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust the position of the x-axis labels to be centered below the data
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
    
    # Save the plot to a file or display it
    if output_path:
        output_path = save_figure(plt.gcf(), output_path,
                                  bbox_inches='tight')
        print(f"Jitter plot saved to {output_path}")
    else:
        plt.show()

    return balanced_df

def create_grouped_plot(df, grouping_column, data_column, graph_type='bar', summary_func='mean', order=None, colors=None, output_dir='./output', save=False, y_lim=None, error_bar_type='std'):
    """Plot grouped data with automatic normality-aware pairwise statistics.

    Runs D'Agostino normality per group, chooses the appropriate
    pairwise test (t-test / Mann-Whitney / ANOVA / Kruskal), adds Tukey
    HSD post-hoc when appropriate, renders the requested plot type and
    optionally persists both plot and stats to ``output_dir``.

    :param df: Source DataFrame.
    :param grouping_column: Categorical grouping variable.
    :param data_column: Numeric column to summarise.
    :param graph_type: One of ``'bar'``, ``'violin'``, ``'jitter'``,
        ``'box'``, ``'jitter_box'``. Default ``'bar'``.
    :param summary_func: Summary function for bar plots. Default
        ``'mean'``.
    :param order: Explicit group ordering. Default: alphabetical.
    :param colors: Colour palette; falls back to a HUSL palette.
    :param output_dir: Save location when ``save=True``.
    :param save: If True, save the plot and per-comparison stats CSV.
    :param y_lim: Two-element y-axis limits.
    :param error_bar_type: ``'std'`` or ``'sem'``. Default ``'std'``.
    :returns: ``(figure, results_df)`` — the displayed matplotlib ``Figure``
        and a DataFrame holding the normality, pairwise and Tukey post-hoc
        rows (``Comparison``, ``Test Statistic``, ``p-value``, ``Test Name``).
    :raises ValueError: if ``error_bar_type`` is not recognised.
    """
    
    # Remove NaN rows in grouping_column
    df = df.dropna(subset=[grouping_column])
    
    # Ensure the output directory exists if save is True
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Sorting and ordering
    if order:
        df[grouping_column] = pd.Categorical(df[grouping_column], categories=order, ordered=True)
    else:
        df[grouping_column] = pd.Categorical(df[grouping_column], categories=sorted(df[grouping_column].unique()), ordered=True)
    
    # Get unique groups
    unique_groups = df[grouping_column].unique()
    
    # Initialize test results
    test_results = []

    # Test normality for each group
    grouped_data = [df.loc[df[grouping_column] == group, data_column] for group in unique_groups]
    normal_p_values = [normaltest(data).pvalue for data in grouped_data]
    normal_stats = [normaltest(data).statistic for data in grouped_data]
    is_normal = all(p > 0.05 for p in normal_p_values)

    # Add normality test results to the results_df
    for group, stat, p_value in zip(unique_groups, normal_stats, normal_p_values):
        test_results.append({
            'Comparison': f'Normality test for {group}',
            'Test Statistic': stat,
            'p-value': p_value,
            'Test Name': 'Normality test'
        })

    # Determine statistical test
    if len(unique_groups) == 2:
        if is_normal:
            stat_test = ttest_ind
            test_name = 'T-test'
        else:
            stat_test = mannwhitneyu
            test_name = 'Mann-Whitney U test'
    else:
        if is_normal:
            stat_test = f_oneway
            test_name = 'One-way ANOVA'
        else:
            stat_test = kruskal
            test_name = 'Kruskal-Wallis test'

    # Perform pairwise statistical tests
    comparisons = list(itertools.combinations(unique_groups, 2))
    p_values = []
    test_statistics = []

    for (group1, group2) in comparisons:
        data1 = df[df[grouping_column] == group1][data_column]
        data2 = df[df[grouping_column] == group2][data_column]
        stat, p = stat_test(data1, data2)
        p_values.append(p)
        test_statistics.append(stat)
        test_results.append({'Comparison': f'{group1} vs {group2}', 'Test Statistic': stat, 'p-value': p, 'Test Name': test_name})
    
    # Post-hoc test (Tukey HSD for ANOVA)
    if is_normal and len(unique_groups) > 2:
        tukey_result = pairwise_tukeyhsd(df[data_column], df[grouping_column], alpha=0.05)
        for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
            test_results.append({
                'Comparison': f'{comparison[0]} vs {comparison[1]}',
                'Test Statistic': None,  # Tukey does not provide a test statistic in the same way
                'p-value': p_value,
                'Test Name': 'Tukey HSD Post-hoc'
            })

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    if colors:
        color_palette = colors
    else:
        color_palette = sns.color_palette("husl", len(unique_groups))
    
    # Choose graph type
    if graph_type == 'bar':
        summary_df = df.groupby(
            grouping_column, observed=False)[data_column].agg(
                [summary_func, 'std', 'sem'])
        
        # Set error bars based on error_bar_type
        if error_bar_type == 'std':
            error_bars = summary_df['std']
        elif error_bar_type == 'sem':
            error_bars = summary_df['sem']
        else:
            raise ValueError(f"Invalid error_bar_type: {error_bar_type}. Choose either 'std' or 'sem'.")

        sns.barplot(
            x=grouping_column, y=summary_func, hue=grouping_column,
            data=summary_df.reset_index(), errorbar=None, order=order,
            palette=color_palette, legend=False)

        # Add error bars (standard deviation or standard error of the mean)
        plt.errorbar(x=np.arange(len(summary_df)), y=summary_df[summary_func], yerr=error_bars, fmt='none', c='black', capsize=5)
    
    elif graph_type == 'violin':
        sns.violinplot(
            x=grouping_column, y=data_column, hue=grouping_column,
            data=df, order=order, palette=color_palette, legend=False)
    elif graph_type == 'jitter':
        sns.stripplot(
            x=grouping_column, y=data_column, hue=grouping_column,
            data=df, jitter=True, order=order, palette=color_palette,
            legend=False)
    elif graph_type == 'box':
        sns.boxplot(
            x=grouping_column, y=data_column, hue=grouping_column,
            data=df, order=order, palette=color_palette, legend=False)
    elif graph_type == 'jitter_box':
        sns.boxplot(
            x=grouping_column, y=data_column, hue=grouping_column,
            data=df, order=order, palette=color_palette, legend=False)
        sns.stripplot(x=grouping_column, y=data_column, data=df, jitter=True, color='black', alpha=0.5, order=order)

    # Create a DataFrame to summarize the test results
    results_df = pd.DataFrame(test_results)

    # Set y-axis start if provided
    if isinstance(y_lim, list) and len(y_lim) == 2:
        plt.ylim(y_lim)

    # If save is True, save the plot and the results table
    if save:
        # No extension: `save_figure` appends the one the figure-format
        # preference selects. Naming the file .png here and then writing a
        # PDF into it was the old behaviour, and it is a file no viewer
        # opens.
        plot_path = os.path.join(output_dir, 'grouped_plot')
        plt.title(f'{test_name} results for {graph_type} plot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = save_figure(plt.gcf(), plot_path)
        print(f"Plot saved to {plot_path}")

        # Save the test results as a CSV file
        results_path = os.path.join(output_dir, 'test_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Test results saved to {results_path}")

    # Show the plot
    plt.show()

    return plt.gcf(), results_df


def _significance_marker(p_value):
    """Return the conventional plot annotation for a statistical p-value."""
    if p_value <= 0.001:
        return '***'
    if p_value <= 0.01:
        return '**'
    if p_value <= 0.05:
        return '*'
    return 'ns'


def _welch_anova(grouped_data):
    """Welch's one-way ANOVA: the >2-group answer when variances differ.

    ``scipy.stats.f_oneway`` assumes equal variance across groups, the same
    assumption Levene's test exists to check. When Levene rejects it, this is
    the standard replacement -- it weights each group by ``n / variance`` and
    corrects the denominator degrees of freedom, so an arm with both a
    different spread and a different size stops borrowing significance from
    the others.

    Computed here rather than through pingouin's ``welch_anova`` because that
    one wants a long-form frame and a formula; this takes the same list of
    arrays every other branch already built.

    :param grouped_data: one 1-D array-like of values per group.
    :returns: ``(F, p)``, or ``(nan, nan)`` when fewer than two groups carry
        the variance the statistic divides by.
    """
    from scipy.stats import f

    groups = [np.asarray(values, dtype=float) for values in grouped_data]
    groups = [values[np.isfinite(values)] for values in groups]
    groups = [values for values in groups
              if values.size >= 2 and np.var(values, ddof=1) > 0]
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    n = np.array([values.size for values in groups], dtype=float)
    mean = np.array([values.mean() for values in groups])
    var = np.array([values.var(ddof=1) for values in groups])

    w = n / var
    w_sum = w.sum()
    grand = (w * mean).sum() / w_sum

    numerator = (w * (mean - grand) ** 2).sum() / (k - 1)
    lam = ((1.0 - w / w_sum) ** 2 / (n - 1.0)).sum()
    denominator = 1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * lam
    statistic = numerator / denominator

    df2 = (k ** 2 - 1.0) / (3.0 * lam)
    p_value = f.sf(statistic, k - 1, df2)
    return float(statistic), float(p_value)


class spacrGraph:
    """Grouped plot + statistical-test helper for spacr experiment DataFrames.

    Wraps preprocessing (aggregation by object / well / plate), normality
    and variance testing, group-wise pairwise stats, and plot rendering
    (bar / jitter / box / violin / jitter_box / jitter_bar / line /
    line_std) in a single object whose output can optionally be persisted
    alongside a CSV of stats.

    :param df: Input DataFrame.
    :param grouping_column: Categorical grouping variable.
    :param data_column: Metric column (or list of columns) to summarise.
    :param graph_type: Plot type. Default ``'bar'``.
    :param summary_func: Aggregator for well/plate level. Default ``'mean'``.
    :param order: Explicit ordering of groups.
    :param colors: Optional colour palette.
    :param output_dir: Save location when ``save=True``.
    :param save: If True, persist plot and stats.
    :param y_lim: Two-element y-axis limits.
    :param log_y: Use log scale for y-axis.
    :param log_x: Use log scale for x-axis.
    :param error_bar_type: ``'std'`` or ``'sem'``. Default ``'std'``.
    :param remove_outliers: Drop 1.5*IQR outliers per group before plotting.
    :param theme: Seaborn palette name. Default ``'pastel'``.
    :param representation: Aggregation level — ``'object'``, ``'well'``
        or ``'plate'``. Default ``'object'``.
    :param paired: Treat groups as paired samples where applicable.
    :param all_to_all: Run every pairwise comparison; ``False`` compares
        each group to ``compare_group``.
    :param compare_group: Reference group when ``all_to_all=False``.
    :param graph_name: Prefix for saved file names.
    """

    def __init__(self, df, grouping_column, data_column, graph_type='bar', summary_func='mean',
                 order=None, colors=None, output_dir='./output', save=False, y_lim=None, log_y=False,
                 log_x=False, error_bar_type='std', remove_outliers=False, theme='pastel', representation='object',
                 paired=False, all_to_all=True, compare_group=None, graph_name=None):
        """Store configuration, set the theme, and preprocess the DataFrame."""

        self.df = df
        self.grouping_column = grouping_column
        #self.order = sorted(df[self.grouping_column].unique().tolist())
        self.order = order or sorted(df[self.grouping_column].dropna().unique().tolist())
        
        self.data_column = data_column if isinstance(data_column, list) else [data_column]
        
        self.graph_type = graph_type
        self.summary_func = summary_func
        #self.order = order
        self.colors = colors
        self.output_dir = output_dir
        self.save = save
        self.error_bar_type = error_bar_type
        self.remove_outliers = remove_outliers
        self.theme = theme
        self.representation = representation
        self.paired = paired
        self.all_to_all = all_to_all
        self.compare_group = compare_group
        self.y_lim = y_lim
        self.graph_name = graph_name
        self.log_x = log_x
        self.log_y = log_y

        self.results_df = pd.DataFrame()
        self.sns_palette = None
        self.fig = None

        self.results_name = str(self.graph_name)+'_'+str(self.data_column[0])+'_'+str(self.grouping_column)+'_'+str(self.graph_type)
        
        self._set_theme()
        self.raw_df = self.df.copy()
        self.df = self.preprocess_data()
        
    def _set_theme(self):
        """Set the Seaborn theme and reorder colors if necessary."""
        integer_list = list(range(1, 81))
        color_order = [7,9,4,0,3,6,2] + integer_list
        self.sns_palette = self._set_reordered_theme(self.theme, color_order, 100)

    def _set_reordered_theme(self, theme='deep', order=None, n_colors=100, show_theme=False):
        """Set and reorder the Seaborn color palette."""
        palette = sns.color_palette(theme, n_colors)
        if order:
            reordered_palette = [palette[i] for i in order]
        else:
            reordered_palette = palette
        if show_theme:
            sns.palplot(reordered_palette)
            plt.show()
        return reordered_palette
  
    def preprocess_data(self):
        """Return a new DataFrame aggregated to the configured representation.

        Drops rows with NaN in the grouping or data columns, aggregates the
        data columns with ``summary_func`` per well (``'prc'``) or per plate
        (``'plateID'``, split out of ``prc`` when needed) — or leaves them
        per object — and makes the grouping column an ordered Categorical.

        :returns: The preprocessed DataFrame; ``__init__`` assigns it back to
            ``self.df`` rather than the frame being modified in place.
        :raises KeyError: if ``representation='plate'`` and neither a
            ``plateID`` nor a ``prc`` column is available.
        :raises ValueError: if ``representation`` is not ``'object'``,
            ``'well'`` or ``'plate'``.
        """
        # 1) Remove NaNs in both the grouping column and each data column
        df = self.df.dropna(subset=[self.grouping_column] + self.data_column)

        # 2) Decide how to handle grouping based on 'representation'
        if self.representation == 'object':
            # -- No grouping at all --
            # We do nothing except keep df as-is after removing NaNs
            group_cols = None

        elif self.representation == 'well':
            # Group by ['prc', grouping_column]
            group_cols = ['prc', self.grouping_column]

        elif self.representation == 'plate':
            # Make sure 'plateID' exists (split from 'prc' if needed)
            if 'plateID' not in df.columns:
                if 'prc' in df.columns:
                    df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)
                else:
                    raise KeyError(
                        "Representation is 'plateID', but no 'plateID' column found. "
                        "Also cannot split from 'prc' because 'prc' column is missing."
                    )
            # If the grouping column IS 'plateID', only group by ['plateID'] once
            if self.grouping_column == 'plateID':
                group_cols = ['plateID']
            else:
                group_cols = ['plateID', self.grouping_column]

        else:
            raise ValueError(f"Unknown representation: {self.representation}, use object, well, or plate")

        # 3) Perform grouping only if group_cols is set
        if group_cols is not None:
            df = df.groupby(
                group_cols, observed=False)[self.data_column].agg(
                    self.summary_func).reset_index()

        # 4) Handle ordering if specified (and if the grouping_column still exists)
        if self.order and (self.grouping_column in df.columns):
            df[self.grouping_column] = pd.Categorical(
                df[self.grouping_column],
                categories=self.order,
                ordered=True
            )
        elif (self.grouping_column in df.columns):
            # Default to sorting unique values
            df[self.grouping_column] = pd.Categorical(
                df[self.grouping_column],
                categories=sorted(df[self.grouping_column].unique()),
                ordered=True
            )

        return df
   
    def remove_outliers_from_plot(self):
        """Remove outliers from the plot but keep them in the data."""
        # self.data_column is a list, so the old code indexed with it and got a
        # DataFrame: the bounds came out as per-column Series and the mask as a
        # DataFrame, which cannot be combined with the group Series. Work one
        # scalar column at a time, and collect the rows to drop instead of
        # dropping inside the loop (that invalidates the group mask's index).
        filtered_df = self.df.copy()
        unique_groups = filtered_df[self.grouping_column].unique()
        drop_index = pd.Index([])
        for group in unique_groups:
            group_mask = filtered_df[self.grouping_column] == group
            for col in self.data_column:
                group_data = filtered_df.loc[group_mask, col]
                q1 = group_data.quantile(0.25)
                q3 = group_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = group_mask & ((filtered_df[col] < lower_bound) | (filtered_df[col] > upper_bound))
                drop_index = drop_index.union(filtered_df.index[outliers])
        return filtered_df.drop(drop_index)

    def perform_normality_tests(self):
        """Perform normality tests for each group and data column."""
        unique_groups = self.df[self.grouping_column].unique()
        normality_results = []

        for column in self.data_column:
            for group in unique_groups:
                data = self.df.loc[self.df[self.grouping_column] == group, column].dropna()
                n_samples = len(data)

                if n_samples < 3 or data.nunique() < 2:
                    reason = (
                        "not enough data" if n_samples < 3
                        else "constant data"
                    )
                    print(f"Skipping normality test for group '{group}' on "
                          f"column '{column}' - {reason}.")
                    normality_results.append({
                        'Comparison': f'Normality test for {group} on {column}',
                        'Test Statistic': None,
                        'p-value': None,
                        'Test Name': 'Skipped',
                        'Column': column,
                        'n': n_samples
                    })
                    continue

                # Choose the appropriate normality test based on the sample size
                if n_samples >= 8:
                    stat, p_value = normaltest(data)
                    test_name = "D'Agostino-Pearson test"
                else:
                    stat, p_value = shapiro(data)
                    test_name = "Shapiro-Wilk test"

                # Store the result for this group and column
                normality_results.append({
                    'Comparison': f'Normality test for {group} on {column}',
                    'Test Statistic': stat,
                    'p-value': p_value,
                    'Test Name': test_name,
                    'Column': column,
                    'n': n_samples
                })

        # No successful normality test is not evidence of normality. This
        # commonly occurs after well-level aggregation leaves one point per
        # group; the old vacuous ``all([])`` then selected a t-test.
        normal_p_values = [
            result['p-value'] for result in normality_results
            if result['p-value'] is not None
        ]
        is_normal = bool(normal_p_values) and all(
            p > 0.05 for p in normal_p_values)

        return is_normal, normality_results
    
    def perform_levene_test(self, unique_groups):
        """Return Levene's test statistic and p-value for the current data column across groups.

        :param unique_groups: Groups to compare.
        :returns: ``(statistic, p_value)`` tuple.
        """
        cols = self.data_column if len(self.data_column) > 1 else [self.data_column[0]]
        # If you only support one column at a time in Levene:
        col = cols[0]
        grouped = [
            self.df.loc[
                self.df[self.grouping_column] == g, col].dropna()
            for g in unique_groups
        ]
        if (len(grouped) < 2
                or any(len(values) < 2 for values in grouped)
                or not any(values.nunique() > 1 for values in grouped)):
            return np.nan, np.nan
        stat, p_value = levene(*grouped)
        return stat, p_value

    def _equal_variance(self, column, unique_groups, alpha=0.05):
        """Does Levene's test allow the equal-variance assumption for ``column``?

        PER COLUMN, unlike :meth:`perform_levene_test`, which only ever looks
        at ``data_column[0]`` while :meth:`perform_statistical_tests` loops
        over every column. Answering once for the first column and applying it
        to the rest would trade one wrong assumption for another.

        :param column: the measurement being tested.
        :param unique_groups: the groups being compared.
        :param alpha: significance at which unequal variance is accepted.
        :returns: True when variances may be treated as equal -- including
            when Levene cannot be computed at all, because the equal-variance
            test is the historical behaviour and silently switching to Welch's
            on a degenerate group would change old numbers for no evidence.
        """
        grouped = [
            self.df.loc[self.df[self.grouping_column] == group,
                        column].dropna()
            for group in unique_groups
        ]
        if (len(grouped) < 2
                or any(len(values) < 2 for values in grouped)
                or not any(values.nunique() > 1 for values in grouped)):
            return True
        try:
            _stat, p_value = levene(*grouped)
        except Exception:
            return True
        if not np.isfinite(p_value):
            return True
        return bool(p_value >= alpha)

    def perform_statistical_tests(self, unique_groups, is_normal):
        """Perform statistical tests separately for each data column.

        :param unique_groups: Groups to compare. Two groups get a pairwise
            test, more get an omnibus test across all of them, but the
            ``Comparison`` label always names only the first two.
        :param is_normal: If True, use the parametric test (t-test, paired
            t-test or one-way ANOVA); otherwise Mann-Whitney, paired
            Wilcoxon or Kruskal-Wallis.
        :returns: One result dict per data column, with keys ``Comparison``,
            ``Test Statistic``, ``p-value``, ``Test Name``, ``Column``,
            ``n_object`` and ``n_well``. Statistic and p-value are ``nan``
            when the data cannot support the chosen test.
        """
        test_results = []
        for column in self.data_column:  # Iterate over each data column
            grouped_data = [
                self.df.loc[
                    self.df[self.grouping_column] == group, column].dropna()
                for group in unique_groups
            ]
            if len(unique_groups) == 2:  # For two groups: class_0 vs class_1
                if is_normal:
                    parametric_testable = (
                        all(len(values) >= 2 for values in grouped_data)
                        and any(values.nunique() > 1
                                for values in grouped_data)
                    )
                    if self.paired and parametric_testable:
                        parametric_testable = (
                            len(grouped_data[0]) == len(grouped_data[1])
                            and np.unique(
                                grouped_data[0].to_numpy()
                                - grouped_data[1].to_numpy()
                            ).size > 1
                        )
                    if not parametric_testable:
                        stat, p = np.nan, np.nan
                        test_name = (
                            'Paired T-test' if self.paired else 'T-test')
                    elif self.paired:
                        stat, p = pg.ttest(grouped_data[0], grouped_data[1], paired=True).iloc[0][['T', 'p-val']]
                        test_name = 'Paired T-test'
                    else:
                        # Levene's test used to be computed and thrown away,
                        # and this line ran Student's t-test regardless
                        # (scipy's equal_var defaults to True). So the
                        # assumption was tested, the answer discarded, and the
                        # test that assumes it run anyway -- which inflates
                        # significance exactly when the groups differ in
                        # spread, the common case for a treated arm.
                        equal_var = self._equal_variance(column, unique_groups)
                        stat, p = ttest_ind(grouped_data[0], grouped_data[1],
                                            equal_var=equal_var)
                        test_name = ('T-test' if equal_var
                                     else "Welch's T-test")
                else:
                    if self.paired:
                        # pingouin's wilcoxon statistic column is 'W-val'; 'T'
                        # belongs to pg.ttest above, so this raised KeyError.
                        test_name = 'Paired Wilcoxon test'
                        paired_testable = (
                            all(len(values) > 0 for values in grouped_data)
                            and len(grouped_data[0]) == len(grouped_data[1])
                            and np.any(
                                grouped_data[0].to_numpy()
                                != grouped_data[1].to_numpy())
                        )
                        if not paired_testable:
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = pg.wilcoxon(
                                grouped_data[0], grouped_data[1]
                            ).iloc[0][['W-val', 'p-val']]
                    else:
                        test_name = 'Mann-Whitney U test'
                        if any(len(values) == 0 for values in grouped_data):
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = mannwhitneyu(
                                grouped_data[0], grouped_data[1])
            else:
                if is_normal:
                    parametric_testable = (
                        all(len(values) >= 2 for values in grouped_data)
                        and any(values.nunique() > 1
                                for values in grouped_data)
                    )
                    if parametric_testable:
                        # Same correction as the two-group branch: f_oneway is
                        # the equal-variance ANOVA, so when Levene rejects
                        # that, use Welch's.
                        equal_var = self._equal_variance(column, unique_groups)
                        if equal_var:
                            stat, p = f_oneway(*grouped_data)
                            test_name = 'One-way ANOVA'
                        else:
                            stat, p = _welch_anova(grouped_data)
                            test_name = "Welch's ANOVA"
                    else:
                        stat, p = np.nan, np.nan
                        test_name = 'One-way ANOVA'
                else:
                    test_name = 'Kruskal-Wallis test'
                    if (any(len(values) == 0 for values in grouped_data)
                            or pd.concat(grouped_data).nunique() < 2):
                        stat, p = np.nan, np.nan
                    else:
                        stat, p = kruskal(*grouped_data)

            test_results.append({
                'Comparison': f'{unique_groups[0]} vs {unique_groups[1]} ({column})',
                'Test Statistic': stat,
                'p-value': p,
                'Test Name': test_name,
                'Column': column,
                # n_object FROM raw_df, n_well from self.df. Both used to come
                # from `grouped_data`, which is built from self.df -- and
                # self.df is what `preprocess_data` AGGREGATED. With
                # representation='well' that made the two columns the same
                # number: a plate of 4,382 cells in 12 wells reported
                # n_object = 12.
                #
                # The post-hoc rows in the same CSV already did it correctly
                # (n_object from raw_df, n_well from self.df), so the two row
                # types disagreed about the same comparison in the same file
                # -- which is how you get a Methods section citing whichever
                # was read first.
                'n_object': sum(
                    len(self.raw_df[self.raw_df[self.grouping_column] == group]
                        [column].dropna())
                    for group in unique_groups),
                'n_well': sum(
                    len(self.df[self.df[self.grouping_column] == group])
                    for group in unique_groups)})

        return test_results
    
    def perform_posthoc_tests(self, is_normal, unique_groups):
        """Perform post-hoc tests for multiple groups based on all_to_all flag.

        :param is_normal: Outcome of the normality check, which selects the
            family of test: True runs Tukey HSD, False runs Dunn's test
            with an automatically chosen p-adjustment. It only matters when
            post-hoc testing runs at all — see ``unique_groups``.
        :param unique_groups: The distinct group labels. Only its *length*
            is read; the comparisons themselves are rebuilt from
            ``self.df[self.grouping_column]``, so reordering or renaming
            entries has no effect. Fewer than three groups returns an empty
            list, as does ``self.all_to_all`` being False, because pairwise
            correction is meaningless for a single comparison.
        :returns: A list of per-comparison dicts with ``Comparison``,
            ``Test Statistic`` (always ``None`` — neither test reports one),
            ``p-value``, ``Test Name`` and the ``n_object`` / ``n_well``
            counts; empty when no post-hoc test was warranted. Only
            ``self.data_column[0]`` is tested, so extra data columns are
            ignored here.
        """

        from .sp_stats import choose_p_adjust_method

        posthoc_results = []
        if is_normal and len(unique_groups) > 2 and self.all_to_all:
            #tukey_result = pairwise_tukeyhsd(self.df[self.data_column], self.df[self.grouping_column], alpha=0.05)
            tukey_result = pairwise_tukeyhsd(self.df[self.data_column[0]], self.df[self.grouping_column], alpha=0.05)
            posthoc_results = []
            for comparison, p_value in zip(tukey_result._results_table.data[1:], tukey_result.pvalues):
                raw_data1 = self.raw_df[self.raw_df[self.grouping_column] == comparison[0]][self.data_column]
                raw_data2 = self.raw_df[self.raw_df[self.grouping_column] == comparison[1]][self.data_column]

                posthoc_results.append({
                    'Comparison': f'{comparison[0]} vs {comparison[1]}',
                    'Test Statistic': None,  # Tukey does not provide a test statistic
                    'p-value': p_value,
                    'Test Name': 'Tukey HSD Post-hoc',
                    'n_object': len(raw_data1) + len(raw_data2),
                    'n_well': len(self.df[self.df[self.grouping_column] == comparison[0]]) + len(self.df[self.df[self.grouping_column] == comparison[1]])})
            return posthoc_results
        
        elif len(unique_groups) > 2 and self.all_to_all:
            print('performing_dunns')

            # Prepare data for Dunn's test in long format
            long_data = self.df[[self.data_column[0], self.grouping_column]].dropna()

            p_adjust_method = choose_p_adjust_method(num_groups=len(long_data[self.grouping_column].unique()),num_data_points=len(long_data) // len(long_data[self.grouping_column].unique()))

            # Perform Dunn's test with Bonferroni correction
            dunn_result = sp.posthoc_dunn(
                long_data, 
                val_col=self.data_column[0], 
                group_col=self.grouping_column, 
                p_adjust=p_adjust_method
            )

            for group_a, group_b in zip(*np.triu_indices_from(dunn_result, k=1)):
                raw_data1 = self.raw_df[self.raw_df[self.grouping_column] == dunn_result.index[group_a]][self.data_column]
                raw_data2 = self.raw_df[self.raw_df[self.grouping_column] == dunn_result.columns[group_b]][self.data_column]

                posthoc_results.append({
                    'Comparison': f"{dunn_result.index[group_a]} vs {dunn_result.columns[group_b]}",
                    'Test Statistic': None,  # Dunn's test does not return a specific test statistic
                    'p-value': dunn_result.iloc[group_a, group_b],  # Extract the p-value from the matrix
                    'Test Name': "Dunn's Post-hoc",
                    'p_adjust_method': p_adjust_method,
                    'n_object': len(raw_data1) + len(raw_data2),  # Total objects
                    # Both terms must index the frame with the mask. Without the
                    # outer self.df[...] the second term is the mask itself, so
                    # its len() is the row count of the whole frame.
                    'n_well': len(self.df[self.df[self.grouping_column] == dunn_result.index[group_a]]) +
                            len(self.df[self.df[self.grouping_column] == dunn_result.columns[group_b]])})

            return posthoc_results

        return posthoc_results
    
    def create_plot(self, ax=None):
        """Build the plot for the chosen graph type onto ``self.fig``.

        Nothing is displayed: retrieve the figure with :meth:`get_figure`
        (and the statistics with :meth:`get_results`), or call ``plt.show()``.

        :param ax: Existing ``Axes`` to draw into, for placing this graph in
            a panel of a larger figure. ``self.fig`` is then set to that
            axes' parent figure, so a later ``save=True`` writes the whole
            enclosing figure, not this panel alone. ``None`` creates a fresh
            figure sized from the group count and ``bar_width`` — and note
            that with a single ``data_column`` the standardisation pass
            still calls ``ax.figure.set_size_inches``, which resizes a
            shared figure underneath its other panels.
        """

        def _generate_tabels(unique_groups):
            """Generate row labels and a symbol table for multi-level grouping."""
            # Create row labels: Include the grouping column and data columns
            row_labels = [self.grouping_column] + self.data_column

            # Initialize table data
            table_data = []

            # Create the grouping row: Alternate each group for every data column
            grouping_row = []
            for _ in self.data_column:
                for group in unique_groups:
                    grouping_row.append(group)
            table_data.append(grouping_row)  # Add the grouping row to the table

            # Create symbol rows for each data column
            for column in self.data_column:
                column_row = []  # Initialize a row for this column
                for data_col in self.data_column:  # Iterate over data columns to align with the structure
                    for group in unique_groups:
                        # Assign '+' if the column matches, otherwise assign '-'
                        if column == data_col:
                            column_row.append('+')
                        else:
                            column_row.append('-')
                table_data.append(column_row)  # Add this row to the table

            # Transpose the table to align with the plot layout
            transposed_table = list(map(list, zip(*table_data)))
            return row_labels, transposed_table


        def _place_symbols(row_labels, transposed_table, x_positions, ax):
            """
            Places symbols and row labels aligned under the bars or jitter points on the graph.
            
            Parameters:
            - row_labels: List of row titles to be displayed along the y-axis.
            - transposed_table: Data to be placed under each bar/jitter as symbols.
            - x_positions: X-axis positions for each group to align the symbols.
            - ax: The matplotlib Axes object where the plot is drawn.
            """
            # Get plot dimensions and adjust for different plot sizes
            y_axis_min = ax.get_ylim()[0]  # Minimum y-axis value (usually 0)
            symbol_start_y = y_axis_min - 0.05 * (ax.get_ylim()[1] - y_axis_min)  # Adjust a bit below the x-axis

            # Calculate spacing for the table rows (adjust as needed)
            y_spacing = 0.04  # Adjust this for better spacing between rows

            # Determine the leftmost x-position for row labels (align with the y-axis)
            label_x_pos = ax.get_xlim()[0] - 0.3  # Adjust offset from the y-axis

            # Place row labels vertically aligned with symbols
            for row_idx, title in enumerate(row_labels):
                y_pos = symbol_start_y - (row_idx * y_spacing)  # Calculate vertical position for each label
                ax.text(label_x_pos, y_pos, title, ha='right', va='center', fontsize=12, fontweight='regular')

            # Place symbols under each bar or jitter point based on x-positions
            for idx, (x_pos, column_data) in enumerate(zip(x_positions, transposed_table)):
                for row_idx, text in enumerate(column_data):
                    y_pos = symbol_start_y - (row_idx * y_spacing)  # Adjust vertical spacing for symbols
                    ax.text(x_pos, y_pos, text, ha='center', va='center', fontsize=12, fontweight='regular')

            # Redraw to apply changes
            ax.figure.canvas.draw()
                    
        def _get_positions(self, ax):
            if self.graph_type in ['bar','jitter_bar']: 
                x_positions = [np.mean(bar.get_paths()[0].vertices[:, 0]) for bar in ax.collections if hasattr(bar, 'get_paths')]

            elif self.graph_type == 'violin':
                x_positions = [np.mean(violin.get_paths()[0].vertices[:, 0]) for violin in ax.collections if hasattr(violin, 'get_paths')]

            elif self.graph_type in ['box', 'jitter_box']:
                x_positions = list(set(line.get_xdata().mean() for line in ax.lines if line.get_linestyle() == '-'))                

            elif self.graph_type == 'jitter': 
                x_positions = [np.mean(collection.get_offsets()[:, 0]) for collection in ax.collections if collection.get_offsets().size > 0]
            
            elif self.graph_type in ['line', 'line_std']:
                x_positions = []
            
            return x_positions
        
        def _draw_comparison_lines(ax, x_positions):
            """Draw comparison lines and annotate significance based on results_df."""
            if self.results_df.empty:
                print("No comparisons available to annotate.")
                return

            y_max = max([bar.get_height() for bar in ax.patches])
            ax.set_ylim(0, y_max * 1.3)

            for idx, row in self.results_df.iterrows():
                group1, group2 = row['Comparison'].split(' vs ')
                p_value = row['p-value']

                significance = _significance_marker(p_value)

                # Find the x positions of the compared groups
                x1 = x_positions[unique_groups.tolist().index(group1)]
                x2 = x_positions[unique_groups.tolist().index(group2)]

                # Stagger lines to avoid overlap
                line_y = y_max + (0.1 * y_max) * (idx + 1)

                # Draw the comparison line
                ax.plot([x1, x1, x2, x2], [line_y - 0.02, line_y, line_y, line_y - 0.02], lw=1.5, c='black')

                # Add the significance marker
                ax.text((x1 + x2) / 2, line_y, significance, ha='center', va='bottom', fontsize=12)

        # Optional: Remove outliers for plotting
        # THE TRIM IS FOR THE PICTURE, NOT FOR THE TEST, and it used to be
        # for both. `remove_outliers_from_plot` drops 1.5*IQR points PER
        # GROUP, and it ran here -- before the normality test, before the
        # comparison and before the post-hoc, all of which then read the
        # trimmed frame.
        #
        # That inflates significance in the one direction nobody checks.
        # Removing a group's tails shrinks its standard deviation, so the
        # t-statistic grows for a difference in means that has not changed.
        # Worse, trimming PER GROUP removes exactly the points that make two
        # groups overlap. A caller asking not to have one point stretch the
        # y-axis was silently also asking for a smaller p-value.
        #
        # No shipped caller passes remove_outliers=True, so nothing published
        # came through here -- but `spacrGraph` is public API and the
        # parameter is documented, so this is a live trap rather than a
        # historical one.
        #
        # The statistics now run on every point, and only the drawing is
        # trimmed. The results table says so, because a reader looking at a
        # trimmed plot beside a p-value has to know which one used what.
        stats_df = self.df
        self.df_melted = pd.melt(stats_df, id_vars=[self.grouping_column], value_vars=self.data_column,var_name='Data Column', value_name='Value')
        unique_groups = stats_df[self.grouping_column].unique()
        is_normal, normality_results = self.perform_normality_tests()
        # The equal-variance check now happens inside
        # `perform_statistical_tests`, PER COLUMN, and decides between the
        # Student and Welch forms. It used to be computed here into
        # `levene_stat, levene_p` and never read again, while the test that
        # depends on the assumption ran regardless. `perform_levene_test`
        # stays as public API for callers that want the statistic itself.
        test_results = self.perform_statistical_tests(unique_groups, is_normal)
        posthoc_results = self.perform_posthoc_tests(is_normal, unique_groups)
        self.results_df = pd.DataFrame(normality_results + test_results + posthoc_results)

        # Now, and only now, trim what gets drawn.
        if self.remove_outliers:
            self.df = self.remove_outliers_from_plot()
            if not self.results_df.empty:
                self.results_df['outliers_removed_from_plot_only'] = True
            trimmed = len(stats_df) - len(self.df)
            if trimmed > 0:
                print(f"remove_outliers: {trimmed} of {len(stats_df)} points "
                      f"are hidden from the plot. THE STATISTICS ABOVE USED "
                      f"ALL {len(stats_df)}.")
            self.df_melted = pd.melt(
                self.df, id_vars=[self.grouping_column],
                value_vars=self.data_column, var_name='Data Column',
                value_name='Value')

        #num_groups = len(self.data_column)*len(self.grouping_column)
        num_groups = len(self.df[self.grouping_column].unique())
        self.bar_width = 0.4
        spacing_between_groups = self.bar_width/0.5

        self.fig_width = (num_groups * self.bar_width) + (spacing_between_groups * num_groups)
        self.fig_height = self.fig_width/2
        
        if  self.graph_type in ['line','line_std']:
            self.fig_height, self.fig_width = 10, 10 

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(self.fig_height, self.fig_width))
        else:
            self.fig = ax.figure

        if len(self.data_column) == 1:
            self.hue=self.grouping_column
            self.jitter_bar_dodge = False
        else:
            self.hue='Data Column'
            self.jitter_bar_dodge = True
        
        # Handle the different plot types based on `graph_type`
        if self.graph_type == 'bar':
            self._create_bar_plot(ax)
        elif self.graph_type == 'jitter':
            self._create_jitter_plot(ax)
        elif self.graph_type == 'box':
            self._create_box_plot(ax)
        elif self.graph_type == 'violin':
            self._create_violin_plot(ax)
        elif self.graph_type == 'jitter_box':
            self._create_jitter_box_plot(ax)
        elif self.graph_type == 'jitter_bar':
            self._create_jitter_bar_plot(ax)
        elif self.graph_type == 'line':
            self._create_line_graph(ax)
        elif self.graph_type == 'line_std':
            self._create_line_with_std_area(ax)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}") 
        
        if len(self.data_column) == 1:
            num_groups = len(self.df[self.grouping_column].unique())
            self._standerdize_figure_format(ax=ax, num_groups=num_groups, graph_type=self.graph_type)

        # Set y-axis start
        if isinstance(self.y_lim, list):
            if len(self.y_lim) == 2:
                ax.set_ylim(self.y_lim[0], self.y_lim[1])
            elif len(self.y_lim) == 1:
                ax.set_ylim(self.y_lim[0], None)

        sns.despine(ax=ax, top=True, right=True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels, loc='center left',
                bbox_to_anchor=(1, 0.5), title='Data Column')
        
        if not self.graph_type in ['line','line_std']:
            ax.set_xlabel('')

        x_positions = _get_positions(self, ax)
        
        if len(self.data_column) == 1 and not self.graph_type in ['line','line_std']:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        elif len(self.data_column) > 1 and not self.graph_type in ['line','line_std']:
            ax.set_xticks([])
            ax.tick_params(bottom=False)
            ax.set_xticklabels([])
            legend_ax = self.fig.add_axes([0.1, -0.2, 0.62, 0.2])  # Position the table closer to the graph
            legend_ax.set_axis_off()

            row_labels, table_data = _generate_tabels(unique_groups)
            _place_symbols(row_labels, table_data, x_positions, ax)
            
        #_draw_comparison_lines(ax, x_positions)    
        
        if self.save:
            self._save_results()

        ax.margins(x=0.12)

    def _standerdize_figure_format(self, ax, num_groups, graph_type):
        """
        Adjusts the figure layout (size, bar width, jitter, and spacing) based on the number of groups.

        Parameters:
        - ax: The matplotlib Axes object.
        - num_groups: Number of unique groups.
        - graph_type: The type of graph (e.g., 'bar', 'jitter', 'box', etc.).

        Returns:
        - None. Modifies the figure and Axes in place.
        """
        if graph_type in ['line', 'line_std']:
            print("Skipping layout adjustment for line graphs.")
            return  # Skip layout adjustment for line graphs
        
        correction_factor = 4

        # Set figure size to ensure it remains square with a minimum size
        fig_size = max(6, num_groups * 2)  / correction_factor
        
        if fig_size < 10:
            fig_size = 10
        
        
        ax.figure.set_size_inches(fig_size, fig_size)

        # Configure layout based on the number of groups
        bar_width = min(0.8, 1.5 / num_groups) / correction_factor
        jitter_amount = min(0.1, 0.2 / num_groups) / correction_factor
        jitter_size = max(50 / num_groups, 200)

        # Adjust axis limits to ensure bars are centered with respect to group labels
        ax.set_xlim(-0.5, num_groups - 0.5)

        # Set ticks to match the group labels in your DataFrame
        #group_labels = self.df[self.grouping_column].unique()
        #group_labels = self.order
        #ax.set_xticks(range(len(group_labels)))
        #ax.set_xticklabels(group_labels, rotation=45, ha='right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Customize elements based on the graph type
        if graph_type == 'bar':
            # Adjust bars' width and position
            for bar in ax.patches:
                bar.set_width(bar_width)
                bar.set_x(bar.get_x() - bar_width / 2)

        elif graph_type in ['jitter', 'jitter_bar', 'jitter_box']:
            # Adjust jitter points' position and size
            for coll in ax.collections:
                offsets = coll.get_offsets()
                offsets[:, 0] += jitter_amount  # Shift jitter points slightly
                coll.set_offsets(offsets)
                coll.set_sizes([jitter_size]  * len(offsets))  # Adjust point size dynamically

        elif graph_type in ['box', 'violin']:
            # Adjust box width for consistent spacing
            for artist in ax.artists:
                artist.set_width(bar_width)

        # Adjust legend and axis labels
        ax.tick_params(axis='x', labelsize=max(10, 15 - num_groups // 2))
        ax.tick_params(axis='y', labelsize=max(10, 15 - num_groups // 2))

        if ax.get_legend():
            ax.get_legend().set_bbox_to_anchor((1.05, 1)) #loc='upper left',borderaxespad=0.
            ax.get_legend().prop.set_size(max(8, 12 - num_groups // 3))

        # Redraw the figure to apply changes
        ax.figure.canvas.draw()
        
    def _create_bar_plot(self, ax):
        """Helper method to create a bar plot with consistent bar thickness and centered error bars."""
        # Flatten DataFrame: Combine grouping column and data column into one group if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        summary_df = self.df_melted.groupby(
            [x_axis_column], observed=False
        ).agg(mean=('Value', 'mean'), std=('Value', 'std'),
              sem=('Value', 'sem')).reset_index()
        self.summary_df = summary_df.copy()
        sns.barplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend, ax=ax,
            dodge=self.jitter_bar_dodge, errorbar=None, order=plot_order)
        
        # Adjust the bar width manually
        if len(self.data_column) > 1:
            bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
            target_width = self.bar_width * 2
            for bar in bars:
                bar.set_width(target_width)  # Set new width
                # Center the bar on its x-coordinate
                bar.set_x(bar.get_x() - target_width / 2)
            
        # Adjust error bars alignment with bars
        bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
        for bar, (_, row) in zip(bars, summary_df.iterrows()):
            x_bar = bar.get_x() + bar.get_width() / 2
            err = row[self.error_bar_type]
            ax.errorbar(x=x_bar, y=bar.get_height(), yerr=err, fmt='none', c='black', capsize=5, lw=2)
    
        # Set legend and labels
        ax.set_xlabel(self.grouping_column)

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_plot(self, ax):
        """Helper method to create a jitter plot (strip plot) with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str)  + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None  # Disable hue to avoid two-level grouping
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        # Create the jitter plot
        self.summary_df = self.df_melted.copy()
        sns.stripplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax,
            alpha=0.6, size=16, order=plot_order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)
       
        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_line_graph(self, ax):
        """Helper method to create a line graph with one line per group based on epochs and accuracy."""
        #display(self.df)
        # Ensure epoch is used on the x-axis and accuracy on the y-axis
        x_axis_column = self.data_column[0]
        y_axis_column = self.data_column[1]

        if self.log_y:
            self.df[y_axis_column] = np.log10(self.df[y_axis_column])
        
        if self.log_x:
            self.df[x_axis_column] = np.log10(self.df[x_axis_column])
        
        # Set hue to the grouping column to get one line per group
        hue = self.grouping_column

        # Check if the required columns exist in the DataFrame
        required_columns = [x_axis_column, y_axis_column, self.grouping_column]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Create the line graph with one line per group
        self.summary_df = self.df.copy()
        line_palette = self.sns_palette[
            :max(1, self.df[hue].nunique(dropna=True))]
        sns.lineplot(
            data=self.df, x=x_axis_column, y=y_axis_column, hue=hue,
            palette=line_palette, ax=ax, marker='o', linewidth=1,
            markersize=6)

        # Adjust axis labels
        ax.set_xlabel(f"{x_axis_column}")
        ax.set_ylabel(f"{y_axis_column}")

    def _create_line_with_std_area(self, ax):
        """Helper method to create a line graph with shaded area representing standard deviation."""

        x_axis_column = self.data_column[0]
        y_axis_column = self.data_column[1]
        y_axis_column_mean = f"mean_{y_axis_column}"
        y_axis_column_std = f"std_{y_axis_column_mean}"
        
        if self.log_y:
            self.df[y_axis_column] = np.log10(self.df[y_axis_column])
        
        if self.log_x:
            self.df[x_axis_column] = np.log10(self.df[x_axis_column])

        # Pivot the DataFrame to get mean and std for each epoch across plates
        summary_df = self.df.pivot_table(index=x_axis_column,values=y_axis_column,aggfunc=['mean', 'std']).reset_index()
        
        # Flatten MultiIndex columns (result of pivoting)
        summary_df.columns = [x_axis_column, y_axis_column_mean, y_axis_column_std]
            
        # Plot the mean accuracy as a line
        self.summary_df = summary_df.copy()
        sns.lineplot(data=summary_df,x=x_axis_column,y=y_axis_column_mean,ax=ax,marker='o',linewidth=1,markersize=0,color='blue',label=y_axis_column_mean)


        # Fill the area representing the standard deviation
        ax.fill_between(summary_df[x_axis_column],summary_df[y_axis_column_mean] - summary_df[y_axis_column_std],summary_df[y_axis_column_mean] + summary_df[y_axis_column_std],color='blue',  alpha=0.1 )

        # Adjust axis labels
        ax.set_xlabel(f"{x_axis_column}")
        ax.set_ylabel(f"{y_axis_column}")
        
    def _create_box_plot(self, ax):
        """Helper method to create a box plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        # Create the box plot
        self.summary_df = self.df_melted.copy()
        sns.boxplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            ax=ax, order=plot_order)

        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)

        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')
    
    def _create_violin_plot(self, ax):
        """Helper method to create a violin plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        # Create the violin plot
        self.summary_df = self.df_melted.copy()
        sns.violinplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            ax=ax, order=plot_order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)
        ax.set_ylabel('Value')
    
        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_bar_plot(self, ax):
        """Helper method to create a bar plot with consistent bar thickness and centered error bars."""
        # Flatten DataFrame: Combine grouping column and data column into one group if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        summary_df = self.df_melted.groupby(
            [x_axis_column], observed=False
        ).agg(mean=('Value', 'mean'), std=('Value', 'std'),
              sem=('Value', 'sem')).reset_index()
        self.summary_df = summary_df
        sns.barplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend, ax=ax,
            dodge=self.jitter_bar_dodge, errorbar=None, order=plot_order)
        sns.stripplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax,
            alpha=0.6, edgecolor='white', linewidth=1, size=16,
            order=plot_order)
        
        # Adjust the bar width manually
        if len(self.data_column) > 1:
            bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
            target_width = self.bar_width * 2
            for bar in bars:
                bar.set_width(target_width)  # Set new width
                # Center the bar on its x-coordinate
                bar.set_x(bar.get_x() - target_width / 2)
            
        # Adjust error bars alignment with bars
        #bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
        #for bar, (_, row) in zip(bars, summary_df.iterrows()):
        #    x_bar = bar.get_x() + bar.get_width() / 2
        #    err = row[self.error_bar_type]
        #    ax.errorbar(x=x_bar, y=bar.get_height(), yerr=err, fmt='none', c='black', capsize=5, lw=2)
    
        # Set legend and labels
        ax.set_xlabel(self.grouping_column)

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')

    def _create_jitter_box_plot(self, ax):
        """Helper method to create a box plot with consistent spacing."""
        # Combine grouping column and data column if needed
        if len(self.data_column) > 1:
            self.df_melted['Combined Group'] = (self.df_melted[self.grouping_column].astype(str) + " - " + self.df_melted['Data Column'].astype(str))
            x_axis_column = 'Combined Group'
            hue = None
            # order must name levels of the column used for x. With multiple
            # data columns x is 'Combined Group', so passing the raw group
            # names selected nothing and seaborn drew an empty plot.
            plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
            ax.set_ylabel('Value')
        else:
            x_axis_column = self.grouping_column
            ax.set_ylabel(self.data_column[0])
            hue = self.hue
            plot_order = self.order

        plot_hue = hue or x_axis_column
        plot_palette = self.sns_palette[:max(1, len(plot_order))]
        show_legend = hue is not None
        # Create the box plot
        self.summary_df = self.df_melted.copy()
        sns.boxplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            ax=ax, order=plot_order)
        sns.stripplot(
            data=self.df_melted, x=x_axis_column, y='Value',
            hue=plot_hue, palette=plot_palette, legend=show_legend,
            dodge=self.jitter_bar_dodge, jitter=self.bar_width, ax=ax,
            alpha=0.6, edgecolor='white', linewidth=1, size=12,
            order=plot_order)
    
        # Adjust legend and labels
        ax.set_xlabel(self.grouping_column)

        # Manage the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

        if self.log_y:
            ax.set_yscale('log')
        if self.log_x:
            ax.set_xscale('log')
        
    def _save_results(self):
        """Save figure, stats, and all data used to generate the plot."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Figure
        plot_path = os.path.join(self.output_dir, f"{self.results_name}.pdf")
        # dpi=600 was hard-coded here, and this is exactly the figure that
        # cannot always take it: `_standerdize_figure_format` pins the
        # canvas to >=10 inches square and grows it with the group count.
        # `save_figure` follows the preference and says so when the number
        # asked for is not deliverable at this size.
        plot_path = save_figure(self.fig, plot_path, bbox_inches='tight',
                                transparent=True)

        # Stats
        stats_path = os.path.join(self.output_dir, f"{self.results_name}_stats.csv")
        self.results_df.to_csv(stats_path, index=False)
        
        # Data
        data_path = os.path.join(self.output_dir, f"{self.results_name}_data.csv")
        self.df.to_csv(data_path, index=False)

        # Data: raw -> preprocessed -> melted (plot input) -> summary (if available)
        #self.raw_df.to_csv(os.path.join(self.output_dir, f"{self.results_name}_raw.csv"), index=False)
        #self.df.to_csv(os.path.join(self.output_dir, f"{self.results_name}_preprocessed.csv"),index=False)
        
        #if hasattr(self, 'df_melted') and self.df_melted is not None:
        #    self.df_melted.to_csv(os.path.join(self.output_dir, f"{self.results_name}_plotdata.csv"),index=False)
        
        if hasattr(self, 'summary_df') and self.summary_df is not None:
            data_path = os.path.join(self.output_dir, f"{self.results_name}_summary.csv")
            self.summary_df.to_csv(data_path, index=False)
            print(f"Data -> {data_path}")
            
        print(f"Plot  -> {plot_path}")
        print(f"Stats -> {stats_path}")

    def get_results(self):
        """Return the results dataframe."""
        return self.results_df
    
    def get_figure(self):
        """Return the generated figure."""
        return self.fig

def plot_data_from_db(settings):
    """Read one or more measurement DBs, annotate conditions and render a ``spacrGraph`` plot.

    Concatenates results across source directories, derives the
    ``recruitment`` column if requested, drops missing rows, then hands
    the data to :class:`spacrGraph` for statistics + plotting.

    :param settings: Settings dict. See
        ``settings.set_default_plot_data_from_db`` for accepted keys
        (notably ``src``, ``database``, ``table_names``, ``data_column``,
        ``grouping_column``, ``graph_type``, ``graph_name``).
    :returns: The plotted DataFrame, or ``None`` when the requested data
        or grouping column is missing.
    :raises ValueError: if ``src`` is neither a string nor a list.
    """
    from .io import _read_db, _read_and_merge_data
    from .utils import annotate_conditions, save_settings
    from .settings import set_default_plot_data_from_db

    """
    Extracts the specified table from the SQLite database and plots a specified column.

    Args:
        db_path (str): The path to the SQLite database.
        table_names (str): The name of the table to extract.
        data_column (str): The column to plot from the table.

    Returns:
        df (pd.DataFrame): The extracted table as a DataFrame.
    """

    settings = set_default_plot_data_from_db(settings)
    
    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    elif isinstance(settings['src'], list):
        srcs = settings['src']
    else:
        raise ValueError("src must be a string or a list of strings.")
    
    if isinstance(settings['database'], str):
        settings['database'] = [settings['database'] for _ in range(len(srcs))]
    
    settings['dst'] = os.path.join(srcs[0], 'results')
    
    save_settings(settings, name=f"{settings['graph_name']}_plot_settings_db", show=True)

    dfs = []
    for i, src in enumerate(srcs):
        db_loc = os.path.join(src, 'measurements', settings['database'][i])
        print(f"Database: {db_loc}")
        if settings['table_names'] in ['saliency_image_correlations']:
            print(f"Database table: {settings['table_names']}")
            [df1] = _read_db(db_loc, tables=[settings['table_names']])
        else:
            df1, _ = _read_and_merge_data(locs=[db_loc],
                                    tables = settings['table_names'],
                                    verbose=settings['verbose'],
                                    nuclei_limit=settings['nuclei_limit'],
                                    pathogen_limit=settings['pathogen_limit'])
            
        dft = annotate_conditions(df1, 
                                cells=settings['cell_types'], 
                                cell_loc=settings['cell_plate_metadata'], 
                                pathogens=settings['pathogen_types'],
                                pathogen_loc=settings['pathogen_plate_metadata'],
                                treatments=settings['treatments'], 
                                treatment_loc=settings['treatment_plate_metadata'])
        dfs.append(dft)
        
    df = pd.concat(dfs, axis=0)
    df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
    
    # Category B, not a per-item skip: the user asked for these conditions,
    # so a missing annotation column means every well below is pooled under
    # the wrong label. Historically this printed one line and produced a plot
    # that looked entirely fine. The ledger makes the damage countable and
    # SPACR_STRICT_ERRORS turns it into a hard stop.
    annotation_ledger = RunLedger('plot_data_from_db:annotation')
    for meta_key, column, label in (
            ('cell_plate_metadata', 'host_cells', 'host_cell'),
            ('pathogen_plate_metadata', 'pathogen', 'pathogen'),
            ('treatment_plate_metadata', 'treatment', 'treatment')):
        if settings[meta_key] != None:
            try:
                df = df.dropna(subset=column)
            except Exception as e:
                print(f"Could not drop NaN values from '{label}' column: {e}")
                annotation_ledger.record_failure(column, stage='annotate_conditions', exc=e)
                raise_if_strict(
                    f"{meta_key} was set but the {column!r} column was never "
                    f"created, so rows cannot be filtered to the requested "
                    f"conditions; every group in this plot is suspect.",
                    exc=e, settings=settings)
            else:
                annotation_ledger.record_success(column, stage='annotate_conditions')
    annotation_ledger.finalize()

    if settings['data_column'] == 'recruitment':
        pahtogen_measurement = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"]
        cytoplasm_measurement = df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
        df['recruitment'] = pahtogen_measurement / cytoplasm_measurement
        
    if settings['data_column'] not in df.columns:
        print(f"Data column {settings['data_column']} not found in DataFrame.")
        print(f'Please use one of the following columns:')
        for col in df.columns:
            print(col)
        display(df)
        return None
    
    df = df.dropna(subset=settings['data_column'])
        
    if settings['grouping_column'] not in df.columns:
        print(f"Grouping column {settings['grouping_column']} not found in DataFrame.")
        print(f'Please use one of the following columns:')
        for col in df.columns:
            print(col)
        display(df)
        return None
    
    df = df.dropna(subset=settings['grouping_column'])

    src = srcs[0] 
    dst = os.path.join(src, 'results', settings['graph_name'])
    os.makedirs(dst, exist_ok=True)
    
    spacr_graph = spacrGraph(
        df=df,                                       # Your DataFrame
        grouping_column=settings['grouping_column'], # Column for grouping the data (x-axis)
        data_column=settings['data_column'],         # Column for the data (y-axis)
        graph_type=settings['graph_type'],           # Type of plot ('bar', 'box', 'violin', 'jitter')
        graph_name=settings['graph_name'],           # Name of the plot
        summary_func='mean',                         # Function to summarize data (e.g., 'mean', 'median')
        colors=None,                                 # Custom colors for the plot (optional)
        output_dir=dst,                              # Directory to save the plot and results
        save=settings['save'],                       # Whether to save the plot and results
        y_lim=settings['y_lim'],                     # Starting point for y-axis (optional)
        error_bar_type='std',                        # Type of error bar ('std' or 'sem')
        representation=settings['representation'],
        theme=settings['theme'],                     # Seaborn color palette theme (e.g., 'pastel', 'muted')
    )

    # Create the plot
    spacr_graph.create_plot()

    # Get the figure object if needed
    fig = spacr_graph.get_figure()
    plt.show()

    # Optional: Get the results DataFrame containing statistical test results
    results_df = spacr_graph.get_results()
    return fig, results_df, df

def plot_data_from_csv(settings):
    """Load per-plate CSVs, filter/outlier-clean and render a ``spacrGraph`` plot.

    :param settings: Settings dict — see
        ``settings.get_plot_data_from_csv_default_settings`` for keys
        (``src``, ``data_column``, ``grouping_column``, ``keep_groups``,
        ``remove_outliers``, ``graph_type``, ``graph_name``, ...).
    :returns: ``(fig, results_df, df)`` — the figure, stats DataFrame
        and plotted DataFrame.
    :raises ValueError: if ``src`` is not a string or list.
    """
    from .utils import remove_outliers_by_group
    """
    Extracts the specified table from the SQLite database and plots a specified column.

    Args:
        db_path (str): The path to the SQLite database.
        table_names (str): The name of the table to extract.
        data_column (str): The column to plot from the table.

    Returns:
        df (pd.DataFrame): The extracted table as a DataFrame.
    """
    

    def filter_rows_by_column_values(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
        """Return a filtered DataFrame where only rows with the column value in the list are kept.

        :param df: Frame to filter; it is not modified, and the result is a
            ``.copy()`` so later assignment to it raises no
            ``SettingWithCopyWarning``.
        :param column: Column to test. Must exist, or ``KeyError`` is
            raised — here it is the caller's ``grouping_column``.
        :param values: Values to keep, matched with ``isin`` so comparison
            is exact and type-sensitive: the string ``'1'`` will not match
            an integer ``1`` read from the CSV. An empty list keeps nothing
            and yields an empty frame rather than passing everything
            through.
        :returns: A new filtered DataFrame.
        """
        return df[df[column].isin(values)].copy()
    
    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    elif isinstance(settings['src'], list):
        srcs = settings['src']
    else:
        raise ValueError("src must be a string or a list of strings.")
    
    dfs = []
    for i, src in enumerate(srcs):
        dft = pd.read_csv(src)
        if 'plateID' not in dft.columns:
            dft['plateID'] = f"plate{i+1}"
            dft['common'] = 'spacr'
        dfs.append(dft)

    df = pd.concat(dfs, axis=0)
    
    if 'prc' in df.columns:
        # Check if 'plateID', 'rowID', and 'columnID' are all missing from df.columns
        if not all(col in df.columns for col in ['plate', 'rowID', 'columnID']):
            try:
                # Split 'prc' into 'plateID', 'rowID', and 'columnID'
                df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)
            except Exception as e:
                # Category B: without plateID/rowID/columnID every downstream
                # grouping falls back to whatever happens to be in the frame,
                # so the plot groups by the wrong thing rather than not at all.
                print(f"Could not split the prc column: {e}")
                raise_if_strict(
                    "The 'prc' column could not be split into "
                    "plateID/rowID/columnID; any grouping in this plot is "
                    "computed on the wrong keys.",
                    exc=e, settings=settings)

    if 'keep_groups' in settings.keys():
        if isinstance(settings['keep_groups'], str):
            settings['keep_groups'] = [settings['keep_groups']]
        elif isinstance(settings['keep_groups'], list):
            df = filter_rows_by_column_values(df, settings['grouping_column'], settings['keep_groups'])
            
    if settings['remove_outliers']:
        df = remove_outliers_by_group(df, settings['grouping_column'], settings['data_column'], method='iqr', threshold=1.5)
    
    if settings['verbose']:       
        display(df)
    
    df = df.dropna(subset=settings['data_column'])
    df = df.dropna(subset=settings['grouping_column'])
    src = srcs[0] 
    dst = os.path.join(os.path.dirname(src), 'results', settings['graph_name'])
    os.makedirs(dst, exist_ok=True)
    
    #data_csv = os.path.join(dst, f"{settings['graph_name']}_data.csv")
    #df.to_csv(data_csv, index=False)
    
    spacr_graph = spacrGraph(
        df=df,                                       # Your DataFrame
        grouping_column=settings['grouping_column'], # Column for grouping the data (x-axis)
        data_column=settings['data_column'],         # Column for the data (y-axis)
        graph_type=settings['graph_type'],           # Type of plot ('bar', 'box', 'violin', 'jitter')
        graph_name=settings['graph_name'],           # Name of the plot
        summary_func='mean',                         # Function to summarize data (e.g., 'mean', 'median')
        colors=None,                                 # Custom colors for the plot (optional)
        output_dir=dst,                              # Directory to save the plot and results
        save=settings['save'],                       # Whether to save the plot and results
        y_lim=settings['y_lim'],                     # Starting point for y-axis (optional)
        log_y=settings['log_y'],                     # Log-transform the y-axis
        log_x=settings['log_x'],                     # Log-transform the x-axis
        error_bar_type='std',                        # Type of error bar ('std' or 'sem')
        representation=settings['representation'],
        theme=settings['theme'],                     # Seaborn color palette theme (e.g., 'pastel', 'muted')
    )

    # Create the plot
    spacr_graph.create_plot()

    fig = spacr_graph.get_figure()
    plt.show()

    # Optional: Get the results DataFrame containing statistical test results
    results_df = spacr_graph.get_results()
    return fig, results_df

def plot_region(settings):
    """Render mask overlay, cropped PNG grid and activation-map grid for one FOV.

    Reads the FOV's merged NPY, resolves its PNG crops and activation
    maps from the measurements and activation DBs, and writes the three
    figures under ``<src>/results/<name>/`` when possible — in the
    configured figure format, so PDF only while that is the preference.

    :param settings: Settings dict with ``src``, ``name``, ``channels``,
        ``cell_channel``, ``nucleus_channel``, ``pathogen_channel``,
        ``percentiles``, ``activation_mode``, ``activation_db``,
        ``mode``, ``export_tiffs``.
    :returns: Tuple ``(fig_mask_overlay, fig_png_grid,
        fig_activation_grid)`` — any element may be ``None`` when the
        corresponding assets were not found.
    """

    def _sort_paths_by_basename(paths):
        """Return ``paths`` sorted by their basename."""
        return sorted(paths, key=lambda path: os.path.basename(path))

    def save_figure_as_pdf(fig, path):
        """Save ``fig`` in the user's chosen figure format.

        Named for the format it used to hard-code; it follows the
        preference now, like every other figure the user keeps, and
        `save_figure` creates the parent directory itself.

        :param fig: Figure to write. It is left open, so the caller can
            still return it to the notebook after saving.
        :param path: Destination path. Its extension is rewritten to
            whichever format the preference selected, so passing a ``.pdf``
            name does not force PDF; missing parent directories are
            created. The path actually written is printed, not returned.
        """
        path = save_figure(fig, path, bbox_inches='tight')
        print(f"Saved {path}")

    from .io import _read_db
    from .utils import correct_paths
    fov_path = os.path.join(settings['src'], 'merged', settings['name'])
    name = os.path.splitext(settings['name'])[0]
    
    db_path = os.path.join(settings['src'], 'measurements', 'measurements.db')
    paths_df = _read_db(db_path, tables=['png_list'])[0]
    paths_df, _ = correct_paths(df=paths_df, base_path=settings['src'], folder='data')
    paths_df = paths_df[paths_df['png_path'].str.contains(name, na=False)]

    activation_mode = f"{settings['activation_mode']}_list"
    activation_db_path = os.path.join(settings['src'], 'measurements', settings['activation_db'])
    activation_paths_df = _read_db(activation_db_path, tables=[activation_mode])[0]
    activation_db = os.path.splitext(settings['activation_db'])[0]
    base_path=os.path.join(settings['src'], 'datasets',activation_db) 
    activation_paths_df, _ = correct_paths(df=activation_paths_df, base_path=base_path, folder=settings['activation_mode'])
    activation_paths_df = activation_paths_df[activation_paths_df['png_path'].str.contains(name, na=False)]

    png_paths = _sort_paths_by_basename(paths_df['png_path'].tolist())
    activation_paths = _sort_paths_by_basename(activation_paths_df['png_path'].tolist())

    
    if activation_paths:
        fig_3 = plot_image_grid(image_paths=activation_paths, percentiles=settings['percentiles'])
    else:
        fig_3 = None
        print(f"Could not find any cropped PNGs")
    if png_paths:
        fig_2 = plot_image_grid(image_paths=png_paths, percentiles=settings['percentiles'])
    else:
        fig_2 = None
        print(f"Could not find any activation maps")
    
    print('fov_path', fov_path)
    fig_1 = plot_image_mask_overlay(file=fov_path,
                                    channels=settings['channels'],
                                    cell_channel=settings['cell_channel'],
                                    nucleus_channel=settings['nucleus_channel'],
                                    pathogen_channel=settings['pathogen_channel'],
                                    figuresize=10,
                                    percentiles=settings['percentiles'],
                                    thickness=3, 
                                    save_pdf=True, 
                                    mode=settings['mode'],
                                    export_tiffs=settings['export_tiffs'])
    
    dst = os.path.join(settings['src'], 'results', name)
    
    if not fig_1 == None:
        save_figure_as_pdf(fig_1, os.path.join(dst, f"{name}_mask_overlay.pdf"))
    if not fig_2 == None:
        save_figure_as_pdf(fig_2, os.path.join(dst, f"{name}_png_grid.pdf"))
    if not fig_3 == None:
        save_figure_as_pdf(fig_3, os.path.join(dst, f"{name}_activation_grid.pdf"))
    
    return fig_1, fig_2, fig_3

def plot_image_grid(image_paths, percentiles):
    """Render a square grid of percentile-normalised images with a black background.

    :param image_paths: Image files to display; extra tiles are filled
        black.
    :param percentiles: Two-element percentile pair used to normalise
        each channel.
    :returns: The generated ``Figure``.
    """

    from PIL import Image
    import matplotlib.pyplot as plt
    import math

    def _normalize_image(image, percentiles=(2, 98)):
        """ Normalize the image to the given percentiles for each channel independently, preserving the input type (either PIL.Image or numpy.ndarray)."""
        
        # Check if the input is a PIL image and convert it to a NumPy array
        is_pil_image = isinstance(image, Image.Image)
        if is_pil_image:
            image = np.array(image)

        # If the image is single-channel, normalize directly
        if image.ndim == 2:
            v_min, v_max = np.percentile(image, percentiles)
            normalized_image = np.clip((image - v_min) / (v_max - v_min), 0, 1)
        else:
            # If multi-channel, normalize each channel independently
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[-1]):
                v_min, v_max = np.percentile(image[..., c], percentiles)
                normalized_image[..., c] = np.clip((image[..., c] - v_min) / (v_max - v_min), 0, 1)

        # If the input was a PIL image, convert the result back to PIL format
        if is_pil_image:
            # Ensure the image is converted back to 8-bit range (0-255) for PIL
            normalized_image = (normalized_image * 255).astype(np.uint8)
            return Image.fromarray(normalized_image)

        return normalized_image

    N = len(image_paths)
    # Calculate the smallest square grid size to fit all images
    grid_size = math.ceil(math.sqrt(N))  

    # Create the square grid of subplots with a black background
    fig, axs = plt.subplots(
        grid_size, grid_size,
        figsize=(grid_size * 2, grid_size * 2),
        facecolor='black',  # Set figure background to black
        # A single image gives a 1x1 grid, which matplotlib otherwise collapses
        # to a bare Axes with no .flatten().
        squeeze=False
    )

    # Flatten axs in case of a 2D array
    axs = axs.flatten()

    for i, img_path in enumerate(image_paths):
        ax = axs[i]

        # Load the image
        img = Image.open(img_path)
        img = _normalize_image(img, percentiles)

        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Fill any unused subplots with black
    for j in range(i + 1, len(axs)):
        axs[j].imshow([[0, 0, 0]], cmap='gray')  # Black square
        axs[j].axis('off')  # Hide axes

    # Adjust layout to minimize white space
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    return fig

def overlay_masks_on_images(img_folder, normalize=True, resize=True, save=False, plot=False, thickness=2):
    """Overlay ``masks/*`` outlines onto matching images from ``img_folder``.

    :param img_folder: Folder containing images; masks live in
        ``img_folder/masks`` with matching filenames.
    :param normalize: If True, percentile-normalise images before
        blending. Default ``True``.
    :param resize: If True, resize the blended overlay to 1000x1000.
        Default ``True``.
    :param save: If True, write PNGs to ``img_folder/overlay/``.
        Default ``False``.
    :param plot: If True, show each overlay via matplotlib.
        Default ``False``.
    :param thickness: Contour line thickness in pixels. Default ``2``.
    :returns: None
    """

    def normalize_image(image):
        """Normalize the image to the 1st and 99th percentiles.

        :param image: Image array of any numeric dtype, typically the raw
            16-bit TIFF. The percentiles are taken over the whole array, so
            a multi-channel image is stretched by one shared window rather
            than per channel, and the brightest and darkest 1% saturate.
            A flat or near-constant image puts both percentiles on the same
            value; the rescale then divides by zero and the ``uint8`` cast
            turns the resulting ``nan`` into an undefined value, so guard
            empty fields upstream.
        :returns: A ``uint8`` array on 0-255, ready to blend with the mask
            overlay.
        """
        lower, upper = np.percentile(image, [1, 99])
        image = np.clip((image - lower) / (upper - lower), 0, 1)
        return (image * 255).astype(np.uint8)

    
    mask_folder = os.path.join(img_folder,'masks')    
    overlay_folder = os.path.join(img_folder, "overlay")
    if save and not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)

    # Get common filenames in both image and mask folders
    image_filenames = set(os.listdir(img_folder))
    mask_filenames = set(os.listdir(mask_folder))
    common_filenames = image_filenames.intersection(mask_filenames)

    if not common_filenames:
        print("No matching filenames found in both folders.")
        return

    for filename in common_filenames:
        # Load image and mask
        img_path = os.path.join(img_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        image = tiff.imread(img_path)
        mask = tiff.imread(mask_path)

        # Normalize the image if requested
        if normalize:
            image = normalize_image(image)

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Resize the mask if it doesn't match the image size
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Generate contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to RGB if grayscale
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
            
        # Draw contours with alpha blending
        overlay = image_rgb.copy()
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness)
        blended = cv2.addWeighted(overlay, 0.7, image_rgb, 0.3, 0)
        
        # Resize the final overlay if requested
        if resize:
            blended = cv2.resize(blended, (1000, 1000), interpolation=cv2.INTER_AREA)

        # Save the overlay if requested
        if save:
            save_path = os.path.join(overlay_folder, filename)
            write_image_rgb(save_path, blended)
        
        if plot:
            # Display the result
            plt.figure(figsize=(10, 10))
            plt.imshow(blended)
            plt.title(f"Overlay: {filename}")
            plt.axis('off')
            plt.show()

def graph_importance(settings):
    """Concatenate feature-importance CSVs and hand off to :class:`spacrGraph` for plotting.

    :param settings: Settings dict with ``csvs`` (single path or list),
        ``grouping_column``, ``data_column``, ``graph_type``, ``save``.
    :returns: None (side-effects: plot shown, artefacts saved).
    """
    from .settings import set_graph_importance_defaults
    from .utils import save_settings
    
    # Wrap a scalar path: the guard used to assign the value to itself, so a
    # single path string fell through and was iterated character by character.
    # Only str/PathLike are wrapped -- a tuple or Series of paths already works.
    if isinstance(settings['csvs'], (str, os.PathLike)):
        settings['csvs'] = [settings['csvs']]

    settings['src'] = os.path.dirname(settings['csvs'][0])
    
    settings = set_graph_importance_defaults(settings)
    save_settings(settings, name='graph_importance')
    
    dfs = []
    for path in settings['csvs']:
        dft = pd.read_csv(path)
        dfs.append(dft)

    df = pd.concat(dfs)
    
    if not all(col in df.columns for col in (settings['grouping_column'], settings['data_column'])):
        print(f"grouping {settings['grouping_column']} and data {settings['data_column']} columns must be in {df.columns.to_list()}")
        return
    
    output_dir = os.path.dirname(settings['csvs'][0])
    
    spacr_graph = spacrGraph(
        df=df,                                     
        grouping_column=settings['grouping_column'],
        data_column=settings['data_column'],   
        graph_type=settings['graph_type'],   
        graph_name=settings['grouping_column'],
        summary_func='mean',                         
        colors=None,                                
        output_dir=output_dir,                              
        save=settings['save'],                       
        y_lim=None,                     
        error_bar_type='std',                       
        representation='object',
        theme='muted',                    
    )

    # Create the plot
    spacr_graph.create_plot()

    plt.show()
    
#: Which column carries the unit of replication for each declared level.
#: `level` chose the FIGURE and nothing else; it now chooses the denominator
#: of the test as well, which is the whole point of declaring it.
REPLICATION_UNIT = {"well": None, "plate": "plateID", "plateid": "plateID"}


def _unit_column(level, prc_column):
    """The column whose distinct values are the independent observations.

    ``well`` resolves to whatever the caller passed as ``prc_column`` -- the
    well identifier is not always spelled ``prc`` -- and ``plate`` to
    ``plateID``.
    """
    key = str(level or "object").strip().lower()
    if key == "object":
        return None
    return REPLICATION_UNIT.get(key, prc_column) or prc_column


def proportions_per_unit(df, group_column, bin_column, unit_column):
    """Each unit's share of every bin, one row per unit.

    :returns: a frame with ``group_column``, ``unit_column`` and one column
        per bin holding a proportion in [0, 1]. Units contributing no
        objects do not appear.
    """
    # Deduplicated, because a caller may GROUP BY the unit -- the
    # replication tables group by `prc`, which is also the well. Passing
    # 'prc' to groupby twice puts it in the index twice, and `reset_index`
    # then raises "cannot insert prc, already exists" instead of choosing.
    keys = list(dict.fromkeys([group_column, unit_column, bin_column]))
    counts = (df.groupby(keys, observed=True).size()
              .unstack(fill_value=0))
    totals = counts.sum(axis=1)
    proportions = counts.div(totals.where(totals > 0), axis=0)
    proportions = proportions.dropna(how="all")
    # `unstack` leaves the bin values as COLUMN names, and a caller's frame
    # can already carry a column spelled like one of the index levels --
    # `prc` is both the unit and, in the replication tables, a plain column.
    # `reset_index` then raises "cannot insert prc, already exists" rather
    # than choosing, so the clash is removed before it can happen.
    clashing = [name for name in proportions.index.names
                if name in proportions.columns]
    if clashing:
        proportions = proportions.drop(columns=clashing)
    return proportions.reset_index()


def _compare_groups(samples):
    """The test spaCR already uses for this shape, chosen the same way.

    Two groups: Shapiro decides normal, then Levene decides Student's or
    Welch's; non-normal goes to Mann-Whitney. More than two: ANOVA, Welch's
    ANOVA or Kruskal-Wallis on the same two questions. Returns
    ``(name, statistic, p)``, or ``(name, nan, nan)`` when there are too few
    units to test -- which is itself the answer worth printing.
    """
    samples = [np.asarray(s, dtype=float) for s in samples]
    samples = [s[np.isfinite(s)] for s in samples]
    if len(samples) < 2 or any(len(s) < 2 for s in samples):
        return "too few units", float("nan"), float("nan")

    normal = True
    for sample in samples:
        if len(sample) >= 3 and float(np.ptp(sample)) > 0:
            try:
                if shapiro(sample)[1] < 0.05:
                    normal = False
            except ValueError:
                pass

    spread_differs = False
    if all(float(np.ptp(s)) > 0 for s in samples):
        try:
            spread_differs = levene(*samples)[1] < 0.05
        except ValueError:
            spread_differs = False

    if len(samples) == 2:
        if not normal:
            stat, p = mannwhitneyu(*samples, alternative="two-sided")
            return "Mann-Whitney U", float(stat), float(p)
        stat, p = ttest_ind(*samples, equal_var=not spread_differs)
        return ("Welch's T-test" if spread_differs else "T-test",
                float(stat), float(p))
    if not normal:
        stat, p = kruskal(*samples)
        return "Kruskal-Wallis", float(stat), float(p)
    if spread_differs:
        stat, p = _welch_anova(samples)
        return "Welch's ANOVA", float(stat), float(p)
    stat, p = f_oneway(*samples)
    return "One-way ANOVA", float(stat), float(p)


def proportion_test_by_unit(df, group_column, bin_column, unit_column):
    """Compare conditions on their PER-UNIT proportions, one row per bin.

    The object-level chi-squared asks whether 20,000 objects came from one
    distribution. Objects in a well share a treatment, a transfection, an
    imaging session and a monolayer, so that is not the question anyone
    asked, and its p-value is smaller than the experiment supports by orders
    of magnitude. This asks the question the design supports: do the WELLS
    differ, with n = the number of wells.
    """
    if unit_column == group_column:
        # The unit of replication IS the thing being compared, so every
        # group holds exactly one unit and there is nothing to test across.
        # Saying so beats returning a p-value computed from one number each.
        return pd.DataFrame([{
            "test": f"not applicable: the groups ARE the {unit_column}s",
            "bin": None,
            "unit": unit_column,
            "n": int(df[unit_column].nunique()) if unit_column in df else 0,
            "n_per_group": "1 each",
            "statistic": float("nan"),
            "p_value": float("nan"),
        }])

    table = proportions_per_unit(df, group_column, bin_column, unit_column)
    bins = [c for c in table.columns if c not in (group_column, unit_column)]
    groups = list(dict.fromkeys(table[group_column].tolist()))

    rows = []
    for bin_value in bins:
        samples = [table.loc[table[group_column] == g, bin_value].to_numpy()
                   for g in groups]
        name, stat, p = _compare_groups(samples)
        rows.append({
            "test": f"{name} on per-{unit_column} proportions",
            "bin": bin_value,
            "unit": unit_column,
            "n": int(table[unit_column].nunique()),
            "n_per_group": ", ".join(f"{g}={len(s)}"
                                     for g, s in zip(groups, samples)),
            "statistic": stat,
            "p_value": p,
        })
    return pd.DataFrame(rows)


def proportion_mixed_model(df, group_column, bin_column, unit_column):
    """A binomial GLM on the per-object outcome, standard errors clustered by unit.

    The proportions test throws away how many objects each well contributed;
    this keeps them while still charging the degrees of freedom the DESIGN
    supports, by clustering on the unit. Reported beside the other two
    because when it disagrees with them, the disagreement is the finding.
    """
    if unit_column == group_column:
        return pd.DataFrame([{
            "test": f"not applicable: the groups ARE the {unit_column}s",
            "bin": None, "unit": unit_column,
            "n": int(df[unit_column].nunique()) if unit_column in df else 0,
            "n_per_group": f"objects={len(df)}",
            "statistic": float("nan"), "p_value": float("nan"),
        }])

    bins = list(dict.fromkeys(df[bin_column].dropna().tolist()))
    groups = list(dict.fromkeys(df[group_column].dropna().tolist()))
    rows = []
    for bin_value in bins:
        outcome = (df[bin_column] == bin_value).astype(float).to_numpy()
        design = pd.get_dummies(df[group_column].astype(str),
                                drop_first=True, dtype=float)
        if design.empty or len(groups) < 2:
            rows.append({"test": "binomial GLM, clustered by " + unit_column,
                         "bin": bin_value, "unit": unit_column,
                         "n": int(df[unit_column].nunique()),
                         "n_per_group": f"objects={len(df)}",
                         "statistic": float("nan"), "p_value": float("nan")})
            continue
        design = sm.add_constant(design, has_constant="add")
        try:
            fit = sm.GLM(outcome, design.to_numpy(),
                         family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": df[unit_column].astype(str).to_numpy()})
            terms = [i for i, name in enumerate(design.columns)
                     if name != "const"]
            wald = fit.wald_test(np.eye(len(design.columns))[terms],
                                 scalar=True)
            statistic, p = float(wald.statistic), float(wald.pvalue)
        except Exception as error:      # singular, separated, or too few clusters
            print(f"mixed model for bin {bin_value!r} did not fit: {error}")
            statistic = p = float("nan")
        rows.append({
            "test": f"binomial GLM, standard errors clustered by {unit_column}",
            "bin": bin_value,
            "unit": unit_column,
            "n": int(df[unit_column].nunique()),
            "n_per_group": f"objects={len(df)}",
            "statistic": statistic,
            "p_value": p,
        })
    return pd.DataFrame(rows)



def plot_proportion_stacked_bars(settings, df, group_column, bin_column, prc_column='prc', level='object', cmap='viridis'):
    """Plot stacked proportion bars per group with chi-squared and pairwise stats.

    :param settings: Settings dict — ``verbose`` toggles pairwise
        chi-squared verbosity.
    :param df: Long-format DataFrame with categorical ``group_column``
        and ``bin_column``.
    :param group_column: Group axis of the stacked bars.
    :param bin_column: Categorical column stacked within each bar.
    :param prc_column: Per-well identifier used when aggregating at the
        well or plate level. Default ``'prc'``.
    :param level: Aggregation level — ``'object'`` for direct counts, or
        ``'well'`` / ``'plateID'`` for per-well means with SD bars.
    :param cmap: Matplotlib colormap. Default ``'viridis'``.
    :returns: ``(results_df, pairwise_results, fig)`` — chi-squared
        summary, pairwise comparison table and the plot figure.
    """

    from .sp_stats import chi_pairwise

    # Calculate contingency table for overall chi-squared test
    raw_counts = df.groupby([group_column, bin_column], observed=True).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(raw_counts)
    print(f"Chi-squared test statistic (raw data): {chi2:.4f}")
    print(f"p-value (raw data): {p:.4e}")

    # Perform pairwise comparisons
    pairwise_results = chi_pairwise(raw_counts, verbose=settings.get('verbose', False))

    # Plot based on level setting.
    #
    # 'plate' USED TO FALL THROUGH HERE. The check read
    # `level in ['well', 'plateID']`, while the setting's own tooltip offers
    # 'object', 'well' and 'plate' -- so a user who asked for plate-level bars
    # got object-level pooling instead: every object in one bar per condition,
    # no per-plate averaging and no SD whiskers, which is a different figure
    # answering a different question with nothing to say it had happened.
    #
    # An unknown level is now named rather than silently pooled, because
    # falling back to 'object' is exactly what made the typo invisible.
    _level = str(level or 'object').strip().lower()
    _AGGREGATED = {'well': prc_column, 'plate': 'plateID', 'plateid': 'plateID'}
    if _level not in _AGGREGATED and _level != 'object':
        raise ValueError(
            f"level={level!r} is not one of 'object', 'well' or 'plate'. "
            f"Pooling every object would have answered a different question "
            f"than the one asked.")
    if _level in _AGGREGATED:
        prc_column = _AGGREGATED[_level]
        if prc_column not in df.columns:
            # 'plateID' used to group by `prc` -- the WELL column -- so a
            # plate-level request averaged wells and called them plates. It
            # now groups by the plate, which means the plate column has to
            # be present, and naming the missing one beats a bare KeyError
            # raised from inside a groupby.
            raise ValueError(
                f"level={level!r} groups by {prc_column!r}, which this table "
                f"does not have. Available: {sorted(df.columns)[:12]}")
        well_proportions = (
            df.groupby([group_column, prc_column, bin_column], observed=True)
            .size()
            .groupby(level=[0, 1], observed=False)
            .apply(lambda x: x / x.sum())
            .unstack(fill_value=0)
        )
        mean_proportions = well_proportions.groupby(
            group_column, observed=False).mean()
        std_proportions = well_proportions.groupby(
            group_column, observed=False).std()

        mean_proportions.plot(
            kind='bar', stacked=True, yerr=std_proportions, capsize=5, colormap=cmap, figsize=(12, 8)
        )
        plt.title(f'Proportion of Volume Bins by Group (Mean ± SD across {"plates" if _level != "well" else "wells"})')
    else:
        group_counts = df.groupby([group_column, bin_column], observed=True).size()
        group_totals = group_counts.groupby(
            level=0, observed=False).sum()
        proportions = group_counts / group_totals
        proportion_df = proportions.unstack(fill_value=0)

        proportion_df.plot(kind='bar', stacked=True, colormap=cmap, figsize=(12, 8))
        plt.title('Proportion of Volume Bins by Group')

    plt.xlabel('Group')
    plt.ylabel('Proportion')

    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    fig = plt.gcf()

    # THREE NUMBERS, EACH LABELLED WITH ITS UNIT AND ITS N.
    #
    # The chi-squared above is computed over OBJECTS and was the only number
    # this function reported, at every level -- object, well and plate gave
    # byte-identical chi2 and p, while the level tooltip promised that "the
    # reported statistics always treat the well as the unit of replication".
    # It never did.
    #
    # The old number is kept as the first row rather than replaced: every
    # figure already published came from it, and a reader comparing an old
    # result with a new one has to be able to see why they differ.
    results_df = pd.DataFrame({
        'chi_squared_stat': [chi2],
        'p_value': [p],
        'degrees_of_freedom': [dof],
        'test': ['chi-squared on object counts'],
        'unit': ['object'],
        'n': [int(len(df))],
        'statistic': [float(chi2)],
    })

    # `level='object'` still gets the well-level tests when a well column is
    # there. Pooling objects does not make them independent, so the honest
    # denominator is reported whether or not it was asked for.
    unit_column = _unit_column(level, prc_column) or prc_column
    if unit_column in df.columns:
        extra = [proportion_test_by_unit(df, group_column, bin_column,
                                         unit_column),
                 proportion_mixed_model(df, group_column, bin_column,
                                        unit_column)]
        results_df = pd.concat([results_df] + extra, ignore_index=True)
        for _, row in pd.concat(extra, ignore_index=True).iterrows():
            print(f"{row['test']} [bin {row['bin']}, n={row['n']} "
                  f"{row['unit']}]: p = {row['p_value']:.4e}")
    else:
        print(f"no {unit_column!r} column, so only the object-level "
              f"chi-squared could be computed; objects in one well are not "
              f"independent and this p-value is smaller than the experiment "
              f"supports")

    return results_df, pairwise_results, fig
    

def create_venn_diagram(file1, file2, gene_column="gene", filter_coeff=0.1, save=True, save_path=None):
    """Compute a two-set gene overlap from CSVs and draw its Venn diagram.

    :param file1: First CSV file.
    :param file2: Second CSV file.
    :param gene_column: Column identifying genes. Default ``'gene'``.
    :param filter_coeff: Threshold on the ``coefficient`` column —
        positive filters ``> threshold``, negative filters ``< threshold``.
    :param save: If True, save as PDF; requires ``save_path``.
    :param save_path: Output PDF path when ``save`` is True.
    :returns: ``{'overlap', 'unique_to_file1', 'unique_to_file2'}`` lists.
    :raises ValueError: if ``save`` is True but ``save_path`` is missing.
    """
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Filter based on coefficient
    if filter_coeff is not None:
        df1 = df1[df1['coefficient'] > filter_coeff] if filter_coeff >= 0 else df1[df1['coefficient'] < filter_coeff]
        df2 = df2[df2['coefficient'] > filter_coeff] if filter_coeff >= 0 else df2[df2['coefficient'] < filter_coeff]

    # Extract gene columns and drop NaN values
    genes1 = set(df1[gene_column].dropna())
    genes2 = set(df2[gene_column].dropna())

    # Calculate overlapping and non-overlapping genes
    overlapping_genes = genes1.intersection(genes2)
    unique_to_file1 = genes1.difference(genes2)
    unique_to_file2 = genes2.difference(genes1)

    # Create a Venn diagram
    plt.figure(figsize=(8, 6))
    venn2([genes1, genes2], ('File 1 Genes', 'File 2 Genes'))
    plt.title("Venn Diagram of Overlapping Genes")

    # Save or show the figure
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")
        save_path = save_figure(plt.gcf(), save_path,
                                bbox_inches="tight")
        print(f"Venn diagram saved to {save_path}")
    else:
        plt.show()

    # Return the results
    return {
        "overlap": list(overlapping_genes),
        "unique_to_file1": list(unique_to_file1),
        "unique_to_file2": list(unique_to_file2)
    }

def volcano_plot(
    data: Union[str, pd.DataFrame],
    *,
    fold_change_col: str,
    p_value_col: str,
    name_col: Optional[str] = None,
    # transforms
    x_transform: str = "none",      # "none" | "log2" | "log10" | "ln"
    y_transform: str = "-log10",    # "none" | "-log10" | "-ln" | "log10" | "ln"
    # thresholds
    fold_change_threshold: Optional[float] = None,
    p_value_threshold: Optional[float] = None,
    # annotation
    annotate: bool = True,
    annotate_max: Optional[int] = None,
    # plotting
    point_size: float = 20.0,
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (8.0, 6.0),
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    threshold_line_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    # excel options
    sheet_name: Union[int, str] = 0,
) -> Tuple[plt.Figure, plt.Axes, list]:
    """Read a table (CSV/TSV/XLS/XLSX or a DataFrame) and render a volcano plot.

    Auto-detects file type from extension (.csv, .tsv/.tab, .xls/.xlsx)
    and applies the requested x/y transforms before drawing.

    :param data: Path to table file or a pandas ``DataFrame``.
    :param fold_change_col: Column of raw fold change (or logFC when
        ``x_transform='none'``).
    :param p_value_col: Column of p-values.
    :param name_col: Optional column supplying point labels.
    :param x_transform: One of ``'none'``, ``'log2'``, ``'log10'``,
        ``'ln'``. Use ``'none'`` when the column already stores logFC
        (may be negative).
    :param y_transform: One of ``'none'``, ``'-log10'``, ``'-ln'``,
        ``'log10'``, ``'ln'``. Default ``'-log10'``.
    :param fold_change_threshold: Threshold on x — in plotted units when
        ``x_transform='none'``, otherwise in raw FC units.
    :param p_value_threshold: Threshold on raw p; drawn as a dashed
        horizontal line in plotted units.
    :param annotate: Annotate significant points when a name column is
        supplied.
    :param annotate_max: Cap on the number of annotated points (highest
        y first).
    :param point_size: Scatter marker size.
    :param alpha: Scatter marker alpha.
    :param figsize: Figure size in inches.
    :param title: Optional figure title.
    :param xlim: Optional x-axis limits.
    :param ylim: Optional y-axis limits.
    :param threshold_line_kwargs: Extra kwargs for threshold lines.
    :param scatter_kwargs: Extra kwargs for the scatter call.
    :param text_kwargs: Extra kwargs for label texts.
    :param save_path: If given, save the figure to this path.
    :param show: Call ``plt.show()`` at the end. Default ``True``.
    :param ax: Existing axes to draw on; a new figure is created if None.
    :param sheet_name: Excel sheet index/name for .xls/.xlsx inputs.
    :returns: ``(fig, ax, hits)`` where ``hits`` are the labels drawn.
    :raises ValueError: on unknown transforms, or numeric columns that
        cannot be coerced.
    """

    # -------------------- I/O helpers --------------------
    def _read_table_auto(path: str) -> pd.DataFrame:
        lower = path.lower()

        # Excel
        if lower.endswith((".xls", ".xlsx")):
            try:
                return pd.read_excel(path, sheet_name=sheet_name)
            except ImportError as e:
                raise ImportError(
                    "Reading Excel requires an engine.\n"
                    "For .xlsx: pip install openpyxl\n"
                    "For .xls:  pip install xlrd\n"
                ) from e

        # TSV-like
        if lower.endswith((".tsv", ".tab")):
            return pd.read_csv(path, sep="\t")

        # CSV
        if lower.endswith(".csv"):
            return pd.read_csv(path)

        # Fallback: sniff delimiter (comma vs tab) and try CSV reader
        # (If it's actually Excel with a missing extension, user should pass a DataFrame or fix extension)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        comma = head.count(",")
        tab = head.count("\t")
        sep = "\t" if tab > comma else ","
        return pd.read_csv(path, sep=sep)

    # -------------------- transform helpers --------------------
    def _as_numeric(s: pd.Series, colname: str) -> np.ndarray:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(arr)):
            raise ValueError(f"Column '{colname}' could not be converted to numeric.")
        return arr

    def _transform_x(x: np.ndarray, mode: str) -> np.ndarray:
        mode = mode.lower()
        if mode == "none":
            return x
        if np.any(x <= 0):
            raise ValueError(
                f"x_transform='{mode}' requires all fold changes > 0. "
                f"If your column is already logFC (can be negative), use x_transform='none'."
            )
        if mode == "log2":
            return np.log2(x)
        if mode == "log10":
            return np.log10(x)
        if mode in ("ln", "log"):
            return np.log(x)
        raise ValueError(f"Unknown x_transform: {mode}")

    def _transform_y(p: np.ndarray, mode: str) -> np.ndarray:
        mode = mode.lower()
        if mode == "none":
            return p
        tiny = np.finfo(float).tiny
        p2 = np.clip(p, tiny, 1.0)
        if mode == "-log10":
            return -np.log10(p2)
        if mode == "-ln":
            return -np.log(p2)
        if mode == "log10":
            return np.log10(p2)
        if mode in ("ln", "log"):
            return np.log(p2)
        raise ValueError(f"Unknown y_transform: {mode}")

    def _threshold_x_in_plot_units(thresh: float) -> float:
        t = float(thresh)
        if x_transform.lower() == "none":
            return abs(t)
        if t <= 0:
            raise ValueError("fold_change_threshold must be > 0 when using a log x_transform.")
        if x_transform.lower() == "log2":
            return abs(np.log2(t))
        if x_transform.lower() == "log10":
            return abs(np.log10(t))
        if x_transform.lower() in ("ln", "log"):
            return abs(np.log(t))
        raise ValueError(f"Unknown x_transform: {x_transform}")

    def _threshold_y_in_plot_units(pthresh: float) -> float:
        pt = float(pthresh)
        if pt <= 0:
            raise ValueError("p_value_threshold must be > 0.")
        return float(_transform_y(np.array([pt], dtype=float), y_transform)[0])

    # -------------------- load --------------------
    df = data.copy() if isinstance(data, pd.DataFrame) else _read_table_auto(str(data))

    if fold_change_col not in df.columns:
        raise KeyError(f"fold_change_col '{fold_change_col}' not found in columns.")
    if p_value_col not in df.columns:
        raise KeyError(f"p_value_col '{p_value_col}' not found in columns.")
    if name_col is not None and name_col not in df.columns:
        raise KeyError(f"name_col '{name_col}' not found in columns.")

    x_raw = _as_numeric(df[fold_change_col], fold_change_col)
    p_raw = _as_numeric(df[p_value_col], p_value_col)

    keep = ~np.isnan(x_raw) & ~np.isnan(p_raw)
    df = df.loc[keep].copy()
    x_raw = x_raw[keep]
    p_raw = p_raw[keep]

    x = _transform_x(x_raw, x_transform)
    y = _transform_y(p_raw, y_transform)

    # -------------------- thresholds & hit mask --------------------
    mask = np.ones(len(df), dtype=bool)

    x_thr_plot = None
    if fold_change_threshold is not None:
        x_thr_plot = _threshold_x_in_plot_units(fold_change_threshold)
        mask &= (np.abs(x) >= x_thr_plot)

    y_thr_plot = None
    if p_value_threshold is not None:
        y_thr_plot = _threshold_y_in_plot_units(p_value_threshold)
        if y_transform.lower() == "none":
            mask &= (p_raw <= float(p_value_threshold))
        else:
            if y_transform.lower().startswith("-"):
                mask &= (y >= y_thr_plot)
            else:
                mask &= (y <= y_thr_plot)

    # -------------------- figure --------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter_defaults = dict(s=point_size, alpha=alpha, edgecolors="none")
    if scatter_kwargs:
        scatter_defaults.update(scatter_kwargs)

    # color hits if thresholds are provided; otherwise all gray
    if (fold_change_threshold is not None) or (p_value_threshold is not None):
        colors = np.where(mask & (x >= 0), "crimson", np.where(mask & (x < 0), "royalblue", "lightgray"))
    else:
        colors = "lightgray"

    ax.scatter(x, y, c=colors, **scatter_defaults)

    # labels
    xlab = fold_change_col if x_transform.lower() == "none" else f"{x_transform}({fold_change_col})"
    ylab = p_value_col if y_transform.lower() == "none" else f"{y_transform}({p_value_col})"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)

    # threshold lines
    line_defaults = dict(color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    if threshold_line_kwargs:
        line_defaults.update(threshold_line_kwargs)

    if x_thr_plot is not None:
        ax.axvline(-x_thr_plot, **line_defaults)
        ax.axvline(+x_thr_plot, **line_defaults)
    if y_thr_plot is not None:
        ax.axhline(y_thr_plot, **line_defaults)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # cosmetics
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)

    # -------------------- annotation --------------------
    hits: list = []
    if annotate and (name_col is not None):
        eligible = mask.copy()

        # If no thresholds were set, annotate nothing unless annotate_max is provided
        if (fold_change_threshold is None) and (p_value_threshold is None) and (annotate_max is None):
            eligible[:] = False

        if np.any(eligible):
            idx = np.where(eligible)[0]
            if annotate_max is not None and len(idx) > int(annotate_max):
                idx = idx[np.argsort(y[idx])[::-1][: int(annotate_max)]]

            try:
                from adjustText import adjust_text
            except ImportError as e:
                raise ImportError(
                    "Annotation requires the 'adjustText' package. Install with:\n"
                    "  pip install adjustText"
                ) from e

            tkw = dict(fontsize=8, ha="center", va="bottom")
            if text_kwargs:
                tkw.update(text_kwargs)

            texts = []
            for i in idx:
                label = str(df.iloc[i][name_col])
                hits.append(label)
                texts.append(ax.text(x[i], y[i], label, **tkw))

            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.7),
            )

    if save_path:
        save_path = save_figure(fig, save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, hits
