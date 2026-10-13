#!/usr/bin/env python3
"""Create 4K real-experiment keyframes for the Mask tutorial."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "live_mask_dataset"
OUTPUT = ROOT / "production" / "07_mask" / "keyframes"
FONT = "/home/carruthers/.local/share/fonts/OpenSans-Regular.ttf"
SIZE = (3840, 2160)
LABELS = ["0  Nuclei", "1  Endoplasmic reticulum", "2  Lipid droplets", "3  Parasites"]
COLORS = [(69, 144, 255), (63, 220, 164), (255, 185, 64), (255, 67, 153)]


def font(size: int):
    return ImageFont.truetype(FONT, size)


def normalize(array: np.ndarray) -> np.ndarray:
    low, high = np.percentile(array, (0.5, 99.75))
    return np.clip((array.astype(np.float32) - low) / max(1, high - low), 0, 1)


def field_channels() -> list[np.ndarray]:
    stem = "W0031F0001T0001Z000C"
    return [tifffile.imread(SOURCE / f"{stem}{index}.tif") for index in range(1, 5)]


def masks(channels: list[np.ndarray]) -> list[np.ndarray]:
    values = [normalize(channel) for channel in channels]
    nuclei_binary = morphology.remove_small_objects(
        filters.gaussian(values[0], 1.2) > filters.threshold_otsu(values[0]), 80)
    nuclei_binary = morphology.remove_small_holes(nuclei_binary, 90)
    nuclei = measure.label(nuclei_binary)

    er = filters.gaussian(values[1], 1.4)
    cell_area = er > max(filters.threshold_otsu(er) * 0.72, np.percentile(er, 52))
    cell_area = morphology.binary_closing(cell_area, morphology.disk(3))
    cell_area = morphology.remove_small_holes(cell_area, 800)
    cell_area = morphology.remove_small_objects(cell_area, 800)
    cells = segmentation.watershed(-er, nuclei, mask=cell_area)

    lipids_enhanced = morphology.white_tophat(values[2], morphology.disk(7))
    lipid_threshold = max(filters.threshold_otsu(lipids_enhanced),
                          np.percentile(lipids_enhanced, 97.5))
    lipids = measure.label(morphology.remove_small_objects(
        lipids_enhanced > lipid_threshold, 5))

    parasite = filters.gaussian(values[3], 0.9)
    parasite_binary = parasite > max(filters.threshold_otsu(parasite),
                                     np.percentile(parasite, 88))
    parasites = measure.label(morphology.remove_small_objects(parasite_binary, 18))
    return [nuclei, cells, lipids, parasites]


def base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, "#070b12")
    draw = ImageDraw.Draw(image)
    draw.text((150, 105), title, font=font(60), fill="#f5f7fb")
    draw.text((153, 188), subtitle, font=font(31), fill="#9cabbf")
    draw.line((150, 255, 3690, 255), fill="#253249", width=2)
    draw.text((150, 2070), "spaCR  •  Mask  •  supplied 13-field experiment",
              font=font(24), fill="#74849a")
    return image, draw


def panel_geometry(index: int) -> tuple[int, int, int, int]:
    column, row = index % 2, index // 2
    x = 150 + column * 1810
    y = 315 + row * 830
    return x, y, 1710, 710


def fitted(array: np.ndarray, width: int, height: int) -> Image.Image:
    gray = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), "L").convert("RGB")
    gray.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#020408")
    canvas.paste(gray, ((width - gray.width) // 2, (height - gray.height) // 2))
    return canvas


def channel_frame(channels: list[np.ndarray]) -> None:
    image, draw = base(
        "Map the four acquisition channels",
        "Zero-based spaCR indices are shown beside their biological roles",
    )
    for index, channel in enumerate(channels):
        x, y, width, height = panel_geometry(index)
        draw.rounded_rectangle((x, y, x + width, y + height), 18,
                               fill="#0b111b", outline="#28364b", width=2)
        draw.text((x + 28, y + 25), LABELS[index], font=font(31),
                  fill=COLORS[index])
        rendered = fitted(normalize(channel), width - 56, height - 100)
        image.paste(rendered, (x + 28, y + 82))
    image.save(OUTPUT / "02_channels.png", compress_level=2)


def colored_overlay(channel: np.ndarray, labels: np.ndarray,
                    color: tuple[int, int, int]) -> Image.Image:
    value = normalize(channel)
    rgb = np.repeat(np.uint8(value[..., None] * 205), 3, axis=2)
    boundary = segmentation.find_boundaries(labels, mode="outer")
    rgb[boundary] = color
    return Image.fromarray(rgb, "RGB")


def preview_frame(channels: list[np.ndarray], labels: list[np.ndarray]) -> None:
    image, draw = base(
        "Preview every segmentation pass",
        "Nuclei from 0  •  cells from ER 1  •  organelle spots from lipids 2  •  parasites from 3",
    )
    for index, (channel, label_image) in enumerate(zip(channels, labels)):
        x, y, width, height = panel_geometry(index)
        draw.rounded_rectangle((x, y, x + width, y + height), 18,
                               fill="#0b111b", outline="#28364b", width=2)
        count = int(label_image.max())
        draw.text((x + 28, y + 25), LABELS[index], font=font(31),
                  fill=COLORS[index])
        draw.text((x + width - 28, y + 31), f"{count:,} objects",
                  font=font(24), fill="#aab6c7", anchor="ra")
        rendered = colored_overlay(channel, label_image, COLORS[index])
        rendered.thumbnail((width - 56, height - 100), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", (width - 56, height - 100), "#020408")
        frame.paste(rendered, ((frame.width - rendered.width) // 2,
                               (frame.height - rendered.height) // 2))
        image.paste(frame, (x + 28, y + 82))
    image.save(OUTPUT / "03_preview.png", compress_level=2)


def output_frame(channels: list[np.ndarray], labels: list[np.ndarray]) -> None:
    image, draw = base(
        "Aligned masks are ready for measurement",
        "The ER channel remains channel of interest 1 for downstream feature extraction",
    )
    value = np.stack([normalize(channels[3]), normalize(channels[1]),
                      normalize(channels[0])], axis=-1)
    value = np.uint8(np.clip(value ** 0.75, 0, 1) * 210)
    colours = np.zeros_like(value)
    for label_image, colour in zip(labels, COLORS):
        colours[segmentation.find_boundaries(label_image, mode="outer")] = colour
    composite = np.maximum(value, colours)
    rendered = Image.fromarray(composite, "RGB")
    rendered.thumbnail((2500, 1660), Image.Resampling.LANCZOS)
    x, y = 150, 320
    image.paste(rendered, (x, y))
    legend_x = 2820
    draw.rounded_rectangle((legend_x, 320, 3690, 1830), 20,
                           fill="#0d1420", outline="#2a3950", width=2)
    draw.text((legend_x + 55, 390), "Mask outputs", font=font(38), fill="#f5f7fb")
    for index, (label, colour, label_image) in enumerate(zip(LABELS, COLORS, labels)):
        yy = 520 + index * 205
        draw.ellipse((legend_x + 58, yy, legend_x + 84, yy + 26), fill=colour)
        draw.text((legend_x + 110, yy - 8), label, font=font(28), fill="#dce5f2")
        draw.text((legend_x + 110, yy + 40), f"{int(label_image.max()):,} labels",
                  font=font(23), fill="#8797ac")
    draw.line((legend_x + 55, 1390, 3635, 1390), fill="#29384e", width=2)
    draw.text((legend_x + 55, 1450), "Downstream", font=font(26), fill="#7faee8")
    draw.text((legend_x + 55, 1505), "Measure  →  Annotate  →  Classify",
              font=font(27), fill="#f5f7fb")
    draw.text((legend_x + 55, 1590), "Channel of interest", font=font(23), fill="#8797ac")
    draw.text((legend_x + 55, 1640), "1  Endoplasmic reticulum",
              font=font(28), fill=COLORS[1])
    image.save(OUTPUT / "04_output.png", compress_level=2)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    channels = field_channels()
    label_images = masks(channels)
    channel_frame(channels)
    preview_frame(channels, label_images)
    output_frame(channels, label_images)
    for filename in ("02_channels.png", "03_preview.png", "04_output.png"):
        print(OUTPUT / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
