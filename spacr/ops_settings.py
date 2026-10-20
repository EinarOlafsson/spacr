"""Types, categories and help for every optical-pooled-screening setting.

Each setting is registered with its type, tooltip and category when it is
added. A setting with no type cannot be validated, a setting with no tooltip
has no help text, and a setting in no category does not appear in the panel.

Tooltips describe the setting's effect on the result rather than repeating
its name. For example, "The number of features" repeats the label, whereas
"More features find more overlap and cost time and memory; below about 2000
a sparse field stops matching" states the effect of raising or lowering it.
"""
from __future__ import annotations

from typing import Dict, List

#: The default of every setting :func:`spacr.ops_engine.run_ops` reads, and
#: of nothing else. A key the engine does not read has no place in its panel:
#: it would be a control that changes nothing.
#:
#: THE FOUR MEASURED NUMBERS REPEAT A CONSTANT IN :mod:`spacr.ops_engine`,
#: which is where each is recorded with the measurement that produced it.
#: They are repeated rather than imported because importing the engine here
#: would pull numpy, scipy and the whole OPS stack into every settings
#: registration, including the GUI's; the two copies are held equal by
#: tests/test_every_ops_setting_is_declared.py.
OPS_DEFAULTS: Dict[str, object] = {
    "genotype_source": None,
    "phenotype_source": None,
    "dst_root": None,
    "plate": "",
    "ops_gpu": True,
    "n_workers": 26,
    "cellpose_model": "cpsam",
    "cellpose_diameter": None,
    "ops_library": None,
    "ops_base_channels": "CY3,A594,CY5,CY7",
    "ops_read_threshold": 315.0,
    "ops_raster_overlap": 213,
    "ops_window_overlap": 96,
    "ops_footprint": 10.0,
    "ops_store_reads": False,
    "ops_spot_detector": "native",
}

#: The type each setting may hold, for :func:`spacr.settings.check_settings`.
#:
#: ``None`` is admissible for every path-like and every "work it out" default,
#: which is why those carry a tuple including ``type(None)``.
OPS_TYPES: Dict[str, object] = {
    "dst_root": (str, type(None)),
    "genotype_source": (str, type(None)),
    "phenotype_source": (str, type(None)),
    "plate": str,
    "cellpose_model": str,
    "cellpose_diameter": (float, int, type(None)),
    "n_workers": int,
    "ops_gpu": bool,
    "ops_library": (str, type(None)),
    "ops_base_channels": str,
    "ops_read_threshold": (float, int),
    "ops_raster_overlap": int,
    "ops_window_overlap": int,
    "ops_footprint": (float, int),
    "ops_store_reads": bool,
    "ops_spot_detector": str,
}

#: Which panel section each setting appears under.
#:
#: A SETTING BELONGS TO EXACTLY ONE CATEGORY. `plate` is shared with the rest
#: of spaCR and is already filed under Plate Layout & Controls, so it is NOT
#: repeated here. Listing one twice is not a display preference: Tk renders
#: each copy separately and Qt drops all but the first, so the second copy is
#: either a duplicate control or an invisible one depending on which toolkit
#: is drawing.
OPS_CATEGORIES: Dict[str, List[str]] = {
    "OPS input": [
        "dst_root", "genotype_source", "phenotype_source", "ops_library",
    ],
    "OPS alignment": [
        "cellpose_model", "cellpose_diameter", "ops_raster_overlap",
        "ops_window_overlap",
    ],
    "OPS decoding": [
        "ops_base_channels", "ops_read_threshold", "ops_footprint",
        "ops_store_reads", "ops_spot_detector",
    ],
    "OPS performance": [
        "ops_gpu", "n_workers",
    ],
}

#: What each setting does to the RESULT.
#:
#: ONE KEY IS DELIBERATELY ABSENT -- `plate` -- because other modules already
#: document it. A second, differently-worded tooltip for a shared setting is
#: worse than none: the reader gets a different explanation depending on which
#: panel they are looking at, for the same key.
OPS_TOOLTIPS: Dict[str, str] = {
    "cellpose_diameter":
        "(float or None) - Expected nucleus diameter in pixels for the "
        "segmentation of each stitched well. When set, each window is "
        "rescaled so that this diameter becomes the 30 pixels the model "
        "expects; empty segments the windows at their own scale. "
        "Default None.",
    "cellpose_model":
        "(str) - Which Cellpose model segments the nuclei of each stitched "
        "well. 'cpsam' runs the default model of the installed Cellpose; any "
        "other name is loaded as that model. Changing it changes which "
        "objects are found, and so which nuclei the reads are attributed "
        "to. Default 'cpsam'.",
    "dst_root":
        "(str or None) - Where measurements.db and each well's "
        "ops_report.json are written. Empty writes them into the source "
        "folder, which mixes outputs with inputs. Default None.",
    "genotype_source":
        "(str or None) - The folder holding the low-magnification "
        "sequencing acquisition that carries the barcodes. Its subfolders "
        "are searched too, for tiles named like "
        "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-0.tif: magnification, cycle, "
        "well, channels and site. Every well found is stitched, segmented "
        "and decoded. Default None.",
    "n_workers":
        "(int) - How many parallel workers to use. More is faster until the "
        "disk becomes the limit; each worker holds its own tiles, so this "
        "multiplies memory. Default is the machine's core count.",
    "ops_base_channels":
        "(str) - Which channels carry the four bases, in the order G, T, A, "
        "C, separated by commas. These names are matched against the channel "
        "token in each tile's file name, so a run whose base channels are "
        "named differently is read by changing this rather than by renaming "
        "files. Exactly four are required: a shorter list decodes a shorter "
        "barcode than the library holds. Default 'CY3,A594,CY5,CY7'.",
    "ops_footprint":
        "(float) - How far beyond a nucleus's boundary a read may lie and "
        "still be counted as that nucleus's read, in pixels. On the "
        "reference plate 10 pixels held 98 % of the detected spots and 3 "
        "held 67 %, and the wider setting assigned 28 % more objects a "
        "barcode at slightly higher purity, because the reads sit around "
        "the rim rather than inside. Too wide starts giving a read to a "
        "neighbour. Default 10.",
    "ops_library":
        "(str or None) - A CSV of guide barcodes, with the column holding "
        "them named prefix, barcode or sequence. Given one, each nucleus's "
        "called barcode is also matched to its closest guide, and the run "
        "reports how many spots match the library exactly, which measures "
        "decoding success. Empty still calls barcodes but cannot score them "
        "against a library. Default None.",
    "ops_raster_overlap":
        "(int) - How far neighbouring sequencing tiles overlap on the "
        "microscope's raster, in pixels. This is the acquisition's own "
        "setting, not a tuning knob: it tells the stitch where to look for "
        "a neighbour, and a value well away from the truth loses edges and "
        "leaves fields unplaced. Default 213, measured on the reference "
        "plate's 1,480 pixel tiles.",
    "ops_read_threshold":
        "(float) - How much brighter than its surroundings a spot must be "
        "to be counted as a read. Lower finds more reads and more debris, "
        "which shows up as a falling library-match rate rather than as an "
        "error; higher loses real reads and leaves nuclei with too few to "
        "vote. Default 315, the reference run's own value for this plate.",
    "ops_store_reads":
        "(bool) - Write every read behind the barcodes into the ops_reads "
        "table: one row per read per cycle, with the base called, its "
        "margin and the four intensities it was called from. This is how a "
        "suspect barcode is traced back to its pixels. It is off by default "
        "because the table is large -- a full well of the reference plate is about "
        "thirty million rows -- so turn it on for a well or two rather than "
        "for a plate. Default False.",
    "ops_spot_detector":
        "(str) - Which detector finds the sequencing spots. 'native' is spaCR's own spot score, the one this pipeline was validated with. 'spotnet' is DeepCell's SpotNet, run in its own environment; its spots go through the same base calls and nucleus assignment, and ops_read_threshold no longer applies. SpotNet is licensed for NON-COMMERCIAL ACADEMIC USE ONLY, installs from the Model Zoo and needs a DeepCell token in ~/.spacr/deepcell_token. Default 'native'.",
    "ops_window_overlap":
        "(int) - How far the segmentation windows overlap each other, in "
        "pixels. A nucleus is only numbered once if at least one window saw "
        "all of it, so this has to exceed the largest nucleus; when it does "
        "not, the run reports the objects no window saw whole and says by "
        "how much. Raising it costs time, because more of the well is "
        "segmented twice. Default 96.",
    "phenotype_source":
        "(str or None) - The folder holding the high-magnification "
        "phenotype acquisition of the same wells. Given one, the run aligns "
        "a few of its fields to the stitched sequencing well, fits the "
        "acquisition raster to those, and records where every phenotype "
        "field lands and which sequencing tile covers it. Empty skips that "
        "step and decodes the sequencing acquisition alone. Default None.",
    "ops_gpu":
        "(bool) - Let this run use the graphics card where spaCR finds a "
        "usable one: the tile registration's FFTs and the Cellpose outlines "
        "both have a GPU path, and both fall back to the CPU on their own if "
        "the card refuses. Turn it off when the card is busy with another "
        "job -- a shared GPU is the common case, and an out-of-memory in the "
        "middle of a plate costs more than the time the GPU saves. "
        "Default True.",
}

#: The blurb the module shows above its settings.
OPS_DESCRIPTION = (
    "Take each well of an optical pooled screen's sequencing acquisition "
    "from tiles to tables: stitch its nuclear tiles, segment and number its "
    "nuclei across the whole well, then decode each field's reads and assign "
    "a barcode to every nucleus whose reads agree. Given a phenotype folder "
    "it also records where each phenotype field lands on the stitched well. "
    "The tables are written to measurements.db. Reads are decoded from the "
    "images, not from FASTQ."
)


def ops_defaults(settings=None):
    """The settings the OPS engine reads, with their defaults filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in OPS_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def register(replace: bool = False) -> bool:
    """Register the OPS defaults, types, categories and help.

    Idempotent by default: a module imported from both the GUI and a headless
    run registers twice, and that must not be an error.

    :param replace: overwrite an existing registration rather than declining.
    :returns: True when this call did the registering.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults("ops") and not replace:
        return False
    register_defaults(
        "ops",
        ops_defaults,
        replace=True,
        expected_types=OPS_TYPES,
        tooltips=OPS_TOOLTIPS,
        categories=OPS_CATEGORIES,
        description=OPS_DESCRIPTION,
    )
    return True


SPATIAL_MASK_DEFAULTS: Dict[str, object] = {
    "spatial_source": None,
    "spatial_platform": "auto",
    "spatial_dst": None,
    "spatial_sample": "",
    "spatial_level": 0,
    "spatial_window": "",
    "spatial_window_unit": "um",
    "spatial_tile_size": 2048,
    "spatial_tile_overlap": 160,
    "spatial_segmenter": "cellpose",
    "spatial_stain": "auto",
    "spatial_cellpose_model": "cpsam",
    "spatial_diameter": None,
    "spatial_cell_expansion_um": 5.0,
    "spatial_gpu": False,
}

SPATIAL_MAP_DEFAULTS: Dict[str, object] = {
    "spatial_source": None,
    "spatial_platform": "auto",
    "spatial_masks_dir": None,
    "spatial_bin_um": 2,
    "spatial_min_qv": 20.0,
    "spatial_objects": "cell,nucleus",
    "spatial_pathogen_genes": "",
    "spatial_min_pathogen_transcripts": 1,
    "spatial_gene": "",
    "spatial_anndata": True,
}

SPATIAL_TYPES: Dict[str, object] = {
    "spatial_source": (str, type(None)),
    "spatial_platform": str,
    "spatial_dst": (str, type(None)),
    "spatial_sample": str,
    "spatial_level": int,
    "spatial_window": str,
    "spatial_window_unit": str,
    "spatial_tile_size": int,
    "spatial_tile_overlap": int,
    "spatial_segmenter": str,
    "spatial_stain": str,
    "spatial_cellpose_model": str,
    "spatial_diameter": (float, int, type(None)),
    "spatial_cell_expansion_um": (float, int),
    "spatial_gpu": bool,
    "spatial_masks_dir": (str, type(None)),
    "spatial_bin_um": int,
    "spatial_min_qv": (float, int),
    "spatial_objects": str,
    "spatial_pathogen_genes": str,
    "spatial_min_pathogen_transcripts": int,
    "spatial_gene": str,
    "spatial_anndata": bool,
}

SPATIAL_MASK_CATEGORIES: Dict[str, List[str]] = {
    "Spatial input": [
        "spatial_source", "spatial_platform", "spatial_dst",
        "spatial_sample",
    ],
    "Spatial region & tiling": [
        "spatial_level", "spatial_window", "spatial_window_unit",
        "spatial_tile_size", "spatial_tile_overlap",
    ],
    "Spatial segmentation": [
        "spatial_segmenter", "spatial_stain", "spatial_cellpose_model",
        "spatial_diameter", "spatial_cell_expansion_um", "spatial_gpu",
    ],
}

SPATIAL_MAP_CATEGORIES: Dict[str, List[str]] = {
    "Spatial input": [
        "spatial_source", "spatial_platform", "spatial_masks_dir",
    ],
    "Spatial mapping": [
        "spatial_bin_um", "spatial_min_qv", "spatial_objects",
        "spatial_gene", "spatial_anndata",
    ],
    "Spatial infection": [
        "spatial_pathogen_genes", "spatial_min_pathogen_transcripts",
    ],
}

SPATIAL_TOOLTIPS: Dict[str, str] = {
    "spatial_source":
        "(str or None) - The platform's output folder: a Xenium output "
        "bundle (transcripts.parquet, experiment.xenium and the morphology "
        "image) or a Space Ranger outs folder of a Visium HD run "
        "(binned_outputs/ and the full-resolution H&E or IF image beside "
        "it). Mask generation segments its image; barcode mapping reads its "
        "transcripts or bins. Default None.",
    "spatial_platform":
        "(str) - Which 10x platform wrote spatial_source: auto, xenium, "
        "visium_hd (Visium HD bins) or visium (standard spot-based Visium). "
        "auto reads it from the files present; set it only when a folder "
        "holds more than one layout. Default auto.",
    "spatial_dst":
        "(str or None) - Where the fields are written: merged/, stack/, "
        "masks/ and measurements/measurements.db with the spatial_fields "
        "and spatial_objects tables, the same layout Measure reads after "
        "Mask Generation. Point Measure's src at <spatial_dst>/merged and "
        "Spatial Map Barcodes' spatial_masks_dir at spatial_dst. Empty "
        "writes a spacr_spatial folder inside spatial_source. Default None.",
    "spatial_sample":
        "(str) - The plate name of the fields, the first part of every "
        "file name and measurement key, so a second slide measured into "
        "the same database stays apart. Empty uses the source folder's "
        "name. Default empty.",
    "spatial_level":
        "(int) - Which pyramid level of the slide image to segment and "
        "write: 0 is full resolution, each level up halves it. Level 1 of "
        "a Xenium morphology image (0.425 um per pixel) still resolves "
        "nuclei and segments four times faster. An image without that "
        "level is read at its coarsest one and the scale is recorded. "
        "Default 0.",
    "spatial_window":
        "(str) - Segment only this part of the slide: x, y, width and "
        "height, in spatial_window_unit. Only the image tiles under it are "
        "read. Empty segments the whole slide. Default empty.",
    "spatial_window_unit":
        "(str) - The unit of spatial_window: um (micrometres, through the "
        "image's pixel size) or px (pixels of the chosen level). "
        "Default um.",
    "spatial_tile_size":
        "(int) - Edge of each segmentation window and written field, in "
        "pixels. Larger windows mean fewer fields and seams but more memory "
        "per Cellpose call and larger arrays for Measure. Default 2048.",
    "spatial_tile_overlap":
        "(int) - How far neighbouring windows overlap, in pixels. An object "
        "is kept from a window that saw all of it, so this has to exceed "
        "the largest cell including spatial_cell_expansion_um; objects no "
        "window saw whole are counted in the report and dropped. Raising "
        "it costs time. Default 160.",
    "spatial_segmenter":
        "(str) - How nuclei are found: cellpose runs the Cellpose model in "
        "spatial_cellpose_model; threshold is an Otsu threshold with a "
        "distance-transform watershed, seconds instead of minutes on a CPU "
        "but it merges touching nuclei more often. Default cellpose.",
    "spatial_stain":
        "(str) - Which image the segmenter sees: auto takes the nuclear "
        "(DAPI) channel of a Xenium image and the haematoxylin of an H&E "
        "image; dapi takes the first channel; hematoxylin unmixes "
        "haematoxylin from RGB; rgb gives Cellpose the colour image "
        "itself. Default auto.",
    "spatial_cellpose_model":
        "(str) - The Cellpose model that segments the nuclei: cpsam is the "
        "installed Cellpose's default model, any other name or path is "
        "loaded as that model. Default cpsam.",
    "spatial_diameter":
        "(float or None) - Expected nucleus diameter in pixels of the "
        "chosen level. Given, each window is rescaled so that this "
        "diameter becomes the size Cellpose expects; empty segments at the "
        "image's own scale. Default None.",
    "spatial_cell_expansion_um":
        "(float) - How far each nucleus is grown into a cell, in "
        "micrometres, without crossing a neighbour (Xenium's own default is "
        "5). The grown objects are the cell masks; the nuclei are written "
        "as the nucleus masks with the same labels. 0 writes the nuclei as "
        "the cells. Default 5.",
    "spatial_gpu":
        "(bool) - Run Cellpose on the graphics card when spaCR finds a "
        "usable one. Off keeps the run on the CPU, which is slower but "
        "leaves a shared card alone. Default False.",
    "spatial_masks_dir":
        "(str or None) - The folder Spatial Mask Generation wrote (its "
        "spatial_dst): the fields in merged/ and their places on the slide "
        "in measurements.db. Every transcript or bin is placed in those "
        "fields and the per-cell counts are written into the same "
        "measurements.db, keyed by prcfo, so they join Measure's tables. "
        "Default None.",
    "spatial_bin_um":
        "(int) - Which Visium HD bins are read: 2, 8 or 16 micrometres. "
        "2 um bins are smaller than a cell and give the sharpest "
        "assignment; 8 and 16 um bins are shared out over the cells they "
        "cover by area. Ignored for Xenium. Default 2.",
    "spatial_min_qv":
        "(float) - The lowest Xenium transcript quality (Phred-scaled QV) "
        "kept. 20 is the threshold of Xenium's own cell-feature matrix; "
        "lower keeps more transcripts and more misreads. Ignored for "
        "Visium HD. Default 20.",
    "spatial_objects":
        "(str) - Which mask planes the reads are counted into, separated "
        "by commas: cell, nucleus, or both. Each gets an "
        "<object>_expression table in measurements.db. Default "
        "cell,nucleus.",
    "spatial_pathogen_genes":
        "(str) - The pathogen's own probes or genes, separated by commas; a "
        "name ending in * matches every gene starting with the rest "
        "(CHIKV*). A cell holding at least spatial_min_pathogen_transcripts "
        "of them is infected; they are left out of the host comparisons. "
        "Empty skips the infection analysis. Default empty.",
    "spatial_min_pathogen_transcripts":
        "(int) - How many pathogen transcripts (Visium HD: UMI shares) make "
        "a cell infected. Raising it from 1 removes cells that carry a "
        "stray viral read from a neighbour. Default 1.",
    "spatial_gene":
        "(str) - A gene whose transcripts or bins colour the overlay "
        "figure drawn over the field with the most cells. Empty draws them "
        "uncoloured. Default empty.",
    "spatial_anndata":
        "(bool) - Also write the per-cell counts as an .h5ad file with the "
        "slide coordinates in obsm['spatial'], for scanpy or squidpy; needs "
        "the anndata package. Default True.",
}

SPATIAL_MASK_DESCRIPTION = (
    "Segment a whole Xenium morphology or Visium HD H&E slide: the slide is "
    "cut into overlapping windows, only the image tiles under each window "
    "are read, nuclei are found and grown into cells, objects seen whole "
    "by more than one window are sewn into one, and every window is "
    "written as a field in merged/ with its masks, the layout Measure "
    "reads after Mask Generation. The fields' places on the slide go to "
    "measurements.db, for Spatial Map Barcodes."
)

SPATIAL_MAP_DESCRIPTION = (
    "Read every Xenium transcript or Visium HD bin, put it in the pixel "
    "frame of the fields Spatial Mask Generation wrote (Xenium micrometres "
    "through the morphology pixel size, Visium HD through the "
    "full-resolution image) and count it into the cell or nucleus it falls "
    "in. Per-cell expression tables are written into measurements.db keyed "
    "like Measure's, with infection calls from pathogen genes, "
    "infected-versus-uninfected tests and an overlay."
)


def spatial_mask_defaults(settings=None):
    """The settings the Spatial Mask Generation engine reads, filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in SPATIAL_MASK_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def spatial_map_defaults(settings=None):
    """The settings the Spatial Map Barcodes engine reads, filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in SPATIAL_MAP_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def register_spatial(replace: bool = False) -> bool:
    """Register both spatial submodules' defaults, types, categories and help.

    :param replace: overwrite existing registrations rather than declining.
    :returns: True when this call did the registering.
    """
    from .settings import has_registered_defaults, register_defaults

    done = False
    for key, factory, keys, cats, text in (
            ("spatial_masks", spatial_mask_defaults, SPATIAL_MASK_DEFAULTS,
             SPATIAL_MASK_CATEGORIES, SPATIAL_MASK_DESCRIPTION),
            ("spatial_map", spatial_map_defaults, SPATIAL_MAP_DEFAULTS,
             SPATIAL_MAP_CATEGORIES, SPATIAL_MAP_DESCRIPTION)):
        if has_registered_defaults(key) and not replace:
            continue
        register_defaults(
            key, factory, replace=True,
            expected_types={k: SPATIAL_TYPES[k] for k in keys},
            tooltips={k: SPATIAL_TOOLTIPS[k] for k in keys},
            categories=cats, description=text)
        done = True
    return done


register()
register_spatial()
