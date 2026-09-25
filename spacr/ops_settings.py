"""Types, categories and help for every optical-pooled-screening setting.

So the registration happens ON THE WAY IN rather than being retrofitted. A
setting with no type cannot be validated; a setting with no tooltip is a
labelled box a user has to guess at; a setting in no category does not appear
in the panel at all.

TOOLTIPS SAY WHAT THE SETTING DOES TO THE RESULT, not what it is named. "The
number of features" tells a reader nothing they could not read off the label.
"More features find more overlap and cost time and memory; below about 2000
a sparse field stops matching" tells them which way to move it and why.
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
        "reports how many spots match the library exactly -- the one number "
        "that says whether the decode worked at all. Empty still calls "
        "barcodes and simply cannot score them. Default None.",
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
        "suspect barcode is traced back to its pixels. It is OFF by default "
        "because it is large -- a full well of the reference plate is about "
        "thirty million rows -- so turn it on for a well or two rather than "
        "for a plate. Default False.",
    "ops_spot_detector":
        "(str) - Which detector finds the sequencing spots each field's "
        "reads are called at. 'native' is spaCR's own spot score and the "
        "only one this plate was validated with. 'spotnet' is DeepCell's "
        "SpotNet, a trained spot detector run in an environment of its own; "
        "its positions go through the same bases, calls and assignment to "
        "nuclei, and ops_read_threshold no longer applies. SpotNet's models "
        "are licensed for NON-COMMERCIAL ACADEMIC USE ONLY, it has to be "
        "installed from the Model Zoo, and its weights need a free DeepCell "
        "access token in DEEPCELL_ACCESS_TOKEN or ~/.spacr/deepcell_token. "
        "It decodes one field at a time. Default 'native'.",
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
        "the card refuses. Turn it OFF when the card is busy with another "
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


register()
