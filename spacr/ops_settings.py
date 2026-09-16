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
OPS_DEFAULTS: Dict[str, object] = {
    "genotype_source": None,
    "dst_root": None,
    "plate": "",
    "ops_gpu": True,
    "n_workers": 26,
    "cellpose_model": "cpsam",
    "cellpose_diameter": None,
}

#: The type each setting may hold, for :func:`spacr.settings.check_settings`.
#:
#: ``None`` is admissible for every path-like and every "work it out" default,
#: which is why those carry a tuple including ``type(None)``.
OPS_TYPES: Dict[str, object] = {
    "dst_root": (str, type(None)),
    "genotype_source": (str, type(None)),
    "plate": str,
    "cellpose_model": str,
    "cellpose_diameter": (float, int, type(None)),
    "n_workers": int,
    "ops_gpu": bool,
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
        "dst_root", "genotype_source",
    ],
    "OPS alignment": [
        "cellpose_model", "cellpose_diameter",
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
    "a barcode to every nucleus whose reads agree. The tables are written to "
    "measurements.db. Reads are decoded from the images, not from FASTQ, and "
    "the phenotype images are not placed by this step."
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
