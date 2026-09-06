"""Types, categories and help for every optical-pooled-screening setting.

WHY THIS FILE EXISTS. `spacr/spacrops.py` carries sixty-three settings and,
until this, not one of them had a declared type, a category or a tooltip.
A settings audit found what that costs: a module absent from the shared tables
is never checked, and that is how a checkbox came to ship the string
``'False'`` -- truthy, silently, for the life of the release.

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

#: The type each setting may hold, for :func:`spacr.settings.check_settings`.
#:
#: ``None`` is admissible for every path-like and every "work it out" default,
#: which is why those carry a tuple including ``type(None)``.
OPS_TYPES: Dict[str, object] = {
    # -- where the data is -------------------------------------------------
    # `src` IS NOT HERE ON PURPOSE. It is already declared as (str, list) by
    # another module, and `register_defaults` refuses a redeclaration --
    # rightly, because two modules disagreeing about what a shared key may
    # hold is a bug, not a preference. OPS uses the shared meaning.
    "dst_root": (str, type(None)),
    "genotype_source": (str, type(None)),
    "phenotype_source": (str, type(None)),
    "tmp_dir": (str, type(None)),
    "plate": str,
    "exts": (list, tuple),
    "recursive": bool,
    "on_missing": str,
    "collision": str,
    "dry_run": bool,
    "do_organize": bool,
    # -- how a filename is read -------------------------------------------
    "meta_regex": str,
    "well_group": str,
    "arr_axes": str,
    "squeeze_singleton": bool,
    "t_index": int,
    "z_index": int,
    "mip": bool,
    "channel_index": int,
    "channel_indices": (list, tuple, type(None)),
    # -- finding the overlap ----------------------------------------------
    "detector": str,
    "nfeatures": int,
    "max_keypoints": int,
    "downsample": float,
    "ransac_thresh_px": float,
    "allow_scale": bool,
    "allow_rotation": bool,
    "max_site_gap": int,
    "pair_batch_size": int,
    "score_threshold": float,
    "all_scores": bool,
    # -- building the mosaic ----------------------------------------------
    "stitch": bool,
    "mosaic": bool,
    "write_mosaic": bool,
    "mosaic_out": (str, type(None)),
    "mosaic_csv_out": (str, type(None)),
    "mosaic_min_score": (float, int, type(None)),
    "do_multichannel": bool,
    "blend": str,
    "out_tif": (str, type(None)),
    "out_png": (str, type(None)),
    "preview_downsample": int,
    "save_stitched_default": bool,
    # -- placing the phenotype images -------------------------------------
    "relative_scale": float,
    "do_nuc_stitch": bool,
    "cellpose_model": str,
    "cellpose_diameter": (float, int, type(None)),
    "outline_source": str,
    "canny": (tuple, list),
    "blur_sigma": float,
    "dilate_ksize": int,
    # -- what it draws for you to check -----------------------------------
    "save_qc": bool,
    "outline_alpha": float,
    "line_thickness": int,
    "verbose": bool,
    "n_workers": int,
    "n_workers_features": (int, type(None)),
    "opencv_threads": int,
    "max_ram_features": int,
    "feature_cache_mode": str,
    "feature_cache_dir": (str, type(None)),
    "stream_csv": bool,
}

#: Which panel section each setting appears under.
OPS_CATEGORIES: Dict[str, List[str]] = {
    "OPS input": [
        "src", "dst_root", "genotype_source", "phenotype_source", "plate",
        "exts", "recursive", "do_organize", "collision", "on_missing",
        "dry_run",
    ],
    "OPS naming": [
        "meta_regex", "well_group", "arr_axes", "squeeze_singleton",
        "t_index", "z_index", "mip", "channel_index", "channel_indices",
    ],
    "OPS stitching": [
        "detector", "nfeatures", "downsample", "max_site_gap",
        "score_threshold", "relative_scale",
    ],
    "OPS stitching advanced": [
        "max_keypoints", "ransac_thresh_px", "allow_scale", "allow_rotation",
        "pair_batch_size", "all_scores",
    ],
    "OPS mosaic": [
        "stitch", "mosaic", "write_mosaic", "do_multichannel", "blend",
        "mosaic_min_score", "save_stitched_default",
    ],
    "OPS mosaic advanced": [
        "mosaic_out", "mosaic_csv_out", "out_tif", "out_png",
        "preview_downsample",
    ],
    "OPS alignment": [
        "do_nuc_stitch", "cellpose_model", "cellpose_diameter",
        "outline_source", "canny", "blur_sigma", "dilate_ksize",
    ],
    "OPS quality control": [
        "save_qc", "outline_alpha", "line_thickness", "verbose",
    ],
    "OPS performance": [
        "n_workers", "n_workers_features", "opencv_threads",
        "max_ram_features", "feature_cache_mode", "feature_cache_dir",
        "stream_csv", "tmp_dir",
    ],
}

#: What each setting does to the RESULT.
#:
#: FIVE KEYS ARE DELIBERATELY ABSENT -- `src`, `plate`, `dry_run`, `verbose`
#: and `score_threshold` -- because other modules already document them. A
#: second, differently-worded tooltip for a shared setting is worse than none:
#: the reader gets a different explanation depending on which panel they are
#: looking at, for the same key.
OPS_TOOLTIPS: Dict[str, str] = {
    "all_scores":
        "(bool) - Keep every scored pair in the report, not only the ones "
        "that passed. Turning it on makes the report larger and lets you see "
        "how close a failed stitch came; it changes nothing about the mosaic. "
        "Default False.",
    "allow_rotation":
        "(bool) - Let the fit rotate one tile relative to another. Leave it "
        "off for a motorised stage, which does not rotate between fields; "
        "turning it on adds a degree of freedom that can absorb a bad match "
        "into a plausible-looking angle. Default False.",
    "allow_scale":
        "(bool) - Let the fit change size between tiles. Off for a single "
        "acquisition, where the magnification cannot differ; on, a weak match "
        "can be explained away as a scale change. Default False.",
    "arr_axes":
        "(str) - How to read the axis order inside each file. AUTO takes it "
        "from the file's own metadata and falls back to guessing from the "
        "shape. Set it explicitly when a stack is being misread as channels "
        "or z. Default 'AUTO'.",
    "blend":
        "(str) - How overlapping tiles are combined where they meet: 'max' "
        "takes the brighter pixel, 'overwrite' lets the later tile win. 'max' "
        "hides a seam, 'overwrite' shows you where one is. Default 'max'.",
    "blur_sigma":
        "(float) - Gaussian blur, in pixels, applied before edges are found "
        "for the quality-control outlines. Higher ignores texture and follows "
        "only the object's shape; 0 does not smooth at all. Affects the drawn "
        "outline, never the mosaic. Default 0.0.",
    "canny":
        "(tuple) - Low and high thresholds, in intensity units, for the edge "
        "detector that draws quality-control outlines. Lower values find more "
        "edge and more noise with it. Default (40, 120).",
    "cellpose_diameter":
        "(float or None) - Expected nucleus diameter in pixels for the "
        "segmentation used to align acquisitions. Empty lets Cellpose "
        "estimate it, which is usually right and occasionally very wrong on a "
        "sparse field; setting it removes that variance. Default None.",
    "cellpose_model":
        "(str) - Which Cellpose model segments the nuclei that the phenotype- "
        "to-genotype alignment matches on. Changing it changes which objects "
        "are found, and so which points the alignment is solved from. Default "
        "'cpsam'.",
    "channel_index":
        "(int) - Which channel the stitcher matches on, zero-indexed. Pick "
        "the one with the most structure, usually the nuclear stain: a sparse "
        "channel gives the detector nothing to align and every pair is "
        "skipped. Default 0.",
    "channel_indices":
        "(list or None) - Which channels go into a multi-channel mosaic, in "
        "the order they are written. Empty uses every channel the tiles "
        "share. Default None.",
    "collision":
        "(str) - What to do when a destination filename already exists: "
        "'rename' the incoming file, 'skip' it, or 'overwrite' it. Overwrite "
        "destroys the earlier file. Default 'rename'.",
    "detector":
        "(str) - Which feature detector finds the same landmark in two "
        "overlapping tiles. ORB is rotation-invariant and free; changing it "
        "changes which pairs match and how long scoring takes. Default 'ORB'.",
    "dilate_ksize":
        "(int) - How many pixels to thicken the drawn quality-control "
        "outline. 0 leaves it one pixel wide, which is hard to see on a large "
        "mosaic. Affects the overlay only. Default 0.",
    "do_multichannel":
        "(bool) - Write one mosaic holding every channel instead of a single "
        "channel. Off gives a smaller file that carries only the channel "
        "named by channel_index. Default True.",
    "do_nuc_stitch":
        "(bool) - Segment nuclei and align on those instead of on raw pixels. "
        "More robust when the two acquisitions use different stains, because "
        "cells correspond even when pixel intensities do not; costs a "
        "segmentation pass. Default True.",
    "do_organize":
        "(bool) - Move each tile into a per-well folder before stitching. ON "
        "MOVES YOUR FILES: run with Dry run first if the current layout "
        "matters to you. Off leaves them where they are and stitches in "
        "place. Default True.",
    "downsample":
        "(float) - Scale tiles down before matching, for speed. THE MOST "
        "COMMON CAUSE OF A RUN THAT FINDS NOTHING: at 0.5 a 256 px tile "
        "becomes 128 px and the detector has almost no corners left, so every "
        "pair is skipped. Raise it to 1.0 if pairs are being skipped. Default "
        "0.5.",
    "dst_root":
        "(str or None) - Where the organised wells, mosaics and reports are "
        "written. Empty writes beside the source, which mixes outputs with "
        "inputs and makes a second run ambiguous about what it is reading. "
        "Default None.",
    "exts":
        "(list) - Which file extensions count as images. Anything else in the "
        "folder is ignored rather than failing the run. Default ['.tif', "
        "'.tiff'].",
    "feature_cache_dir":
        "(str or None) - Where the computed feature cache is written. Empty "
        "puts it beside the outputs. Point it at a fast local disk when the "
        "outputs are on a network share. Default None.",
    "feature_cache_mode":
        "(str) - Whether features are cached on 'disk', held in memory, or "
        "not cached. Disk pays the cost once and makes a re-run fast; memory "
        "is faster and bounded by max_ram_features. Default 'disk'.",
    "genotype_source":
        "(str or None) - The folder holding the low-magnification acquisition "
        "that carries the barcodes. This is the one that gets stitched into "
        "per-well mosaics; the phenotype images are placed onto its output. "
        "Default None.",
    "line_thickness":
        "(int) - How many pixels wide the quality-control outlines are drawn. "
        "Larger is easier to see on a downsampled preview and obscures more "
        "of the image under it. Default 1.",
    "max_keypoints":
        "(int) - Cap on features kept per tile after detection, which is what "
        "bounds memory on a dense field. Lowering it speeds scoring and can "
        "drop the match that would have joined two tiles. Default 4000.",
    "max_ram_features":
        "(int) - How many tiles' features to hold in memory before spilling "
        "to the cache, in images. Lower it on a small machine; raising it "
        "trades memory for fewer disk reads. Default 256.",
    "max_site_gap":
        "(int) - How far apart two site numbers may be and still be treated "
        "as neighbours. Large enough to cover the turn at the end of a snake "
        "pattern; too small and the tiles at a row end never get compared. "
        "Default 64.",
    "meta_regex":
        "(str) - How the well, site, channel and magnification are read OUT "
        "OF THE FILENAME. Every tile that does not match is invisible to the "
        "run, so a wrong pattern looks like missing data rather than an "
        "error. Default matches '10X_c1_A1_Site-1.tif'.",
    "mip":
        "(bool) - Take the maximum across z instead of a single plane. "
        "Usually right for spots, which sit at different depths across a "
        "field; off reads the plane named by z_index. Default True.",
    "mosaic":
        "(bool) - Assemble the stitched tiles into one image. The same switch "
        "as write_mosaic; either turns it on. Default False.",
    "mosaic_csv_out":
        "(str or None) - Explicit path for the manifest listing each tile's "
        "position and transform in the mosaic. Empty writes it beside the "
        "well's other outputs. The manifest is enough to rebuild the mosaic "
        "later without re-scoring. Default None.",
    "mosaic_min_score":
        "(float or None) - The lowest pair score allowed to place a tile in "
        "the mosaic. Empty uses the automatic knee of the score distribution, "
        "which adapts to the run; a fixed value is reproducible but can drop "
        "a whole well on a dim plate. Default None.",
    "mosaic_out":
        "(str or None) - Explicit path for the assembled mosaic image. Empty "
        "writes it beside the well's other outputs under its own name. "
        "Default None.",
    "n_workers":
        "(int) - How many parallel workers to use. More is faster until the "
        "disk becomes the limit; each worker holds its own tiles, so this "
        "multiplies memory. Default is the machine's core count.",
    "n_workers_features":
        "(int or None) - Workers for feature detection specifically. Empty "
        "follows n_workers. Lower it when feature extraction is what is "
        "exhausting memory. Default None.",
    "nfeatures":
        "(int) - How many features to look for per tile. More finds overlap "
        "in sparser fields and costs time and memory; below about 2000 a "
        "sparse field stops matching at all. Default 8000.",
    "on_missing":
        "(str) - What to do when a file named in the plan is not there: "
        "'error' stops the run, 'skip' carries on without it and leaves a "
        "hole in the mosaic. Default 'error'.",
    "opencv_threads":
        "(int) - Threads OpenCV may use INSIDE each worker. Leave at 1 when "
        "running many workers: the two multiply, and oversubscribing a "
        "machine makes it slower rather than faster. Default 1.",
    "out_png":
        "(str or None) - Explicit path for the downsampled preview PNG. Empty "
        "writes it beside the mosaic. The preview is for looking at; the TIFF "
        "is for measuring. Default None.",
    "out_tif":
        "(str or None) - Explicit path for a single-channel output image. "
        "Empty writes it beside the well's other outputs. Default None.",
    "outline_alpha":
        "(float) - How opaque the quality-control outlines are drawn, 0 to 1. "
        "Lower lets more of the image show through the line. Affects the "
        "overlay only. Default 1.0.",
    "outline_source":
        "(str) - How the foreground is found when drawing quality-control "
        "outlines: 'otsu' thresholds the intensity, other values use the edge "
        "detector above. Changes what the overlay traces, never the mosaic. "
        "Default 'otsu'.",
    "pair_batch_size":
        "(int) - How many candidate pairs are scored per batch. Only affects "
        "peak memory and how often progress is reported; the result is "
        "identical. Default 8192.",
    "phenotype_source":
        "(str or None) - The folder holding the high-magnification "
        "acquisition that carries the morphology. These images are PLACED "
        "onto the stitched genotype mosaic, not stitched themselves. Default "
        "None.",
    "preview_downsample":
        "(int) - How much to shrink the preview PNG, as a divisor. The mosaic "
        "itself is unaffected. Larger is a smaller file that hides fine "
        "seams. Default 8.",
    "ransac_thresh_px":
        "(float) - How far a matched feature may sit, in pixels, from where "
        "the fitted transform predicts and still count as agreeing with it. "
        "Larger accepts looser fits and more of them; smaller rejects real "
        "matches on a distorted field. Default 3.0.",
    "recursive":
        "(bool) - Search sub-folders as well as the source folder. Turn it "
        "off when a plate's wells are already separated and you want only "
        "this level. Default True.",
    "relative_scale":
        "(float) - How much bigger the phenotype magnification is than the "
        "genotype one -- 2.0 for 20x onto 10x. Wrong here and the alignment "
        "cannot converge, because it is solving for a scale it has been told "
        "is different. Default 2.0.",
    "save_qc":
        "(bool) - Write overlay images showing where each tile was placed. "
        "The cheapest way to see that a stitch is right; costs one image per "
        "well. Default False.",
    "save_stitched_default":
        "(bool) - Also write each stitched PAIR, not only the whole-well "
        "mosaic. A great many files; useful when diagnosing one bad seam and "
        "wasteful otherwise. Default False.",
    "squeeze_singleton":
        "(bool) - Drop axes of length one when reading a file. Off keeps a "
        "(1, Y, X) file three-dimensional, which matters when downstream code "
        "counts dimensions. Default True.",
    "stitch":
        "(bool) - Run the pairwise stitch. Off scores no pairs and only "
        "organises the plate into per-well folders, which is what you want "
        "when the images are already stitched. Default False.",
    "stream_csv":
        "(bool) - Write each result to the report as it is produced rather "
        "than at the end, so a long run can be watched and an interrupted one "
        "keeps what it had. Off is marginally faster and loses everything on "
        "a crash. Default True.",
    "t_index":
        "(int) - Which timepoint to take from a time series, zero-indexed. "
        "Only read when the file has a time axis. Default 0.",
    "tmp_dir":
        "(str or None) - Scratch space for intermediate files and the "
        "mosaic's memory map. Empty uses the system temporary folder, which "
        "may be too small for a large plate. Default None.",
    "well_group":
        "(str) - Which named group in meta_regex holds the well identifier. "
        "Change it when your filenames name the well under a different group; "
        "getting it wrong groups every tile into one well. Default 'well'.",
    "write_mosaic":
        "(bool) - Write the assembled mosaic to disk. Off still produces the "
        "pairwise report and the manifest, which is enough to assemble it "
        "later without re-scoring. Default False.",
    "z_index":
        "(int) - Which z-plane to take when not projecting, zero-indexed. "
        "Ignored when mip is on. Default 0.",
}

#: The blurb the module shows above its settings.
OPS_DESCRIPTION = (
    "Stitch a low-magnification genotype acquisition into per-well mosaics "
    "and place the high-magnification phenotype images onto them. This is the "
    "preprocessing half of optical pooled screening; barcode decoding is a "
    "separate step and does not read FASTQ."
)


def ops_defaults(settings=None):
    """The OPS settings, with `spacrops` imported only when they are wanted.

    A LAZY FACTORY, AND THAT IS THE POINT. `spacr.settings` imports this
    module so the registration happens at startup, and `spacrops` reaches
    OpenCV and SciPy -- a third of a second, on the path that a module screen
    opens through. Registering the FUNCTION rather than calling it keeps that
    cost until someone actually asks for an OPS default.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    from .spacrops import get_preprocess_ops_settings

    return get_preprocess_ops_settings(settings if settings is not None else {})


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
