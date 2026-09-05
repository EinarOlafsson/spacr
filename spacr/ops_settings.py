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
    # -- where the data is -------------------------------------------------
    "dst_root": "Where the organised wells, mosaics and reports are written. "
                "Empty writes beside the source, which mixes outputs with "
                "inputs and makes a second run ambiguous.",
    "genotype_source": "The low-magnification acquisition that carries the "
                       "barcodes. This is the one that gets stitched.",
    "phenotype_source": "The high-magnification acquisition that carries the "
                        "morphology. These images are placed onto the "
                        "stitched genotype mosaic, not stitched themselves.",
    "exts": "Which file extensions count as images. Anything else in the "
            "folder is ignored rather than failing the run.",
    "recursive": "Search sub-folders as well. Off when a plate's wells are "
                 "already separated and you want only this level.",
    "do_organize": "Move each tile into a per-well folder before stitching. "
                   "OFF LEAVES YOUR FILES WHERE THEY ARE; on, they are moved, "
                   "so run with Dry run first if the layout matters to you.",
    "collision": "What to do when a destination filename already exists: "
                 "rename the incoming file, skip it, or overwrite it.",
    "on_missing": "What to do when a file named in the plan is not there: "
                  "stop, or carry on without it.",
    "meta_regex": "How the well, site, channel and magnification are read "
                  "OUT OF THE FILENAME. Every tile that does not match is "
                  "invisible to the run, so a wrong pattern looks like "
                  "missing data rather than an error.",
    "well_group": "Which named group in the pattern above holds the well.",
    "arr_axes": "The axis order inside each file. AUTO reads it from the "
                "file's own metadata and falls back to guessing from the "
                "shape, which is right for a plainly stacked array.",
    "squeeze_singleton": "Drop axes of length one. Off keeps a (1, Y, X) "
                         "file three-dimensional.",
    "t_index": "Which timepoint to take from a time series.",
    "z_index": "Which z-plane to take, when not projecting.",
    "mip": "Take the maximum across z instead of one plane. Usually right "
           "for spots, which sit at different depths across a field.",
    "channel_index": "Which channel the stitcher matches on. Pick the one "
                     "with the most structure -- usually the nuclear stain; "
                     "a sparse channel gives it nothing to align.",
    "channel_indices": "Which channels go into a multi-channel mosaic, in "
                       "order. Empty uses every channel the tiles share.",
    # -- finding the overlap ----------------------------------------------
    "detector": "The feature detector used to find the same landmark in two "
                "overlapping tiles.",
    "nfeatures": "How many features to look for per tile. More finds overlap "
                 "in sparser fields and costs time and memory; below about "
                 "2000 a sparse field stops matching at all.",
    "downsample": "Scale tiles down before matching, for speed. THE MOST "
                  "COMMON CAUSE OF A RUN THAT FINDS NOTHING: at 0.5 a 256 px "
                  "tile becomes 128 px and the detector has almost no corners "
                  "left. Use 1.0 if pairs are being skipped.",
    "max_site_gap": "How far apart two site numbers may be and still be "
                    "treated as neighbours. Large enough to cover the turn at "
                    "the end of a snake pattern.",
    "relative_scale": "How much bigger the phenotype magnification is than "
                      "the genotype one -- 2.0 for 20x onto 10x. Wrong here "
                      "and the alignment cannot converge.",
    # -- stitching advanced ------------------------------------------------
    "max_keypoints": "Cap on features actually kept per tile after detection, "
                     "which is what bounds memory on a dense field.",
    "ransac_thresh_px": "How far a matched feature may sit from where the "
                        "fitted transform predicts, in pixels, and still "
                        "count as agreeing with it.",
    "allow_scale": "Let the fit change size between tiles. Off for a single "
                   "acquisition, where the magnification cannot differ.",
    "allow_rotation": "Let the fit rotate. Off for a motorised stage, which "
                      "does not rotate between fields.",
    "pair_batch_size": "How many candidate pairs are scored per batch. Only "
                       "affects memory and progress reporting.",
    "all_scores": "Keep every scored pair in the report, not only the ones "
                  "that passed. Useful when a stitch fails and you want to "
                  "see how close it came.",
    # -- mosaic -------------------------------------------------------------
    "stitch": "Run the pairwise stitch. Off scores nothing and only "
              "organises the plate.",
    "mosaic": "Assemble the stitched tiles into one image. Same meaning as "
              "Write mosaic; either switches it on.",
    "write_mosaic": "Write the assembled mosaic to disk. Off still produces "
                    "the pairwise report and the manifest, which is enough to "
                    "assemble it later.",
    "do_multichannel": "Write one mosaic holding every channel rather than "
                       "one channel alone.",
    "blend": "How overlapping tiles are combined where they meet: take the "
             "brighter pixel, or let the later tile overwrite.",
    "mosaic_min_score": "The lowest pair score allowed to place a tile in the "
                        "mosaic. Empty uses the automatic knee of the score "
                        "distribution, which adapts to the run.",
    "save_stitched_default": "Also write each stitched PAIR, not only the "
                             "whole-well mosaic. A lot of files; useful when "
                             "diagnosing one bad seam.",
    "mosaic_out": "Explicit path for the mosaic image. Empty puts it beside "
                  "the well's other outputs.",
    "mosaic_csv_out": "Explicit path for the manifest listing each tile's "
                      "position in the mosaic.",
    "out_tif": "Explicit path for a single-channel output image.",
    "out_png": "Explicit path for a preview PNG.",
    "preview_downsample": "How much to shrink the preview PNG. The mosaic "
                          "itself is unaffected.",
    # -- alignment ----------------------------------------------------------
    "do_nuc_stitch": "Segment nuclei and align on those instead of on raw "
                     "pixels. More robust when the two acquisitions use "
                     "different stains, because cells correspond even when "
                     "pixels do not.",
    "cellpose_model": "Which Cellpose model segments the nuclei used for "
                      "alignment.",
    "cellpose_diameter": "Expected nucleus diameter in pixels. Empty lets "
                         "Cellpose estimate it, which is usually right and "
                         "occasionally very wrong on a sparse field.",
    "outline_source": "How the foreground is found when drawing quality-"
                      "control outlines.",
    "canny": "Low and high thresholds for edge detection, when outlines come "
             "from edges.",
    "blur_sigma": "How much to smooth before finding edges. Higher ignores "
                  "texture; 0 does not smooth.",
    "dilate_ksize": "How much to thicken the detected outline. 0 leaves it "
                    "one pixel wide.",
    # -- quality control ----------------------------------------------------
    "save_qc": "Write overlay images showing where each tile was placed. The "
               "cheapest way to see that a stitch is right.",
    "outline_alpha": "How opaque the quality-control outlines are drawn.",
    "line_thickness": "How thick those outlines are drawn.",
    "n_workers": "How many parallel workers to use. More is faster until the "
                 "disk becomes the limit.",
    "n_workers_features": "Workers for feature detection specifically. Empty "
                          "follows Workers.",
    "opencv_threads": "Threads OpenCV may use INSIDE each worker. Leave at 1 "
                      "when running many workers: the two multiply, and "
                      "oversubscribing a machine makes it slower, not faster.",
    "max_ram_features": "How many tiles' features to hold in memory before "
                        "spilling to the cache. Lower on a small machine.",
    "feature_cache_mode": "Whether computed features are cached on disk, kept "
                          "in memory, or not cached. Disk pays once and makes "
                          "a re-run fast.",
    "feature_cache_dir": "Where that cache lives. Empty puts it beside the "
                         "outputs.",
    "stream_csv": "Write each result to the report as it is produced rather "
                  "than at the end, so a long run can be watched and an "
                  "interrupted one keeps what it had.",
    "tmp_dir": "Scratch space for intermediate files. Empty uses the system "
               "temporary folder, which may be too small for a large plate.",
}

#: The blurb the module shows above its settings.
OPS_DESCRIPTION = (
    "Stitch a low-magnification genotype acquisition into per-well mosaics "
    "and place the high-magnification phenotype images onto them. This is the "
    "preprocessing half of optical pooled screening; barcode decoding is a "
    "separate step and does not read FASTQ."
)


def register() -> None:
    """Register the OPS defaults, types, categories and help.

    Idempotent: registering twice is what happens when a module is imported
    from both the GUI and a headless run, and that must not be an error.

    :returns: nothing; it mutates the shared settings tables.
    """
    from . import settings as _settings
    from .spacrops import get_preprocess_ops_settings

    _settings.register_defaults(
        "ops",
        get_preprocess_ops_settings,
        replace=True,
        expected_types=OPS_TYPES,
        tooltips=OPS_TOOLTIPS,
        categories=OPS_CATEGORIES,
        description=OPS_DESCRIPTION,
    )
