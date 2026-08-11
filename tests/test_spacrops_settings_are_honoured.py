"""`cap_one_per_dir` was threaded through three functions and never read.

It is declared on `spacrStitcher._compute_mosaic_transforms` and on both
public mosaic APIs, documented on all three, and passed down through both --
and the per-direction cap ran unconditionally.

The cap is right by default: keep the best-scoring edge per (tile,
direction). Turning it OFF is the escape hatch for the case it exists for --
when the best-scoring edge in a direction is a FALSE match, a repeated
background pattern outscoring the true neighbour. Keeping only the winner
then hands the spanning tree the wrong edge and no alternative, which is
exactly what a user passing cap_one_per_dir=False is trying to avoid.

Also covered here: `qc_outlines` on `align_image_to_stitch`, which is a
DIFFERENT shape of defect and is deliberately NOT wired up. See below.
"""

import inspect

from spacr import spacrops


def test_the_direction_cap_is_conditional():
    source = inspect.getsource(spacrops.spacrStitcher._compute_mosaic_transforms)
    assert "if cap_one_per_dir:" in source, (
        "cap_one_per_dir is not consulted; the cap is unconditional again")


def test_there_is_a_real_uncapped_branch():
    """A conditional that picks between two identical paths is not a fix."""
    source = inspect.getsource(spacrops.spacrStitcher._compute_mosaic_transforms)
    assert "all_edges" in source
    # The capped branch keys on (src, dbin) and keeps a best; the uncapped one
    # must not, or False still collapses to one edge per direction.
    capped = source.index("if cap_one_per_dir:")
    uncapped = source.index("else:", capped)
    tail = source[uncapped:uncapped + 1200]
    assert "best_per_node_dir[key]" not in tail, (
        "the uncapped branch still keeps one best per direction")


def test_qc_outlines_says_it_is_inert_and_names_where_to_go():
    """This one is NOT wired up, on purpose, and that is the finding.

    `qc_outlines` reads like the switch for the `*__qc_outlines.png`
    overlays. Those are written by `spacrStitcher`, gated on ITS `save_qc`.
    `align_image_to_stitch` builds a `FOVAlignAndCropper`, which has no
    save_qc argument and draws no overlays -- so there is no destination to
    connect the parameter to, and an attempt to pass it through raises
    TypeError on every call.

    So the docstring is the fix: say it is inert and say which knob is not.
    """
    assert "save_qc" not in inspect.signature(
        spacrops.FOVAlignAndCropper.__init__).parameters, (
        "FOVAlignAndCropper grew a save_qc; qc_outlines can now be wired to it")

    doc = inspect.getdoc(spacrops.align_image_to_stitch)
    assert "NOT used" in doc
    assert "spacrStitcher" in doc, "the docstring does not say where to go instead"


def test_qc_outlines_is_no_longer_a_second_settings_key():
    """It defaulted True in the ops settings while save_qc, the key that
    actually gates the overlays, defaults False -- so leaving the defaults
    alone announced QC output that was never written.

    BOTH ENDS had to go. The key did have a reader:
    `qc_outlines=stitch_settings["qc_outlines"]` in ops_preprocess. But that
    reader only handed it to the inert parameter, so removing the default
    alone turned a silent no-op into a KeyError -- which is how this was
    caught. A key whose one reader feeds a parameter that does nothing is not
    read; it is laundered.
    """
    source = inspect.getsource(spacrops)
    assert 'setdefault("qc_outlines"' not in source, (
        "the ops defaults declare qc_outlines again")
    assert 'qc_outlines=stitch_settings' not in source, (
        "ops_preprocess passes it to the inert parameter again")
    assert 'setdefault("save_qc"' in source, "save_qc is the key that remains"
