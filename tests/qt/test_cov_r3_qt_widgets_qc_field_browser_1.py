"""Edge inputs the QC field browser must survive without hiding a field.

Every path here is one a real plate reaches: a scorecard written before the
plate path was recorded, a field name still carrying its ``.npy`` suffix, a
legacy merged array with no plane manifest, a mask stack whose files were
replaced by something that is not a 2-D mask, and flat or all-NaN planes.
The browser is the last step before a user quarantines data, so each of
these must produce an explained, still-usable dialog rather than an
exception or a silently dropped field.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import seg_qc  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldImage,
    QCFieldTarget,
    QCFieldVerdict,
    finding_targets,
    load_qc_field,
    render_qc_field,
    targets_from_digest,
)

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"


def _qc(field, object_type="cell", severity="fail", flags=("under_segmented",)):
    return seg_qc.FieldQC(field, object_type, 1, list(flags), {}, severity, "")


def _digest(root, cards, findings=()):
    return seg_qc.QCDigest(
        root=str(root), verdict="fail", headline="review",
        scorecards=list(cards), findings=list(findings))


def _legacy_plate(tmp_path, planes=5, stacks=None, manifest=None,
                  field=FIELD, name="plate1"):
    """A merged array with no manifest, plus whatever mask stacks are asked for."""
    plate = tmp_path / name
    merged = plate / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 10, planes), dtype=np.uint16)
    ramp = np.arange(80, dtype=np.uint16).reshape(8, 10)
    for index in range(planes):
        array[..., index] = ramp * (index + 1)
    np.save(merged / f"{field}.npy", array)
    if manifest is not None:
        (merged / ".spacr_plane_layout.json").write_text(
            manifest, encoding="utf-8")
    for object_type, mask in (stacks or {}).items():
        folder = plate / "norm_channel_stack" / f"{object_type}_mask_stack"
        folder.mkdir(parents=True, exist_ok=True)
        if mask is not None:
            np.save(folder / f"{field}.npy", mask)
    return plate, array


def _target(plate, object_types=("cell",), field=FIELD):
    return QCFieldTarget(
        field=field, plate_root=str(plate),
        merged_dir=str(plate / "merged"),
        verdicts=tuple(
            QCFieldVerdict(object_type, "fail", ("under_segmented",), "")
            for object_type in object_types))


def test_a_pathless_scorecard_and_a_dot_npy_field_still_reach_the_browser(
        tmp_path):
    """A scorecard with no file path must not orphan its fields.

    Older QC runs recorded no scorecard path and wrote field names with the
    ``.npy`` suffix still attached.  If the browser derived the plate root
    only from the scorecard path it would open ``/merged`` at the filesystem
    root, and a suffixed name would never match the file on disk -- the
    banner would list fields that open to "already gone".  A blank field
    name, which a truncated CSV row produces, must be dropped rather than
    become a target that points at the plate folder itself.
    """
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [
        _qc("   "),
        _qc(f"{FIELD}.NPY"),
        _qc("plate1_A02_1"),
    ])

    targets = targets_from_digest(_digest(plate, [card]))

    assert [target.field for target in targets] == [FIELD, "plate1_A02_1"]
    assert targets[0].plate_root == os.path.abspath(str(plate))
    assert targets[0].merged_dir == os.path.join(
        os.path.abspath(str(plate)), "merged")


def test_each_finding_opens_only_the_fields_it_actually_implicates(tmp_path):
    """A finding's browser list must not spill into unrelated fields.

    The banner offers one browser per finding.  If the field/plate/object
    filters were dropped, clicking a single-field cell finding would open
    every field on every plate in the digest and the user would quarantine
    the wrong array.  All three filters are exercised against the same
    digest so the empty result below is proved to be a filter, not an empty
    digest.
    """
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [_qc(FIELD), _qc("plate1_A02_1")])
    other = seg_qc.Scorecard("", "nucleus", [_qc("plate2_B01_1", "nucleus")])
    exact = seg_qc.Finding(
        severity="fail", kind="flag", headline="one field",
        flag="under_segmented", plate="plate1", object_type="cell",
        fields=(FIELD,))
    positional = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="plate 2 drifts",
        plate="plate2")
    wrong_type = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="no pathogens here",
        object_type="pathogen")
    digest = _digest(plate, [card, other], [exact, positional, wrong_type])

    assert [t.field for t in finding_targets(digest, exact)] == [FIELD]
    assert [t.field for t in finding_targets(digest, positional)] == [
        "plate2_B01_1"]
    assert finding_targets(digest, wrong_type) == ()


def test_plate_matching_falls_back_to_the_name_prefix(tmp_path, monkeypatch):
    """A plate-scoped finding must still find its fields if parsing fails.

    ``parse_field_name`` is the strict reader; when it cannot be used the
    browser falls back to the leading ``plate_`` segment.  Without that
    fallback a positional finding would match nothing and the banner's
    "inspect" link would open an empty dialog for the very pattern it just
    reported.
    """
    def _boom(name):
        raise RuntimeError("field-name parser unavailable")

    monkeypatch.setattr(seg_qc, "parse_field_name", _boom)
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [_qc(FIELD)])
    other = seg_qc.Scorecard("", "cell", [_qc("plate2_B01_1")])
    positional = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="plate 1 drifts",
        plate="plate1")
    digest = _digest(plate, [card, other], [positional])

    assert [t.field for t in finding_targets(digest, positional)] == [FIELD]


def test_an_unreadable_merged_array_is_explained_not_raised(tmp_path):
    """A truncated merged file must report itself, not crash the dialog.

    Merged arrays are written during long runs and a killed run leaves a
    partial file.  The browser loads on a worker thread; an uncaught
    exception there loses the whole triage session instead of letting the
    user step past one bad field.
    """
    plate = tmp_path / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True)
    (merged / f"{FIELD}.npy").write_bytes(b"\x93NUMPY truncated")

    payload = load_qc_field(_target(plate))

    assert payload.intensities is None
    assert FIELD in payload.error and "Could not read" in payload.error
    assert payload.path.endswith(f"{FIELD}.npy")


def test_a_two_dimensional_merged_array_names_the_shape_it_found(tmp_path):
    """A single-plane file saved where a merged stack belongs must say so.

    Renderers index ``array[..., channel]``; a 2-D array would raise deep in
    the render worker with an IndexError that names nothing.  Reporting the
    shape tells the user their merged folder holds the wrong artifact.
    """
    plate = tmp_path / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True)
    np.save(merged / f"{FIELD}.npy", np.zeros((8, 10), dtype=np.uint16))

    payload = load_qc_field(_target(plate))

    assert payload.intensities is None
    assert "Expected a merged" in payload.error
    assert "(8, 10)" in payload.error


def test_a_corrupt_plane_manifest_warns_and_the_field_still_opens(tmp_path):
    """Unreadable layout metadata must degrade, not block triage.

    The manifest is a convenience written next to the merged arrays; a
    half-written or hand-edited one is recoverable because the legacy
    plane-count rules still apply.  Refusing to open the field would make a
    stray sidecar file hide every image on the plate.
    """
    plate, _array = _legacy_plate(tmp_path, planes=5, manifest="{not json")

    payload = load_qc_field(_target(plate))

    assert payload.error == ""
    assert payload.intensities is not None
    assert any("Plane-layout metadata could not be read" in warning
               for warning in payload.warnings), payload.warnings


def test_a_legacy_plate_counts_intensities_from_the_stacks_it_finds(tmp_path):
    """Discovered mask stacks, not a guess, set the legacy channel count.

    A legacy six-plane array with one mask stack holds five intensity
    channels.  Assuming the default four would render a mask plane as if it
    were intensity data -- a black-and-white label image the user would read
    as a broken channel.  The nucleus plane, which has no stack folder, must
    still come from its default plane index, and an object type with neither
    a stack nor a default plane must be named in a warning rather than
    silently missing from the layer list.
    """
    cell = np.zeros((8, 10), dtype=np.uint16)
    cell[2:6, 3:7] = 1
    plate, array = _legacy_plate(tmp_path, planes=6, stacks={"cell": cell})

    payload = load_qc_field(
        _target(plate, ("cell", "nucleus", "mito")))

    assert payload.channel_names == ("1", "2", "3", "4", "5")
    assert payload.intensities.shape == (8, 10, 5)
    assert set(payload.masks) == {"cell", "nucleus"}
    assert np.array_equal(payload.masks["nucleus"], array[..., 5])
    assert int(payload.masks["cell"].max()) == 1
    assert any("mito" in warning for warning in payload.warnings), \
        payload.warnings


def test_a_legacy_plate_with_no_stacks_falls_back_to_the_default_planes(
        tmp_path):
    """With no stack folders the default plane layout must still find masks.

    Archived plates keep the merged arrays and lose the per-object stacks.
    Their contract is spaCR's default four intensity channels followed by
    the mask planes; without that fallback the browser would show the images
    with no outlines at all and the user could not tell an over-segmented
    field from a clean one.
    """
    plate, array = _legacy_plate(tmp_path, planes=6)

    payload = load_qc_field(_target(plate, ("cell", "nucleus")))

    assert payload.channel_names == ("1", "2", "3", "4")
    assert payload.intensities.shape == (8, 10, 4)
    assert np.array_equal(payload.masks["cell"], array[..., 4])
    assert np.array_equal(payload.masks["nucleus"], array[..., 5])


def test_a_short_legacy_array_treats_every_plane_as_intensity(tmp_path):
    """A three-plane array holds no mask planes and must not invent one.

    Clamping to the default four-channel layout on a three-plane array would
    index past the end, or worse, present plane 2 as a mask.  Every plane is
    intensity here, and the missing cell mask has to be stated so the user
    knows the outline toggle is empty for a reason.
    """
    plate, _array = _legacy_plate(tmp_path, planes=3)

    payload = load_qc_field(_target(plate, ("cell",)))

    assert payload.channel_names == ("1", "2", "3")
    assert payload.intensities.shape == (8, 10, 3)
    assert payload.masks == {}
    assert any("cell" in warning for warning in payload.warnings), \
        payload.warnings


def test_each_broken_mask_stack_is_named_and_the_good_ones_still_load(
        tmp_path):
    """One unusable mask file must not cost the user the other outlines.

    Mask stacks are written by separate segmentation passes, so a plate
    routinely mixes a healthy stack with one that holds a stale shape or a
    stack of frames.  Loading must keep the masks it can read, keep the
    trailing singleton axis that some writers add, and name each failure
    with its object type so the user knows which segmentation to rerun.
    """
    cell = np.zeros((8, 10, 1), dtype=np.uint16)
    cell[2:6, 3:7, 0] = 1
    plate, _array = _legacy_plate(tmp_path, planes=5, stacks={
        "cell": cell,
        "nucleus": np.zeros((2, 3, 4), dtype=np.uint16),
        "pathogen": np.zeros((4, 4), dtype=np.uint16),
    })

    payload = load_qc_field(
        _target(plate, ("cell", "nucleus", "pathogen")))

    assert set(payload.masks) == {"cell"}
    assert payload.masks["cell"].shape == (8, 10)
    joined = " | ".join(payload.warnings)
    assert "expected a 2-D mask" in joined, joined
    assert "does not match image" in joined, joined
    assert "nucleus" in joined and "pathogen" in joined, joined


def test_a_single_channel_field_renders_grey_and_survives_all_nan():
    """One-channel and all-NaN planes must render, not divide by nothing.

    A brightfield-only plate has a single intensity channel: the composite
    view has to repeat it across R, G and B instead of leaving two empty
    planes.  An all-NaN plane -- what a failed flat-field correction writes
    -- has no finite percentile, and normalising it would produce NaN
    pixels that reach QImage as noise.  Both are rendered here so the black
    frame below is proved to come from the NaN input.
    """
    ramp = np.arange(16, dtype=np.float32).reshape(4, 4)
    real = render_qc_field(
        QCFieldImage(intensities=ramp[..., None].copy()), -1, ())
    blank = render_qc_field(
        QCFieldImage(intensities=np.full((4, 4, 1), np.nan, dtype=np.float32)),
        -1, ())

    assert real.shape == (4, 4, 3)
    assert np.array_equal(real[..., 0], real[..., 2]), "grey means R == B"
    assert int(real.max()) == 255 and int(real.min()) == 0
    assert blank.shape == (4, 4, 3)
    assert int(blank.max()) == 0


def test_a_flat_channel_renders_black_instead_of_amplifying_noise():
    """A constant plane has no contrast and must not be stretched.

    A dead or saturated channel is constant; a 2-98 percentile stretch on it
    divides by zero and, with the fallback also degenerate, would emit
    garbage.  The neighbouring live channel is rendered in the same test so
    the black frame is attributable to the flat data, not to the renderer
    being broken.
    """
    intensities = np.zeros((4, 4, 2), dtype=np.uint16)
    intensities[..., 0] = 7
    intensities[..., 1] = np.arange(16, dtype=np.uint16).reshape(4, 4)
    payload = QCFieldImage(intensities=intensities)

    flat = render_qc_field(payload, 0, ())
    live = render_qc_field(payload, 1, ())

    assert int(flat.max()) == 0
    assert int(live.max()) == 255
    assert np.any(live != flat)


def test_custom_object_roles_get_their_own_stable_outline_colour():
    """Organelle variants and custom roles must be told apart on screen.

    The browser draws every object type over the same image, so two roles
    sharing a colour makes the overlay unreadable.  ``organelle_b`` belongs
    to the organelle family and takes its colour; a project-specific role
    like ``mito`` gets a derived colour that must be distinct and identical
    on every redraw, or the outline would flicker colour as the user steps
    through fields.
    """
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    payload = QCFieldImage(
        intensities=np.zeros((4, 4, 3), dtype=np.uint16),
        masks={"organelle_b": mask, "mito": mask})

    organelle = render_qc_field(payload, -1, ("organelle_b",))
    custom = render_qc_field(payload, -1, ("mito",))
    again = render_qc_field(payload, -1, ("mito",))

    assert tuple(int(v) for v in organelle[1, 1]) == (35, 205, 235)
    assert tuple(int(v) for v in custom[1, 1]) != (35, 205, 235)
    assert int(np.asarray(custom[1, 1]).max()) > 0
    assert np.array_equal(custom, again), "the derived colour must be stable"
    assert np.any(custom != organelle)
