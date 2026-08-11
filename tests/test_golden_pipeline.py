"""A whole pipeline run, end to end, against numbers that were known in advance.

Silent numeric drift is this codebase's signature failure: a Poisson model
that ignored its exposure offset, an illumination correction with a 0.65
bias slope, a measure run that wrote no organelle table. Every one of them
ran green, because every test that covered them asked whether *an artefact
appeared*, not what was *in* it.

This module runs mask -> measure -> classify -> regression on one tiny
fixture and asserts the number at every stage. It is deliberately built so
that the fixture's ground truth is known **analytically**: every object is
an axis-aligned square of a chosen side, and every pixel of the image is
painted with one of five chosen constants. A 12x12 square has an area of
144; the mean of a region painted 3000 is 3000; a 4-neighbourhood perimeter
is ``4 * (side - 1)``. Nothing here was read off a run and blessed.

Where a number *could* only come from a run it is marked (RECORDED) in the
test's docstring and carries an independent invariant beside it - a
conservation law, a monotone ordering, or a sum that must equal a total.
Everything else is marked (DERIVED) and the derivation is written out.

The four stages and what each is driven through
----------------------------------------------

1. **mask** - :func:`spacr.external_masks.plan_external_masks` /
   :func:`~spacr.external_masks.run_external_masks`. This is the real
   mask-ingest writer: it converts the intensity images, pairs each label
   image to its field, and builds the canonical ``stack/``, ``masks/`` and
   ``merged/`` folders that Measure consumes. It is used in place of
   :func:`spacr.core.preprocess_generate_masks` because that one segments
   with Cellpose-SAM, which needs a GPU and a model download, and because a
   segmentation whose output is not known in advance cannot be a golden
   input. The *ingest* half - plane order, dtype, label preservation - is
   what is checked here; see the note at the bottom of this docstring.

2. **measure** - :func:`spacr.measure.measure_crop`, reached through
   ``run_external_masks`` exactly as a real external-mask project reaches
   it. Writes ``measurements.db`` and the per-object PNG crops.

3. **classify** - :func:`spacr.deep_spacr.apply_model` over the crops
   Measure wrote, using a hand-built one-layer torch probe on the CPU whose
   weights are set (not trained), so the score of every crop is the sigmoid
   of a linear functional of pixels this file also predicts.

4. **regression** - :func:`spacr.ml.process_scores` (the per-well
   aggregation) and :func:`spacr.ml.regression_model` /
   :func:`~spacr.ml.process_model_coefficients` (the fitting core that
   ``perform_regression`` dispatches to). The response is built so that the
   model fits it *exactly*, which makes the maximum-likelihood estimate an
   analytic quantity rather than a recorded one.

What this module found
----------------------

``TestTheStockImporterLosesTheOrganelleSummary`` is ``xfail(strict=True)``:
:func:`spacr.external_masks.run_external_masks` does ``list(...)`` on a
setting that defaults to the *string* ``'cell'``, turning it into
``['c', 'e', 'l', 'l']``, so ``measure.py``'s ``if "cell" in
settings['summarize_organelles_by']`` is False and the
``cell_organelle_summary`` table is never written. Full detail on the test.

Runtime and determinism
-----------------------

15-30 s wall clock for the whole module (the spread is machine load; the
work is 13 measured fields plus 36 crop inferences). No network, no GPU, no
Cellpose, no TensorFlow; the classifier is pinned to the CPU so a machine
with a GPU and one without produce the same numbers. The measure stage was
run twice from scratch on the same fixture and every numeric column of
every table was bit-identical, so the tolerances below are about float
representation, not about run-to-run noise.

Tolerances: integers and counts are compared exactly; derived float
quantities at ``rel=1e-12``; the two fitted models at ``rel=1e-9`` (the
solvers' linear algebra, measured at 4e-16 and 6e-16); and only the
classifier's scores at ``rel=1e-5``, for the reason spelled out on
:data:`SCORE_TOLERANCE`.

Not covered here
----------------

Cellpose segmentation itself. A golden test cannot assert what a neural
network will label without pinning the weights, and the weights are a
download. Stage 1 therefore covers mask *ingest* and the merged-array
contract that segmentation feeds; the segmentation call is covered by the
GPU-marked suite (``tests/test_full_pipeline_e2e.py``).
"""
from __future__ import annotations

import math
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# The fixture, stated as numbers before a line of spaCR runs
# ---------------------------------------------------------------------------

FIELD = 64                      #: side of every synthetic field, in pixels

#: Painted constants. Every pixel of channel 0 is one of ``BG0``, ``CYTO0``,
#: ``NUC0`` or that well's pathogen level; every pixel of channel 1 is one of
#: ``BG1``, ``CYTO1``, ``NUC1``, ``PATH1``. Organelles are painted at the
#: cytoplasm level, so they change the *areas* without changing the *means*.
BG0, CYTO0, NUC0 = 100, 1000, 3000
BG1, CYTO1, NUC1, PATH1 = 200, 2000, 400, 800

#: Channel-0 pathogen level of well ``i`` (1-based). Linear in ``i`` on
#: purpose: it is the per-well response the regression stage recovers.
def pathogen_level(well_index: int) -> int:
    return 5000 + 500 * well_index


#: ``label -> (cell, nucleus, pathogen, [organelle, organelle])`` as
#: ``(rows, cols)`` slices. Every region is an axis-aligned square, every
#: child sits strictly inside its cell, and no two children touch.
GEOMETRY = {
    1: ((slice(8, 20), slice(8, 20)), (slice(10, 16), slice(10, 16)),
        (slice(16, 19), slice(16, 19)),
        [(slice(8, 10), slice(8, 10)), (slice(8, 10), slice(12, 14))]),
    2: ((slice(8, 18), slice(24, 34)), (slice(10, 14), slice(26, 30)),
        (slice(14, 17), slice(30, 33)),
        [(slice(8, 10), slice(24, 26)), (slice(8, 10), slice(28, 30))]),
    3: ((slice(30, 39), slice(8, 17)), (slice(32, 36), slice(10, 14)),
        (slice(36, 38), slice(14, 16)),
        [(slice(30, 32), slice(8, 10)), (slice(30, 32), slice(12, 14))]),
    4: ((slice(30, 38), slice(24, 32)), (slice(32, 35), slice(26, 29)),
        (slice(35, 37), slice(29, 31)),
        [(slice(30, 32), slice(24, 26)), (slice(30, 32), slice(28, 30))]),
}

#: Side of each cell square, read straight off ``GEOMETRY``.
SIDE = {1: 12, 2: 10, 3: 9, 4: 8}

#: (DERIVED) Areas, in pixels, of every compartment of every cell label.
#: ``cell`` is ``side**2``; the rest are the products of their own slices.
#: ``organelle_total`` is two 2x2 squares.
CELL_AREA = {1: 144, 2: 100, 3: 81, 4: 64}
NUCLEUS_AREA = {1: 36, 2: 16, 3: 16, 4: 9}
PATHOGEN_AREA = {1: 9, 2: 9, 3: 4, 4: 4}
ORGANELLE_COUNT = {label: 2 for label in GEOMETRY}
ORGANELLE_AREA = {label: 4 for label in GEOMETRY}       # each 2x2
ORGANELLE_TOTAL = {label: 8 for label in GEOMETRY}      # 2 x 4

#: (DERIVED) Measure defines cytoplasm as the cell minus every interior
#: object it was given - nucleus, pathogen AND organelle.
CYTOPLASM_AREA = {
    label: CELL_AREA[label] - NUCLEUS_AREA[label]
    - PATHOGEN_AREA[label] - ORGANELLE_TOTAL[label]
    for label in GEOMETRY
}

#: (DERIVED) Every cell pixel that is neither nucleus nor pathogen carries
#: the cytoplasm intensity, because organelles are painted at that level.
CYTO_LEVEL_PIXELS = {
    label: CELL_AREA[label] - NUCLEUS_AREA[label] - PATHOGEN_AREA[label]
    for label in GEOMETRY
}

#: 12 wells over two rows, so both the row and the column of the well name
#: have to be parsed correctly for the per-well numbers to line up.
WELLS = ["A01", "A02", "A03", "A04", "A05", "A06",
         "B01", "B02", "B03", "B04", "B05", "B06"]

#: Cells present in each well, 1-based-index aligned with ``WELLS``. Varying
#: it is what makes the Poisson exposure below non-constant.
CELLS_PER_WELL = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]

#: (DERIVED) How many wells hold a given cell label: every well has labels
#: 1 and 2, the eight three-cell-and-up wells have label 3, the four
#: four-cell wells have label 4. 12 + 12 + 8 + 4 == 36 objects.
WELLS_WITH_LABEL = {label: sum(1 for n in CELLS_PER_WELL if n >= label)
                    for label in (1, 2, 3, 4)}

PNG_SIDE = 24                   #: crop size written by Measure
PNG_PIXELS = 3 * PNG_SIDE * PNG_SIDE

#: The linear probe used as the classifier. ``logit = WEIGHT * mean(x) +
#: BIAS`` where ``x`` is the normalised image, because every weight of the
#: single Linear layer is set to ``WEIGHT / PNG_PIXELS``.
PROBE_WEIGHT, PROBE_BIAS = 4.0, 4.0

#: Tolerance for a score that came back through :func:`apply_model`.
#:
#: LOOSER THAN THIS FILE'S 1e-12 DEFAULT, AND THIS IS WHY. ``apply_model``
#: casts every batch with ``dtype=torch.float`` - float32, hard-coded - so
#: the logit is a 1728-term float32 dot product, while the expected value
#: this file computes is float64. Measured worst case over all 36 crops:
#: 4.1e-6 relative on the CPU (float32 eps is 6e-8 and the accumulation is
#: over 1728 terms), 1.6e-7 on CUDA. 1e-5 is the smallest round number
#: above the CPU figure. It is not slack for drift: the failures this
#: stage exists to catch - a lost normalisation, a mis-centred crop, a
#: missing sigmoid, an illumination slope - are tens of percent, and the
#: *input* to the classifier is asserted exactly, as integer pixel counts,
#: by ``test_the_crops_carry_the_object_areas_as_pixel_counts``.
SCORE_TOLERANCE = 1e-5

#: Poisson coefficients the count frame is built to reproduce exactly.
POISSON_B0, POISSON_B1 = -0.75, 1.5
POISSON_COUNTS = [1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12]


def well_prc(well: str) -> str:
    """``'A01' -> 'plate1_r1_c1'`` - the key spaCR builds from a well name."""
    return f"plate1_r{'AB'.index(well[0]) + 1}_c{int(well[1:])}"


def cell_mean(label: int, channel: int, well_index: int) -> float:
    """(DERIVED) Area-weighted mean of the constants painted inside a cell.

    Every cell pixel is one of three levels, and how many of each is fixed
    by :data:`GEOMETRY`, so the mean is a rational number known in advance.
    """
    if channel == 0:
        cyto, nucleus, pathogen = CYTO0, NUC0, pathogen_level(well_index)
    else:
        cyto, nucleus, pathogen = CYTO1, NUC1, PATH1
    total = (CYTO_LEVEL_PIXELS[label] * cyto
             + NUCLEUS_AREA[label] * nucleus
             + PATHOGEN_AREA[label] * pathogen)
    return total / CELL_AREA[label]


def nucleus_grey_level(well_index: int) -> int:
    """(DERIVED) The 8-bit value Measure writes for a nucleus pixel.

    ``normalize_to_dtype`` stretches the crop's non-zero pixels between the
    0th and 100th percentile - here the cytoplasm level and the pathogen
    level - across the full ``uint16`` range and **truncates**, and
    ``crops.narrow_to_uint8`` then takes the high byte. So the nucleus,
    which sits at ``NUC0``, lands at::

        int((NUC0 - CYTO0) / (pathogen - CYTO0) * 65535) >> 8
    """
    span = pathogen_level(well_index) - CYTO0
    return int((NUC0 - CYTO0) / span * 65535) >> 8


#: (DERIVED) Same arithmetic on channel 1, where the low end is the nucleus
#: and the high end the cytoplasm: ``int(400/1600 * 65535) >> 8`` == 63.
PATHOGEN_GREY_CH1 = int((PATH1 - NUC1) / (CYTO1 - NUC1) * 65535) >> 8


# ---------------------------------------------------------------------------
# Building the inputs
# ---------------------------------------------------------------------------

def _planes(n_cells: int, pathogen: int):
    """Return ``(channel0, channel1, cell, nucleus, pathogen, organelle)``."""
    cell = np.zeros((FIELD, FIELD), np.uint16)
    nucleus = np.zeros((FIELD, FIELD), np.uint16)
    parasite = np.zeros((FIELD, FIELD), np.uint16)
    organelle = np.zeros((FIELD, FIELD), np.uint16)
    for label in range(1, n_cells + 1):
        cell_sl, nucleus_sl, pathogen_sl, organelle_sl = GEOMETRY[label]
        cell[cell_sl] = label
        nucleus[nucleus_sl] = label
        parasite[pathogen_sl] = label
        for index, one in enumerate(organelle_sl):
            organelle[one] = 2 * (label - 1) + index + 1

    channel0 = np.full((FIELD, FIELD), BG0, np.uint16)
    channel0[cell > 0] = CYTO0
    channel0[nucleus > 0] = NUC0
    channel0[parasite > 0] = pathogen
    channel1 = np.full((FIELD, FIELD), BG1, np.uint16)
    channel1[cell > 0] = CYTO1
    channel1[nucleus > 0] = NUC1
    channel1[parasite > 0] = PATH1
    return channel0, channel1, cell, nucleus, parasite, organelle


def _write_inputs(root, wells, cells_per_well):
    """Write ``images/`` and one folder of label images per object type.

    ``plate/well/field`` on both sides, because that is what pairs a mask
    folder to an image folder: :func:`spacr.external_masks._pair_masks`
    matches on ``(plate, well, field)``, and only the ``plate_well`` layout
    gives a mask folder the same plate key as the image folder.
    """
    images = os.path.join(root, "images")
    mask_roots = {name: os.path.join(root, f"{name}_masks")
                  for name in ("cell", "nucleus", "pathogen", "organelle")}
    truth = {}
    for index, well in enumerate(wells, start=1):
        planes = _planes(cells_per_well[index - 1], pathogen_level(index))
        channel0, channel1 = planes[0], planes[1]
        folder = os.path.join(images, "plate1", well)
        os.makedirs(folder, exist_ok=True)
        tifffile.imwrite(os.path.join(folder, "f01_C1.tif"), channel0)
        tifffile.imwrite(os.path.join(folder, "f01_C2.tif"), channel1)
        for name, array in zip(("cell", "nucleus", "pathogen", "organelle"),
                               planes[2:]):
            folder = os.path.join(mask_roots[name], "plate1", well)
            os.makedirs(folder, exist_ok=True)
            tifffile.imwrite(os.path.join(folder, f"f01_{name}_mask.tif"),
                             array)
        truth[well] = {"channel0": channel0, "channel1": channel1,
                       "cell": planes[2], "nucleus": planes[3],
                       "pathogen": planes[4], "organelle": planes[5]}
    return images, mask_roots, truth


def _import(root, destination, *, wells, cells_per_well, extra):
    """Run the real ingest + measure chain and return ``(result, truth)``."""
    from spacr import external_masks as em

    images, mask_roots, truth = _write_inputs(root, wells, cells_per_well)
    groups = em.detect_inputs([images, *mask_roots.values()])
    settings = {
        "inputs": [group.to_dict() for group in groups],
        "dst": destination,
        "layout": "plate_well",
        "experiment": "golden",
        "n_jobs": 1,
        "plot": False,
        "verbose": False,
        # Every object is drawn at its exact size, so nothing may be
        # filtered by area: a min_size default that moved would otherwise
        # silently change which objects the goldens are about.
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "organelle_min_size": 0,
    }
    settings.update(extra)
    plan = em.plan_external_masks(settings)
    assert plan.ok, plan.summary()
    return em.run_external_masks(plan, settings), truth, plan


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """The whole project: 12 wells ingested and measured, once."""
    root = tmp_path_factory.mktemp("golden")
    result, truth, plan = _import(
        str(root), os.path.join(str(root), "project"),
        wells=WELLS, cells_per_well=CELLS_PER_WELL,
        extra={
            "save_png": True, "crop_mode": ["cell"],
            "png_size": [PNG_SIDE, PNG_SIDE], "png_dims": [0, 1],
            # 0/100 percentiles: min-max over the crop's own object pixels,
            # so the grey level of every crop pixel is a closed form of the
            # constants painted into it (see nucleus_grey_level).
            "normalize": [0, 100], "normalize_by": "png",
            # Explicit, because the stock value is the string 'cell' and
            # the importer mangles it - the bug this module records.
            "summarize_organelles_by": ["cell", "nucleus", "pathogen",
                                        "organelle"],
        })
    return {"result": result, "truth": truth, "plan": plan,
            "root": str(root), "db": result.db_path,
            "dst": result.destination}


@pytest.fixture(scope="module")
def tables(project):
    """Every measurement table as a DataFrame, read once."""
    connection = sqlite3.connect(project["db"])
    try:
        names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        frames = {name: pd.read_sql(f'SELECT * FROM "{name}"', connection)
                  for name in names}
    finally:
        connection.close()
    return frames


def _by_well_label(frame):
    """Index a measurement frame by ``(well_index, object_label)``."""
    order = {well_prc(well): index for index, well in enumerate(WELLS, start=1)}
    keyed = frame.copy()
    keyed["well_index"] = (
        "plate1_" + keyed["rowID"].astype(str) + "_"
        + keyed["columnID"].astype(str)).map(order)
    assert keyed["well_index"].notna().all()
    return keyed.set_index(["well_index", "object_label"]).sort_index()


# ===========================================================================
# Stage 1 - mask ingest
# ===========================================================================

class TestStage1MaskIngest:
    """What the mask stage hands to Measure: planes, order, dtype, labels."""

    def test_the_plan_puts_the_masks_after_the_intensity_channels(self,
                                                                  project):
        """(DERIVED) Two intensity channels, then one plane per object type.

        The plane index of each mask *is* the ``<object>_mask_dim`` Measure
        is then driven with, so an off-by-one here measures the wrong array
        and every number downstream is somebody else's.
        """
        plan = project["plan"]
        assert plan.n_channels == 2
        assert plan.object_types == ["cell", "nucleus", "pathogen",
                                     "organelle"]
        assert plan.mask_dims == {"cell": 2, "nucleus": 3, "pathogen": 4,
                                  "organelle": 5}
        assert len(plan.stems) == len(WELLS) == 12

    def test_every_field_becomes_one_merged_stack_of_the_declared_shape(
            self, project):
        """(DERIVED) 12 fields; ``stack`` (64, 64, 2), ``merged`` (64, 64, 6).

        ``stack`` is the intensity channels alone and ``merged`` is those
        channels with one plane appended per object type, so the two shapes
        differ by exactly the number of mask planes. uint16 in both, which
        is the dtype the whole measure path is written against.
        """
        merged = sorted(os.listdir(os.path.join(project["dst"], "merged")))
        stacks = sorted(os.listdir(os.path.join(project["dst"], "stack")))
        assert len(merged) == len(stacks) == 12
        assert merged == stacks
        for name in merged:
            array = np.load(os.path.join(project["dst"], "merged", name))
            assert array.shape == (FIELD, FIELD, 6), name
            assert array.dtype == np.uint16, name
            intensity = np.load(os.path.join(project["dst"], "stack", name))
            assert intensity.shape == (FIELD, FIELD, 2), name
            assert intensity.dtype == np.uint16, name
            assert np.array_equal(intensity, array[:, :, :2]), name

    def test_the_label_images_survive_the_ingest_bit_for_bit(self, project):
        """(DERIVED) The merged mask planes equal the label images written.

        Not "a mask is there": the same integers, in the same places. A
        relabel, a resize or a dtype narrowing anywhere in the ingest shows
        up here as an array comparison rather than as a wrong area later.
        """
        for well in WELLS:
            stem = f"plate1_{well}_1"
            array = np.load(os.path.join(project["dst"], "merged",
                                         f"{stem}.npy"))
            truth = project["truth"][well]
            assert np.array_equal(array[:, :, 0], truth["channel0"]), well
            assert np.array_equal(array[:, :, 1], truth["channel1"]), well
            assert np.array_equal(array[:, :, 2], truth["cell"]), well
            assert np.array_equal(array[:, :, 3], truth["nucleus"]), well
            assert np.array_equal(array[:, :, 4], truth["pathogen"]), well
            assert np.array_equal(array[:, :, 5], truth["organelle"]), well
            # ...and the per-object mask stack the same masks were split into.
            for index, name in enumerate(("cell", "nucleus", "pathogen",
                                          "organelle"), start=2):
                stack = np.load(os.path.join(
                    project["dst"], "masks", f"{name}_mask_stack",
                    f"{stem}.npy"))
                assert np.array_equal(stack, truth[name]), (well, name)

    def test_the_well_name_becomes_the_row_and_column_it_encodes(self,
                                                                 tables):
        """(DERIVED) A01 -> r1/c1 ... B06 -> r2/c6.

        Two rows on purpose: a parser that ignores the letter puts all
        twelve wells on r1, which is invisible in a one-row plate and
        collapses six pairs of wells in every per-well aggregate.
        """
        cells = tables["cell"]
        seen = set(zip(cells["rowID"], cells["columnID"]))
        assert seen == {(f"r{'AB'.index(w[0]) + 1}", f"c{int(w[1:])}")
                        for w in WELLS}
        assert set(cells["plateID"]) == {"plate1"}
        assert set(cells["fieldID"]) == {"f1"}


# ===========================================================================
# Stage 2 - measure
# ===========================================================================

class TestStage2Measure:
    """Every measurement, against the square that was drawn."""

    def test_every_table_holds_exactly_the_objects_that_were_drawn(self,
                                                                   tables):
        """(DERIVED) 36 cells (2+3+4 twice over, x2 rows), 72 organelles.

        ``sum(CELLS_PER_WELL) == 36``; each cell carries one nucleus, one
        pathogen, one cytoplasm and two organelles. A count that is a
        multiple of the truth is the double-counting failure; one that is
        short is a filter nobody asked for.
        """
        assert sum(CELLS_PER_WELL) == 36
        for name in ("cell", "nucleus", "pathogen", "cytoplasm",
                     "cell_organelle_summary", "nucleus_organelle_summary",
                     "pathogen_organelle_summary", "png_list"):
            assert name in tables, f"{name} was not written"
            assert len(tables[name]) == 36, name
        assert len(tables["organelle"]) == 72

    @pytest.mark.parametrize("label", sorted(GEOMETRY))
    def test_object_areas_are_the_pixel_counts_that_were_painted(self, tables,
                                                                 label):
        """(DERIVED) 12x12 = 144, 6x6 = 36, 3x3 = 9, 2x2 = 4.

        Integers, so compared exactly. ``area`` is the count of pixels
        carrying the label - if it ever becomes an area in micrometres the
        equality breaks, which is the point.
        """
        for table, expected in (("cell", CELL_AREA[label]),
                                ("nucleus", NUCLEUS_AREA[label]),
                                ("pathogen", PATHOGEN_AREA[label]),
                                ("cytoplasm", CYTOPLASM_AREA[label])):
            frame = tables[table]
            rows = frame[frame["object_label"] == label]
            values = set(rows[f"{table}_area"].tolist())
            assert values == {float(expected)}, (table, label, values)
        organelles = tables["organelle"]
        theirs = organelles[organelles["cell_id"] == label]
        assert set(theirs["organelle_area"].tolist()) == {4.0}

    def test_the_units_stamp_says_the_areas_are_pixels(self, tables):
        """(RECORDED, with the invariant beside it) 2-D, ``px``, no voxel size.

        The stamp is what tells a reader whether ``cell_area`` is 144
        pixels or 144 um^2. The invariant is the one above: the numbers in
        those columns really are the pixel counts, so a stamp that said
        anything else would be a lie about numbers this file has checked.
        """
        for name in ("cell", "nucleus", "pathogen", "cytoplasm"):
            frame = tables[name]
            assert set(frame["measurement_ndim"]) == {2}, name
            assert set(frame["measurement_units"]) == {"px"}, name
            assert frame["voxel_size_xy_um"].isna().all(), name

    @pytest.mark.parametrize("label", sorted(GEOMETRY))
    def test_shape_descriptors_are_the_closed_forms_for_a_square(self, tables,
                                                                 label):
        """(DERIVED) Perimeter ``4*(side-1)``; equivalent diameter
        ``2*sqrt(area/pi)``; a filled convex square has solidity and extent
        1, eccentricity 0 and Euler number 1.

        ``perimeter`` is the crossing-counted boundary length skimage
        reports for a solid square, which is ``4*(side-1)``: 44 px for the
        12x12 cell, 36 for the 10x10, 32 for the 9x9, 28 for the 8x8.
        """
        rows = tables["cell"]
        rows = rows[rows["object_label"] == label]
        side, area = SIDE[label], CELL_AREA[label]
        assert set(rows["cell_perimeter"]) == {float(4 * (side - 1))}
        assert rows["cell_equivalent_diameter_area"].to_numpy() == pytest.approx(
            2 * math.sqrt(area / math.pi), rel=1e-12)
        assert set(rows["cell_solidity"]) == {1.0}
        assert set(rows["cell_extent"]) == {1.0}
        assert set(rows["cell_eccentricity"]) == {0.0}
        assert set(rows["cell_euler_number"]) == {1}
        assert set(rows["cell_area_bbox"]) == {float(area)}
        assert set(rows["cell_convex_area"]) == {float(area)}

    @pytest.mark.parametrize("well_index", range(1, 13))
    def test_mean_intensities_are_the_constants_that_were_painted(
            self, tables, well_index):
        """(DERIVED) A region painted at one constant has that mean.

        Nucleus 3000 / 400, pathogen ``5000 + 500*i`` / 800, cytoplasm 1000
        / 2000 - each region is a single value, so its mean is that value
        exactly, whatever its shape. The cell is the area-weighted mean of
        the three, e.g. the 12x12 cell of well 1::

            (99*1000 + 36*3000 + 9*5500) / 144 == 1781.25
        """
        pathogen = pathogen_level(well_index)
        for table, channel0, channel1 in (("nucleus", NUC0, NUC1),
                                          ("pathogen", pathogen, PATH1),
                                          ("cytoplasm", CYTO0, CYTO1),
                                          ("organelle", CYTO0, CYTO1)):
            rows = _by_well_label(tables[table]).loc[well_index]
            assert set(rows[f"{table}_channel_0_mean_intensity"]) == {
                float(channel0)}, table
            assert set(rows[f"{table}_channel_1_mean_intensity"]) == {
                float(channel1)}, table

        cells = _by_well_label(tables["cell"]).loc[well_index]
        for label, row in cells.iterrows():
            for channel in (0, 1):
                assert row[f"cell_channel_{channel}_mean_intensity"] == (
                    pytest.approx(cell_mean(label, channel, well_index),
                                  rel=1e-12)), (well_index, label, channel)

    def test_integrated_intensity_is_the_area_times_the_mean(self, tables):
        """CONSERVATION LAW. Holds to the last bit, for every object.

        The sum of a region and its mean cannot drift apart: if one of them
        starts counting a different set of pixels than the other, this is
        the equality that breaks, and it breaks on every row at once.
        """
        for name in ("cell", "nucleus", "pathogen", "cytoplasm", "organelle"):
            frame = tables[name]
            for channel in (0, 1):
                product = (frame[f"{name}_area"]
                           * frame[f"{name}_channel_{channel}_mean_intensity"])
                assert np.array_equal(
                    frame[f"{name}_channel_{channel}_integrated_intensity"]
                    .to_numpy(dtype=float), product.to_numpy(dtype=float)), (
                        name, channel)

    def test_the_compartments_partition_the_cell_exactly(self, tables):
        """CONSERVATION LAW. cytoplasm + nucleus + pathogen + organelles == cell.

        Measure defines the cytoplasm by subtraction, so this is the check
        that the subtraction used the same objects the other tables
        describe. Every one of the 36 cells, in pixels, with no slack.
        """
        key = ["plateID", "rowID", "columnID", "fieldID", "object_label"]
        joined = (tables["cell"][key + ["cell_area"]]
                  .merge(tables["cytoplasm"][key + ["cytoplasm_area"]], on=key)
                  .merge(tables["nucleus"][key + ["nucleus_area"]], on=key)
                  .merge(tables["pathogen"][key + ["pathogen_area"]], on=key)
                  .merge(tables["cell_organelle_summary"][
                      key + ["organelle_summary_organelle_total_area"]], on=key))
        assert len(joined) == 36
        total = (joined["cytoplasm_area"] + joined["nucleus_area"]
                 + joined["pathogen_area"]
                 + joined["organelle_summary_organelle_total_area"])
        assert np.array_equal(total.to_numpy(dtype=float),
                              joined["cell_area"].to_numpy(dtype=float))

    def test_children_are_attributed_to_the_cell_that_contains_them(self,
                                                                    tables):
        """(DERIVED) Each nucleus/pathogen carries its own cell's id; each
        cell owns exactly two organelles.

        The geometry puts child ``n`` inside cell ``n`` and nowhere else,
        so ``cell_id`` is knowable in advance. An assignment that drifted by
        one would leave every count right and every attribution wrong.
        """
        for name in ("nucleus", "pathogen"):
            frame = tables[name]
            assert np.array_equal(frame["cell_id"].to_numpy(dtype=float),
                                  frame["object_label"].to_numpy(dtype=float)
                                  ), name
        organelles = tables["organelle"]
        per_cell = organelles.groupby(
            ["rowID", "columnID", "cell_id"]).size()
        assert set(per_cell) == {2}
        assert len(per_cell) == 36
        # ...and organelle labels are 2*(cell-1)+1 and +2, as written.
        for label in sorted(GEOMETRY):
            theirs = organelles[organelles["cell_id"] == label]
            assert set(theirs["object_label"]) == {2 * (label - 1) + 1,
                                                   2 * (label - 1) + 2}

    def test_the_innermost_radial_shell_is_cytoplasm_and_not_background(
            self, tables):
        """(DERIVED) ``rad_dist_..._bin_0`` == 1000 (ch0) / 2000 (ch1).

        Bin 0 is the shell at distance zero from the nucleus's outer
        boundary - the ring of pixels immediately outside it, which the
        fixture paints entirely cytoplasm. The regression this guards is
        named in ``measure._calculate_radial_distribution``: multiplying
        the distance map by the cell mask set every pixel *outside* the
        cell to distance 0 and dumped the field background into bin 0, so
        the feature reported ~100 (``BG0``) instead of 1000 and its meaning
        was inverted. Both numbers are painted constants, so either
        behaviour is unmistakable.
        """
        nuclei = tables["nucleus"]
        assert set(nuclei["nucleus_rad_dist_channel_0_bin_0"]) == {
            float(CYTO0)}
        assert set(nuclei["nucleus_rad_dist_channel_1_bin_0"]) == {
            float(CYTO1)}
        # The periphery of the nucleus is still nucleus, at its own level.
        assert set(nuclei["nucleus_channel_0_periphery_mean"]) == {float(NUC0)}
        assert set(nuclei["nucleus_channel_1_periphery_mean"]) == {float(NUC1)}

    def test_no_measurement_column_is_entirely_missing(self, tables):
        """A NaN column is a measurement that silently did not happen.

        Not a golden - a floor. Every numeric column of every object table
        has to hold a number for every object, because every object here is
        a well-formed solid square with a positive area in both channels.
        """
        for name in ("cell", "nucleus", "pathogen", "cytoplasm", "organelle",
                     "cell_organelle_summary"):
            frame = tables[name].select_dtypes(include=[np.number])
            empty = [column for column in frame.columns
                     if frame[column].isna().any()]
            assert not empty, f"{name} has NaN in {empty}"


class TestStage2OrganelleSummaries:
    """The per-parent roll-up, which is where a whole table went missing."""

    @pytest.mark.parametrize("label", sorted(GEOMETRY))
    def test_each_cell_summarises_its_two_organelles(self, tables, label):
        """(DERIVED) count 2, total area 8, mean area 4, std 0.

        Two 2x2 squares per cell: ``2 * 4 == 8`` pixels, identical, so the
        standard deviation of their areas is exactly zero.
        """
        rows = tables["cell_organelle_summary"]
        rows = rows[rows["object_label"] == label]
        assert len(rows) == WELLS_WITH_LABEL[label]
        assert set(rows["organelle_summary_organelle_count"]) == {2}
        assert set(rows["organelle_summary_organelle_total_area"]) == {8.0}
        assert set(rows["organelle_summary_organelle_mean_area"]) == {4.0}
        assert set(rows["organelle_summary_organelle_std_area"]) == {0.0}

    @pytest.mark.parametrize("label", sorted(GEOMETRY))
    def test_the_area_fraction_is_the_organelle_area_over_the_cell_area(
            self, tables, label):
        """(DERIVED) 8/144, 8/100, 8/81, 8/64.

        A fraction is where a unit error hides: the same number divided by
        the bounding box, by the convex hull or by the cytoplasm would all
        look plausible. Here the four cells have four different areas, so
        only the right denominator produces all four values.
        """
        rows = tables["cell_organelle_summary"]
        rows = rows[rows["object_label"] == label]
        expected = ORGANELLE_TOTAL[label] / CELL_AREA[label]
        assert rows["organelle_summary_organelle_fraction"].to_numpy() == (
            pytest.approx(expected, rel=1e-12))

    def test_a_parent_that_no_organelle_touches_summarises_to_zero(self,
                                                                   tables):
        """(DERIVED) Zero, not NaN and not the cell's numbers.

        No organelle overlaps any nucleus or pathogen in this fixture, so
        both roll-ups are a row of zeros for all 36 objects. An empty
        result that came back as NaN, or as the cell summary, would be
        indistinguishable downstream from "no organelles were found".
        """
        for parent in ("nucleus", "pathogen"):
            rows = tables[f"{parent}_organelle_summary"]
            assert len(rows) == 36
            assert set(rows["organelle_summary_organelle_count"]) == {0}
            assert set(rows["organelle_summary_organelle_total_area"]) == {0}
            assert set(rows["organelle_summary_organelle_fraction"]) == {0.0}


# ===========================================================================
# Stage 3 - classify
# ===========================================================================

def _probe_model(path):
    """A one-layer linear probe whose weights are set, not trained.

    Every weight is ``PROBE_WEIGHT / PNG_PIXELS``, so the single logit is
    ``PROBE_WEIGHT * mean(x) + PROBE_BIAS`` over the normalised image - a
    quantity this file can predict from the crop's grey levels. Nothing is
    trained, so there is no seed to get wrong and no epoch count to drift.
    Saved as a whole module, which
    :func:`spacr.torch_artifacts.load_model_artifact` accepts as its legacy
    format.
    """
    import torch
    from torch import nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(PNG_PIXELS, 1))
    with torch.no_grad():
        model[1].weight.fill_(PROBE_WEIGHT / PNG_PIXELS)
        model[1].bias.fill_(PROBE_BIAS)
    model.eval()
    torch.save(model, path)
    return path


def _expected_score(grey_sum):
    """(DERIVED) ``sigmoid(W * mean(2x/255 - 1) + B)`` for a crop.

    ``ToTensor`` divides a uint8 PNG by 255 and ``Normalize(0.5, 0.5)``
    maps that to ``2x/255 - 1``; the probe averages the result and the head
    is a sigmoid. ``grey_sum`` is the sum of all 3 * 24 * 24 grey levels.
    """
    mean = 2.0 * (grey_sum / PNG_PIXELS) / 255.0 - 1.0
    return 1.0 / (1.0 + math.exp(-(PROBE_WEIGHT * mean + PROBE_BIAS)))


def _crop_grey_sum(label, well_index):
    """(DERIVED) Sum of the grey levels of one crop, from its histogram.

    Channel 0 holds the nucleus at :func:`nucleus_grey_level` and the
    pathogen at 255; channel 1 holds the cytoplasm-level pixels at 255 and
    the pathogen at 63; channel 2 is zero-padding. Everything else is 0.
    """
    channel0 = (nucleus_grey_level(well_index) * NUCLEUS_AREA[label]
                + 255 * PATHOGEN_AREA[label])
    channel1 = (255 * CYTO_LEVEL_PIXELS[label]
                + PATHOGEN_GREY_CH1 * PATHOGEN_AREA[label])
    return channel0 + channel1


@pytest.fixture(scope="module")
def scores(project, tables):
    """Score every crop with the probe, then merge the scores into the DB.

    Inference is pinned to the CPU with a single BLAS thread. The golden
    scores are meant to be the *probe's* arithmetic and nothing else, and
    ``apply_model`` picks its device from ``torch.cuda.is_available()``, so
    without the pin the same assertion would be checking CUDA's summation
    on a workstation and the CPU's on CI - which differ by more than
    :data:`SCORE_TOLERANCE`'s margin (4.1e-6 vs 1.6e-7 measured). Pinning
    also makes "no GPU required" true rather than merely usually true.
    """
    import torch

    from spacr.deep_spacr import apply_model, merge_predictions_into_db

    model_path = _probe_model(os.path.join(project["root"], "probe.pth"))
    folders = sorted({os.path.dirname(path)
                      for path in tables["png_list"]["png_path"]})
    threads = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        torch.set_num_threads(1)
        try:
            frames = [apply_model(src=folder, model_path=model_path,
                                  image_size=PNG_SIDE, batch_size=8,
                                  normalize=True, n_jobs=0)
                      for folder in folders]
        finally:
            torch.set_num_threads(threads)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["cv_predictions"] = (predictions["pred"] >= 0.5).astype(int)
    matched = merge_predictions_into_db(predictions, project["db"])
    return {"folders": folders, "predictions": predictions,
            "matched": matched, "model_path": model_path}


def _crop_key(path):
    """``plate1_A03_1_2.png -> (3, 2)`` - (well index, object label)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    _, well, _field, label = stem.split("_")
    return WELLS.index(well) + 1, int(label)


class TestStage3Classify:

    def test_one_crop_per_object_in_one_folder_per_well(self, project,
                                                        tables, scores):
        """(DERIVED) 12 folders, 36 crops, keyed by well and label."""
        assert len(scores["folders"]) == 12
        assert len(scores["predictions"]) == 36
        keys = {_crop_key(path) for path in scores["predictions"]["path"]}
        assert keys == {(index, label)
                        for index, count in enumerate(CELLS_PER_WELL, start=1)
                        for label in range(1, count + 1)}

    def test_the_crops_carry_the_object_areas_as_pixel_counts(self, tables,
                                                              scores):
        """(DERIVED) The crop's histogram *is* the segmentation, in 8 bits.

        WHICH SLOT HOLDS WHICH SOURCE CHANNEL. ``png_dims=[0, 1]`` is the
        legacy spelling, and :func:`spacr.crops.png_dims_to_channel_mapping`
        translates it to ``{'r': None, 'g': 1, 'b': 0}``: entry 0 is BLUE,
        entry 1 is green, red is the empty plane. That is not an accident of
        cv2 -- microscope channels arrive in wavelength order, so channel 0
        is the 405 nuclear stain and belongs in blue, and every crop spaCR
        wrote before 2026-07-26 is in exactly this layout.

        This test used to read source channel 0 out of the RED slot. That is
        format 2, which `spacr/crops.py` documents outright as "the format
        that is wrong": it was written between 2026-07-26 and 2026-08-06,
        this file landed on 2026-08-03 inside that window, and when 2cab81f7
        restored the declared mapping the expectation was left behind. The
        pixels were never wrong; the assertion was.

        So: source channel 0 (nucleus grey and pathogen 255) in BLUE, source
        channel 1 (cytoplasm 255, pathogen ``PATHOGEN_GREY_CH1``) in GREEN,
        and RED the zero plane -- because two channels were asked for and
        the mapping leaves red empty rather than padding at the end.
        """
        from PIL import Image

        for path in scores["predictions"]["path"]:
            well_index, label = _crop_key(path)
            image = np.asarray(Image.open(path).convert("RGB"))
            assert image.shape == (PNG_SIDE, PNG_SIDE, 3), path
            blue = dict(zip(*[part.tolist() for part in np.unique(
                image[:, :, 2], return_counts=True)]))
            green = dict(zip(*[part.tolist() for part in np.unique(
                image[:, :, 1], return_counts=True)]))
            grey = nucleus_grey_level(well_index)
            assert blue[grey] == NUCLEUS_AREA[label], path
            assert blue[255] == PATHOGEN_AREA[label], path
            assert green[255] == CYTO_LEVEL_PIXELS[label], path
            assert green[PATHOGEN_GREY_CH1] == PATHOGEN_AREA[label], path
            # png_dims named two channels, so the mapping leaves red empty.
            assert set(np.unique(image[:, :, 0]).tolist()) == {0}, path

    @pytest.mark.parametrize("well_index", range(1, 13))
    def test_the_crop_grey_levels_are_the_stretch_of_the_painted_constants(
            self, scores, well_index):
        """(DERIVED) 113, 102, 93, 85, 78, 73, 68, 63, 60, 56, 53, 51.

        One value per well, and every one of them is
        ``int((3000-1000)/(pathogen-1000) * 65535) >> 8`` for that well's
        pathogen level - the percentile stretch across the uint16 range
        (truncating) followed by the high-byte narrowing. They fall
        monotonically because the stretch's upper end rises with the
        pathogen level, which is the invariant beside the twelve values.

        Read out of BLUE, for the reason spelled out in
        ``test_the_crops_carry_the_object_areas_as_pixel_counts``: source
        channel 0 is the 405 stain and the declared mapping puts it there.
        """
        from PIL import Image

        expected = [113, 102, 93, 85, 78, 73, 68, 63, 60, 56, 53, 51]
        assert nucleus_grey_level(well_index) == expected[well_index - 1]
        assert PATHOGEN_GREY_CH1 == 63
        assert expected == sorted(expected, reverse=True)

        paths = [path for path in scores["predictions"]["path"]
                 if _crop_key(path)[0] == well_index]
        for path in paths:
            image = np.asarray(Image.open(path).convert("RGB"))
            assert sorted(np.unique(image[:, :, 2]).tolist()) == [
                0, expected[well_index - 1], 255], path

    def test_the_reference_crop_scores_are_the_sigmoid_of_the_probe(self,
                                                                    scores):
        """(DERIVED) 0.6420218363106915 and 0.6063530554404933.

        Well A01's two crops, spelled out. Cell 1's channel-0 pixels sum to
        ``113*36 + 255*9 = 6363`` and its channel-1 pixels to
        ``255*99 + 63*9 = 25812``; over 3*24*24 = 1728 pixels that is a
        mean of 32175/1728, which normalises to ``2m/255 - 1`` and goes
        through ``4x + 4`` into a sigmoid.

        See :data:`SCORE_TOLERANCE` for why this comparison is 1e-5 and
        not 1e-12 like the rest of the file.
        """
        assert _crop_grey_sum(1, 1) == 6363 + 25812 == 32175
        assert _crop_grey_sum(2, 1) == (113 * 16 + 255 * 9) + (255 * 75
                                                               + 63 * 9)
        by_key = {_crop_key(row.path): row.pred
                  for row in scores["predictions"].itertuples()}
        assert by_key[(1, 1)] == pytest.approx(0.6420218363106915,
                                               rel=SCORE_TOLERANCE)
        assert by_key[(1, 2)] == pytest.approx(0.6063530554404933,
                                               rel=SCORE_TOLERANCE)
        # ...and both agree with the closed form this file computes.
        assert by_key[(1, 1)] == pytest.approx(
            _expected_score(_crop_grey_sum(1, 1)), rel=SCORE_TOLERANCE)
        assert by_key[(1, 2)] == pytest.approx(
            _expected_score(_crop_grey_sum(2, 1)), rel=SCORE_TOLERANCE)

    def test_every_crop_scores_what_its_grey_levels_predict(self, scores):
        """(DERIVED) The same closed form, for all 36 crops.

        The reference pair above pins the arithmetic; this pins that the
        arithmetic was applied to the right image, 36 times, with no crop
        picking up its neighbour's pixels.
        """
        for row in scores["predictions"].itertuples():
            label_key = _crop_key(row.path)
            expected = _expected_score(_crop_grey_sum(label_key[1],
                                                      label_key[0]))
            assert row.pred == pytest.approx(
                expected, rel=SCORE_TOLERANCE), row.path

    def test_scores_fall_with_the_pathogen_level_within_every_cell_size(
            self, scores):
        """MONOTONICITY. A brighter pathogen darkens the rest of the crop.

        The stretch is min-max over the crop, so raising the pathogen level
        (well 1 -> well 12) pushes every other grey level down, which
        lowers the mean and therefore the score. The ordering has to hold
        across all twelve wells for each of the four cell sizes separately;
        a per-crop normalisation that had become per-field would flatten it.
        """
        by_key = {_crop_key(row.path): row.pred
                  for row in scores["predictions"].itertuples()}
        for label in sorted(GEOMETRY):
            series = [by_key[(index, label)]
                      for index, count in enumerate(CELLS_PER_WELL, start=1)
                      if label <= count]
            assert series == sorted(series, reverse=True), label

    def test_every_score_is_written_back_onto_its_own_crop_row(self, project,
                                                               scores):
        """(DERIVED) 36 of 36 rows matched, and each holds its own score.

        The merge keys on ``prcfo``, so this is also the check that the
        crop file name and the measurement row agree about which object
        they are: 36 matches with the values transposed would still be 36.
        """
        assert scores["matched"] == 36
        connection = sqlite3.connect(project["db"])
        try:
            merged = pd.read_sql(
                'SELECT prcfo, pred, cv_predictions FROM png_list', connection)
        finally:
            connection.close()
        assert len(merged) == 36
        assert merged["pred"].notna().all()
        for row in merged.itertuples():
            plate, rowid, columnid, _field, objectid = row.prcfo.split("_")
            well_index = WELLS.index(
                "AB"[int(rowid[1:]) - 1] + f"{int(columnid[1:]):02d}") + 1
            label = int(objectid[1:])
            assert row.pred == pytest.approx(
                _expected_score(_crop_grey_sum(label, well_index)),
                rel=SCORE_TOLERANCE), row.prcfo
        # Every crop here is above the 0.5 cut, so the class column is 1s.
        assert set(merged["cv_predictions"]) == {1}


# ===========================================================================
# Stage 4 - regression
# ===========================================================================

@pytest.fixture(scope="module")
def wells(tables):
    """Per-well aggregation through :func:`spacr.ml.process_scores`.

    The response is the pathogen's channel-0 mean intensity, which is one
    painted constant per well, so its per-well mean is that constant and
    the aggregation cannot hide behind a plausible-looking average.
    """
    from spacr.ml import process_scores

    frame = tables["pathogen"][
        ["plateID", "rowID", "columnID", "fieldID",
         "pathogen_channel_0_mean_intensity"]]
    aggregated, response = process_scores(
        frame, "pathogen_channel_0_mean_intensity", plate=None,
        min_cell_count=1, agg_type="mean")
    order = {well_prc(well): index for index, well in enumerate(WELLS, start=1)}
    aggregated = aggregated.copy()
    aggregated["well_index"] = aggregated["prc"].map(order)
    aggregated = aggregated.sort_values("well_index").reset_index(drop=True)
    return {"frame": aggregated, "response": response}


class TestStage4Aggregation:

    def test_each_well_appears_once_and_carries_its_own_object_count(self,
                                                                     wells,
                                                                     tables):
        """(DERIVED) 12 rows; counts 2, 3, 4 repeating.

        ``cell_count`` is the number of rows the well contributed - one per
        pathogen, and the fixture gives every cell exactly one pathogen, so
        it equals ``CELLS_PER_WELL``. This is the double-count detector:
        a merge that fanned out would leave the means unchanged and the
        counts multiplied, and the counts are what the Poisson exposure
        below is taken from.
        """
        frame = wells["frame"]
        assert len(frame) == 12
        assert frame["prc"].tolist() == [well_prc(w) for w in WELLS]
        assert frame["cell_count"].tolist() == CELLS_PER_WELL
        # ...and the cell table agrees, well for well.
        per_well = tables["cell"].groupby(["rowID", "columnID"]).size()
        assert [int(per_well[(f"r{'AB'.index(w[0]) + 1}", f"c{int(w[1:])}")])
                for w in WELLS] == CELLS_PER_WELL

    def test_the_aggregate_is_the_constant_that_was_painted(self, wells):
        """(DERIVED) 5500, 6000, ... 11000 - one per well.

        Every pathogen in well ``i`` is painted at ``5000 + 500*i``, so the
        mean over the well is that value however many objects it holds. A
        weighting, a re-normalisation or a well mix-up all move it.
        """
        frame = wells["frame"]
        assert wells["response"] == "pathogen_channel_0_mean_intensity"
        assert frame[wells["response"]].tolist() == [
            float(pathogen_level(index)) for index in range(1, 13)]


class TestStage4Regression:

    @pytest.fixture()
    def design(self, wells):
        """``y = 5000 + 8000 * x`` by construction, with ``x = i/16``.

        ``i/16`` is exact in binary and ``8000 * i/16 == 500 * i``, so the
        response the wells actually produced is an exact linear function of
        the covariate and the least-squares solution is analytic.
        """
        frame = wells["frame"]
        design = pd.DataFrame({
            "Intercept": 1.0,
            "fraction": frame["well_index"].to_numpy(dtype=float) / 16.0})
        return design, frame[wells["response"]]

    def test_ols_recovers_the_designed_effect_exactly(self, design):
        """(DERIVED) intercept 5000.0, slope 8000.0, R-squared 1.0.

        The design is exactly satisfiable, so the least-squares minimum is
        the zero-residual solution and there is only one. ``rel=1e-9``
        leaves room for the solver's linear algebra (measured: 4e-16) and
        none for a scaling, a centring or a dropped intercept.
        """
        from spacr.ml import regression_model

        matrix, response = design
        model = regression_model(matrix, response, regression_type="ols")

        assert model.params["Intercept"] == pytest.approx(5000.0, rel=1e-9)
        assert model.params["fraction"] == pytest.approx(8000.0, rel=1e-9)
        assert model.rsquared == pytest.approx(1.0, rel=1e-12)
        assert float(np.abs(model.resid).max()) < 1e-8

    def test_the_coefficient_table_reports_the_same_numbers(self, design):
        """(DERIVED for the coefficients) The table the volcano plot reads.

        ``process_model_coefficients`` is the one place a fitted model
        becomes the hit table, so the coefficients have to survive the
        trip: this asserts the same 5000 / 8000 come out of the table that
        came out of the fit.

        The p-values are NOT golden and are not asserted as such - a
        perfect fit drives them to ~1e-152, where the exact value is a
        property of the solver's precision rather than of the data. What
        is checked is that they are finite and that the column the volcano
        plot actually plots, ``-log10(p_value)``, really is the negative
        log of the column beside it.
        """
        from spacr.ml import process_model_coefficients, regression_model

        matrix, response = design
        model = regression_model(matrix, response, regression_type="ols")
        table = process_model_coefficients(model, "ols", matrix, response,
                                           "nc", "pc", [])

        assert table["feature"].tolist() == ["Intercept", "fraction"]
        assert table["coefficient"].tolist() == pytest.approx(
            [5000.0, 8000.0], rel=1e-9)
        assert np.isfinite(table["p_value"]).all()
        assert table["-log10(p_value)"].to_numpy() == pytest.approx(
            -np.log10(table["p_value"].to_numpy()), rel=1e-12)

    @pytest.fixture()
    def counts(self, wells):
        """A count response whose Poisson MLE is analytic.

        The exposure is the wells' real object counts from
        :func:`~spacr.ml.process_scores` (2, 3, 4 repeating). The counts
        are chosen integers, and the covariate is then solved for::

            z_i = (log(k_i / n_i) - b0) / b1

        so that ``n_i * exp(b0 + b1*z_i)`` equals ``k_i`` for every well.
        The Poisson score equations are ``sum((k - mu) * X) == 0``, which
        that makes exactly zero at ``(b0, b1)``; the log-likelihood is
        strictly concave on a full-rank design, so ``(b0, b1)`` is the
        unique maximum. The coefficients are therefore derived, not
        recorded - and this is a count model with a genuinely varying
        exposure, which is the shape the offset exists for.
        """
        exposure = wells["frame"]["cell_count"].to_numpy(dtype=float)
        assert exposure.tolist() == [float(n) for n in CELLS_PER_WELL]
        response = np.asarray(POISSON_COUNTS, dtype=float)
        covariate = (np.log(response / exposure) - POISSON_B0) / POISSON_B1
        matrix = pd.DataFrame({"Intercept": 1.0, "z": covariate})
        return matrix, pd.Series(response), exposure

    def test_poisson_recovers_the_rate_coefficients_through_the_offset(
            self, counts):
        """(DERIVED) intercept -0.75, slope 1.5, deviance 0.

        The offset ``log(exposure)`` is what turns a per-well headcount
        into a per-well rate. With it, the fit reproduces every count
        exactly - deviance 0 to solver precision - and lands on the two
        coefficients the frame was built from. Measured: -0.7500000000000004
        and 1.5000000000000009, so ``rel=1e-9`` is not a loose tolerance.
        """
        from spacr.ml import regression_model

        matrix, response, exposure = counts
        model = regression_model(matrix, response, regression_type="poisson",
                                 exposure=exposure)

        assert model.params["Intercept"] == pytest.approx(POISSON_B0,
                                                          rel=1e-9)
        assert model.params["z"] == pytest.approx(POISSON_B1, rel=1e-9)
        assert model.deviance < 1e-9
        assert np.asarray(model.fittedvalues) == pytest.approx(
            response.to_numpy(), rel=1e-9)

    def test_dropping_the_exposure_moves_the_answer(self, counts):
        """THE OFFSET IS LOAD-BEARING. (RECORDED, with invariants.)

        The same frame fitted with no exposure gives intercept 0.11646443
        and slope 1.80330179 - recorded from a run, because a mis-specified
        model has no closed form. What makes them meaningful is what sits
        beside them: the slope is 20% away from the 1.5 the data were built
        with, and the deviance rises from 0 to 3.57, i.e. the offset-free
        model cannot reproduce the counts at all. If a change ever makes
        this fit agree with the one above, the offset has stopped being
        applied and the test above would keep passing.
        """
        from spacr.ml import regression_model

        matrix, response, _exposure = counts
        model = regression_model(matrix, response, regression_type="poisson",
                                 exposure=None)

        assert model.params["Intercept"] == pytest.approx(0.11646443180997579,
                                                          rel=1e-6)
        assert model.params["z"] == pytest.approx(1.8033017870986545,
                                                  rel=1e-6)
        assert abs(model.params["z"] - POISSON_B1) > 0.25
        assert model.deviance == pytest.approx(3.5683, rel=1e-3)
        assert model.deviance > 1.0


# ===========================================================================
# The bug this module found
# ===========================================================================

@pytest.fixture(scope="module")
def stock_import(tmp_path_factory):
    """One well, imported with the importer's own defaults.

    Nothing about ``summarize_organelles_by`` is said here - which is the
    whole point: this is what a user gets.
    """
    root = tmp_path_factory.mktemp("golden_stock")
    result, _truth, _plan = _import(
        str(root), os.path.join(str(root), "project"),
        wells=["A01"], cells_per_well=[4],
        extra={"save_png": False})
    return result


class TestTheStockImporterLosesTheOrganelleSummary:
    """``list('cell')`` is ``['c', 'e', 'l', 'l']``.

    :func:`spacr.external_masks.run_external_masks`, at
    ``spacr/external_masks.py:661-665``, does::

        summaries = list(measure_settings.get("summarize_organelles_by") or [])
        if "organelle" not in summaries:
            summaries.append("organelle")
        measure_settings["summarize_organelles_by"] = summaries

    and the value it reads is the default from
    :func:`spacr.settings.get_measure_crop_settings`, which is the **string**
    ``'cell'`` (``spacr/settings.py:497``). ``list()`` of a string is its
    characters, so what reaches Measure is
    ``['c', 'e', 'l', 'l', 'organelle']``.

    ``spacr/measure.py:2396`` then asks ``if "cell" in
    settings['summarize_organelles_by']``. On the original string that is a
    substring test and True; on the character list it is a membership test
    and False. So ``cell_organelle_summary`` - per-cell organelle count,
    total area, area fraction, per-channel intensity - is never written for
    an external-mask import, while the raw ``organelle`` table is (because
    ``"organelle"`` really was appended as an element). Nothing warns:
    ``run_external_masks``'s own completeness check only requires a table
    per imported object type, and ``organelle`` is there.

    The same ``list()`` also silently discards any *other* parent the user
    chose, since ``'nucleus'`` and ``'pathogen'`` become characters too.

    Measured on this fixture: tables are ``cell, cytoplasm, nucleus,
    organelle, pathogen`` and no ``*_organelle_summary`` at all. The
    control below shows the same import produces the table as soon as the
    setting arrives as a list, so the geometry is not the reason.

    The fix belongs in ``spacr/external_masks.py``: normalise the setting
    the way Measure reads it (a str is one name, not four characters)
    before appending to it - and, better, stop
    ``get_measure_crop_settings`` handing out a bare string for a setting
    every reader treats as a collection.
    """

    @pytest.mark.xfail(strict=True, reason=(
        "spacr/external_masks.py:661 does list() on summarize_organelles_by, "
        "whose default (spacr/settings.py:497) is the string 'cell', so "
        "Measure receives ['c','e','l','l','organelle'] and the "
        "`if \"cell\" in settings['summarize_organelles_by']` test at "
        "spacr/measure.py:2396 is False. An external-mask import with "
        "organelle masks therefore writes no cell_organelle_summary table, "
        "and run_external_masks' completeness check does not notice because "
        "it only requires one table per imported object type."))
    def test_a_stock_import_still_summarises_organelles_per_cell(self,
                                                                  stock_import):
        assert "cell_organelle_summary" in stock_import.tables, (
            f"tables written: {stock_import.tables}")

    def test_the_raw_organelle_table_is_written_either_way(self,
                                                            stock_import):
        """The half that survives, which is why the loss is quiet.

        ``'organelle'`` is appended as a whole element, so that membership
        test passes and the per-organelle table appears. A reader who
        checks that organelles were measured at all sees a table.
        """
        assert "organelle" in stock_import.tables
        connection = sqlite3.connect(stock_import.db_path)
        try:
            organelles = pd.read_sql('SELECT * FROM organelle', connection)
        finally:
            connection.close()
        assert len(organelles) == 8               # 4 cells x 2 organelles

    def test_the_same_import_writes_it_when_the_setting_is_a_list(self,
                                                                   project):
        """The control. Same code path, same geometry, setting as a list.

        ``project`` passes ``summarize_organelles_by`` as a list and gets
        all three roll-ups, so nothing about this fixture's organelles
        prevents the summary - only the ``list()`` of a string does.
        """
        assert "cell_organelle_summary" in project["result"].tables
        assert "nucleus_organelle_summary" in project["result"].tables
        assert "pathogen_organelle_summary" in project["result"].tables


def test_the_setting_that_causes_it_still_has_the_shape_described():
    """Pins the two halves of the bug so the xfail cannot rot.

    If either the default stops being a bare string or Measure stops using
    a containment test, the analysis above is out of date and the marker
    needs re-reading rather than silently flipping.
    """
    import inspect
    import re

    from spacr.settings import get_measure_crop_settings
    import spacr.external_masks as external_masks
    import spacr.measure as measure

    # Half one: the setting arrives as a bare string, and list()ing it
    # destroys the only name in it.
    default = get_measure_crop_settings({})["summarize_organelles_by"]
    assert isinstance(default, str) and default == "cell"
    assert "cell" in default                      # substring test: True
    assert "cell" not in list(default)            # membership test: False

    # Half two: the importer really does list() it...
    importer = inspect.getsource(external_masks.run_external_masks)
    assert re.search(
        r'list\(\s*measure_settings\.get\("summarize_organelles_by"\)',
        importer), "external_masks no longer list()s the setting"

    # ...and Measure really does read it with `in`, which is what makes
    # the difference between the two spellings observable.
    measured = inspect.getsource(measure._measure_crop_core)
    assert '"cell" in settings[\'summarize_organelles_by\']' in measured
