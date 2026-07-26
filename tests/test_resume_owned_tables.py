"""A measure resume must delete only the tables the measure stage wrote.

``measurements.db`` is not measure's private file. Three other modules
write into the same database, keyed on exactly the same four columns so
that they *join* to the measurements:

* ``convert.populate_db_from_map`` writes ``conversion_map`` — the only
  record anywhere of which vendor file became ``plate1_A01_1``;
* ``align.save_coordinates`` writes ``align_coordinates`` — where each
  tile was actually stitched, and by which method;
* ``foreign._write_tables`` writes ``foreign_<object>`` — somebody
  else's measurements, imported and matched to spaCR's objects.

``discover_field_tables`` enumerated tables by *structure* ("has all four
key columns") minus a deny-list of four names, so all three qualified as
"per-field measure output" and ``clear_field_rows`` deleted from them.
None of it can be recomputed from the database: the conversion map lives
in a CSV beside the converted images, the stitch coordinates only exist
if align is re-run, and the foreign import needs the original file.

Everything here is built with the real writers — ``convert.convert_folder``
, ``align.save_coordinates``, ``utils._merge_and_save_to_database`` and
``utils.filepaths_to_database`` — because a hand-built schema that happens
to match is precisely what let this survive: the schema *is* the bug.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

import spacr.align as align
import spacr.convert as cv
from spacr.resume import (
    FIELD_KEY_COLUMNS, MEASURE_OWNED_TABLES, NON_FIELD_TABLES,
    REASON_NOT_MEASURED, REASON_PARTIAL_DB, clear_field_rows,
    completed_fields_in_db, discover_field_tables, plan_measure_resume,
)
from spacr.utils import _merge_and_save_to_database, filepaths_to_database


# ---------------------------------------------------------------------------
# Builders — every one of them a real spaCR writer
# ---------------------------------------------------------------------------

N_FIELDS = 4
N_OBJECTS = 3


def _write_merged(root, n_fields=N_FIELDS, planes=2):
    """``merged/<stem>.npy`` exactly as ``io._load_and_concatenate_arrays``."""
    folder = os.path.join(root, 'merged')
    os.makedirs(folder, exist_ok=True)
    for i in range(1, n_fields + 1):
        np.save(os.path.join(folder, f'plate1_A01_{i}.npy'),
                np.zeros((16, 16, planes), np.uint16))
    return folder


def _run_convert(tmp_path, db_path, n_fields=N_FIELDS):
    """Convert a vendor folder for real, and populate ``conversion_map``.

    ``run1/wt/fov01_C1.tif`` is the layout ``convert.scan`` was written
    for; it yields ``plate1`` / ``A01`` / ``F001..`` — the same stems the
    merged folder carries, which is the whole point of the table.
    """
    vendor = os.path.join(str(tmp_path), 'vendor')
    for field in range(1, n_fields + 1):
        for channel in (1, 2):
            path = os.path.join(vendor, 'run1', 'wt',
                                f'fov{field:02d}_C{channel}.tif')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tifffile.imwrite(path, np.full((8, 8), field * 10 + channel,
                                           np.uint16))
    return cv.convert_folder({'src': vendor,
                              'dst': os.path.join(str(tmp_path), 'yokogawa'),
                              'db_path': db_path,
                              'preview_rows': 0})


def _run_align(tmp_path, db_path, n_fields=N_FIELDS):
    """Stitch a real 2x2 grid and write ``align_coordinates`` for real."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(7)
    raw = gaussian_filter(rng.random((300, 300)).astype(np.float32), 2.0)
    span = float(raw.max() - raw.min())
    big = ((raw - raw.min()) / max(span, 1e-9) * 30000 + 1000).astype(np.uint16)

    folder = os.path.join(str(tmp_path), 'tiles')
    os.makedirs(folder, exist_ok=True)
    index = 0
    for row in range(2):
        for column in range(2):
            y, x = row * 100, column * 100
            np.save(os.path.join(folder, f'plate1_A01_{index + 1:03d}.npy'),
                    big[y:y + 128, x:x + 128])
            index += 1
    tiles = align.scan_tiles(folder, grid=(2, 2), overlap=1 - 100 / 128)
    plan = align.estimate_offsets(tiles)
    return align.save_coordinates(plan, db_path)


def _measure_field(root, stem, tables=('cell', 'nucleus', 'png_list'),
                   n_objects=N_OBJECTS):
    """Write one field's rows through the writers measure actually calls."""
    labels = list(range(1, n_objects + 1))
    for table in tables:
        if table == 'png_list':
            continue
        morphology = pd.DataFrame({
            'label': labels,
            f'{table}_area': [100.0 + i for i in range(n_objects)]})
        intensity = pd.DataFrame({
            'label': labels,
            f'{table}_channel_0_mean_intensity': [5.0] * n_objects})
        _merge_and_save_to_database(morphology, intensity, table,
                                    root, stem, 'exp', False)
    if 'png_list' in tables:
        paths = [os.path.join(root, 'data', 'cell_png', f'{stem}_{i + 1}.png')
                 for i in range(n_objects)]
        filepaths_to_database(paths, {'timelapse': False}, root, 'cell')


def _counts(db_path):
    """``{table: row count}`` for every table in the database."""
    conn = sqlite3.connect(db_path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                for n in names}
    finally:
        conn.close()


def _fields_in(db_path, table):
    """The distinct ``fieldID`` values a table still carries."""
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute(
            f'SELECT DISTINCT fieldID FROM "{table}"')}
    finally:
        conn.close()


@pytest.fixture()
def project(tmp_path):
    """A whole project database, written the way a real project writes it.

    Four fields in ``merged/``. ``conversion_map`` and ``align_coordinates``
    cover all four — they are written before measure ever runs. Measure
    then completes fields 1 and 2, and is killed part-way through field 3
    (cell rows written, nucleus and png_list not), which is the exact
    state a resume exists to repair. Field 4 was never measured.
    """
    root = os.path.join(str(tmp_path), 'exp')
    merged = _write_merged(root)
    db_path = os.path.join(root, 'measurements', 'measurements.db')

    _run_convert(tmp_path, db_path)
    _run_align(tmp_path, db_path)

    _measure_field(root, 'plate1_A01_1')
    _measure_field(root, 'plate1_A01_2')
    _measure_field(root, 'plate1_A01_3', tables=('cell',))   # crash here

    settings = {'src': merged, 'resume': True, 'timelapse': False,
                'channels': [0, 1], 'cell_mask_dim': 1}
    return {'root': root, 'merged': merged, 'db': db_path,
            'settings': settings}


# ---------------------------------------------------------------------------
# The starting state, asserted — so the numbers after the resume mean something
# ---------------------------------------------------------------------------

def test_the_project_database_holds_all_four_modules_output(project):
    """convert, align and measure all wrote into one measurements.db."""
    counts = _counts(project['db'])
    assert counts['conversion_map'] == 8      # 4 fields x 2 channels
    assert counts['align_coordinates'] == 4   # 4 tiles
    assert counts['cell'] == 9                # fields 1, 2, 3 x 3 objects
    assert counts['nucleus'] == 6             # fields 1, 2 only
    assert counts['png_list'] == 6

    # And every one of them is keyed on the four columns, which is exactly
    # why a structural "has the key columns" test cannot tell them apart.
    conn = sqlite3.connect(project['db'])
    try:
        for table in ('conversion_map', 'align_coordinates', 'cell'):
            columns = {r[1] for r in conn.execute(
                f'PRAGMA table_info("{table}")')}
            assert set(FIELD_KEY_COLUMNS) <= columns, table
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

class TestResumeDoesNotDeleteOtherModulesTables:

    def test_conversion_map_and_align_coordinates_survive_a_resume(self,
                                                                   project):
        """THE BUG. A resume used to delete every pending field's row from
        ``conversion_map`` and ``align_coordinates``.

        Measured before the fix: ``conversion_map`` 8 -> 4 rows and
        ``align_coordinates`` 4 -> 2, because fields 3 and 4 are pending
        and both tables carry a row for them. The map for those two fields
        — which vendor file they came from — was gone, and the only other
        copy is a CSV next to the converted images.
        """
        before = _counts(project['db'])

        plan_measure_resume(project['settings'], verbose=False)

        after = _counts(project['db'])
        assert after['conversion_map'] == before['conversion_map'] == 8
        assert after['align_coordinates'] == before['align_coordinates'] == 4
        assert _fields_in(project['db'], 'conversion_map') == {
            'f1', 'f2', 'f3', 'f4'}
        assert _fields_in(project['db'], 'align_coordinates') == {
            'f1', 'f2', 'f3', 'f4'}

    def test_the_partial_measure_rows_it_is_meant_to_clear_are_gone(self,
                                                                   project):
        """The other half: the resume must still do its actual job.

        Field 3 crashed after its cell rows and before its nucleus rows.
        Those 3 cell rows have to go, or re-measuring the field appends a
        second copy of every object.
        """
        assert _counts(project['db'])['cell'] == 9
        assert 'f3' in _fields_in(project['db'], 'cell')

        state = plan_measure_resume(project['settings'], verbose=False)

        assert _counts(project['db'])['cell'] == 6
        assert _fields_in(project['db'], 'cell') == {'f1', 'f2'}
        # Only the three stale cell rows — not conversion_map's four and
        # align_coordinates' two, which the old count of 9 included.
        assert state.cleared_rows == 3

    def test_completed_fields_are_untouched(self, project):
        """Fields 1 and 2 finished; nothing about them may move."""
        before = _counts(project['db'])
        plan_measure_resume(project['settings'], verbose=False)
        after = _counts(project['db'])
        assert after['nucleus'] == before['nucleus'] == 6
        assert after['png_list'] == before['png_list'] == 6

    def test_an_unmeasured_field_is_not_reported_as_partial(self, project):
        """Field 4 has no measurement rows at all — it is simply pending.

        It used to come back as ``partial-db-rows``, a *rejection* reason,
        because ``conversion_map`` and ``align_coordinates`` had rows for
        it and the resume counted those as "measured in some tables".
        """
        state = plan_measure_resume(project['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_3', 'plate1_A01_4'}
        assert set(state.done) == {'plate1_A01_1', 'plate1_A01_2'}
        assert state.reasons.get('plate1_A01_3') == REASON_PARTIAL_DB
        assert state.reasons.get('plate1_A01_4') == REASON_NOT_MEASURED

    def test_resuming_twice_still_never_touches_the_other_tables(self, project):
        plan_measure_resume(project['settings'], verbose=False)
        plan_measure_resume(project['settings'], verbose=False)
        counts = _counts(project['db'])
        assert counts['conversion_map'] == 8
        assert counts['align_coordinates'] == 4
        assert counts['cell'] == 6

    def test_resume_then_remeasure_is_still_idempotent(self, project):
        """The deliverable the deletion exists for, end to end.

        Resume, measure what it says is pending, resume again, measure
        what it says is pending again. Nothing grows and nothing is lost —
        which is the property the delete-before-insert buys, and the one
        that must survive being restricted to measure's own tables.
        """
        for _ in range(2):
            state = plan_measure_resume(project['settings'], verbose=False)
            for stem in state.pending:
                _measure_field(project['root'], stem)
            counts = _counts(project['db'])
            assert counts['cell'] == 12          # 4 fields x 3 objects
            assert counts['nucleus'] == 12
            assert counts['png_list'] == 12
            assert counts['conversion_map'] == 8
            assert counts['align_coordinates'] == 4


# ---------------------------------------------------------------------------
# The allow-list itself
# ---------------------------------------------------------------------------

class TestMeasureOwnedTables:

    def test_discovery_excludes_other_modules_field_tables(self, project):
        """The list the resume deletes from is measure's output, nothing else."""
        assert set(discover_field_tables(project['db'])) == {
            'cell', 'nucleus', 'png_list'}

    def test_the_deny_list_alone_would_have_included_them(self, project):
        """Proof that the deny-list was the wrong shape, not merely stale.

        ``owned_only=False`` is the old structural rule. It still finds
        ``conversion_map`` and ``align_coordinates``, because they really
        do carry the four key columns — deliberately, so they join.
        """
        structural = set(discover_field_tables(project['db'],
                                               owned_only=False))
        assert {'conversion_map', 'align_coordinates'} <= structural
        assert structural & NON_FIELD_TABLES == set()

    def test_clear_field_rows_refuses_a_table_it_does_not_own(self, project):
        """Naming the table by hand does not get past the allow-list."""
        before = _counts(project['db'])
        with pytest.raises(ValueError, match='conversion_map'):
            clear_field_rows(project['db'], ['cell', 'conversion_map'],
                             'plate1_A01_3')
        assert _counts(project['db']) == before      # pre-flight, nothing ran

    def test_the_allow_list_matches_what_measure_actually_writes(self):
        """Keeps the literal honest.

        The set is spelled out in ``resume`` because that module must stay
        stdlib-only — it runs before torch is imported, and ``utils``
        pulls torch in. This test is the link that would otherwise be
        missing: add a table to measure and this fails until the
        allow-list learns about it, which is the opposite failure mode
        from the deny-list that let convert, align and foreign through.
        """
        from spacr.utils import (_CHILD_OBJECT_TABLES,
                                 _ORGANELLE_SUMMARY_TABLES,
                                 _PARENT_OBJECT_TABLES)
        expected = (set(_PARENT_OBJECT_TABLES) | set(_CHILD_OBJECT_TABLES)
                    | set(_ORGANELLE_SUMMARY_TABLES) | {'png_list'})
        assert MEASURE_OWNED_TABLES == expected

    def test_no_other_module_owns_a_name_on_the_allow_list(self):
        """The three tables that were being deleted are not measure's."""
        assert cv.CONVERSION_TABLE not in MEASURE_OWNED_TABLES
        assert align.ALIGN_TABLE not in MEASURE_OWNED_TABLES
        assert not any(t.startswith('foreign_') for t in MEASURE_OWNED_TABLES)

    def test_the_old_rule_really_did_destroy_them(self, project, monkeypatch):
        """THE BUG, asserted — so the allow-list is provably load-bearing.

        Widening :data:`spacr.resume.MEASURE_OWNED_TABLES` to every table
        in the database restores the old rule exactly: "has the four key
        columns and is not on the deny-list". Under it, resuming this
        project deletes fields 3 and 4 out of ``conversion_map``
        (8 -> 4 rows) and ``align_coordinates`` (4 -> 2), and reports 9
        cleared rows where only 3 of them were measure's.
        """
        import spacr.resume as resume
        conn = sqlite3.connect(project['db'])
        try:
            every = frozenset(r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"))
        finally:
            conn.close()
        monkeypatch.setattr(resume, 'MEASURE_OWNED_TABLES', every)

        state = resume.plan_measure_resume(project['settings'], verbose=False)

        after = _counts(project['db'])
        assert after['conversion_map'] == 4
        assert after['align_coordinates'] == 2
        assert _fields_in(project['db'], 'conversion_map') == {'f1', 'f2'}
        assert state.cleared_rows == 9
        # And the spurious rejection that came with it.
        assert state.reasons['plate1_A01_4'] == REASON_PARTIAL_DB

    def test_completed_fields_ignores_other_modules_tables(self, project):
        """The read side too: 'measured' means measured, not 'registered'."""
        partial = {}
        done = completed_fields_in_db(
            project['db'],
            fields=[f'plate1_A01_{i}' for i in range(1, 5)],
            partial=partial)
        assert done == {'plate1_A01_1', 'plate1_A01_2'}
        assert partial == {'plate1_A01_3': REASON_PARTIAL_DB}


# ---------------------------------------------------------------------------
# foreign_* — the same shape, from a third module
# ---------------------------------------------------------------------------

SIZE = 24


def _their_labels():
    mask = np.zeros((SIZE, SIZE), np.uint16)
    mask[2:8, 2:8] = 1
    mask[12:20, 12:20] = 2
    return mask


@pytest.fixture()
def imported(tmp_path):
    """A project produced by a real ``foreign.run_import``.

    Their images, their masks and their CSV go through
    :func:`spacr.foreign.plan_import` / :func:`~spacr.foreign.run_import`,
    which converts, merges, populates ``conversion_map`` and writes
    ``foreign_cell``. Then one field is measured with spaCR's own writer,
    which leaves the second field pending — the state a resume acts on.
    """
    import spacr.foreign as fg

    images = os.path.join(str(tmp_path), 'their_images')
    masks = os.path.join(str(tmp_path), 'their_cell_masks')
    os.makedirs(images)
    os.makedirs(masks)
    rows = []
    for field in (1, 2):
        for channel in (1, 2):
            tifffile.imwrite(
                os.path.join(images, f'fov{field:02d}_C{channel}.tif'),
                np.full((SIZE, SIZE), field * 10 + channel, np.uint16))
        tifffile.imwrite(
            os.path.join(masks, f'fov{field:02d}_cell_mask.tif'),
            _their_labels())
        for label, area in ((1, 36.0), (2, 64.0)):
            rows.append({'ImageNumber': f'fov{field:02d}_C1.tif',
                         'ObjectNumber': label, 'AreaShape_Area': area})
    csv_path = os.path.join(str(tmp_path), 'results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    plan = fg.plan_import(images, {'cell': masks}, csv_path, um_per_px=0.5)
    assert plan.ok
    dst = os.path.join(str(tmp_path), 'imported')
    fg.run_import(plan, dst)

    # spaCR measures field 1 and dies before field 2.
    _measure_field(dst, 'plate1_A01_1', tables=('nucleus',), n_objects=2)

    settings = {'src': os.path.join(dst, 'merged'), 'resume': True,
                'timelapse': False, 'channels': [0, 1], 'cell_mask_dim': 1}
    return {'root': dst, 'db': os.path.join(dst, 'measurements',
                                            'measurements.db'),
            'settings': settings}


def test_a_foreign_object_table_is_not_deleted(imported):
    """``foreign_cell`` and ``conversion_map`` survive; measure's do not.

    ``foreign._foreign_frame`` writes ``plateID`` / ``rowID`` /
    ``columnID`` / ``fieldID`` onto every row so the import joins to the
    measurements, which made it indistinguishable from measure output
    under the structural rule. Measured before the fix: ``foreign_cell``
    4 -> 2 rows and ``conversion_map`` 4 -> 2.
    """
    before = _counts(imported['db'])
    assert before['foreign_cell'] == 4
    assert before['conversion_map'] == 4

    state = plan_measure_resume(imported['settings'], verbose=False)
    assert set(state.pending) == {'plate1_A01_2'}

    after = _counts(imported['db'])
    assert after['foreign_cell'] == 4
    assert after['conversion_map'] == 4
    assert after['foreign_columns'] == before['foreign_columns']
    assert after['foreign_import'] == before['foreign_import']
    # ...while the object table measure owns is cleared of the pending field.
    assert _fields_in(imported['db'], 'cell') == {'f1'}
