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

The allow-list that replaced the deny-list answers "could measure have
written this table?", and there is one case where the honest answer to
"did it?" is still no: ``foreign.run_import`` copies the imported rows
into the *canonical* ``cell`` table when the destination is empty, so a
project built purely by import is readable by every spaCR tool. Those
rows sit under spaCR's own column names in a table whose name is on the
allow-list, and a resume clears them along with the pending field. That
is a real, open bug (F34), and ``TestACanonicalTableTheImporterFilled``
below records it in ``xfail(strict=True)`` tests rather than leaving it
undocumented: they assert the behaviour spaCR should have, they fail
today, and anything that fixes one of them fails the suite until the
marker is removed. A guard inside :func:`clear_field_rows` was written
for this and backed out — the ``spacr.resume`` module docstring lists
the five ways it was worse than the bug, all of them measured. What
bounds the damage is asserted here too: the canonical copy is
byte-identical to ``foreign_cell``, which no resume may touch.

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


def _their_project(tmp_path):
    """Their images, masks and CSV on disk; returns (images, masks, csv)."""
    images = os.path.join(str(tmp_path), 'their_images')
    masks = os.path.join(str(tmp_path), 'their_cell_masks')
    os.makedirs(images, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
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
    return images, masks, csv_path


def _resume_settings(dst):
    return {'src': os.path.join(dst, 'merged'), 'resume': True,
            'timelapse': False, 'channels': [0, 1], 'cell_mask_dim': 1}


@pytest.fixture()
def imported(tmp_path):
    """A project produced by a real ``foreign.run_import``.

    Their images, their masks and their CSV go through
    :func:`spacr.foreign.plan_import` / :func:`~spacr.foreign.run_import`,
    which converts, merges, populates ``conversion_map`` and writes
    ``foreign_cell`` — **and**, because the destination was empty, copies
    those same rows into the canonical ``cell`` table so a purely-imported
    project is readable by every spaCR tool. Then one field is measured
    with spaCR's own writer, which leaves the second field pending — the
    state a resume acts on.
    """
    import spacr.foreign as fg

    images, masks, csv_path = _their_project(tmp_path)
    plan = fg.plan_import(images, {'cell': masks}, csv_path, um_per_px=0.5)
    assert plan.ok
    dst = os.path.join(str(tmp_path), 'imported')
    fg.run_import(plan, dst)

    # spaCR measures field 1 and dies before field 2.
    _measure_field(dst, 'plate1_A01_1', tables=('nucleus',), n_objects=2)

    return {'root': dst, 'db': os.path.join(dst, 'measurements',
                                            'measurements.db'),
            'settings': _resume_settings(dst), 'their_csv': csv_path}


@pytest.fixture()
def imported_beside(tmp_path):
    """An import that landed *beside* spaCR's own measurements.

    The destination already holds spaCR measurements, so
    ``_may_write_canonical`` refuses the canonical copy: their rows go to
    ``foreign_cell`` only and ``foreign_import`` records
    ``canonical_table_written = 0``. ``cell`` is therefore measure's own
    table in fact as well as in name, and a resume must clear it exactly
    as it would in any other project — this is the control that keeps the
    refusal below from being a blanket ban on imported projects.

    Field 1 is measured in full; field 2 has only its ``cell`` rows, the
    crash-mid-field state the delete-before-insert exists for.
    """
    import spacr.foreign as fg

    images, masks, csv_path = _their_project(tmp_path)
    dst = os.path.join(str(tmp_path), 'beside')
    os.makedirs(os.path.join(dst, 'measurements'), exist_ok=True)
    _measure_field(dst, 'plate1_A01_1')
    _measure_field(dst, 'plate1_A01_2', tables=('cell',))

    plan = fg.plan_import(images, {'cell': masks}, csv_path, um_per_px=0.5)
    assert plan.ok
    result = fg.run_import(plan, dst)
    assert 'cell' not in result.rows, result.rows

    return {'root': dst, 'db': os.path.join(dst, 'measurements',
                                            'measurements.db'),
            'settings': _resume_settings(dst)}


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


def test_the_canonical_copy_is_a_duplicate_of_the_foreign_table(imported):
    """What bounds the damage of the bug below — assert it, do not assume it.

    ``run_import`` writes *one* frame twice: to ``foreign_cell``, which
    this module may never delete from, and to the canonical ``cell``,
    which it may. Same columns, same rows. So a resume that clears
    imported rows out of ``cell`` destroys a copy, not a measurement —
    which is the whole reason the bug below is recorded and lived with
    rather than guarded against at this layer.

    If this ever stops holding, the ``xfail`` below stops being an
    acceptable state of affairs and becomes data loss.
    """
    conn = sqlite3.connect(imported['db'])
    try:
        canonical = pd.read_sql('SELECT * FROM "cell"', conn)
        theirs = pd.read_sql('SELECT * FROM "foreign_cell"', conn)
    finally:
        conn.close()
    assert list(canonical.columns) == list(theirs.columns)
    assert canonical.equals(theirs)
    assert 'foreign_cell' not in MEASURE_OWNED_TABLES
    assert 'cell' in MEASURE_OWNED_TABLES


class TestACanonicalTableTheImporterFilled:
    """``cell`` is measure's table by name; here the rows in it are theirs.

    The allow-list answers "could measure have written this table?". In a
    project built by ``foreign.run_import`` the honest answer to "did
    it?" is no: the importer copies the imported rows into the canonical
    ``cell`` table when nothing of anyone else's is there. Nothing *in*
    ``cell`` distinguishes those rows from measure's, and a resume clears
    them along with the pending field.

    **These are ``xfail(strict=True)``: they assert the behaviour spaCR
    should have, they fail today, and a fix that makes one of them pass
    fails the suite until it is unmarked.** A guard was written for
    exactly this and backed out — see the ``spacr.resume`` module
    docstring for the five ways it was worse than the bug. The next
    attempt belongs where the canonical table is *appended to*, not where
    it is deleted from, and should read ``foreign_columns`` (which
    records every column the importer wrote, per table) rather than a
    marker row of its own.
    """

    @pytest.mark.xfail(strict=True, reason=(
        'F34: a resume clears the pending field out of the canonical table '
        'an import filled, and reports the imported rows as its own stale '
        'output. The rows survive in foreign_cell, but `cell` is left half '
        "theirs and half spaCR's. Guarding it at this layer was tried and "
        'backed out; see the spacr.resume module docstring.'))
    def test_a_resume_leaves_the_imported_rows_alone(self, imported):
        """Measured: ``cell`` 4 -> 2 rows, ``state.cleared_rows == 2``,
        field ``f2``'s two imported objects gone, and no error."""
        assert _counts(imported['db'])['cell'] == 4
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

        state = plan_measure_resume(imported['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_2'}
        assert state.cleared_rows == 0
        assert _counts(imported['db'])['cell'] == 4
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

    @pytest.mark.xfail(strict=True, reason=(
        'F34, primary flow: a project built purely by import has `cell` as '
        'its only field table, holding every field, so completed_fields_in_db '
        'reports the whole plate measured. measure_crop then runs nothing and '
        "reports the collaborator's numbers as spaCR's own output. This is "
        'the shape any future guard has to reach, and the reason a check '
        'inside clear_field_rows could not: no delete is ever planned here.'))
    def test_a_purely_imported_project_is_not_reported_as_measured(
            self, tmp_path):
        import spacr.foreign as fg

        images, masks, csv_path = _their_project(tmp_path)
        plan = fg.plan_import(images, {'cell': masks}, csv_path, um_per_px=0.5)
        assert plan.ok
        dst = os.path.join(str(tmp_path), 'pristine')
        result = fg.run_import(plan, dst)
        assert result.rows['cell'] == 4          # the canonical copy
        db = os.path.join(dst, 'measurements', 'measurements.db')
        assert discover_field_tables(db) == ['cell']

        state = plan_measure_resume(_resume_settings(dst), verbose=False)

        # spaCR has measured nothing here. Both fields are still to do.
        assert set(state.pending) == {'plate1_A01_1', 'plate1_A01_2'}

    @pytest.mark.xfail(strict=True, reason=(
        'F34, spread by measure: measure_crop appends its own rows into the '
        'canonical table the import filled, in the same columns, with no '
        'column marking the seam — with or without a resume. This is where '
        'the fix belongs, and it is not in this module.'))
    def test_measure_does_not_append_into_a_table_an_import_filled(
            self, imported):
        _measure_field(imported['root'], 'plate1_A01_2', tables=('cell',),
                       n_objects=2)
        conn = sqlite3.connect(imported['db'])
        try:
            mixed = pd.read_sql(
                'SELECT * FROM "cell" WHERE "fieldID" = ?', conn, params=('f2',))
        finally:
            conn.close()
        # Two imported rows and two measured ones, side by side, and the
        # only thing telling them apart is which columns are NULL.
        assert len(mixed) == 2

    def test_a_field_the_import_never_covered_is_still_clearable(self,
                                                                 imported):
        """The control that the backed-out guard failed.

        It refused *table*-wide once an import had filled ``cell``, so a
        field ``measure_crop`` itself wrote could not be cleared and the
        project could never be resumed. Clearing measure's own rows must
        keep working.
        """
        import numpy as np
        np.save(os.path.join(imported['root'], 'merged', 'plate1_A01_3.npy'),
                np.zeros((16, 16, 2), np.uint16))
        _measure_field(imported['root'], 'plate1_A01_3', tables=('cell',))
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2', 'f3'}

        assert clear_field_rows(imported['db'], ['cell'], 'plate1_A01_3') == 3
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

    def test_an_import_beside_spacrs_own_tables_still_resumes(self,
                                                             imported_beside):
        """The other control. ``_may_write_canonical`` refused the canonical
        copy here, so ``cell`` really is measure's, and the
        delete-before-insert must run exactly as it always did — taking
        only measure's rows with it."""
        before = _counts(imported_beside['db'])
        assert before['foreign_cell'] == 4
        assert _fields_in(imported_beside['db'], 'cell') == {'f1', 'f2'}

        state = plan_measure_resume(imported_beside['settings'],
                                    verbose=False)

        assert set(state.pending) == {'plate1_A01_2'}
        assert state.cleared_rows == 3        # field 2's three cell rows
        assert _fields_in(imported_beside['db'], 'cell') == {'f1'}
        after = _counts(imported_beside['db'])
        assert after['foreign_cell'] == 4     # theirs untouched
        assert after['conversion_map'] == before['conversion_map']


def test_foreign_columns_records_what_the_importer_wrote_per_table(imported):
    """The signal a future fix should read, asserted so it cannot rot.

    ``foreign_columns`` names every column the importer put in every table
    it wrote — including the canonical ``cell`` — which is what
    ``foreign._importer_owns`` already uses to decide whether a table is
    still the importer's. Unlike a marker row it is written by every
    importer that ever existed, including the first one, so it does not
    fail open on the databases most likely to carry the bug.

    Read through ``foreign`` rather than by hand: this fixture has had a
    real ``measure_crop`` writer over it, which renames the provenance
    column (see ``tests/test_foreign_preserves_object_tables.py``), and a
    hand-written ``SELECT "column"`` here would quietly answer with the
    word ``'column'`` instead of failing.
    """
    import spacr.foreign as fg

    conn = sqlite3.connect(imported['db'])
    try:
        name_column = fg._provenance_name_column(conn)
        assert name_column is not None
        recorded = {r[0] for r in conn.execute(
            f'SELECT [{name_column}] FROM "foreign_columns" '
            f'WHERE [table] = ?', ('cell',))}
        have = {r[1] for r in conn.execute('PRAGMA table_info("cell")')}
        assert fg._importer_owns(conn, 'cell')
    finally:
        conn.close()
    assert recorded                       # the canonical table is recorded
    assert have <= recorded               # nothing but the importer's columns
