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
"did it?" is no: ``foreign.run_import`` copies the imported rows into
the *canonical* ``cell`` table when the destination is empty, so a
project built purely by import is readable by every spaCR tool. Those
rows sit under spaCR's own metadata columns in a table whose name is on
the allow-list, and a resume cleared them along with the pending field
— and, on a pristine imported project, reported the whole plate as
measured and ran nothing at all.

Both are closed, and the second half of this file is what closes them:

* ownership is decided **per row**, from ``foreign_columns`` — the
  importer's own record of which columns it wrote into which table.
  ``TestOwnershipIsDecidedPerRow`` measures it on a ``cell`` table that
  holds both writers' rows for the *same field*, which is the case a
  table-scoped claim (the shape that was tried here and backed out)
  cannot get right in either direction;
* a copy that measure is about to supersede is handed back before the
  first insert, and only when the hand-back is complete and provably
  lossless — ``TestTheImportersCopyIsReleasedBeforeMeasuring``, which
  also follows the printed remedy and checks what it produced, and which
  measures the release on a ``cell`` holding *both* writers' rows. That
  last case is the one that was missing: the first release deleted by
  ``rowid``, an object table declares a column called ``rowID`` that
  shadows it, and on a mixed table the statement took spaCR's
  measurements along with the copy. Every release test before it ran
  against a table that was 100% the importer's, where deleting too much
  is invisible;
* everything unreadable preserves rather than deletes —
  ``TestUnreadableProvenanceFailsSafe``, including the database shape
  the previous attempt left unprotected.

The last ``xfail(strict=True)`` here — ``measure_crop`` appending into
such a table with no resume anywhere near it — is closed.
``utils._merge_and_save_to_database`` now hands back the import's copy
of the field it is about to write, and refuses to write at all when it
cannot prove the hand-back lossless: see
``TestMeasureRefusesRatherThanMixes``. Scoping the release to the one
field is what makes it safe at the writer, where the whole-table release
``supersede_imported_copies`` performs is not: the released field's
replacement rows are the very next statement, so no field is ever left
with none. The count-and-delete-on-one-predicate property that gates it
is asserted directly, by making the count lie.

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
from spacr.errors import ConfigurationError
from spacr.resume import (
    FIELD_KEY_COLUMNS, MEASURE_OWNED_TABLES, NON_FIELD_TABLES,
    REASON_NOT_MEASURED, REASON_PARTIAL_DB, clear_field_rows,
    completed_fields_in_db, discover_field_tables, format_resume,
    plan_measure_resume,
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


def _measure_field_pre_fix(root, stem, tables=('cell', 'nucleus'),
                           n_objects=N_OBJECTS):
    """``_measure_field`` as spaCR wrote it *before* F34 was fixed at the writer.

    ``utils._merge_and_save_to_database`` now hands back an import's copy of
    the field it is about to write, so the mixed state — one canonical table
    holding both writers' rows for one field — can no longer be produced by
    measuring. Every database written by a spaCR release before that fix can
    still be in it, which is precisely why the resume-side protections below
    must go on being measured on a real mixed table rather than on the
    all-imported one where over-deleting is invisible.

    So the writer-side release is disabled for the duration of the write, and
    nothing else is: the rows, the schema and the provenance are produced by
    the same real writers as everywhere else in this file.
    """
    import spacr.utils as u

    guard = u._release_imported_rows_for_field
    u._release_imported_rows_for_field = lambda *a, **k: 0
    try:
        _measure_field(root, stem, tables=tables, n_objects=n_objects)
    finally:
        u._release_imported_rows_for_field = guard


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
    """``foreign_cell`` and ``conversion_map`` survive a resume.

    ``foreign._foreign_frame`` writes ``plateID`` / ``rowID`` /
    ``columnID`` / ``fieldID`` onto every row so the import joins to the
    measurements, which made it indistinguishable from measure output
    under the structural rule. Measured before the fix: ``foreign_cell``
    4 -> 2 rows and ``conversion_map`` 4 -> 2.

    The last assertion here used to read ``== {'f1'}`` — it *encoded*
    F34, asserting that the pending field's imported rows were cleared
    out of ``cell``. They are the import's rows in a table that is
    measure's only by name, and ``TestACanonicalTableTheImporterFilled``
    below is where that is pinned; this line agrees with it now.
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
    # ...and the canonical table holds the import's rows for both fields,
    # because measure never wrote one of them.
    assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}
    assert after['cell'] == before['cell'] == 4


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
    ``cell`` table when nothing of anyone else's is there. Nothing about
    the *table* distinguishes those rows from measure's, and a resume
    cleared them along with the pending field.

    What does distinguish them is per row and was already in the
    database: ``foreign_columns`` names every column that importer wrote
    into every table it wrote, and a measure row always holds a value in
    a column that list does not contain. ``resume.measure_rows_clause``
    turns that into a WHERE fragment, so one DELETE can take measure's
    rows for a field and leave the import's beside them.

    The first two tests here were ``xfail(strict=True)`` and are now
    ordinary tests. The third still is: it is about the *append*, which
    happens in ``spacr.utils._merge_and_save_to_database``, and no amount
    of care in this module can undo a row that has already been written
    into the wrong table.
    """

    def test_a_resume_leaves_the_imported_rows_alone(self, imported):
        """THE BUG. Measured before the fix: ``cell`` 4 -> 2 rows,
        ``state.cleared_rows == 2``, field ``f2``'s two imported objects
        gone, and no error anywhere.

        Nothing is released here — see
        ``test_a_partially_covered_copy_is_reported_not_released`` for
        why — so the assertion is the strong one: the resume plans its
        work and touches not a single imported row.
        """
        assert _counts(imported['db'])['cell'] == 4
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

        state = plan_measure_resume(imported['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_2'}
        assert state.cleared_rows == 0
        assert state.released_rows == 0
        assert _counts(imported['db'])['cell'] == 4
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

    def test_a_purely_imported_project_is_not_reported_as_measured(
            self, tmp_path):
        """THE BUG, on the flow that reaches it first.

        A project built purely by import has ``cell`` as its only field
        table and a row in it for every field, so
        ``completed_fields_in_db`` reported the whole plate measured,
        ``measure_crop`` ran nothing, and the collaborator's numbers stood
        as spaCR's own output. Measured before the fix:
        ``state.pending == ()``.
        """
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

    def test_measure_releases_the_imports_copy_of_the_field_it_writes(
            self, imported):
        """F34's residual, closed. Was ``xfail(strict=True)``.

        ``measure_crop`` with no resume anywhere near it, writing ``f2``
        into a ``cell`` an import filled. Measured before the fix: four
        rows for ``f2`` — two imported and two measured, side by side,
        the only thing telling them apart being which columns are NULL,
        and every ``count_cell`` for that well the sum of two
        populations.

        The copy of *that field* is handed back first, so the table holds
        one population per field. Nothing is lost: what went is a
        duplicate of ``foreign_cell``, checked row by row against it
        before the delete.
        """
        _measure_field(imported['root'], 'plate1_A01_2', tables=('cell',),
                       n_objects=2)
        conn = sqlite3.connect(imported['db'])
        try:
            mixed = pd.read_sql(
                'SELECT * FROM "cell" WHERE "fieldID" = ?', conn, params=('f2',))
        finally:
            conn.close()
        assert len(mixed) == 2
        # ...and they are spaCR's, not the import's.
        assert mixed['cell_area'].notna().all()
        assert mixed['foreign_areashape_area'].isna().all()
        # The other field the import covers is untouched — a per-field
        # release, not a table-wide one, so f1 keeps its rows and its
        # provenance until a run comes to replace them.
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}
        assert _counts(imported['db'])['cell'] == 4
        # Their numbers are exactly where they were.
        assert _counts(imported['db'])['foreign_cell'] == 4

    def test_the_release_is_idempotent_over_a_second_measure(self, imported):
        """Measuring the same field twice releases once and never refuses.

        The second call finds no imported rows for that field — the first
        took them — so it adds nothing to the statement it runs. What the
        second write leaves behind is the pre-existing append behaviour of
        ``measure_crop``, which is not what this fix is about; what it
        must not do is raise, or take the first write's rows with it.
        """
        _measure_field(imported['root'], 'plate1_A01_2', tables=('cell',),
                       n_objects=2)
        _measure_field(imported['root'], 'plate1_A01_2', tables=('cell',),
                       n_objects=2)

        rows = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(rows) == 6                    # f1 theirs x2, f2 ours x4
        assert int(rows['cell_area'].notna().sum()) == 4
        assert _counts(imported['db'])['foreign_cell'] == 4

    def test_a_field_the_import_never_covered_costs_nothing(self, imported):
        """The control. ``f3`` is measure's alone, so nothing is released.

        The release must be invisible on every field an import did not
        cover — otherwise the guard would be deleting measure's own rows,
        which is the failure it exists to prevent.
        """
        np.save(os.path.join(imported['root'], 'merged', 'plate1_A01_3.npy'),
                np.zeros((16, 16, 2), np.uint16))
        _measure_field(imported['root'], 'plate1_A01_3', tables=('cell',),
                       n_objects=3)

        rows = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(rows) == 7                    # theirs 4 + ours 3
        assert set(rows['fieldID']) == {'f1', 'f2', 'f3'}
        assert int(rows['cell_area'].notna().sum()) == 3

    def test_a_project_that_never_saw_an_import_is_unchanged(self, project):
        """No ``foreign_columns``, no ``foreign_*`` table, no clause, no cost.

        The ordinary case, which must run exactly the statements it always
        did — a guard that changed the common path would be a far bigger
        risk than the bug it closes.
        """
        before = _counts(project['db'])
        _measure_field(project['root'], 'plate1_A01_4', tables=('cell',),
                       n_objects=3)
        after = _counts(project['db'])
        assert after['cell'] == before['cell'] + 3
        assert set(_fields_in(project['db'], 'cell')) == {'f1', 'f2', 'f3',
                                                          'f4'}


class TestMeasureRefusesRatherThanMixes:
    """When the copy cannot be handed back losslessly, nothing is written.

    Refusing costs one field's measurements, which a re-run replaces.
    Mixing costs every count in the project, and nothing detects it. So
    every path that cannot *prove* the removal lossless raises, and the
    message names the remedy.
    """

    def test_it_refuses_when_the_importers_own_copy_is_gone(self, imported):
        """No ``foreign_cell`` to check against — so no delete, and no append.

        Deleting would destroy the only copy; appending would mix. The
        third option is the correct one.
        """
        from spacr.utils import ImportedCopyNotReleased

        conn = sqlite3.connect(imported['db'])
        try:
            conn.execute('DROP TABLE "foreign_cell"')
            conn.commit()
        finally:
            conn.close()
        before = _rows(imported['db'], 'SELECT * FROM cell')

        with pytest.raises(ImportedCopyNotReleased) as excinfo:
            _measure_field(imported['root'], 'plate1_A01_2',
                           tables=('cell',), n_objects=2)

        message = str(excinfo.value)
        assert 'foreign_cell' in message
        assert 'only copy' in message
        after = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(after) == len(before) == 4
        assert 'cell_area' not in after.columns     # not one row written

    def test_it_refuses_when_a_copied_row_has_no_twin(self, imported):
        """One imported row edited out of ``foreign_cell``: it exists nowhere else.

        The lossless check is row by row, not table by table, so a single
        orphan stops the whole write.
        """
        from spacr.utils import ImportedCopyNotReleased

        conn = sqlite3.connect(imported['db'])
        try:
            conn.execute('DELETE FROM "foreign_cell" WHERE "fieldID" = ? '
                         'AND "object_label" = 1', ('f2',))
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ImportedCopyNotReleased) as excinfo:
            _measure_field(imported['root'], 'plate1_A01_2',
                           tables=('cell',), n_objects=2)

        assert 'no matching row' in str(excinfo.value)
        after = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(after) == 4
        assert 'cell_area' not in after.columns

    def test_it_refuses_a_timelapse_rather_than_key_on_fewer_columns(
            self, imported):
        """The importer writes no ``timeID``; a frame cannot be identified.

        ``resume.supersede_imported_copies`` declines a timelapse for
        exactly this reason, and the writer has to agree with it — a
        release keyed on the four field columns alone would take every
        frame of the field.
        """
        from spacr.utils import ImportedCopyNotReleased, _merge_and_save_to_database

        morphology = pd.DataFrame({'label': [1, 2],
                                   'cell_area': [100.0, 101.0]})
        intensity = pd.DataFrame(
            {'label': [1, 2], 'cell_channel_0_mean_intensity': [5.0, 5.0]})
        with pytest.raises(ImportedCopyNotReleased) as excinfo:
            _merge_and_save_to_database(morphology, intensity, 'cell',
                                        imported['root'],
                                        'plate1_A01_2_t1', 'exp', True)

        assert 'timeID' in str(excinfo.value)
        after = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(after) == 4
        assert 'cell_area' not in after.columns

    def test_the_predicate_refuses_a_frame_with_no_field_key(self):
        """A frame missing a key column cannot say which field it is for.

        The message has to name the columns, because the caller that hits
        this is a direct one -- ``measure_crop``'s own frames always carry
        the four -- and "cannot establish the field" with no field named
        is not actionable.
        """
        from spacr.utils import ImportedCopyNotReleased, _field_key_predicate
        from spacr import schema

        frame = pd.DataFrame({'plateID': ['plate1'], 'rowID': ['r1']})
        with pytest.raises(ImportedCopyNotReleased) as excinfo:
            _field_key_predicate(frame, schema.FIELD_KEY_COLUMNS, 's')

        assert 'columnID' in str(excinfo.value)
        assert 'fewer columns than the writer used' in str(excinfo.value)

    def test_an_empty_frame_matches_no_row_rather_than_failing_to_parse(self):
        """The defensive branch, exercised for real.

        ``()`` -- an empty OR -- is not valid SQL, and a predicate that
        will not parse inside a DELETE is a worse answer than one that
        selects nothing. Asserted against SQLite rather than by reading
        the string, because "is this valid SQL" is not a question a test
        should answer by eye.
        """
        from spacr.utils import _field_key_predicate
        from spacr import schema

        empty = pd.DataFrame({c: [] for c in schema.FIELD_KEY_COLUMNS})
        predicate, params = _field_key_predicate(
            empty, schema.FIELD_KEY_COLUMNS, 's')

        assert params == []
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE TABLE t (a)')
            conn.execute('INSERT INTO t VALUES (1)')
            assert conn.execute(
                f'SELECT COUNT(*) FROM t AS s WHERE {predicate}'
            ).fetchone()[0] == 0
        finally:
            conn.close()

    def test_a_delete_that_takes_a_number_nobody_counted_rolls_back(
            self, imported):
        """The property that is actually true, asserted by making it false.

        Count and delete run on one predicate string, and the equality of
        the two numbers is what the write is gated on — not the row
        identity, which is what destroyed this table twice. Here the count
        is made to lie; the delete must roll back and nothing at all may
        be written.
        """
        import spacr.utils as u

        class _Result:
            def __init__(self, count=None, rowcount=0):
                self._count = count
                self.rowcount = rowcount

            def fetchone(self):
                return (self._count,)

        class _LyingConnection:
            """Answers the gating count with a number the delete cannot match."""

            def __init__(self):
                self.statements = []

            def execute(self, sql, params=()):
                self.statements.append(sql)
                if sql.lstrip().upper().startswith('SELECT COUNT(*)'):
                    return _Result(count=99)
                return _Result(rowcount=2)

        conn = _LyingConnection()
        with pytest.raises(u.ImportedCopyNotReleased) as excinfo:
            u._verified_delete(conn, 'cell', 's', '"fieldID" = ?', ['f2'],
                               "release the import's copy of field f2")

        message = str(excinfo.value)
        assert 'did not act on the rows that were checked' in message
        assert '99' in message and '2' in message
        # One predicate string, interpolated into both statements, so the
        # two cannot drift apart in a later edit.
        assert len(conn.statements) == 2
        assert all('"fieldID" = ?' in sql for sql in conn.statements)
        assert conn.statements[0].startswith('SELECT COUNT(*)')
        assert conn.statements[1].startswith('DELETE')

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


# ---------------------------------------------------------------------------
# Row-scoped, not table-scoped: one table, both writers, one DELETE
# ---------------------------------------------------------------------------

def _pristine_import(tmp_path, name='pristine'):
    """A project built by a real import and nothing else.

    The primary flow, and the one every claim about reachability has to
    be measured on: ``cell`` is the only field table, it holds a row for
    every field, and every one of those rows is the importer's.
    """
    import spacr.foreign as fg

    images, masks, csv_path = _their_project(tmp_path)
    plan = fg.plan_import(images, {'cell': masks}, csv_path, um_per_px=0.5)
    assert plan.ok, fg.format_plan(plan)
    dst = os.path.join(str(tmp_path), name)
    result = fg.run_import(plan, dst)
    assert result.rows == {'foreign_cell': 4, 'cell': 4}
    return {'root': dst, 'db': result.db_path,
            'settings': _resume_settings(dst)}


def _rows(db_path, sql, *params):
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


class TestOwnershipIsDecidedPerRow:
    """One ``cell`` table holding both writers' rows, told apart in SQL.

    This is the property the backed-out attempt did not have: it claimed
    the whole table for the importer as soon as an import had filled it,
    so a field ``measure_crop`` itself wrote could not be cleared and the
    project could never be resumed.
    """

    def test_a_mixed_table_is_cleared_of_measures_rows_only(self, imported):
        """Both writers, the same field, one DELETE.

        ``f2`` has two imported rows. A ``measure_crop`` then writes two
        of its own for the same field into the same table — the state F34
        was, written here by :func:`_measure_field_pre_fix` because the
        writer no longer produces it (see
        ``test_measure_releases_the_imports_copy_of_the_field_it_writes``)
        and every database an older spaCR wrote still can be in it.
        Clearing ``f2`` has to take exactly the two measured rows: a
        table-scoped claim keeps all four (unresumable), and no claim at
        all takes all four (F34).
        """
        _measure_field_pre_fix(imported['root'], 'plate1_A01_2',
                               tables=('cell',), n_objects=2)
        assert len(_rows(imported['db'], 'SELECT * FROM cell')) == 6

        assert clear_field_rows(imported['db'], ['cell'],
                                'plate1_A01_2') == 2

        after = _rows(imported['db'], 'SELECT * FROM cell')
        assert len(after) == 4
        # theirs, for both fields, untouched...
        assert after['foreign_areashape_area'].tolist() == [
            36.0, 64.0, 36.0, 64.0]
        # ...and not one measured row left to be doubled by the re-run.
        assert after['cell_area'].isna().all()

    def test_a_field_the_import_never_covered_is_untouched_by_the_clause(
            self, imported):
        """The control, from the other side.

        ``f3`` exists only because measure wrote it. The row clause must
        not protect a single one of its rows — protecting a measure row
        is how a resume comes to double it.
        """
        np.save(os.path.join(imported['root'], 'merged', 'plate1_A01_3.npy'),
                np.zeros((16, 16, 2), np.uint16))
        _measure_field(imported['root'], 'plate1_A01_3', tables=('cell',))

        assert clear_field_rows(imported['db'], ['cell'],
                                'plate1_A01_3') == 3
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

    def test_the_clause_is_read_from_the_importers_own_record(self, imported):
        """Not from a marker this module invented — from ``foreign_columns``.

        Widening the record so that it claims a measure column turns
        measure's rows into the importer's, which is the mechanism, made
        visible. If this ever stops holding, the protection above is
        coming from somewhere else and nobody has noticed.
        """
        from spacr.resume import measure_rows_clause

        _measure_field(imported['root'], 'plate1_A01_2', tables=('cell',),
                       n_objects=2)
        conn = sqlite3.connect(imported['db'])
        try:
            name_column = _provenance_column(conn)
            clause = measure_rows_clause(conn, 'cell')
            assert 'cell_area' in clause          # measure's, and unrecorded
            conn.execute(
                f'INSERT INTO "foreign_columns" ("table", "{name_column}") '
                f'VALUES (?, ?)', ('cell', 'cell_area'))
            conn.commit()
            widened = measure_rows_clause(conn, 'cell')
        finally:
            conn.close()
        assert 'cell_area' not in widened


def _provenance_column(conn):
    """The spelling ``foreign_columns`` uses in this database.

    Through ``spacr.resume`` rather than by hand: a measure write renames
    it, and a hand-written ``SELECT "column"`` answers with the word
    ``'column'`` instead of failing.
    """
    from spacr.resume import _foreign_name_column

    name_column = _foreign_name_column(conn)
    assert name_column is not None
    return name_column


# ---------------------------------------------------------------------------
# The copy measure is about to supersede
# ---------------------------------------------------------------------------

class TestTheImportersCopyIsReleasedBeforeMeasuring:
    """``measure_crop`` appends, so the copy cannot still be there when it runs.

    Leaving the imported rows alone is only half an answer: measure would
    then add its own to the same table and every per-well count would be
    the sum of two populations. The copy is handed back first — when the
    hand-back is *complete*, which is when every field it covers is
    either already measured into that table or queued to be measured now.
    """

    def test_it_fires_on_a_pristine_imported_project(self, tmp_path):
        """The primary flow, end to end, with the numbers.

        Nothing about this project has ever been measured, so the copy is
        wholly superseded by the run that is about to start.
        """
        project = _pristine_import(tmp_path)

        state = plan_measure_resume(project['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_1', 'plate1_A01_2'}
        assert state.released_rows == 4
        assert state.notes == ()
        counts = _counts(project['db'])
        assert counts['cell'] == 0             # spaCR's to fill
        assert counts['foreign_cell'] == 4     # theirs, untouched
        # ...and their numbers are still one query away.
        assert 'cell_with_foreign' in _views(project['db'])

    def test_the_measured_table_is_then_spacrs_alone(self, tmp_path):
        """The point of all of it: measure runs and nothing is doubled.

        Measured before the fix, on this exact project: ``cell`` reported
        as fully measured, ``measure_crop`` skipping every field, and the
        collaborator's four rows standing as spaCR's output. Measured
        with the release but without it: eight rows, four of each, and
        ``count_cell`` twice what it should be.
        """
        project = _pristine_import(tmp_path)
        state = plan_measure_resume(project['settings'], verbose=False)
        for stem in state.pending:
            _measure_field(project['root'], stem, tables=('cell', 'nucleus'),
                           n_objects=2)

        cells = _rows(project['db'], 'SELECT * FROM cell')
        assert len(cells) == 4                       # not 8, and not 0
        assert cells['cell_area'].notna().all()      # every row is measured
        assert 'foreign_areashape_area' not in cells.columns
        assert sorted(cells['fieldID'].unique()) == ['f1', 'f2']

        # Theirs beside spaCR's, one object at a time, through the view.
        joined = _rows(project['db'], 'SELECT * FROM cell_with_foreign')
        assert len(joined) == 4
        assert joined['foreign_areashape_area'].tolist() == [
            36.0, 64.0, 36.0, 64.0]

    def test_the_claim_does_not_outlive_the_rows(self, tmp_path):
        """Nothing may still say the importer owns a table it no longer fills.

        The backed-out attempt had no way to un-claim at all, so its own
        remedy left a refusal standing over rows that were gone. Both
        records are updated in the same transaction as the delete.
        """
        import spacr.foreign as fg

        project = _pristine_import(tmp_path)
        plan_measure_resume(project['settings'], verbose=False)

        provenance = _rows(project['db'],
                           'SELECT * FROM foreign_columns WHERE "table" = ?',
                           'cell')
        assert len(provenance) == 0
        run = _rows(project['db'], 'SELECT * FROM foreign_import')
        assert int(run['canonical_table_written'].iloc[0]) == 0
        assert 'released' in run['canonical_table_note'].iloc[0]
        conn = sqlite3.connect(project['db'])
        try:
            assert fg._importer_owns(conn, 'cell') is False
            assert fg._importer_owns(conn, 'foreign_cell') is True
        finally:
            conn.close()

    def test_releasing_twice_is_a_no_op(self, tmp_path):
        """Idempotent, so a second resume is not a second surprise."""
        import spacr.foreign as fg

        project = _pristine_import(tmp_path)
        assert fg.release_canonical_copy(project['db'], 'cell') == 4
        assert fg.release_canonical_copy(project['db'], 'cell') == 0
        assert _counts(project['db'])['foreign_cell'] == 4

    def test_two_imported_object_tables_are_both_released(self, tmp_path):
        """A database can hold more than one import; each table is its own.

        Their cells from one run and their nuclei from another, both
        copied into their canonical tables. One resume has to hand both
        back — and give each its own view — or the half it missed is the
        one that ends up doubled.
        """
        import spacr.foreign as fg

        images, masks, csv_path = _their_project(tmp_path)
        nuclei = os.path.join(str(tmp_path), 'their_nucleus_masks')
        os.makedirs(nuclei, exist_ok=True)
        for field in (1, 2):
            tifffile.imwrite(
                os.path.join(nuclei, f'fov{field:02d}_nucleus_mask.tif'),
                _their_labels())
        dst = os.path.join(str(tmp_path), 'two')
        fg.run_import(fg.plan_import(images, {'cell': masks}, csv_path,
                                     um_per_px=0.5), dst)
        fg.run_import(fg.plan_import(images, {'nucleus': nuclei}, csv_path,
                                     um_per_px=0.5,
                                     measurement_object='nucleus'), dst)
        db = os.path.join(dst, 'measurements', 'measurements.db')
        assert discover_field_tables(db) == ['cell', 'nucleus']

        state = plan_measure_resume(_resume_settings(dst), verbose=False)

        assert state.released_rows == 8            # 4 + 4
        assert state.notes == ()
        counts = _counts(db)
        assert counts['cell'] == 0 and counts['nucleus'] == 0
        assert counts['foreign_cell'] == 4 and counts['foreign_nucleus'] == 4
        assert _views(db) == {'cell_with_foreign', 'nucleus_with_foreign'}

    def test_a_project_with_no_import_at_all_is_untouched(self, project):
        """The overwhelmingly common case pays none of this.

        No ``foreign_columns``, no ``foreign_*`` table, so no clause is
        added to any statement and no release is even considered.
        """
        state = plan_measure_resume(project['settings'], verbose=False)
        assert state.released_rows == 0
        assert state.notes == ()
        assert state.cleared_rows == 3        # field 3's stale cell rows

    def test_it_releases_only_the_copy_from_a_table_measure_has_grown(
            self, tmp_path):
        """THE BUG, on the flow that reaches it — a *mixed* canonical table.

        The state a real resume finds: the import covers ``f1`` and
        ``f2`` and copied both into ``cell``; ``measure_crop`` then wrote
        ``f1`` into that same table and died; the resume queues ``f2``.
        Every field the copy covers is now either measured or pending, so
        the release fires — and it fires on a table holding both writers'
        rows for ``f1``, under identical
        ``plateID``/``rowID``/``columnID``/``fieldID``/``object_label``.

        Measured before the fix: ``release_canonical_copy`` issued
        ``DELETE … WHERE rowid IN (SELECT s.rowid …)``, every ``rowid``
        resolved to the ``rowID`` *column* (``'r1'`` for all six rows),
        and the statement took all six — spaCR's two measurements of
        ``f1`` went with the import's four, and the resume reported six
        rows released and no error. The measurements were the only copy.

        Every other release test in this suite runs against a ``cell``
        that is 100% the importer's, where over-deleting is invisible
        because everything was going anyway. That is the gap this closes.
        """
        project = _pristine_import(tmp_path, name='mixed')
        # measure_crop wrote f1 -- the field the import also covers. Through
        # the pre-fix writer: this is a database an older spaCR left behind,
        # and the resume has to go on handling it correctly forever.
        _measure_field_pre_fix(project['root'], 'plate1_A01_1',
                               tables=('cell', 'nucleus'), n_objects=2)
        before = _rows(project['db'], 'SELECT * FROM cell')
        assert len(before) == 6
        assert int(before['cell_area'].notna().sum()) == 2

        state = plan_measure_resume(project['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_2'}
        assert state.released_rows == 4          # theirs, and only theirs
        assert state.notes == ()
        after = _rows(project['db'], 'SELECT * FROM cell')
        assert len(after) == 2                   # spaCR's f1, still there
        assert after['cell_area'].tolist() == [100.0, 101.0]
        assert set(after['fieldID']) == {'f1'}
        assert _counts(project['db'])['foreign_cell'] == 4
        assert 'cell_with_foreign' in _views(project['db'])

        # ...and finishing the run leaves one population, not two.
        for stem in state.pending:
            _measure_field(project['root'], stem,
                           tables=('cell', 'nucleus'), n_objects=2)
        cells = _rows(project['db'], 'SELECT * FROM cell')
        assert len(cells) == 4                   # not 8, not 2
        assert cells['cell_area'].notna().all()
        assert sorted(cells['fieldID'].unique()) == ['f1', 'f2']

    def test_a_partially_covered_copy_is_reported_not_released(self,
                                                               imported):
        """A half-released table is worse than an unreleased one.

        Here ``f1`` is covered by the import and is *done* — measure has
        nucleus rows for it and will not run it again — so releasing the
        copy would leave ``f1`` with no rows in ``cell`` at all and no run
        coming to replace them. The copy stays whole, and the user is told
        what to run and what it will do.
        """
        state = plan_measure_resume(imported['settings'], verbose=False)

        assert state.released_rows == 0
        assert _counts(imported['db'])['cell'] == 4
        assert len(state.notes) == 1
        note = state.notes[0]
        assert 'plate1_r1_c1_f1' in note                  # the field at issue
        assert 'release_canonical_copy' in note           # the command
        assert 'foreign_cell' in note and 'cell_with_foreign' in note
        # ...and it reaches the user, not just the object.
        printed = format_resume(state)
        assert 'ACTION NEEDED' in printed
        assert 'release_canonical_copy' in printed

    def test_following_that_advice_does_what_it_says(self, imported):
        """The remedy, run exactly as printed, and checked afterwards.

        A remedy that does not work is worse than none — the backed-out
        attempt's first one silently destroyed the rows it was about. This
        one is followed here: release, resume again, measure what the
        resume says is pending, and count.
        """
        import spacr.foreign as fg

        state = plan_measure_resume(imported['settings'], verbose=False)
        assert 'release_canonical_copy' in state.notes[0]

        released = fg.release_canonical_copy(imported['db'], 'cell')

        assert released == 4
        assert _counts(imported['db'])['cell'] == 0
        assert _counts(imported['db'])['foreign_cell'] == 4

        again = plan_measure_resume(imported['settings'], verbose=False)
        # The advice promised these fields would be measured. Both were
        # only ever in `cell` as the import's rows, so both come back.
        assert set(again.pending) == {'plate1_A01_1', 'plate1_A01_2'}
        assert again.notes == ()
        # f1's stale nucleus rows go, so re-running it cannot double them.
        assert again.cleared_rows == 2
        for stem in again.pending:
            _measure_field(imported['root'], stem,
                           tables=('cell', 'nucleus'), n_objects=2)

        counts = _counts(imported['db'])
        assert counts['cell'] == 4            # two fields, two objects
        assert counts['nucleus'] == 4         # not 6
        assert counts['foreign_cell'] == 4
        joined = _rows(imported['db'], 'SELECT * FROM cell_with_foreign')
        assert len(joined) == 4
        assert joined['foreign_areashape_area'].tolist() == [
            36.0, 64.0, 36.0, 64.0]


# ---------------------------------------------------------------------------
# Failing safe: everything unreadable must preserve, never delete
# ---------------------------------------------------------------------------

class TestUnreadableProvenanceFailsSafe:

    def test_a_release_without_the_other_copy_is_refused(self, tmp_path):
        """The copy is only a duplicate while the original is there.

        With ``foreign_cell`` gone the rows in ``cell`` are the only ones
        left, so nothing may remove them — not the release, and not the
        resume that would have called it.
        """
        import spacr.foreign as fg

        project = _pristine_import(tmp_path)
        conn = sqlite3.connect(project['db'])
        try:
            conn.execute('DROP TABLE "foreign_cell"')
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ConfigurationError, match='foreign_cell'):
            fg.release_canonical_copy(project['db'], 'cell')

        state = plan_measure_resume(project['settings'], verbose=False)
        assert state.released_rows == 0
        assert _counts(project['db'])['cell'] == 4
        assert len(state.notes) == 1
        assert 'foreign_cell' in state.notes[0]

    def test_a_row_with_no_twin_stops_the_whole_release(self, tmp_path):
        """One unmatched row and none of them go.

        A ``cell`` row the importer wrote but ``foreign_cell`` does not
        hold exists nowhere else. Deleting the other three and leaving it
        would be a half-release; the call refuses and writes nothing.
        """
        import spacr.foreign as fg

        project = _pristine_import(tmp_path)
        conn = sqlite3.connect(project['db'])
        try:
            conn.execute('DELETE FROM "foreign_cell" WHERE fieldID = ?',
                         ('f2',))
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ConfigurationError, match='2 of its 4'):
            fg.release_canonical_copy(project['db'], 'cell')
        assert _counts(project['db'])['cell'] == 4

    def test_a_database_whose_provenance_was_replaced_still_protects_it(
            self, imported):
        """The failure mode the last attempt left live, closed.

        The release this module fixes wrote ``foreign_columns`` with
        ``to_sql(if_exists='replace')``, so importing a second object type
        into the same database erased the first import's record — and with
        it any claim on ``cell``. Reproduced here by deleting exactly
        those rows. The importer's own copy, ``foreign_cell``, is still a
        faithful record of what it wrote, and the imported rows survive on
        the strength of it.
        """
        conn = sqlite3.connect(imported['db'])
        try:
            conn.execute('DELETE FROM "foreign_columns" WHERE "table" = ?',
                         ('cell',))
            conn.commit()
        finally:
            conn.close()

        state = plan_measure_resume(imported['settings'], verbose=False)

        assert set(state.pending) == {'plate1_A01_2'}
        assert state.cleared_rows == 0
        assert _counts(imported['db'])['cell'] == 4
        assert _fields_in(imported['db'], 'cell') == {'f1', 'f2'}

    def test_a_provenance_column_renamed_away_still_protects_it(self,
                                                                imported):
        """Neither ``column`` nor ``columnID``: unreadable, so preserve.

        ``foreign._importer_owns`` answers *False* to the same database,
        because it gates a DROP-and-rewrite and the protective answer
        there is the opposite one. The two must not be collapsed.
        """
        import spacr.foreign as fg

        conn = sqlite3.connect(imported['db'])
        try:
            name_column = _provenance_column(conn)
            conn.execute(f'ALTER TABLE "foreign_columns" '
                         f'RENAME COLUMN "{name_column}" TO "whatever"')
            conn.commit()
            assert fg._importer_owns(conn, 'cell') is False
        finally:
            conn.close()

        state = plan_measure_resume(imported['settings'], verbose=False)

        assert state.cleared_rows == 0
        assert _counts(imported['db'])['cell'] == 4

    def test_a_resume_that_cannot_import_foreign_preserves_rather_than_deletes(
            self, tmp_path, monkeypatch):
        """No importer, no release — and certainly no blind delete.

        The lossless check lives in ``spacr.foreign``. Without it there is
        no way to know a row still exists elsewhere, so the copy stays and
        the user is told why.
        """
        import sys

        project = _pristine_import(tmp_path)
        monkeypatch.setitem(sys.modules, 'spacr.foreign', None)

        state = plan_measure_resume(project['settings'], verbose=False)

        assert state.released_rows == 0
        assert _counts(project['db'])['cell'] == 4
        assert len(state.notes) == 1
        assert 'spacr.foreign' in state.notes[0]
        # ...and the fields are still queued, so nothing is reported as
        # measured that was not.
        assert set(state.pending) == {'plate1_A01_1', 'plate1_A01_2'}

    def test_a_timelapse_resume_releases_nothing(self, tmp_path):
        """The importer writes no ``timeID``, so it cannot own a frame.

        Rather than key a delete on fewer columns than the writer used,
        the release is declined outright and reported.
        """
        from spacr.resume import supersede_imported_copies

        project = _pristine_import(tmp_path)
        released, notes = supersede_imported_copies(
            project['db'], ['cell'], ['plate1_A01_1', 'plate1_A01_2'],
            timelapse=True)

        assert released == 0
        assert len(notes) == 1
        assert _counts(project['db'])['cell'] == 4


def _views(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'")}
    finally:
        conn.close()


class TestTheOwnershipReaderOnItsOwn:
    """The predicate, asked directly, on the cases the callers guard.

    Every one of these is a path a resume takes on a real database; they
    are exercised here rather than only through ``plan_measure_resume``
    so that a wrong answer is reported as a wrong answer rather than as
    some downstream count.
    """

    def test_a_table_no_import_ever_wrote_needs_no_clause(self, project):
        """No condition at all, so an ordinary project pays nothing.

        ``None`` and ``'0'`` are not interchangeable: ``None`` means "add
        nothing to the WHERE", and returning ``'0'`` here would make every
        delete a no-op and every resume a duplication.
        """
        from spacr.resume import (importer_recorded_columns,
                                  importer_rows_clause,
                                  importer_written_columns,
                                  measure_rows_clause)

        conn = sqlite3.connect(project['db'])
        try:
            for table in ('cell', 'nucleus', 'png_list'):
                assert importer_recorded_columns(conn, table) is None, table
                assert importer_written_columns(conn, table) is None, table
                assert measure_rows_clause(conn, table) is None, table
                assert importer_rows_clause(conn, table) is None, table
        finally:
            conn.close()

    def test_a_wholly_imported_table_has_no_measure_rows(self, imported):
        """``'0'`` — the whole point, spelled as SQL that selects nothing."""
        from spacr.resume import importer_rows_clause, measure_rows_clause

        conn = sqlite3.connect(imported['db'])
        try:
            assert measure_rows_clause(conn, 'cell') == '0'
            assert importer_rows_clause(conn, 'cell') == '1'
            # ...and the database agrees, which is what actually matters.
            assert conn.execute(
                'SELECT COUNT(*) FROM cell WHERE 0').fetchone()[0] == 0
            assert conn.execute(
                'SELECT COUNT(*) FROM cell WHERE 1').fetchone()[0] == 4
        finally:
            conn.close()

    def test_a_database_that_does_not_exist_releases_nothing(self, tmp_path):
        from spacr.resume import supersede_imported_copies

        assert supersede_imported_copies(
            str(tmp_path / 'nothing.db'), ['cell'], ['plate1_A01_1']) == (0, [])

    def test_a_table_not_on_the_allow_list_is_never_released(self, imported):
        """Named by hand or not, only measure's own tables are considered.

        ``foreign_cell`` is entirely the importer's, so every test above
        would call it releasable. It is not measure's to release.
        """
        from spacr.resume import supersede_imported_copies

        released, notes = supersede_imported_copies(
            imported['db'], ['foreign_cell', 'conversion_map'],
            ['plate1_A01_1', 'plate1_A01_2'])

        assert (released, notes) == (0, [])
        assert _counts(imported['db'])['foreign_cell'] == 4

    def test_an_unparseable_pending_name_is_skipped_not_guessed(self,
                                                                imported):
        """A name that is not a field stem cannot vouch for a field.

        It must not be treated as covering one — that would release a
        copy on the strength of a name nobody can resolve.
        """
        from spacr.resume import supersede_imported_copies

        released, notes = supersede_imported_copies(
            imported['db'], ['cell'], ['not-a-field-stem'])

        assert released == 0
        assert len(notes) == 1
        assert _counts(imported['db'])['cell'] == 4


# ---------------------------------------------------------------------------
# One vocabulary, spelled once
# ---------------------------------------------------------------------------

def test_resume_and_foreign_name_the_same_provenance():
    """The link that would otherwise be missing.

    ``resume`` must stay stdlib-only — it runs at the top of
    ``measure_crop``, before any model is loaded — so it cannot import
    ``foreign`` to learn where the provenance lives. The names are
    therefore defined in ``resume`` and *aliased* by ``foreign``, rather
    than written out twice: two spellings of ``foreign_columns`` that
    drifted apart would be a delete on one side and a refusal on the
    other.
    """
    import spacr.foreign as fg
    import spacr.resume as resume

    assert fg.FOREIGN_COLUMNS_TABLE is resume.FOREIGN_COLUMNS_TABLE
    assert fg.FOREIGN_PREFIX is resume.FOREIGN_PREFIX
    assert fg._PROVENANCE_NAME_COLUMNS is resume.FOREIGN_NAME_COLUMNS
    assert fg._provenance_name_column is resume._foreign_name_column


def test_resume_still_imports_nothing_heavy():
    """The release reaches ``foreign`` lazily, and only when it has to.

    ``spacr.foreign`` pulls in pandas, numpy and ``spacr.convert``.
    Importing it from ``resume`` at module scope would cost that on every
    process that merely wants to know whether it can skip a field, which
    is the constraint the whole module is written under.
    """
    import spacr.resume as module

    source = open(module.__file__).read()
    assert 'from .foreign import' in source          # it is reached...
    assert '\nfrom .foreign import' not in source    # ...but never at top level
    assert 'import pandas' not in source
    assert 'import numpy' not in source
