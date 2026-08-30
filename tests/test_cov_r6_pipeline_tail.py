"""The last branch in eight pipeline modules.

These are the biggest modules in the tree and each has one arc left. Four
of them turn out to be guards that a line above has already made true --
those get a proof and a test that pins the invariant, never a
``pragma``. The rest are driven.

Module by module:

``spacr.remote_execution``
    Slurm's ``sacct`` printing a state with no exit code, and a
    cloud/custom profile whose job-ID pattern matches nothing.
``spacr.object``
    The ``cellpose_*_channel`` fallback reads DENSE stack positions, and
    every raw role channel it looks up is one ``dense_mask_channel_
    positions`` has already recorded.
``spacr.settings``
    An advanced family with no members contributes no heading; and every
    organelle slot's ``channel``/``mask_dim``/``chann_dim`` key is a
    declared setting, which is why the import-time loop never skips one.
``spacr.core``
    ``preprocess_generate_masks`` admits only ``str`` or ``list`` for
    ``src``, so by the time the list branch is reached it is a list.
``spacr.convert``
    ``read_map`` requires a ``target`` column, so the map is always
    indexed on it.
``spacr.seg_qc``
    A flag finding whose fields have no names at all still says which
    plate it is about.
``spacr.foreign``
    An import into a project that already has a ``conversion_map``
    merges, and clears any staging table a crashed import left behind.
``spacr.crops``
    ``_region_for`` always returns a region, so a crop is always masked
    by one -- an all-ones one when the bounding box is the object.
``spacr.layers``
    ``lock`` marks the panel locked before it looks for the leader, so
    there is always at least one leader to align to.

Not covered here because the arcs are gone from this tree:
``spacr.align``'s ``if span > 0`` in ``_feather_width`` and ``if parent``
in the coordinate writer have both been deleted, each replaced by the
one-line comment that says why they could not fail
(``_overlap_windows`` returns ``None`` unless both dimensions are
positive; ``dirname(abspath(...))`` is never empty).
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# spacr.remote_execution
# ---------------------------------------------------------------------------

class _QueueRunner:
    """Deterministic command runner; records every argument vector."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv}")
        return self.results.pop(0)


class TestSlurmAccountingWithoutAnExitCode:
    """``if len(columns) > 1 and ":" in columns[1]:`` in ``_SlurmBackend.refresh``."""

    @staticmethod
    def _job():
        from spacr.remote_execution import RemoteJob

        return RemoteJob("j", "mask", "hpc", "slurm", external_id="4711")

    @staticmethod
    def _profile():
        from spacr.remote_execution import ExecutionProfile

        return ExecutionProfile("hpc", "slurm", host="login", workdir="/work")

    def test_a_state_with_no_exit_code_column_still_sets_the_state(self):
        """``sacct --format=State,ExitCode`` can print State alone.

        A job cancelled before it ran has no exit code to report. The
        state is still the answer, and inventing a code for it would be
        worse than leaving the one the job carries.
        """
        from spacr import remote_execution as rx

        job = self._job()
        job.exit_code = None
        runner = _QueueRunner(
            rx.CommandResult(1, "", "slurm_load_jobs error"),   # squeue: gone
            rx.CommandResult(0, "CANCELLED\n", ""))             # sacct: no |

        rx._SlurmBackend(runner).refresh(self._profile(), job)

        assert job.status == "cancelled"
        assert job.exit_code is None, (
            "with no ExitCode column there is nothing to parse; got "
            f"{job.exit_code!r}")

    def test_an_exit_code_column_is_read_when_it_is_there(self):
        """The contrast that makes the assertion above a real absence."""
        from spacr import remote_execution as rx

        job = self._job()
        runner = _QueueRunner(
            rx.CommandResult(1, "", "slurm_load_jobs error"),
            rx.CommandResult(0, "FAILED|2:0\n", ""))

        rx._SlurmBackend(runner).refresh(self._profile(), job)

        assert job.status == "failed"
        assert job.exit_code == 2


class TestACloudPatternThatMatchesNothing:
    """``if match:`` in ``_CommandBackend.submit``."""

    @staticmethod
    def _profile(pattern):
        from spacr.remote_execution import ExecutionProfile

        return ExecutionProfile(
            "cloud", "command",
            submit_command="cloud submit {settings}",
            status_command="cloud status {external_id}",
            cancel_command="cloud cancel {external_id}",
            job_id_pattern=pattern)

    @staticmethod
    def _job():
        from spacr.remote_execution import RemoteJob

        return RemoteJob("j", "mask", "cloud", "command",
                         settings_path="/tmp/s.csv")

    def test_output_the_pattern_does_not_recognise_is_refused_by_name(self):
        """A submit that printed something else must not become job ''.

        Silently keeping an empty external id would leave a job that can
        never be polled or cancelled, reported as queued.
        """
        from spacr import remote_execution as rx

        runner = _QueueRunner(
            rx.CommandResult(0, "submitted, see the console\n", ""))

        with pytest.raises(rx.RemoteExecutionError, match="unsafe or empty"):
            rx._CommandBackend(runner).submit(
                self._profile(r"job-(\d+)"), self._job(), "payload")

    def test_output_the_pattern_does_recognise_becomes_the_job_id(self):
        """The contrast: the same profile, output the pattern matches."""
        from spacr import remote_execution as rx

        runner = _QueueRunner(rx.CommandResult(0, "queued job-8123 ok\n", ""))
        job = self._job()

        rx._CommandBackend(runner).submit(
            self._profile(r"job-(\d+)"), job, "payload")

        assert job.external_id == "8123"
        assert job.status == "queued"


# ---------------------------------------------------------------------------
# spacr.object -- the dense-position fallback
# ---------------------------------------------------------------------------

class TestEveryRoleChannelHasADensePosition:
    """``if _raw in _dense:`` in both Cellpose mask generators.

    ``_dense = dense_mask_channel_positions(settings)`` walks
    ``utils.MASK_CHANNEL_ROLE_ORDER`` -- ``nucleus_channel``,
    ``cell_channel``, ``pathogen_channel`` and every
    ``<organelle role>_channel`` -- coercing each with ``int()`` and
    giving each newly seen raw channel the next dense position. The loop
    underneath then reads exactly those same keys, coerces them the same
    way with ``int()``, and asks whether the result is in ``_dense``.

    It always is: the key set the loop iterates is a subset of the key
    set the map was built from, and the coercion is identical, so a raw
    channel that survives ``int()`` in the loop survived it in the map
    too. Both copies of the guard (in ``generate_cellpose_masks_sam``
    and in ``generate_cellpose_masks``) are re-checks of what the call
    on the line above has already established.

    Pinned instead is the property the fallback exists for, and the trap
    the module's own comment records: the copied value is the DENSE
    position, which differs from the raw channel whenever the roles are
    not in ascending channel order.
    """

    def test_the_map_covers_every_role_channel_the_fallback_reads(self):
        from spacr.utils import (MASK_CHANNEL_ROLE_ORDER,
                                 dense_mask_channel_positions)

        for role in ("nucleus", "cell", "pathogen", "organelle"):
            assert f"{role}_channel" in MASK_CHANNEL_ROLE_ORDER, (
                f"the fallback reads {role}_channel, so the dense map has "
                "to be built from it too")

        settings = {"nucleus_channel": 2, "cell_channel": 0,
                    "organelle_channel": 1, "pathogen_channel": None}
        dense = dense_mask_channel_positions(settings)

        for role in ("nucleus", "cell", "organelle"):
            assert int(settings[f"{role}_channel"]) in dense, (
                f"{role}'s raw channel must have a dense position; {dense}")

    def test_the_dense_position_is_role_order_not_sorted_order(self):
        """The trap: sorted() and role order disagree, and one is wrong.

        With ``nucleus=2, cell=0, organelle=1`` the merged stack is built
        ``[2, 0, 1]``, so raw channel 1 -- the organelle -- sits at
        position 2. Reading ``sorted({2, 0, 1})`` instead would say
        position 1, which holds the CELL image.
        """
        from spacr.utils import dense_mask_channel_positions

        dense = dense_mask_channel_positions(
            {"nucleus_channel": 2, "cell_channel": 0, "organelle_channel": 1})

        assert dense == {2: 0, 0: 1, 1: 2}, (
            "positions must follow MASK_CHANNEL_ROLE_ORDER, not the sorted "
            f"channel numbers; got {dense}")

    def test_a_channel_that_is_not_a_number_is_absent_from_both(self):
        """The one way a role has no dense position -- and the loop skips it too.

        ``int('later')`` raises in ``dense_mask_channel_positions`` and
        raises again in the generators' own ``try``, so the role is
        dropped by both, in step. That is why "in the map" and "readable
        by the loop" cannot come apart.
        """
        from spacr.utils import dense_mask_channel_positions

        dense = dense_mask_channel_positions(
            {"nucleus_channel": "later", "cell_channel": 0})

        assert dense == {0: 0}
        with pytest.raises((TypeError, ValueError)):
            int("later")


# ---------------------------------------------------------------------------
# spacr.settings
# ---------------------------------------------------------------------------

class TestAnAdvancedFamilyWithNoMembers:
    """``if members:`` in ``_regroup_advanced``."""

    def test_a_table_with_no_family_members_grows_no_headings(self):
        from spacr.settings import _ADVANCED_FAMILIES, _regroup_advanced

        table = {"General": ["src", "metadata_type"]}

        out = _regroup_advanced(table)

        assert out == {"General": ["src", "metadata_type"]}, (
            "a table with nothing to move must come back unchanged; got "
            f"{out}")
        for heading, _s, _p in _ADVANCED_FAMILIES:
            assert heading not in out, \
                f"{heading!r} was invented for a family with no members"

    def test_a_table_with_a_family_member_gets_that_heading(self):
        """The contrast: one real member, and the heading appears with it."""
        from spacr.settings import (_ADVANCED_FAMILIES,
                                    _advanced_family_members,
                                    _regroup_advanced)

        # Find a family this build actually has members for, and a key in it.
        for heading, suffixes, prefixes in _ADVANCED_FAMILIES:
            members = _advanced_family_members(
                {"General": list(_all_setting_keys())}, suffixes, prefixes)
            if members:
                break
        else:                                        # pragma: no branch
            pytest.fail("no advanced family matches any declared setting")

        out = _regroup_advanced({"General": list(members)})

        assert heading in out, (
            f"{heading!r} must appear once its members are in the table")
        assert set(out[heading]) == set(members)
        assert out.get("General", []) == [], \
            "the members moved rather than being duplicated"


def _all_setting_keys():
    from spacr.settings import expected_types

    return list(expected_types)


class TestEveryOrganelleSlotKeyIsADeclaredSetting:
    """``if _key in expected_types and _key not in categories['General']:``

    That loop runs once, at import, over
    ``ORGANELLE_SLOT_ROLES[1:]`` x ``('channel', 'mask_dim',
    'chann_dim')``. Its false side is never taken, and the reason is a
    fact about the settings contract rather than about this file: every
    one of those keys IS declared in ``expected_types``, and
    ``categories['General']`` does not already list any of them (the
    slot roles are exactly the ones the base ``organelle`` category does
    not cover). There is no second import in which the loop could see a
    different table.

    So the invariant is pinned directly. It fails the day a slot role is
    added without its three keys being declared -- which is the bug the
    guard was written against, and which coverage could never report.
    """

    def test_each_slot_declares_channel_mask_dim_and_chann_dim(self):
        from spacr.settings import (ORGANELLE_SLOT_ROLES, categories,
                                    expected_types)

        assert len(ORGANELLE_SLOT_ROLES) > 1, \
            "with one slot the loop below has nothing to say"

        general = categories["General"]
        for role in ORGANELLE_SLOT_ROLES[1:]:
            for suffix in ("channel", "mask_dim", "chann_dim"):
                key = f"{role}_{suffix}"
                assert key in expected_types, (
                    f"{key} is placed in the General category at import, so "
                    "it has to be a declared setting")
                assert general.count(key) == 1, (
                    f"{key} must appear in General exactly once; got "
                    f"{general.count(key)}")

    def test_the_first_slot_is_the_plain_organelle_and_is_not_re_added(self):
        """``[1:]`` is deliberate: slot one is the base ``organelle_*`` keys."""
        from spacr.settings import ORGANELLE_SLOT_ROLES, categories

        assert ORGANELLE_SLOT_ROLES[0] == "organelle"
        general = categories["General"]
        assert general.count("organelle_channel") <= 1


# ---------------------------------------------------------------------------
# spacr.core -- src is a str or a list, and nothing else
# ---------------------------------------------------------------------------

class TestSrcIsAlwaysAListByTheTimeItIsUsed:
    """``if isinstance(settings['src'], list):`` in ``preprocess_generate_masks``.

    Three lines decide this, all above it:

    * ``if not isinstance(settings['src'], (str, list)): raise
      ValueError`` refuses anything else outright;
    * ``normalize_src_path`` returns a ``list`` for a list, and a ``str``
      (or a real list, for a stringified one) for a string -- it raises
      for anything else and can return nothing else;
    * ``if isinstance(settings['src'], str): settings['src'] = [...]``
      converts what is left.

    The ``consolidate`` branch between them assigns a list too. So the
    ``list`` check is a re-check of what those three have guaranteed, and
    its false side -- a silent ``return`` that would look exactly like a
    finished run -- cannot be reached. Both of the guards that make that
    true are pinned below.
    """

    def test_a_src_that_is_neither_string_nor_list_is_refused_out_loud(self):
        from spacr.core import preprocess_generate_masks

        with pytest.raises(ValueError,
                           match="src must be a string or a list of strings"):
            preprocess_generate_masks({"src": ("a", "b")})

    def test_src_is_a_required_parameter(self):
        from spacr.core import preprocess_generate_masks

        with pytest.raises(ValueError, match="src is a required parameter"):
            preprocess_generate_masks({})

    def test_normalize_src_path_returns_only_a_string_or_a_list(self):
        from spacr.utils import normalize_src_path

        assert normalize_src_path("/data/plate1") == "/data/plate1"
        assert normalize_src_path(["/a", "/b"]) == ["/a", "/b"]
        # A stringified list -- what a settings CSV round trip produces --
        # comes back as a real list rather than as a path with brackets.
        assert normalize_src_path("['/a', '/b']") == ["/a", "/b"]
        with pytest.raises(ValueError, match="Invalid type for 'src'"):
            normalize_src_path(3)


# ---------------------------------------------------------------------------
# spacr.convert -- the map always has a target column
# ---------------------------------------------------------------------------

def _minimal_map(path, *, with_prcf=True):
    row = {
        "target": "plate1_A01_T0001F001L01A01Z01C01.tif",
        "source": "/raw/whatever.tif",
        "plate": "plate1", "well": "A01", "field": 1,
        "channel": 1, "z": 1, "t": 1,
    }
    if with_prcf:
        row["prcf"] = "plate1_A_01_1"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


class TestTheConversionMapIsAlwaysIndexedOnTarget:
    """``if 'target' in frame.columns:`` in ``populate_db_from_map``.

    ``frame = read_map(map_path)`` on the line above refuses any file
    missing one of ``_REQUIRED_MAP_COLUMNS``, and ``target`` is the first
    of them. So by the time the index is created the column is there and
    the false side cannot be taken -- the same re-check pattern as the
    rest of this file, and one worth keeping honest because ``prcf``
    beside it IS optional and does need its guard.
    """

    def test_a_map_without_target_never_reaches_the_index(self, tmp_path):
        from spacr.convert import populate_db_from_map
        from spacr.errors import ConfigurationError

        path = tmp_path / "no_target.csv"
        frame = pd.read_csv(_minimal_map(tmp_path / "full.csv"))
        frame.drop(columns=["target"]).to_csv(path, index=False)

        with pytest.raises(ConfigurationError, match="missing column"):
            populate_db_from_map(str(tmp_path / "x.db"), str(path))

        assert not (tmp_path / "x.db").exists() or _indexes(
            tmp_path / "x.db") == [], \
            "nothing may be written from a map that was refused"

    def test_a_map_that_passes_read_map_is_indexed_on_target(self, tmp_path):
        from spacr.convert import CONVERSION_TABLE, populate_db_from_map

        db = tmp_path / "m.db"
        assert populate_db_from_map(
            str(db), str(_minimal_map(tmp_path / "map.csv"))) == 1

        assert f"idx_{CONVERSION_TABLE}_target" in _indexes(db)


def _indexes(db_path):
    if not os.path.isfile(db_path):
        return []
    connection = sqlite3.connect(str(db_path))
    try:
        return [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# spacr.seg_qc -- a finding whose fields have no names
# ---------------------------------------------------------------------------

class TestAFlagFindingWithNothingToName:
    """``if located: ... elif named_fields: ...`` in ``_flag_findings``."""

    @staticmethod
    def _finding(field):
        from spacr.seg_qc import FLAG_EMPTY, FieldQC, _flag_findings

        qc = FieldQC(field=field, object_type="cell", n_objects=0,
                     flags=[FLAG_EMPTY], severity="fail")
        found = _flag_findings([qc], max_named=3)
        assert len(found) == 1, f"one flag, one finding; got {found}"
        return found[0]

    def test_a_field_with_no_name_at_all_falls_back_to_the_project(self):
        """A mask whose stem parses to nothing still has to be reported.

        Neither a well nor a field name can be printed, and inventing one
        would be worse than saying where it is not known.
        """
        headline = self._finding("").headline

        assert "this project" in headline, (
            "with no plate, no well and no field name the finding names the "
            f"project; got {headline!r}")
        assert "wells" not in headline and "fields" not in headline

    def test_a_named_well_is_named(self):
        headline = self._finding("plate1_A01_1").headline

        assert "plate1" in headline and "wells A01" in headline, headline

    def test_a_field_name_with_no_well_is_named_as_a_field(self):
        """The middle branch: a name, but not one an address parses out of."""
        headline = self._finding("weird").headline

        assert "fields weird" in headline, headline
        assert "wells" not in headline


# ---------------------------------------------------------------------------
# spacr.foreign -- importing into a project that already converted
# ---------------------------------------------------------------------------

class TestImportingAConversionMapIntoAnExistingProject:
    """``if staging in names:`` in ``_populate_conversion_map``."""

    @staticmethod
    def _seed(db_path, target):
        """A destination that already carries a conversion_map row."""
        from spacr import convert as cv

        frame = pd.DataFrame([{
            "target": target, "source": "/old/raw.tif", "plate": "plate0",
            "well": "A01", "field": 1, "channel": 1, "z": 1, "t": 1,
        }])
        connection = sqlite3.connect(str(db_path))
        try:
            frame.to_sql(cv.CONVERSION_TABLE, connection,
                         if_exists="replace", index=False)
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _targets(db_path):
        from spacr import convert as cv

        connection = sqlite3.connect(str(db_path))
        try:
            return sorted(row[0] for row in connection.execute(
                f'SELECT target FROM "{cv.CONVERSION_TABLE}"'))
        finally:
            connection.close()

    def test_a_clean_destination_keeps_the_rows_it_had_and_gains_the_new(
            self, tmp_path):
        from spacr import foreign

        db = tmp_path / "measurements.db"
        self._seed(db, "old_row.tif")
        map_path = _minimal_map(tmp_path / "map.csv")

        foreign._populate_conversion_map(str(db), str(map_path))

        assert self._targets(db) == sorted(
            ["old_row.tif",
             "plate1_A01_T0001F001L01A01Z01C01.tif"]), (
            "an import must merge into the project's own provenance, not "
            "replace it")
        assert foreign._CONVERSION_STAGING not in _table_names(db), \
            "the staging table is dropped once it has been merged"

    def test_a_staging_table_left_by_a_crash_is_cleared_first(self, tmp_path):
        """A previous import that died mid-merge leaves scratch rows behind.

        They must not be merged in a second time: the drop at the top is
        what makes a retry produce the same table as a first attempt.
        """
        from spacr import foreign

        db = tmp_path / "measurements.db"
        self._seed(db, "old_row.tif")
        connection = sqlite3.connect(str(db))
        try:
            pd.DataFrame([{"target": "ghost_from_a_crash.tif",
                           "source": "/raw/ghost.tif"}]).to_sql(
                foreign._CONVERSION_STAGING, connection,
                if_exists="replace", index=False)
            connection.commit()
        finally:
            connection.close()
        assert foreign._CONVERSION_STAGING in _table_names(db)

        foreign._populate_conversion_map(
            str(db), str(_minimal_map(tmp_path / "map.csv")))

        assert "ghost_from_a_crash.tif" not in self._targets(db), (
            "rows a crashed import staged must not be adopted by the next "
            f"one; got {self._targets(db)}")
        assert self._targets(db) == sorted(
            ["old_row.tif", "plate1_A01_T0001F001L01A01Z01C01.tif"])


def _table_names(db_path):
    connection = sqlite3.connect(str(db_path))
    try:
        return [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# spacr.crops -- a crop is always masked by a region
# ---------------------------------------------------------------------------

class TestACropIsAlwaysMaskedByARegion:
    """``if region is not None:`` in ``_crop_from_field``.

    ``_region_for`` returns ``(centroid, bounds, region)`` and its
    docstring states the contract: "``region_mask`` is always the boolean
    region restricted to ``region_bounds``". Both of its branches build
    one -- ``np.ones(...)`` for the bounding-box mode, ``window ==
    spec.label`` for the object mode, which additionally raises
    ``LabelMissing`` if that came back all-false. There is no path on
    which it returns ``None``, so the guard is another re-check of what
    the call above guarantees.

    What the two modes actually DO is the thing worth pinning, and it is
    exactly what the region carries: outside the object is zeroed in the
    object mode and kept in the bounding-box mode.
    """

    @staticmethod
    def _merged(tmp_path):
        """One 32x32 field: two graded intensity channels and a cell mask.

        Graded on purpose. ``_normalize_to_dtype`` stretches between the
        percentiles of the NON-ZERO pixels, and a flat channel gives it a
        degenerate range it fills with the constant -- which would put the
        background back after the mask had removed it.
        """
        yy, xx = np.indices((32, 32))
        merged = np.zeros((32, 32, 3), dtype=np.uint16)
        merged[..., 0] = (yy * 32 + xx + 100).astype(np.uint16)
        merged[..., 1] = (xx * 32 + yy + 100).astype(np.uint16)
        merged[12:20, 12:20, 2] = 1                 # one square cell, label 1
        path = tmp_path / "merged" / "fov1.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, merged)
        return path

    def test_the_object_mode_zeroes_everything_outside_the_label(
            self, tmp_path):
        from spacr.crops import extract_crop

        crop = extract_crop(str(self._merged(tmp_path)), "cell", 1,
                            size=(16, 16), channels=(0, 1),
                            mask_dims={"cell": 2},
                            use_bounding_box=False, normalize=None)

        # Two channels come back as three: the PNG path pads a two-channel
        # crop to RGB with a zero plane before writing, and this is the
        # pre-write array.
        assert crop.shape == (16, 16, 3)
        assert int(crop[..., 2].max()) == 0, "the pad plane is empty"
        # The object is 8x8 inside a 16x16 window, so the whole outer ring
        # is background and the region masks it away.
        assert int(crop[0, :, 0].max()) == 0, \
            "the top row of the window is outside the cell"
        assert int(crop[:, 0, 0].max()) == 0
        assert int((crop[..., 0] == 0).sum()) >= 192, (
            "at least the 256 - 64 background pixels must be zero; got "
            f"{int((crop[..., 0] == 0).sum())}")
        assert int(crop[8, 8, 0]) > 0, "the middle of the cell survives"

    def test_the_bounding_box_mode_keeps_the_whole_window(self, tmp_path):
        """The contrast: an all-ones region masks nothing away."""
        from spacr.crops import extract_crop

        crop = extract_crop(str(self._merged(tmp_path)), "cell", 1,
                            size=(16, 16), channels=(0, 1),
                            mask_dims={"cell": 2},
                            use_bounding_box=True, normalize=None)

        assert crop.shape == (16, 16, 3)
        assert int(crop[0, :, 0].max()) > 0, (
            "with a bounding-box region the background of the window is "
            "kept, not masked off")
        assert int((crop[..., 0] == 0).sum()) <= 1, (
            "only the dimmest pixel lands on zero, from the percentile "
            f"stretch; got {int((crop[..., 0] == 0).sum())}")


# ---------------------------------------------------------------------------
# spacr.layers -- locking makes its own leader
# ---------------------------------------------------------------------------

class TestLockingAPanelAlwaysHasALeader:
    """``if leader is not None:`` in ``CanvasLink.lock``.

    ``lock`` does ``self._locked[str(key)] = True`` and only then calls
    ``self._leader()``, which returns the first canvas in ``_canvases``
    whose ``_locked`` entry is true. ``canvas = self[key]`` on the line
    before has already established that ``key`` IS in ``_canvases``, so
    the panel just locked is itself a candidate: ``_leader()`` cannot
    come back ``None`` here. (``add`` calls ``_leader()`` BEFORE
    inserting, which is why the same guard is load-bearing there and
    dead here.)

    Pinned instead: what locking does to a panel that had drifted.
    """

    @staticmethod
    def _canvas(origin, step=(1.0, 1.0), shape=(8, 8)):
        from spacr.layers import Canvas

        return Canvas(origin=origin, step=step, shape=shape)

    def test_locking_the_only_unlocked_panel_moves_it_onto_the_others(self):
        from spacr.layers import CanvasLink

        link = CanvasLink({"a": self._canvas((0.0, 0.0)),
                           "b": self._canvas((0.0, 0.0))})
        link.unlock("b")
        link.pan(5.0, 7.0, key="b")
        assert link["b"].origin != link["a"].origin, \
            "the unlocked panel has to have actually drifted"

        link.lock("b")

        assert link.is_locked("b")
        assert link["b"].origin == link["a"].origin, (
            "locking puts the panel back on the shared window now, not at "
            f"the next pan; got {link['b'].origin} vs {link['a'].origin}")

    def test_locking_the_last_panel_when_none_are_locked_leads_itself(self):
        """Every panel unlocked: the one being locked becomes the leader.

        This is the case the ``leader is not None`` guard looks like it is
        for, and it is why the guard cannot fire: the panel is marked
        locked before the leader is looked up, so it finds itself and
        aligning to itself is a no-op.
        """
        from spacr.layers import CanvasLink

        # `add(locked=False)` keeps b's own window: a panel joining a link
        # unlocked does not adopt the shared one.
        link = CanvasLink({"a": self._canvas((1.0, 2.0))})
        link.add("b", self._canvas((3.0, 4.0)), locked=False)
        link.unlock("a")

        link.lock("b")

        assert link["b"].origin == (3.0, 4.0), (
            "the first panel locked keeps its own window and becomes the "
            f"leader; got {link['b'].origin}")
        assert link["a"].origin == (1.0, 2.0), \
            "a panel that is still unlocked is not moved"
        # ...and b really is the leader now: a panel joining locked adopts
        # b's window rather than keeping its own.
        link.add("c", self._canvas((9.0, 9.0)))
        assert link["c"].origin == (3.0, 4.0), (
            "the newly locked panel is the leader the next one follows; got "
            f"{link['c'].origin}")
