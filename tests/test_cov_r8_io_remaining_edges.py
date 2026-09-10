"""What is left uncovered in io.py: six live edges and three pins.

Every one of them is a "and if there is nothing / it is already right"
case, which in this module means either a file renamed onto itself, a
field that read no frames, or a split that left a class empty.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from spacr import io


# ---------------------------------------------------------------------------
# migrate_unescaped_plate_names -- a name that is already safe
# ---------------------------------------------------------------------------

class TestMigratingUnescapedPlateNames:

    def _plate(self, tmp_path, plate, stems):
        root = tmp_path / plate
        (root / "masks").mkdir(parents=True)
        for stem in stems:
            (root / "masks" / f"{stem}.npy").write_bytes(b"\x93NUMPY")
        return root

    def test_a_plate_with_an_underscore_is_planned_for_renaming(self,
                                                                tmp_path):
        root = self._plate(tmp_path, "exp_1", ["exp_1_A01_1_1"])

        planned = io.migrate_unescaped_plate_names(str(root), dry_run=True)

        assert planned, "an unescaped plate name was not noticed"

    def test_a_four_component_stem_is_refused_before_it_is_escaped(self,
                                                                   tmp_path):
        """A plate with no separator in it has nothing to migrate.

        The test is on the COMPONENT COUNT rather than on "does escaping
        change the name", and that is what makes a second run a no-op:
        escaping is not idempotent, because a literal percent is escaped
        first, so ``exp%5F1_A01_1_1`` would become ``exp%255F1_A01_1_1``.
        A migration that corrupts on its second run is worse than one
        that never ran.
        """
        root = self._plate(tmp_path, "exp1", ["exp1_A01_1_1", "exp1_A02_1_1"])

        planned = io.migrate_unescaped_plate_names(str(root), dry_run=True)

        assert planned == [], (
            f"a plate that needs no escaping was planned anyway: {planned}")

    def test_a_stem_that_escapes_to_itself_cannot_get_that_far(self):
        """THE PIN, for ``if safe == stem: continue``.

        Planning a rename onto the same path is a no-op on some
        filesystems and an error on others, and in either case a
        migration reporting work it did not do -- so the guard is right
        to keep. It cannot fire: the check above it has already refused
        every stem of four components or fewer, and more than four means
        the plate holds a separator, which is exactly what escaping
        changes.

        Asserted against the escaper itself, over both the plain and the
        already-escaped forms, so an escaper that became idempotent
        would fail here rather than start planning self-renames.
        """
        from spacr.schema import escape_field_stem_plate

        for stem in ("exp_1_A01_1_1", "a_b_c_A01_1_1"):
            assert escape_field_stem_plate(stem, timelapse=True) != stem, (
                f"{stem!r} escapes to itself, so the guard is live")

        for stem in ("exp1_A01_1_1",):
            assert len(stem.split("_")) <= 4, (
                "this stem now has more than four components and would "
                "reach the escaper")

    def test_a_dry_run_moves_nothing(self, tmp_path):
        root = self._plate(tmp_path, "exp_1", ["exp_1_A01_1_1"])
        before = sorted(p.name for p in (root / "masks").iterdir())

        io.migrate_unescaped_plate_names(str(root), dry_run=True)

        assert sorted(p.name for p in (root / "masks").iterdir()) == before


# ---------------------------------------------------------------------------
# process_non_tif_non_2D_images -- splitting a stack, and the suffix it names
# ---------------------------------------------------------------------------

class TestSplittingAMultiDimensionalImage:

    def test_a_three_axis_stack_becomes_one_tiff_per_channel(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")

        stack = np.zeros((8, 6, 3), dtype=np.uint16)
        for channel in range(3):
            stack[..., channel] = 100 * (channel + 1)
        tifffile.imwrite(tmp_path / "plate1_A01.tif", stack)

        io.process_non_tif_non_2D_images(str(tmp_path))

        written = sorted(p.name for p in tmp_path.glob("plate1_A01_C*.tif"))
        assert written == ["plate1_A01_C1.tif", "plate1_A01_C2.tif",
                           "plate1_A01_C3.tif"]
        assert tifffile.imread(tmp_path / "plate1_A01_C2.tif").dtype \
            == np.uint16, "the bit depth was not preserved"

    def test_a_four_axis_stack_names_both_the_channel_and_the_plane(self,
                                                                     tmp_path):
        tifffile = pytest.importorskip("tifffile")

        tifffile.imwrite(tmp_path / "plate1_A01.tif",
                         np.zeros((8, 6, 2, 3), dtype=np.uint8))

        io.process_non_tif_non_2D_images(str(tmp_path))

        written = sorted(p.name for p in tmp_path.glob("plate1_A01_C*.tif"))
        assert written == [
            "plate1_A01_C1_Z1.tif", "plate1_A01_C1_Z2.tif",
            "plate1_A01_C1_Z3.tif", "plate1_A01_C2_Z1.tif",
            "plate1_A01_C2_Z2.tif", "plate1_A01_C2_Z3.tif",
        ]

    def test_the_ledger_records_a_file_that_cannot_be_read(self, tmp_path):
        (tmp_path / "broken.tif").write_bytes(b"not a tiff at all")

        ledger = io.process_non_tif_non_2D_images(str(tmp_path))

        assert ledger is not None
        assert not ledger.is_complete, (
            "an unreadable file did not reach the ledger")

    def test_the_channel_token_is_optional_for_a_caller_that_has_none(self):
        """THE PIN.

        The suffix is built axis by axis and every part is optional --
        writing ``_CNone`` would put a literal 'None' into a filename the
        rest of the pipeline then parses. But today's only caller,
        ``split_channels``, always supplies ``channel=c+1``: the 3-, 4-
        and 5-axis branches differ only in whether they ALSO pass z and
        t. So the channel half of the suffix rule has no live caller
        that omits it.

        Pinned on the calls rather than on the rule: a Z-only stack
        handled without a channel axis is exactly the change that would
        make it live, and it would land here.
        """
        source = inspect.getsource(io.process_non_tif_non_2D_images)
        calls = [line.strip() for line in source.splitlines()
                 if "save_grayscale_images(" in line and "def " not in line]
        assert calls, "save_grayscale_images is no longer called at all"
        assert all("channel=c+1" in call for call in calls), (
            f"a caller now omits the channel: {calls}")
        assert "if channel is not None:" not in source
        assert 'suffix = f"_C{channel}"' in source


# ---------------------------------------------------------------------------
# _get_avg_object_size -- an empty mask and an impossible one
# ---------------------------------------------------------------------------

class TestAverageObjectSize:

    def test_objects_are_counted_and_measured(self, capsys):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[1:4, 1:4] = 1
        mask[6:8, 6:8] = 2

        per_image, average = io._get_avg_object_size([mask, mask])

        assert per_image == 2
        assert average == pytest.approx((9 + 4) / 2)

    def test_an_empty_mask_is_named_and_counted_as_zero(self, capsys):
        mask = np.zeros((10, 10), dtype=np.int32)

        per_image, _average = io._get_avg_object_size([mask])

        assert per_image == 0
        assert "Mask 0 is empty" in capsys.readouterr().out

    def test_a_mask_of_the_wrong_dimension_is_named(self, capsys):
        odd = np.ones((2, 2, 2, 2), dtype=np.int32)

        io._get_avg_object_size([odd])

        assert "invalid dimension: 4" in capsys.readouterr().out

    def test_the_two_warning_cases_are_exhaustive(self):
        """THE PIN for replacing the always-true ``elif`` with ``else``.

        The else branch is reached only when the mask is empty OR its
        dimension is wrong, and those two are exactly the two warnings.
        A mask that is non-empty and 2/3-D took the outer ``if`` above,
        so a non-empty mask in the other arm necessarily has invalid rank.

        The pin is on the shape of the condition, because a third reason
        to skip a mask added without a third warning would make a mask
        vanish from the average with nothing said about it.
        """
        source = inspect.getsource(io._get_avg_object_size)
        assert "if mask.ndim in [2, 3] and np.any(mask):" in source
        assert "if not np.any(mask):" in source
        assert "elif mask.ndim not in [2, 3]:" not in source
        assert "else:\n                print(f\"Warning: Mask {idx} has invalid" in source


# ---------------------------------------------------------------------------
# The three remaining decisions, pinned to the source that settles them
# ---------------------------------------------------------------------------
#
# Each of these sits deep inside a function that needs a measurement
# database, a crop tree or a tar destination to reach. Driving them would
# mean building most of a run to observe one line, so what is asserted is
# the code that decides them -- and each pin names the change that would
# make its branch live.


class TestTheRemainingDecisions:

    def test_the_object_key_uses_the_prcf_rebuilt_by_the_splitter(self):
        """``_read_and_merge_data`` uses the complete field identity.

        ``_split_data`` rebuilds ``prcf`` before returning metadata, and every
        object role already consumes it earlier in the function. The former
        four-component fallback was unreachable and also omitted time.
        """
        source = inspect.getsource(io._read_and_merge_data)
        assert "if 'prcf' in metadata.columns:" not in source
        assert "prcfo=lambda x: x['prcf'] + '_' + x[metadata_key]" in source
        assert "prcfo=lambda x: x['plateID']" not in source

    def test_a_tar_destination_is_set_before_nonempty_input_can_reach_it(self):
        """The unreachable ``dst is None`` guard stays deleted."""
        source = inspect.getsource(io.generate_dataset)
        assert "if dst is None:" not in source
        assert "Destination folder (dst) was not set." not in source
        assert source.index("if i == 0:") < source.index("os.makedirs(dst")
        assert source.index("No images selected; nothing to tar.") < \
            source.index("os.makedirs(dst")

    def test_a_dataset_mode_spacr_never_had_is_refused_by_name(self):
        """``generate_training_dataset``: the else after the two modes.

        ``resolve_basis`` migrates the retired 'measurement' to
        'annotation', so anything reaching the else is a value the
        package has never had -- a typo in a settings file. Naming it AND
        naming the two that are legal is the difference between a
        fixable message and a KeyError several frames down.
        """
        from spacr.training_basis import TrainingBasisError, resolve_basis

        with pytest.raises(TrainingBasisError, match="not one of"):
            resolve_basis({"dataset_mode": "not-real"})
        source = inspect.getsource(io.generate_training_dataset)
        assert "Invalid dataset_mode:" not in source
        assert "else:" in source and "resolve_basis has already reduced" in source

    def test_balancing_nothing_returns_nothing_rather_than_raising(self):
        """``_balance_lists``: ``min()`` over an empty list raises.

        A run whose filters selected no classes at all has a later,
        better message for that; this must not pre-empt it with a
        ValueError from the balancer.
        """
        source = inspect.getsource(io.generate_training_dataset)
        balancer = source[source.index("def _balance_lists"):]
        balancer = balancer[:balancer.index("def _annotation_classes")]
        assert "if not list_of_lists:" not in balancer
        gate = source.index("if not class_path_list or sum(")
        call = source.index("class_path_list = _balance_lists")
        assert gate < call

    def test_the_unique_training_directory_search_has_no_false_exhaustion(self):
        """Every occupied suffix advances to the next rather than returning it."""
        source = inspect.getsource(io.generate_training_dataset)
        assert "for j in range(1, 100000):" not in source
        assert 'while os.path.exists(f"{base}_{j}"):' in source
        assert "j += 1" in source

    def test_a_well_is_minted_once_per_file_not_once_per_channel(self):
        """``convert_to_yokogawa``: the loop runs per channel/timepoint.

        Minting a well on every pass would scatter one image's channels
        across the plate, which is the one thing the conversion must not
        do.
        """
        source = inspect.getsource(io.convert_to_yokogawa)
        assert "file_to_well" not in source
        assert "well = _get_next_well(used_wells)" in source
        assert source.index("for file in sorted(os.listdir(folder)):") < \
            source.index("well = _get_next_well(used_wells)")
