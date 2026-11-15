"""What is left uncovered in io.py: six live edges and three pins.

Every one of them is a "and if there is nothing / it is already right"
case, which in this module means either a file renamed onto itself, a
field that read no frames, or a split that left a class empty.
"""
from __future__ import annotations

import inspect
import os

import numpy as np

from spacr import io


# ---------------------------------------------------------------------------
# process_non_tif_non_2D_images -- splitting a stack, and the suffix it names
# ---------------------------------------------------------------------------

class TestSplittingAMultiDimensionalImage:


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

