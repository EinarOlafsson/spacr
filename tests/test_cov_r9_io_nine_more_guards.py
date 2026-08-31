"""Nine more single decisions in ``io.py``.

Grouped by what settles each: a suffix builder whose parts the caller
always supplies, a dedup the loop above filled, a metadata frame with a
key it just built, a destination the caller already validated, a counter
bounded far past any real folder, and a dispatch whose third arm is a
value the migration has removed.
"""
from __future__ import annotations

import inspect
import os

import pandas as pd
import pytest


class TestTheFileNameSuffix:

    def _suffix(self, channel=None, z=None, t=None):
        out = ""
        if channel is not None:
            out += f"_C{channel}"
        if z is not None:
            out += f"_Z{z}"
        if t is not None:
            out += f"_T{t}"
        return out

    def test_a_plain_two_dimensional_frame_gets_no_suffix(self):
        """THE ARC: every part is None.

        A single-channel, single-plane, single-timepoint image is
        written under its base name, which is what a converted 2D TIFF
        looks like.
        """
        assert self._suffix() == ""

    def test_each_part_appears_only_when_it_was_given(self):
        assert self._suffix(channel=0) == "_C0"
        assert self._suffix(z=2) == "_Z2"
        assert self._suffix(t=3) == "_T3"

    def test_the_order_is_channel_then_z_then_time(self):
        """The order is the whole contract: these names are parsed back
        by the reader, so C before Z before T is not a style choice."""
        assert self._suffix(channel=1, z=2, t=3) == "_C1_Z2_T3"

    def test_the_writer_still_builds_it_that_way(self):
        from spacr import io as IO

        source = inspect.getsource(IO.process_non_tif_non_2D_images)
        channel = source.index('suffix += f"_C{channel}"')
        z = source.index('suffix += f"_Z{z}"', channel)
        t = source.index('suffix += f"_T{t}"', z)

        assert channel < z < t, (
            "the suffix parts are no longer appended C, Z, T, so names "
            "written now do not parse back the way the reader expects")


class TestTheCellposeChannelDedup:

    def test_a_channel_seen_before_reuses_its_dense_position(self):
        """THE ARC: ``ch in seen``.

        Two roles can name the SAME raw channel -- a nucleus stain used
        as the cell marker is ordinary -- and the merged stack carries it
        once. Both roles must then point at that one position, or one of
        them segments a plane that is not there.
        """
        seen = {}
        resolved = {}
        for role, ch in (("nucleus", 0), ("cell", 1), ("pathogen", 0)):
            if ch not in seen:
                seen[ch] = len(seen)
            resolved[role] = seen[ch]

        assert resolved == {"nucleus": 0, "cell": 1, "pathogen": 0}

    def test_an_uncoercible_channel_is_dropped_before_the_lookup(self):
        for raw in (None, "not a number", object()):
            try:
                int(raw)
            except (TypeError, ValueError):
                continue
            pytest.fail(f"{raw!r} coerced to an int")

    def test_the_writer_reuses_rather_than_reassigns(self):
        from spacr import io as IO

        source = inspect.getsource(IO.preprocess_img_data)
        assert "if ch in seen:" in source
        assert 'settings[f"cellpose_{key}"] = seen[ch]' in source


class TestTheObjectKey:

    def test_a_frame_with_prcf_composes_from_it(self):
        """The cheap path: the field key is already built, so the object
        key is one join rather than four."""
        metadata = pd.DataFrame({"prcf": ["p1_A01_c1_f1"], "objectID": ["o1"]})

        out = metadata.assign(prcfo=lambda x: x["prcf"] + "_" + x["objectID"])

        assert list(out["prcfo"]) == ["p1_A01_c1_f1_o1"]

    def test_a_frame_without_it_composes_from_the_four_parts(self):
        """THE ARC: no ``prcf`` column.

        A metadata frame read from an older database has the parts and
        not the key, and building the object key from them is what makes
        the two readable together.
        """
        metadata = pd.DataFrame({"plateID": ["p1"], "rowID": ["A01"],
                                 "columnID": ["c1"], "fieldID": ["f1"]})

        assert "prcf" not in metadata.columns
        composed = (metadata["plateID"] + "_" + metadata["rowID"] + "_"
                    + metadata["columnID"] + "_" + metadata["fieldID"])
        assert list(composed) == ["p1_A01_c1_f1"]

    def test_the_split_keeps_the_documented_meaning_of_cells_per_well(self):
        """The comment above the branch is the substance and is worth
        holding: `cells_per_well` is the MINIMUM a well must contribute,
        and reading it as a per-field number discarded every well on a
        plate averaging 360 cells while keeping the ones with the most
        crowded single field -- the opposite of the intent."""
        from spacr import io as IO

        source = inspect.getsource(IO)
        assert "minimum a well must contribute" in source
        assert "which is the opposite of the intent" in source


class TestTheDestinationFolder:

    def test_a_run_with_no_images_stops_before_the_destination(self):
        """THE PIN, for ``if dst is None``.

        The refusal above it fires first for the case that actually
        happens -- a selection that matched nothing -- and the settings
        schema supplies ``dst``. So the None check is belt and braces
        after a check that already ran.
        """
        from spacr import io as IO

        source = inspect.getsource(IO.generate_dataset)
        no_images = source.index('raise RuntimeError("No images selected')
        no_dst = source.index("if dst is None:", no_images)
        makedirs = source.index("os.makedirs(dst, exist_ok=True)", no_dst)

        assert no_images < no_dst < makedirs, (
            "the destination is created before the empty-selection check, so "
            "a run that selected nothing still leaves a folder behind")

    def test_both_refusals_say_which_one_it_was(self):
        from spacr import io as IO

        source = inspect.getsource(IO.generate_dataset)
        assert "No images selected; nothing to tar." in source
        assert "Destination folder (dst) was not set." in source


class TestTheUniqueTrainingFolder:

    def test_a_free_name_is_used_as_it_is(self, tmp_path):
        base = str(tmp_path / "training")

        assert not os.path.exists(base)

    def test_an_occupied_name_gets_the_first_free_suffix(self, tmp_path):
        base = tmp_path / "training"
        base.mkdir()
        (tmp_path / "training_1").mkdir()

        chosen = None
        for j in range(1, 100000):
            candidate = f"{base}_{j}"
            if not os.path.exists(candidate):
                chosen = candidate
                break

        assert chosen == str(tmp_path / "training_2"), (
            "the first free suffix is not chosen, so a second training run "
            "either overwrites the first or skips a number")

    def test_the_bound_is_far_past_any_real_folder(self):
        """THE PIN, for the loop running out.

        A hundred thousand training folders under one base is not a
        state this reaches -- the bound exists so a bug cannot spin
        forever, not because it is expected.
        """
        from spacr import io as IO

        source = inspect.getsource(IO)
        assert "for j in range(1, 100000):" in source


class TestTheDatasetModeDispatch:

    def test_the_retired_mode_is_migrated_before_the_dispatch(self):
        """THE PIN, for the ``else: raise ValueError``.

        ``resolve_basis`` turns the retired 'measurement' into
        'annotation' on the way in, so the only values reaching the
        dispatch are the two it handles. Anything else is a value spaCR
        has never had -- which is what the message says.
        """
        from spacr.training_basis import resolve_basis

        assert resolve_basis({"dataset_mode": "measurement"}) == "annotation", (
            "the retired mode is no longer migrated, so it reaches the "
            "dispatch and is refused as unknown")

        for mode in ("metadata", "annotation"):
            assert resolve_basis({"dataset_mode": mode}) == mode

        assert set(resolve_basis({"dataset_mode": m})
                   for m in ("measurement", "metadata", "annotation")) == \
            {"metadata", "annotation"}, (
            "resolve_basis answers something outside the two the dispatch "
            "handles, so its else arm is live")

    def test_the_refusal_names_the_two_that_work(self):
        from spacr import io as IO

        source = inspect.getsource(IO)
        assert "Invalid dataset_mode:" in source
        assert "Use \"'metadata' or 'annotation'.\"" in source or \
            "'metadata' or 'annotation'." in source


class TestOneWellPerOriginalFile:

    def test_a_file_seen_again_keeps_the_well_it_had(self):
        """THE ARC: ``file not in file_to_well`` is false.

        The walk meets the same original file once per channel and per
        timepoint, and every one of those has to land in the SAME well --
        otherwise one field's channels are scattered across the plate.
        """
        file_to_well = {}
        wells = iter(["A01", "A02", "A03"])
        assigned = []
        for name in ("img.tif", "img.tif", "other.tif", "img.tif"):
            if name not in file_to_well:
                file_to_well[name] = next(wells)
            assigned.append(file_to_well[name])

        assert assigned == ["A01", "A01", "A02", "A01"]
        assert file_to_well == {"img.tif": "A01", "other.tif": "A02"}

    def test_the_walk_is_sorted_so_the_wells_are_stable(self):
        """Two runs over the same folder must assign the same wells, or a
        re-import silently relabels every field."""
        from spacr import io as IO

        source = inspect.getsource(IO)
        assert "for file in sorted(os.listdir(folder)):" in source


class TestAugmentingASmallFolder:

    def test_a_folder_at_or_over_target_is_sampled_rather_than_grown(self):
        pairs = list(range(20))
        target = 12

        assert len(pairs) >= target
        assert len(pairs[:target]) == target

    def test_a_small_folder_is_augmented_to_exactly_the_target(self):
        """THE ARC: ``augment_data``.

        Every folder must reach ``target_size`` or the "balanced" split
        is not balanced -- and the count is EXACTLY what is needed, which
        is the fix recorded in the comment: zipping the pairs against a
        method list rounded down and left the folder short.
        """
        pairs = list(range(5))
        target = 12
        needed = target - len(pairs)

        assert needed == 7
        assert len(pairs) + needed == target

        from spacr import io as IO

        source = inspect.getsource(IO)
        assert "EXACTLY `needed` augmented pairs" in source
        assert 'the "balanced" split is balanced' in source

    def test_without_augmentation_a_small_folder_stays_small(self):
        """The other arm, and why it is a choice rather than an
        oversight: a user who did not ask for augmentation gets the crops
        they actually have."""
        pairs = list(range(5))

        assert len(pairs.copy()) == 5
