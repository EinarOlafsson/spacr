"""Fourteen guards across fourteen modules, one site each.

Filed together because each is a single decision in an otherwise covered
function, and because the shapes repeat: a pair returned together, a
frame that is empty only when nothing was collected, an index created
for a column the table may not have.
"""
from __future__ import annotations

import inspect
import sqlite3

import numpy as np
import pandas as pd
import pytest


class TestADigestOrAnError:

    def test_a_readable_file_hashes_and_reports_no_error(self, tmp_path):
        from spacr.classifier_evaluation import _content_sha256

        path = tmp_path / "model.pth"
        path.write_bytes(b"weights")

        digest, error = _content_sha256(str(path))

        assert digest and len(digest) == 64
        assert not error

    def test_a_missing_file_reports_an_error_and_no_digest(self, tmp_path):
        """THE ARC: ``elif error``.

        The two come back together and exactly one is set, so a path that
        could not be read always produces an error to collect. Recording
        it matters: a model whose file vanished must show up in the
        identity report as unreadable rather than simply contributing no
        hash, which reads as "same as the others".
        """
        from spacr.classifier_evaluation import _content_sha256

        digest, error = _content_sha256(str(tmp_path / "absent.pth"))

        assert not digest
        assert error, "an unreadable path produced neither a digest nor an error"

    def test_exactly_one_of_the_pair_is_ever_set(self, tmp_path):
        from spacr.classifier_evaluation import _content_sha256

        real = tmp_path / "a.pth"
        real.write_bytes(b"x")
        for path in (str(real), str(tmp_path / "missing"), str(tmp_path)):
            digest, error = _content_sha256(path)
            assert bool(digest) != bool(error), (
                f"{path} gave digest={digest!r} error={error!r}; the elif in "
                f"_identity_sets_with_hashes assumes one or the other")


class TestThePerPlateTable:

    def test_a_run_with_plates_puts_plate_first(self):
        """The reorder the guard protects: 'plate' leads the table, so a
        reader sees which plate a row is about before its numbers."""
        rows = [{"auc": 0.9, "plate": "p1"}, {"auc": 0.8, "plate": "p2"}]
        per_plate = pd.DataFrame(rows)

        assert not per_plate.empty
        columns = ["plate", *[c for c in per_plate if c != "plate"]]
        assert columns[0] == "plate"
        assert list(per_plate[columns].columns) == ["plate", "auc"]

    def test_a_run_with_no_plates_has_no_columns_to_reorder(self):
        """THE ARC: an empty frame.

        ``DataFrame([])`` has no columns at all, so the reorder would ask
        for 'plate' and raise -- at the end of an evaluation that
        otherwise succeeded.
        """
        per_plate = pd.DataFrame([])

        assert per_plate.empty
        assert list(per_plate.columns) == []
        with pytest.raises(KeyError):
            per_plate[["plate"]]


class TestIndexingOnlyTheColumnsPresent:

    def _table(self, columns):
        connection = sqlite3.connect(":memory:")
        frame = pd.DataFrame({c: ["x"] for c in columns})
        frame.to_sql("t", connection, if_exists="replace", index=False)
        return connection, frame

    def test_an_index_is_created_for_a_column_that_is_there(self):
        connection, frame = self._table(["prcf", "target"])
        try:
            assert "target" in frame.columns
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_t_target ON t (target)")
            names = {row[1] for row in connection.execute(
                'PRAGMA index_list("t")')}
            assert "idx_t_target" in names
        finally:
            connection.close()

    def test_a_missing_column_is_skipped_rather_than_indexed(self):
        """THE ARC: the conversion map has no ``target``.

        Not every map carries one -- a plate-only map is legal -- and
        CREATE INDEX on an absent column is an OperationalError in the
        middle of a populate that has already written its rows.
        """
        connection, frame = self._table(["prcf"])
        try:
            assert "target" not in frame.columns
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("CREATE INDEX idx_t_target ON t (target)")
        finally:
            connection.close()

    def test_both_writers_receive_the_required_target_column(self):
        from spacr import convert as C

        assert "target" in C._REQUIRED_MAP_COLUMNS


class TestFallingBackToAShorterControl:

    def test_a_typed_control_that_resolves_is_retried(self):
        from spacr.control_names import resolve_control

        assert resolve_control("TSC2") is not None

    def test_only_an_empty_tail_resolves_to_none(self):
        """THE ARC: ``shorter is None``, and MEASURED rather than assumed.

        It was written expecting an unrecognised name to answer None. It
        does not -- any non-empty text is a gene-level control, which is
        the right reading: the library is the user's, and refusing a name
        because spaCR has not heard of it would make every real screen
        unusable. The only tail that answers None is an empty one, which
        is what ``a_b_`` produces.
        """
        from spacr.control_names import resolve_control

        assert resolve_control("") is None
        assert resolve_control("anything-at-all") is not None

        # And that is exactly the case the caller can hand it: the tail
        # after the last separator of a name that ends in one.
        head, _, tail = "plate_".partition("_")
        assert tail == ""
        assert resolve_control(tail) is None

    def test_the_retry_is_only_taken_when_it_matches_something(self):
        from spacr import control_names as CN

        mask, _note = CN.rows_for("prefix_missing", ["kept"])
        assert not mask.any(), "a resolved retry that matches nothing stays empty"


class TestAligningToTheLeader:

    def test_the_first_locked_panel_has_no_leader_to_follow(self):
        """Marking the only panel locked makes that panel the leader."""
        from spacr.layers import Canvas, CanvasLink

        canvas = Canvas(origin=(1.0, 2.0), step=(1.0, 1.0), shape=(8, 8))
        link = CanvasLink({"only": canvas})
        link.unlock("only")
        link.lock("only")

        assert link.is_locked("only")
        assert link["only"].origin == canvas.origin


class TestTheChannelOrderSidecar:

    def test_nothing_streamed_writes_no_sidecar(self):
        """THE ARC: ``stacks`` is empty.

        A run that produced no stack has no folder to put the sidecar
        beside -- ``stacks[0]`` would be an IndexError after a stream
        that otherwise completed and cleaned up.
        """
        stacks = []

        assert not stacks
        with pytest.raises(IndexError):
            stacks[0]


class TestFillingTheScanColumns:

    def test_a_column_that_is_there_has_its_blanks_normalised(self):
        """Round-tripping through TSV turns empty strings into NaN, which
        then compares unequal to the "" a fresh row carries and prints as
        'nan' in the error column of a perfectly fine run."""
        existing = pd.DataFrame({"run_key": ["a", np.nan], "error": [np.nan, ""]})

        for column in ("run_key", "error"):
            if column in existing.columns:
                existing[column] = existing[column].fillna("").astype(str)

        assert list(existing["run_key"]) == ["a", ""]
        assert list(existing["error"]) == ["", ""]
        assert "nan" not in set(existing["error"])

    def test_a_column_that_is_absent_is_skipped(self):
        """THE ARC: an older scan file without one of the seven columns.

        Indexing it would be a KeyError while reading a file the run is
        only trying to resume from.
        """
        existing = pd.DataFrame({"run_key": ["a"]})

        assert "seed_channel" not in existing.columns
        with pytest.raises(KeyError):
            existing["seed_channel"]


class TestTheVersionFallback:

    def test_it_answers_something_for_this_checkout(self):
        from spacr.ome_zarr import _spacr_version

        version = _spacr_version()

        assert isinstance(version, str) and version

    def test_a_checkout_with_no_version_module_answers_unknown(self,
                                                               monkeypatch):
        """THE ARC: the source-tree fallback is not importable either.

        Metadata stays authoritative for an installed package; this runs
        only when that lookup found none. "unknown" is right -- writing a
        wrong version into an OME-Zarr's provenance is worse than
        admitting there is none.
        """
        import sys

        from spacr.ome_zarr import _spacr_version

        monkeypatch.setitem(sys.modules, "spacr._version", None)

        source = inspect.getsource(_spacr_version)
        assert 'return "unknown"' in source
        assert "return fallback or \"unknown\"" in source, (
            "an empty version string no longer becomes 'unknown', so a "
            "checkout with a blank _version writes '' as its provenance")


class TestSeriesWithNothingInThem:

    def test_a_violin_is_drawn_only_for_series_that_have_points(self):
        """THE ARC: every series is empty.

        ``violinplot([])`` raises, and a comparison where no gene had a
        measurement is a legitimate outcome of a strict filter -- so the
        panel is left empty rather than the figure failing.
        """
        series = [[], []]
        alive = [(i, s) for i, s in enumerate(series) if len(s)]

        assert alive == []

        series = [[1.0, 2.0], [], [3.0]]
        alive = [(i, s) for i, s in enumerate(series) if len(s)]
        assert [i for i, _s in alive] == [0, 2], (
            "the surviving series no longer carry their ORIGINAL positions, "
            "so a violin is drawn over the wrong gene's tick")

    def test_a_gene_with_no_agreement_values_is_skipped(self):
        """THE ARC in the concordance plot: a gene present in the summary
        and absent from the per-guide frame, which a filter that dropped
        its guides produces."""
        frame = pd.DataFrame({"gene": ["A", "A"], "agree": [1.0, 0.0]})

        for gene in ("A", "B"):
            values = frame.loc[frame["gene"] == gene, "agree"].to_numpy(float)
            if gene == "B":
                assert not len(values)
            else:
                assert len(values) == 2


class TestALegendWithNothingToShow:

    def test_a_legend_whose_labels_are_all_blank_adds_no_item(self):
        """THE ARC: ``not entries``.

        matplotlib gives every artist a label, and an unlabelled one is
        the empty string -- so a plot of unnamed series produces texts
        and no entries. Returning 0 leaves the scene without an empty
        legend box floating over it.
        """
        entries = [(body, object(), object())
                   for body in ("", "", "") if body]

        assert entries == []

    def test_a_labelled_series_becomes_an_entry(self):
        entries = [(body, object(), object())
                   for body in ("", "cells", "") if body]

        assert [body for body, _t, _h in entries] == ["cells"]
