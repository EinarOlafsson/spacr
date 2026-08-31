"""Annotate's model sizer, its cache release, and two empty-result paths.

`_model_bytes` measures a warm Cellpose model so the GUI can decide
whether to release it under memory pressure. It runs over somebody
else's object graph -- a torch module whose tensors may or may not
answer the questions asked of them -- so almost every line in it is a
fallback, and none of those fallbacks had been run.

Nothing here needs torch: the sizer only asks for `parameters()`,
`buffers()`, `numel()`, `element_size()` and `nbytes`, so stand-ins say
exactly as much as a real tensor would and say it deterministically.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.qt import annotate_engine as AE


class _Tensor:
    """Answers `numel`/`element_size`, like a torch tensor."""

    def __init__(self, numel=10, element_size=4):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class TestMeasuringAModel:

    def test_parameters_and_buffers_are_both_counted(self):
        class _Net:
            @staticmethod
            def parameters():
                return [_Tensor(10, 4)]

            @staticmethod
            def buffers():
                return [_Tensor(5, 8)]

        assert AE._model_bytes(_Net()) == 10 * 4 + 5 * 8

    def test_the_inner_net_is_preferred_when_there_is_one(self):
        """Cellpose wraps the module; `model.net` is where the weights are."""
        class _Inner:
            @staticmethod
            def parameters():
                return [_Tensor(2, 4)]

        class _Wrapper:
            net = _Inner()

        assert AE._model_bytes(_Wrapper()) == 8

    def test_a_model_with_no_accessors_measures_nothing(self):
        assert AE._model_bytes(object()) == 0

    def test_an_accessor_that_is_not_callable_is_skipped(self):
        """A tensor attribute NAMED `parameters` is not the method."""
        class _Odd:
            parameters = [1, 2, 3]          # not callable

            @staticmethod
            def buffers():
                return [_Tensor(1, 4)]

        assert AE._model_bytes(_Odd()) == 4

    def test_an_accessor_that_raises_is_skipped(self):
        class _Hostile:
            @staticmethod
            def parameters():
                raise RuntimeError("the module is on a dead device")

            @staticmethod
            def buffers():
                return [_Tensor(3, 4)]

        assert AE._model_bytes(_Hostile()) == 12

    def test_a_tensor_shared_between_the_two_is_counted_once(self):
        """Tied weights appear in both lists and are ONE allocation.

        Counting them twice would overstate the model and could make the
        GUI release a cache it did not need to.
        """
        shared = _Tensor(100, 4)

        class _Tied:
            @staticmethod
            def parameters():
                return [shared]

            @staticmethod
            def buffers():
                return [shared]

        assert AE._model_bytes(_Tied()) == 400

    def test_a_tensor_that_only_knows_nbytes_is_measured_by_it(self):
        """numpy-backed buffers answer `nbytes` and not `numel`."""
        class _Array:
            nbytes = 64

        class _Net:
            @staticmethod
            def parameters():
                return [_Array()]

        assert AE._model_bytes(_Net()) == 64

    def test_a_value_that_answers_neither_is_skipped(self):
        """Measuring is best-effort; it must not raise into a budget tick."""
        class _Opaque:
            pass

        class _Net:
            @staticmethod
            def parameters():
                return [_Opaque(), _Tensor(1, 4)]

        assert AE._model_bytes(_Net()) == 4

    def test_a_negative_size_never_subtracts(self):
        """`max(0, ...)` -- a nonsense answer must not shrink the total."""
        class _Net:
            @staticmethod
            def parameters():
                return [_Tensor(-100, 4), _Tensor(2, 4)]

        assert AE._model_bytes(_Net()) == 8


class TestReleasingTheWarmModel:

    def test_a_busy_lock_releases_nothing_and_returns_at_once(self):
        """A five-second GUI budget tick must never wait behind inference.

        The lock is taken without blocking: if native Cellpose is
        running, the tick reports nothing released and the NEXT tick
        retries. Blocking here would freeze the interface for as long as
        the inference takes.
        """
        assert AE._cellpose_outline_lock.acquire(), "could not take the lock"
        try:
            # Held on this thread, but `acquire(blocking=False)` on an
            # RLock re-entered from the same thread would succeed -- so the
            # contention is made from another thread.
            import threading

            answer = []
            done = threading.Event()

            def ask():
                answer.append(AE._release_cached_models())
                done.set()

            threading.Thread(target=ask, daemon=True).start()
            assert done.wait(5), "the release blocked on the held lock"
            assert answer == [0]
        finally:
            AE._cellpose_outline_lock.release()


class TestTwoEmptyResults:

    def test_an_expression_of_only_whitespace_means_no_filter(self):
        """A field the user has cleared to spaces means "no filter".

        It is the FIRST guard that answers, not the token one: the
        expression is stripped before it is tested, so whitespace becomes
        the empty string.
        """
        for blank in ("   ", "\t\n", " \t "):
            assert AE.parse_image_type(blank) == ("", []), blank

    def test_the_no_tokens_guard_cannot_be_reached(self):
        """`if not tokens: return "", []` is dead, and pinned here.

        `text` is stripped before the empty check, so anything reaching
        the tokeniser has at least one non-whitespace character -- and
        the tokeniser's pattern, `\\(|\\)|[^\\s()]+`, matches every such
        character. It cannot return an empty list for a non-empty input.

        Checked over every printable character and every two-character
        combination of the punctuation that might plausibly be dropped.
        """
        import itertools
        import string

        for ch in string.printable:
            if not ch.strip():
                continue
            assert AE._tokenise_image_type(ch), (
                f"{ch!r} tokenises to nothing; the `not tokens` guard in "
                "parse_image_type is now REACHABLE and wants its own test")
        for a, b in itertools.product("()!$%^&*.,;:\"'\\/", repeat=2):
            text = (a + b).strip()
            if text:
                assert AE._tokenise_image_type(text), repr(text)

    def test_a_comma_is_content_and_not_a_separator(self):
        """Recorded because it is easy to assume otherwise.

        Commas are not tokenised away -- ",," is a substring to match on,
        so it parses to a LIKE. A test that used commas expecting the
        empty result would be asserting the wrong thing.
        """
        sql, params = AE.parse_image_type("   ,,  ")
        assert sql == "png_path LIKE ?"
        assert params == ["%,,%"]

    def test_an_empty_expression_parses_to_nothing(self):
        assert AE.parse_image_type("") == ("", [])
        assert AE.parse_image_type(None) == ("", [])

    def test_a_missing_annotation_column_cannot_reach_the_late_guard(self):
        """`if annotation_column not in df.columns: return []` is dead.

        A database written before that column existed is handled far
        earlier: `fetch_filtered_paths` CREATES the column, filled with
        None, before it applies any threshold --

            if annotation_column not in df.columns:
                df[annotation_column] = None

        -- and the thresholds filter rows, not columns. So by the time
        the late guard is reached the column is always present.

        Answering [] there would be wrong anyway: the rows exist and are
        simply unannotated, which is exactly what the annotator opens a
        database to fix.

        Pinned from the producing side rather than forced.
        """
        import inspect

        source = inspect.getsource(AE.fetch_filtered_paths)
        assert "df[annotation_column] = None" in source, (
            "the annotation column is no longer created up front; the late "
            "`not in df.columns` guard may now be reachable")
        create_at = source.index("df[annotation_column] = None")
        guard_at = source.rindex("if annotation_column not in df.columns:")
        assert create_at < guard_at, (
            "the column is now created AFTER the guard that checks for it")
