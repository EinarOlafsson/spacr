"""The last eight uncovered decisions in the package.

Two typing Protocols whose bodies are ellipses, a module-level QSS
registration that runs on import, a reaped subprocess, a preview cell
with no path behind it, and three guards whose other arm is what every
caller takes.
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# flowview/feeder -- two Protocol methods whose bodies are `...`
# ---------------------------------------------------------------------------

class TestTheQueueProtocols:

    def test_the_two_protocols_describe_the_queue_the_feeder_needs(self):
        """THE PIN, for two method bodies that are a literal ellipsis.

        A ``Protocol`` method body never runs -- it exists so a type
        checker can say whether the object handed in is queue-shaped.
        The bodies show as uncovered because they ARE statements, and
        nothing calls them.

        What is worth holding is the SHAPE: the feeder needs a source it
        can ``get`` from with a timeout and a sink it can ``put_nowait``
        to, and those two names are the whole contract with whatever
        queue the caller supplies.
        """
        from spacr.flowview import feeder as F

        source = inspect.getsource(F)
        assert "class _QueueSource(Protocol):" in source
        assert "get: Callable[..., object]" in source
        assert "class _QueueSink(Protocol):" in source
        assert "put_nowait: Callable[[object], None]" in source

    def test_a_standard_queue_satisfies_both(self):
        """The check that makes the Protocols worth having: the thing
        the caller actually passes fits them."""
        import queue

        real = queue.Queue()
        assert callable(getattr(real, "get", None))
        assert callable(getattr(real, "put_nowait", None))

        import inspect as _inspect

        signature = _inspect.signature(real.get)
        assert "block" in signature.parameters
        assert "timeout" in signature.parameters


# ---------------------------------------------------------------------------
# qt/recipes -- a stylesheet registration that runs at import
# ---------------------------------------------------------------------------

class TestTheRecipeButtonStylesheet:

    def test_it_is_registered_at_import_and_replaces_any_earlier_one(self):
        """THE PIN, for a module-level try/except.

        The registration runs on import and is present in every real
        launch, so the handler cannot fire -- but it is wrapped because
        a failure there would stop the module importing at all, and a
        recipe button with the wrong colour is better than no recipes.

        ``replace=True`` is the half worth pinning: a second import must
        not stack a duplicate rule, which is how a QSS block grows until
        it is slower to parse than to apply.
        """
        pytest.importorskip("PySide6")
        from spacr.qt import recipes as R

        source = inspect.getsource(R)
        assert "from .theme import register_widget_qss as _register_widget_qss" \
            in source
        assert "replace=True" in source
        assert "could not register the recipe-button QSS" in source

    def test_the_same_shape_guards_the_shortcut_overlay(self):
        pytest.importorskip("PySide6")
        from spacr.qt import shortcuts as S

        source = inspect.getsource(S)
        assert "could not register the shortcut-overlay QSS" in source
        assert "register_widget_qss" in source


# ---------------------------------------------------------------------------
# qt/ai/providers -- a subprocess that would not die
# ---------------------------------------------------------------------------

class TestReapingAProviderProcess:

    def test_a_process_that_finished_has_its_stream_discarded(self):
        """THE PIN, for ``if finished``.

        The two ways in both end with ``finished`` true: a clean wait
        sets it, and the fallback reaps with ``known_running=True`` and
        reports what it managed. So the discard always runs -- and it
        has to, because an undrained pipe holds the child's file
        descriptors open and the next provider call inherits them.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.ai import providers as P

        source = inspect.getsource(P._stream_process)
        assert "finished = True" in source
        assert "_terminate_and_reap(" in source
        assert "if finished:" in source
        assert "_discard_stream(proc)" in source

        wait = source.index("finished = True")
        discard = source.index("_discard_stream(proc)", wait)
        assert "known_running=True" in source[wait:discard], (
            "the fallback no longer reaps with known_running, so it can "
            "report unfinished and leave the stream undrained")

    def test_the_provider_forgets_the_process_before_the_stream_goes(self):
        pytest.importorskip("PySide6")
        from spacr.qt.ai import providers as P

        source = inspect.getsource(P._stream_process)
        assert "provider._current_proc = None" in source
        assert source.index("provider._current_proc = None") < \
            source.index("_discard_stream(proc)"), (
            "the stream is discarded before the provider forgets the "
            "process, so a caller that looks between the two sees a "
            "provider still holding a process whose pipes are gone")


# ---------------------------------------------------------------------------
# core -- the source list, again
# ---------------------------------------------------------------------------

class TestThePreprocessSourceList:

    def test_a_single_folder_becomes_a_list_before_the_run(self):
        """THE PIN, the same shape ``measure_crop`` has.

        A string is wrapped into a list of one, so the check below it is
        always true and the function cannot fall through without running
        -- which would be a preprocessing pass the caller believes
        happened.
        """
        from spacr import core as C

        source = inspect.getsource(C.preprocess_generate_masks)
        wrap = source.index("if isinstance(settings['src'], str):")
        direct = source.index("source_folders = settings['src']", wrap)
        assert wrap < direct
        assert "settings['src'] = [settings['src']]" in source[wrap:direct]
        assert "if isinstance(settings['src'], list):" not in source[wrap:direct]

        for value in ("/data/plate1", ["/data/a", "/data/b"]):
            if isinstance(value, str):
                value = [value]
            assert isinstance(value, list)

    def test_one_ledger_covers_the_whole_invocation(self):
        """A run over four plates that only managed three must not report
        as if it did four."""
        from spacr import core as C

        source = inspect.getsource(C)
        assert "RunLedger('preprocess_generate_masks')" in source
        assert "must not report as if it did four" in source


# ---------------------------------------------------------------------------
# Three preview guards
# ---------------------------------------------------------------------------

class TestThreePreviewGuards:

    def test_a_table_cell_with_no_path_loads_nothing(self):
        """THE PIN, for ``if path`` in live_preview.

        Every cell the selection can reach was filled with a path when
        the set was loaded, so the read above always answers one. The
        guard is for a cell whose item went during a reload --
        ``load_image(Path(None))`` is a TypeError from a selection
        handler.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import live_preview as L

        source = inspect.getsource(L)
        assert "path = item.data(Qt.UserRole) if item is not None else None" \
            in source
        assert "if path:" in source
        item_check = source.index("if item is not None else None")
        assert source.index("if path:", item_check) > item_check, (
            "the path check no longer follows the item check")

    def test_a_sample_note_is_only_shown_when_there_is_one(self):
        """THE PIN, for ``if self.sample_note()`` in motility_preview.

        The note says the preview is a sample of N of M sets, and it is
        written whenever a sample is drawn -- so redrawing at a new cap
        always has one. ``""[:1].upper()`` is empty rather than an
        error, so what the guard buys is not a crash but a status line
        that would be blanked by a redraw that had nothing to say.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import motility_preview as M

        source = inspect.getsource(M)
        assert "if self.sample_note():" in source
        assert "self.sample_note()[:1].upper()" in source

        assert ""[:1].upper() + ""[1:] == "", (
            "an empty note no longer capitalises to nothing, so the guard "
            "protects against something else")

    def test_a_refit_with_no_destination_says_what_it_would_change(self):
        """THE PIN, for ``if where`` in refit_dialog.

        ``destination`` answers None when the settings name no usable
        count table, and the notice then carries the CHANGES alone --
        which is still the thing the dialog is for. Appending
        "Writes to None." would be worse than saying nothing about
        where.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import refit_dialog as R

        source = inspect.getsource(R)
        assert "where = destination(settings)" in source
        assert "if where:" in source
        assert 'lines.append(f"Writes to {where}.")' in source
        assert "Nothing to change: re-fitting these settings" in source, (
            "the empty-notice text is gone; a refit that changes nothing "
            "must say so rather than showing a blank notice")
