"""The last untaken edges in five pieces of spaCR infrastructure.

Five modules that are not analysis and not GUI, each with one or two arcs the
suite had never taken:

* ``spacr.schema`` -- the row-letter guard, and the object-role registry that
  validates itself at import.
* ``spacr.logging_util`` -- the trace hook surviving a logging call that
  fails, and a level handler that is carrying somebody else's filter.
* ``spacr.figures.fast_render`` -- the offscreen platform NOT being forced
  when a display is advertised, and a destination with no folder in it.
* ``spacr.pipeline_v2`` -- a non-``cell`` object type claiming no nucleus
  channel, and the empty-stack early return.
* ``spacr.resources.home.versions._generators.common`` -- the generator's
  path fix when ``spacr`` has not been imported yet.

Three of the targets turned out to be guards no caller can fail. Those are
not silenced: each has a proof beside it and a test that pins the invariant
the proof rests on, so the guard becomes reachable again the moment somebody
breaks the invariant.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr import logging_util as lu
from spacr import pipeline_v2 as PV
from spacr import schema
from spacr.figures import fast_render as FR

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


# ---------------------------------------------------------------------------
# spacr.schema
# ---------------------------------------------------------------------------

def test_every_token_row_id_hands_to_the_letter_parser_comes_back_a_number():
    """``row_id``'s ``if index is not None`` cannot be false. Proof, and pin.

    ``row_id`` only calls :func:`row_index_from_letters` after
    ``_ROW_ONLY.match(letters)`` has succeeded, and ``_ROW_ONLY`` is
    ``^([A-Za-z]{1,2})$``. Every one of the 702 tokens that matches is a
    non-empty ASCII-letter string, which is exactly the input for which
    ``row_index_from_letters`` cannot return None: it returns None only for a
    non-``str``, for an empty ``strip()``, or for a character outside
    ``A``-``Z`` after ``upper()``. So the ``is not None`` re-check is a
    defensive re-test of something the caller has already guaranteed.

    The proof is exhaustive rather than argued, and the second half drives the
    tokens that DO make the parser answer None -- so this is not the vacuous
    "no counterexample found" that an absence assertion on its own would be.
    """
    accepted = [chr(a) + tail
                for tail in ("",) + tuple(chr(b) for b in range(65, 91))
                for a in range(65, 91)]
    assert len(accepted) == 702

    for token in accepted:
        assert schema._ROW_ONLY.match(token), token
        assert schema._PREFIXED_INT.match(token) is None, token
        index = schema.row_index_from_letters(token)
        assert index is not None, f"{token!r} reached row_id's guard as None"
        assert schema.row_id(token) == f"r{index}"

    assert schema.row_id("a") == "r1"
    assert schema.row_id("AF") == "r32"

    # The tokens for which the parser DOES answer None -- so the guard is a
    # real question in general, just not one this caller can ask. Not one of
    # them survives _ROW_ONLY, which is the whole argument above.
    for parser_says_none in ("A1", "A-1", "", "   ", None, 3):
        assert schema.row_index_from_letters(parser_says_none) is None
        assert schema._ROW_ONLY.match(str(parser_says_none)) is None

    # And _ROW_ONLY also stops tokens the parser would happily number, so the
    # call under the guard is only ever reached with 1-2 letters.
    assert schema.row_index_from_letters("ABC") == 731
    assert schema._ROW_ONLY.match("ABC") is None
    assert schema.row_id("ABC") == "rABC"
    assert schema.row_id(3) == "r3"


def test_no_declared_object_role_can_trip_the_registry_guard_at_import():
    """The ``raise`` under ``for _registered_role in OBJECT_TYPES`` is dead.

    That loop runs once, at module import, over a tuple literal declared
    twenty lines above it in the same file. Nothing can substitute a different
    registry before it runs -- there is no import-time hook between the two
    statements -- so the ``raise`` fires only if somebody edits
    ``OBJECT_TYPES`` to hold a role that is empty, contains ``KEY_SEPARATOR``,
    or contains a digit.

    This test is that guard, moved to where it can be seen: it fails for the
    same edit, and it fails BEFORE the import-time raise would, because it
    also checks the round trip the guard exists to protect.
    """
    def trips_the_guard(role):
        return bool(not role
                    or schema.KEY_SEPARATOR in role
                    or any(character.isdigit() for character in role))

    assert schema.OBJECT_TYPES
    for role in schema.OBJECT_TYPES:
        assert not trips_the_guard(role), f"{role!r} would fail the import"
        # What the guard protects: role and label concatenate with no
        # separator, so a digit or an underscore in a role is unrecoverable.
        token = schema.object_id(17, object_type=role)
        assert token == f"{role}17"
        assert schema.split_object_id(token) == (role, "17")

    # The guard is not vacuous: each shape it names really is rejected.
    assert trips_the_guard("")
    assert trips_the_guard(f"cell{schema.KEY_SEPARATOR}wall")
    assert trips_the_guard("cell1")


# ---------------------------------------------------------------------------
# spacr.logging_util
# ---------------------------------------------------------------------------

@pytest.fixture
def logging_sandbox():
    """Put back every piece of process-wide logging state a test moves."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_level_handlers = dict(lu._LEVEL_HANDLERS)
    saved_file_filter = lu._FILE_FILTER
    yield
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    lu._LEVEL_HANDLERS.clear()
    lu._LEVEL_HANDLERS.update(saved_level_handlers)
    lu._FILE_FILTER = saved_file_filter
    lu._TRACE_STATE.busy = False


def _a_frame_from_a_spacr_module(module_name="spacr.measure"):
    """A real frame whose module name and filename both look like spaCR's."""
    source = "def f():\n    import sys\n    return sys._getframe()\n"
    parts = module_name.split(".")
    filename = str(Path(lu._TRACE_ROOT, *parts[1:]).with_suffix(".py"))
    namespace = {"__name__": module_name}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["f"]()


def test_a_trace_write_that_raises_leaves_the_traced_call_untouched():
    """``except Exception: pass`` inside the hook, and the flag it must clear.

    The comment on that handler is the promise: "a tracing aid must never
    alter the code it observes". A logger that raises -- a handler wired to a
    closed stream, a formatter that cannot format -- would otherwise raise
    inside whatever function happened to be called next, which is every
    function in the process.
    """
    logger = logging.getLogger("spacr.trace")
    previous_level, previous_disabled = logger.level, logger.disabled
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    lu._TRACE_STATE.busy = False
    frame = _a_frame_from_a_spacr_module()
    try:
        # The working case first, so the failure below is a difference and
        # not a test that would pass against a hook that does nothing.
        assert lu._trace_one_event(frame, "call") is None
        assert len(records) == 1
        assert records[0].getMessage().startswith("→ spacr.measure.")
        assert lu._TRACE_STATE.busy is False

        boom = logger.debug

        def explode(*args, **kwargs):
            raise RuntimeError("the trace's own logger is broken")

        logger.debug = explode
        try:
            assert lu._trace_one_event(frame, "return") is None
        finally:
            logger.debug = boom

        assert len(records) == 1, "a failed write must not invent a record"
        assert lu._TRACE_STATE.busy is False, (
            "the finally must clear the flag even when the write raised")

        # And the hook still works afterwards -- the failure was swallowed,
        # not latched.
        lu._trace_one_event(frame, "return")
        assert len(records) == 2
        assert records[1].getMessage().startswith("← spacr.measure.")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled
        lu._TRACE_STATE.busy = False


def test_a_foreign_filter_on_a_level_handler_survives_and_still_sees_records(
        tmp_path, logging_sandbox):
    """Reinstalling the policy walks past filters it did not install.

    ``_install_level_handlers`` keeps one handler per level for the life of
    the process and re-points its ``LevelSetFilter`` instead of detaching it.
    Anything else attached to that handler -- a caller's own filter, a
    diagnostic counter -- must be left exactly where it is, and must still be
    consulted, or switching a level on and off would quietly disarm it.
    """
    class CountingFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def filter(self, record):
            self.seen += 1
            return True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    master = tmp_path / "spacr.log"
    lu._LEVEL_HANDLERS.clear()
    lu._install_level_handlers(master, lu.LEVELS)

    handler = lu._LEVEL_HANDLERS[logging.ERROR]
    counter = CountingFilter()
    handler.addFilter(counter)
    assert isinstance(handler.filters[0], lu.LevelSetFilter)

    # ERROR switched off: the loop must step over `counter` to find the
    # LevelSetFilter and empty it.
    lu._install_level_handlers(master, {logging.WARNING})
    assert counter in handler.filters
    level_filter = next(f for f in handler.filters
                        if isinstance(f, lu.LevelSetFilter))
    assert level_filter.levels == set()

    error_log = master.parent / lu.LEVEL_LOG_FILENAMES[logging.ERROR]
    logging.getLogger("spacr.probe").error("while the level is off")
    off_bytes = error_log.stat().st_size if error_log.exists() else 0
    assert off_bytes == 0

    # ERROR switched back on: same handler, same foreign filter, and now the
    # record reaches the file THROUGH it.
    lu._install_level_handlers(master, {logging.ERROR})
    assert lu._LEVEL_HANDLERS[logging.ERROR] is handler
    assert counter in handler.filters
    assert level_filter.levels == {logging.ERROR}

    before = counter.seen
    logging.getLogger("spacr.probe").error("while the level is on")
    assert counter.seen > before, "the foreign filter was bypassed"
    assert "while the level is on" in error_log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# spacr.figures.fast_render
# ---------------------------------------------------------------------------

def test_an_advertised_display_is_not_overridden_with_the_offscreen_platform(
        monkeypatch):
    """``_pyqtgraph_ready`` forces ``offscreen`` only when there is no display.

    The setdefault exists so a headless machine can still export a scene. On a
    machine that HAS a display, forcing offscreen would render the export
    through a different platform plugin than the one the user is looking at,
    which is the disagreement this module exists to end.
    """
    ok, why = FR._pyqtgraph_ready(create=True)
    if not ok:
        pytest.skip(why)

    from PySide6.QtWidgets import QApplication
    existing = QApplication.instance()
    assert existing is not None

    saved_application = FR._APPLICATION
    # The branch under test is only reached when the module believes there is
    # no QApplication; there is no public way to unmake one, and unmaking the
    # session's app to test a branch would be worse than reaching past it.
    monkeypatch.setattr(FR, "qt_application", lambda: None)
    try:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        assert FR._pyqtgraph_ready(create=True) == (True, "")
        assert os.environ["QT_QPA_PLATFORM"] == "offscreen", (
            "with no display the export platform must be forced")
        assert FR._APPLICATION is existing

        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.setenv("DISPLAY", ":99")
        assert FR._pyqtgraph_ready(create=True) == (True, "")
        assert "QT_QPA_PLATFORM" not in os.environ, (
            "an advertised display must be left to choose its own platform")
        assert FR._APPLICATION is existing
    finally:
        FR._APPLICATION = saved_application


def test_a_destination_with_no_folder_in_it_creates_no_folder(
        tmp_path, monkeypatch):
    """A bare filename is written where it stands, not under a new directory.

    ``os.makedirs`` on ``os.path.dirname('volcano.pdf')`` -- the empty string
    -- raises, so the guard is what lets a relative destination work at all.
    """
    class FakePlot:
        def __init__(self):
            self.exported = []

        def export(self, destination):
            self.exported.append(str(destination))
            Path(destination).write_bytes(b"%PDF-fake")
            return str(destination)

    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))

    bare = FakePlot()
    rendered = FR.render_panel("volcano", plot=bare, path="volcano.png",
                               announce=False)
    assert rendered.drawn is True
    assert rendered.renderer == "pyqtgraph"
    assert os.path.dirname(rendered.path) == ""
    assert (tmp_path / rendered.path).is_file()
    assert set(os.listdir(tmp_path)) - before == {os.path.basename(
        rendered.path)}, "a bare name must not have made a directory"

    # The same call with a folder in the destination: now the folder IS made.
    nested = FakePlot()
    rendered_nested = FR.render_panel(
        "volcano", plot=nested, path=str(tmp_path / "nested" / "volcano.png"),
        announce=False)
    assert rendered_nested.drawn is True
    assert (tmp_path / "nested").is_dir()
    assert Path(rendered_nested.path).parent == tmp_path / "nested"


def test_every_fast_panel_key_has_a_branch_that_feeds_its_plot():
    """``build_fast_plot``'s final fall-through is unreachable. Proof, and pin.

    The function raises ``KeyError`` for any key outside ``FAST_PANELS``, so
    by the time the ``if/elif`` chain starts, ``key`` is one of the seven
    entries in that dict. The chain names all seven. The path where every
    branch is false and the plot is returned unfed therefore cannot be taken.

    That is only true while the dict and the chain agree, which is precisely
    what this test holds: add a panel to ``FAST_PANELS`` without giving it a
    branch and this fails, naming the key that would come back empty.
    """
    fed_by_the_chain = {
        "volcano", "effect_rank", "effect_distribution",
        "p_histogram", "qq", "controls", "agreement",
    }
    assert set(FR.FAST_PANELS) == fed_by_the_chain, (
        "a FAST_PANELS key with no branch would return an unfed plot")

    import pandas as pd
    with pytest.raises(KeyError) as caught:
        FR.build_fast_plot("not_a_panel", pd.DataFrame({"a": [1]}))
    assert "no pyqtgraph twin" in str(caught.value)


# ---------------------------------------------------------------------------
# spacr.pipeline_v2
# ---------------------------------------------------------------------------

def _make_plate(dst: Path, name="plate1", channels=3, size=12) -> Path:
    import tifffile

    plate = dst / name
    plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for field in (1,):
        for channel in range(channels):
            arr = rng.integers(0, 2000, size=(size, size)).astype(np.uint16)
            tifffile.imwrite(
                str(plate / f'{name}_A01_T01F0{field}L01A01Z01C0{channel}.tif'),
                arr)
    return plate


class _ConstantMaskModel:
    def __init__(self, *args, **kwargs):
        self.pretrained_model = None

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis)
        return ([np.ones(np.asarray(image).shape[:2], dtype=np.uint16)
                 for image in x], None, None)


def test_only_a_cell_run_claims_the_second_plane_as_its_nucleus(
        tmp_path, monkeypatch):
    """The nucleus slot is a cell-specific remap, not a rule about plane 1.

    ``stream_masks_from_stack`` reuses V1's normaliser by remapping the
    channel roles onto the compact Cellpose input. Handing plane 1 to
    ``nucleus_channel`` is right for a cell run, where the second selected
    plane IS the nucleus; doing it for a nucleus or pathogen run would
    normalise the object's own second plane as if it were a different
    organelle, and the masks would differ from V1's for the same input.
    """
    seen = []

    def recording_normalise(*, stack, channels, save_dtype, settings):
        seen.append(dict(settings))
        return np.asarray(stack, dtype=np.float32)

    import spacr.io as spacr_io
    monkeypatch.setattr(spacr_io, "_normalize_img_batch", recording_normalise)
    monkeypatch.setattr("cellpose.models.CellposeModel", _ConstantMaskModel)

    for index, object_type in enumerate(("nucleus", "cell")):
        plate = _make_plate(tmp_path, name=f"plate{index}")
        mapper = PV.FilenameMapper.discover(plate,
                                            metadata_type="cellvoyager")
        stacks = PV.stream_originals_to_stack(plate, mapper,
                                              channels=(0, 1, 2))
        PV.stream_masks_from_stack(
            stacks, model_name="cyto", batch_fields=1,
            channels_for_cellpose=(0, 1), object_type=object_type,
            postprocess_settings={"percentiles": [2, 98]})

    nucleus_run, cell_run = seen
    # Both runs hand Cellpose the same two-plane input...
    assert nucleus_run["nucleus_channel"] == 0
    assert nucleus_run["cell_channel"] is None
    # ...and only the cell run reads plane 1 as a nucleus.
    assert cell_run["cell_channel"] == 0
    assert cell_run["nucleus_channel"] == 1


def test_an_empty_stack_list_returns_before_cellpose_is_ever_loaded(
        tmp_path, monkeypatch):
    """The ``if stacks:`` before the sidecar update cannot be false. Proof.

    ``stream_masks_from_stack`` opens with ``if not stacks: return stacks``,
    and ``stacks`` is never rebound anywhere in the body -- the batching does
    ``batch = stacks[a:b]`` and the writeback mutates ``StackFile`` attributes.
    So by the time the sidecar block is reached, ``stacks`` is the same
    non-empty list the guard let through, and its false arm is dead.

    What is pinned here is the early return the proof rests on: an empty list
    must cost nothing at all -- no scratch directory, no Cellpose import, no
    model construction -- and the non-empty case in the same test shows all
    three of those really do happen otherwise.
    """
    class ExplodingModel:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cellpose must not be loaded for no stacks")

    monkeypatch.setattr("cellpose.models.CellposeModel", ExplodingModel)
    empty = []
    assert PV.stream_masks_from_stack(empty, model_name="cyto") is empty
    assert not list(tmp_path.iterdir())

    monkeypatch.setattr("cellpose.models.CellposeModel", _ConstantMaskModel)
    plate = _make_plate(tmp_path, name="plateA")
    mapper = PV.FilenameMapper.discover(plate, metadata_type="cellvoyager")
    stacks = PV.stream_originals_to_stack(plate, mapper, channels=(0, 1, 2))
    assert stacks

    returned = PV.stream_masks_from_stack(stacks, model_name="cyto",
                                          batch_fields=1)
    assert returned is stacks
    sidecar = stacks[0].path.parent / "channel_order.json"
    assert json.loads(sidecar.read_text())["mask_channels"] == ["mask"]


# ---------------------------------------------------------------------------
# spacr.resources.home.versions._generators.common
# ---------------------------------------------------------------------------

def test_the_generator_fixes_sys_path_even_before_spacr_is_imported():
    """``_prefer_checkout_package`` returns early when there is no spacr yet.

    The generator runs the checkout's own spaCR, and it is called at import
    time of its own module -- so it has to work in a process where ``spacr``
    has not been imported at all. There is nothing to check the origin of
    then, and nothing to evict: the path fix alone is the whole job.
    """
    from spacr.resources.home.versions._generators import common

    root = common.repo_root()
    sentinel = "spacr.zz_generator_probe"
    saved_path = list(sys.path)
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name == "spacr" or name.startswith("spacr.")}
    try:
        sys.modules[sentinel] = sys.modules["spacr.schema"]

        # No spacr in sys.modules: the early return, and nothing evicted.
        sys.modules.pop("spacr", None)
        common._prefer_checkout_package()
        assert os.path.realpath(sys.path[0]) == os.path.realpath(root)
        assert [entry for entry in sys.path
                if os.path.realpath(entry or os.getcwd())
                == os.path.realpath(root)] == [sys.path[0]]
        assert sentinel in sys.modules, (
            "with no spacr loaded there is no origin to disagree with")

        # A spacr loaded from somewhere else: now the eviction really runs,
        # which is what the early return is skipping past.
        import types
        foreign = types.ModuleType("spacr")
        foreign.__file__ = os.path.join(os.sep, "elsewhere", "spacr",
                                        "__init__.py")
        sys.modules["spacr"] = foreign
        common._prefer_checkout_package()
        assert sentinel not in sys.modules
    finally:
        sys.path[:] = saved_path
        for name in [n for n in sys.modules
                     if n == "spacr" or n.startswith("spacr.")]:
            sys.modules.pop(name, None)
        sys.modules.pop(sentinel, None)
        sys.modules.update(saved_modules)
