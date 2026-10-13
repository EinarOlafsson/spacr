"""measure.py: the scikit-image capability probe, and one dead guard.

`expand_labels` grew a `spacing` argument after scikit-image 0.22, and
setup.py's floor is >=0.22.0 -- so spaCR PROBES for it rather than
assuming. The comment says why in the sharpest possible terms: without
the argument a 3-D run silently measures an unscaled radius, measured
wrong by 2.000x on a (2.0, 0.2, 0.2) voxel.

A silently wrong measurement is the worst failure this package can have,
so both halves of that probe are worth holding: the answer it gives on a
normal install, and what it does when the signature cannot be read at
all.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest


@pytest.fixture(autouse=True)
def _only_one_spacr_measure_survives_this_file():
    """Put the ORIGINAL ``spacr.measure`` back in ``sys.modules`` afterwards.

    THIS FILE POISONED EVERY LATER TEST IN THE PROCESS, and the mechanism is
    the nastiest kind: nothing fails where the mistake is.

    `_reimport_measure` deletes `spacr.measure` from `sys.modules` so the
    module-level capability probe runs again. The next `import spacr.measure`
    then builds a SECOND, DIFFERENT module object -- and anything still
    holding the first one is now talking to a module the pipeline does not
    use.

    `tests/test_measure_hooks.py` is exactly that: it registers hooks through
    its own reference and then runs a measurement, which imports the other
    copy and finds no hooks registered at all. Ten tests failed in company and
    passed alone (instruction 346), and the bisect that found it took 112
    files down to this one.

    Restoring the original objects is enough, and is better than forbidding
    the reload: the reload is the point of this file, and the probe genuinely
    has to run twice to be tested.
    """
    # THE SAME PREDICATE `_reimport_measure` DELETES BY, and getting this
    # wrong is how the first attempt at this fixture failed. That function
    # says `startswith("spacr.measure")` -- no dot -- so it also removes
    # `spacr.measure_hooks`, WHICH IS WHERE THE HOOK REGISTRY LIVES. A
    # restore that matched only `spacr.measure` and `spacr.measure.*` put the
    # module back and left the registry module rebuilt, which is exactly the
    # state that made `tests/test_measure_hooks.py` register its hooks into a
    # copy nobody reads.
    #
    # WIDENING IT ACROSS `sys.modules` DOES NOT HELP AND WAS TRIED: restoring
    # the whole `spacr.` namespace leaves the same failures, because the
    # surviving stale reference is not in `sys.modules` at all.
    #
    # IT IS THE PARENT PACKAGE'S ATTRIBUTE. Importing `spacr.measure` sets
    # `spacr.measure` as an attribute ON THE `spacr` MODULE OBJECT, and that
    # binding is a SECOND piece of state that `sys.modules` does not own.
    # Putting the original back in `sys.modules` leaves `spacr.measure` --
    # the attribute -- pointing at the rebuilt copy, so the process ends up
    # with the two copies reachable by two different routes:
    #
    #   sys.modules["spacr.measure"]  -> the ORIGINAL   (what `import` finds)
    #   spacr.measure                 -> the REBUILT    (what getattr finds)
    #
    # and that split is what actually broke the later tests. `monkeypatch`
    # resolves a dotted target by GETATTR from the package down
    # (`_pytest.monkeypatch.resolve`), so `monkeypatch.setattr(
    # "spacr.measure.measure_crop", fake)` patched the REBUILT copy, while
    # `from .measure import measure_crop` inside the pipeline imported the
    # ORIGINAL and ran the real thing. The same split raises
    # `PicklingError: ... not the same object as spacr.measure.
    # _measure_crop_core` when a pool pickles a worker function by name.
    #
    # So the restore has to put BOTH back, keyed the same way: the
    # `sys.modules` entries, and the attribute each of them is bound to on
    # its parent package.
    def _ours(name):
        return name.startswith("spacr.measure")

    def _rebind(name, module):
        """Make the parent package's attribute agree with ``sys.modules``.

        UNCONDITIONALLY, and that is the point: by the time the second test
        in this file runs the two are ALREADY split, so a restore that only
        put back an attribute it had seen agreeing at setup would decline to
        repair exactly the case it exists for.
        """
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            return
        if module is None:
            # Never imported before this file ran; leave nothing bound, so
            # the next import builds one copy and binds it in both places.
            if hasattr(parent, leaf):
                delattr(parent, leaf)
        else:
            setattr(parent, leaf, module)

    saved = {name: module for name, module in sys.modules.items()
             if _ours(name)}
    try:
        yield
    finally:
        rebuilt = [name for name in list(sys.modules) if _ours(name)]
        for name in rebuilt:
            del sys.modules[name]
        sys.modules.update(saved)
        for name in set(rebuilt) | set(saved):
            _rebind(name, saved.get(name))


def _reimport_measure():
    for name in [n for n in list(sys.modules) if n.startswith("spacr.measure")]:
        del sys.modules[name]
    return importlib.import_module("spacr.measure")


class TestTheExpandLabelsProbe:

    def test_the_probe_matches_the_installed_signature(self):
        """Whatever it decides must agree with the real function."""
        from skimage.segmentation import expand_labels

        import spacr.measure as measure

        expected = "spacing" in inspect.signature(expand_labels).parameters
        assert measure._EXPAND_LABELS_TAKES_SPACING is expected

    def test_a_signature_that_cannot_be_read_answers_no(self, monkeypatch):
        """THE UNCOVERED PAIR.

        A C-implemented `expand_labels` -- which is what a differently
        built scikit-image can present -- makes `inspect.signature`
        raise. Answering "no spacing argument" is the safe direction: the
        caller then does its own scaling instead of passing an argument
        that would be rejected.
        """
        import skimage.segmentation as seg

        # `dict.update` is C-implemented and has no readable signature,
        # which is exactly the shape being guarded against.
        with pytest.raises((TypeError, ValueError)):
            inspect.signature(dict.update)

        monkeypatch.setattr(seg, "expand_labels", dict.update)
        measure = _reimport_measure()
        try:
            assert measure._EXPAND_LABELS_TAKES_SPACING is False
        finally:
            monkeypatch.undo()
            _reimport_measure()

    def test_the_probe_is_restored_afterwards(self):
        """The re-import above must not leave a poisoned module behind."""
        from skimage.segmentation import expand_labels

        import spacr.measure as measure

        expected = "spacing" in inspect.signature(expand_labels).parameters
        assert measure._EXPAND_LABELS_TAKES_SPACING is expected
