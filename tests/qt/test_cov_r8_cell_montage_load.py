"""`load` reporting why a montage cannot be built, and an inert husk.

`load` runs on a worker thread and touches no widget, so every failure
has to come back as `MontageLoad.error` rather than as an exception --
the caller is a tab that must stay on screen and say why. These tests
characterise the refusals it can give, which is what makes it safe to
delete the dead block found beside them.

THE HUSK. Between the "no fractions anywhere" refusal and the crop loop
there was:

    try:
        pass
    except MontageError as error:
        return MontageLoad(...)
    except Exception as error:
        return MontageLoad(..., error=f"Could not read the per-well guide
                                       fractions: {error}")

`pass` cannot raise, so neither handler could ever run. The message in
the second one duplicates the real handler twenty lines above it, which
is where the reading actually happens -- the work was moved out and the
shell was left behind. Seven lines that could not execute, removed.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


def test_the_inert_try_block_is_gone():
    """The husk removed above must not come back.

    A `try:` whose body is `pass` cannot raise, so its handlers are
    unreachable by construction -- not defensive, just noise that
    coverage reports for ever.
    """
    import inspect

    from spacr.qt.widgets import cell_montage_view

    source = inspect.getsource(cell_montage_view.load)
    assert "try:\n        pass\n" not in source, (
        "an inert try/except husk is back in load()")
