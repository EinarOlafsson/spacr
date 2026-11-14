"""Answering "will this model work?" before an hour of training finds out.

:mod:`spacr.model_check` never raises -- it runs from a click. What it does
instead is turn each way a chosen model can be wrong into a sentence naming
the setting that would fix it, and the sentences are what this drives.
"""
from __future__ import annotations

from spacr import model_check as MC


def test_a_channel_count_that_is_not_a_number_falls_through_to_the_next_key():
    """One unreadable channels entry does not hide the readable one after it."""
    assert MC.expected_channels({"train_channels": "all of them",
                                 "extract_channels": [0, 1, 2]}) == 3
    assert MC.expected_channels({"train_channels": "3"}) == 3
    assert MC.expected_channels({"train_channels": "all",
                                 "extract_channels": None,
                                 "channels": object()}) is None


def test_a_model_whose_head_cannot_be_walked_reports_no_size_rather_than_a_guess():
    """Returning None is the honest answer; a made-up number is worse."""
    class Unwalkable:
        def modules(self):
            raise RuntimeError("this model does not expose its modules")

    assert MC._head_size(Unwalkable()) is None


def test_a_custom_path_that_does_not_exist_falls_back_to_the_model_type(tmp_path):
    """A missing path is a mistake, not a preference for the built-in."""
    kind, name = MC.resolve_model_source(
        {"custom_model_path": str(tmp_path / "gone.pth"),
         "model_type": "maxvit_t"})
    assert (kind, name) == ("builtin", "maxvit_t")
