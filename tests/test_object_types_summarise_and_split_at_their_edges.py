"""Item 288: three edges of ``spacr.schema``'s object-type vocabulary.

* ``object_type_summary`` collapses organelle slots into one pattern. With
  only SOME two-letter slots in use it lists them by their first letter
  rather than claiming the whole two-letter range.
* ``split_object_id`` does not read the bare word ``organelle`` -- a role
  with no label after it -- as an organelle object.
* ``add_screen_column`` fills a ``None`` screen label with the default
  screen, as it does ``NaN`` and blanks, and keeps a real one.
"""
from __future__ import annotations

import pandas as pd

from spacr import schema
from spacr.organelle_types import organelle_role


def test_some_two_letter_slots_are_listed_by_their_first_letter():
    roles = ["cell", organelle_role(27), organelle_role(28),
             organelle_role(60)]
    assert [organelle_role(n) for n in (27, 28, 60)] == \
        ["organelleaa", "organelleab", "organellebh"]
    summary = schema.object_type_summary(roles)
    assert summary == "cell, organelle(?:a[ab]|b[h])"


def test_the_bare_role_word_is_not_an_organelle_object():
    assert schema._split_organelle_id("organelle", "organelle") is None
    kind, _label = schema.split_object_id("organelle")
    assert kind != "organelle"
    assert schema.split_object_id("organelle7") == ("organelle", "7")


def test_a_missing_screen_label_takes_the_default_screen():
    frame = pd.DataFrame({schema.SCREEN_KEY: [None, "screenB", float("nan"),
                                              " "]})
    before = frame.copy()
    out = schema.add_screen_column(frame)
    assert list(out[schema.SCREEN_KEY]) == [
        schema.DEFAULT_SCREEN, "screenB", schema.DEFAULT_SCREEN,
        schema.DEFAULT_SCREEN]
    # pandas 3 stores the None of a string column as its own missing value,
    # so the input is compared with a copy taken before the call.
    pd.testing.assert_frame_equal(frame, before, obj="the input is not edited")
    assert pd.isna(frame[schema.SCREEN_KEY].iloc[0])
