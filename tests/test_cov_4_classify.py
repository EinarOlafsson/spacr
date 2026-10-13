"""An unknown classifier family is refused rather than silently answered.

``inapplicable_settings`` drives which settings the panel greys out. If a
misspelled family returned an empty tuple instead of raising, the panel would
grey out nothing and every setting from both families would look applicable.
"""
from __future__ import annotations

import pytest

from spacr.classify import (CLASSIFIER_FAMILIES, ClassifierFamilyError,
                            inapplicable_settings)


@pytest.mark.parametrize("family", ["", "xgboost", "nn", None])
def test_an_unknown_family_names_the_ones_that_exist(family):
    """The refusal lists the real families so the caller can correct it."""
    with pytest.raises(ClassifierFamilyError) as excinfo:
        inapplicable_settings(family)
    message = str(excinfo.value)
    for known in CLASSIFIER_FAMILIES:
        assert known in message, message


def test_a_known_family_still_returns_the_other_familys_settings():
    """The guard does not reject the families the panel actually uses."""
    for known in CLASSIFIER_FAMILIES:
        greyed = inapplicable_settings(known)
        assert isinstance(greyed, tuple)
