"""Item 288: the Activation sweep keeps a ``cam_type`` the registry does not know.

``_applicable_activation_space`` drops the swept ``cam_type`` values the
loaded model cannot use, each with its reason. A value the attribution
registry has never heard of is NOT dropped: its trial then fails naming the
unknown method, which is the sentence the user needs, whereas dropping it
here would make a typo in the grid disappear without a word.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.test_18_modern_attribution_methods import CornerCNN  # noqa: E402


def test_an_unknown_cam_type_stays_in_the_sweep():
    from spacr.hyperparam import (ActivationSearchData, SearchSpace,
                                  _applicable_activation_space)

    data = ActivationSearchData(model=CornerCNN(), images=[])
    space, notes = _applicable_activation_space(
        SearchSpace({"cam_type": ["hirescam", "hirescamm"]}), data)
    assert list(space.params["cam_type"]) == ["hirescam", "hirescamm"]
    assert notes == []
