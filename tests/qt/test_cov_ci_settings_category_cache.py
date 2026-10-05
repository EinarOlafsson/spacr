"""Category caching stays bounded and can bypass an unmeasurable shape."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import plugins  # noqa: E402
from spacr.qt.screens import settings_model  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_full_category_cache_is_replaced_without_losing_the_new_layout(
        monkeypatch):
    old = {(str(index),): {"Old": ["unused"]} for index in range(257)}
    monkeypatch.setattr(settings_model, "_CATEGORIES_FOR_APP_MEMO", old)
    monkeypatch.setattr(plugins, "get_app", lambda _key: None)
    source = {"General": ["src"]}

    first = settings_model.categories_for_app("probe", source)
    first["General"].append("not persisted")
    second = settings_model.categories_for_app("probe", source)

    assert second == {"General": ["src"]}
    assert len(old) == 1


def test_an_unmeasurable_category_shape_still_produces_a_layout(monkeypatch):
    class Unmeasurable(list):
        def __len__(self):
            if not hasattr(self, "_measured"):
                self._measured = True
                raise RuntimeError("shape cannot be measured")
            return super().__len__()

    cache = {}
    monkeypatch.setattr(settings_model, "_CATEGORIES_FOR_APP_MEMO", cache)
    monkeypatch.setattr(plugins, "get_app", lambda _key: None)
    source = {"General": Unmeasurable(["src"])}

    assert settings_model.categories_for_app("probe", source) == source
    assert cache == {}
