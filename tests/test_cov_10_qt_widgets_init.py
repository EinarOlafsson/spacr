"""The widgets package keeps its deferred names visible.

Heavy widgets are imported on first use, so they are not in the module's
globals until something asks for one. Without an explicit ``__dir__`` they
would be invisible to ``dir()`` and to tab-completion in a notebook -- a
name that imports fine but that nobody can discover.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt import widgets                            # noqa: E402


def test_a_lazily_imported_widget_is_still_listed_by_dir():
    """``GraphCanvas`` is not imported until something asks for it, but it
    has to be findable before anyone knows to ask."""
    names = dir(widgets)
    assert "GraphCanvas" in names
    assert names == sorted(set(names))


def test_dir_lists_every_deferred_name_and_the_eager_ones_too():
    """The two halves of the package must not be discoverable by different
    means; one listing covers both."""
    names = set(dir(widgets))
    assert set(widgets._LAZY).issubset(names)
    assert "Card" in names


def test_a_name_the_package_does_not_have_is_still_an_attribute_error():
    """The lazy hook must not turn a typo into an import of some other
    module."""
    with pytest.raises(AttributeError, match="nosuchwidget"):
        widgets.nosuchwidget
