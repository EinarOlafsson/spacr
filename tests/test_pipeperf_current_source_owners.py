"""The current source owners remain active when their cached extraction policy changes."""
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("owner", [
    "_ops_profile_cell_cycle_method_sources",
    "_flatfield_fit_runtime_sources",
    "_cpg_measure_runtime_sources",
    "_ngff06_runtime_sources",
])
def test_current_owner_callback_changes_invalidate_the_file_caption_cache(monkeypatch, owner):
    """A changed current owner contributes new text through a previously inventoried real tree."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    import build_i18n_catalogs as builder

    before = builder.extract_static_ui_sources()
    original = getattr(builder, owner)
    marker = f"Current source owner cache validation: {owner}."

    def changed(*args, **kwargs):
        """Add an observable caption through the real current owner callback."""
        return original(*args, **kwargs) | {marker}

    monkeypatch.setattr(builder, owner, changed)
    after = builder.extract_static_ui_sources()
    assert marker not in before
    assert marker in after
