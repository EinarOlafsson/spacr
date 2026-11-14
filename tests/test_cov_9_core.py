"""Branches of ``spacr.core`` that only a damaged or plural input reaches.

Four situations, each of which produces a wrong number rather than an error
when it is handled badly:

* ``test_mode`` deciding how many overlay figures to draw;
* a ``merged/`` folder the pipeline cannot list;
* two source databases that call different experiments by the same plate id;
* a screen-graph run given several source folders instead of one.
"""
from __future__ import annotations

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


class _Recorder:
    """Records every call and returns ``ret``; stands in for an externality."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret

    @property
    def n(self):
        return len(self.calls)


# ---------------------------------------------------------------------------
# preprocess_generate_masks: the overlay-plot budget
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# generate_image_umap: two databases that share a plate id
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """Ends the run once the branch under test has been passed."""


# ---------------------------------------------------------------------------
# generate_screen_graphs: several source folders
# ---------------------------------------------------------------------------

def test_a_list_of_screen_sources_is_read_source_by_source(tmp_path, monkeypatch):
    """Several source folders must each be read, not joined into one path.

    ``src`` accepts a string or a list. Wrapping an already-plural value would
    build ``"['/a', '/b']/measurements/measurements.db"`` -- a path that
    exists nowhere -- so the plural form has to be used as it stands.
    """
    import spacr.core as core
    import spacr.io as sio

    seen = []

    def _record(db_loc, *args, **kwargs):
        seen.append(list(db_loc))
        raise _Stop()

    monkeypatch.setattr(sio, "_read_and_merge_data", _record)

    with pytest.raises(_Stop):
        core.generate_screen_graphs({
            "src": [str(tmp_path / "plateA"), str(tmp_path / "plateB")],
            "tables": ["cell"], "nuclei_limit": 10, "pathogen_limit": 10,
        })

    assert seen == [[os.path.join(str(tmp_path / "plateA"),
                                  "measurements", "measurements.db")]]
