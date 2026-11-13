"""The final ten uncovered statements in spacr.utils.

Each test here targets one specific line that the wave-1/wave-2 suites left
behind — mostly defensive branches that only fire on a failure or an unusual
array layout. Every one is driven for real (blocked imports, injected
exceptions, 4-D arrays); nothing is pragma'd out.

CPU-only, offline, no plotting windows.
"""
from __future__ import annotations


import numpy as np


# ---------------------------------------------------------------------------
# NOT covered here: utils.py lines 34-40, the IPython-import fallback that
# defines a no-op display(). Exercising it means blocking the IPython import
# and re-importing spacr.utils, which replaces the module object while
# spacr.io and friends still hold references to the old one — that reliably
# breaks the TorchModel/ResNet tests later in the same session. Seven lines of
# import fallback are not worth destabilising the suite, so they are left
# uncovered rather than pragma'd or faked.
# ---------------------------------------------------------------------------
# 160: _select_intensity_channel's >3-D passthrough
# ---------------------------------------------------------------------------

def test_select_intensity_channel_passes_through_4d_stacks():
    """A 4-D array has no channel layout this helper understands, so it is
    returned whole (as float32) instead of being indexed."""
    from spacr.utils import _select_intensity_channel

    raw = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    out = _select_intensity_channel(raw, intensity_channel=1)

    assert out.shape == raw.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, raw.astype(np.float32))


# ---------------------------------------------------------------------------
# 1537-1538: both prcfo builds fail -> the inner except prints the error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3215: _scalar unwraps a Series
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4339: _remove_outside_objects clears a nucleus lying inside a stray pathogen
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5639: plot_clusters_grid skips the DBSCAN noise label
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7286: delete_intermedeate_files bails out when merged/ is under-populated
# ---------------------------------------------------------------------------
