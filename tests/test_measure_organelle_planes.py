"""Three organelle defects queued under instruction 76. Two real, one not.

(i)  The filtered organelle mask was never written back into `data`, so the
     PNG crops and region arrays kept the UNFILTERED plane while the
     measurements used the filtered one.
(ii) `settings['cytoplasm_mask_dim']` raised KeyError for anyone who asked to
     summarise organelles by cytoplasm -- the key does not exist at all.
(iii) `organelle_min_size` is applied twice. DISPROVEN as a defect: the
     filter is idempotent and the first pass is load-bearing.
"""

import inspect

import numpy as np
import pytest

from spacr.utils import _filter_object


# ---------------------------------------------------------------------------
# (i) the crops must cover the objects the measurements did
# ---------------------------------------------------------------------------

def measure_source():
    from spacr import measure

    return inspect.getsource(measure)


def test_every_mask_plane_is_written_back_including_organelle():
    """Cell, nucleus and pathogen were written back; organelle was not.

    The comment above that block states the invariant it exists for -- "the
    PNG crops and region arrays cover the same objects the measurements do"
    -- and organelle sat outside it, so a crop could show debris the
    measurement table had already dropped.
    """
    source = measure_source()
    for role in ("cell", "nucleus", "pathogen", "organelle"):
        needle = f"data[..., settings['{role}_mask_dim']] = {role}_mask"
        alt = f"data[..., settings.get('{role}_mask_dim')] = {role}_mask"
        assert needle in source or alt in source, (
            f"the {role} plane is not written back; its crops will show "
            f"objects the measurements filtered out")


def test_the_organelle_write_back_follows_the_filter():
    """Writing back BEFORE the filter would put the unfiltered plane in."""
    source = measure_source()
    filtered_at = source.index(
        "organelle_mask = _filter_object(organelle_mask")
    written_at = source.index(
        "data[..., settings['organelle_mask_dim']] = organelle_mask")
    assert filtered_at < written_at


# ---------------------------------------------------------------------------
# (ii) cytoplasm is a boolean, not a dim
# ---------------------------------------------------------------------------

def test_there_is_no_cytoplasm_mask_dim_setting():
    """The premise of the bug: the key never existed.

    The cytoplasm mask is DERIVED -- cell minus its interior objects -- and
    appended as a new plane rather than read from one, so it has no dim.
    """
    import spacr.settings as S
    from spacr.settings import get_measure_crop_settings

    assert "cytoplasm_mask_dim" not in get_measure_crop_settings({"src": "/tmp"})
    assert "cytoplasm_mask_dim" not in S.expected_types


def test_the_cytoplasm_summary_no_longer_reads_a_key_that_does_not_exist():
    source = measure_source()
    assert "settings['cytoplasm_mask_dim']" not in source, (
        "a KeyError for everyone who summarises organelles by cytoplasm")


def test_it_gates_on_the_boolean_instead():
    source = measure_source()
    marker = '"cytoplasm" in settings[\'summarize_organelles_by\']'
    assert marker in source
    # Comments strip first: the fix's own comment explains at length why
    # there is no dim, which pushes the code line well past any fixed window.
    after = source[source.index(marker):source.index(marker) + 2000]
    code = "\n".join(line.split("#", 1)[0] for line in after.splitlines())
    assert "settings['cytoplasm']" in code


# ---------------------------------------------------------------------------
# (iii) DISPROVEN -- and removing it would have introduced a bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_size", [1, 10, 50])
def test_the_size_filter_is_idempotent(min_size):
    """Applying it twice cannot change the answer, so "applied twice" is not
    by itself a defect."""
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[2:5, 2:5] = 1        # 9 px
    mask[10:20, 10:20] = 2    # 100 px
    mask[25:30, 25:30] = 3    # 25 px

    once = _filter_object(mask.copy(), min_size)
    twice = _filter_object(_filter_object(mask.copy(), min_size), min_size)
    assert np.array_equal(once, twice)


def test_the_first_filter_runs_before_the_cytoplasm_is_built():
    """THE REASON THE FIRST PASS CANNOT BE REMOVED.

    The cytoplasm mask is "cell minus every interior object", built FROM the
    organelle mask between the two filter calls. Filtering only at the later
    site would carve sub-threshold organelle debris out of the cytoplasm --
    a silent change to a measured area.
    """
    source = measure_source()
    first_filter = source.index(
        "organelle_mask = _filter_object(organelle_mask")
    cytoplasm_built = source.index("cytoplasm_mask = np.where(interior, 0, cell_mask)")
    second_filter = source.index(
        "organelle_mask = _filter_object(organelle_mask",
        first_filter + 1)

    assert first_filter < cytoplasm_built < second_filter


def test_the_cytoplasm_subtracts_the_organelle():
    """If it did not, the ordering above would not matter."""
    source = measure_source()
    interior = source[source.index("interior = np.zeros_like(cell_mask"):]
    interior = interior[:interior.index("cytoplasm_mask = np.where")]
    assert "organelle_mask" in interior
    for role in ("nucleus_mask", "pathogen_mask"):
        assert role in interior
