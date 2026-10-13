"""Library reconciliation cannot hide an absent lesson or stale retained media."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_library_checkpoint import require_partition, require_retention_receipt


def test_real_explicit_dispositions_preserve_every_remaining_lesson():
    assert require_partition(['new', 'held', 'kept'], {'held'}, {'kept'}) == ['new']
    assert require_partition(['new', 'held', 'kept'], set(), set()) == ['new', 'held', 'kept']


@pytest.mark.parametrize('ids,held,retained', [
    (['a', 'a'], set(), set()), (['a'], {'typo'}, set()),
    (['a'], set(), {'typo'}), (['a'], {'a'}, {'a'}),
])
def test_unknown_duplicate_or_overlapping_dispositions_rejected(ids, held, retained):
    with pytest.raises(ValueError):
        require_partition(ids, held, retained)


@pytest.fixture
def evidence():
    current = {'tracks': [{'language': 'en', 'voice': 'v', 'audio_sha256': 'a',
                           'timing_sha256': 't'}],
               'media': {'video': {'sha256': 'm'}}, 'catalogs': [{'sha256': 'c'}]}
    saved = deepcopy(current)
    saved.update(passed=True, video_fully_decoded=True, tracks_checked=1)
    saved['tracks'][0]['errors'] = []
    return saved, current


def test_exact_retained_receipt_is_accepted(evidence):
    require_retention_receipt(*evidence)


@pytest.mark.parametrize('change', [
    lambda r: r.update(passed=False), lambda r: r.update(video_fully_decoded=False),
    lambda r: r.update(tracks_checked=0), lambda r: r.update(media={}),
    lambda r: r.update(catalogs=[]), lambda r: r.update(tracks=[]),
    lambda r: r['tracks'][0].update(errors=['decode failure']),
    lambda r: r['tracks'][0].update(audio_sha256='changed'),
    lambda r: r['tracks'][0].update(timing_sha256='changed'),
    lambda r: r['tracks'][0].update(voice='different'),
])
def test_actual_failed_or_stale_retained_receipt_is_rejected(evidence, change):
    saved, current = evidence
    change(saved)
    with pytest.raises(ValueError):
        require_retention_receipt(saved, current)
