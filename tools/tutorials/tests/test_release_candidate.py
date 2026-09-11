"""Copied assets and accepted reports must agree on exact, current bytes."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_release_candidate import copy_checked, require_web_receipt
from check_completed_matrix import digest


@pytest.fixture
def evidence():
    return ({'lesson': 'a', 'accepted': True, 'rendition_sha256': 'hash',
             'all_frame_presentation_times_match': True, 'full_decode_passed': True},
            {'lesson': 'a', 'passed': True, 'checked_web_rendition': {'sha256': 'hash'}})


def test_matching_reports_and_current_video_accepted(evidence):
    require_web_receipt('a', *evidence, 'hash')


@pytest.mark.parametrize('target,key,value', [
    (0, 'lesson', 'wrong'), (0, 'accepted', False), (0, 'rendition_sha256', 'old'),
    (0, 'all_frame_presentation_times_match', False), (0, 'full_decode_passed', False),
    (1, 'lesson', 'wrong'), (1, 'passed', False), (1, 'checked_web_rendition', {'sha256': 'old'}),
])
def test_failed_or_stale_report_rejected(evidence, target, key, value):
    require_web_receipt('a', *evidence, 'hash')
    broken = deepcopy(evidence)
    broken[target][key] = value
    with pytest.raises(ValueError, match='Stale or failed'):
        require_web_receipt('a', *broken, 'hash')


def test_actual_asset_changed_after_measurement_is_rejected(tmp_path):
    source = tmp_path / 'original.mp4'
    source.write_bytes(b'original test asset')
    checksum = digest(source)
    records = []
    copy_checked(source, tmp_path / 'ready/video.mp4', records, tmp_path, checksum)
    assert (tmp_path / 'ready/video.mp4').read_bytes() == source.read_bytes()
    assert records[0]['sha256'] == checksum
    source.write_bytes(b'changed test asset')
    with pytest.raises(ValueError, match='Source changed'):
        copy_checked(source, tmp_path / 'refused/video.mp4', records, tmp_path, checksum)
    assert not (tmp_path / 'refused/video.mp4').exists()
