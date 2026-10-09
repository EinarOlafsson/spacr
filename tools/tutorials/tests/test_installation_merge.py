from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lesson_redirects import installation_placeholder, LESSON_ALIASES


def test_merge_preserves_number_position_and_old_links_without_renumbering():
    original = {'lessons': [
        {'id': '01_pypi_github', 'number': 1},
        {'id': '02_conda_install', 'number': 2, 'app_key': None},
        {'id': '03_pip_install', 'number': 3, 'app_key': None},
        {'id': '04_platform_installers', 'number': 4},
    ]}
    merged = installation_placeholder(original, original)
    assert [item['number'] for item in merged['lessons']] == [1, 2, 4]
    assert merged['lessons'][1]['id'] == '02_install_spacr'
    assert merged['lesson_aliases'] == LESSON_ALIASES
    assert original['lessons'][1]['id'] == '02_conda_install'
    assert installation_placeholder(merged, original) == merged


def test_merge_requires_both_archive_identities():
    with pytest.raises(ValueError):
        installation_placeholder({'lessons': []}, {'lessons': []})
