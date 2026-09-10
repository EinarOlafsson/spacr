"""Exact asset and checksum selection must fail closed."""
from pathlib import Path

import pytest


@pytest.fixture
def checks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import check_release_installer
    return check_release_installer


def test_checksum_requires_the_exact_filename_once(checks):
    digest = 'a' * 64
    manifest = digest + '  release.run\n' + 'b' * 64 + ' *another.run\n'
    assert checks.checksum_for(manifest, 'release.run') == digest
    assert checks.checksum_for(manifest, 'another.run') == 'b' * 64
    with pytest.raises(ValueError, match='Exactly one checksum'):
        checks.checksum_for(manifest, 'missing.run')
    with pytest.raises(ValueError, match='Exactly one checksum'):
        checks.checksum_for(manifest + digest + ' *release.run\n', 'release.run')


def test_malformed_checksum_is_not_accepted(checks):
    with pytest.raises(ValueError, match='Exactly one checksum'):
        checks.checksum_for('a' * 63 + '  release.run\n', 'release.run')


@pytest.fixture
def asset():
    return dict(name='release.run', size=1000,
                browser_download_url='https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.5/release.run')


def test_asset_requires_the_named_public_release(checks, asset):
    assert checks.release_asset([asset], 'release.run', 'v1.5.0.5') == asset
    with pytest.raises(ValueError, match='missing or ambiguous'):
        checks.release_asset([], 'release.run', 'v1.5.0.5')
    with pytest.raises(ValueError, match='missing or ambiguous'):
        checks.release_asset([asset, asset], 'release.run', 'v1.5.0.5')
    with pytest.raises(ValueError, match='Unexpected release'):
        checks.release_asset([asset], 'release.run', 'v1.5.0.4')


@pytest.mark.parametrize('change', [
    dict(size=0), dict(size=30 * 1024**2),
    dict(browser_download_url='http://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.5/release.run'),
    dict(browser_download_url='https://other.example/release.run'),
])
def test_unexpected_origin_or_oversized_asset_is_rejected(checks, asset, change):
    asset.update(change)
    with pytest.raises(ValueError, match='Unexpected release'):
        checks.release_asset([asset], 'release.run', 'v1.5.0.5')
