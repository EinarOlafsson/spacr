"""The optional synthetic-tap budget must not bypass or silently miss validation."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="the player checks drive a browser through Playwright")

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('mobile_tap_verifier', TOOLS / 'verify_release_candidate.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '3000'])
def test_invalid_tap_budget_is_rejected_before_network(value, monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid tap budget reached the network')

    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    with pytest.raises(ValueError, match='positive integer'):
        verifier.verify_live_mobile('https://invalid.example/', tap_timeout_ms=value)


def test_cli_forwards_invalid_budget_to_validation():
    result = subprocess.run([sys.executable, str(TOOLS / 'verify_release_candidate.py'),
                             '--live-mobile', 'https://example.invalid/', '--tap-timeout-ms', '0'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'Synthetic tap timeout must be a positive integer' in result.stderr
    assert 'urlopen' not in result.stderr
