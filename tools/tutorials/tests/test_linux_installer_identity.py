"""The installation runner may only execute its verified private asset."""
import hashlib
import json
from pathlib import Path

import pytest


def test_private_receipt_and_exact_installer_bytes_are_required(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from check_linux_installation import verified_installer
    root = tmp_path / 'installation_runs' / 'verified'
    root.mkdir(parents=True)
    installer = root / 'test.run'
    installer.write_bytes(b'# a checksum fixture, never executed\n')
    value = dict(accepted=True, installer=str(installer),
                 sha256=hashlib.sha256(installer.read_bytes()).hexdigest())
    receipt = root / 'receipt.json'
    receipt.write_text(json.dumps(value))
    assert verified_installer(tmp_path, receipt)[0] == installer

    unaccepted = dict(value, accepted=False)
    receipt.write_text(json.dumps(unaccepted))
    with pytest.raises(ValueError, match='exact checksum-verified'):
        verified_installer(tmp_path, receipt)
    receipt.write_text(json.dumps(value))
    installer.write_bytes(installer.read_bytes() + b'# changed\n')
    with pytest.raises(ValueError, match='exact checksum-verified'):
        verified_installer(tmp_path, receipt)

    outside = tmp_path / 'other.run'
    outside.write_bytes(installer.read_bytes())
    escaped = dict(value, installer=str(outside), sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    receipt.write_text(json.dumps(escaped))
    with pytest.raises(ValueError, match='exact checksum-verified'):
        verified_installer(tmp_path, receipt)
    other_receipt = tmp_path / 'other-receipt.json'
    other_receipt.write_text(json.dumps(escaped))
    with pytest.raises(ValueError, match='private release-verification'):
        verified_installer(tmp_path, other_receipt)
