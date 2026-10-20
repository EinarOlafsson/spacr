"""Checkpoint allocation preserves occupied dangling links without spinning."""
import hashlib
import os

import pytest

from spacr import model_zoo as zoo


@pytest.mark.parametrize("requested,occupied,expected", [
    ("m.CP_model", ["m.CP_model"], "m_v2.CP_model"),
    ("m_v2.CP_model", ["m_v2.CP_model"], "m_v3.CP_model"),
    ("m.CP_model", ["m.CP_model", "m_v2.CP_model"], "m_v3.CP_model"),
    ("m", ["m", "m_v2"], "m_v3"),
])
@pytest.mark.needs_symlink
def test_version_allocation_steps_over_dangling_checkpoint_links(
        tmp_path, requested, occupied, expected):
    for name in occupied:
        (tmp_path / name).symlink_to("missing-user-checkpoint")
    assert zoo.versioned_path(tmp_path, requested) == tmp_path / expected
    for name in occupied:
        assert (tmp_path / name).is_symlink()
        assert os.readlink(tmp_path / name) == "missing-user-checkpoint"
    assert not (tmp_path / "missing-user-checkpoint").exists()


@pytest.mark.parametrize("boundary", ["claim", "fetch"])
@pytest.mark.needs_symlink
def test_native_exclusive_claim_is_bounded_and_preserves_dangling_link(
        tmp_path, monkeypatch, boundary):
    dest = tmp_path / "models"
    dest.mkdir()
    link = dest / "m.CP_model"
    link.symlink_to("missing-user-checkpoint")
    body = b"PK\x03\x04exact offline checkpoint bytes"
    original_open = os.open
    attempts = []

    def bounded_open(path, flags, *args, **kwargs):
        if flags & os.O_EXCL and str(path).endswith(".CP_model"):
            if len(attempts) == 2:
                raise RuntimeError("bounded diagnostic: two native exclusive-open refusals")
            attempts.append(str(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", bounded_open)
    if boundary == "claim":
        stage = tmp_path / "owned-stage"
        stage.write_bytes(body)
        landed = zoo._claim(stage, dest, "m.CP_model")
        assert not stage.exists()
    else:
        entry = zoo.ModelEntry(key="exact-model", name="m.CP_model",
                               uri="https://unused.invalid/checkpoint",
                               sha256=hashlib.sha256(body).hexdigest())
        landed = zoo.fetch(entry, dest, opener=lambda uri: ([body], len(body)))
    assert attempts == [str(dest / "m_v2.CP_model")]
    assert landed == dest / "m_v2.CP_model"
    assert landed.read_bytes() == body
    assert zoo.sha256_file(landed) == hashlib.sha256(body).hexdigest()
    assert link.is_symlink()
    assert os.readlink(link) == "missing-user-checkpoint"
    assert not (dest / "missing-user-checkpoint").exists()
    assert sorted(path.name for path in dest.iterdir()) == ["m.CP_model", "m_v2.CP_model"]
