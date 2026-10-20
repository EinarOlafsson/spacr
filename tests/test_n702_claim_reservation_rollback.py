"""Failed checkpoint publication rolls back only its own reservation."""
import errno
import hashlib
import os

import pytest

from spacr import model_zoo as zoo


def _fetch(root, body):
    entry = zoo.ModelEntry(key="exact-offline", name="model.CP_model",
                           uri="https://unused.invalid/weights",
                           sha256=hashlib.sha256(body).hexdigest())
    return zoo.fetch(entry, root, opener=lambda uri: ([body], len(body)))


@pytest.mark.parametrize("boundary", ["claim", "fetch"])
@pytest.mark.parametrize("failure", ["io-error", "interruption"])
def test_failed_replace_removes_own_reservation_and_retains_original_error(
        tmp_path, monkeypatch, boundary, failure):
    root = tmp_path / "models"
    root.mkdir()
    body = b"PK\x03\x04exact original checkpoint bytes"
    stage = tmp_path / "owned-stage"
    stage.write_bytes(body)
    problem = OSError(errno.EIO, "injected rename failure") if failure == "io-error" else KeyboardInterrupt()

    def interrupted(source, target):
        assert (root / "model.CP_model").read_bytes() == b""
        raise problem

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(type(problem)) as raised:
        if boundary == "claim":
            zoo._claim(stage, root, "model.CP_model")
        else:
            _fetch(root, body)
    assert raised.value is problem
    assert stage.read_bytes() == body
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("foreign", [
    "regular", pytest.param("dangling-link", marks=pytest.mark.needs_symlink)])
def test_failed_fetch_never_removes_foreign_replacement(
        tmp_path, monkeypatch, foreign):
    root = tmp_path / "models"
    root.mkdir()
    replacement = tmp_path / "pre-existing-foreign"
    if foreign == "regular":
        replacement.write_bytes(b"foreign original bytes")
    else:
        replacement.symlink_to("missing-user-checkpoint")
    foreign_stat = replacement.lstat()
    native_replace = os.replace
    problem = OSError(errno.EIO, "injected error after foreign replacement")

    def interrupted(source, target):
        native_replace(replacement, target)
        raise problem

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError) as raised:
        _fetch(root, b"PK\x03\x04offline weights")
    assert raised.value is problem
    target = root / "model.CP_model"
    assert os.path.samestat(target.lstat(), foreign_stat)
    if foreign == "regular":
        assert target.read_bytes() == b"foreign original bytes"
    else:
        assert target.is_symlink()
        assert os.readlink(target) == "missing-user-checkpoint"
        assert not (root / "missing-user-checkpoint").exists()
    assert [p.name for p in root.iterdir()] == ["model.CP_model"]
