"""OPS segments with pinned Cellpose-SAM weights (settings schema 975).

``ops_cellpose_model`` picks the stock weights the OPS ``cpsam`` model name
loads: ``cpsam_v2`` (the new default) by its full path, ``cpsam_v1`` as the
installed Cellpose default, exactly as before. Each file is checked by size
and md5 first, and a missing or different file is refused: Cellpose loads
``MODEL_DIR/cpsam`` for any bare name it does not ship, so a name alone
would silently run v1. Settings files from before schema 975 keep v1.

The weights here are small fake files whose md5 is pinned in their place;
the construction is recorded against the installed ``CellposeModel``
signature, so no weights are loaded.
"""
from __future__ import annotations

import hashlib
import inspect
import os

import pytest

models = pytest.importorskip("cellpose.models")

import spacr.accelerator
from spacr import ops_engine, utils
from spacr.model_zoo import ModelUnreadable
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, _fold_older_defaults


@pytest.fixture
def weights(monkeypatch, tmp_path):
    """Two fake weight files in a fake Cellpose model folder, pinned by md5.

    :returns: ``{choice: absolute path}``.
    """
    paths = {}
    pins = {}
    for choice, name, body in (("cpsam_v1", "cpsam", b"v1 weights"),
                               ("cpsam_v2", "cpsam_v2", b"v2 weights!")):
        path = tmp_path / name
        path.write_bytes(body)
        paths[choice] = str(path)
        pins[choice] = {"file": name, "size": len(body),
                        "md5": hashlib.md5(body).hexdigest()}
    monkeypatch.setattr(utils, "CPSAM_WEIGHTS", pins)
    monkeypatch.setattr(models, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(utils, "_CPSAM_MD5_SEEN", {})
    return paths


@pytest.fixture
def built(monkeypatch):
    """Replace ``CellposeModel`` with a recorder bound to the real signature.

    :returns: the list each construction's passed arguments are appended to.
    """
    signature = inspect.signature(models.CellposeModel.__init__)
    calls = []

    class _RecordingModel:
        """A CellposeModel that loads no weights and remembers how it was built."""

        def __init__(self, *args, **kwargs):
            """Record the arguments as the installed signature binds them."""
            bound = signature.bind(self, *args, **kwargs)
            calls.append({name: value for name, value in bound.arguments.items()
                          if name != "self"})

    monkeypatch.setattr(models, "CellposeModel", _RecordingModel)
    monkeypatch.setattr(spacr.accelerator, "cellpose_kwargs",
                        lambda: pytest.fail("ops_gpu is off; the card was asked"))
    return calls


def test_the_real_pins_are_the_two_cpsam_files():
    """v1 is the July 2025 cpsam file, v2 the cpsam_v2 file, each by md5 and size."""
    assert utils.CPSAM_WEIGHTS["cpsam_v1"] == {
        "file": "cpsam", "size": 1233587898, "md5": "aad12e2d04a28eabee0d210ae23d46a3"}
    assert utils.CPSAM_WEIGHTS["cpsam_v2"] == {
        "file": "cpsam_v2", "size": 1233586851, "md5": "5d43876c953f7ad4a685acc05c959f16"}


@pytest.mark.parametrize("choice", ["cpsam_v1", "cpsam_v2"])
def test_the_resolver_returns_the_full_path_after_reading_its_md5(weights, monkeypatch,
                                                                    choice):
    """The path is absolute, and the file's md5 is read once and then kept."""
    read = []
    real = utils._file_md5
    monkeypatch.setattr(utils, "_file_md5", lambda path: read.append(path) or real(path))

    path = utils.cpsam_weights(choice)
    assert path == weights[choice] and os.path.isabs(path)
    assert utils.cpsam_weights(f" {choice} ") == path
    assert read == [path]


def test_different_bytes_of_the_same_size_are_refused(weights):
    """A file whose md5 is not the pinned one is not loaded."""
    with open(weights["cpsam_v2"], "wb") as handle:
        handle.write(b"v1 weights?")
    with pytest.raises(ModelUnreadable, match="md5"):
        utils.cpsam_weights("cpsam_v2")


def test_a_file_of_another_size_is_refused(weights):
    """The v1 file sitting where v2 should be is refused by its size."""
    with open(weights["cpsam_v2"], "wb") as handle:
        handle.write(b"v1 weights")
    with pytest.raises(ModelUnreadable, match="bytes"):
        utils.cpsam_weights("cpsam_v2")


def test_a_missing_file_is_refused_and_nothing_is_built(weights, built):
    """No v2 file: the run stops; Cellpose is never handed a name to substitute."""
    os.remove(weights["cpsam_v2"])
    with pytest.raises(ModelUnreadable, match="missing"):
        ops_engine._cellpose_model({"ops_cellpose_model": "cpsam_v2"}, gpu=False)
    assert built == []


def test_an_unknown_choice_is_refused(weights):
    """Only the pinned choices resolve; a bare file name is not one of them."""
    with pytest.raises(ValueError, match="ops_cellpose_model"):
        utils.cpsam_weights("cpsam_v3")


@pytest.mark.parametrize("name", [None, "", "cpsam"])
def test_cpsam_v1_builds_exactly_the_old_call(weights, built, name):
    """v1 passes no ``pretrained_model``, as the validated wells were built."""
    ops_engine._cellpose_model({"ops_sbs_nucleus_model": name,
                                "ops_cellpose_model": "cpsam_v1"}, gpu=False)
    assert built == [{"gpu": False, "use_bfloat16": False}]


@pytest.mark.parametrize("key", ["ops_sbs_nucleus_model", "ops_ph_nucleus_model",
                                 "ops_ph_cell_model"])
def test_v2_is_the_default_and_loads_by_full_path_for_every_ops_model(weights, built, key):
    """Sequencing nuclei, phenotype nuclei and cells all go through the one resolver."""
    ops_engine._cellpose_model({key: "cpsam"}, gpu=False, key=key)
    assert built == [{"gpu": False, "use_bfloat16": False,
                      "pretrained_model": weights["cpsam_v2"]}]


def test_the_phenotype_pair_loads_v2_once(weights, built):
    """Same nucleus and cell model: one v2 model, by full path."""
    held = {}
    pair = ops_engine._ph_model_pair({"ops_cellpose_model": "cpsam_v2"}, False, held)
    assert pair[0] is pair[1]
    assert built == [{"gpu": False, "use_bfloat16": False,
                      "pretrained_model": weights["cpsam_v2"]}]


def test_a_checkpoint_is_not_affected(weights, built, tmp_path):
    """A model named by path loads that path whatever the stock choice is."""
    checkpoint = tmp_path / "ops_nuclei"
    checkpoint.write_bytes(b"weights")
    ops_engine._cellpose_model({"ops_sbs_nucleus_model": str(checkpoint),
                                "ops_cellpose_model": "cpsam_v2"}, gpu=False)
    assert built == [{"gpu": False, "use_bfloat16": False,
                      "pretrained_model": str(checkpoint)}]


def test_files_before_975_replay_v1(weights, built):
    """An older settings file gets cpsam_v1 and so the old call; a new one keeps v2."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert SETTINGS_SCHEMA_VERSION >= 975
    assert DEFAULT_CHANGES[975] == {"ops_cellpose_model": "cpsam_v1"}
    assert OPS_DEFAULTS["ops_cellpose_model"] == "cpsam_v2"

    old_file = {"ops_sbs_nucleus_model": "cpsam"}
    applied = _fold_older_defaults(old_file, 965, known=set(OPS_DEFAULTS))
    assert applied["ops_cellpose_model"] == ("cpsam_v1", 975)
    ops_engine._cellpose_model(old_file, gpu=False)
    assert built == [{"gpu": False, "use_bfloat16": False}]

    assert "ops_cellpose_model" not in _fold_older_defaults(
        {"ops_sbs_nucleus_model": "cpsam"}, 975, known=set(OPS_DEFAULTS))
    chosen = {"ops_cellpose_model": "cpsam_v2"}
    _fold_older_defaults(chosen, 965, known=set(OPS_DEFAULTS))
    assert chosen["ops_cellpose_model"] == "cpsam_v2"


def test_the_window_key_and_live_cache_tell_the_weights_apart():
    """A window segmented under v1 is not reused for v2; v1 keeps its old key."""
    basis = {"ops_sbs_nucleus_model": "cpsam"}
    v1 = ops_engine._objects_window_basis({**basis, "ops_cellpose_model": "cpsam_v1"},
                                          "average", {}, (10, 10), (5, 5))
    v2 = ops_engine._objects_window_basis({**basis, "ops_cellpose_model": "cpsam_v2"},
                                          "average", {}, (10, 10), (5, 5))
    assert "weights" not in v1["model"]
    assert v2["model"]["weights"] == "cpsam_v2"
    from spacr import ops_live

    assert set(ops_live._steps_of("ops_cellpose_model", "Nuclei segmentation")) == {
        "spots", "objects"}


def test_phenotype_masks_of_two_weights_are_not_mixed(tmp_path):
    """The option marker of a v1 folder refuses a v2 rerun; v1 keeps its old marker."""
    old = {"mask_eval_batch_size": 0, "ops_ph_tile_batch": 8,
           "mask_inference_precision": "auto"}
    ops_engine._ph_options_marker(str(tmp_path), old)
    (tmp_path / "site-1.npz").write_bytes(b"")
    ops_engine._ph_options_marker(str(tmp_path), old)
    with pytest.raises(ValueError):
        ops_engine._ph_options_marker(str(tmp_path), {**old, "cellpose_weights": "cpsam_v2"})


def test_v1_refuses_a_default_model_cellpose_found_elsewhere(weights, monkeypatch, tmp_path):
    """Cellpose reads a file named cpsam in the working folder first; that is refused."""
    stray = tmp_path / "elsewhere" / "cpsam"

    class _LoadsElsewhere:
        """A model that reports loading a file other than the verified one."""

        def __init__(self, **kwargs):
            """Report the stray file as loaded."""
            self.pretrained_model = str(stray)

    monkeypatch.setattr(models, "CellposeModel", _LoadsElsewhere)
    with pytest.raises(ModelUnreadable, match="working folder"):
        ops_engine._cellpose_model({"ops_cellpose_model": "cpsam_v1"}, gpu=False)

    class _LoadsTheFile(_LoadsElsewhere):
        """A model that reports loading the verified file."""

        def __init__(self, **kwargs):
            """Report the verified v1 file as loaded."""
            self.pretrained_model = weights["cpsam_v1"]

    monkeypatch.setattr(models, "CellposeModel", _LoadsTheFile)
    model = ops_engine._cellpose_model({"ops_cellpose_model": "cpsam_v1"}, gpu=False)
    assert model.pretrained_model == weights["cpsam_v1"]
