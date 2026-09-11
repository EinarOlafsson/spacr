"""386 step 3: the encoder is a model, so it goes in the zoo with a checksum.

386: "Register the encoder in `spacr/model_zoo.py` like any other published
model, with its checksum. Instruction 370's scorecard applies: an embedding
that ships without one is a black box twice over."

The interesting part is what an encoder's provenance IS. It has no spaCR
checkpoint -- the weights are public, resolved by timm and cached by
HuggingFace -- so "its checksum" can only mean the digest of the bytes on
this machine. That is also the only thing a later run can be made to match,
which is what makes it the right thing to record.
"""

import pytest

from spacr.embeddings import (
    CHANNEL_PER_CHANNEL, CHANNEL_PROJECT, DEFAULT_BACKBONE,
    ENCODER_KEY_PREFIX, EmbeddingSpec, encoder_entry, encoder_key,
)
from spacr.model_zoo import KINDS, ModelEntry


def test_encoder_is_a_kind_the_zoo_knows():
    """A separate kind, because nothing that eats a classifier eats one."""
    assert "encoder" in KINDS


def test_the_entry_is_an_ordinary_zoo_entry():
    """'like any other published model' -- the same record type, not a cousin."""
    entry = encoder_entry()
    assert isinstance(entry, ModelEntry)
    assert entry.kind == "encoder"
    assert entry.key.startswith(ENCODER_KEY_PREFIX)
    assert DEFAULT_BACKBONE in entry.name


def test_the_key_names_the_policy_as_well_as_the_backbone():
    """The two policies are not comparable and must not share an id.

    Same backbone, same column count, same dtype -- and one is per-stain
    while the other is a mixture. A key naming only the backbone would let a
    per-channel run and a projected run be compared silently, which is the
    one confusion this family's whole design exists to prevent.
    """
    per_channel = encoder_key(EmbeddingSpec(channel_policy=CHANNEL_PER_CHANNEL))
    projected = encoder_key(EmbeddingSpec(channel_policy=CHANNEL_PROJECT))

    assert per_channel != projected
    assert CHANNEL_PER_CHANNEL in per_channel
    assert CHANNEL_PROJECT in projected


def test_the_policy_is_stated_in_the_notes_too():
    """A reader of the zoo listing sees it without parsing the key."""
    entry = encoder_entry(EmbeddingSpec(channel_policy=CHANNEL_PROJECT))
    assert any("not comparable" in note for note in entry.notes)


def test_a_missing_scorecard_is_said_out_loud():
    """370's rule, applied: silence would make it a black box twice over."""
    entry = encoder_entry()
    assert entry.metrics == {}
    assert any("black box twice over" in note for note in entry.notes)


def test_a_scorecard_is_carried_where_370_reads_it():
    """Attached as metrics, which is where `scorecard_lines` looks."""
    numbers = {"knn_accuracy": 0.82, "knn_accuracy_measured_panel": 0.61,
               "n_objects": 12517}
    entry = encoder_entry(scorecard=numbers)

    assert entry.metrics == numbers
    assert not any("black box twice over" in note for note in entry.notes)


def test_the_checksum_is_of_the_bytes_on_this_machine():
    """Not a published digest taken on trust.

    ModelEntry's own contract: "for a downloaded model this is the digest of
    the bytes that were actually written". An encoder arrives through the
    HuggingFace cache, so the local bytes are the only checkable thing.
    """
    entry = encoder_entry()
    if not entry.sha256:
        pytest.skip("pretrained weights are not cached on this machine")

    assert len(entry.sha256) == 64
    assert entry.size_bytes > 0
    assert entry.path
    # hashing the same file twice must agree, or the record means nothing
    assert encoder_entry().sha256 == entry.sha256


def test_an_uncached_encoder_admits_it_rather_than_inventing_a_digest(
        monkeypatch):
    """An empty digest is a refusal downstream; a fabricated one is a lie.

    `spacr.model_zoo.fetch` already treats '' as a refusal rather than a
    pass, so admitting the gap routes into behaviour that already exists.
    """
    import spacr.embeddings as module

    monkeypatch.setattr(module, "_weights_on_disk", lambda backbone: ("", "", 0))
    entry = encoder_entry()

    assert entry.sha256 == ""
    assert entry.source == "remote"
    assert any("No checksum" in note for note in entry.notes)


def test_the_entry_survives_a_machine_with_no_timm(monkeypatch):
    """Describing the encoder must not require being able to run it.

    The zoo is browsable without torch -- that is the premise `spacr.scorecard`
    was written for -- so an entry has to be constructible there too.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.split(".")[0] in {"timm", "huggingface_hub"}:
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    entry = encoder_entry()

    assert entry.kind == "encoder"
    assert entry.sha256 == ""
    assert any("No checksum" in note for note in entry.notes)


def test_the_uri_says_where_the_weights_come_from():
    """A reader should not have to know timm's resolution rules to find them."""
    entry = encoder_entry()
    assert entry.uri == f"timm:{DEFAULT_BACKBONE}"


def test_provenance_never_reads_as_no_constraints():
    """ModelEntry replaces a blank with UNKNOWN; assert we did not defeat it."""
    entry = encoder_entry()
    assert entry.trained_on and entry.trained_by
    assert entry.trained_on.strip() != ""
