"""386 step 4: the emb_ family reaches the pickers, and keeps its channel.

Step 4 asks for "a documented prefix so embedding dimensions never collide
with measured features, and so `spacr/columns.py` and `spacr/column_groups.py`
can group them for the UI and for `spacr/attribution_columns.py`". The prefix
and the naming were done with the engine; this is the wiring that makes the
rest of that sentence true.

The wiring point is `spacr.feature_dict.parse_column`, because
`column_groups` deliberately has no taxonomy of its own -- "A second taxonomy
would be a second thing to keep in step with the measurement code, and it
would disagree first in exactly the corners nobody checks." So an embedding
column that parse_column cannot classify is invisible to every picker, and
that is what these tests hold.
"""

import pytest

from spacr.column_groups import classify, group_names
from spacr.embeddings import EMBEDDING_PREFIX, embedding_column_names
from spacr.feature_dict import FEATURE_FAMILIES, parse_column


def test_the_family_is_documented_like_every_other():
    """A family with no entry here is a family the UI cannot explain."""
    assert "embedding" in FEATURE_FAMILIES
    text = FEATURE_FAMILIES["embedding"]
    assert "spacr.embeddings" in text
    # the caveat that matters most for a reader who finds one in a table
    assert "never be read as a phenotype" in text


def test_a_per_channel_dimension_keeps_its_channel():
    """The whole argument for the per-channel default.

    Per-channel encoding costs N forward passes and buys one thing: an
    attribution that can say WHICH STAIN carried the signal. It can only say
    that if the column still knows its channel after parsing.
    """
    entry = parse_column("emb_c2_017")
    assert entry.family == "embedding"
    assert entry.channel == 2
    assert entry.computed_by == "spacr.embeddings.embed_array"


def test_a_projected_dimension_has_no_channel_and_says_so():
    """Projection mixes the channels, so there is no channel to report.

    Reporting one anyway -- 0, or the first -- would let an attribution claim
    a stain that did not exist as a separate input.
    """
    entry = parse_column("emb_rgb_004")
    assert entry.family == "embedding"
    assert entry.channel is None
    assert "projected" in (entry.description or "")


def test_every_generated_column_name_parses_back():
    """The generator and the parser must agree, in both policies.

    They are written in different modules against the same grammar, which is
    exactly the arrangement that drifts.
    """
    from spacr.embeddings import CHANNEL_PER_CHANNEL, CHANNEL_PROJECT

    channels = (0, 1, 2, 3)
    for column in embedding_column_names(8, channels, CHANNEL_PER_CHANNEL):
        entry = parse_column(column)
        assert entry.family == "embedding", column
        assert entry.channel in channels, column

    for column in embedding_column_names(8, channels, CHANNEL_PROJECT):
        entry = parse_column(column)
        assert entry.family == "embedding", column
        assert entry.channel is None, column


def test_the_family_is_offered_as_a_group():
    """Step 4's stated purpose: column_groups can group them for the UI."""
    columns = ["emb_c2_017", "emb_c2_018", "emb_c3_001",
               "cell_channel_1_mean_intensity", "cell_area"]
    families = classify(columns)["family"]

    assert set(families["embedding"]) == {"emb_c2_017", "emb_c2_018",
                                          "emb_c3_001"}
    assert "embedding" in group_names(columns)["family"]


def test_per_channel_dimensions_group_by_channel_too():
    """A user asking for 'channel 2' should get its embedding dimensions.

    This is the second half of keeping the channel: not just recoverable from
    the name, but actually joined to the channel group the measured columns
    already use, so the two feature sources are selectable together.
    """
    columns = ["emb_c2_017", "emb_c3_001", "cell_channel_2_mean_intensity"]
    channels = classify(columns)["channel"]

    assert set(channels["channel_2"]) == {"emb_c2_017",
                                          "cell_channel_2_mean_intensity"}
    assert set(channels["channel_3"]) == {"emb_c3_001"}


def test_an_embedding_column_is_never_mistaken_for_a_measurement():
    """The collision the prefix exists to prevent, asserted rather than assumed."""
    measured = parse_column("cell_channel_1_mean_intensity")
    embedded = parse_column("emb_c1_000")

    assert measured.family == "intensity"
    assert embedded.family == "embedding"
    assert measured.family != embedded.family


def test_the_whole_prefix_is_claimed_even_when_malformed():
    """A deliberate choice, recorded here so it is not read as an accident.

    `emb_` is a reserved prefix. A name carrying it that does not match the
    grammar is still classified as an embedding rather than falling through
    to `unknown` -- because `unknown` columns stay available as individual
    features, and a malformed embedding dimension offered as a measured
    feature is the worse of the two failures.
    """
    entry = parse_column(f"{EMBEDDING_PREFIX}bogus")
    assert entry.family == "embedding"
    assert entry.channel is None


def test_describing_a_column_never_needs_the_engine(monkeypatch):
    """The dictionary must work where the embedding engine cannot import.

    A table can be described on a machine with no torch -- that is the same
    reason `spacr.scorecard` uses the standard library -- so the import is
    lazy and its failure is handled rather than raised.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if "embeddings" in name:
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    entry = parse_column("emb_c2_017")
    assert entry.family == "embedding"
    assert entry.channel == 2          # recovered without the engine


def test_a_note_warns_that_dimensions_are_not_portable():
    """Two runs' dimension 17 are the same number and not the same thing."""
    entry = parse_column("emb_c2_017")
    assert entry.notes and "not comparable between runs" in entry.notes
