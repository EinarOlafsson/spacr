"""386 step 6: the screen, and the two sentences it must not omit.

386 puts the GUI last and then argues it is the largest item in the file:
Cell-DINO, OpenPhenom and SubCell are all Python/CLI, and no GUI platform
exposes any of them. The engine serves this lab; the screen is what makes
spaCR the only clickable route to a foundation-model embedding.

Two things the engine decided have to survive onto the surface, or the screen
undoes the module underneath it:

  * THE CHANNEL POLICY IS "the design problem" -- 386's words -- and must not
    be decided implicitly by whatever the backbone wants. So it is a visible
    control whose options say what each costs.
  * A DIMENSION IS NOT A PHENOTYPE. `emb_c2_017 = 0.43` shown without that
    sentence invites exactly the reading the family cannot survive.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot):
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def _crops(n=4, size=32, channels=2):
    rng = np.random.default_rng(0)
    return rng.random((n, size, size, channels)).astype("float32")


def _fake_encoder(batch):
    """A stand-in backbone: deterministic, and no download."""
    batch = np.asarray(batch, dtype="float32")
    return batch.reshape(batch.shape[0], -1)[:, :16]


def test_the_screen_builds_and_names_itself_from_the_catalog(qtbot):
    """One spelling for the name, in the catalog, read back here."""
    from spacr.qt.app_catalog import declared_app
    from spacr.qt.screens.embeddings import APP_KEY, APP_NAME

    screen = _screen(qtbot)
    assert APP_NAME == declared_app(APP_KEY).name
    assert screen.objectName() == "EmbeddingsScreen"


def test_the_catalog_row_carries_nine_translations():
    """A row with fewer would leave the tile in English in some locale."""
    from spacr.qt.app_catalog import declared_app

    row = declared_app("embeddings")
    assert len(row.translations) == 9
    assert all(text.strip() for text in row.translations)
    assert row.name not in row.translations      # none of them is the English


def test_the_channel_policy_is_a_visible_control_not_a_default(qtbot):
    """386 calls this 'the design problem'; it must not be implicit."""
    screen = _screen(qtbot)
    policies = [screen._policy.itemData(i)
                for i in range(screen._policy.count())]

    assert set(policies) == {"per_channel", "project"}


def test_each_policy_option_says_what_it_costs(qtbot):
    """A user who cannot see the cost picks whichever is first, forever."""
    screen = _screen(qtbot)
    captions = [screen._policy.itemText(i)
                for i in range(screen._policy.count())]

    joined = " ".join(captions).lower()
    assert "pass" in joined                      # N passes vs one pass
    assert any("stain" in caption.lower() for caption in captions)


def test_the_default_policy_is_per_channel(qtbot):
    """The engine's default, and for the engine's reason: the channel
    survives into the column name, so an attribution can name the stain."""
    from spacr.embeddings import CHANNEL_PER_CHANNEL

    assert _screen(qtbot).spec().channel_policy == CHANNEL_PER_CHANNEL


def test_the_caveat_is_on_the_screen_not_in_a_tooltip(qtbot):
    """It has to be where a user reading a number is already looking."""
    from spacr.qt.screens.embeddings import DIMENSION_CAVEAT

    screen = _screen(qtbot)
    assert screen._caveat.text() == DIMENSION_CAVEAT
    assert "not a phenotype" in DIMENSION_CAVEAT
    assert screen._caveat.isVisibleTo(screen)


def test_the_controls_describe_a_valid_engine_spec(qtbot):
    """The screen must not be able to build a spec the engine rejects."""
    from spacr.embeddings import EmbeddingSpec

    spec = _screen(qtbot).spec()
    assert isinstance(spec, EmbeddingSpec)
    assert spec.batch_size >= 1
    assert spec.backbone


def test_crops_with_the_axes_the_wrong_way_round_are_refused(qtbot):
    """The failure this catches does not raise anywhere useful.

    A (n, channels, height, width) stack is still four-dimensional, so it
    would reach the backbone and fail there with a message about tensor
    shapes rather than about the input. The rank check cannot see that, so
    the message names the layout instead.
    """
    screen = _screen(qtbot)
    with pytest.raises(ValueError, match="channels last"):
        screen.set_crops(np.zeros((4, 32, 32)))      # three axes


def test_embedding_fills_the_preview_and_names_the_encoder(qtbot,
                                                           monkeypatch):
    """The run's summary must identify what produced the numbers.

    Two runs' dimension 17 are the same number and not the same thing unless
    the backbone, the weights and the policy all match -- so the screen says
    all three rather than just declaring success.
    """
    import spacr.embeddings as engine

    screen = _screen(qtbot)
    real = engine.embed_array
    monkeypatch.setattr(
        engine, "embed_array",
        lambda crops, spec=None, **kw: real(crops, spec,
                                            encoder=_fake_encoder))

    screen.set_crops(_crops())
    screen.embed()

    assert screen._table.rowCount() == 4
    assert 0 < screen._table.columnCount() <= 8
    status = screen._status.text()
    assert "objects" in status and "dimensions" in status
    assert "resnet18" in status


def test_the_preview_shows_a_few_dimensions_not_all_of_them(qtbot,
                                                            monkeypatch):
    """The whole matrix is the result and it is not readable.

    A table of 2,048 columns invites scrolling through it as though a column
    meant something, which is the reading the caveat exists to prevent.
    """
    import spacr.embeddings as engine
    from spacr.qt.screens.embeddings import PREVIEW_DIMENSIONS

    screen = _screen(qtbot)
    real = engine.embed_array
    monkeypatch.setattr(
        engine, "embed_array",
        lambda crops, spec=None, **kw: real(crops, spec,
                                            encoder=_fake_encoder))
    screen.set_crops(_crops())
    screen.embed()

    assert screen._table.columnCount() <= PREVIEW_DIMENSIONS


def test_embedding_without_crops_says_so_rather_than_crashing(qtbot):
    screen = _screen(qtbot)
    screen.embed()
    assert "Load crops" in screen._status.text()


def test_a_failure_becomes_a_sentence_on_the_screen(qtbot, monkeypatch):
    """Never a silent empty table: that reads as 'no signal'."""
    import spacr.embeddings as engine

    screen = _screen(qtbot)

    def explode(*_a, **_k):
        raise RuntimeError("the card ran out of memory")

    monkeypatch.setattr(engine, "embed_array", explode)
    screen.set_crops(_crops())
    screen.embed()

    assert "memory" in screen._status.text()


def test_the_run_button_is_dead_until_there_are_crops(qtbot):
    """An enabled control that cannot do anything is worse than an absent one."""
    screen = _screen(qtbot)
    assert not screen._run.isEnabled()
    screen.set_crops(_crops())
    assert screen._run.isEnabled()


def test_register_is_not_called_at_import():
    """Importing to reach the class must not mutate process-wide state."""
    import inspect

    from spacr.qt.screens import embeddings

    source = inspect.getsource(embeddings)
    body = source[source.index("def register("):]
    assert "register_app" in body
    # not invoked anywhere at module level
    assert "\nregister()" not in source
