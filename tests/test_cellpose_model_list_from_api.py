"""``Z12``/``A5`` — the Cellpose model list comes from Cellpose.

It used to be a literal in four places, which meant spaCR could be wrong
in both directions: it offered models Cellpose 4 had removed, and it could
not offer a checkpoint the user had registered with
``cellpose.io.add_model``.

:func:`spacr.settings.cellpose_model_choices` reads
``cellpose.models.MODEL_NAMES`` and ``cellpose.models.get_user_models()``
instead. The awkward part is *when*: importing ``cellpose.models`` pulls in
torch and costs ~2.5 s, and the accessor is called while a settings page is
being laid out. So it reads the API only when Cellpose is already imported
and degrades to :data:`spacr.settings.CELLPOSE_MODEL_CHOICES` otherwise —
the same bargain ``settings_spec._torchvision_model_names`` already strikes
for the torchvision zoo.

Two properties matter more than the contents: the list is never empty, and
it never costs the import.
"""
from __future__ import annotations

import sys
import types

import pytest

import spacr.settings as S


@pytest.fixture(autouse=True)
def _clear_cache():
    """The accessor caches per process; every test starts cold."""
    S._CELLPOSE_MODELS_CACHE = None
    yield
    S._CELLPOSE_MODELS_CACHE = None


def _fake_cellpose(monkeypatch, *, names, user=(), user_raises=False):
    """Install a stand-in ``cellpose.models`` in ``sys.modules``."""
    module = types.ModuleType("cellpose.models")
    module.MODEL_NAMES = list(names)

    def get_user_models():
        if user_raises:
            raise OSError("gui_models.txt is a directory")
        return list(user)

    module.get_user_models = get_user_models
    package = types.ModuleType("cellpose")
    package.models = module
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", module)
    return module


# ---------------------------------------------------------------------------
# It reads the API
# ---------------------------------------------------------------------------

def test_the_stock_models_come_from_the_api(monkeypatch):
    _fake_cellpose(monkeypatch, names=["cpsam"])
    assert S.cellpose_model_choices() == ("cpsam",)


def test_a_model_cellpose_adds_later_shows_up_without_a_spacr_release(
        monkeypatch):
    """The whole point: the list is Cellpose's, not spaCR's."""
    _fake_cellpose(monkeypatch, names=["cpsam", "cpsam_v5"])
    assert S.cellpose_model_choices() == ("cpsam", "cpsam_v5")


def test_a_user_registered_checkpoint_is_offered(monkeypatch):
    """``cellpose.io.add_model`` writes the registry; this reads it."""
    _fake_cellpose(monkeypatch, names=["cpsam"], user=["my_cells"])
    assert S.cellpose_model_choices() == ("cpsam", "my_cells")


def test_the_default_model_is_always_first(monkeypatch):
    """Whatever order Cellpose reports, cpsam is the one preselected."""
    _fake_cellpose(monkeypatch, names=["zebra", "cpsam"], user=["mine"])
    assert S.cellpose_model_choices()[0] == "cpsam"


def test_duplicates_between_the_stock_and_user_lists_collapse(monkeypatch):
    _fake_cellpose(monkeypatch, names=["cpsam"], user=["cpsam", "mine"])
    assert S.cellpose_model_choices() == ("cpsam", "mine")


# ---------------------------------------------------------------------------
# It degrades rather than failing
# ---------------------------------------------------------------------------

def test_no_cellpose_at_all_falls_back_to_the_shipped_list(monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    assert S.cellpose_model_choices(block=True) == S.CELLPOSE_MODEL_CHOICES


def test_a_broken_user_registry_still_yields_the_stock_models(monkeypatch):
    """A malformed gui_models.txt must not cost the stock list too."""
    _fake_cellpose(monkeypatch, names=["cpsam"], user_raises=True)
    assert S.cellpose_model_choices() == ("cpsam",)


def test_an_api_that_reports_nothing_falls_back(monkeypatch):
    """Empty is not an answer — a dropdown with nothing in it is a dead end."""
    _fake_cellpose(monkeypatch, names=[], user=[])
    assert S.cellpose_model_choices() == S.CELLPOSE_MODEL_CHOICES


def test_the_fallback_is_never_cached(monkeypatch):
    """A miss because Cellpose is not loaded yet must not pin the fallback."""
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    assert S.cellpose_model_choices() == S.CELLPOSE_MODEL_CHOICES
    assert S._CELLPOSE_MODELS_CACHE is None

    _fake_cellpose(monkeypatch, names=["cpsam", "later"])
    assert S.cellpose_model_choices() == ("cpsam", "later")


def test_refresh_asks_again(monkeypatch):
    _fake_cellpose(monkeypatch, names=["cpsam"])
    assert S.cellpose_model_choices() == ("cpsam",)
    _fake_cellpose(monkeypatch, names=["cpsam"], user=["mine"])
    assert S.cellpose_model_choices() == ("cpsam",), "the cache did not hold"
    assert S.cellpose_model_choices(refresh=True) == ("cpsam", "mine")


# ---------------------------------------------------------------------------
# It does not cost the import
# ---------------------------------------------------------------------------

def test_building_a_settings_page_does_not_import_cellpose():
    """The measured constraint. ~2.5 s per settings page is not acceptable.

    Run in a clean interpreter because the test session has almost
    certainly imported torch by now.
    """
    import subprocess
    code = (
        "import sys;"
        "from spacr.settings_spec import convert_settings_dict_for_gui as f;"
        "spec = f({'model_name': 'cpsam'});"
        "assert spec['model_name'][1], 'the model combo is empty';"
        "assert 'cellpose.models' not in sys.modules, 'cellpose was imported';"
        "assert 'torch' not in sys.modules, 'torch was imported';"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ---------------------------------------------------------------------------
# What a dropdown offers
# ---------------------------------------------------------------------------

def test_the_menu_keeps_the_legacy_spellings(monkeypatch):
    """A saved ``cyto2`` has to be visible in the combo, not silently swapped.

    They are not four choices — Cellpose resolves all of them to cpsam,
    which is what :func:`spacr.settings.normalize_cellpose_model_name` is
    for — but a value the user's settings file holds must be selectable.
    """
    _fake_cellpose(monkeypatch, names=["cpsam"])
    menu = S.cellpose_model_menu()
    assert menu[0] == "cpsam"
    for legacy in ("cyto3", "cyto2", "nuclei"):
        assert legacy in menu


def test_the_menu_never_repeats_a_name_the_api_already_reported(monkeypatch):
    _fake_cellpose(monkeypatch, names=["cpsam", "nuclei"])
    menu = S.cellpose_model_menu()
    assert menu.count("nuclei") == 1


def test_the_shipped_constant_is_still_the_cellpose_4_answer():
    """The fallback is a fallback, not a second source of truth."""
    assert S.CELLPOSE_MODEL_CHOICES == ("cpsam",)
