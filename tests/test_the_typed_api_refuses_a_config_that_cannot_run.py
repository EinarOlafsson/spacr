"""The typed API's refusals, which happen before a run starts rather than during.

Each of these is checked when ``to_settings`` is called, so a script that
assembled a nonsense configuration is stopped at the line that built it. The
alternative is a pipeline that starts, reads images, and fails an hour later --
which is where the same mistakes used to surface.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# _source_value — one path or several
# ---------------------------------------------------------------------------

def test_a_single_source_becomes_a_string():
    """Both a str and a Path, since a caller may hand either."""
    from spacr.api import _source_value

    assert _source_value("/data/plate1") == "/data/plate1"
    assert _source_value(Path("/data/plate1")) == "/data/plate1"


def test_several_sources_become_a_list_of_strings():
    """The other branch: a multi-plate run names each folder.

    Every element is stringified, so a list of Paths -- which is what
    ``Path.glob`` returns -- does not reach the settings as Path objects that
    a later json.dump would refuse.
    """
    from spacr.api import _source_value

    out = _source_value([Path("/data/plate1"), "/data/plate2"])

    assert out == ["/data/plate1", "/data/plate2"]
    assert all(isinstance(item, str) for item in out)


# ---------------------------------------------------------------------------
# MaskConfig.to_settings
# ---------------------------------------------------------------------------

def test_a_mask_config_with_no_segmentation_channel_is_refused():
    """Nothing to segment, so there is nothing for Mask to do.

    Caught here rather than after the images are read, which is where it
    surfaced before -- as an empty mask folder and no explanation.
    """
    from spacr.api import MaskConfig

    with pytest.raises(ValueError, match="at least one segmentation channel"):
        MaskConfig(src="/data/plate1").to_settings()


@pytest.mark.parametrize("style", ["v3", "", "V1", None, 1])
def test_a_pipeline_style_that_is_not_v1_or_v2_is_refused(style):
    """The vocabulary, including the near-misses.

    'V1' is the spelling a user reaches for, and accepting it silently would
    run the other pipeline -- a different mask for the same settings.
    """
    from spacr.api import MaskConfig

    with pytest.raises(ValueError, match="pipeline_style"):
        MaskConfig(src="/data/plate1", cell_channel=0,
                   pipeline_style=style).to_settings()


def test_a_complete_mask_config_produces_settings():
    """The valid case, so the refusals above are visibly the exceptions."""
    from spacr.api import MaskConfig

    for field in ("test_mode", "dry_run"):
        assert f":param {field}:" in (MaskConfig.__doc__ or "")
    settings = MaskConfig(src="/data/plate1", cell_channel=0,
                          pipeline_style="v2").to_settings()

    assert settings["src"] == "/data/plate1"
    assert settings["cell_channel"] == 0


def test_one_named_channel_is_enough():
    """Each of the three on its own, since a screen may segment only one."""
    from spacr.api import MaskConfig

    for field in ("cell_channel", "nucleus_channel", "pathogen_channel"):
        settings = MaskConfig(src="/data/plate1", **{field: 1}).to_settings()
        assert settings[field] == 1


# ---------------------------------------------------------------------------
# MeasureConfig.to_settings
# ---------------------------------------------------------------------------

def test_a_measure_config_with_no_mask_plane_is_refused():
    """Nothing to measure.

    The defaults name all three planes, so reaching this needs every one
    cleared -- which is what a user does when they mean "measure the image"
    and there is no such mode.
    """
    from spacr.api import MeasureConfig

    with pytest.raises(ValueError, match="at least one mask plane"):
        MeasureConfig(src="/data/plate1", cell_mask_dim=None,
                      nucleus_mask_dim=None,
                      pathogen_mask_dim=None).to_settings()


def test_a_measure_config_with_one_plane_is_accepted():
    """One plane is a measurement; the other two may legitimately be absent."""
    from spacr.api import MeasureConfig

    for field in ("save_png", "test_mode", "dry_run", "resume"):
        assert f":param {field}:" in (MeasureConfig.__doc__ or "")
    settings = MeasureConfig(src="/data/plate1", cell_mask_dim=4,
                             nucleus_mask_dim=None,
                             pathogen_mask_dim=None).to_settings()

    assert settings["cell_mask_dim"] == 4


# ---------------------------------------------------------------------------
# run_mask — a mapping instead of a config
# ---------------------------------------------------------------------------

def test_run_mask_accepts_a_plain_mapping_and_copies_it(monkeypatch):
    """The ``dict(config)`` branch, which is the escape hatch for old scripts.

    It COPIES rather than passing the caller's mapping through, so the
    pipeline's own edits do not reach back into the caller's dict -- a script
    reusing one settings object for two plates would otherwise carry the first
    run's mutations into the second.
    """
    from spacr import api

    seen = {}

    def fake_pipeline(settings):
        seen.update(settings)
        settings["added_by_the_pipeline"] = True
        return None

    monkeypatch.setattr("spacr.core.preprocess_generate_masks", fake_pipeline)

    mine = {"src": "/data/plate1", "cell_channel": 0}
    api.run_mask(mine)

    assert seen["src"] == "/data/plate1"
    assert "added_by_the_pipeline" not in mine


def test_extra_that_repeats_a_typed_setting_is_refused_by_name():
    """_merge_extra's raise, which names every clash it found.

    ``extra`` is the escape hatch for settings the dataclass does not model,
    and a key that IS modelled would be set twice with no rule about which
    wins. Naming them all matters: a user fixing one at a time would run the
    same failure once per key.
    """
    from spacr.api import MaskConfig

    with pytest.raises(ValueError) as excinfo:
        MaskConfig(src="/data/plate1", cell_channel=0,
                   extra={"cell_channel": 2, "src": "/elsewhere"}
                   ).to_settings()

    message = str(excinfo.value)
    assert "cell_channel" in message and "src" in message
    assert "on the configuration object instead" in message


def test_extra_that_adds_a_new_setting_is_merged():
    """The taken side: that is what extra is for.

    A setting spaCR grew after this dataclass was written must still be
    reachable without editing the API.
    """
    from spacr.api import MaskConfig

    settings = MaskConfig(src="/data/plate1", cell_channel=0,
                          extra={"a_setting_not_yet_typed": 3}).to_settings()

    assert settings["a_setting_not_yet_typed"] == 3


def test_run_measure_accepts_a_config_and_a_mapping(monkeypatch):
    """Both branches of the second entry point.

    run_mask and run_measure are the same shape, and covering only one would
    leave the other free to drift -- which is how two entry points come to
    disagree about whether they copy the caller's mapping.
    """
    from spacr import api

    seen = []
    monkeypatch.setattr("spacr.measure.measure_crop",
                        lambda settings: seen.append(dict(settings)))

    api.run_measure(api.MeasureConfig(src="/data/plate1"))
    mine = {"src": "/data/plate2"}
    api.run_measure(mine)

    assert len(seen) == 2
    assert seen[0]["src"] == "/data/plate1"
    assert seen[1]["src"] == "/data/plate2"
