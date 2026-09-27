"""SVG artwork that cannot be drawn gives no icon, and one bad icon does not
stop the others being warmed.

An SVG Qt cannot parse decodes to nothing rather than to a garbage
picture, so the caller falls back to its default glyph. The background
warm-up counts the icons it re-inked and skips one whose decode raises
instead of abandoning the rest.
"""
import numpy as np
import pytest

from spacr.qt import iconset


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(tmp_path / "icons"))
    themed = iconset._themed_array
    themed.cache_clear()
    iconset._source_digest.cache_clear()
    yield tmp_path / "icons"
    themed.cache_clear()
    iconset._source_digest.cache_clear()


def test_an_svg_qt_cannot_parse_decodes_to_nothing(tmp_path):
    path = tmp_path / "broken.svg"
    path.write_text("this is not <svg at all")
    assert iconset._load_rgba(str(path)) is None


def test_an_svg_with_no_size_is_drawn_at_the_working_size(tmp_path):
    """Qt scales a zero default size up to the whole working square, so a
    sizeless SVG is rendered there rather than refused."""
    path = tmp_path / "empty.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" '
                    'width="0.1" height="0.1"></svg>')
    rgba = iconset._load_rgba(str(path))
    assert rgba is not None
    assert rgba.shape[:2] == (iconset.MAX_WORK_SIZE, iconset.MAX_WORK_SIZE)


def test_a_drawable_svg_decodes_to_rgba(tmp_path):
    path = tmp_path / "dot.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16" '
                    'height="16"><circle cx="8" cy="8" r="6" fill="black"/>'
                    '</svg>')
    rgba = iconset._load_rgba(str(path))
    assert rgba is not None and rgba.shape[2] == 4
    assert rgba[..., 3].max() > 0


def test_a_themed_icon_is_written_to_the_cache(cache, tmp_path):
    from PIL import Image
    path = tmp_path / "src.png"
    Image.fromarray(np.full((8, 8, 4), 200, dtype=np.uint8), "RGBA").save(path)

    inked = iconset.themed_array(str(path), "dark")

    assert inked is not None
    assert list(cache.glob("src-*.png"))


def test_the_warm_up_skips_an_icon_whose_decode_raises(cache, tmp_path,
                                                        monkeypatch):
    from PIL import Image
    good = tmp_path / "good.png"
    Image.fromarray(np.full((8, 8, 4), 200, dtype=np.uint8), "RGBA").save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"")
    monkeypatch.setattr(iconset, "bundled_icon_paths",
                        lambda: (str(bad), str(good)))
    real = iconset._themed_array

    def refusing(stamp, theme):
        if stamp[0] == str(bad):
            raise OSError("unreadable")
        return real(stamp, theme)

    monkeypatch.setattr(iconset, "_themed_array", refusing)

    assert iconset._warm_the_bundled_icons("dark") == 1
