"""Re-inked icons persist between launches.

Re-inking 50 icons cold was 2.8 s of a 4.8 s startup: the source art is
large (logo_spacr.png is 3334x3334) and every launch paid full decode plus
LANCZOS downscale plus re-ink. The lru_cache covers repeats within one run;
this covers repeats across runs, which is the case a user actually meets.

The cold path is capped at the largest physical size the supported UI can
request. Keeping a 512 px working image for a 256 px maximum display path
made icon processing 37 percent of a measured real Home launch.
"""

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import iconset


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(tmp_path / "icons"))
    iconset._themed_array.cache_clear()
    iconset._source_digest.cache_clear()
    yield tmp_path / "icons"
    iconset._themed_array.cache_clear()
    iconset._source_digest.cache_clear()


@pytest.fixture
def icon(tmp_path):
    from PIL import Image
    path = tmp_path / "src.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (64, 64, 4), dtype=np.uint8),
                    "RGBA").save(path)
    return str(path)


class TestTheRoundTrip:

    def test_the_second_read_is_byte_identical(self, cache, icon):
        """Lossless, or the cache is a subtle rendering bug."""
        first = iconset.themed_array(icon, "dark")
        iconset._themed_array.cache_clear()      # forget in-process only
        second = iconset.themed_array(icon, "dark")
        assert np.array_equal(first, second)

    def test_it_writes_one_png_per_theme(self, cache, icon):
        iconset.themed_array(icon, "dark")
        iconset.themed_array(icon, "light")
        assert len(list(cache.glob("*.png"))) == 2

    def test_the_second_read_does_not_touch_the_source(self, cache, icon,
                                                       monkeypatch):
        """The point is skipping decode + LANCZOS + re-ink, so prove the
        expensive path is not entered rather than timing it."""
        iconset.themed_array(icon, "dark")
        iconset._themed_array.cache_clear()

        def _boom(_path):
            raise AssertionError("the source was re-decoded despite a cache hit")

        monkeypatch.setattr(iconset, "_load_rgba", _boom)
        assert iconset.themed_array(icon, "dark") is not None


class TestTheColdWorkCeiling:

    def test_it_covers_the_largest_supported_physical_tile(self):
        """The speedup may not make an accessibility/HiDPI path upscale."""
        from spacr.qt import preferences
        from spacr.qt.widgets.tile import Tile

        slot = inspect.signature(Tile.__init__).parameters["icon_size"].default
        physical = slot * preferences.FONT_SCALE_MAX * 2

        assert iconset.MAX_WORK_SIZE >= physical

    def test_a_large_master_is_reduced_to_that_ceiling(self, tmp_path):
        from PIL import Image

        source = tmp_path / "master.png"
        Image.fromarray(
            np.full((1024, 1024, 4), 255, dtype=np.uint8), "RGBA"
        ).save(source)

        rgba = iconset._load_rgba(str(source))

        assert rgba.shape == (
            iconset.MAX_WORK_SIZE, iconset.MAX_WORK_SIZE, 4)


class TestInvalidation:

    def test_an_edited_icon_gets_a_new_entry(self, cache, icon):
        """The stamp carries mtime and size, so a changed file simply keys
        somewhere else. No invalidation logic, and none to get wrong."""
        from PIL import Image
        iconset.themed_array(icon, "dark")
        before = {p.name for p in cache.glob("*.png")}
        Image.fromarray(np.full((64, 64, 4), 7, np.uint8), "RGBA").save(icon)
        iconset._themed_array.cache_clear()
        iconset.themed_array(icon, "dark")
        assert {p.name for p in cache.glob("*.png")} - before

    def test_the_version_is_part_of_the_key(self, cache, icon):
        """A cached icon from an older re-inking formula is WRONG, not
        stale, so the version has to be in the name."""
        path = iconset._cache_path(iconset._file_stamp(icon), "dark")
        assert f"v{iconset.ICON_CACHE_VERSION}" not in path.name  # hashed
        other = iconset._cache_path(iconset._file_stamp(icon), "light")
        assert path.name != other.name

    def test_an_unchanged_icon_survives_an_mtime_only_rewrite(
            self, cache, icon, monkeypatch):
        """Editable checkouts rewrite mtimes even when artwork is identical."""
        first = iconset.themed_array(icon, "dark")
        names_before = {path.name for path in cache.glob("*.png")}
        source = Path(icon)
        stamp = source.stat()
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
        iconset._themed_array.cache_clear()

        def _boom(_path):
            raise AssertionError("byte-identical artwork was decoded again")

        monkeypatch.setattr(iconset, "_load_rgba", _boom)
        second = iconset.themed_array(icon, "dark")

        assert np.array_equal(first, second)
        assert {path.name for path in cache.glob("*.png")} == names_before

    def test_same_size_but_changed_bytes_get_a_new_disk_key(
            self, cache, tmp_path):
        """Content, rather than size or mtime alone, decides correctness."""
        source = tmp_path / "raw-icon"
        source.write_bytes(b"first payload")
        before_stamp = iconset._file_stamp(str(source))
        before = iconset._cache_path(before_stamp, "dark")

        source.write_bytes(b"other payload")
        assert source.stat().st_size == before_stamp[2]
        os.utime(source, ns=(source.stat().st_atime_ns,
                             before_stamp[1] + 1_000_000))
        after = iconset._cache_path(iconset._file_stamp(str(source)), "dark")

        assert after != before


class TestItNeverBreaksIcons:
    """A cache that can break icon loading is a liability (INVARIANTS 10)."""

    def test_an_unwritable_cache_dir_still_returns_an_icon(self, icon,
                                                            monkeypatch,
                                                            tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("this is a file, not a directory")
        monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(blocker / "icons"))
        iconset._themed_array.cache_clear()
        assert iconset.themed_array(icon, "dark") is not None

    def test_a_corrupt_cache_entry_is_re_rendered(self, cache, icon):
        iconset.themed_array(icon, "dark")
        for png in cache.glob("*.png"):
            png.write_bytes(b"not a png at all")
        iconset._themed_array.cache_clear()
        assert iconset.themed_array(icon, "dark") is not None

    def test_a_missing_source_is_still_none(self, cache, tmp_path):
        assert iconset.themed_array(str(tmp_path / "nope.png"), "dark") is None

    def test_no_part_files_are_left_behind(self, cache, icon):
        """Write-then-rename, so a crash cannot leave a half-written PNG
        that every later launch reads as corrupt."""
        iconset.themed_array(icon, "dark")
        assert list(cache.glob("*.part")) == []
