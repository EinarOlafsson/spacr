from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacr.flowview import thumbs
from spacr.flowview.thumbs import ThumbnailCache, make_thumbnail, thumbnail_png


def gradient(shape=(200, 100)):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def test_thumbnail_downsamples_stretches_nonfinite_values_and_is_deterministic():
    source = gradient()
    source[0, 0] = np.nan
    source[-1, -1] = np.inf

    first = make_thumbnail(source, max_size=128)
    second = make_thumbnail(source, max_size=128)

    assert first.size == (64, 128)
    assert first.mode == "L"
    assert np.asarray(first).min() == 0
    assert np.asarray(first).max() == 255
    assert first.tobytes() == second.tobytes()
    assert thumbnail_png(source) == thumbs.encode_thumbnail(source)


def test_constant_empty_finite_and_alpha_channels_are_normalised():
    assert np.asarray(make_thumbnail(np.full((3, 3), np.nan))).sum() == 0
    assert np.asarray(make_thumbnail(np.full((3, 3), 0.5))).tolist() == [
        [128, 128, 128],
        [128, 128, 128],
        [128, 128, 128],
    ]
    assert np.asarray(make_thumbnail(np.full((2, 2), 300))).min() == 255

    one_channel = make_thumbnail(np.ones((2, 2, 1), dtype=np.float32))
    assert one_channel.mode == "L"
    la = make_thumbnail(
        np.dstack((gradient((3, 3)), np.full((3, 3), 0.5))),
    )
    assert la.mode == "LA" and np.asarray(la)[0, 0, 1] == 128
    rgba = make_thumbnail(
        np.dstack(
            (
                gradient((3, 3)),
                gradient((3, 3))[::-1],
                np.ones((3, 3)),
                np.full((3, 3), np.nan),
            )
        )
    )
    assert rgba.mode == "RGBA"
    assert np.asarray(rgba)[..., 3].sum() == 0
    rgb = make_thumbnail(np.dstack((gradient((3, 3)),) * 3))
    assert rgb.mode == "RGB"


def test_paths_pillow_palette_images_and_one_pixel_outlines(tmp_path):
    palette = Image.new("P", (5, 5))
    palette.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    palette.putpixel((2, 2), 1)
    path = tmp_path / "palette.png"
    palette.save(path)

    from_path = make_thumbnail(path)
    from_pillow = make_thumbnail(palette)
    assert from_path.mode == from_pillow.mode == "RGBA"

    normal = Image.fromarray(np.arange(25, dtype=np.uint8).reshape(5, 5))
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)
    assert make_thumbnail(normal_path).mode == "L"
    assert make_thumbnail(normal).mode == "L"

    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    outlined = np.asarray(make_thumbnail(np.zeros((5, 5)), outline_mask=mask))
    assert outlined[1, 1] == 255
    assert outlined[2, 2] == 0

    colour = np.zeros((5, 5, 4), dtype=np.uint8)
    colour[..., 3] = 7
    colour_outline = np.asarray(make_thumbnail(colour, outline_mask=mask))
    assert colour_outline[1, 1].tolist() == [255, 255, 255, 255]
    assert colour_outline[2, 2].tolist() == [0, 0, 0, 7]
    rgb_outline = np.asarray(
        make_thumbnail(np.zeros((5, 5, 3), dtype=np.uint8), outline_mask=mask)
    )
    assert rgb_outline[1, 1].tolist() == [255, 255, 255]


@pytest.mark.parametrize("max_size", [0, 129])
def test_thumbnail_size_is_strictly_bounded(max_size):
    with pytest.raises(ValueError, match="between 1 and 128"):
        make_thumbnail(np.ones((2, 2)), max_size=max_size)


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2, 3]),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.zeros((2, 2, 5)),
    ],
)
def test_invalid_image_shapes_are_refused(array):
    with pytest.raises(ValueError, match="thumbnail images"):
        make_thumbnail(array)


def test_invalid_outline_shapes_are_refused():
    image = np.zeros((3, 4))
    with pytest.raises(ValueError, match="two-dimensional"):
        make_thumbnail(image, outline_mask=np.zeros((3, 4, 1)))
    with pytest.raises(ValueError, match="match the image"):
        make_thumbnail(image, outline_mask=np.zeros((3, 3)))


def test_cache_uses_safe_names_and_evicts_oldest_first(tmp_path):
    payload = thumbnail_png(gradient((24, 24)))
    cache = ThumbnailCache(
        tmp_path / "nested" / "run",
        max_bytes=len(payload) * 2,
        max_size=64,
    )
    assert cache.get("missing") is None
    first = cache.put("../../first", gradient((24, 24)))
    second = cache.store("second", gradient((24, 24)))
    assert first.parent == cache.directory
    assert first.name == cache.path_for("../../first").name
    assert ".." not in first.name
    assert cache.get("second") == second
    assert cache.total_bytes == len(payload) * 2

    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    third = cache.store("third", gradient((24, 24)))
    assert not first.exists()
    assert second.exists() and third.exists()
    assert cache.total_bytes <= cache.max_bytes

    assert cache.clear() == 2
    assert cache.clear() == 0
    assert cache.discard() == 0
    assert not cache.directory.exists()


def test_cache_rejects_impossible_limits_and_preserves_foreign_files(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        ThumbnailCache(tmp_path / "zero", max_bytes=0)
    with pytest.raises(ValueError, match="between 1 and 128"):
        ThumbnailCache(tmp_path / "large", max_size=200)

    payload = thumbnail_png(np.zeros((4, 4)))
    tiny = ThumbnailCache(tmp_path / "tiny", max_bytes=len(payload) - 1)
    with pytest.raises(ValueError, match="exceeds"):
        tiny.store("too-large", np.zeros((4, 4)))

    kept = ThumbnailCache(tmp_path / "kept")
    sentinel = kept.directory / "record.json"
    sentinel.write_text("keep", encoding="utf-8")
    kept.store("one", np.zeros((4, 4)))
    assert kept.discard() == 1
    assert kept.directory.exists() and sentinel.exists()


def test_foreign_png_is_not_owned_counted_evicted_or_cleared(tmp_path):
    payload = thumbnail_png(np.zeros((4, 4), dtype=np.uint8))
    cache = ThumbnailCache(tmp_path / "shared", max_bytes=len(payload))
    foreign = cache.directory / "figure.png"
    foreign.write_bytes(b"user-owned image")
    os.utime(foreign, ns=(1, 1))

    assert cache.total_bytes == 0
    generated = cache.store("generated", np.zeros((4, 4), dtype=np.uint8))
    assert foreign.read_bytes() == b"user-owned image"
    assert cache.total_bytes == generated.stat().st_size == len(payload)

    assert cache.clear() == 1
    assert not generated.exists()
    assert foreign.read_bytes() == b"user-owned image"
    assert cache.discard() == 0
    assert cache.directory.exists()


def test_store_publishes_complete_pngs_atomically_and_coordinates_get(
    tmp_path,
    monkeypatch,
):
    directory = tmp_path / "shared"
    writer = ThumbnailCache(directory)
    reader = ThumbnailCache(directory)
    key = "same-key"
    path = writer.store(key, np.zeros((4, 4), dtype=np.uint8))

    replacement_reached = threading.Event()
    allow_replacement = threading.Event()
    real_replace = os.replace

    def pause_before_publication(source, destination):
        if Path(destination) == path:
            replacement_reached.set()
            assert allow_replacement.wait(5), "test did not release atomic publication"
        real_replace(source, destination)

    monkeypatch.setattr(thumbs.os, "replace", pause_before_publication)
    with ThreadPoolExecutor(max_workers=2) as pool:
        store_future = pool.submit(
            writer.store,
            key,
            np.full((4, 4), 255, dtype=np.uint8),
        )
        get_future = None
        try:
            assert replacement_reached.wait(2), "store did not reach atomic publication"

            old_path = reader.get(key)
            assert old_path == path
            with Image.open(old_path) as old_image:
                old_image.load()
                assert np.asarray(old_image).tolist() == [[0] * 4 for _ in range(4)]

            get_future = pool.submit(writer.get, key)
            assert not get_future.done(), "same-cache get bypassed the publication lock"
        finally:
            allow_replacement.set()

        assert store_future.result(timeout=5) == path
        assert get_future is not None and get_future.result(timeout=5) == path

    with Image.open(path) as new_image:
        new_image.load()
        assert np.asarray(new_image).tolist() == [[255] * 4 for _ in range(4)]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_atomic_publication_preserves_the_previous_png_and_error(
    tmp_path,
    monkeypatch,
    cleanup_fails,
):
    cache = ThumbnailCache(tmp_path / "cache")
    key = "same-key"
    path = cache.store(key, np.zeros((4, 4), dtype=np.uint8))
    previous = path.read_bytes()

    def reject_publication(source, destination):
        raise OSError("publication failed")

    monkeypatch.setattr(thumbs.os, "replace", reject_publication)
    real_unlink = Path.unlink
    if cleanup_fails:

        def reject_cleanup(temporary, *args, **kwargs):
            if temporary.suffix == ".tmp":
                raise OSError("cleanup failed")
            return real_unlink(temporary, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", reject_cleanup)

    with pytest.raises(OSError, match="publication failed"):
        cache.store(key, np.full((4, 4), 255, dtype=np.uint8))

    assert path.read_bytes() == previous
    temporary_paths = list(cache.directory.glob(".*.tmp"))
    assert bool(temporary_paths) is cleanup_fails
    if cleanup_fails:
        monkeypatch.setattr(Path, "unlink", real_unlink)
        for temporary in temporary_paths:
            temporary.unlink()
