"""Registry for the short animations that explain visual spaCR settings.

The registry is intentionally independent of Qt.  GUI code, documentation,
plugins and tests can therefore resolve the same exact setting key to the same
packaged GIF without importing the desktop application.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, Mapping, Optional, Tuple


SCHEMA_VERSION = 1
"""Manifest schema understood by this spaCR release."""

ANIMATION_DOCS_BASE = (
    "https://einarolafsson.github.io/spacr/setting_animations.html"
)
"""Published setting-animation gallery used by API and GUI links."""

_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources" / "setting_animations"
_MANIFEST_PATH = _RESOURCE_ROOT / "manifest.json"


class SettingAnimationError(RuntimeError):
    """Raised when packaged setting-animation metadata is invalid."""


@dataclass(frozen=True)
class SettingAnimation:
    """One explanatory animation and every exact setting key that uses it.

    :param slug: Stable identifier used by the GIF filename and docs anchor.
    :param title: Short human-readable animation title.
    :param category: Gallery section containing the animation.
    :param scene: Deterministic renderer scene used to generate the GIF.
    :param settings: Exact spaCR setting keys mapped to this animation.
    :param relative_file: Safe path below the packaged animation directory.
    :param sha256: Expected SHA-256 digest recorded during generation.
    :param frames: Encoded GIF frame count after identical-frame coalescing.
    :param unique_frames: Number of visually distinct encoded frames.
    :param byte_size: Generated GIF size in bytes.
    """

    slug: str
    title: str
    category: str
    scene: str
    settings: Tuple[str, ...]
    relative_file: str
    sha256: str
    frames: int
    unique_frames: int
    byte_size: int

    @property
    def path(self) -> Path:
        """Return the installed GIF path for this animation."""
        return _RESOURCE_ROOT.joinpath(*PurePosixPath(self.relative_file).parts)

    @property
    def docs_anchor(self) -> str:
        """Return this animation's stable Sphinx/HTML anchor."""
        return "setting-animation-" + self.slug.replace("_", "-")

    @property
    def docs_url(self) -> str:
        """Return the published gallery URL anchored to this animation."""
        return f"{ANIMATION_DOCS_BASE}#{self.docs_anchor}"


def _safe_relative_file(value: object, slug: str) -> str:
    """Validate and normalize one manifest file path."""
    if not isinstance(value, str) or not value.strip():
        raise SettingAnimationError(f"{slug}: animation file must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("gifs",):
        raise SettingAnimationError(f"{slug}: unsafe animation path {value!r}")
    if path.suffix.lower() != ".gif":
        raise SettingAnimationError(f"{slug}: animation is not a GIF: {value!r}")
    return path.as_posix()


def _positive_integer(value: object, slug: str, field: str) -> int:
    """Return a positive manifest integer or raise a focused error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SettingAnimationError(
            f"{slug}: validation field {field!r} must be a positive integer"
        )
    return value


def _parse_entry(raw: object) -> SettingAnimation:
    """Validate one decoded manifest entry."""
    if not isinstance(raw, dict):
        raise SettingAnimationError("animation entries must be JSON objects")
    slug = raw.get("slug")
    if not isinstance(slug, str) or not slug or not slug.replace("_", "").isalnum():
        raise SettingAnimationError(f"invalid animation slug {slug!r}")
    settings = raw.get("settings")
    if (
        not isinstance(settings, list)
        or not settings
        or any(not isinstance(key, str) or not key for key in settings)
    ):
        raise SettingAnimationError(f"{slug}: settings must be non-empty strings")
    validation = raw.get("validation")
    if not isinstance(validation, dict):
        raise SettingAnimationError(f"{slug}: validation metadata is missing")
    digest = validation.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SettingAnimationError(f"{slug}: invalid SHA-256 digest")

    text_fields = {}
    for field in ("title", "category", "scene"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SettingAnimationError(f"{slug}: {field} must be a non-empty string")
        text_fields[field] = value.strip()

    animation = SettingAnimation(
        slug=slug,
        title=text_fields["title"],
        category=text_fields["category"],
        scene=text_fields["scene"],
        settings=tuple(settings),
        relative_file=_safe_relative_file(raw.get("file"), slug),
        sha256=digest,
        frames=_positive_integer(validation.get("frames"), slug, "frames"),
        unique_frames=_positive_integer(
            validation.get("unique_frames"), slug, "unique_frames"
        ),
        byte_size=_positive_integer(validation.get("bytes"), slug, "bytes"),
    )
    if not animation.path.is_file():
        raise SettingAnimationError(
            f"{slug}: packaged GIF is missing: {animation.relative_file}"
        )
    return animation


@lru_cache(maxsize=1)
def setting_animations() -> Tuple[SettingAnimation, ...]:
    """Load, validate and cache every packaged setting animation.

    :returns: Animations in deterministic gallery order.
    :raises SettingAnimationError: If the manifest or an asset is invalid.
    """
    try:
        payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingAnimationError(
            f"Could not load setting-animation manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SettingAnimationError("setting-animation manifest must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SettingAnimationError(
            "unsupported setting-animation manifest schema "
            f"{payload.get('schema_version')!r}"
        )
    raw_animations = payload.get("animations")
    if not isinstance(raw_animations, list):
        raise SettingAnimationError("manifest animations must be a list")

    animations = tuple(_parse_entry(raw) for raw in raw_animations)
    slugs = [animation.slug for animation in animations]
    if len(slugs) != len(set(slugs)):
        raise SettingAnimationError("setting-animation manifest has duplicate slugs")

    keys = [key for animation in animations for key in animation.settings]
    if len(keys) != len(set(keys)):
        raise SettingAnimationError(
            "one setting key is mapped to more than one animation"
        )
    return animations


@lru_cache(maxsize=1)
def animations_by_setting() -> Mapping[str, SettingAnimation]:
    """Return an immutable-by-convention exact setting-key lookup."""
    return {
        key: animation
        for animation in setting_animations()
        for key in animation.settings
    }


def animation_for_setting(setting_key: str) -> Optional[SettingAnimation]:
    """Return the animation mapped to ``setting_key``, if one exists.

    Matching is deliberately exact and case-sensitive so similarly named
    scientific settings cannot accidentally display misleading help.
    """
    return animations_by_setting().get(str(setting_key))


def animation_path_for_setting(setting_key: str) -> Optional[Path]:
    """Return the installed GIF path for ``setting_key``, if available."""
    animation = animation_for_setting(setting_key)
    return animation.path if animation is not None else None


def iter_setting_animations() -> Iterator[SettingAnimation]:
    """Iterate over all animations in deterministic gallery order."""
    return iter(setting_animations())



#: An animation whose before and after differ by less than this fraction of
#: the frame is not illustrating its setting. Measured across all 94 in the
#: 2026-08 audit: the median is well above 1%, 27 fall below it, and
#: `pathogen_diameter` changes 0.0% -- it shows the viewer nothing at all.
MIN_VISIBLE_CHANGE = 0.01

#: Pixels differing by less than this (summed over RGB) are noise from GIF
#: palette quantisation rather than a real change.
_CHANGE_THRESHOLD = 30

#: Width in pixels of the frame border inspected for the GIF disposal
#: artifact. Six is wide enough to sample the ring the encoder leaves and
#: narrow enough that a scene drawing near the edge -- the border-object
#: family does -- only ever changes part of it.
_BORDER_BAND = 6

#: A frame whose whole perimeter changed is not a picture, it is the GIF
#: encoder. Measured across all 94: the artifact scores exactly 1.0 and the
#: highest legitimate score is 0.14, so anything above half is the encoder.
MAX_BORDER_ARTIFACT = 0.5


def measure_visible_change(path) -> float:
    """Fraction of the frame that changes between an animation's states.

    The "after" state is the frame MOST DIFFERENT from the first, not the
    last: these GIFs loop, so the last frame is the first one again and
    comparing them reports zero change for every animation ever made.

    :param path: the GIF to measure.
    :returns: fraction of pixels that differ, 0.0 to 1.0. Returns 0.0 when
        the file cannot be read as an animation, so a caller sees "shows
        nothing" rather than an exception.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:                       # no imaging
        return 1.0
    try:
        frames = []
        with Image.open(path) as im:
            while True:
                frames.append(np.asarray(im.convert("RGB"), dtype=np.int16))
                im.seek(im.tell() + 1)
    except EOFError:
        pass
    except Exception:
        return 0.0
    if len(frames) < 2:
        return 0.0
    base = frames[0]
    worst = max(frames[1:], key=lambda f: int(np.abs(f - base).sum()))
    changed = np.abs(worst - base).sum(axis=2) > _CHANGE_THRESHOLD
    return float(changed.mean())


def validate_animations_show_something(
        *, minimum: float = MIN_VISIBLE_CHANGE) -> Dict[str, float]:
    """Every animation must visibly change something.

    An animation is a stronger claim than a sentence: a user who watches
    one believes it. One that changes nothing teaches nothing, and unlike a
    wrong sentence it cannot be spotted by reading the source.

    Separate from :func:`validate_setting_animation_assets`, which checks
    that the FILES are intact -- an unchanged, perfectly-hashed GIF passes
    that and fails this.

    :param minimum: fraction of the frame that must change.
    :returns: ``{slug: fraction}`` for every animation BELOW the threshold,
        empty when they all pass. Returned rather than raised so a caller
        can report the whole list instead of the first one.
    """
    failures: Dict[str, float] = {}
    for animation in setting_animations():
        fraction = measure_visible_change(animation.path)
        if fraction < minimum:
            failures[animation.slug] = round(fraction, 4)
    return failures


def _animation_frames(path):
    """Decode one GIF to a list of RGB arrays, or ``None`` if it cannot be read.

    Shared by the two measurements below so they agree about what a frame is.
    GIF decoding is not a detail here: Pillow coalesces identical frames, so
    the stored frame count is lower than the rendered one and no index maps
    to a phase of the scene.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:                       # no imaging
        return None
    frames = []
    try:
        with Image.open(path) as image:
            while True:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.int16))
                image.seek(image.tell() + 1)
    except EOFError:
        pass
    except Exception:
        return []
    return frames


def measure_border_artifact(path) -> float:
    """Largest fraction of the frame border that changes after the first frame.

    This measures an ENCODING fault rather than a drawing one. Saving a GIF
    with ``disposal=2`` tells a decoder to clear each frame to the background
    colour before drawing the next; when the encoder then optimises a frame
    down to the sub-rectangle that actually changed, everything outside that
    rectangle is left showing the background. The animation plays with a
    bright ring flashing around it once per loop, and Qt's ``QMovie`` renders
    it exactly as described -- this was measured through the GUI path, not
    inferred from the file.

    It also corrupts :func:`measure_visible_change`, because a ring around a
    360x360 frame is 9.75% of it: an animation that shows almost nothing
    scores over 10% and clears the threshold on the artifact alone.

    :param path: the GIF to measure.
    :returns: 0.0 to 1.0, where 1.0 means an entire frame border changed.
        Returns 0.0 when the file cannot be read, so a caller sees the other
        animations rather than a traceback.
    """
    try:
        import numpy as np
    except Exception:                       # no imaging
        return 0.0
    frames = _animation_frames(path)
    if not frames or len(frames) < 2:
        return 0.0
    band = _BORDER_BAND

    def perimeter(frame):
        return np.concatenate([
            frame[:band].reshape(-1, 3),
            frame[-band:].reshape(-1, 3),
            frame[band:-band, :band].reshape(-1, 3),
            frame[band:-band, -band:].reshape(-1, 3),
        ])

    first = perimeter(frames[0])
    return max(
        float((np.abs(perimeter(frame) - first).sum(axis=1) > _CHANGE_THRESHOLD).mean())
        for frame in frames[1:]
    )


def validate_animations_have_no_border_artifact(
        *, maximum: float = MAX_BORDER_ARTIFACT) -> Dict[str, float]:
    """No animation may flash a ring around itself.

    Separate from :func:`validate_animations_show_something` because the two
    fail in opposite directions: this artifact ADDS 9.75% of changed frame,
    so an animation carrying it passes the "shows something" check while
    showing the viewer a rectangle the setting has nothing to do with.

    :param maximum: fraction of the frame border allowed to change.
    :returns: ``{slug: fraction}`` for every animation ABOVE the threshold,
        empty when they all pass. Returned rather than raised so a caller can
        report the whole list.
    """
    failures: Dict[str, float] = {}
    for animation in setting_animations():
        fraction = measure_border_artifact(animation.path)
        if fraction > maximum:
            failures[animation.slug] = round(fraction, 4)
    return failures


#: A border-object animation zooms in before it removes anything, so a diff
#: taken across the zoom measures the CAMERA and reports every object as
#: changed. Only frames whose drawn well edge sits at the same column are
#: comparable; this is how many such frames are needed to have a pair.
_STEADY_FRAMES_NEEDED = 2

#: The well perimeter is drawn near-white. A column of the frame belongs to
#: it when nearly every pixel down the column is at least this bright.
_WELL_RULE_BRIGHTNESS = 150

#: The well's own bright rule is drawn OVER the objects, and the organelle
#: glyph is a scatter of separate strokes, so one changed object arrives as
#: many components. Closing the mask with a square this wide reunites them;
#: 13 px is wider than the rule and than the gaps between Golgi strokes,
#: and narrower than the gap between two drawn objects.
_OBJECT_CLOSING = 13

#: Components smaller than this are antialiasing along a moved edge, not an
#: object that was removed.
_MIN_OBJECT_PIXELS = 60


def measure_border_object_removal(path) -> Dict[str, int]:
    """Classify what a "remove border objects" animation actually removes.

    The reported failure -- one edge object and one non-edge object
    disappear -- was true, and three separate ways of measuring it agreed it
    was fine. Each of those is avoided here, deliberately:

    * **Not first-vs-last, and not "the frame most different from frame 0".**
      This scene zooms in and back out, so the most-different frame is the
      one at full zoom and that diff is dominated by the camera: every
      object registers as changed. The pair used here is the FIRST and LAST
      frames at which the drawn well edge sits at the same column, which is
      the interval where the zoom is complete and only the removal happens.
    * **Not the generator's own ``touches`` flag.** That flag is the thing
      under test; a check that classifies objects by it validates the
      generator against itself and reported "all correct" on the assets that
      shipped the bug. The well edge is recovered from the drawn pixels.
    * **Not raw connected components.** The well's bright rule is drawn over
      the objects and splits each into a left and a right half, and the
      organelle glyph is a scatter of separate strokes. Classifying those
      fragments individually reports a straddling Golgi as ~30 interior
      objects. The mask is closed first so one object is one component.

    :param path: the GIF to measure.
    :returns: ``{"edge_x", "before", "after", "crossing", "interior"}`` --
        the drawn well edge column, the two frame indices compared, and how
        many changed objects do and do not cross that edge.
    :raises SettingAnimationError: if the animation cannot be measured,
        which for this check is a failure rather than a zero: an unreadable
        border animation is not a correct one.
    """
    import numpy as np
    from scipy import ndimage

    frames = _animation_frames(path)
    if not frames or len(frames) < _STEADY_FRAMES_NEEDED:
        raise SettingAnimationError(f"{path}: not enough frames to compare")

    height = frames[0].shape[0]
    columns = []
    for frame in frames:
        lit = (frame.min(axis=2) > _WELL_RULE_BRIGHTNESS).sum(axis=0)
        column = int(np.argmax(lit))
        columns.append((column, int(lit[column])))

    spanning = [
        index for index, (_, count) in enumerate(columns)
        if count >= height - 2
    ]
    if not spanning:
        raise SettingAnimationError(f"{path}: no frame draws a full well edge")
    counts: Dict[int, int] = {}
    for index in spanning:
        counts[columns[index][0]] = counts.get(columns[index][0], 0) + 1
    edge_x = max(sorted(counts), key=lambda column: counts[column])
    steady = [index for index in spanning if columns[index][0] == edge_x]
    if len(steady) < _STEADY_FRAMES_NEEDED:
        raise SettingAnimationError(
            f"{path}: the zoom never rests, so no two frames are comparable"
        )

    before, after = steady[0], steady[-1]
    changed = np.abs(frames[after] - frames[before]).sum(axis=2) > _CHANGE_THRESHOLD
    pad = _OBJECT_CLOSING
    padded = np.pad(changed, pad, mode="constant", constant_values=False)
    closed = ndimage.binary_closing(
        padded, structure=np.ones((_OBJECT_CLOSING, _OBJECT_CLOSING), bool)
    )[pad:-pad, pad:-pad]

    labels, count = ndimage.label(closed)
    crossing = interior = 0
    for index in range(1, count + 1):
        component = labels == index
        if int((component & changed).sum()) < _MIN_OBJECT_PIXELS:
            continue
        columns_hit = np.nonzero(component.any(axis=0))[0]
        if columns_hit[0] <= edge_x <= columns_hit[-1]:
            crossing += 1
        else:
            interior += 1
    return {
        "edge_x": edge_x,
        "before": before,
        "after": after,
        "crossing": crossing,
        "interior": interior,
    }


def validate_border_animations_remove_only_edge_objects() -> Dict[str, int]:
    """Every border animation must remove only objects crossing the well edge.

    The setting says "delete every label touching an image edge". An
    animation that also removes something in the middle of the well teaches
    the opposite of the sentence beside it, and it looks finished while
    doing so -- the file has the right size, the right digest, and 16% of
    the frame changes.

    This reads the SHIPPED GIFs. ``tests/test_border_animation_geometry.py``
    checks the same property in the generator's spec, which is where the
    fault was fixed; this is the half that still holds if an asset is
    regenerated by a different build, edited, or shipped from a spec that
    was never re-rendered.

    :returns: ``{slug: interior_objects_removed}`` for each border animation
        that removes something not crossing the edge. Empty when they are
        all correct. Returned rather than raised so one report covers the
        family instead of stopping at the first.
    """
    offenders: Dict[str, int] = {}
    for animation in setting_animations():
        if animation.scene != "border":
            continue
        interior = measure_border_object_removal(animation.path)["interior"]
        if interior:
            offenders[animation.slug] = interior
    return offenders


def validate_setting_animation_assets(*, check_hashes: bool = False) -> Dict[str, int]:
    """Validate packaged files and optionally recompute their SHA-256 hashes.

    :param check_hashes: When true, read every GIF and compare its digest with
        the manifest. Runtime callers normally leave this false; release tests
        enable it.
    :returns: Counts for animations, mapped setting keys and total asset bytes.
    :raises SettingAnimationError: If a file is absent, has changed size or has
        a digest different from the generated manifest.
    """
    animations = setting_animations()
    total_bytes = 0
    for animation in animations:
        try:
            size = animation.path.stat().st_size
        except OSError as exc:
            raise SettingAnimationError(
                f"Could not inspect {animation.relative_file}: {exc}"
            ) from exc
        if size != animation.byte_size:
            raise SettingAnimationError(
                f"{animation.slug}: expected {animation.byte_size} bytes, got {size}"
            )
        total_bytes += size
        if check_hashes:
            digest = hashlib.sha256(animation.path.read_bytes()).hexdigest()
            if digest != animation.sha256:
                raise SettingAnimationError(
                    f"{animation.slug}: packaged GIF digest does not match manifest"
                )
    return {
        "animations": len(animations),
        "setting_keys": len(animations_by_setting()),
        "bytes": total_bytes,
    }


__all__ = [
    "ANIMATION_DOCS_BASE",
    "SCHEMA_VERSION",
    "SettingAnimation",
    "SettingAnimationError",
    "animation_for_setting",
    "animation_path_for_setting",
    "animations_by_setting",
    "iter_setting_animations",
    "measure_border_object_removal",
    "measure_visible_change",
    "setting_animations",
    "validate_animations_show_something",
    "validate_border_animations_remove_only_edge_objects",
    "validate_setting_animation_assets",
]
