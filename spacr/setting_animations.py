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
    "setting_animations",
    "validate_setting_animation_assets",
]
