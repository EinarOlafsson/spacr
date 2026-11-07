"""Where tutorial media is written and read, now that it is kept out of git.

Posters, narration, example downloads and the release-candidate snapshot's
media are not committed to spaCR. ``docs/media_revision.json`` pins them in
the Hugging Face dataset ``einarolafsson/spacr-tutorials`` (groups
``tutorials`` and ``tutorial_archive``), and :mod:`fetch_docs_media` puts
them back into the tree for a docs build.

The tutorial tools write new media to the staging folder instead of the
repository. It mirrors the dataset: a file whose place in spaCR is
``docs/source/_extra/tutorials/examples/x.zip`` is written to
``<staging>/hf/tree/docs/source/_extra/tutorials/examples/x.zip``, and
``python tools/fetch_docs_media.py publish --group tutorials --source
<staging>/hf/tree`` uploads it and moves the pin.

The staging folder is ``$SPACR_TUTORIALS_STAGING`` when set, otherwise
``~/spacr-tutorials-staging``.

Reading goes the other way: :func:`media_source` returns the file from the
tree when it has been fetched, from staging when a tool has just written it,
and otherwise from the pinned dataset revision.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional, Union

REPO = Path(__file__).resolve().parents[2]

#: Environment variable naming the staging folder.
STAGING_ENV = "SPACR_TUTORIALS_STAGING"

_FETCH = None


def _fetch_module():
    """``tools/fetch_docs_media.py``, loaded once."""
    global _FETCH
    if _FETCH is None:
        spec = importlib.util.spec_from_file_location(
            "fetch_docs_media", REPO / "tools" / "fetch_docs_media.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FETCH = module
    return _FETCH


def staging_root() -> Path:
    """The staging folder the tutorial tools write media to.

    :returns: ``$SPACR_TUTORIALS_STAGING``, or ``~/spacr-tutorials-staging``.
    """
    configured = os.environ.get(STAGING_ENV, "").strip()
    return Path(configured).expanduser() if configured else Path.home() / "spacr-tutorials-staging"


def staging_tree() -> Path:
    """The staging mirror of the repository tree, ``<staging>/hf/tree``."""
    return staging_root() / "hf" / _fetch_module().TUTORIAL_PREFIX.rstrip("/")


def _relative(path: Union[str, Path]) -> Optional[str]:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO).as_posix()
        except ValueError:
            return None
    return candidate.as_posix()


def is_media(path: Union[str, Path]) -> bool:
    """Whether a path in the repository is tutorial media kept out of git.

    :param path: absolute, or relative to the repository root.
    """
    relative = _relative(path)
    return relative is not None and _fetch_module().is_tutorial_media(relative)


def media_target(path: Union[str, Path]) -> Path:
    """Where a tool writes a file: staging for media, the repository otherwise.

    :param path: the file's place in the repository, absolute or relative.
    :returns: the staging copy's path for tutorial media, with its folder
        created; the repository path for everything else.
    """
    relative = _relative(path)
    if relative is None or not _fetch_module().is_tutorial_media(relative):
        return Path(path) if Path(path).is_absolute() else REPO / path
    target = staging_tree() / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def media_source(path: Union[str, Path]) -> Path:
    """Where a tool reads a file from.

    :param path: the file's place in the repository, absolute or relative.
    :returns: the repository file when it exists; for tutorial media, the
        staging copy, else the verified copy of the pinned revision.
    :raises FileNotFoundError: when tutorial media is in none of them.
    """
    relative = _relative(path)
    local = Path(path) if Path(path).is_absolute() else REPO / path
    if local.exists() or relative is None or not _fetch_module().is_tutorial_media(relative):
        return local
    staged = staging_tree() / relative
    if staged.exists():
        return staged
    fetch = _fetch_module()
    for group in ("tutorials", "tutorial_archive"):
        found = fetch.available(relative, group, REPO)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"{relative} is tutorial media that is not downloaded: run "
        "'python tools/fetch_docs_media.py fetch --group tutorials' "
        "(or --group tutorial_archive), or write it with the tutorial tools")
