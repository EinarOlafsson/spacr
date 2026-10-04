"""What was dropped on Make Masks, decided before anything is opened.

Make Masks takes a drop of files, folders or both, and what the
user means depends on what they dropped. :func:`classify_drop` reads the
paths -- names, folder layout and, for a few TIFFs, their pixels -- and says
which of these it is, without a single question or widget:

``images``
    Image files (and possibly folders) to open as one queue, in drop order.
``folder``
    One folder of images, opened as it is (masks in its ``masks/``).
``nested``
    One folder whose images sit in subfolders: when the subfolders look like
    channels (:attr:`channel_like`), "Organize for Measure" with one channel
    column per subfolder; otherwise the consolidation question.
``folders``
    Several folders, each holding images: Make Masks opens "Organize for
    Measure" with one channel column per folder.
``images_with_masks``
    Images dropped together with their masks (a masks folder, files whose
    name says mask, or integer label TIFFs named after a dropped image):
    :attr:`masks` pairs them.
``spacr_output``
    A folder spaCR wrote (``merged/*.npy``, a ``sorted_channels`` folder, or
    a ``merged`` folder itself): :attr:`description` says what it is and
    :attr:`open_folder` is the images folder to open, if any.
``nothing``
    Nothing Make Masks can use.

Whatever the drop held that is none of these -- a ``.npy``, a text file, an
empty folder, a mask no image claims -- is in :attr:`unrecognised` or
:attr:`unpaired_masks`, so the screen can list it instead of losing it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .channel_sorting import (DEFAULT_DEST_NAME, IMAGE_EXTS, list_folder_images,
                              natural_key, split_extension)

#: Subfolders that never hold images to edit or channels to sort.
_SKIPPED_DIRS = ("masks", "orig", DEFAULT_DEST_NAME, "merged", "stack")

#: Subfolder names that read as a channel on their own.
_CHANNEL_NAME = re.compile(
    r"(?i)^(?:dapi|hoechst|gfp|egfp|rfp|mcherry|yfp|cfp|fitc|tritc|cy3|cy5|"
    r"far[-_ ]?red|brightfield|bf|phase|dic|nuc(?:leus|lei)?|cell|cyto|"
    r"(?:ch|chan|channel|c|w|wave|wavelength)[-_ ]?\d{1,2})$")

#: Tokens stripped from file names before comparing fields across folders.
_CHANNEL_TOKEN = re.compile(
    r"(?i)(?:dapi|hoechst|gfp|egfp|rfp|mcherry|yfp|cfp|fitc|tritc|cy3|cy5|"
    r"(?:ch|chan|channel|c|w)\d{1,2})")

#: Suffixes and prefixes a mask's name carries beyond its image's stem.
_MASK_AFFIX = re.compile(
    r"(?i)(?:^masks?[-_ ]|[-_ ]?(?:cp_)?masks?$|[-_ ]?labels?$|[-_ ]?seg$)")

#: A folder name that says it holds masks.
_MASKS_FOLDER = re.compile(r"(?i)(?:.*[-_ ])?(?:cp_)?masks?(?:[-_ ]\d+)?")

#: How many loose TIFFs are read to tell label masks from images.
_PIXEL_CHECK_LIMIT = 64

#: The largest TIFF read for that check, in bytes.
_PIXEL_CHECK_BYTES = 64 * 1024 * 1024


@dataclass
class DropClassification:
    """What a drop on Make Masks is, and everything needed to act on it.

    :ivar kind: one of ``images``, ``folder``, ``nested``, ``folders``,
        ``images_with_masks``, ``spacr_output`` or ``nothing``.
    :ivar images: image files, absolute, in drop order (``images`` and
        ``images_with_masks``).
    :ivar folders: the folder dropped (``folder``, ``nested``), the folders
        (``folders``) or the folders dropped with loose images (``images``).
    :ivar channel_folders: the subfolders of a ``nested`` folder that hold
        images, naturally sorted; for ``folders``, the dropped folders.
    :ivar channel_like: whether :attr:`channel_folders` look like one
        channel each -- names such as DAPI or ch1, or the same fields in
        every one.
    :ivar masks: ``{image path: mask path}`` for ``images_with_masks``.
    :ivar unpaired_masks: masks that no dropped image claims.
    :ivar open_folder: for ``spacr_output``, the image folder to open, or
        None when there is none.
    :ivar description: one plain sentence saying what a ``spacr_output``
        drop is.
    :ivar unrecognised: dropped paths used for nothing, each with the reason.
    """

    kind: str
    images: List[str] = field(default_factory=list)
    folders: List[str] = field(default_factory=list)
    channel_folders: List[str] = field(default_factory=list)
    channel_like: bool = False
    masks: Dict[str, str] = field(default_factory=dict)
    unpaired_masks: List[str] = field(default_factory=list)
    open_folder: Optional[str] = None
    description: str = ""
    unrecognised: List[str] = field(default_factory=list)


def _is_image(path: str) -> bool:
    """Whether ``path`` names an image Make Masks opens.

    :param path: a file path.
    """
    return path.lower().endswith(IMAGE_EXTS)


def _skipped(name: str) -> bool:
    """Whether a subfolder ``name`` is one spaCR keeps its own things in.

    ``masks``, ``masks_2``, ``sorted_channels_3``... all count.

    :param name: a folder name.
    """
    base = re.sub(r"_\d+$", "", name.casefold())
    return base in _SKIPPED_DIRS or name.startswith(".")


def _is_masks_folder_name(name: str) -> bool:
    """Whether a folder name says it holds masks.

    ``masks``, ``masks_2``, ``cell_masks``, ``drawn-mask``: the name ENDS in
    mask(s), numbered or not. A name that merely contains the word
    (``maskless_run``, ``test_mask_run``) does not count.

    :param name: a folder name.
    """
    return bool(_MASKS_FOLDER.fullmatch(name))


def _has_mask_word(path: str) -> bool:
    """Whether a file's name or its folder's name says it is a mask.

    The file's stem carries a mask affix (``_mask``, ``_cp_masks``,
    ``mask_``...), or its folder is a masks folder
    (:func:`_is_masks_folder_name`).

    :param path: a file path.
    """
    stem = split_extension(os.path.basename(path))[0]
    return bool(_MASK_AFFIX.search(stem)) or _is_masks_folder_name(
        os.path.basename(os.path.dirname(path)))


def _looks_like_labels(path: str) -> bool:
    """Whether a TIFF holds an integer label image rather than intensities.

    A label image is integer, 2-D, has background (0), few distinct values,
    and is made of flat regions: nearly every pixel equals its right-hand
    neighbour. A microscope image, even an 8-bit one, has noise there.

    :param path: a TIFF path.
    :returns: False for anything that cannot be read or is too big to read.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        return False
    try:
        if os.path.getsize(path) > _PIXEL_CHECK_BYTES:
            return False
        import tifffile

        array = np.squeeze(np.asarray(tifffile.imread(path)))
    except Exception:
        return False
    if array.ndim != 2 or array.size < 4 or not np.issubdtype(
            array.dtype, np.integer):
        return False
    values = np.unique(array)
    if values[0] != 0 or values.size < 2 or values.size > 4096:
        return False
    flat = float(np.mean(array[:, 1:] == array[:, :-1]))
    return flat >= 0.9


def _mask_stem(path: str) -> str:
    """A mask's name without ``_mask``/``_cp_masks``/``mask_``, lower case.

    :param path: a mask path.
    """
    stem = split_extension(os.path.basename(path))[0]
    return _MASK_AFFIX.sub("", stem).lower()


def _image_stem(path: str) -> str:
    """An image's stem, lower case, for pairing with its mask.

    :param path: an image path.
    """
    return split_extension(os.path.basename(path))[0].lower()


def _pair_masks(images: Sequence[str], masks: Sequence[str]):
    """Pair each mask with the image whose stem it carries.

    An exact stem wins; otherwise the stem left after dropping a mask's
    affixes (``img1_cp_masks`` -> ``img1``). An image gets one mask at most.

    :param images: image paths.
    :param masks: mask paths.
    :returns: ``({image: mask}, [masks left over])``.
    """
    by_stem: Dict[str, List[str]] = {}
    for image in images:
        by_stem.setdefault(_image_stem(image), []).append(image)
    paired: Dict[str, str] = {}
    left: List[str] = []
    for mask in masks:
        candidates = (by_stem.get(_image_stem(mask))
                      or by_stem.get(_mask_stem(mask)) or [])
        free = [image for image in candidates if image not in paired]
        if free:
            paired[free[0]] = mask
        else:
            left.append(mask)
    return paired, left


def _image_subfolders(folder: str) -> List[str]:
    """``folder``'s direct subfolders that hold images, naturally sorted.

    :param folder: a folder.
    """
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = []
    for name in names:
        path = os.path.join(folder, name)
        if (os.path.isdir(path) and not os.path.islink(path)
                and not _skipped(name) and list_folder_images(path)):
            found.append(path)
    return sorted(found, key=lambda p: natural_key(os.path.basename(p)))


def _has_nested_images(folder: str, depth: int = 3) -> bool:
    """Whether images sit in subfolders of ``folder``, up to ``depth`` down.

    Stops at the first one found; spaCR's own folders are not searched.

    :param folder: a folder.
    :param depth: how many levels below ``folder`` to look.
    """
    if depth <= 0:
        return False
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return False
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isdir(path) and not os.path.islink(path) and not _skipped(name):
            if list_folder_images(path) or _has_nested_images(path, depth - 1):
                return True
    return False


def _field_names(folder: str) -> set:
    """The fields of a channel folder: stems without channel words.

    The folder's own name and tokens such as ``DAPI``, ``ch1`` or ``w2`` are
    taken out, so ``DAPI/f1_dapi.tif`` and ``GFP/f1_gfp.tif`` both read
    ``f1``.

    :param folder: a folder of images.
    """
    own = re.escape(os.path.basename(folder).lower())
    fields = set()
    for name in list_folder_images(folder):
        stem = split_extension(name)[0].lower()
        if own:
            stem = re.sub(own, "", stem)
        stem = _CHANNEL_TOKEN.sub("", stem)
        fields.add(re.sub(r"[-_. ]+", "_", stem).strip("_"))
    return fields


def _channel_like(folders: Sequence[str]) -> bool:
    """Whether each folder looks like one channel of the same fields.

    True for two or more folders when every folder's name reads as a
    channel (DAPI, GFP, ch1, C01, w1...), or when every folder holds the
    same fields once the channel words are taken out of the names.

    :param folders: folders of images.
    """
    if len(folders) < 2:
        return False
    if all(_CHANNEL_NAME.match(os.path.basename(f)) for f in folders):
        return True
    fields = [_field_names(f) for f in folders]
    return bool(fields[0]) and all(f == fields[0] for f in fields[1:])


def _spacr_output(folder: str) -> Optional[DropClassification]:
    """Recognise a folder spaCR wrote, and what Make Masks should open of it.

    :param folder: a dropped folder.
    :returns: a ``spacr_output`` classification, or None for any other folder.
    """
    name = os.path.basename(os.path.normpath(folder))
    base = re.sub(r"_\d+$", "", name.casefold())
    channel_dirs = sorted(
        d for d in (os.listdir(folder) if os.path.isdir(folder) else [])
        if re.fullmatch(r"C\d{2,}", d) and os.path.isdir(os.path.join(folder, d)))
    if base == DEFAULT_DEST_NAME or (channel_dirs and os.path.isfile(
            os.path.join(folder, "channel_sorting_manifest.csv"))):
        first = os.path.join(folder, channel_dirs[0]) if channel_dirs else None
        return DropClassification(
            "spacr_output", folders=[folder],
            open_folder=first if first and list_folder_images(first) else None,
            description=(f"{folder} is a folder of channels sorted by spaCR; "
                         + (f"opening its first channel, {first}." if first
                            else "it has no channel folders to open.")))

    def npys(path: str) -> int:
        """How many ``.npy`` files sit directly in ``path``.

        :param path: a folder.
        """
        try:
            return sum(1 for n in os.listdir(path) if n.lower().endswith(".npy"))
        except OSError:
            return 0

    if base == "merged" and npys(folder) and not list_folder_images(folder):
        parent = os.path.dirname(os.path.normpath(folder))
        return _spacr_output_of(parent, merged=folder)
    merged = os.path.join(folder, "merged")
    if os.path.isdir(merged) and npys(merged):
        return _spacr_output_of(folder, merged=merged)
    return None


def _spacr_output_of(folder: str, merged: str) -> DropClassification:
    """Describe a folder spaCR merged, and find its images, if any are left.

    :param folder: the folder holding ``merged/``.
    :param merged: its ``merged`` folder.
    """
    for candidate in (folder, os.path.join(folder, "orig")):
        if list_folder_images(candidate):
            return DropClassification(
                "spacr_output", folders=[folder], open_folder=candidate,
                description=(f"{folder} holds spaCR merged arrays in {merged} "
                             "(.npy, not images); opening its images in "
                             f"{candidate}."))
    return DropClassification(
        "spacr_output", folders=[folder], open_folder=None,
        description=(f"{folder} holds spaCR merged arrays in {merged}; they are "
                     ".npy arrays, not images, so Make Masks has nothing "
                     "to open there. Measure reads them."))


def classify_drop(paths: Iterable) -> DropClassification:
    """Say what a drop on Make Masks is, without asking or opening anything.

    :param paths: the dropped files and folders, in drop order.
    :returns: a :class:`DropClassification`.

    spaCR's own output counts only when it is the whole drop. Files named as
    masks are masks; beside images, label TIFFs are too, but only when a
    dropped image claims them by name, since a clean synthetic image can look
    like labels. Masks dropped alone open the images they belong to, when those
    are present.
    """
    paths = [os.path.abspath(os.fspath(p)) for p in paths]
    unrecognised: List[str] = []
    files: List[str] = []
    folders: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            folders.append(path)
        elif os.path.isfile(path) and _is_image(path):
            files.append(path)
        elif os.path.isfile(path) and path.lower().endswith(".npy"):
            unrecognised.append(f"{path} (a .npy array, not an image)")
        elif os.path.exists(path):
            unrecognised.append(f"{path} (not an image Make Masks opens)")
        else:
            unrecognised.append(f"{path} (not found)")

    if len(folders) == 1 and not files:
        found = _spacr_output(folders[0])
        if found is not None:
            found.unrecognised = unrecognised
            return found

    image_folders: List[str] = []
    mask_folders: List[str] = []
    for folder in folders:
        named_mask = _is_masks_folder_name(os.path.basename(folder))
        if list_folder_images(folder):
            (mask_folders if named_mask else image_folders).append(folder)
        elif len(folders) == 1 and not files and _has_nested_images(folder):
            image_folders.append(folder)
        else:
            unrecognised.append(f"{folder} (no images directly in it)")

    mask_files = [f for f in files if _has_mask_word(f)]
    loose = [f for f in files if f not in mask_files]
    for folder in mask_folders:
        mask_files.extend(os.path.join(folder, n)
                          for n in list_folder_images(folder))
    if (loose or image_folders) and len(loose) <= _PIXEL_CHECK_LIMIT and \
            len(loose) + len(mask_files) + len(image_folders) > 1:
        labels = [f for f in loose if _looks_like_labels(f)]
        others = [f for f in loose if f not in labels]
        for folder in image_folders:
            others.extend(os.path.join(folder, n)
                          for n in list_folder_images(folder))
        claimed = list(_pair_masks(others, labels)[0].values())
        mask_files.extend(claimed)
        loose = [f for f in loose if f not in claimed]

    if mask_files and (loose or image_folders):
        images = list(loose)
        for folder in image_folders:
            images.extend(os.path.join(folder, n)
                          for n in list_folder_images(folder))
        paired, left = _pair_masks(images, mask_files)
        if paired:
            return DropClassification(
                "images_with_masks", images=images, folders=image_folders,
                masks=paired, unpaired_masks=left, unrecognised=unrecognised)
        unrecognised.extend(f"{m} (a mask no dropped image claims)"
                            for m in left)
    elif mask_files and not (loose or image_folders):
        parents = {os.path.dirname(m) for m in mask_files}
        if len(parents) == 1:
            parent = parents.pop()
            owner = os.path.dirname(parent)
            if _is_masks_folder_name(os.path.basename(parent)) and \
                    list_folder_images(owner):
                return DropClassification(
                    "folder", folders=[owner], unrecognised=unrecognised,
                    description=(f"{parent} is a masks folder; opening the "
                                 f"images it belongs to, {owner}."))
        loose = list(files)
        image_folders = list(mask_folders)

    if not loose and not image_folders:
        return DropClassification("nothing", unrecognised=unrecognised)
    if loose:
        return DropClassification("images", images=loose, folders=image_folders,
                                  unrecognised=unrecognised)
    if len(image_folders) == 1:
        folder = image_folders[0]
        subfolders = _image_subfolders(folder)
        if _has_nested_images(folder):
            return DropClassification(
                "nested", folders=[folder], channel_folders=subfolders,
                channel_like=_channel_like(subfolders),
                unrecognised=unrecognised)
        return DropClassification("folder", folders=[folder],
                                  unrecognised=unrecognised)
    return DropClassification(
        "folders", folders=image_folders, channel_folders=list(image_folders),
        channel_like=_channel_like(image_folders), unrecognised=unrecognised)


def _folder_images(folder: str, recursive: bool = True) -> List[str]:
    """Every image under ``folder``, absolute, spaCR's own folders skipped.

    :param folder: a folder.
    :param recursive: descend into subfolders.
    :returns: paths, naturally sorted by their path below ``folder``.
    """
    found: List[str] = []
    for current, directories, filenames in os.walk(folder):
        directories[:] = [d for d in directories if not _skipped(d)
                          and not os.path.islink(os.path.join(current, d))]
        found.extend(os.path.join(current, n) for n in filenames
                     if _is_image(n) and not n.startswith("."))
        if not recursive:
            break
    return sorted(found, key=lambda p: natural_key(os.path.relpath(p, folder)))


def _accepts(path: str) -> bool:
    """Whether Make Masks' drop handler should take ``path`` and classify it.

    An image file, a ``.npy`` (to say what it is), a folder with images in
    it or in subfolders, or a folder spaCR wrote. Worker thread only.

    :param path: a dropped path.
    """
    if os.path.isfile(path):
        return _is_image(path) or path.lower().endswith(".npy")
    if not os.path.isdir(path):
        return False
    return bool(list_folder_images(path) or _has_nested_images(path)
                or _spacr_output(path) is not None)
