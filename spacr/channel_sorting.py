"""Sort a folder of images (and their masks) into channels, rename, and merge.

The case this module exists for: masks were drawn in Make Masks on a folder
holding two kinds of image at once -- nucleus-stain images with nucleus
masks and cell-stain images with cell masks, say -- and those have to become
one multi-channel field per position, in the layout spaCR's Mask/Measure
pipeline reads.

It happens in three separate steps, and nothing is moved until the last:

1. **Channels and sets.** Each image gets a CHANNEL, and images that show the
   same field in different channels share a SET. Channels come from the
   user's selection (:func:`parse_names` ``channels=``) or from a regex's
   ``chanID`` group; sets come from the regex's other named groups
   (:func:`parse_names`, checked by :func:`check_sets`), from a regex spaCR
   proposes (:func:`infer_regex`), or, ignoring names, from pairing each
   image with the most similar image of the same size in every other channel
   (:func:`detect_sets`).
2. **The plan.** :func:`build_plan` names every set in the Yokogawa /
   CellVoyager form ``plate1_A01_T0001F001L01A01Z01C01.tif`` -- exactly what
   ``spacr.utils._get_regex('cellvoyager', ...)`` parses -- and lists every
   move. It writes nothing.
3. **Apply.** :func:`apply_plan` moves each image into ``C01/``, ``C02/``...
   under a new folder, its mask into that channel folder's ``masks/`` under
   the new name (and its ``.curation.json`` ledger with it, so the pairing
   survives), records every move in ``channel_sorting_manifest.csv``, and
   then :func:`merge_sorted` builds ``stack/``, ``masks/<role>_mask_stack/``
   and ``merged/`` with spaCR's own functions
   (:func:`spacr.utils._extract_filename_metadata`,
   :func:`spacr.io._escaped_field_stem` and
   :func:`spacr.io._load_and_concatenate_arrays`), so each merged array has the
   pipeline's own layout: intensity channels first, then one label plane per
   mask role in the order cell, nucleus, pathogen.

Regex conventions: ``chanID`` is the channel; ``plateID``, ``wellID``,
``fieldID`` and ``timeID`` say where a set goes; any other named group only
helps tell sets apart. A group that did not take part in a match counts as
the empty string, so an optional ``(?:_(?P<fieldID>\\d+))?`` reads the
numbered copies :mod:`spacr.folder_consolidation` makes.
"""
from __future__ import annotations

import csv
import difflib
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Image extensions listed from a folder -- Make Masks' own list.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

#: The mask roles a channel's masks can take, in the pipeline's plane order.
MASK_ROLES = ("cell", "nucleus", "pathogen", "organelle")

#: The regex group that names the channel.
CHANNEL_GROUP = "chanID"

#: The key group :func:`detect_sets` gives each set it pairs.
DETECTED_GROUP = "set"

#: Where the channel folders, stacks and ``merged/`` go, beside the images.
DEFAULT_DEST_NAME = "sorted_channels"

#: The move/rename record written into the destination.
MANIFEST_NAME = "channel_sorting_manifest.csv"

#: spaCR's own Yokogawa pattern, in a form that works for any extension.
CELLVOYAGER_EXAMPLE = (
    r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
    r"L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*)\.tif")

#: Wells on one plate before a set is put on the next plate (384-well).
WELLS_PER_PLATE = 384

#: A set key: the named groups other than ``chanID``, as (name, value) pairs.
SetKey = Tuple[Tuple[str, str], ...]


def natural_key(text) -> tuple:
    """Sort key that orders ``img2`` before ``img10``.

    :param text: a name.
    :returns: a tuple comparing numbers numerically and text case-blind.
    """
    parts = re.split(r"(\d+)", str(text))
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.lower())
                 for p in parts if p != "")


def split_extension(name: str) -> Tuple[str, str]:
    """Split ``name`` into stem and extension, keeping ``.ome.tif`` whole.

    :param name: a file name.
    :returns: ``(stem, extension)``; the extension keeps its dot.
    """
    lower = name.lower()
    for double in (".ome.tiff", ".ome.tif"):
        if lower.endswith(double):
            return name[:-len(double)], name[-len(double):]
    return os.path.splitext(name)


def list_folder_images(folder: str) -> List[str]:
    """Return the image file names directly in ``folder``, naturally sorted.

    :param folder: the folder; a missing one holds none.
    """
    if not folder or not os.path.isdir(folder):
        return []
    return sorted((name for name in os.listdir(folder)
                   if not name.startswith(".")
                   and name.lower().endswith(IMAGE_EXTS)
                   and os.path.isfile(os.path.join(folder, name))),
                  key=natural_key)


def masks_folder(folder: str, masks_dir: Optional[str] = None) -> str:
    """Where Make Masks keeps the masks of ``folder``'s images.

    :param folder: the image folder.
    :param masks_dir: an explicit masks folder, when not ``<folder>/masks``.
    """
    return os.fspath(masks_dir) if masks_dir else os.path.join(folder, "masks")


def mask_for(folder: str, name: str,
             masks_dir: Optional[str] = None) -> Optional[str]:
    """Return the path of the saved mask of image ``name``, or None.

    Make Masks saves every mask as ``<masks folder>/<stem>.tif``
    (:func:`spacr.qt.mask_engine.mask_save_path`). An image named by an
    absolute path outside ``folder`` -- one dropped into the sort dialog
    from elsewhere, or a channel subfolder's -- has its mask in its own
    folder's ``masks/``.

    :param folder: the image folder.
    :param name: the image file name, or an absolute path.
    :param masks_dir: an explicit masks folder, for images in ``folder``.
    """
    base = masks_folder(folder, masks_dir)
    if os.path.isabs(name) and (os.path.normpath(os.path.dirname(name))
                                != os.path.normpath(os.path.abspath(folder))):
        base = os.path.join(os.path.dirname(name), "masks")
    path = os.path.join(base,
                        split_extension(os.path.basename(name))[0] + ".tif")
    return path if os.path.isfile(path) else None


def _plane_conversion(shape: Optional[Tuple[int, ...]]) -> Optional[str]:
    """How an image that is not one 2-D plane could become one.

    :param shape: the image's shape from :func:`image_shape`.
    :returns: ``"rgb"`` for ``(H, W, 3|4)`` (converted to grey by the mean of
        its colours), ``"zstack"`` for any other 3-D shape (converted by a
        maximum projection over the first axis), None for a 2-D plane or a
        shape that cannot be converted.
    """
    if shape is None or len(shape) != 3:
        return None
    return "rgb" if shape[-1] in (3, 4) else "zstack"


def _converted_shape(shape: Tuple[int, ...], kind: Optional[str]) -> Tuple[int, ...]:
    """The 2-D shape an image has after :func:`_convert_plane`.

    :param shape: the image's shape.
    :param kind: ``"rgb"``, ``"zstack"`` or None.
    :returns: the shape after conversion (unchanged when ``kind`` is None).
    """
    if kind == "rgb":
        return tuple(shape[:2])
    if kind == "zstack":
        return tuple(shape[1:])
    return tuple(shape)


def _convert_plane(array: np.ndarray, kind: str) -> np.ndarray:
    """Turn an RGB image or a z-stack into one 2-D plane.

    RGB (and RGBA, whose alpha is dropped) becomes the mean of its colours in
    the original dtype; a z-stack becomes its maximum projection.

    :param array: the image's pixels, singleton axes already squeezed.
    :param kind: ``"rgb"`` or ``"zstack"``.
    :returns: a 2-D array.
    """
    if kind == "rgb":
        colours = array[..., :3].astype(np.float64).mean(axis=-1)
        if np.issubdtype(array.dtype, np.integer):
            return np.round(colours).astype(array.dtype)
        return colours.astype(array.dtype)
    return array.max(axis=0)


def image_shape(path: str) -> Optional[Tuple[int, ...]]:
    """Read an image's pixel shape from its header, without its pixels.

    Singleton axes are dropped, so a ``(1, H, W)`` TIFF and an ``(H, W)``
    PNG compare equal.

    :param path: an image file.
    :returns: the shape, or None when the file cannot be read.
    """
    shape = None
    try:
        if path.lower().endswith((".tif", ".tiff")):
            import tifffile

            with tifffile.TiffFile(path) as handle:
                shape = tuple(int(n) for n in handle.series[0].shape)
        else:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                bands = len(image.getbands())
                shape = (height, width) if bands == 1 else (height, width, bands)
    except Exception:
        return None
    return tuple(n for n in shape if n != 1) or (1,)


def thumbnail(path: str, size: int = 96) -> Optional[np.ndarray]:
    """A small 8-bit grey preview of an image, contrast-stretched.

    :param path: the image file.
    :param size: the longest side, in pixels.
    :returns: a 2-D ``uint8`` array, or None when it cannot be read.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image)
    except Exception:
        try:
            import tifffile

            array = tifffile.imread(path)
        except Exception:
            return None
    array = np.squeeze(np.asarray(array))
    if array.ndim == 3:
        array = array.max(axis=-1 if array.shape[-1] <= 4 else 0)
    if array.ndim != 2 or array.size == 0:
        return None
    step = max(1, int(np.ceil(max(array.shape) / float(size))))
    array = array[::step, ::step].astype(np.float32)
    low, high = np.percentile(array, (1, 99.5))
    if high <= low:
        high = low + 1.0
    return (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def mask_thumbnail(path: str, size: int = 96) -> Optional[np.ndarray]:
    """A small preview of a label mask: 255 where any object is, else 0.

    :param path: the mask file.
    :param size: the longest side, in pixels.
    :returns: a 2-D ``uint8`` array, or None when it cannot be read.
    """
    try:
        import tifffile

        array = np.squeeze(np.asarray(tifffile.imread(path)))
    except Exception:
        return None
    if array.ndim != 2 or array.size == 0:
        return None
    step = max(1, int(np.ceil(max(array.shape) / float(size))))
    return ((array[::step, ::step] > 0) * 255).astype(np.uint8)




@dataclass
class ParsedName:
    """One image name read through a regex.

    :ivar name: the file name.
    :ivar matched: whether the regex matched it.
    :ivar groups: every named group's value; one that did not take part is
        the empty string.
    :ivar channel: the 1-based channel, from the selection or ``chanID``;
        None when neither gives one.
    :ivar set_key: the set the image belongs to; None when unmatched.
    """

    name: str
    matched: bool
    groups: Dict[str, str] = field(default_factory=dict)
    channel: Optional[int] = None
    set_key: Optional[SetKey] = None


def compile_regex(pattern: str):
    """Compile ``pattern``, or return the error message.

    :param pattern: the regex text.
    :returns: ``(compiled, None)`` or ``(None, message)``.
    """
    try:
        return re.compile(pattern), None
    except re.error as exc:
        return None, str(exc)


def parse_names(names: Sequence[str], pattern: Optional[str],
                channels: Optional[Dict[str, int]] = None) -> List[ParsedName]:
    """Read each name's channel and set through ``pattern``.

    The channel of a name the user assigned (``channels``) is that
    assignment; otherwise it is its ``chanID`` value's position among all
    ``chanID`` values, naturally sorted, from 1. The set key is every other
    named group, in the pattern's order. A name the pattern does not match
    has no set.

    :param names: image file names, or absolute paths for images outside
        the folder; the regex reads the file name alone.
    :param pattern: the regex, matched from the start of each name; empty or
        None matches nothing.
    :param channels: ``{name: channel}`` from the selection strategy.
    :returns: one :class:`ParsedName` per name, in order.
    """
    channels = dict(channels or {})
    compiled = compile_regex(pattern)[0] if pattern else None
    parsed: List[ParsedName] = []
    for name in names:
        match = (compiled.match(os.path.basename(name))
                 if compiled is not None else None)
        if match is None:
            parsed.append(ParsedName(name, False, {}, channels.get(name), None))
            continue
        groups = {key: (value if value is not None else "")
                  for key, value in match.groupdict().items()}
        key = tuple((group, groups[group])
                    for group in sorted(compiled.groupindex,
                                        key=compiled.groupindex.get)
                    if group != CHANNEL_GROUP)
        parsed.append(ParsedName(name, True, groups, channels.get(name), key))
    values = sorted({p.groups[CHANNEL_GROUP] for p in parsed
                     if p.matched and p.channel is None
                     and CHANNEL_GROUP in p.groups},
                    key=natural_key)
    index = {value: position + 1 for position, value in enumerate(values)}
    for item in parsed:
        if item.channel is None and item.matched and CHANNEL_GROUP in item.groups:
            item.channel = index[item.groups[CHANNEL_GROUP]]
    return parsed


@dataclass
class SetReport:
    """Whether a set of parsed names forms complete, unique sets.

    :ivar channels: the channels present, sorted.
    :ivar complete: ``{set key: {channel: name}}`` for sets with exactly one
        image in every channel.
    :ivar incomplete: ``{set key: [missing channels]}``.
    :ivar duplicated: ``{(set key, channel): [names]}`` -- two images
        claiming the same place.
    :ivar unmatched: names the regex did not match.
    :ivar no_channel: matched names with no channel.
    """

    channels: List[int] = field(default_factory=list)
    complete: Dict[SetKey, Dict[int, str]] = field(default_factory=dict)
    incomplete: Dict[SetKey, List[int]] = field(default_factory=dict)
    duplicated: Dict[Tuple[SetKey, int], List[str]] = field(default_factory=dict)
    unmatched: List[str] = field(default_factory=list)
    no_channel: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every image is in exactly one complete set."""
        return bool(self.complete) and not (
            self.incomplete or self.duplicated or self.unmatched
            or self.no_channel)

    def summary(self) -> str:
        """A few plain lines saying what is and is not right."""
        lines = [f"{len(self.complete)} complete set(s) across "
                 f"{len(self.channels)} channel(s)."]
        if self.unmatched:
            lines.append(f"{len(self.unmatched)} file(s) not matched: "
                         + _few(self.unmatched))
        if self.no_channel:
            lines.append(f"{len(self.no_channel)} file(s) have no channel: "
                         + _few(self.no_channel))
        if self.incomplete:
            lines.append(f"{len(self.incomplete)} set(s) incomplete: "
                         + _few([f"{set_label(key)} lacks channel(s) "
                                 + ", ".join(str(c) for c in missing)
                                 for key, missing in self.incomplete.items()]))
        if self.duplicated:
            lines.append(f"{len(self.duplicated)} duplicate(s): " + _few(
                [f"{set_label(key)} channel {channel}: " + " / ".join(names)
                 for (key, channel), names in self.duplicated.items()]))
        return "\n".join(lines)


def _few(items: Sequence[str], limit: int = 3) -> str:
    """Join the first few of ``items`` and say how many more there are.

    :param items: texts.
    :param limit: how many to show.
    """
    shown = ", ".join(str(item) for item in list(items)[:limit])
    more = len(items) - limit
    return shown + (f" and {more} more" if more > 0 else "")


def set_label(key: Optional[SetKey]) -> str:
    """A set key as short text, ``wellID=A01 fieldID=2``.

    :param key: the set key.
    """
    if key is None:
        return ""
    if not key:
        return "(one set)"
    return " ".join(f"{name}={value}" for name, value in key)


def check_sets(parsed: Sequence[ParsedName]) -> SetReport:
    """Group parsed names into sets and report every way they fail to fit.

    :param parsed: the output of :func:`parse_names`.
    :returns: a :class:`SetReport`.
    """
    report = SetReport()
    members: Dict[SetKey, Dict[int, List[str]]] = defaultdict(
        lambda: defaultdict(list))
    for item in parsed:
        if not item.matched:
            report.unmatched.append(item.name)
        elif item.channel is None:
            report.no_channel.append(item.name)
        else:
            members[item.set_key][item.channel].append(item.name)
    report.channels = sorted({channel for sets in members.values()
                              for channel in sets})
    for key, by_channel in members.items():
        missing = [c for c in report.channels if c not in by_channel]
        doubles = {c: names for c, names in by_channel.items() if len(names) > 1}
        for channel, names in doubles.items():
            report.duplicated[(key, channel)] = list(names)
        if missing:
            report.incomplete[key] = missing
        if not missing and not doubles:
            report.complete[key] = {c: names[0]
                                    for c, names in sorted(by_channel.items())}
    return report



_SEPARATORS = re.compile(r"([_\-. ]+)")
_RUNS = re.compile(r"\d+|[A-Za-z]+|[^A-Za-z\d]+")
_WELL = re.compile(r"^(?:[A-Pa-p]\d{1,2}|r\d{1,2}c\d{1,2})$")
_CHANNEL_WORDS = re.compile(r"(?i)(?:c|ch|chan|channel|w|wave|wavelength)$")


def _separator_tokens(stem: str) -> List[Tuple[str, bool]]:
    """Split ``stem`` at ``_ - . space`` runs.

    :param stem: a file name without extension.
    :returns: ``[(text, is_separator)]``.
    """
    return [(part, bool(_SEPARATORS.fullmatch(part)))
            for part in _SEPARATORS.split(stem) if part != ""]


def _run_tokens(stem: str) -> List[Tuple[str, bool]]:
    """Split ``stem`` into digit, letter and other runs.

    :param stem: a file name without extension.
    :returns: ``[(text, is_separator)]``; non-alphanumeric runs are
        separators.
    """
    return [(part, not part.isalnum()) for part in _RUNS.findall(stem)]


def _uniform(tokenised: List[List[Tuple[str, bool]]]) -> bool:
    """Whether every name has the same token layout and separators.

    :param tokenised: per name, its tokens.
    """
    first = tokenised[0]
    for tokens in tokenised[1:]:
        if len(tokens) != len(first):
            return False
        for (text, sep), (text0, sep0) in zip(tokens, first):
            if sep != sep0 or (sep and text != text0):
                return False
            if not sep and text.isdigit() != text0.isdigit():
                return False
    return True


def _class_for(values: Sequence[str]) -> str:
    """The narrowest simple character class that matches all ``values``.

    :param values: the texts a group must match.
    """
    if all(value.isdigit() for value in values):
        return r"\d+"
    seen = set("".join(values))
    pieces = []
    if any(c.isalpha() for c in seen):
        pieces.append("A-Za-z")
    if any(c.isdigit() for c in seen):
        pieces.append("0-9")
    others = "".join(sorted(c for c in seen if not c.isalnum()))
    if others:
        pieces.append(re.escape(others))
    return "[" + "".join(pieces) + "]+"


def _factor(values: Sequence[str]) -> Tuple[str, str, List[str]]:
    """Split each value into a shared literal head, a varying middle and tail.

    Digit-only values are never factored (``09`` and ``10`` share no head),
    and a factoring that would leave any middle empty is not used.

    :param values: one token position's values.
    :returns: ``(head, tail, middles)``.
    """
    if all(value.isdigit() for value in values):
        return "", "", list(values)
    head = os.path.commonprefix(list(values))
    rest = [value[len(head):] for value in values]
    tail = os.path.commonprefix([r[::-1] for r in rest])[::-1]
    middles = [r[:len(r) - len(tail)] if tail else r for r in rest]
    if head and head[-1].isdigit():
        stripped = head.rstrip("0123456789")
        middles = [head[len(stripped):] + m for m in middles]
        head = stripped
    if tail and tail[0].isdigit():
        stripped = tail.lstrip("0123456789")
        middles = [m + tail[:len(tail) - len(stripped)] for m in middles]
        tail = stripped
    if any(m == "" for m in middles):
        return "", "", list(values)
    return head, tail, middles


def _role_hint(literal: str) -> str:
    """The role a literal right before a slot suggests, if any.

    :param literal: text before the varying part.
    :returns: a group name from :func:`spacr.regex_infer.hint_for`, or ``""``.
    """
    from .regex_infer import hint_for

    return hint_for(literal) if literal else ""


def _whole_word_role(word: str) -> str:
    """The role a whole separate word names, such as ``well`` or ``field``.

    Unlike :func:`_role_hint`, the word must BE a hint (``well_01``), not
    merely end in one: ``wt_01`` says nothing about time.

    :param word: the token before a separator.
    """
    from .regex_infer import LITERAL_HINTS

    return LITERAL_HINTS.get(str(word).upper(), "")


def _candidate_regexes(names: Sequence[str], channels: Optional[Dict[str, int]]
                       ) -> List[str]:
    """Propose regexes for ``names``, most plausible first.

    :param names: image file names.
    :param channels: the selection's channels, when every name has one.
    """
    stems, exts = zip(*(split_extension(os.path.basename(n)) for n in names))
    ext_values = sorted({e for e in exts}, key=str.lower)
    ext_regex = (re.escape(ext_values[0]) if len(ext_values) == 1 else
                 "(?:" + "|".join(re.escape(e) for e in ext_values) + ")")
    variants = [(list(stems), "")]
    suffix = [re.match(r"^(.*?)_(\d+)$", s) for s in stems]
    if any(suffix) and not all(suffix):
        stripped = [m.group(1) if m else s for m, s in zip(suffix, stems)]
        variants.append((stripped, r"(?:_(?P<fieldID>\d+))?"))
    out: List[str] = []
    for variant_stems, tail in variants:
        for tokeniser in (_separator_tokens, _run_tokens):
            tokenised = [tokeniser(s) for s in variant_stems]
            if not tokenised[0] or not _uniform(tokenised):
                continue
            out.extend(_regexes_for_family(names, tokenised, channels,
                                           tail, ext_regex))
    return out


def _regexes_for_family(names, tokenised, channels, tail, ext_regex) -> List[str]:
    """Regexes for names that share one token layout, best first.

    :param names: the file names, in the order of ``tokenised``.
    :param tokenised: each name's tokens.
    :param channels: ``{name: channel}`` for every name, or None.
    :param tail: regex text for an optional numbered suffix, or ``""``.
    :param ext_regex: regex text for the extension.
    """
    width = len(tokenised[0])
    columns = [[tokens[i][0] for tokens in tokenised] for i in range(width)]
    varying = [i for i in range(width)
               if not tokenised[0][i][1] and len(set(columns[i])) > 1]
    tail_values = None
    if tail:
        tail_values = [m.group(1) if m else "" for m in
                       (re.match(r"^.*?_(\d+)$",
                                 split_extension(os.path.basename(n))[0])
                        for n in names)]

    def function_of(position: int, labels: Sequence) -> bool:
        """Whether column ``position`` is fixed by ``labels``.

        :param position: a token column.
        :param labels: one label per name.
        """
        seen: Dict = {}
        for label, value in zip(labels, columns[position]):
            if seen.setdefault(label, value) != value:
                return False
        return True

    def complete(identity: Sequence[int], labels: Sequence) -> bool:
        """Whether ``identity`` columns give each label exactly once per set.

        :param identity: the columns that identify a set.
        :param labels: the channel label of each name.
        """
        sets: Dict[tuple, List] = defaultdict(list)
        for row, label in enumerate(labels):
            key = tuple(columns[i][row] for i in identity)
            if tail_values is not None:
                key += (tail_values[row],)
            sets[key].append(label)
        wanted = sorted(set(labels), key=str)
        return len(wanted) >= 1 and all(
            sorted(members, key=str) == wanted for members in sets.values())

    choices = []
    if channels and all(n in channels for n in names):
        labels = [channels[n] for n in names]
        determined = [i for i in varying if function_of(i, labels)]
        identity = [i for i in varying if i not in determined]
        if complete(identity, labels):
            choices.append((None, determined, identity))
    else:
        for position in varying:
            labels = columns[position]
            distinct = len(set(labels))
            if distinct < 2 or distinct > 12:
                continue
            determined = [i for i in varying if i != position
                          and function_of(i, labels)]
            identity = [i for i in varying
                        if i != position and i not in determined]
            if not complete(identity, labels):
                continue
            before = "".join(t[0] for t in tokenised[0][:position])
            hinted = bool(_CHANNEL_WORDS.search(
                before.rstrip("_-. ") + _factor(labels)[0]))
            choices.append(((not hinted, distinct, -position), position,
                            determined, identity))
        choices.sort(key=lambda item: item[0])
        choices = [(position, determined, identity)
                   for _score, position, determined, identity in choices]

    regexes = []
    for channel_position, determined, identity in choices:
        pieces = []
        used_roles = set()
        generic = 0
        for i in range(width):
            text, separator = tokenised[0][i]
            if i not in varying:
                pieces.append(re.escape(text))
                continue
            is_well = (i != channel_position and i not in determined
                       and all(_WELL.match(v) for v in columns[i]))
            head, tail_text, middles = (("", "", columns[i]) if is_well
                                        else _factor(columns[i]))
            cls = _class_for(middles)
            if i == channel_position:
                role = CHANNEL_GROUP
            elif i in determined:
                pieces.append(re.escape(head) + cls + re.escape(tail_text))
                continue
            else:
                before = [j for j in range(i) if not tokenised[0][j][1]]
                previous = ""
                if before and before[-1] not in varying:
                    previous = tokenised[0][before[-1]][0]
                role = ""
                if is_well:
                    role = "wellID"
                if not role:
                    role = _role_hint(head) or _whole_word_role(previous)
                if role in ("chanID", "") or role in used_roles:
                    role = "fieldID" if "fieldID" not in used_roles else ""
                if not role:
                    generic += 1
                    role = f"id{generic}"
            used_roles.add(role)
            pieces.append(re.escape(head) + f"(?P<{role}>{cls})"
                          + re.escape(tail_text))
        body = "".join(pieces)
        suffix = tail
        if tail and "fieldID" in used_roles:
            suffix = r"(?:_(?P<copy>\d+))?"
        regexes.append(body + suffix + ext_regex + "$")
    return regexes


def infer_regex(names: Sequence[str],
                channels: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Propose a regex that maps every image to a unique, complete set.

    Names are split into tokens (at ``_ - . space``, and failing that into
    digit and letter runs); the tokens that vary are found; one is tried as
    the channel -- or, when the user assigned channels, the tokens that only
    follow the channel are set aside -- and the rest must identify a set
    that holds exactly one image per channel. Each candidate is checked with
    :func:`check_sets` before it is returned. spaCR's own Yokogawa pattern is
    tried first, and :func:`spacr.regex_infer.propose` last. Of the
    candidates that pass, the one with the FEWEST channels wins: any name can
    be read as a channel of its own in a single set, which passes and means
    nothing.

    :param names: image file names.
    :param channels: ``{name: channel}`` from the selection strategy; when
        every name has one, the regex needs no ``chanID`` group.
    :returns: the first regex that passes, or None.
    """
    names = [n for n in names if n]
    if not names:
        return None
    channels = dict(channels or {})

    def channel_count(pattern: str) -> Optional[int]:
        """How many channels ``pattern`` gives, when it puts every name in a set.

        :param pattern: a candidate regex.
        :returns: the channel count, or None when the pattern fails.
        """
        compiled, _error = compile_regex(pattern)
        if compiled is None:
            return None
        if CHANNEL_GROUP not in compiled.groupindex and not (
                channels and all(n in channels for n in names)):
            return None
        report = check_sets(parse_names(names, pattern, channels))
        return len(report.channels) if report.ok else None

    cellvoyager = CELLVOYAGER_EXAMPLE.replace(r"\.tif", r"\.[A-Za-z]+$")
    if channel_count(cellvoyager):
        return cellvoyager
    candidates = list(_candidate_regexes(names, channels))
    try:
        from .regex_infer import propose

        candidates += [p.pattern for p in propose(
            [os.path.basename(n) for n in names])]
    except Exception:
        pass
    passing = []
    for order, pattern in enumerate(candidates):
        count = channel_count(pattern)
        if count:
            passing.append((count, order, pattern))
    return min(passing)[2] if passing else None




def _name_tokens(name: str) -> List[str]:
    """Letter and number tokens of a name, numbers unpadded, lower case.

    :param name: a file name.
    """
    stem = split_extension(os.path.basename(name))[0]
    return [str(int(t)) if t.isdigit() else t.lower()
            for t in re.findall(r"\d+|[A-Za-z]+", stem)]


def name_distance(a: str, b: str) -> float:
    """How different two names are, 0 (same tokens) to 1 (nothing shared).

    :param a: a file name.
    :param b: another.
    """
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta and not tb:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


@dataclass
class DetectedSets:
    """What :func:`detect_sets` paired.

    :ivar sets: ``{set key: {channel: name}}``, keyed ``(("set", "000001"),)``.
    :ivar unpaired: images left without a partner in some channel.
    :ivar problems: reasons the pairing cannot be used as it is.
    """

    sets: Dict[SetKey, Dict[int, str]] = field(default_factory=dict)
    unpaired: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def detect_sets(folder: str, channels: Dict[str, int],
                masks_dir: Optional[str] = None,
                shapes: Optional[Dict[str, Optional[tuple]]] = None
                ) -> DetectedSets:
    """Pair images across channels by name similarity, order and size.

    The channel with the most images is the reference. Each other channel's
    images are matched one-to-one to the reference images by the smallest
    total cost, where the cost is :func:`name_distance` plus a small term
    for how far apart the two sit in their channel's sorted order, and a pair
    whose images differ in size is not allowed. Names play no other part:
    each resulting set is keyed by its number alone. A mask whose size is not
    its image's is reported as a problem.

    :param folder: the image folder.
    :param channels: ``{name: channel}`` for every image to pair.
    :param masks_dir: an explicit masks folder.
    :param shapes: ``{name: shape}`` already read; read here otherwise.
    :returns: a :class:`DetectedSets`.
    """
    result = DetectedSets()
    by_channel: Dict[int, List[str]] = defaultdict(list)
    for name, channel in channels.items():
        by_channel[int(channel)].append(name)
    if not by_channel:
        result.problems.append("No image has a channel yet.")
        return result
    for names in by_channel.values():
        names.sort(key=natural_key)
    shapes = dict(shapes or {})
    for name in channels:
        if name not in shapes:
            shapes[name] = image_shape(os.path.join(folder, name))
    for name in channels:
        mask = mask_for(folder, name, masks_dir)
        if mask is not None and shapes.get(name) is not None:
            mask_shape = image_shape(mask)
            if mask_shape is not None and mask_shape != shapes[name][:2] and \
                    mask_shape != shapes[name]:
                result.problems.append(
                    f"The mask of {name} is {mask_shape}, its image "
                    f"{shapes[name]}.")
    reference = sorted(by_channel, key=lambda c: (-len(by_channel[c]), c))[0]
    ref_names = by_channel[reference]
    partners: Dict[str, Dict[int, str]] = {n: {reference: n} for n in ref_names}
    forbidden = 1e6
    for channel, names in sorted(by_channel.items()):
        if channel == reference:
            continue
        cost = np.zeros((len(ref_names), len(names)))
        for i, a in enumerate(ref_names):
            for j, b in enumerate(names):
                if shapes.get(a) != shapes.get(b) or shapes.get(a) is None:
                    cost[i, j] = forbidden
                    continue
                order = abs(i / max(len(ref_names), 1) - j / max(len(names), 1))
                cost[i, j] = name_distance(a, b) + 0.01 * order
        rows, cols = _assign(cost)
        taken = set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= forbidden:
                continue
            partners[ref_names[i]][channel] = names[j]
            taken.add(names[j])
        result.unpaired.extend(n for n in names if n not in taken)
    wanted = sorted(by_channel)
    number = 0
    for name in ref_names:
        members = partners[name]
        if sorted(members) != wanted:
            result.unpaired.append(name)
            result.unpaired.extend(v for c, v in members.items() if v != name)
            continue
        number += 1
        result.sets[((DETECTED_GROUP, f"{number:06d}"),)] = dict(
            sorted(members.items()))
    if result.unpaired:
        result.problems.append(
            f"{len(result.unpaired)} image(s) have no partner of the same size "
            f"in every channel: " + _few(sorted(set(result.unpaired),
                                                key=natural_key)))
    return result


def _assign(cost: np.ndarray):
    """Minimum-cost one-to-one assignment of rows to columns.

    :param cost: a rows-by-columns cost matrix.
    :returns: ``(rows, columns)`` index arrays.
    """
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        rows, cols, used_rows, used_cols = [], [], set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = divmod(int(flat), cost.shape[1])
            if i not in used_rows and j not in used_cols:
                rows.append(i)
                cols.append(j)
                used_rows.add(i)
                used_cols.add(j)
        return np.array(rows), np.array(cols)




@dataclass(frozen=True)
class SetName:
    """Where one set goes in the Yokogawa naming.

    :ivar plate: the plate token, with no underscore.
    :ivar well: e.g. ``A01``.
    :ivar field: 1-based field.
    :ivar time: 1-based timepoint.
    """

    plate: str
    well: str
    field: int
    time: int = 1


def well_name(index: int) -> str:
    """The ``index``-th well of a 384-well plate, row by row: 0 is ``A01``.

    :param index: 0-based, below :data:`WELLS_PER_PLATE`.
    """
    row, column = divmod(int(index), 24)
    return f"{chr(ord('A') + row)}{column + 1:02d}"


def canonical_well(text: str) -> Optional[str]:
    """``A1``, ``a01`` or ``r01c01`` as ``A01``; None for anything else.

    :param text: a well token.
    """
    text = str(text)
    match = re.fullmatch(r"([A-Pa-p])(\d{1,2})", text)
    if match and 1 <= int(match.group(2)) <= 24:
        return f"{match.group(1).upper()}{int(match.group(2)):02d}"
    match = re.fullmatch(r"(?i)r(\d{1,2})c(\d{1,2})", text)
    if match and 1 <= int(match.group(1)) <= 16 and 1 <= int(match.group(2)) <= 24:
        return f"{chr(ord('A') + int(match.group(1)) - 1)}{int(match.group(2)):02d}"
    return None


def _plate_token(text: str) -> str:
    """A plate name safe for the Yokogawa pattern (no underscores).

    :param text: the plate group's value.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-")
    return cleaned or "plate1"


def name_sets(keys: Iterable[SetKey]) -> Dict[SetKey, SetName]:
    """Give every set a unique plate, well, field and time.

    Sets keyed by :func:`detect_sets` get one well each, ``A01`` onward,
    field 1, 384 wells to a plate. Otherwise the plate is ``plateID``
    (or ``plate1``); the well is ``wellID`` when every value reads as a well,
    else wells are numbered in natural order; the field is ``fieldID`` when
    it is the only other group and every value is a whole number from 1 to
    999, else fields are numbered within their well; without a ``wellID``
    group the fields fill ``A01`` to 999 and then move on to ``A02``. The
    time is ``timeID`` when every value is a whole number from 1, else
    numbered; 1 without one.

    :param keys: set keys, as :func:`check_sets` or :func:`detect_sets`
        return them.
    :returns: ``{key: SetName}``.
    :raises ValueError: when two sets would get one name.
    """
    keys = list(keys)
    names: Dict[SetKey, SetName] = {}
    if keys and all(dict(k).keys() == {DETECTED_GROUP} for k in keys):
        for index, key in enumerate(sorted(keys, key=lambda k: natural_key(
                dict(k)[DETECTED_GROUP]))):
            plate, well = divmod(index, WELLS_PER_PLATE)
            names[key] = SetName(f"plate{plate + 1}", well_name(well), 1, 1)
        return names

    table = [dict(key) for key in keys]
    plate_raw = sorted({d.get("plateID", "") for d in table}, key=natural_key)
    plates: Dict[str, str] = {}
    for raw in plate_raw:
        token = _plate_token(raw) if raw else "plate1"
        while token in plates.values():
            token = f"{token}-{len(plates) + 1}"
        plates[raw] = token

    time_raw = sorted({d.get("timeID", "") for d in table}, key=natural_key)
    if all(t.isdigit() and 1 <= int(t) <= 9999 for t in time_raw):
        times = {t: int(t) for t in time_raw}
    else:
        times = {t: i + 1 for i, t in enumerate(time_raw)}

    constant = {k for k in {k for d in table for k in d}
                if len({d.get(k, "") for d in table}) == 1}

    def identity(d: Dict[str, str]) -> tuple:
        """The groups that tell fields apart within a well.

        Groups with one value across every set (the ``L01``/``A01`` of a
        Yokogawa name) tell nothing apart and are left out.

        :param d: one set's groups.
        """
        return tuple((k, v) for k, v in d.items()
                     if k not in ("plateID", "wellID", "timeID")
                     and k not in constant)

    has_well = any("wellID" in d for d in table)
    wells: Dict[Tuple[str, str], str] = {}
    if has_well:
        for raw_plate in plate_raw:
            raws = sorted({d.get("wellID", "") for d in table
                           if d.get("plateID", "") == raw_plate}, key=natural_key)
            canon = [canonical_well(r) for r in raws]
            if all(canon) and len(set(canon)) == len(canon):
                wells.update({(raw_plate, r): c for r, c in zip(raws, canon)})
            else:
                if len(raws) > WELLS_PER_PLATE:
                    raise ValueError("More than 384 wells on one plate.")
                wells.update({(raw_plate, r): well_name(i)
                              for i, r in enumerate(raws)})

    fields_of: Dict[tuple, Dict[tuple, Tuple[str, int]]] = {}
    groups = defaultdict(set)
    for d in table:
        groups[(d.get("plateID", ""),
                d.get("wellID", "") if has_well else "")].add(identity(d))
    for group_key, idents in groups.items():
        ordered = sorted(idents, key=lambda ident: natural_key(
            " ".join(v for _k, v in ident)))
        only_field = all(len(i) == 1 and i[0][0] == "fieldID" for i in ordered)
        numbers = [i[0][1] for i in ordered] if only_field else []
        keep = (only_field and has_well and all(
            n.isdigit() and 1 <= int(n) <= 999 for n in numbers)
            and len({int(n) for n in numbers}) == len(numbers))
        mapping = {}
        for index, ident in enumerate(ordered):
            if keep:
                mapping[ident] = ("", int(ident[0][1]))
            elif has_well:
                if index >= 999:
                    raise ValueError("More than 999 fields in one well.")
                mapping[ident] = ("", index + 1)
            else:
                well_index, field_index = divmod(index, 999)
                if well_index >= WELLS_PER_PLATE:
                    raise ValueError("More than 384 x 999 sets on one plate.")
                mapping[ident] = (well_name(well_index), field_index + 1)
        fields_of[group_key] = mapping

    seen = {}
    for key, d in zip(keys, table):
        raw_plate = d.get("plateID", "")
        group_key = (raw_plate, d.get("wellID", "") if has_well else "")
        well_override, field_number = fields_of[group_key][identity(d)]
        well = (wells[(raw_plate, d.get("wellID", ""))] if has_well
                else well_override)
        name = SetName(plates[raw_plate], well, field_number,
                       times[d.get("timeID", "")])
        if name in seen:
            raise ValueError(
                f"Sets {set_label(seen[name])} and {set_label(key)} would "
                f"both be named {yokogawa_name(name, 1)}.")
        seen[name] = key
        names[key] = name
    return names


def yokogawa_name(name: SetName, channel: int, extension: str = ".tif") -> str:
    """The Yokogawa file name of one channel of one set.

    Built by :func:`spacr.convert.target_name`, so it is the exact form
    ``spacr.utils._get_regex('cellvoyager', ...)`` parses.

    :param name: the set's place.
    :param channel: 1-based channel.
    :param extension: the image's own extension, kept.
    """
    from .convert import target_name

    base = target_name(name.plate, name.well, name.field, channel, t=name.time)
    return base[:-len(".tif")] + extension.lower()




@dataclass
class PlanRow:
    """One image to move, with its mask and the mask's ledger.

    :ivar channel: 1-based channel.
    :ivar name: the set's place.
    :ivar set_key: the set the image is in.
    :ivar source_image: current image path.
    :ivar target_image: where it goes.
    :ivar source_mask: current mask path, or None.
    :ivar target_mask: where the mask goes, or None.
    :ivar source_ledger: the mask's ``.curation.json``, or None.
    :ivar target_ledger: where the ledger goes, or None.
    :ivar convert: ``"rgb"`` or ``"zstack"`` when the image is written as one
        converted 2-D plane (the original is kept under ``originals/``).
    """

    channel: int
    name: SetName
    set_key: SetKey
    source_image: str
    target_image: str
    source_mask: Optional[str] = None
    target_mask: Optional[str] = None
    source_ledger: Optional[str] = None
    target_ledger: Optional[str] = None
    convert: Optional[str] = None


@dataclass
class SortPlan:
    """Everything :func:`apply_plan` will do, decided before it does any.

    :ivar folder: the image folder.
    :ivar dest: the new folder the channels, stacks and ``merged/`` go in.
    :ivar rows: one per image.
    :ivar channels: channels, sorted.
    :ivar mask_roles: ``{channel: role}`` for channels whose masks are merged.
    :ivar problems: reasons the plan cannot be applied.
    :ivar warnings: things the user should know before it is.
    :ivar convertible: RGB images and z-stacks that :func:`build_plan` would
        convert when asked to (``convert=True``).
    """

    folder: str
    dest: str
    rows: List[PlanRow] = field(default_factory=list)
    channels: List[int] = field(default_factory=list)
    mask_roles: Dict[int, str] = field(default_factory=dict)
    convertible: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there is something to do and nothing stops it."""
        return bool(self.rows) and not self.problems

    @property
    def n_sets(self) -> int:
        """How many sets the plan holds."""
        return len({row.name for row in self.rows})

    def summary(self) -> str:
        """Plain lines saying what applying the plan will do."""
        masks = sum(1 for row in self.rows if row.source_mask)
        lines = [f"Move {len(self.rows)} image(s) and {masks} mask(s) into "
                 f"{len(self.channels)} channel folder(s) under {self.dest}, "
                 f"as {self.n_sets} set(s) in Yokogawa naming, and merge them "
                 f"into {os.path.join(self.dest, 'merged')}."]
        if self.mask_roles:
            lines.append("Mask planes: " + ", ".join(
                f"channel {c} -> {role}"
                for c, role in sorted(self.mask_roles.items())))
        lines.extend(self.warnings)
        lines.extend(self.problems)
        return "\n".join(lines)


def _word_role(names: Sequence[str]) -> str:
    """The mask role the names of a channel's images suggest, if any.

    :param names: the channel's file names; a path counts with its folder's
        name, so ``DAPI/f1.tif`` says nucleus.
    """
    text = " ".join(
        os.path.join(os.path.basename(os.path.dirname(n)), os.path.basename(n))
        for n in names).lower()
    for role, words in (("nucleus", ("nuc", "dapi", "hoechst", "h2b")),
                        ("pathogen", ("pathogen", "parasite", "bact", "virus")),
                        ("organelle", ("punct", "autophag", "lc3", "vesicle", "organelle")),
                        ("cell", ("cell", "cyto", "membrane", "actin", "cyst"))):
        if any(word in text for word in words):
            return role
    return ""


def default_mask_roles(members: Dict[int, List[str]],
                       with_masks: Iterable[int]) -> Dict[int, str]:
    """Guess which role each masked channel's masks have.

    A channel whose names say ``nuc``/``dapi``/``hoechst`` is ``nucleus``,
    ``cell``/``cyto`` is ``cell``, ``parasite``/``pathogen`` is
    ``pathogen``; the rest take the first unused of cell, nucleus, pathogen.

    :param members: ``{channel: [names]}``.
    :param with_masks: the channels that have masks.
    :returns: ``{channel: role}`` for those channels, roles distinct.
    """
    roles: Dict[int, str] = {}
    for channel in sorted(with_masks):
        guess = _word_role(members.get(channel, []))
        if guess and guess not in roles.values():
            roles[channel] = guess
    for channel in sorted(with_masks):
        if channel in roles:
            continue
        free = [r for r in MASK_ROLES if r not in roles.values()]
        if free:
            roles[channel] = free[0]
    return roles


def unused_folder(parent: str, name: str) -> str:
    """``parent/name``, or ``name_2``, ``name_3``... when that exists.

    :param parent: the folder to create in.
    :param name: the name wanted.
    """
    candidate = os.path.join(parent, name)
    number = 2
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{name}_{number}")
        number += 1
    return candidate


def _plan_mask(folder: str, image: str, masks_dir: Optional[str],
               masks: Optional[Dict[str, Optional[str]]]) -> Optional[str]:
    """The mask :func:`build_plan` moves with ``image``, or None.

    :param folder: the image folder.
    :param image: the image's name or path.
    :param masks_dir: an explicit masks folder.
    :param masks: explicit ``{image: mask}``, or None to look it up.
    """
    if masks is None:
        return mask_for(folder, image, masks_dir)
    mask = masks.get(image)
    return mask if mask and os.path.isfile(mask) else None


def _match_columns(columns: Sequence[Sequence[str]],
                   partners: Optional[Dict[int, int]] = None,
                   shapes: Optional[Dict[str, Optional[tuple]]] = None
                   ) -> List[List[Optional[str]]]:
    """Line files up in rows across columns, the way Detect sets pairs them.

    The Organize for Measure table: each column is a channel or a mask,
    each row one field. The column with the most files is the reference;
    its files, naturally sorted, start one row each. Every other column is matched to
    the rows one-to-one at the least total cost -- :func:`name_distance`
    plus a small term for how far apart two files sit in their sorted
    order -- and a pair of different image size is never made. A column in
    ``partners`` (a mask column) is matched against its partner column's
    file in each row (the image it outlines) rather than the reference.
    Files no row takes start rows of their own, so nothing dropped is lost.

    :param columns: the files of each column, absolute paths.
    :param partners: ``{column: column it is matched against}``.
    :param shapes: ``{path: shape}`` already read; read here otherwise.
    :returns: rows, each a list with one path or None per column.
    """
    columns = [sorted(dict.fromkeys(c), key=natural_key) for c in columns]
    partners = dict(partners or {})
    width = len(columns)
    if not width or not any(columns):
        return []
    shapes = dict(shapes or {})
    for files in columns:
        for path in files:
            if path not in shapes:
                shapes[path] = image_shape(path)

    def plane(path: Optional[str]):
        """A file's 2-D size, for comparing an image with a mask.

        :param path: a file, or None.
        """
        shape = shapes.get(path) if path else None
        return tuple(shape[:2]) if shape else None

    order = sorted(range(width), key=lambda c: (c in partners, -len(columns[c]), c))
    reference = order[0]
    rows: List[List[Optional[str]]] = []
    for path in columns[reference]:
        row: List[Optional[str]] = [None] * width
        row[reference] = path
        rows.append(row)
    forbidden = 1e6
    for column in order[1:]:
        files = columns[column]
        if not files:
            continue
        against = partners.get(column, reference)
        anchors = [row[against] for row in rows]
        cost = np.full((len(rows), len(files)), forbidden)
        for i, anchor in enumerate(anchors):
            if anchor is None or rows[i][column] is not None:
                continue
            for j, path in enumerate(files):
                if plane(anchor) is None or plane(anchor) != plane(path):
                    continue
                spread = abs(i / max(len(rows), 1) - j / max(len(files), 1))
                cost[i, j] = name_distance(anchor, path) + 0.01 * spread
        taken = set()
        if rows:
            for i, j in zip(*_assign(cost)):
                if cost[i, j] < forbidden:
                    rows[i][column] = files[j]
                    taken.add(j)
        for j, path in enumerate(files):
            if j not in taken:
                row = [None] * width
                row[column] = path
                rows.append(row)
    return rows


def build_plan(folder: str, sets: Dict[SetKey, Dict[int, str]], *,
               masks_dir: Optional[str] = None,
               mask_roles: Optional[Dict[int, str]] = None,
               dest: Optional[str] = None,
               check_shapes: bool = True,
               convert: bool = False,
               masks: Optional[Dict[str, Optional[str]]] = None) -> SortPlan:
    """Decide every move, name and check before anything is touched.

    :param folder: the image folder.
    :param sets: ``{set key: {channel: name}}``, complete sets only.
    :param masks_dir: an explicit masks folder.
    :param mask_roles: ``{channel: role}``; ``None`` guesses
        (:func:`default_mask_roles`); a channel missing from it, or given
        ``"none"``, has its masks moved but not merged.
    :param dest: the new folder; default ``<folder>/sorted_channels``.
    :param check_shapes: read each image's and mask's size and refuse a set
        whose images differ in size, a mask that is not its image's size or
        an image that is not 2-D.
    :param masks: ``{image: mask path}`` naming each image's mask outright
        (the organizer's mask columns); an image missing from it has none. None
        looks each mask up with :func:`mask_for`.
    :param convert: write RGB images as grey and z-stacks as their maximum
        projection instead of refusing them; without it they are listed in
        :attr:`SortPlan.convertible` so the caller can ask.
    :returns: a :class:`SortPlan`.
    """
    plan = SortPlan(folder=folder,
                    dest=dest or unused_folder(folder, DEFAULT_DEST_NAME))
    if not sets:
        plan.problems.append("There are no complete sets to sort.")
        return plan
    if os.path.exists(plan.dest):
        plan.problems.append(f"{plan.dest} already exists.")
    try:
        names = name_sets(sets)
    except ValueError as exc:
        plan.problems.append(str(exc))
        return plan
    plan.channels = sorted({c for members in sets.values() for c in members})
    members_by_channel: Dict[int, List[str]] = defaultdict(list)
    masked: set = set()
    for key, members in sets.items():
        for channel, image in members.items():
            members_by_channel[channel].append(image)
            if _plan_mask(folder, image, masks_dir, masks):
                masked.add(channel)
    if mask_roles is None:
        plan.mask_roles = default_mask_roles(members_by_channel, masked)
    else:
        plan.mask_roles = {int(c): r for c, r in mask_roles.items()
                           if r in MASK_ROLES and int(c) in plan.channels}
    doubled = [r for r, n in Counter(plan.mask_roles.values()).items() if n > 1]
    if doubled:
        plan.problems.append("Two channels cannot both give the "
                             + ", ".join(doubled) + " mask.")

    lacking: Dict[int, int] = Counter()
    for key in sorted(sets, key=lambda k: natural_key(set_label(k))):
        members = sets[key]
        place = names[key]
        set_shapes = {}
        for channel, image in sorted(members.items()):
            source = os.path.join(folder, image)
            ext = split_extension(image)[1] or ".tif"
            channel_dir = os.path.join(plan.dest, f"C{channel:02d}")
            target = os.path.join(channel_dir, yokogawa_name(place, channel, ext))
            row = PlanRow(channel, place, key, source, target)
            mask = _plan_mask(folder, image, masks_dir, masks)
            if mask:
                row.source_mask = mask
                row.target_mask = os.path.join(
                    channel_dir, "masks",
                    split_extension(os.path.basename(target))[0] + ".tif")
                ledger = mask + ".curation.json"
                if os.path.isfile(ledger):
                    row.source_ledger = ledger
                    row.target_ledger = row.target_mask + ".curation.json"
            elif channel in plan.mask_roles:
                lacking[channel] += 1
            if check_shapes:
                shape = image_shape(source)
                kind = _plane_conversion(shape)
                if kind:
                    plan.convertible.append(image)
                    if convert:
                        row.convert = kind
                        row.target_image = split_extension(target)[0] + ".tif"
                        shape = _converted_shape(shape, kind)
                set_shapes[image] = shape
                if shape is None:
                    plan.problems.append(f"{image} cannot be read.")
                elif len(shape) != 2:
                    plan.problems.append(
                        f"{image} is {shape}, not a single 2-D plane.")
                if mask and shape is not None:
                    mask_shape = image_shape(mask)
                    if mask_shape != shape:
                        plan.problems.append(
                            f"The mask of {image} is {mask_shape}; the image "
                            f"is {shape}.")
            plan.rows.append(row)
        if check_shapes and len({s for s in set_shapes.values()}) > 1:
            plan.problems.append(
                f"The images of set {set_label(key)} differ in size: "
                + ", ".join(f"{n} {s}" for n, s in set_shapes.items()))
    for channel, count in sorted(lacking.items()):
        plan.warnings.append(
            f"{count} set(s) have no mask in channel {channel} "
            f"({plan.mask_roles[channel]}); they get no merged array.")
    if not plan.mask_roles:
        plan.warnings.append("No channel has masks, so merged/ will not be "
                             "written; stack/ still will.")
    return plan


@dataclass
class ApplyResult:
    """What :func:`apply_plan` did.

    :ivar dest: the folder written.
    :ivar manifest: the move/rename record.
    :ivar moved: files moved (images, masks and ledgers).
    :ivar stacks: ``stack/*.npy`` written.
    :ivar merged: ``merged/*.npy`` written.
    """

    dest: str
    manifest: str
    moved: int = 0
    stacks: List[str] = field(default_factory=list)
    merged: List[str] = field(default_factory=list)


MANIFEST_COLUMNS = ("kind", "channel", "plate", "well", "field", "time",
                    "set", "original_path", "new_path", "status", "error")


def _write_converted(source: str, target: str, kind: str, dest: str) -> None:
    """Write ``source`` as one converted 2-D plane, then keep the original.

    The original is moved to ``<dest>/originals/`` rather than deleted, so a
    conversion can always be checked against what it came from.

    :param source: the RGB image or z-stack.
    :param target: the TIFF to write.
    :param kind: ``"rgb"`` or ``"zstack"``.
    :param dest: the sort's output folder.
    :raises OSError: when the image cannot be read or written.
    """
    from .tiff_io import write_tiff

    try:
        if source.lower().endswith((".tif", ".tiff")):
            import tifffile

            array = tifffile.imread(source)
        else:
            from PIL import Image

            with Image.open(source) as image:
                array = np.asarray(image)
    except Exception as exc:
        raise OSError(f"cannot read {source}: {exc}") from exc
    write_tiff(target, _convert_plane(np.squeeze(np.asarray(array)), kind))
    originals = os.path.join(dest, "originals")
    os.makedirs(originals, exist_ok=True)
    shutil.move(source, os.path.join(originals, os.path.basename(source)))


def apply_plan(plan: SortPlan, *, merge: bool = True,
               log: Optional[Callable[[str], None]] = None) -> ApplyResult:
    """MOVE the images and masks as planned, then merge.

    Every move is written to ``<dest>/channel_sorting_manifest.csv`` as it
    happens, so a run that stops half-way still says which file went where.
    Nothing is overwritten: every target is checked to be free before the
    first move.

    :param plan: an ``ok`` plan from :func:`build_plan`.
    :param merge: run :func:`merge_sorted` afterwards.
    :param log: called with progress lines; default prints.
    :returns: an :class:`ApplyResult`.
    :raises ValueError: when the plan has problems, a source is gone or a
        target is taken.
    :raises OSError: when a move fails; the manifest records it and every
        move before it.
    """
    say = log or (lambda text: print(text, flush=True))
    if not plan.ok:
        raise ValueError("The plan cannot be applied:\n" + "\n".join(
            plan.problems or ["it is empty"]))
    moves = []
    for row in plan.rows:
        for kind, source, target in (("image", row.source_image, row.target_image),
                                     ("mask", row.source_mask, row.target_mask),
                                     ("curation", row.source_ledger,
                                      row.target_ledger)):
            if source and target:
                moves.append((kind, row, source, target))
    missing = [source for _k, _r, source, _t in moves if not os.path.isfile(source)]
    if missing:
        raise ValueError("Missing before the move: " + _few(missing))
    if os.path.exists(plan.dest):
        raise ValueError(f"{plan.dest} already exists.")
    targets = [target for _k, _r, _s, target in moves]
    if len(set(targets)) != len(targets):
        raise ValueError("Two files would be moved to one name.")
    os.makedirs(plan.dest)
    result = ApplyResult(dest=plan.dest,
                         manifest=os.path.join(plan.dest, MANIFEST_NAME))
    with open(result.manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for index, (kind, row, source, target) in enumerate(moves):
            place = row.name
            base = [kind, row.channel, place.plate, place.well, place.field,
                    place.time, set_label(row.set_key), source, target]
            status = "moved"
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if kind == "image" and row.convert:
                    _write_converted(source, target, row.convert, plan.dest)
                    status = f"converted {row.convert}"
                else:
                    shutil.move(source, target)
            except OSError as exc:
                writer.writerow(base + ["error", str(exc)])
                handle.flush()
                raise
            writer.writerow(base + [status, ""])
            handle.flush()
            result.moved += 1
            if (index + 1) % 50 == 0:
                say(f"Moved {index + 1} of {len(moves)} files…")
    say(f"Moved {result.moved} file(s) into {plan.dest}; every move is in "
        f"{result.manifest}.")
    if merge:
        result.stacks, result.merged = merge_sorted(
            plan.dest, plan.mask_roles, log=say)
    return result


def merge_sorted(dest: str, mask_roles: Dict[int, str], *,
                 log: Optional[Callable[[str], None]] = None
                 ) -> Tuple[List[str], List[str]]:
    """Build ``stack/``, the mask stacks and ``merged/`` from sorted channels.

    The Yokogawa names in ``C01/``, ``C02/``... are read with spaCR's own
    ``cellvoyager`` pattern and :func:`spacr.utils._extract_filename_metadata`,
    each field's stem is :func:`spacr.io._escaped_field_stem` -- the stem the
    pipeline gives it -- and its channels are stacked in channel order into
    ``stack/<stem>.npy``. Each channel's masks are copied to
    ``masks/<role>_mask_stack/<stem>.tif``, and
    :func:`spacr.io._load_and_concatenate_arrays` writes ``merged/``.

    :param dest: the folder :func:`apply_plan` filled.
    :param mask_roles: ``{channel: role}``.
    :param log: called with progress lines.
    :returns: ``(stack files, merged files)``.
    """
    say = log or (lambda text: print(text, flush=True))
    from .io import (_escaped_field_stem, _load_and_concatenate_arrays,
                     _save_array_atomic, load_images_from_paths)
    from .utils import _extract_filename_metadata, _get_regex

    fields: Dict[str, Dict[int, str]] = defaultdict(dict)
    channel_dirs = sorted(d for d in os.listdir(dest)
                          if re.fullmatch(r"C\d{2,}", d)
                          and os.path.isdir(os.path.join(dest, d)))
    for channel_dir in channel_dirs:
        folder = os.path.join(dest, channel_dir)
        names = list_folder_images(folder)
        by_ext: Dict[str, List[str]] = defaultdict(list)
        for name in names:
            by_ext[split_extension(name)[1].lstrip(".")].append(name)
        for ext, group in by_ext.items():
            regex = re.compile(_get_regex("cellvoyager", ext))
            parsed = _extract_filename_metadata(group, folder, regex,
                                                "cellvoyager")
            for key, paths in parsed.items():
                stem = _escaped_field_stem(key[0], key[1], key[2], key[4])
                fields[stem][int(key[3])] = paths[0]

    stack_dir = os.path.join(dest, "stack")
    os.makedirs(stack_dir, exist_ok=True)
    stacks: List[str] = []
    for stem in sorted(fields, key=natural_key):
        by_channel = fields[stem]
        loaded = load_images_from_paths(
            {c: [path] for c, path in sorted(by_channel.items())})
        planes = [np.expand_dims(np.squeeze(images[0]), axis=2)
                  for _c, images in sorted(loaded.items()) if images]
        if len(planes) != len(by_channel):
            say(f"{stem}: a channel could not be read; no stack written.")
            continue
        path = os.path.join(stack_dir, stem + ".npy")
        _save_array_atomic(path, np.concatenate(planes, axis=2))
        stacks.append(path)
        for channel, image_path in by_channel.items():
            role = mask_roles.get(channel)
            if role not in MASK_ROLES:
                continue
            mask = os.path.join(os.path.dirname(image_path), "masks",
                                split_extension(os.path.basename(image_path))[0]
                                + ".tif")
            if os.path.isfile(mask):
                target = os.path.join(dest, "masks", f"{role}_mask_stack",
                                      stem + ".tif")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(mask, target)
    say(f"Wrote {len(stacks)} field stack(s) to {stack_dir}.")
    if not any(role in MASK_ROLES for role in mask_roles.values()):
        return stacks, []
    _load_and_concatenate_arrays(dest, None, None, None, None, None)
    merged_dir = os.path.join(dest, "merged")
    merged = sorted(os.path.join(merged_dir, n) for n in os.listdir(merged_dir)
                    if n.endswith(".npy")) if os.path.isdir(merged_dir) else []
    say(f"Wrote {len(merged)} merged array(s) to {merged_dir}.")
    return stacks, merged



#: The file :mod:`spacr.folder_consolidation` writes beside its copies.
CONSOLIDATION_MANIFEST = "rename_manifest.csv"


def _original_names(folder: str) -> Dict[str, str]:
    """The names a consolidated folder's files had before they were copied.

    :mod:`spacr.folder_consolidation` names each copy after its folders and
    numbers collisions, which can drop what told two channels apart -- a
    ``1.tif`` / ``1c.tif`` pair becomes ``exp_KO.tif`` / ``exp_KO_2.tif``.
    Its ``rename_manifest.csv`` still has the original paths; this gives
    each copy the original path below the common root, joined by ``_``
    (``Experiment 1_ATG2 KO_1c.tif``), so a regex can read the channel.

    :param folder: a folder that may hold ``rename_manifest.csv``.
    :returns: ``{current file name: original name}``; empty without a
        readable manifest.
    """
    path = os.path.join(folder, CONSOLIDATION_MANIFEST)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r.get("original_path") and r.get("new_filename")]
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}
    if not rows:
        return {}
    originals = [os.path.normpath(r["original_path"]) for r in rows]
    root = (os.path.commonpath([os.path.dirname(p) for p in originals])
            if len(originals) > 1 else os.path.dirname(originals[0]))
    names = {}
    for row, original in zip(rows, originals):
        relative = os.path.relpath(original, root)
        names[row["new_filename"]] = "_".join(
            part for part in relative.replace("\\", "/").split("/") if part)
    return names


def _diff_region(a: str, b: str):
    """The one place two file names differ, with what follows it.

    :param a: a file name.
    :param b: another.
    :returns: ``(a's text, b's text, text after)`` over the stems, or None
        when the names differ in more than one place or not at all.

    Several edits inside one word (DAPI and GFP share a P) form one region;
    edits separated by a separator form two, and then there is no marker. The
    region grows to whole letter runs, so DAPI/GFP is not read as DA/GF.
    """
    stem_a, stem_b = split_extension(a)[0], split_extension(b)[0]
    ops = [op for op in difflib.SequenceMatcher(
        None, stem_a, stem_b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    _tag, i1, _i2, j1, _j2 = ops[0]
    _tag, _i1, i2, _j1, j2 = ops[-1]
    if len(ops) > 1 and (re.search(r"[-_. ]", stem_a[i1:i2])
                         or re.search(r"[-_. ]", stem_b[j1:j2])):
        return None
    while (i1 > 0 and j1 > 0 and stem_a[i1 - 1].isalpha()
           and stem_a[i1 - 1] == stem_b[j1 - 1]
           and ((i1 < i2 and stem_a[i1].isalpha())
                or (j1 < j2 and stem_b[j1].isalpha()))):
        i1 -= 1
        j1 -= 1
    while (i2 < len(stem_a) and j2 < len(stem_b) and stem_a[i2].isalpha()
           and stem_a[i2] == stem_b[j2]
           and ((i1 < i2 and stem_a[i2 - 1].isalpha())
                or (j1 < j2 and stem_b[j2 - 1].isalpha()))):
        i2 += 1
        j2 += 1
    return stem_a[i1:i2], stem_b[j1:j2], stem_a[i2:]


def _teach_markers(answers: Dict[str, object]) -> Dict[object, set]:
    """What in the names marks each answered label.

    Each answered name is compared with the nearest answered name of
    another label; when they differ in one place, that text is a marker of
    its label (``""`` counts: a name with nothing where the other has
    ``c``).

    :param answers: ``{name: label}`` -- the user's answers.
    :returns: ``{label: {marker texts}}``.
    """
    markers: Dict[object, set] = defaultdict(set)
    names = list(answers)
    for name in names:
        others = sorted((n for n in names if answers[n] != answers[name]),
                        key=lambda n: (name_distance(name, n), natural_key(n)))
        for other in others:
            region = _diff_region(os.path.basename(name),
                                  os.path.basename(other))
            if region is not None:
                markers[answers[name]].add(region[0])
                break
    return dict(markers)


def _teach_regex(answers: Dict[str, object], extensions: Iterable[str]):
    """A regex whose ``chanID`` group reads the markers the answers show.

    The marker sits either at the end of the stem (``1c.tif``) or before
    fixed text; what comes before it (and after it) names the set.

    :param answers: ``{name: label}``.
    :param extensions: every extension among the names, with its dot.
    :returns: ``(regex, {marker: label})``, or ``(None, {})`` while the
        answers show no marker yet, or show one text for two labels.

    The pattern anchors on what sits just before the marker, so an empty marker
    cannot swallow an unknown one (7C is not 7 followed by nothing).
    """
    markers = _teach_markers(answers)
    by_marker: Dict[str, object] = {}
    for label, texts in markers.items():
        for text in texts:
            if by_marker.setdefault(text, label) != label:
                return None, {}
    if len(set(by_marker.values())) < 2:
        return None, {}
    after = set()
    before = set()
    names = list(answers)
    for name in names:
        for other in names:
            if answers[other] != answers[name]:
                region = _diff_region(os.path.basename(name),
                                      os.path.basename(other))
                if region is not None:
                    after.add(region[2])
                    stem = split_extension(os.path.basename(name))[0]
                    prefix = stem[:len(stem) - len(region[2]) - len(region[0])]
                    before.add(prefix[-1:])
    if before and all(c.isdigit() for c in before):
        edge = r"\d"
    elif before and all(c.isalpha() for c in before):
        edge = "[A-Za-z]"
    elif len(before) == 1 and before != {""}:
        edge = re.escape(next(iter(before)))
    else:
        edge = ""
    alternation = "|".join(re.escape(m) for m in
                           sorted(by_marker, key=lambda m: (-len(m), m)))
    exts = "|".join(sorted({re.escape(str(e).lstrip("."))
                            for e in extensions if e}))
    tail = r"\.(?:" + (exts or "[A-Za-z0-9]+") + ")$"
    head = r"(?P<wellID>.*?" + edge + ")(?P<chanID>" + alternation + ")"
    if not after or after == {""}:
        return head + tail, by_marker
    anchor = sorted(after, key=len)[0][:1]
    return (head + r"(?P<fieldID>" + re.escape(anchor) + ".*?)" + tail,
            by_marker)


def _teach_step(names: Sequence[str], answers: Dict[str, object]):
    """One round of "Teach me": the regex so far, and which image to ask about.

    Without a regex yet, the next image is the unanswered one most like an
    answered one -- most often the same field in another channel. With a
    regex, it is the first name the regex does not read, or the first
    unanswered one when the regex reads an answered name as another label;
    None means every name is placed.

    :param names: every image name (as the regex will read them).
    :param answers: ``{name: label}`` so far; a label is anything hashable,
        e.g. ``("channel", 1)`` or ``("mask", 1, "cell")``, or ``"skip"``
        for an image the user does not want placed.
    :returns: ``(regex or None, {marker: label}, next name or None)``.

    An answer the pattern cannot explain needs its neighbour: the same field in
    another channel shows what marks it.
    """
    names = list(names)
    placed = {n: label for n, label in answers.items() if label != "skip"}
    unanswered = [n for n in names if n not in answers]
    if not placed:
        return None, {}, (unanswered[0] if unanswered else None)
    extensions = {split_extension(os.path.basename(n))[1] for n in names}
    pattern, by_marker = _teach_regex(placed, extensions)
    if pattern is None:
        if not unanswered:
            return None, {}, None
        nearest = min(unanswered, key=lambda n: (
            min(name_distance(n, a) for a in placed), natural_key(n)))
        return None, {}, nearest
    compiled = re.compile(pattern)
    unread: List[str] = []
    for name in names:
        if answers.get(name) == "skip":
            continue
        match = compiled.match(os.path.basename(name))
        if match is None:
            if name not in answers:
                return pattern, by_marker, name
            unread.append(name)
            continue
        if name in placed and by_marker.get(match.group(CHANNEL_GROUP)) \
                != placed[name]:
            unread.append(name)
    if unread and unanswered:
        return pattern, by_marker, min(unanswered, key=lambda n: (
            name_distance(n, unread[0]), natural_key(n)))
    return pattern, by_marker, None


def _teach_labels(names: Sequence[str], pattern: str,
                  by_marker: Dict[str, object]) -> Dict[str, object]:
    """Each name's label, as the learned regex reads it.

    :param names: image names.
    :param pattern: a regex from :func:`_teach_step`.
    :param by_marker: ``{marker: label}`` from :func:`_teach_step`.
    :returns: ``{name: label}`` for the names the regex reads.
    """
    compiled = re.compile(pattern)
    labels = {}
    for name in names:
        match = compiled.match(os.path.basename(name))
        if match is not None and match.group(CHANNEL_GROUP) in by_marker:
            labels[name] = by_marker[match.group(CHANNEL_GROUP)]
    return labels
