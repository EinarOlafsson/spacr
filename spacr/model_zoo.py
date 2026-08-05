"""Browse, verify, fetch and bench the models spaCR can segment or classify with.

Why this exists
---------------
spaCR can run a stock Cellpose model, a Cellpose model somebody on the team
fine-tuned last year, a classifier checkpoint from a run three folders up, or a
checkpoint downloaded from Hugging Face. Today the only way to find out which
of those exist on a machine is ``find / -name '*.pth'``, the only way to know
what one of them was trained on is to remember, and the only way to know
whether the copy on disk is the file the author published is to hope.

This module answers those three questions and nothing else:

* **what is here** — :func:`discover_local` walks folders and returns the
  checkpoints, classified as Cellpose or classifier, with whatever provenance
  is recoverable from the settings snapshots spaCR already writes;
* **is it the right bytes** — :func:`sha256_file` / :func:`verify` /
  :func:`fetch`, which downloads atomically, checksums what arrived, and
  refuses to install a mismatch;
* **what does it do on my data** — :func:`benchmark`, which is
  :mod:`spacr.model_compare`'s "three fields" harness pointed at one model
  instead of two.

Nothing here imports torch or cellpose at module import time. Browsing the zoo,
reading provenance, checksumming a file and rendering the table all work on a
machine with neither installed; only :func:`benchmark` (through
:func:`spacr.model_compare.segment_with_cellpose`) needs them, and only when it
is called. ``tests/test_model_zoo.py`` asserts that.

The four things that make this trustworthy rather than merely convenient
-----------------------------------------------------------------------
**Every download is checksummed, and verification happens before use.**
    A checkpoint truncated by a dropped connection, or swapped for a different
    one at the same URL, still loads. It does not raise; it produces silently
    different masks, and the run that used it looks exactly like the run that
    did not. So :func:`fetch` hashes what arrived and compares it to the hash
    the catalogue published: a mismatch deletes the file and raises
    :class:`ChecksumMismatch` — the entry is never registered. The hash that
    was actually computed is stored on the returned entry, so "verified"
    means "these bytes", not "this filename".

    A catalogue entry with no published hash cannot be verified at all, and
    that is refused by default (``require_checksum=True``) rather than quietly
    treated as fine. Callers that knowingly accept an unverifiable source pass
    ``require_checksum=False``; the resulting entry carries
    ``verified=False`` and says so in :func:`format_zoo`.

**Every download is atomic, and never overwrites.**
    Bytes stream into a temporary file *in the destination directory* (so the
    final ``os.replace`` is a same-filesystem rename, which is atomic) and the
    rename happens only after the checksum passes. An interrupted or cancelled
    download therefore leaves nothing behind that looks like a model — the
    failure mode where half a checkpoint sits at the real filename, loads, and
    segments badly, cannot happen.

    The destination is versioned rather than overwritten
    (:func:`versioned_path`: ``foo.CP_model``, ``foo_v2.CP_model``, …). Two
    models with the same filename are a normal thing to have; losing the first
    one to the second is not.

**Provenance is recorded, and "unknown" is written out in full.**
    A Cellpose model fine-tuned on 60x confluent HeLa is not interchangeable
    with one trained on 20x sparse fibroblasts, and no amount of benchmark
    score makes it so. What a model was trained on is the single most useful
    thing the zoo can show, so :class:`ModelEntry` carries
    :attr:`~ModelEntry.trained_on` and :attr:`~ModelEntry.trained_by`, both
    recovered from the settings snapshots spaCR already writes beside its
    models (``<file>_settings.csv`` for a Cellpose model,
    ``<dst>/settings.csv`` for a classifier — read through
    :func:`spacr.train_compare.load_run`, which already knows where to look).

    Where it could not be recovered the field reads ``'unknown'``, never ``''``.
    A blank cell in a provenance table reads as "no constraints"; that is the
    opposite of what it means.

**Benchmarks are only comparable inside one field set.**
    A model's score on your three fields says nothing whatsoever about its
    score on somebody else's, and a table that sorts the two together invents a
    ranking out of two unrelated numbers. So every :class:`BenchmarkResult`
    records a :func:`fieldset_id` — a hash of the actual pixels, not the folder
    name — and :func:`rank` **raises** :class:`IncomparableBenchmarks` when
    handed results from more than one field set. :func:`rank_groups` and
    :func:`format_benchmarks` are the supported alternative: they group by
    field set, rank within each group, and label the groups.

And a fifth, smaller one: a model file that is missing, empty, or not a torch
checkpoint at all fails in :func:`inspect_checkpoint` with a message naming the
file, before anything tries to load it. The default failure — a ``KeyError`` on
a state-dict key from deep inside torch — names nothing the user chose.

What the benchmark can and cannot say
-------------------------------------
There is no ground truth here. :func:`benchmark` runs one model over N fields
and reports what came out: object counts, timings and the
:mod:`spacr.seg_qc` verdict per field (fused? shattered? empty? all on the
border?). That is a *quality-control* score, not an accuracy — it can tell you
a model collapsed on your data, it cannot tell you which of two plausible
segmentations is right. :data:`RANK_KEYS` therefore offers exactly two keys,
``'qc'`` and ``'seconds'``, and no key that would read as accuracy. To compare
two models against each other, use :func:`compare_entries`, which hands both to
:func:`spacr.model_compare.compare_models` — the A/B harness that is explicit
about neither side being the truth.

Cellpose 4 accepts and ignores ``model_type``, ``diam_mean``, ``nchan``,
``channels`` and ``rescale``; only ``diameter`` at ``eval`` still changes the
masks. :class:`BenchmarkResult` carries the resolved ``honoured`` and
``ignored`` parameter dicts straight from
:class:`spacr.model_compare.ModelConfig` so a benchmark cannot silently be a
benchmark of settings nothing read.

Example::

    from spacr import model_zoo as zoo

    entries = zoo.catalogue() + zoo.discover_local('/data/screen1')
    print(zoo.format_zoo(entries))

    entry = zoo.resolve('toxo_plaque_cyto_e25000_X1120_Y1120.CP_model', entries)
    result = zoo.benchmark(entry, source='/data/screen1/plate1/1', n_fields=3)
    print(zoo.format_benchmarks([result]))

See Also:
    :mod:`spacr.model_compare` — the A/B harness this module reuses for
    segmentation and for the two-model comparison.
    :mod:`spacr.train_compare` — run discovery and settings recovery, reused
    wholesale for the classifier half of the zoo.
    :func:`spacr.utils.download_models` — the legacy bulk pull of the bundled
    Hugging Face model pack, wrapped by :func:`download_bundled_models`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field as _dc_field, replace
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence,
    Tuple,
)

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "BUNDLED_REMOTE_MODELS",
    "BenchmarkResult",
    "CATALOGUE_ENV_VAR",
    "CLASSIFIER_SUFFIXES",
    "CELLPOSE_SUFFIXES",
    "ChecksumMismatch",
    "DEFAULT_N_FIELDS",
    "DEFAULT_SCAN_DEPTH",
    "DownloadCancelled",
    "FieldBenchmark",
    "HF_MODELS_REPO",
    "IncomparableBenchmarks",
    "KINDS",
    "ModelEntry",
    "ModelUnreadable",
    "ModelZooError",
    "RANK_KEYS",
    "UNKNOWN",
    "benchmark",
    "catalogue",
    "classify_kind",
    "compare_entries",
    "default_local_roots",
    "discover_local",
    "download_bundled_models",
    "entry_from_file",
    "fetch",
    "fieldset_id",
    "format_benchmarks",
    "format_zoo",
    "group_by_fieldset",
    "hf_uri",
    "inspect_checkpoint",
    "install",
    "load_catalogue_file",
    "open_uri",
    "package_model_root",
    "rank",
    "rank_groups",
    "resolve",
    "sha256_file",
    "verify",
    "versioned_path",
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: What an unrecoverable provenance field says. Never ``''``: a blank cell in a
#: provenance table reads as "no constraints", which is the opposite of "we do
#: not know what this model was trained on".
UNKNOWN = "unknown"

#: The two kinds of model spaCR runs.
KINDS = ("cellpose", "classifier")

#: Filename endings that mark a Cellpose checkpoint. ``.CP_model`` is what
#: :func:`spacr.submodules.train_cellpose` names its output.
CELLPOSE_SUFFIXES = (".cp_model", ".cpmodel")

#: Filename endings that mark a torch checkpoint. A ``.pth`` inside a Cellpose
#: folder is still a Cellpose model — see :func:`classify_kind`.
CLASSIFIER_SUFFIXES = (".pth", ".pt")

#: Directory names that mean "the files in here are Cellpose checkpoints".
#: ``cellpose_model`` is :func:`spacr.submodules.train_cellpose`'s output
#: folder; ``cp`` is where :func:`spacr.utils.download_models` lands the
#: bundled pack; ``models`` is what ``cellpose.train.train_seg`` creates under
#: whatever ``save_path`` it is given.
CELLPOSE_DIR_NAMES = ("cellpose_model", "cp", "models")

#: How deep :func:`discover_local` walks below each root.
DEFAULT_SCAN_DEPTH = 6

#: Ceiling on how many files one :func:`discover_local` call will *examine*, so
#: pointing it at ``/`` is slow rather than fatal.
DEFAULT_SCAN_LIMIT = 20000

#: Fields a benchmark uses by default — the number a human actually looks at.
DEFAULT_N_FIELDS = 3

#: Seconds before a download gives up on the server.
DEFAULT_TIMEOUT = 30

#: Bytes per chunk while streaming a download.
DEFAULT_CHUNK = 1 << 16

#: The Hugging Face dataset repo :func:`spacr.utils.download_models` pulls the
#: bundled model pack from. Same repo, same URL scheme — this module adds the
#: checksum, the atomic write and the versioned destination that one lacks.
HF_MODELS_REPO = "einarolafsson/models"

#: Environment variable naming a JSON catalogue of remote models. See
#: :func:`load_catalogue_file` for the format.
CATALOGUE_ENV_VAR = "SPACR_MODEL_CATALOGUE"

#: Remote entries spaCR knows about out of the box.
#:
#: ``sha256`` is empty because this pack publishes no checksum, and an empty
#: hash here is a *statement*, not an oversight: :func:`fetch` refuses to
#: install an entry it cannot verify unless the caller passes
#: ``require_checksum=False``. The honest fix is a catalogue file
#: (:func:`load_catalogue_file`) carrying hashes for the copies your lab
#: actually blessed.
BUNDLED_REMOTE_MODELS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "toxo_plaque_cyto",
        "name": "toxo_plaque_cyto_e25000_X1120_Y1120.CP_model",
        "kind": "cellpose",
        "uri": None,        # filled in from HF_MODELS_REPO below
        "sha256": "",
        "trained_on": (
            "Toxoplasma plaque assay, /nas_mnt/carruthers/patrick/"
            "Plaque_assay_training/train — 1120x1120 crops, diameter 30, "
            "25000 epochs, greyscale"
        ),
        "trained_by": "einarolafsson (spaCR bundled model pack)",
        "notes": (
            "publishes no checksum; fetch refuses it unless you pass "
            "require_checksum=False or supply expected_sha256=",
        ),
    },
)

#: Keys :func:`rank` will sort on, with the direction and what the number is.
#:
#: There is deliberately no accuracy key. A benchmark here has no ground truth
#: (see the module docstring), so a column called "score" that sorts models
#: would be inventing one.
RANK_KEYS: Dict[str, str] = {
    "qc": "fraction of fields spacr.seg_qc scored 'ok' — a quality-control "
          "verdict on this model's own masks, not an accuracy (higher first)",
    "seconds": "wall-clock segmentation time over the field set (lower first)",
}

#: :func:`rank`'s default key.
DEFAULT_RANK_KEY = "qc"

#: First bytes of a torch checkpoint. Everything torch has saved since 1.6 is a
#: zip; ``\\x80`` opens the legacy pickle protocol.
_TORCH_MAGICS = (b"PK\x03\x04", b"\x80")

#: ``name_v3`` -> ``('name', 3)``. See :func:`versioned_path`.
_VERSION_RE = re.compile(r"^(?P<base>.+)_v(?P<n>\d+)$")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class ModelZooError(Exception):
    """Base class for every refusal in this module."""


class ChecksumMismatch(ModelZooError):
    """What arrived is not what the catalogue published. Nothing was installed."""


class ModelUnreadable(ModelZooError):
    """A model file is missing, empty, or not a checkpoint. Names the file."""


class DownloadCancelled(ModelZooError):
    """The caller cancelled a fetch. Nothing was left at the destination."""


class IncomparableBenchmarks(ModelZooError):
    """Benchmarks from different field sets cannot be ranked against each other."""


# ---------------------------------------------------------------------------
# the entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """One model the zoo knows about, wherever it lives.

    Frozen because an entry is a *record of a file at a moment* — the hash, the
    size and the provenance describe those bytes. Changing one in place would
    silently invalidate the other two; :func:`dataclasses.replace` makes the
    new record explicit.

    :param key: stable id, unique within a listing. For a local file this is
        derived from the filename; for a catalogue entry it is whatever the
        catalogue declared.
    :param name: the filename (or the published name) — what a human reads.
    :param kind: ``'cellpose'`` or ``'classifier'``; see :data:`KINDS`.
    :param source: ``'bundled'`` (ships with spaCR), ``'local'`` (found on this
        machine) or ``'remote'`` (declared in a catalogue, not yet fetched).
    :param path: absolute path on this machine, or ``''`` for a remote entry.
    :param uri: where a remote entry is fetched from, or ``''``.
    :param version: the zoo's own version number for a filename. ``'1'`` for a
        plain name, ``'2'`` for ``foo_v2.CP_model`` (see
        :func:`versioned_path`), or whatever a catalogue declared.
    :param sha256: hex digest. For a downloaded model this is the digest of the
        bytes that were actually written; for a catalogue entry it is the
        published digest to check against; ``''`` means "no checksum known",
        which :func:`fetch` treats as a refusal rather than a pass.
    :param size_bytes: file size, ``0`` when unknown.
    :param trained_on: what data produced this model, in prose, or
        :data:`UNKNOWN`. Never ``''``.
    :param trained_by: who produced it, or :data:`UNKNOWN`. Never ``''``.
    :param metrics: whatever numbers came with it — for a classifier, the
        best/last epoch metrics :func:`spacr.train_compare.load_run` recovered.
        Excluded from equality: two records of the same bytes are the same
        model whether or not somebody attached numbers to one of them.
    :param notes: everything the reader needs to know that is not a field:
        missing provenance, an unverified download, a file that does not look
        like a checkpoint.
    :param verified: True only when :attr:`sha256` was checked against a
        published digest. A downloaded file whose hash was merely *recorded* is
        not verified, and says so.
    :param settings_path: where the provenance came from, for the reader who
        wants to go and look at it.
    """

    key: str
    name: str
    kind: str = "cellpose"
    source: str = "local"
    path: str = ""
    uri: str = ""
    version: str = "1"
    sha256: str = ""
    size_bytes: int = 0
    trained_on: str = UNKNOWN
    trained_by: str = UNKNOWN
    metrics: Dict[str, Any] = _dc_field(default_factory=dict, compare=False)
    notes: Tuple[str, ...] = ()
    verified: bool = False
    settings_path: str = ""

    def __post_init__(self):
        # A blank provenance field reads as "no constraints"; it has to say
        # "unknown" out loud instead. object.__setattr__ because frozen.
        for attribute in ("trained_on", "trained_by"):
            value = str(getattr(self, attribute) or "").strip()
            object.__setattr__(self, attribute, value or UNKNOWN)
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def exists(self) -> bool:
        """True when :attr:`path` names a file that is here now."""
        return bool(self.path) and os.path.isfile(self.path)

    @property
    def provenance_known(self) -> bool:
        """True when this model says what it was trained on."""
        return self.trained_on != UNKNOWN

    @property
    def checksum_state(self) -> str:
        """What the checksum column says, in one word.

        ``'none'``
            no hash at all — nothing can be checked, and :func:`fetch` refuses
            such an entry unless the caller overrides it.
        ``'published'``
            a hash came with the entry but the bytes are not here yet, so it is
            a promise about what will arrive.
        ``'recorded'``
            the honest middle: the hash of the file on disk is known, but
            nobody published one to compare it against. It proves the file has
            not changed *since we looked*, and nothing more.
        ``'verified'``
            the bytes on disk were compared with a published digest and match.
        """
        if not self.sha256:
            return "none"
        if self.verified:
            return "verified"
        return "recorded" if self.exists else "published"

    def summary_line(self) -> str:
        """One line for a list widget."""
        bits = [self.name, self.kind, self.source, f"v{self.version}"]
        bits.append(f"trained on: {_shorten(self.trained_on, 48)}")
        bits.append(f"checksum {self.checksum_state}")
        if self.notes:
            bits.append(f"! {len(self.notes)} note"
                        f"{'s' if len(self.notes) > 1 else ''}")
        return " · ".join(bits)

    def describe(self) -> str:
        """The multi-line provenance card shown next to a selected model."""
        lines = [
            f"{self.name}  [{self.kind} · {self.source} · v{self.version}]",
            f"  path       {self.path or '(not downloaded)'}",
        ]
        if self.uri:
            lines.append(f"  uri        {self.uri}")
        lines.append(f"  size       {_human_bytes(self.size_bytes)}")
        lines.append(f"  sha256     {self.sha256 or '(none published)'} "
                     f"({self.checksum_state})")
        lines.append(f"  trained on {self.trained_on}")
        lines.append(f"  trained by {self.trained_by}")
        if self.settings_path:
            lines.append(f"  provenance {self.settings_path}")
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name:<10} {value}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        if not self.provenance_known:
            lines.append(
                "  ! this model does not say what it was trained on, so "
                "nothing here tells you whether it suits your images.")
        return "\n".join(lines)


def _shorten(text: Any, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + "…"


def _human_bytes(size: Any) -> str:
    """Bytes as something a person reads, ``'unknown'`` for 0/None."""
    try:
        n = float(size)
    except (TypeError, ValueError):
        return UNKNOWN
    if n <= 0:
        return UNKNOWN
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------

def sha256_file(path: Any, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks so a 2 GB checkpoint is not RAM.

    :param path: the file.
    :param chunk_size: bytes per read.
    :returns: the lowercase hex digest.
    :raises ModelUnreadable: when the file is missing or cannot be read, with
        the path in the message.
    """
    p = Path(path)
    digest = hashlib.sha256()
    try:
        with p.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except FileNotFoundError:
        raise ModelUnreadable(f"no such model file: {p}") from None
    except OSError as e:
        raise ModelUnreadable(f"could not read {p}: {e}") from None
    return digest.hexdigest()


def verify(entry: ModelEntry, expected: Optional[str] = None) -> bool:
    """Hash the file this entry points at and compare it to a known digest.

    :param entry: the entry to check.
    :param expected: the digest to compare against; defaults to
        :attr:`ModelEntry.sha256`.
    :returns: True when the file's digest matches.
    :raises ModelUnreadable: when the entry has no local file, naming it.
    :raises ModelZooError: when there is no digest to compare against — that is
        a caller error, and returning False for it would read as "the file is
        wrong" when what happened is "nobody said what right looks like".
    """
    if not entry.path:
        raise ModelUnreadable(
            f"{entry.name} has not been downloaded, so there is nothing to "
            f"verify (source={entry.source}, uri={entry.uri or 'none'})")
    if not os.path.isfile(entry.path):
        raise ModelUnreadable(f"no such model file: {entry.path}")
    want = (expected if expected is not None else entry.sha256) or ""
    want = want.strip().lower()
    if not want:
        raise ModelZooError(
            f"no checksum recorded for {entry.name} ({entry.path}) — there is "
            f"nothing to verify it against. Compute one with sha256_file() and "
            f"put it in the catalogue.")
    return sha256_file(entry.path) == want


# ---------------------------------------------------------------------------
# recognising a model file
# ---------------------------------------------------------------------------

def _looks_like_checkpoint(path: Path) -> bool:
    """True when the first bytes are a torch save (zip or legacy pickle)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in _TORCH_MAGICS)


def classify_kind(path: Any) -> Optional[str]:
    """Say whether a file is a Cellpose model, a classifier, or not a model.

    The rules, in order:

    1. ``*.CP_model`` is a Cellpose checkpoint — that is what
       :func:`spacr.submodules.train_cellpose` names its output.
    2. ``*.pth`` / ``*.pt`` is a classifier checkpoint
       (:func:`spacr.io._save_model` writes
       ``<model_type>_epoch_<n>_channels_<ch>.pth``) **unless** it sits in a
       Cellpose folder or has ``cellpose``/``cp_model`` in its name.
    3. An **extensionless** file inside a Cellpose folder is a Cellpose
       checkpoint only if its first bytes are a torch save. ``cellpose.train``
       writes ``<save_path>/models/<name>`` with no suffix, and that folder
       also holds READMEs and logs — the magic-byte check is what keeps a
       ``README`` out of the zoo.
    4. Anything else is not a model. CSVs, PNGs, ``.npy`` masks and settings
       snapshots all land here and are ignored.

    :param path: a file path.
    :returns: ``'cellpose'``, ``'classifier'`` or None.
    """
    p = Path(path)
    low = p.name.lower()
    # Only the two nearest folders, not every ancestor: a classifier under
    # ``/data/models/screen1/run/`` is still a classifier, and matching any
    # ancestor called "models" would silently relabel a whole tree.
    near = {p.parent.name.lower(), p.parent.parent.name.lower()}
    in_cellpose_dir = bool(near & set(CELLPOSE_DIR_NAMES))

    if low.endswith(CELLPOSE_SUFFIXES):
        return "cellpose"
    if low.endswith(CLASSIFIER_SUFFIXES):
        if "cellpose" in low or "cp_model" in low or in_cellpose_dir:
            return "cellpose"
        return "classifier"
    if not p.suffix and in_cellpose_dir:
        return "cellpose" if _looks_like_checkpoint(p) else None
    return None


def inspect_checkpoint(path: Any, loader: Optional[Callable[[str], Any]] = None,
                       deep: bool = False) -> Dict[str, Any]:
    """Check a file is a loadable checkpoint, failing with the filename in it.

    The default failure for a wrong or corrupt checkpoint is a ``KeyError`` on
    a state-dict key raised somewhere inside torch, which names nothing the
    user chose and reads like a spaCR bug. This turns all of it —  missing,
    empty, truncated, a PNG somebody renamed, a Cellpose model handed to the
    classifier path — into one :class:`ModelUnreadable` naming the file.

    The shallow check needs no torch at all: it is a stat and four bytes.

    :param path: the checkpoint.
    :param loader: ``fn(path) -> object`` used for the deep check; defaults to
        ``torch.load(..., map_location='cpu')``, imported only if used.
    :param deep: actually load the file. Off by default because loading a 2 GB
        checkpoint to populate a list widget is not acceptable.
    :returns: ``{'path', 'size_bytes', 'format', 'loaded'}``.
    :raises ModelUnreadable: naming the file, always.
    """
    p = Path(path)
    if not p.exists():
        raise ModelUnreadable(f"no such model file: {p}")
    if p.is_dir():
        raise ModelUnreadable(
            f"{p} is a directory, not a model checkpoint — point at the file "
            f"inside it")
    size = p.stat().st_size
    if size == 0:
        raise ModelUnreadable(
            f"{p} is empty (0 bytes) — an interrupted download or copy leaves "
            f"exactly this")
    try:
        with p.open("rb") as handle:
            head = handle.read(4)
    except OSError as e:
        raise ModelUnreadable(f"could not read {p}: {e}") from None
    if not any(head.startswith(magic) for magic in _TORCH_MAGICS):
        raise ModelUnreadable(
            f"{p} is not a PyTorch checkpoint: it starts with {head!r}, and a "
            f".pth / .CP_model file starts with a zip header or a pickle "
            f"opcode. Check the path — this is usually a text file, an HTML "
            f"error page saved by a failed download, or the wrong file "
            f"entirely.")
    fmt = "zip" if head.startswith(b"PK\x03\x04") else "pickle"

    out: Dict[str, Any] = {"path": str(p), "size_bytes": int(size),
                           "format": fmt, "loaded": False}
    if not deep:
        return out
    load = loader if loader is not None else _torch_loader
    try:
        load(str(p))
    except ModelUnreadable:
        raise
    except Exception as e:
        raise ModelUnreadable(
            f"{p} could not be loaded as a model checkpoint "
            f"({type(e).__name__}: {e}). The file is a torch save but not the "
            f"architecture that was asked for — check that this is a "
            f"{'Cellpose' if p.name.lower().endswith(CELLPOSE_SUFFIXES) else 'classifier'} "
            f"model and not the other kind.") from None
    out["loaded"] = True
    return out


def _torch_loader(path: str) -> Any:
    """``torch.load`` on the CPU. Imported here so the module stays torch-free."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _read_key_value_csv(path: Path) -> Dict[str, Any]:
    """Read a ``Key,Value`` settings CSV the way the run diff reads one.

    Reuses :func:`spacr.run_journal._read_settings_csv` rather than writing a
    second parser: that one already handles the JSON/CSV/live-dict round trips
    a spaCR settings dict takes, and the zoo has to agree with the run diff
    about what a settings file says.
    """
    from .run_journal import _read_settings_csv

    return _read_settings_csv(path)


def _settings_beside(path: Path) -> Tuple[Dict[str, Any], str]:
    """Find the settings snapshot beside a Cellpose checkpoint.

    :func:`spacr.submodules.train_cellpose` calls ``save_settings(settings,
    name=model_name)``, which writes ``<src>/settings/<model_name>.csv``; the
    bundled pack ships ``<model_file>_settings.csv`` next to the weights. Both
    are looked for, nearest first.

    :returns: ``(settings, where)``; ``({}, '')`` when there is none.
    """
    candidates = [
        path.with_name(path.name + "_settings.csv"),
        path.with_name(path.stem + "_settings.csv"),
        path.with_suffix(".csv"),
    ]
    node = path.parent
    for _ in range(DEFAULT_SCAN_DEPTH):
        candidates.append(node / "settings" / f"{path.name}.csv")
        candidates.append(node / "settings" / f"{path.stem}.csv")
        node = node.parent
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            settings = _read_key_value_csv(candidate)
        except Exception:
            continue
        if settings:
            return settings, str(candidate)
    return {}, ""


def _describe_cellpose_training(settings: Mapping[str, Any]) -> str:
    """One prose line saying what a Cellpose model saw, or :data:`UNKNOWN`.

    Deliberately concrete: magnification and confluence are not in the settings
    file, but the source folder, the crop size, the diameter and the epoch
    count are, and together they are enough for a reader to tell "this is not
    my data".
    """
    source = settings.get("img_src") or settings.get("src") or ""
    bits: List[str] = []
    if source:
        bits.append(str(source))
    shape = settings.get("width_height") or settings.get("target_size")
    if shape:
        bits.append(f"crops {shape}")
    if settings.get("diameter"):
        bits.append(f"diameter {settings['diameter']}")
    if settings.get("n_epochs"):
        bits.append(f"{settings['n_epochs']} epochs")
    if settings.get("grayscale") is True or str(settings.get("grayscale")).lower() == "true":
        bits.append("greyscale")
    return ", ".join(bits) if bits else UNKNOWN


def _describe_classifier_training(settings: Mapping[str, Any]) -> str:
    """One prose line saying what a classifier saw, or :data:`UNKNOWN`."""
    bits: List[str] = []
    source = settings.get("src") or ""
    if source:
        bits.append(str(source))
    if settings.get("model_type"):
        bits.append(str(settings["model_type"]))
    if settings.get("classes"):
        bits.append(f"classes {settings['classes']}")
    if settings.get("image_size"):
        bits.append(f"{settings['image_size']}px crops")
    if settings.get("epochs"):
        bits.append(f"{settings['epochs']} epochs")
    return ", ".join(bits) if bits else UNKNOWN


def _who(settings: Mapping[str, Any]) -> str:
    """Who trained it, from whatever the settings recorded, or :data:`UNKNOWN`."""
    for key in ("user", "author", "trained_by", "operator", "hostname", "host"):
        value = settings.get(key)
        if value not in (None, "", "nan"):
            return str(value)
    return UNKNOWN


# ---------------------------------------------------------------------------
# building entries from files
# ---------------------------------------------------------------------------

def _version_of(name: str) -> str:
    """The zoo's version number for a filename (see :func:`versioned_path`)."""
    stem = Path(name).stem
    match = _VERSION_RE.match(stem)
    return match.group("n") if match else "1"


def _key_for(path: Path) -> str:
    """A short, stable key for a local file: its name, made filesystem-safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)


def entry_from_file(path: Any, kind: Optional[str] = None,
                    source: str = "local", key: Optional[str] = None,
                    runs: Optional[Mapping[str, Any]] = None,
                    compute_hash: bool = False,
                    sha256: str = "", verified: bool = False,
                    extra_notes: Sequence[str] = ()) -> ModelEntry:
    """Build a :class:`ModelEntry` for a checkpoint on this machine.

    Provenance is recovered from whatever spaCR already wrote beside the model:
    a ``*_settings.csv`` or ``<src>/settings/<name>.csv`` for a Cellpose model,
    and — for a classifier — the training run the checkpoint sits in, loaded
    through :func:`spacr.train_compare.load_run` so the zoo and the training-run
    comparison agree about where settings live and what they say.

    :param path: the checkpoint.
    :param kind: override :func:`classify_kind`.
    :param source: ``'local'`` or ``'bundled'``.
    :param key: override the generated key.
    :param runs: ``{folder: TrainingRun}`` from :func:`_runs_under`, so a scan
        of 40 checkpoints in one run folder reads that folder once.
    :param compute_hash: hash the file now. Off by default: hashing every
        checkpoint on a machine to populate a list widget is minutes.
    :param sha256: a digest already known for these bytes.
    :param verified: whether ``sha256`` was checked against a published digest.
    :param extra_notes: notes to carry onto the entry.
    :returns: the entry.
    :raises ModelUnreadable: when the path is not a file.
    """
    p = Path(path)
    if not p.is_file():
        raise ModelUnreadable(f"no such model file: {p}")
    kind = kind or classify_kind(p) or "classifier"
    notes: List[str] = list(extra_notes)

    settings: Dict[str, Any] = {}
    settings_path = ""
    metrics: Dict[str, Any] = {}
    if kind == "classifier":
        run = _run_for(p, runs)
        if run is not None:
            settings = dict(run.settings)
            settings_path = run.settings_path
            metrics = _metrics_from_run(run)
        if not settings:
            settings, settings_path = _settings_beside(p)
        trained_on = _describe_classifier_training(settings)
    else:
        settings, settings_path = _settings_beside(p)
        trained_on = _describe_cellpose_training(settings)

    if trained_on == UNKNOWN:
        notes.append(
            "no settings snapshot found beside this model, so what it was "
            "trained on is unknown — treat it as untested on your images")
    if not _looks_like_checkpoint(p):
        notes.append(
            f"{p.name} does not start with a torch header; it may be a Git LFS "
            f"pointer, a failed download, or not a model at all")

    try:
        size = p.stat().st_size
    except OSError:
        size = 0

    digest = sha256
    if compute_hash and not digest:
        digest = sha256_file(p)

    return ModelEntry(
        key=key or _key_for(p),
        name=p.name,
        kind=kind,
        source=source,
        path=str(p.resolve()),
        version=_version_of(p.name),
        sha256=digest,
        size_bytes=int(size),
        trained_on=trained_on,
        trained_by=_who(settings),
        metrics=metrics,
        notes=tuple(notes),
        verified=bool(verified and digest),
        settings_path=settings_path,
    )


def _metrics_from_run(run: Any) -> Dict[str, Any]:
    """Best/last accuracy off a :class:`spacr.train_compare.TrainingRun`.

    Both, never one: the best epoch of a validation curve was chosen using that
    curve, so it is optimistically biased; the last epoch is unbiased but may
    be well past the optimum. :mod:`spacr.train_compare` makes that argument at
    length and reports both for it — the zoo shows the same pair for the same
    reason.
    """
    out: Dict[str, Any] = {}
    final = getattr(run, "final_metrics", None) or {}
    for label, entry in final.items():
        if str(entry.get("split")) != "val":
            continue
        best = (entry.get("best") or {}).get("accuracy")
        last = (entry.get("last") or {}).get("accuracy")
        if best:
            out[f"{label} best accuracy"] = f"{best['value']:.4f} @ epoch {best['epoch']}"
        if last:
            out[f"{label} last accuracy"] = f"{last['value']:.4f} @ epoch {last['epoch']}"
    if not out:
        for label, entry in final.items():
            best = (entry.get("best") or {}).get("accuracy")
            if best:
                out[f"{label} best accuracy"] = (
                    f"{best['value']:.4f} @ epoch {best['epoch']} "
                    f"(train split — not held out)")
    return out


def _run_for(path: Path, runs: Optional[Mapping[str, Any]]) -> Any:
    """The discovered training run a checkpoint belongs to, or one loaded now."""
    if runs:
        for folder in (path.parent, path.parent.parent):
            run = runs.get(str(folder.resolve()))
            if run is not None:
                return run
    from .train_compare import load_run

    for folder in (path.parent, path.parent.parent):
        try:
            return load_run(folder)
        except Exception:
            continue
    return None


def _runs_under(root: Path) -> Dict[str, Any]:
    """``{folder: TrainingRun}`` for every training run below ``root``.

    Straight reuse of :func:`spacr.train_compare.find_runs` — the classifier
    half of the zoo *is* the training-run scan, and a second discovery pass
    would drift out of step with it the first time the on-disk layout changed.
    Failures are swallowed: a zoo that cannot list local models because one
    folder was unreadable is worse than one with thinner provenance.
    """
    try:
        from .train_compare import find_runs

        return {str(Path(run.path).resolve()): run for run in find_runs(root)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def package_model_root() -> Path:
    """``<spacr>/resources/models`` — where the bundled pack lives.

    The same folder :func:`spacr.utils.download_models` fills and
    :func:`spacr.submodules.analyze_plaques` reads from.
    """
    return Path(__file__).resolve().parent / "resources" / "models"


def default_local_roots() -> List[Path]:
    """Folders worth scanning when the caller has not named one.

    The bundled pack, the Cellpose user folder, and spaCR's own model cache.
    Only the ones that exist come back.
    """
    roots = [
        package_model_root(),
        Path.home() / ".cellpose" / "models",
        Path.home() / ".spacr" / "models",
    ]
    return [r for r in roots if r.is_dir()]


def _walk(base: Path, max_depth: int, limit: int) -> Iterator[Path]:
    """Every file at most ``max_depth`` levels below ``base``, hidden dirs skipped.

    ``limit`` caps the files *examined*, not the models found, so a folder with
    a million PNGs in it is slow rather than fatal and does not silently drop
    the models below them.
    """
    stack: List[Tuple[Path, int]] = [(base, 0)]
    seen = 0
    while stack and seen < limit:
        node, depth = stack.pop(0)
        try:
            children = sorted(node.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                if depth < max_depth:
                    stack.append((child, depth + 1))
            elif child.is_file():
                seen += 1
                yield child
                if seen >= limit:
                    return


def discover_local(roots: Any = None, max_depth: int = DEFAULT_SCAN_DEPTH,
                   compute_hashes: bool = False,
                   limit: int = DEFAULT_SCAN_LIMIT) -> List[ModelEntry]:
    """Find the model checkpoints already on this machine.

    Cellpose models and classifier checkpoints are told apart by
    :func:`classify_kind`; everything else in the folders — settings CSVs, mask
    ``.npy`` files, montage PNGs, logs — is ignored.

    Nothing is downloaded and nothing is hashed unless ``compute_hashes`` is
    set: this is the function behind a list widget, and it has to be fast
    enough to run on a folder the user just typed.

    :param roots: a folder, a file, or an iterable of them; None uses
        :func:`default_local_roots`.
    :param max_depth: how deep below each root to look.
    :param compute_hashes: hash every file found (minutes on a big folder).
    :param limit: stop after examining this many files per root.
    :returns: entries, Cellpose first, then by name.
    """
    entries: List[ModelEntry] = []
    seen: set = set()
    for root in _as_paths(roots if roots is not None else default_local_roots()):
        if root.is_file():
            candidates: List[Path] = [root]
            runs: Dict[str, Any] = {}
        elif root.is_dir():
            candidates = list(_walk(root, max_depth, limit))
            runs = _runs_under(root)
        else:
            continue
        for path in candidates:
            kind = classify_kind(path)
            if kind is None:
                continue
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            source = ("bundled" if _under(path, package_model_root())
                      else "local")
            try:
                entries.append(entry_from_file(path, kind=kind, source=source,
                                               runs=runs,
                                               compute_hash=compute_hashes))
            except ModelUnreadable:
                continue
    entries.sort(key=lambda e: (0 if e.kind == "cellpose" else 1, e.name))
    return entries


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _as_paths(roots: Any) -> List[Path]:
    """Normalise a folder / path / iterable-of-those into a list of Paths."""
    if roots is None:
        return []
    if isinstance(roots, (str, os.PathLike)):
        return [Path(roots)]
    return [Path(r) for r in roots]


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------

def hf_uri(repo_id: str, filename: str) -> str:
    """The download URL for a file in a Hugging Face **dataset** repo.

    Exactly the URL :func:`spacr.utils.download_models` and
    :func:`spacr.qt.hf_download._download_one` build, kept in one place so the
    zoo cannot drift away from the downloader spaCR already ships.
    """
    return (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
            f"{filename}?download=true")


def _entry_from_mapping(data: Mapping[str, Any],
                        source: str = "remote") -> ModelEntry:
    """One catalogue record -> a :class:`ModelEntry`."""
    name = str(data.get("name") or data.get("key") or "")
    if not name:
        raise ValueError("a catalogue entry needs at least a name")
    uri = data.get("uri")
    if not uri:
        uri = hf_uri(str(data.get("repo_id") or HF_MODELS_REPO), name)
    notes = tuple(str(n) for n in (data.get("notes") or ()))
    sha = str(data.get("sha256") or "").strip().lower()
    if not sha:
        notes = notes + (
            "no published checksum — this entry cannot be verified, and fetch "
            "refuses it unless you explicitly accept that",)
    return ModelEntry(
        key=str(data.get("key") or name),
        name=name,
        kind=str(data.get("kind") or "cellpose"),
        source=str(data.get("source") or source),
        path=str(data.get("path") or ""),
        uri=str(uri),
        version=str(data.get("version") or "1"),
        sha256=sha,
        size_bytes=int(data.get("size_bytes") or 0),
        trained_on=data.get("trained_on") or UNKNOWN,
        trained_by=data.get("trained_by") or UNKNOWN,
        metrics=dict(data.get("metrics") or {}),
        notes=notes,
    )


def load_catalogue_file(path: Any) -> List[ModelEntry]:
    """Read a JSON catalogue of remote models.

    Format — a list, or an object with a ``models`` list::

        {"models": [
          {"key": "hela_60x",
           "name": "hela_60x_confluent.CP_model",
           "kind": "cellpose",
           "uri": "https://…/hela_60x_confluent.CP_model",
           "sha256": "9f86d0…",
           "size_bytes": 26566572,
           "trained_on": "HeLa, 60x, confluent monolayer, 512px crops",
           "trained_by": "A. Researcher, 2026-02",
           "metrics": {"note": "benchmarked on plate3 fields 1-3"}}
        ]}

    ``sha256`` is the field that decides whether the entry is usable without an
    explicit override, so a catalogue is worth exactly as much as its hashes.

    :param path: the JSON file.
    :returns: the entries.
    :raises ModelZooError: when the file cannot be read or is not a catalogue,
        naming the file.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        raise ModelZooError(f"no such catalogue file: {p}") from None
    except (OSError, ValueError) as e:
        raise ModelZooError(f"could not read the catalogue {p}: {e}") from None
    records = data.get("models") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ModelZooError(
            f"{p} is not a model catalogue — expected a list of entries, or an "
            f"object with a 'models' list, got {type(records).__name__}")
    out: List[ModelEntry] = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ModelZooError(
                f"{p}: entry {i} is a {type(record).__name__}, not an object")
        try:
            out.append(_entry_from_mapping(record))
        except ValueError as e:
            raise ModelZooError(f"{p}: entry {i} is unusable — {e}") from None
    return out


def catalogue(include_bundled: bool = True, remote: bool = True,
              catalogue_path: Any = None,
              include_plugins: bool = True) -> List[ModelEntry]:
    """Everything the zoo knows about without scanning the user's disks.

    That is: the models bundled with the installed package (whatever
    :func:`spacr.utils.download_models` has put in ``resources/models``), plus
    the declared remote entries — :data:`BUNDLED_REMOTE_MODELS` and, if one is
    configured, the JSON catalogue named by ``catalogue_path`` or the
    :data:`CATALOGUE_ENV_VAR` environment variable.

    Purely local: this reads files and an environment variable and makes no
    network call, so it works offline and on a machine with no torch.

    :param include_bundled: list the models in the package resources folder.
    :param remote: list declared remote entries.
    :param catalogue_path: a JSON catalogue to add; defaults to
        ``$SPACR_MODEL_CATALOGUE`` when that names a file.
    :param include_plugins: include entries returned by installed spaCR model
        providers. Provider failures are recorded in plugin diagnostics and do
        not hide built-in entries.
    :returns: bundled entries first, then remote ones already present locally
        are dropped (a downloaded model is listed once, as the local file).
    """
    entries: List[ModelEntry] = []
    if include_bundled:
        root = package_model_root()
        if root.is_dir():
            entries.extend(discover_local(root, max_depth=2))

    if remote:
        have = {(e.key, e.name) for e in entries}
        for record in BUNDLED_REMOTE_MODELS:
            entry = _entry_from_mapping(record)
            if (entry.key, entry.name) not in have:
                entries.append(entry)
                have.add((entry.key, entry.name))
        path = catalogue_path or os.environ.get(CATALOGUE_ENV_VAR, "")
        if path and os.path.isfile(str(path)):
            for entry in load_catalogue_file(path):
                if (entry.key, entry.name) not in have:
                    entries.append(entry)
                    have.add((entry.key, entry.name))
    if include_plugins:
        try:
            from .plugins import (
                load_object, model_providers, record_diagnostic,
            )
            have = {(entry.key, entry.name) for entry in entries}
            for plugin_name, contribution in model_providers():
                try:
                    provider = load_object(contribution.provider)
                    if not callable(provider):
                        raise TypeError(
                            f"{contribution.provider!r} is not callable"
                        )
                    produced = provider()
                    if isinstance(produced, (ModelEntry, Mapping)):
                        produced = (produced,)
                    for item in produced or ():
                        entry = (
                            item if isinstance(item, ModelEntry)
                            else _entry_from_mapping(item)
                        )
                        identity = (entry.key, entry.name)
                        if identity not in have:
                            entries.append(entry)
                            have.add(identity)
                except Exception as exc:
                    record_diagnostic(
                        plugin_name,
                        f"Model provider {contribution.key!r} failed",
                        exc,
                    )
        except Exception:
            LOG.exception("Could not initialise plugin model providers")
    return entries


def resolve(key_or_path: Any,
            entries: Optional[Sequence[ModelEntry]] = None) -> ModelEntry:
    """Turn a key, a name or a path into a :class:`ModelEntry`.

    A path that exists wins over a key: pointing the zoo at a file you just
    trained has to work without registering it anywhere first.

    :param key_or_path: an entry key, a model filename, or a path to a file.
    :param entries: the listing to search; defaults to :func:`catalogue`.
    :returns: the entry.
    :raises ModelUnreadable: when it looks like a path and no file is there.
    :raises ModelZooError: when no entry matches, listing the near misses.
    """
    text = str(key_or_path or "").strip()
    if not text:
        raise ModelZooError("no model given")
    if os.path.isfile(text):
        return entry_from_file(text)
    if os.sep in text or text.startswith("~"):
        # It was written as a path, so "no entry called that" would be the
        # wrong complaint: the user pointed at a file and the file is not there.
        raise ModelUnreadable(
            f"no such model file: {text} — and nothing in the zoo is called "
            f"that either")

    pool = list(entries) if entries is not None else catalogue()
    for entry in pool:
        if entry.key == text or entry.name == text:
            return entry
    lowered = text.lower()
    near = [e.key for e in pool if lowered in e.key.lower()
            or lowered in e.name.lower()]
    if len(near) == 1:
        return next(e for e in pool if e.key == near[0])
    raise ModelZooError(
        f"no model called {text!r} in the zoo"
        + (f" — did you mean one of: {', '.join(near[:5])}?" if near
           else f" ({len(pool)} entries known; pass a path to use a file "
                f"that is not registered)"))


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def versioned_path(dest: Any, filename: str) -> Path:
    """The first free destination for ``filename`` in ``dest``.

    ``foo.CP_model`` -> ``foo.CP_model``, then ``foo_v2.CP_model``,
    ``foo_v3.CP_model``… An existing checkpoint is never overwritten: two
    models with the same filename are a normal thing to have (the same author
    retrained, or two people picked the same name), and the failure mode of
    overwriting — a run that used the old weights becoming unreproducible with
    no trace — is silent.

    An input that already carries ``_vN`` counts from there rather than
    becoming ``foo_v2_v2``.

    :param dest: destination directory.
    :param filename: the name to place there.
    :returns: a path that does not exist yet.
    """
    folder = Path(dest)
    p = Path(filename)
    suffix = p.suffix
    match = _VERSION_RE.match(p.stem)
    base = match.group("base") if match else p.stem
    n = int(match.group("n")) if match else 1

    candidate = folder / (f"{base}{suffix}" if n == 1
                          else f"{base}_v{n}{suffix}")
    while candidate.exists():
        n += 1
        candidate = folder / f"{base}_v{n}{suffix}"
    return candidate


def open_uri(uri: str, timeout: int = DEFAULT_TIMEOUT,
             chunk_size: int = DEFAULT_CHUNK
             ) -> Tuple[Iterable[bytes], int]:
    """Open a model URI for streaming. ``(chunks, total_bytes)``.

    ``http://`` and ``https://`` stream over ``requests`` — the same call
    :func:`spacr.utils.download_models` and
    :func:`spacr.qt.hf_download._download_one` make, imported here so this
    module has no hard dependency on it. ``file://`` and a plain existing path
    are read from disk, which is what a lab mirror on a NAS looks like and what
    the tests use, so the whole fetch path is exercised without a network.

    :param uri: where the model lives.
    :param timeout: seconds, HTTP only.
    :param chunk_size: bytes per chunk.
    :returns: ``(iterable of byte chunks, total size or 0 when unknown)``.
    :raises ModelZooError: for a scheme this does not speak.
    """
    text = str(uri or "")
    if text.startswith(("http://", "https://")):
        import requests

        response = requests.get(text, stream=True, timeout=timeout)
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        return response.iter_content(chunk_size=chunk_size), total

    if text.startswith("file://"):
        local = Path(text[len("file://"):])
    elif os.path.exists(text):
        local = Path(text)
    else:
        raise ModelZooError(
            f"do not know how to fetch {text!r} — expected an http(s):// URL, "
            f"a file:// URL, or a path that exists")
    if not local.is_file():
        raise ModelUnreadable(f"no such model file: {local}")
    return _read_chunks(local, chunk_size), local.stat().st_size


def _read_chunks(path: Path, chunk_size: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return
            yield block


def fetch(entry: ModelEntry, dest: Any,
          expected_sha256: Optional[str] = None,
          require_checksum: bool = True,
          opener: Optional[Callable[[str], Any]] = None,
          progress: Optional[Callable[[int, int], None]] = None,
          cancel: Optional[Callable[[], bool]] = None,
          chunk_size: int = DEFAULT_CHUNK,
          timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Download a model, verify it, and only then put it where it belongs.

    The order is the whole point:

    1. bytes stream into a temporary file **inside** ``dest``, so the rename in
       step 4 is a same-filesystem ``os.replace`` and therefore atomic;
    2. the checksum of what actually arrived is computed;
    3. if it does not match the published digest the temporary file is deleted
       and :class:`ChecksumMismatch` is raised — nothing is installed, and the
       destination still holds whatever it held before;
    4. only now is the temporary file renamed, to a
       :func:`versioned_path` that does not exist yet.

    Every failure — a dead server, a cancel, a bad hash, a full disk — leaves
    the destination directory exactly as it was. There is no window in which a
    half-written file sits at a name that looks like a model.

    :param entry: what to fetch. :attr:`ModelEntry.uri` is the source.
    :param dest: destination directory; created if missing.
    :param expected_sha256: digest to require, overriding
        :attr:`ModelEntry.sha256`.
    :param require_checksum: refuse to install when no digest is known. True by
        default — a download nobody can check is exactly the thing this module
        exists to stop being routine. Pass False to accept one knowingly; the
        entry :func:`install` returns then reports ``verified=False``.
    :param opener: ``fn(uri) -> chunks`` or ``fn(uri) -> (chunks, total)``;
        defaults to :func:`open_uri`.
    :param progress: ``fn(done_bytes, total_bytes)``; ``total`` is 0 when the
        server did not say.
    :param cancel: ``fn() -> bool``, polled between chunks. Returning True
        deletes the partial file and raises :class:`DownloadCancelled`.
    :param chunk_size: bytes per read.
    :param timeout: seconds, HTTP only.
    :returns: the path the model was written to.
    :raises ChecksumMismatch: the bytes are not the published bytes.
    :raises DownloadCancelled: ``cancel()`` returned True.
    :raises ModelZooError: no URI, or no checksum with ``require_checksum``.
    """
    if not entry.uri:
        raise ModelZooError(
            f"{entry.name} has no uri to fetch it from (source={entry.source})")
    want = (expected_sha256 if expected_sha256 is not None
            else entry.sha256) or ""
    want = want.strip().lower()
    if require_checksum and not want:
        raise ModelZooError(
            f"refusing to install {entry.name}: no sha256 was published for "
            f"it, so a truncated or substituted checkpoint could not be told "
            f"from the real one. Supply expected_sha256=…, put a hash in the "
            f"catalogue, or pass require_checksum=False to accept it "
            f"unverified.")

    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)

    import tempfile

    handle = tempfile.NamedTemporaryFile(
        dir=str(folder), prefix=f".{Path(entry.name).stem}.", suffix=".part",
        delete=False)
    temp = Path(handle.name)
    done = 0
    try:
        stream = (opener(entry.uri) if opener is not None
                  else open_uri(entry.uri, timeout=timeout,
                                chunk_size=chunk_size))
        chunks, total = stream if isinstance(stream, tuple) else (stream, 0)
        total = int(total or entry.size_bytes or 0)
        if progress is not None:
            progress(0, total)
        for block in chunks:
            if cancel is not None and cancel():
                raise DownloadCancelled(
                    f"download of {entry.name} cancelled after {done} byte(s) "
                    f"— nothing was written to {folder}")
            if not block:
                continue
            handle.write(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

        if done == 0:
            raise ModelZooError(
                f"{entry.uri} returned no data for {entry.name} — nothing was "
                f"written to {folder}")

        got = sha256_file(temp)
        if want and got != want:
            raise ChecksumMismatch(
                f"{entry.name} does not match its published checksum and was "
                f"NOT installed in {folder}.\n"
                f"  expected sha256 {want}\n"
                f"  got      sha256 {got}\n"
                f"  ({done} bytes from {entry.uri})\n"
                f"A checkpoint that fails this still loads and still produces "
                f"masks — they are just not the masks the author's model "
                f"produces. Re-download, or get the right hash from whoever "
                f"published it.")
        return _claim(temp, folder, entry.name)
    except BaseException:
        # Every failure path leaves the destination untouched: no partial file
        # at a real model name, ever.
        try:
            handle.close()
        except Exception:
            pass
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass
        raise


def _claim(temp: Path, folder: Path, name: str) -> Path:
    """Move ``temp`` onto the first free version of ``name``, race-free.

    :func:`versioned_path` alone is check-then-act: two downloads of the same
    model finishing together both see ``foo.CP_model`` free, and the second
    ``os.replace`` silently destroys the first. So the name is *reserved* with
    ``O_CREAT | O_EXCL`` — which the kernel makes atomic — before the rename,
    and a lost race just tries the next version.

    The reservation is a zero-byte file that exists for microseconds and is
    then replaced. If the process is killed inside that window what remains is
    an empty file, which :func:`inspect_checkpoint` rejects by name — unlike a
    truncated checkpoint, which loads.
    """
    while True:
        target = versioned_path(folder, name)
        try:
            os.close(os.open(str(target),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        except FileExistsError:
            continue
        os.replace(str(temp), str(target))
        return target


def install(entry: ModelEntry, dest: Any, **kwargs: Any) -> ModelEntry:
    """:func:`fetch` the model and return the registered local entry.

    The returned entry carries the digest of the bytes that were actually
    written — not the one the catalogue claimed — and
    :attr:`ModelEntry.verified` is True only when the two were compared and
    matched. Provenance from the catalogue entry is carried over, because that
    is the whole reason for having had a catalogue.

    :param entry: the remote entry.
    :param dest: destination directory.
    :param kwargs: passed to :func:`fetch`.
    :returns: a ``source='local'`` entry pointing at the new file.
    """
    expected = kwargs.get("expected_sha256")
    want = ((expected if expected is not None else entry.sha256) or "").strip()
    path = fetch(entry, dest, **kwargs)
    digest = sha256_file(path)
    notes = tuple(n for n in entry.notes if "no published checksum" not in n)
    if not want:
        notes = notes + (
            f"installed without a published checksum to check against; the "
            f"recorded sha256 {digest[:12]}… is of the bytes that arrived, "
            f"which proves nothing about where they came from",)
    return replace(
        entry,
        key=_key_for(path),
        name=path.name,
        source="local",
        path=str(path.resolve()),
        version=_version_of(path.name),
        sha256=digest,
        size_bytes=path.stat().st_size,
        verified=bool(want),
        notes=notes,
    )


def download_bundled_models(**kwargs: Any) -> str:
    """Pull the bundled Hugging Face model pack via the existing downloader.

    Thin, deliberate wrapper over :func:`spacr.utils.download_models` — the
    downloader spaCR already ships and the one
    :func:`spacr.submodules.analyze_plaques` depends on. It is *not*
    reimplemented here, so there is one code path that fills
    ``resources/models`` and one place to fix when the repo moves.

    It is also the unverified path: that function has no checksum, writes
    straight to the destination filename, and skips the whole pull when the
    folder is non-empty. Prefer a catalogue entry with a hash and
    :func:`install`; this exists so the zoo can offer the legacy pack rather
    than pretend it does not exist.

    ``spacr.utils`` imports torch, so it is imported here and not at module
    level. Nothing else in this module reaches for it.

    :param kwargs: forwarded to :func:`spacr.utils.download_models`.
    :returns: the local directory the pack landed in.
    """
    return _bulk_downloader()(**kwargs)


def _bulk_downloader() -> Callable[..., str]:
    """The legacy bulk downloader, imported late (it pulls torch in with it)."""
    from .utils import download_models

    return download_models


# ---------------------------------------------------------------------------
# benchmarking — "test on 3 fields"
# ---------------------------------------------------------------------------

@dataclass
class FieldBenchmark:
    """One field's result for one model.

    :param field: the field name.
    :param n_objects: labels in the mask this model produced.
    :param severity: :mod:`spacr.seg_qc`'s verdict — ``'ok'``, ``'warn'`` or
        ``'fail'``, or ``'-'`` when QC was off.
    :param flags: the named defects seg_qc raised.
    :param note: seg_qc's verdict in prose, with its numbers in it.
    """

    field: str
    n_objects: int = 0
    severity: str = "-"
    flags: Tuple[str, ...] = ()
    note: str = ""


@dataclass
class BenchmarkResult:
    """One model over one field set. Only comparable to results on the *same* set.

    :param entry: the model that ran.
    :param fieldset: :func:`fieldset_id` of the images — a hash of the pixels,
        so two runs over the same three fields share it and two runs over
        different fields never do, whatever the folders were called.
    :param fieldset_label: the same thing for a human.
    :param rows: one :class:`FieldBenchmark` per field, in field order.
    :param seconds: wall-clock seconds the model spent segmenting.
    :param honoured: the parameters that reached the model.
    :param ignored: what was set and Cellpose 4 dropped (``diam_mean`` and
        friends) — carried so a benchmark cannot silently be a benchmark of
        settings nothing read.
    :param notes: warnings the reader must see before the numbers.
    :param masks: the label images, when kept, so a GUI can draw them.
    :param images: the source fields, likewise.
    :param object_type: what was segmented.
    """

    entry: ModelEntry
    fieldset: str = ""
    fieldset_label: str = ""
    rows: List[FieldBenchmark] = _dc_field(default_factory=list)
    seconds: float = 0.0
    honoured: Dict[str, Any] = _dc_field(default_factory=dict)
    ignored: Dict[str, Any] = _dc_field(default_factory=dict)
    notes: List[str] = _dc_field(default_factory=list)
    masks: List[Any] = _dc_field(default_factory=list)
    images: List[Any] = _dc_field(default_factory=list)
    object_type: str = "cell"

    @property
    def fields(self) -> List[str]:
        return [r.field for r in self.rows]

    @property
    def n_fields(self) -> int:
        return len(self.rows)

    @property
    def total_objects(self) -> int:
        return sum(r.n_objects for r in self.rows)

    @property
    def mean_objects(self) -> float:
        return self.total_objects / self.n_fields if self.rows else float("nan")

    @property
    def n_failed(self) -> int:
        """Fields :mod:`spacr.seg_qc` scored ``'fail'``."""
        return sum(1 for r in self.rows if r.severity == "fail")

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r.severity == "ok")

    @property
    def qc_score(self) -> float:
        """Fraction of fields seg_qc scored ``'ok'``; ``nan`` without QC.

        A quality-control verdict on this model's own masks — it says the masks
        are not obviously broken, not that they are right. There is no ground
        truth in a benchmark (see the module docstring), so this is as close to
        a score as the zoo will produce.
        """
        scored = [r for r in self.rows if r.severity != "-"]
        if not scored:
            return float("nan")
        return sum(1 for r in scored if r.severity == "ok") / len(scored)

    @property
    def summary(self) -> str:
        score = self.qc_score
        return (f"{self.entry.name}: {self.total_objects} "
                f"{self.object_type}(s) over {self.n_fields} field(s) "
                f"({self.mean_objects:.1f}/field) in {self.seconds:.1f}s; "
                f"seg_qc scored {self.n_ok}/{self.n_fields} field(s) ok"
                + ("" if score != score else f" ({score * 100:.0f}%)")
                + f", {self.n_failed} fail.")


def fieldset_id(names: Sequence[str], images: Sequence[Any]) -> str:
    """A stable id for a set of fields, taken from the **pixels**.

    Folder names are not identity: ``plate1/1`` on two machines is two
    different sets of images, and the same three images copied to a new folder
    are the same benchmark input. So the id hashes each array's bytes, shape and
    dtype together with its name.

    This is what makes :func:`rank` able to refuse. Without it, two benchmarks
    run on different data are two numbers, and two numbers always sort.

    :param names: field names, in order.
    :param images: the arrays, in the same order.
    :returns: a 16-character hex id.
    """
    digest = hashlib.sha256()
    for name, image in zip(names, images):
        array = np.ascontiguousarray(np.asarray(image))
        digest.update(str(name).encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(hashlib.sha256(array.tobytes()).digest())
    return digest.hexdigest()[:16]


def _model_compare():
    """:mod:`spacr.model_compare`, imported late so browsing stays cheap."""
    from . import model_compare

    return model_compare


def config_for(entry: ModelEntry, overrides: Optional[Mapping[str, Any]] = None):
    """The :class:`spacr.model_compare.ModelConfig` that runs this entry.

    A local checkpoint is passed by path, which
    :func:`spacr.utils._choose_model` and
    :func:`spacr.model_compare.segment_with_cellpose` both load as
    ``pretrained_model``; anything else goes through by name and is subject to
    Cellpose 4's legacy-name remapping, which the config reports.

    :param entry: the model.
    :param overrides: eval settings (``diameter``, ``flow_threshold``, …).
    :returns: the config.
    """
    mc = _model_compare()
    settings: Dict[str, Any] = dict(overrides or {})
    settings["model"] = entry.path or entry.name
    settings.setdefault("name", entry.name)
    return mc.ModelConfig.from_mapping(settings)


def benchmark(entry: ModelEntry, images: Optional[Sequence[Any]] = None,
              source: Any = None, n_fields: int = DEFAULT_N_FIELDS,
              field_names: Optional[Sequence[str]] = None,
              segment_fn: Optional[Callable] = None,
              settings: Optional[Mapping[str, Any]] = None,
              object_type: str = "cell", qc: bool = True,
              keep_images: bool = True, channel: Optional[int] = None,
              progress: Optional[Callable[[str, int, int], None]] = None,
              ) -> BenchmarkResult:
    """Run one model over N fields and report what came out. "Test on 3 fields".

    This is :mod:`spacr.model_compare`'s harness with one model instead of two:
    the same :func:`~spacr.model_compare.load_fields` reader, the same
    :class:`~spacr.model_compare.ModelConfig` (so the same arguments are
    honoured and the same ones reported as ignored), the same
    :func:`~spacr.model_compare.segment_with_cellpose` backend, and the same
    :mod:`spacr.seg_qc` scorecards. To put two models side by side use
    :func:`compare_entries`, which calls
    :func:`~spacr.model_compare.compare_models` proper.

    The checkpoint is checked before it is loaded, so a missing or corrupt file
    fails with its own name in the message rather than a torch ``KeyError``.

    :param entry: the model to run.
    :param images: fields already in memory; None loads them from ``source``.
    :param source: a folder of fields (``.tif`` / ``.png`` / ``.npy`` /
        ``.npz``), read by :func:`spacr.model_compare.load_fields`.
    :param n_fields: how many fields to take from ``source``.
    :param field_names: names for the rows.
    :param segment_fn: ``fn(images, config) -> masks``; defaults to
        :func:`spacr.model_compare.segment_with_cellpose`. This is the seam the
        GUI and the tests use, and the reason no test here loads Cellpose.
    :param settings: eval overrides (``diameter``, ``flow_threshold``, …).
    :param object_type: what is being segmented, for the seg_qc scorecards.
    :param qc: score the masks with :mod:`spacr.seg_qc`.
    :param keep_images: keep images and masks on the result for a GUI to draw.
    :param channel: index into the last axis for multi-channel fields.
    :param progress: ``fn(message, done, total)``.
    :returns: a :class:`BenchmarkResult`.
    :raises ModelUnreadable: when the checkpoint is missing or not a checkpoint.
    :raises ValueError: when there is no field, or the model returned the wrong
        number of masks.
    """
    mc = _model_compare()

    if images is None:
        if source is None:
            raise ValueError(
                "benchmark needs either images= or source= (a folder of fields)")
        field_names, images = mc.load_fields(source, n_fields=n_fields,
                                             channel=channel)
    fields = [np.asarray(image) for image in images]
    if not fields:
        raise ValueError("no field to benchmark: pass at least one image")
    names = ([str(n) for n in field_names] if field_names is not None
             else [f"field_{i:04d}" for i in range(len(fields))])
    if len(names) != len(fields):
        raise ValueError(
            f"got {len(names)} field name(s) for {len(fields)} field(s)")

    config = config_for(entry, settings)
    notes = list(entry.notes) + list(config.notes())
    if not entry.provenance_known:
        notes.append(
            f"{entry.name} does not record what it was trained on, so a good "
            f"score here says it works on these fields and nothing more.")

    # Fail on the file, not on a state-dict key three frames inside torch.
    if entry.path:
        inspect_checkpoint(entry.path)

    total_steps = 2

    def _tick(message: str, done: int) -> None:
        if progress is not None:
            progress(message, done, total_steps)

    _tick(f"Segmenting {len(fields)} field(s) with {entry.name}…", 0)
    run = segment_fn if segment_fn is not None else mc.segment_with_cellpose
    started = time.perf_counter()
    produced = list(run(fields, config))
    seconds = time.perf_counter() - started
    if len(produced) != len(fields):
        raise ValueError(
            f"{entry.name} returned {len(produced)} mask(s) for "
            f"{len(fields)} field(s)")
    masks = [mc._as_labels(m) for m in produced]

    _tick("Scoring masks…", 1)
    # mc._score is spacr.seg_qc.score_masks over the whole set at once, which
    # is what gives the plate-relative flags something to compare against.
    scores = mc._score(masks, names, object_type) if qc else [None] * len(fields)

    rows = [
        FieldBenchmark(
            field=names[i],
            n_objects=int(np.unique(masks[i]).size - (1 if (masks[i] == 0).any()
                                                      else 0)),
            severity=scores[i].severity if scores[i] else "-",
            flags=tuple(scores[i].flags) if scores[i] else (),
            note=scores[i].note if scores[i] else "",
        )
        for i in range(len(fields))
    ]
    _tick("Done", 2)

    return BenchmarkResult(
        entry=entry,
        fieldset=fieldset_id(names, fields),
        fieldset_label=_fieldset_label(names, source),
        rows=rows,
        seconds=seconds,
        honoured=config.honoured_parameters(),
        ignored=config.ignored_parameters(),
        notes=notes,
        masks=masks if keep_images else [],
        images=fields if keep_images else [],
        object_type=object_type,
    )


def _fieldset_label(names: Sequence[str], source: Any) -> str:
    """"3 field(s) from …: a, b, c" — what a group header says."""
    where = f" from {os.fspath(source)}" if isinstance(
        source, (str, os.PathLike)) else ""
    listed = ", ".join(str(n) for n in list(names)[:4])
    if len(names) > 4:
        listed += f", … (+{len(names) - 4})"
    return f"{len(names)} field(s){where}: {listed}"


def compare_entries(entry_a: ModelEntry, entry_b: ModelEntry,
                    images: Optional[Sequence[Any]] = None,
                    source: Any = None, n_fields: int = DEFAULT_N_FIELDS,
                    field_names: Optional[Sequence[str]] = None,
                    settings_a: Optional[Mapping[str, Any]] = None,
                    settings_b: Optional[Mapping[str, Any]] = None,
                    **kwargs: Any):
    """Put two zoo entries head to head on the same fields.

    Straight delegation to :func:`spacr.model_compare.compare_models` — the
    metrics, the split/merge attribution and the "neither model is ground
    truth" wording all come from there, unchanged. This function's only job is
    turning two :class:`ModelEntry` objects into two
    :class:`~spacr.model_compare.ModelConfig` objects.

    :param entry_a: the A side.
    :param entry_b: the B side.
    :param images: fields already in memory; None loads them from ``source``.
    :param source: a folder of fields.
    :param n_fields: how many fields to take from ``source``.
    :param field_names: names for the rows.
    :param settings_a: eval overrides for A.
    :param settings_b: eval overrides for B.
    :param kwargs: forwarded to :func:`spacr.model_compare.compare_models`.
    :returns: a :class:`spacr.model_compare.ComparisonReport`.
    """
    mc = _model_compare()
    if images is None:
        if source is None:
            raise ValueError(
                "compare_entries needs either images= or source=")
        field_names, images = mc.load_fields(source, n_fields=n_fields)
    for entry in (entry_a, entry_b):
        if entry.path:
            inspect_checkpoint(entry.path)
    return mc.compare_models(
        images,
        config_for(entry_a, settings_a),
        config_for(entry_b, settings_b),
        field_names=field_names,
        **kwargs)


# ---------------------------------------------------------------------------
# ranking — the part that refuses
# ---------------------------------------------------------------------------

def group_by_fieldset(results: Sequence[BenchmarkResult]
                      ) -> Dict[str, List[BenchmarkResult]]:
    """Bucket benchmarks by the field set they ran on, first-seen order.

    :param results: benchmarks.
    :returns: ``{fieldset_id: [results]}``.
    """
    groups: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault(result.fieldset, []).append(result)
    return groups


def _rank_value(result: BenchmarkResult, key: str) -> Tuple:
    if key == "qc":
        score = result.qc_score
        # nan sorts last rather than first: a model nobody scored is not the
        # best model.
        return (-(score if score == score else -1.0), result.seconds,
                result.entry.name)
    if key == "seconds":
        return (result.seconds, result.entry.name)
    raise ValueError(
        f"rank key must be one of {tuple(RANK_KEYS)}, got {key!r}. There is "
        f"deliberately no accuracy key: a benchmark here has no ground truth, "
        f"so a column that sorted models by 'score' would be inventing one.")


def rank(results: Sequence[BenchmarkResult],
         key: str = DEFAULT_RANK_KEY) -> List[BenchmarkResult]:
    """Order benchmarks best-first — **within one field set only**.

    A model's numbers on your three fields say nothing about its numbers on
    somebody else's: different cell density, different exposure, different
    magnification. Sorting results from two field sets into one list produces a
    ranking that looks exactly like a real one and means nothing, which is the
    failure this function exists to prevent. So it refuses.

    :param results: benchmarks, all from the same field set.
    :param key: one of :data:`RANK_KEYS`.
    :returns: the results, best first.
    :raises IncomparableBenchmarks: when the results span more than one field
        set. Use :func:`rank_groups` or :func:`format_benchmarks`, which group
        and label instead.
    :raises ValueError: on an unknown ``key``.
    """
    groups = group_by_fieldset(results)
    if len(groups) > 1:
        detail = "; ".join(
            f"{members[0].fieldset_label} "
            f"[{', '.join(m.entry.name for m in members)}]"
            for members in groups.values())
        raise IncomparableBenchmarks(
            f"refusing to rank {len(results)} benchmark(s) that ran on "
            f"{len(groups)} different field sets — a score on one set of "
            f"fields says nothing about a score on another, so sorting them "
            f"together would invent a ranking. The sets were: {detail}. Use "
            f"rank_groups() or format_benchmarks() to rank within each set.")
    return sorted(results, key=lambda r: _rank_value(r, key))


def rank_groups(results: Sequence[BenchmarkResult],
                key: str = DEFAULT_RANK_KEY
                ) -> Dict[str, List[BenchmarkResult]]:
    """Rank inside each field set, keeping the sets apart. The safe alternative.

    :param results: benchmarks from any number of field sets.
    :param key: one of :data:`RANK_KEYS`.
    :returns: ``{fieldset_id: [results, best first]}``.
    """
    return {fieldset: sorted(members, key=lambda r: _rank_value(r, key))
            for fieldset, members in group_by_fieldset(results).items()}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _render_table(rows: Sequence[Sequence[str]],
                  header: Sequence[str]) -> List[str]:
    """A fixed-width text table.

    :func:`spacr.model_compare._render_table`, reused so the zoo's console
    output is the same shape as the comparison's rather than a second table
    style two lines apart in the same terminal.
    """
    return _model_compare()._render_table(rows, header)


_ZOO_COLUMNS = (
    ("model", lambda e: e.name),
    ("kind", lambda e: e.kind),
    ("source", lambda e: e.source),
    ("v", lambda e: e.version),
    ("size", lambda e: _human_bytes(e.size_bytes)),
    ("checksum", lambda e: e.checksum_state),
    ("trained on", lambda e: _shorten(e.trained_on, 46)),
    ("trained by", lambda e: _shorten(e.trained_by, 22)),
)


def format_zoo(entries: Sequence[ModelEntry]) -> str:
    """Render a listing a human reads before choosing a model.

    Provenance is a column, not a footnote: "trained on" is the field that
    decides whether a model is applicable to your images at all, and it is
    printed for every row — reading ``unknown`` where it is unknown, because a
    blank there would read as "no constraints".

    :param entries: what to list.
    :returns: a multi-line string.
    """
    entries = list(entries)
    if not entries:
        return ("Model zoo: nothing found.\n"
                "  Scan a folder with discover_local(), or point "
                "$SPACR_MODEL_CATALOGUE at a catalogue file.")

    lines = [f"Model zoo — {len(entries)} model(s)"]
    rows = [[str(fmt(e)) for _, fmt in _ZOO_COLUMNS] for e in entries]
    lines.extend(_render_table(rows, [name for name, _ in _ZOO_COLUMNS]))

    unknown = [e for e in entries if not e.provenance_known]
    if unknown:
        lines.append("")
        lines.append(
            f"  {len(unknown)} model(s) do not record what they were trained "
            f"on: {', '.join(e.name for e in unknown[:6])}"
            f"{'…' if len(unknown) > 6 else ''}. A Cellpose model fine-tuned "
            f"on confluent 60x cells is not interchangeable with one trained "
            f"on sparse 20x ones, and nothing here tells you which this is.")

    unchecked = [e for e in entries if e.checksum_state == "none" and e.path]
    if unchecked:
        lines.append(
            f"  {len(unchecked)} local model(s) have no checksum on record; "
            f"run verify() against a published digest before trusting one.")

    notes = [(e.name, n) for e in entries for n in e.notes]
    if notes:
        lines.append("")
        for name, note in notes:
            lines.append(f"  ! {name}: {note}")
    return "\n".join(lines)


_BENCH_COLUMNS = (
    ("field", lambda r: r.field),
    ("objects", lambda r: str(r.n_objects)),
    ("seg_qc", lambda r: r.severity),
    ("flags", lambda r: ", ".join(r.flags) if r.flags else "-"),
)


def format_benchmarks(results: Sequence[BenchmarkResult],
                      key: str = DEFAULT_RANK_KEY) -> str:
    """Render benchmarks **grouped by field set**, ranked only within a group.

    Two models benchmarked on different fields appear under two headers with a
    line saying the two blocks cannot be compared. That is the alternative to
    :func:`rank`'s refusal, and it is the only way this module will ever put
    incomparable numbers on the same page.

    :param results: benchmarks, from any number of field sets.
    :param key: one of :data:`RANK_KEYS`.
    :returns: a multi-line string.
    """
    results = list(results)
    if not results:
        return "No benchmark to show."

    groups = rank_groups(results, key=key)
    lines: List[str] = [
        f"Model benchmarks — {len(results)} run(s) over "
        f"{len(groups)} field set(s), ranked by {key} "
        f"({RANK_KEYS[key] if key in RANK_KEYS else ''})"
    ]
    if len(groups) > 1:
        lines.append(
            "  The blocks below ran on DIFFERENT fields and are not comparable "
            "with each other: a score on one set of images says nothing about "
            "a score on another. Compare within a block only.")

    for fieldset, members in groups.items():
        lines.append("")
        lines.append(f"  field set {fieldset} — {members[0].fieldset_label}")
        for rank_index, result in enumerate(members, start=1):
            lines.append("")
            lines.append(f"    {rank_index}. {result.summary}")
            entry = result.entry
            lines.append(f"       trained on: {entry.trained_on}")
            diameter = result.honoured.get("diameter")
            lines.append(
                f"       model {result.honoured.get('model', entry.name)}, "
                f"diameter {diameter if diameter is not None else 'native'}")
            if result.ignored:
                lines.append(
                    f"       set but ignored by Cellpose 4: "
                    + ", ".join(f"{k}={v!r}" for k, v in result.ignored.items()))
            rows = [[str(fmt(r)) for _, fmt in _BENCH_COLUMNS]
                    for r in result.rows]
            lines.extend("    " + line for line in
                         _render_table(rows, [n for n, _ in _BENCH_COLUMNS]))
            for note in result.notes:
                lines.append(f"       ! {note}")

    lines.append("")
    lines.append("  seg_qc is a quality-control verdict on each model's own "
                 "masks — it catches a model that collapsed on your data. It "
                 "is not an accuracy: there is no ground truth here. To put "
                 "two models against each other, use compare_entries().")
    return "\n".join(lines)
