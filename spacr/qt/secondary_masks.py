"""Pair a read-only primary mask with one editable secondary-mask field.

Folder sources match the image's stem. Explicit files are bound to one image
so moving through a queue cannot accidentally reuse yesterday's nuclei on a
same-sized new field. The immutable snapshot records class names and the
source checksum; neither loading nor validation writes a primary mask.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import mask_engine


def _same_file(first, second):
    """Recognize identical paths, symbolic links and existing hard links."""
    if os.path.realpath(first) == os.path.realpath(second):
        return True
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError):
        return False


@dataclass(frozen=True, eq=False)
class PrimaryMaskSource:
    """A validated primary mask and the field and object classes it belongs to.

    :param path: resolved source-mask path.
    :param image_path: resolved image path identifying this field.
    :param primary_class: primary object class, for example ``nucleus``.
    :param secondary_class: distinct output object class, for example ``cell``.
    :param sha256: checksum of the source file actually read.
    :param labels: owned, read-only uint16 primary labels matching the image.
    """

    path: str
    image_path: str
    primary_class: str
    secondary_class: str
    sha256: str
    labels: np.ndarray

    @property
    def identity(self):
        """Hashable field/source/class token for asynchronous request keys."""
        return (self.path, self.image_path, self.primary_class,
                self.secondary_class, self.sha256)

    def crop(self, box):
        """Copy primary labels inside a rectangle for a worker request.

        :param box: ``(x0,y0,x1,y1)`` pixel bounds, with exclusive upper bounds.
        :returns: independent label array preserving the source object IDs.
        :raises ValueError: the rectangle is empty or extends outside the image.
        """
        x0, y0, x1, y1 = (int(value) for value in box)
        height, width = self.labels.shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Primary-mask crop is outside its image.')
        return self.labels[y0:y1, x0:x1].copy()

    def validate_destination(self, path):
        """Refuse any output that would overwrite the primary, including aliases.

        :param path: proposed secondary-mask destination, as a string or path.
        :raises ValueError: the destination is the primary file, a symbolic
            link to it or an existing hard link to it.
        """
        if _same_file(self.path, os.fspath(path)):
            raise ValueError('Primary and secondary masks must be saved to different files.')

    def provenance(self):
        """Return JSON-safe source identity for the mask's curation ledger."""
        return {'path': self.path, 'image_path': self.image_path,
                'primary_class': self.primary_class,
                'secondary_class': self.secondary_class,
                'sha256': self.sha256, 'shape': list(self.labels.shape)}


def read_primary_source(source, image_path, shape, output_path, *,
                        primary_class='nucleus', secondary_class='cell',
                        bound_image=None):
    """Load a field's primary labels without modifying either mask file.

    :param source: one TIFF/PNG/NPY/Cellpose bundle or a folder containing
        a matching ``<image stem>`` mask. Multiple matching files are refused.
    :param image_path: image currently being edited; field identity is its path.
    :param shape: expected ``(height,width)`` of the primary mask.
    :param output_path: destination of the editable secondary mask. Must not
        alias the primary through a path, symbolic link or hard link.
    :param primary_class: nonempty name of the source object class.
    :param secondary_class: distinct nonempty name of the output class.
    :param bound_image: field for which an explicit file was chosen. Required
        for file sources; ignored for folders that resolve each field by name.
    :returns: immutable :class:`PrimaryMaskSource` with read-only uint16 labels.
    :raises ValueError: ambiguous/missing pairing, incompatible classes, shape,
        labels, field binding, output alias or a source changed during reading.
    :raises OSError: an input cannot be read.
    """
    source = Path(source).expanduser().resolve()
    field = Path(image_path).expanduser().resolve()
    first, second = str(primary_class).strip(), str(secondary_class).strip()
    if not first or not second or first.casefold() == second.casefold():
        raise ValueError('Primary and secondary object classes must be distinct and nonempty.')
    if source.is_dir():
        stem = field.name[:-len(mask_engine.SEG_SUFFIX)] if mask_engine.is_seg_bundle(field.name) else field.stem
        candidates = {candidate.resolve() for suffix in ('.tif', '.tiff', '.png', '.npy', '_seg.npy')
                      if (candidate := source / (stem + suffix)).is_file()}
        if len(candidates) != 1:
            raise ValueError(f'Expected one primary mask for {field.name}; found {len(candidates)}. Select its file explicitly.')
        source = candidates.pop()
    elif bound_image is None or Path(bound_image).expanduser().resolve() != field:
        raise ValueError('This primary-mask file belongs to another field. Choose the matching file or a primary-mask folder.')
    if _same_file(source, output_path):
        raise ValueError('Primary and secondary masks must be saved to different files.')
    before = source.stat()
    if mask_engine.is_seg_bundle(source.name):
        labels = mask_engine.read_seg_bundle(str(source))['masks']
    elif source.suffix.lower() == '.npy':
        labels = np.load(source, allow_pickle=False)
    else:
        labels = mask_engine.imageio.imread(source)
    labels = mask_engine.canonical_labels(labels, preserve_ids=True)
    if tuple(labels.shape) != tuple(shape):
        raise ValueError(f'Primary mask shape {labels.shape} does not match image shape {tuple(shape)}.')
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('The primary mask changed while it was being read. Reload it.')
    labels.setflags(write=False)
    return PrimaryMaskSource(str(source), str(field), first, second,
                             digest.hexdigest(), labels)
