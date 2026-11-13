"""Read the Cell Painting Gallery's cropped-site layout as OPS input (item 919, alpha).

PERISCOPE (Ramezani et al. 2025, cpg0021) publishes the pixels its
CellProfiler analysis read as one TIFF per channel and site::

    images_corrected_cropped/<plate>_Well<N>/<Channel>/<Channel>_Site_<S>.tiff

Painting channels are named after the stain (``CorrDNA``, ``CorrER``,
``CorrMito``, ``CorrPhalloidin``, ``CorrWGA``); sequencing channels are
``Cycle<CC>_<base>`` for each cycle and base, plus ``Cycle01_DAPI``. The
painting and sequencing crops of a site share one pixel grid and are already
illumination-corrected and aligned, so this reader stacks them as they are:
no stitching, no cycle registration, no rescaling.

The channel and base order is never taken from the folder listing. It is
pinned in the settings (``cpg_phenotype_channels``, ``cpg_sbs_bases``,
``cpg_sbs_dapi``) and written to the run folder, so a rerun from the saved
settings stacks the same planes in the same order.

Typical use::

    from spacr.cpg_cropped import cpg_cropped_index, load_cpg_site

    index = cpg_cropped_index(".../images_corrected_cropped/CP186A_Well1")
    site = load_cpg_site(index, "CP186A", "Well1", 45)
    site["phenotype"].shape     # (5500, 5500, 5), uint16
    site["sbs"].shape           # (12, 5500, 5500, 4), uint16
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: The painting channels in the order item 919 stacks them by default.
DEFAULT_PHENOTYPE_CHANNELS = ("CorrDNA", "CorrPhalloidin", "CorrMito",
                              "CorrER", "CorrWGA")

#: The base of each sequencing channel, in spaCR's ``ops_base_channels``
#: order (CY3, A594, CY5, CY7 carry G, T, A, C).
DEFAULT_SBS_BASES = ("G", "T", "A", "C")

#: The phases this module runs for :func:`spacr.ops_engine.run_ops`.
CPG_PHASES = ("cpg_index", "cpg_merged")

#: The columns of the index table, in order.
INDEX_COLUMNS = ("plate", "well", "well_id", "site", "modality", "cycle",
                 "channel", "base", "path", "shape", "dtype", "size")

_LOG = logging.getLogger(__name__)

_WELL_FOLDER = re.compile(r"^(?P<plate>.+)_Well(?P<number>\d+)$", re.IGNORECASE)
_SITE_FILE = re.compile(r"^(?P<channel>.+)_Site_(?P<site>\d+)\.tiff?$",
                        re.IGNORECASE)
_SBS_CHANNEL = re.compile(r"^Cycle(?P<cycle>\d+)_(?P<base>[A-Za-z]+)$",
                          re.IGNORECASE)
_RESAMPLE = ("none", "nearest", "linear")
_MISSING = ("error", "skip_site")


def parse_well_folder(name: str) -> Dict[str, Any]:
    """Split a ``<plate>_Well<N>`` folder name.

    :param name: the folder's base name, for example ``CP186A_Well1``.
    :returns: ``plate`` (``CP186A``), ``well`` (``Well1``), ``well_id``
        (``W1``, the form spaCR tables use) and ``number`` (1).
    :raises ValueError: when the name has no ``_Well<N>`` ending.
    """
    found = _WELL_FOLDER.match(str(name))
    if not found:
        raise ValueError(
            f"{name!r} is not a <plate>_Well<N> folder; the cropped-site "
            "reader takes the plate and well from that name (for example "
            "CP186A_Well1)")
    number = int(found.group("number"))
    return {"plate": found.group("plate"), "well": f"Well{number}",
            "well_id": f"W{number}", "number": number}


def parse_sites(value: Any) -> Optional[List[int]]:
    """The sites a ``cpg_sites`` setting names, or None for every site.

    :param value: None or empty for all; an int; a list of ints; or a string
        of comma-separated numbers and ``a-b`` ranges (``"1-3,7"``).
    :returns: the sorted site numbers, or None.
    :raises ValueError: for a part that is not a number or a range.
    """
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, str):
        out: List[int] = []
        for part in (p.strip() for p in value.split(",")):
            if not part:
                continue
            if "-" in part:
                start, stop = part.split("-", 1)
                out.extend(range(int(start), int(stop) + 1))
            else:
                out.append(int(part))
        return sorted(set(out))
    return sorted({int(v) for v in value})


def _names(value: Any, default: Sequence[str]) -> List[str]:
    """A list setting given as a list or a comma-separated string."""
    if value is None or value == "" or value == []:
        return list(default)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def _well_folders(root: str) -> List[str]:
    """The ``<plate>_Well<N>`` folders under ``root``, or ``root`` itself.

    :param root: an ``images_corrected_cropped`` folder or one well folder.
    :returns: the well folders, sorted by name.
    :raises ValueError: when ``root`` does not exist, holds channel folders
        under a name without ``_Well<N>``, or holds sub-folders that are not
        well folders.
    """
    if not os.path.isdir(root):
        raise ValueError(f"cpg_root {root!r} is not a folder")
    base = os.path.basename(os.path.normpath(root))
    if _WELL_FOLDER.match(base):
        return [os.path.normpath(root)]
    children = sorted(name for name in os.listdir(root)
                      if os.path.isdir(os.path.join(root, name)))
    wells = [name for name in children if _WELL_FOLDER.match(name)]
    others = [name for name in children if not _WELL_FOLDER.match(name)]
    if others and not wells:
        parse_well_folder(base)
    if others:
        raise ValueError(
            f"{root!r} holds folders that are not <plate>_Well<N> folders: "
            f"{others[:10]}")
    return [os.path.join(root, name) for name in wells]


def _tiff_header(path: str):
    """The first page's shape and dtype of a TIFF, without decoding pixels."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        return tuple(int(v) for v in page.shape), str(page.dtype)


def file_md5(path: str, chunk: int = 1 << 22) -> str:
    """The md5 of a file's bytes.

    :param path: the file.
    :param chunk: bytes read at a time.
    :returns: the hex digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _drop_sbs_only_well(rows: List[Dict[str, Any]], first: int,
                        well: Mapping[str, Any], folder: str,
                        keep_wells: Optional[set]) -> str:
    """Remove a well's rows when it has sequencing images but no phenotype ones.

    Cycle and channel folder names match without regard to case, so the index
    holds ``Cycle<CC>_<BASE>`` in canonical case and the real path. A well
    with sequencing images but no phenotype images (``CP186A_Well6`` in
    cpg0021) is left out of the index with a logged warning; the caller keeps
    its name in ``index.attrs["skipped_wells"]``, which
    :func:`cpg_site_index` passes on as ``skipped_wells`` and the run report
    and ``cpg_channels.json`` record. Naming such a well in ``cpg_wells``
    raises an error that says it has no phenotype images. A well with
    phenotype images and no sequencing images is kept.

    :param rows: the index rows so far; this well's start at ``first``.
    :param first: the position of the well's first row.
    :param well: the parsed well folder name.
    :param folder: the well folder, for the error text.
    :param keep_wells: the wells asked for, or None.
    :returns: ``<plate>_Well<N>`` when the well was removed, else ``""``.
    :raises ValueError: when the removed well was asked for by name.
    """
    added = rows[first:]
    if not added or any(row["modality"] != "sbs" for row in added):
        return ""
    name = f"{well['plate']}_{well['well']}"
    if keep_wells is not None:
        raise ValueError(
            f"well {name} has no phenotype images (only sequencing cycles "
            f"under {folder!r}); it cannot be read as a cropped site. "
            "Remove it from wells= to skip it")
    del rows[first:]
    _LOG.warning("skipping well %s: it has SBS images but no phenotype "
                 "images", name)
    return name


def cpg_cropped_index(root: str, *, wells: Optional[Sequence[str]] = None,
                      sites: Any = None, md5: bool = False,
                      headers: bool = True) -> pd.DataFrame:
    """Index a cropped-site tree from its paths, one row per file.

    :param root: ``images_corrected_cropped/`` or one ``<plate>_Well<N>/``.
    :param wells: keep only these wells, as ``Well1``, ``W1`` or
        ``CP186A_Well1``; None keeps all.
    :param sites: keep only these sites (:func:`parse_sites`).
    :param md5: add an ``md5`` column of each file's bytes.
    :param headers: read each TIFF's header for ``shape`` and ``dtype``. No
        pixels are decoded either way; False leaves both empty.
    :returns: the index (:data:`INDEX_COLUMNS`, plus ``md5``), sorted by
        plate, well, site, modality (phenotype first), cycle and channel.
        ``modality`` is ``phenotype`` or ``sbs``; ``cycle`` is 0 for a
        painting channel; ``base`` is the letter of a sequencing channel,
        ``DAPI`` for its nuclear channel, and empty for a painting channel.
    :raises ValueError: for a well folder without ``_Well<N>``, a file whose
        name does not match its channel folder, or a tree with no site files.
    """
    keep_wells = None
    if wells:
        keep_wells = set()
        for name in _names(wells, ()):
            keep_wells.add(name)
            if _WELL_FOLDER.match(name):
                keep_wells.add(parse_well_folder(name)["well"])
            elif re.fullmatch(r"[Ww](?:ell)?(\d+)", name):
                keep_wells.add(f"Well{int(re.sub(r'[^0-9]', '', name))}")
    keep_sites = parse_sites(sites)
    keep_sites = None if keep_sites is None else set(keep_sites)
    rows: List[Dict[str, Any]] = []
    skipped_wells: List[str] = []
    for folder in _well_folders(root):
        well = parse_well_folder(os.path.basename(folder))
        if keep_wells is not None and well["well"] not in keep_wells:
            continue
        first = len(rows)
        for channel in sorted(os.listdir(folder)):
            channel_dir = os.path.join(folder, channel)
            if not os.path.isdir(channel_dir):
                continue
            sbs = _SBS_CHANNEL.match(channel)
            for name in sorted(os.listdir(channel_dir)):
                found = _SITE_FILE.match(name)
                if not found:
                    continue
                if found.group("channel").casefold() != channel.casefold():
                    raise ValueError(
                        f"{os.path.join(channel_dir, name)} is in the "
                        f"{channel} folder but names channel "
                        f"{found.group('channel')}")
                site = int(found.group("site"))
                if keep_sites is not None and site not in keep_sites:
                    continue
                path = os.path.join(channel_dir, name)
                rows.append({
                    "plate": well["plate"], "well": well["well"],
                    "well_id": well["well_id"], "site": site,
                    "modality": "sbs" if sbs else "phenotype",
                    "cycle": int(sbs.group("cycle")) if sbs else 0,
                    "channel": (f"Cycle{int(sbs.group('cycle')):02d}_"
                                f"{sbs.group('base').upper()}") if sbs else channel,
                    "base": sbs.group("base").upper() if sbs else "",
                    "path": path, "size": int(os.path.getsize(path))})
        name = _drop_sbs_only_well(rows, first, well, folder, keep_wells)
        if name:
            skipped_wells.append(name)
    if not rows:
        raise ValueError(f"no <Channel>_Site_<S>.tiff files under {root!r}"
                         + (f" (skipped wells without phenotype images: "
                            f"{skipped_wells})" if skipped_wells else ""))
    frame = pd.DataFrame(rows)
    if headers:
        found = [_tiff_header(path) for path in frame["path"]]
        frame["shape"] = ["x".join(str(v) for v in shape) for shape, _ in found]
        frame["dtype"] = [dtype for _, dtype in found]
    else:
        frame["shape"] = ""
        frame["dtype"] = ""
    frame["_order"] = (frame["modality"] != "phenotype").astype(int)
    frame = frame.sort_values(["plate", "well", "site", "_order", "cycle",
                               "channel"], kind="mergesort")
    frame = frame[list(INDEX_COLUMNS)].reset_index(drop=True)
    if md5:
        frame["md5"] = [file_md5(path) for path in frame["path"]]
    frame.attrs["skipped_wells"] = list(skipped_wells)
    return frame


def expected_channels(phenotype_channels: Sequence[str] = DEFAULT_PHENOTYPE_CHANNELS,
                      sbs_cycles: int = 12,
                      sbs_bases: Sequence[str] = DEFAULT_SBS_BASES,
                      sbs_dapi: Optional[str] = "Cycle01_DAPI") -> List[str]:
    """Every channel folder a complete site has, in stacking order.

    :param phenotype_channels: the painting channels.
    :param sbs_cycles: how many sequencing cycles; 0 for none.
    :param sbs_bases: the base letters of each cycle.
    :param sbs_dapi: the sequencing nuclear channel, or None.
    :returns: the painting channels, ``Cycle<CC>_<base>`` for every cycle
        and base, then the nuclear channel.
    """
    out = list(phenotype_channels)
    out += [f"Cycle{cycle:02d}_{base}" for cycle in range(1, int(sbs_cycles) + 1)
            for base in sbs_bases]
    if sbs_dapi:
        out.append(str(sbs_dapi))
    return out


def cpg_missing_files(index: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    """The (site, channel) files a well lacks.

    A site counts as present when any of its channels is; each present site
    must then hold every channel in ``channels``.

    :param index: from :func:`cpg_cropped_index`.
    :param channels: from :func:`expected_channels`.
    :returns: one row per missing file: ``plate``, ``well``, ``site``,
        ``channel`` and the ``expected`` file name.
    """
    have = set(zip(index["plate"], index["well"], index["site"], index["channel"]))
    rows = []
    for (plate, well, site) in sorted(set(zip(index["plate"], index["well"],
                                              index["site"]))):
        for channel in channels:
            if (plate, well, site, channel) not in have:
                rows.append({"plate": plate, "well": well, "site": int(site),
                             "channel": channel,
                             "expected": f"{channel}/{channel}_Site_{int(site)}.tiff"})
    return pd.DataFrame(rows, columns=["plate", "well", "site", "channel",
                                       "expected"])


def check_cpg_complete(index: pd.DataFrame, channels: Sequence[str], *,
                       missing: str = "error") -> Dict[str, Any]:
    """Refuse an incomplete well, or drop its incomplete sites when asked.

    :param index: from :func:`cpg_cropped_index`.
    :param channels: from :func:`expected_channels`.
    :param missing: ``'error'`` raises naming every missing file;
        ``'skip_site'`` drops each incomplete site and lists it.
    :returns: ``index`` (only complete sites, every file of them), ``missing`` (the table of
        :func:`cpg_missing_files`) and ``skipped`` (``[(plate, well, site)]``).
    :raises ValueError: for an unknown ``missing`` or, with ``'error'``, a
        missing file.
    """
    if missing not in _MISSING:
        raise ValueError(f"cpg_missing must be one of {list(_MISSING)}; got {missing!r}")
    absent = cpg_missing_files(index, channels)
    if len(absent) and missing == "error":
        listed = [f"{r.plate}_{r.well} site {r.site}: {r.expected}"
                  for r in absent.head(20).itertuples()]
        more = f" and {len(absent) - 20} more" if len(absent) > 20 else ""
        raise ValueError(f"{len(absent)} files missing: " + "; ".join(listed) + more)
    skipped = sorted({(r.plate, r.well, int(r.site)) for r in absent.itertuples()})
    keep = index
    if skipped:
        drop = set(skipped)
        mask = [(p, w, int(s)) not in drop for p, w, s in
                zip(index["plate"], index["well"], index["site"])]
        keep = index[mask].reset_index(drop=True)
    return {"index": keep, "missing": absent, "skipped": skipped}


def _site_rows(index: pd.DataFrame, plate: str, well: str, site: int) -> Dict[str, str]:
    """``channel -> path`` for one site."""
    well_name = well
    if re.fullmatch(r"[Ww](\d+)", str(well)):
        well_name = f"Well{int(str(well)[1:])}"
    rows = index[(index["plate"] == plate) & (index["well"] == well_name)
                 & (index["site"] == int(site))]
    return dict(zip(rows["channel"], rows["path"]))


def _resample(plane: np.ndarray, shape, method: str) -> np.ndarray:
    """Resize ``plane`` onto ``shape`` keeping its dtype."""
    from skimage.transform import resize

    order = 0 if method == "nearest" else 1
    out = resize(plane, shape, order=order, preserve_range=True,
                 anti_aliasing=False, mode="edge")
    if np.issubdtype(plane.dtype, np.integer):
        info = np.iinfo(plane.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(plane.dtype)


def load_cpg_site(index: pd.DataFrame, plate: str, well: str, site: int, *,
                  phenotype_channels: Sequence[str] = DEFAULT_PHENOTYPE_CHANNELS,
                  sbs_cycles: int = 12,
                  sbs_bases: Sequence[str] = DEFAULT_SBS_BASES,
                  sbs_dapi: Optional[str] = "Cycle01_DAPI",
                  resample: str = "none",
                  parts: Sequence[str] = ("phenotype", "sbs")) -> Dict[str, Any]:
    """One site's painting field and sequencing stack, exactly as stored.

    Each plane is ``tifffile.imread`` of its file, unchanged: same dtype, no
    rescaling, no registration.

    :param index: from :func:`cpg_cropped_index`.
    :param plate: the plate, for example ``CP186A``.
    :param well: ``Well1`` or ``W1``.
    :param site: the site number.
    :param phenotype_channels: painting channels, in stacking order.
    :param sbs_cycles: sequencing cycles to stack, 1 to this.
    :param sbs_bases: the base of each of the four planes per cycle, in order.
    :param sbs_dapi: the sequencing nuclear channel, or None.
    :param resample: what to do when a sequencing plane's grid differs from
        the painting grid: ``'none'`` refuses; ``'nearest'`` or ``'linear'``
        resizes it onto the painting grid and records the factor.
    :param parts: which of ``'phenotype'`` and ``'sbs'`` to read.
    :returns: ``phenotype`` ``(Y, X, C)``, ``sbs`` ``(cycles, Y, X, bases)``,
        ``dapi`` ``(Y, X)`` (each None when not read), ``channels``,
        ``sbs_channels`` (``[cycle][base]`` names), ``resample_factor``
        (``(fy, fx)`` from sequencing to painting grid, ``(1.0, 1.0)`` when
        they agree) and ``paths``.
    :raises ValueError: for a missing channel, planes of one modality that
        differ in shape, a grid mismatch with ``resample='none'``, or an
        unknown ``resample``.
    """
    import tifffile

    if resample not in _RESAMPLE:
        raise ValueError(f"cpg_sbs_resample must be one of {list(_RESAMPLE)}; "
                         f"got {resample!r}")
    files = _site_rows(index, plate, well, site)
    sbs_names = [[f"Cycle{cycle:02d}_{base}" for base in sbs_bases]
                 for cycle in range(1, int(sbs_cycles) + 1)]
    wanted: List[str] = []
    if "phenotype" in parts:
        wanted += list(phenotype_channels)
    if "sbs" in parts:
        wanted += [name for names in sbs_names for name in names]
        if sbs_dapi:
            wanted.append(str(sbs_dapi))
    absent = [name for name in wanted if name not in files]
    if absent:
        raise ValueError(f"{plate} {well} site {site} has no file for "
                         f"{absent[:10]}")

    def read(name: str) -> np.ndarray:
        """Read one channel's plane as stored."""
        return tifffile.imread(files[name])

    out: Dict[str, Any] = {"phenotype": None, "sbs": None, "dapi": None,
                           "channels": list(phenotype_channels),
                           "sbs_channels": sbs_names,
                           "resample_factor": (1.0, 1.0),
                           "paths": {name: files[name] for name in wanted}}
    grid = None
    if "phenotype" in parts:
        planes = [read(name) for name in phenotype_channels]
        shapes = {plane.shape for plane in planes}
        if len(shapes) > 1:
            raise ValueError(f"{plate} {well} site {site}: painting channels "
                             f"differ in shape {sorted(shapes)}")
        out["phenotype"] = np.stack(planes, axis=-1)
        grid = planes[0].shape
    if "sbs" in parts:
        planes = [[read(name) for name in names] for names in sbs_names]
        dapi = read(str(sbs_dapi)) if sbs_dapi else None
        shapes = {p.shape for names in planes for p in names} | (
            {dapi.shape} if dapi is not None else set())
        if len(shapes) > 1:
            raise ValueError(f"{plate} {well} site {site}: sequencing planes "
                             f"differ in shape {sorted(shapes)}")
        cycles = [np.stack(names, axis=-1) for names in planes]
        sbs_shape = next(iter(shapes)) if shapes else grid
        if grid is not None and sbs_shape != grid:
            if resample == "none":
                raise ValueError(
                    f"{plate} {well} site {site}: the sequencing grid "
                    f"{sbs_shape} differs from the painting grid {grid}; set "
                    "cpg_sbs_resample to 'nearest' or 'linear' to resize it "
                    "(track P1 must run with 'none')")
            out["resample_factor"] = (grid[0] / sbs_shape[0], grid[1] / sbs_shape[1])
            cycles = [np.stack([_resample(c[..., k], grid, resample)
                                for k in range(c.shape[-1])], axis=-1)
                      for c in cycles]
            if dapi is not None:
                dapi = _resample(dapi, grid, resample)
        out["sbs"] = np.stack(cycles, axis=0) if cycles else None
        out["dapi"] = dapi
    return out


def array_md5(array: np.ndarray) -> str:
    """The md5 of an array's bytes in C order.

    :param array: any array.
    :returns: the hex digest.
    """
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def cpg_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The pinned reader settings, with defaults filled in.

    :param settings: the OPS settings; read for ``cpg_root``, ``cpg_wells``,
        ``cpg_sites``, ``cpg_phenotype_channels``, ``cpg_sbs_cycles``,
        ``cpg_sbs_bases``, ``cpg_sbs_dapi``, ``cpg_sbs_resample``,
        ``cpg_missing`` and ``cpg_md5``.
    :returns: the same keys, normalised (lists of names, ints, lower-case
        choices).
    :raises ValueError: when ``cpg_root`` is empty.
    """
    root = settings.get("cpg_root")
    if not root:
        raise ValueError("ops_layout 'cpg_cropped' needs cpg_root: an "
                         "images_corrected_cropped folder or one <plate>_Well<N> folder")
    dapi = settings.get("cpg_sbs_dapi", "Cycle01_DAPI")
    return {
        "cpg_root": str(root),
        "cpg_wells": _names(settings.get("cpg_wells"), ()) or None,
        "cpg_sites": parse_sites(settings.get("cpg_sites")),
        "cpg_phenotype_channels": _names(settings.get("cpg_phenotype_channels"),
                                         DEFAULT_PHENOTYPE_CHANNELS),
        "cpg_sbs_cycles": int(settings.get("cpg_sbs_cycles", 12) or 0),
        "cpg_sbs_bases": [b.upper() for b in _names(settings.get("cpg_sbs_bases"),
                                                    DEFAULT_SBS_BASES)],
        "cpg_sbs_dapi": str(dapi) if dapi else None,
        "cpg_sbs_resample": str(settings.get("cpg_sbs_resample") or "none").lower(),
        "cpg_missing": str(settings.get("cpg_missing") or "error").lower(),
        "cpg_md5": bool(settings.get("cpg_md5", False)),
    }


def cpg_site_index(settings: Mapping[str, Any], *, md5: Optional[bool] = None
                   ) -> Dict[str, Any]:
    """Index, check and pin a cropped-site run's input.

    :param settings: the OPS settings (:func:`cpg_settings`).
    :param md5: overrides ``cpg_md5``.
    :returns: ``index`` (the complete sites), ``missing``,
        ``skipped``, ``channels`` (every expected channel, in order) and
        ``pinned`` (:func:`cpg_settings`).
    """
    pinned = cpg_settings(settings)
    index = cpg_cropped_index(pinned["cpg_root"], wells=pinned["cpg_wells"],
                              sites=pinned["cpg_sites"],
                              md5=pinned["cpg_md5"] if md5 is None else md5)
    channels = expected_channels(pinned["cpg_phenotype_channels"],
                                 pinned["cpg_sbs_cycles"], pinned["cpg_sbs_bases"],
                                 pinned["cpg_sbs_dapi"])
    checked = check_cpg_complete(index, channels, missing=pinned["cpg_missing"])
    return {**checked, "channels": channels, "pinned": pinned,
            "skipped_wells": list(index.attrs.get("skipped_wells", []))}


def _write_json(path: str, record: Mapping[str, Any]) -> None:
    """Write ``record`` as indented JSON through a ``.part`` rename."""
    with open(path + ".part", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    os.replace(path + ".part", path)


def merged_path(root: str, plate: str, well_id: str, site: int) -> str:
    """One site's ``merged/phenotype/<plate>/<well_id>/<site>.npz``.

    :param root: the run folder.
    :param plate: the plate.
    :param well_id: the normalised well id, for example ``W1``.
    :param site: the site.
    :returns: the path.
    """
    return os.path.join(root, "merged", "phenotype", str(plate), str(well_id),
                        f"{int(site)}.npz")


def _merge_site(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Write one site's painting stack to ``merged/``; the ``cpg_merged`` worker."""
    loaded = load_cpg_site(task["index"], task["plate"], task["well"], task["site"],
                           phenotype_channels=task["channels"], parts=("phenotype",))
    stack = loaded["phenotype"]
    meta = {"plate": task["plate"], "well": task["well"], "well_id": task["well_id"],
            "site": int(task["site"]), "layout": "cpg_cropped",
            "channels": list(task["channels"]), "paths": loaded["paths"],
            "md5": {name: array_md5(stack[..., k])
                    for k, name in enumerate(task["channels"])},
            "spacr_version": task["version"]}
    os.makedirs(os.path.dirname(task["out_path"]), exist_ok=True)
    part = task["out_path"] + ".part.npz"
    np.savez_compressed(part, channels=np.moveaxis(stack, -1, 0),
                        meta=np.array(json.dumps(meta, default=str)))
    os.replace(part, task["out_path"])
    return {"site": int(task["site"]), "shape": list(stack.shape),
            "dtype": str(stack.dtype), "md5": meta["md5"]}


def run_cpg_phases(settings: Mapping[str, Any], phases: Sequence[str] = CPG_PHASES,
                   report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Index a cropped-site tree and write its sites to ``merged/`` (item 919).

    ``cpg_index`` builds the index from the paths, checks every site has every
    pinned channel and cycle, and writes ``<dst_root>/cpg_index.csv`` (one row
    per file, with ``md5`` when ``cpg_md5`` is on) and
    ``<dst_root>/cpg_channels.json`` (the pinned channel and base order and
    every other reader setting). ``cpg_merged`` writes each complete site's
    painting channels, stacked in the pinned order and exactly as stored, to
    ``<dst_root>/merged/phenotype/<plate>/<well_id>/<site>.npz`` (``channels``
    as ``(C, Y, X)`` and a JSON ``meta`` with each plane's md5); masks are
    added by the segmentation items. A site whose file exists is not redone.
    The sequencing stack is not copied: the decode reads it from the index.

    :param settings: the OPS settings: ``dst_root`` (required), the
        ``cpg_*`` reader settings (:func:`cpg_settings`) and ``n_workers``.
    :param phases: ``"cpg_index"``, ``"cpg_merged"`` or both.
    :param report: called with progress lines.
    :returns: ``{"index": path, "channels_record": path, "files", "sites",
        "skipped", "merged": {...}}``.
    :raises ValueError: without ``dst_root``, for an incomplete well under
        ``cpg_missing='error'``, or for an unknown phase.
    """
    from concurrent.futures import ThreadPoolExecutor

    from . import __version__
    from .tabular import write_table

    say = report or (lambda text: None)
    unknown = [phase for phase in phases if phase not in CPG_PHASES]
    if unknown:
        raise ValueError(f"unknown cropped-site phases {unknown}")
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError("the cropped-site phases need dst_root")
    destination = str(destination)
    os.makedirs(destination, exist_ok=True)
    started = time.perf_counter()
    built = cpg_site_index(settings)
    index, pinned = built["index"], built["pinned"]
    result: Dict[str, Any] = {
        "index": os.path.join(destination, "cpg_index.csv"),
        "channels_record": os.path.join(destination, "cpg_channels.json"),
        "files": int(len(index)),
        "sites": int(index[["plate", "well", "site"]].drop_duplicates().shape[0]),
        "skipped": [list(s) for s in built["skipped"]],
        "skipped_wells": list(built["skipped_wells"])}
    if "cpg_index" in phases:
        write_table(index, result["index"], canonicalise=False)
        if len(built["missing"]):
            write_table(built["missing"], os.path.join(destination, "cpg_missing.csv"),
                        canonicalise=False)
        _write_json(result["channels_record"], {
            "layout": "cpg_cropped", "phenotype_channels": pinned["cpg_phenotype_channels"],
            "sbs_cycles": pinned["cpg_sbs_cycles"], "sbs_bases": pinned["cpg_sbs_bases"],
            "sbs_dapi": pinned["cpg_sbs_dapi"], "sbs_resample": pinned["cpg_sbs_resample"],
            "missing": pinned["cpg_missing"], "md5": pinned["cpg_md5"],
            "registration": "none: the crops are already aligned",
            "stacking_order": built["channels"], "settings": pinned,
            "skipped_wells": built["skipped_wells"],
            "spacr_version": __version__})
        say(f"cpg_index: {result['files']} files, {result['sites']} sites")
    if "cpg_merged" in phases:
        sites = index[["plate", "well", "well_id", "site"]].drop_duplicates()
        tasks = []
        for row in sites.itertuples(index=False):
            out_path = merged_path(destination, row.plate, row.well_id, row.site)
            if os.path.isfile(out_path):
                continue
            tasks.append({"index": index, "plate": row.plate, "well": row.well,
                          "well_id": row.well_id, "site": int(row.site),
                          "channels": pinned["cpg_phenotype_channels"],
                          "out_path": out_path, "version": __version__})
        os.makedirs(os.path.join(destination, "merged"), exist_ok=True)
        _write_json(os.path.join(destination, "merged", "LAYOUT.json"), {
            "layout": "ops_merged_v1", "source": "cpg_cropped",
            "channels": pinned["cpg_phenotype_channels"], "masks": [],
            "dtype": "as stored", "axes": "CYX"})
        workers = max(1, min(int(settings.get("n_workers") or 1), 4, len(tasks) or 1))
        if workers == 1:
            written = [_merge_site(task) for task in tasks]
        else:
            with ThreadPoolExecutor(workers) as pool:
                written = list(pool.map(_merge_site, tasks))
        result["merged"] = {"written": len(written), "resumed": int(len(sites) - len(tasks)),
                            "sites": written}
        say(f"cpg_merged: {len(written)} sites written")
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result
