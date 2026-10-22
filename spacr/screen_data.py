"""The published TSG101 screen, as separately downloadable pieces.

The screen is four plates of about 8 GB of crops each plus a half-gigabyte
database each -- 33 GB in total. The regression, measurement and cell
functions read only the databases; the crops are needed only to display
images. A single archive would require a 33 GB download to try one
function.

So each plate's database and each plate's crop folder is its own archive, and
:class:`ScreenAsset` says how big each one is BEFORE it is fetched. The picker
in the Regression screen lists them with their sizes and downloads only what is
selected.

The `merged/` folders are deliberately absent. They are about 300 GB per plate
-- 1.2 TB for the screen -- which is past what a public dataset host will take
without arrangement, and nothing in Regression reads them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("spacr.screen_data")

__all__ = [
    "SCREEN_REPO",
    "ScreenAsset",
    "SCREEN_ASSETS",
    "assets_for",
    "published_archives",
    "human_size",
    "total_size",
]

#: Where the screen pieces are published.
SCREEN_REPO = "einarolafsson/spacr-example-screen"


@dataclass(frozen=True)
class ScreenAsset:
    """One separately downloadable piece of the screen.

    :param archive: the ``.tar`` in :data:`SCREEN_REPO`.
    :param plate: which plate it belongs to.
    :param kind: ``"measurements"`` or ``"crops"``.
    :param bytes: the archive's size in bytes, shown by the picker before
        the archive is downloaded.
    :param unpacks_to: the path inside the plate folder the archive fills, used
        to tell an already-downloaded piece from one that is missing.
    """

    archive: str
    plate: int
    kind: str
    bytes: int
    unpacks_to: str

    @property
    def label(self) -> str:
        """Return the plate-and-content label shown by the data picker.

        :returns: human-readable plate and archive-content label.
        """
        what = ("measurements.db" if self.kind == "measurements"
                else "data/ (object crops)")
        return f"Plate {self.plate} — {what}"

    def is_present(self, folder) -> bool:
        """Whether this piece is already unpacked under ``folder``.

        Checks the files the archive writes, not the folder it writes into:
        every piece shares one plate directory, so the directory's existence
        does not show which pieces are in it.

        :param folder: plate directory under which this asset would unpack.
        :returns: whether the database file exists, or the crop directory
            contains at least one PNG, according to this asset's ``kind``.
        """
        from pathlib import Path

        target = Path(folder) / self.unpacks_to
        if self.kind == "measurements":
            return target.is_file()
        return target.is_dir() and any(target.rglob("*.png"))


#: Every piece, in the order the picker lists them.
#:
#: Sizes are the ARCHIVE's, measured at publication. They are approximate for
#: the crop folders -- tar adds a little per member and 60,000 members is not
#: nothing -- and exact enough for a user deciding whether to spend the
#: download.
SCREEN_ASSETS: Tuple[ScreenAsset, ...] = (
    ScreenAsset("plate1-measurements.tar", 1, "measurements",
                555673600, "measurements/measurements.db"),
    ScreenAsset("plate2-measurements.tar", 2, "measurements",
                530165760, "measurements/measurements.db"),
    ScreenAsset("plate3-measurements.tar", 3, "measurements",
                513576960, "measurements/measurements.db"),
    ScreenAsset("plate4-measurements.tar", 4, "measurements",
                455454720, "measurements/measurements.db"),
    ScreenAsset("plate1-data.tar", 1, "crops", 8874557440, "data"),
    ScreenAsset("plate2-data.tar", 2, "crops", 8403578880, "data"),
    ScreenAsset("plate3-data.tar", 3, "crops", 8040058880, "data"),
    ScreenAsset("plate4-data.tar", 4, "crops", 7375370240, "data"),
)


def assets_for(kind: Optional[str] = None,
               plate: Optional[int] = None) -> List[ScreenAsset]:
    """The pieces matching ``kind`` and ``plate``.

    :param kind: ``"measurements"``, ``"crops"``, or ``None`` for both.
    :param plate: a plate number, or ``None`` for all of them.
    :returns: assets matching both supplied filters, in picker order.
    """
    return [a for a in SCREEN_ASSETS
            if (kind is None or a.kind == kind)
            and (plate is None or a.plate == plate)]


def published_archives(repo: str = SCREEN_REPO, *, timeout: float = 8.0):
    """The archive names actually present in ``repo``, or ``None``.

    ``None`` means the lookup failed -- offline, or the hub did not answer --
    and is distinct from an empty set, which means the repository has no
    archives. A caller that treated a failed lookup as an empty set would
    disable every row of the picker without explanation.

    :param repo: Hugging Face dataset repository to inspect.
    :param timeout: give up rather than hold a dialog open on a slow network.
    :returns: names of published ``.tar`` archives, or ``None`` when the
        repository could not be inspected.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo, repo_type="dataset", timeout=timeout)
        return {sibling.rfilename for sibling in info.siblings
                if sibling.rfilename.endswith(".tar")}
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not list %s", repo, exc_info=True)
        return None


def total_size(assets) -> int:
    """Return how many bytes a selection will cost.

    :param assets: iterable of :class:`ScreenAsset` objects.
    :returns: sum of the assets' published archive sizes in bytes.
    """
    return sum(int(a.bytes) for a in assets)


def human_size(count: int) -> str:
    """Format a byte count with decimal units, e.g. ``1234567`` as ``1.2 MB``.

    Decimal units (1 KB = 1000 B) match what download managers and disk
    vendors report.

    :param count: byte count to format.
    :returns: decimal-unit size from bytes through terabytes, with one decimal
        above the byte unit.
    """
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"
