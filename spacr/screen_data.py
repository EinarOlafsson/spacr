"""The published TSG101 screen, as separately downloadable pieces.

WHY PIECES. The screen is four plates of about 8 GB of crops each plus a
half-gigabyte database each -- 33 GB in total. Almost nobody wants all of it:
the regression measurement and cell functions read the DATABASES, and the crops
are only needed when something has to display an image. Shipping it as one
download would make trying one function cost 33 GB.

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
    :param bytes: the archive's size, so the picker can say what a tick costs
        before it is paid.
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
        """What the picker shows."""
        what = ("measurements.db" if self.kind == "measurements"
                else "data/ (object crops)")
        return f"Plate {self.plate} — {what}"

    def is_present(self, folder) -> bool:
        """Whether this piece is already unpacked under ``folder``.

        Checks what the archive WRITES, not the folder it writes into: every
        piece shares one plate directory, so the directory existing says
        nothing about which pieces are in it.
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
                590745600, "measurements/measurements.db"),
    ScreenAsset("plate2-measurements.tar", 2, "measurements",
                565401600, "measurements/measurements.db"),
    ScreenAsset("plate3-measurements.tar", 3, "measurements",
                548567040, "measurements/measurements.db"),
    ScreenAsset("plate4-measurements.tar", 4, "measurements",
                486932480, "measurements/measurements.db"),
    ScreenAsset("plate1-data.tar", 1, "crops", 8_300_000_000, "data"),
    ScreenAsset("plate2-data.tar", 2, "crops", 7_900_000_000, "data"),
    ScreenAsset("plate3-data.tar", 3, "crops", 7_600_000_000, "data"),
    ScreenAsset("plate4-data.tar", 4, "crops", 6_900_000_000, "data"),
)


def assets_for(kind: Optional[str] = None,
               plate: Optional[int] = None) -> List[ScreenAsset]:
    """The pieces matching ``kind`` and ``plate``.

    :param kind: ``"measurements"``, ``"crops"``, or ``None`` for both.
    :param plate: a plate number, or ``None`` for all of them.
    """
    return [a for a in SCREEN_ASSETS
            if (kind is None or a.kind == kind)
            and (plate is None or a.plate == plate)]


def published_archives(repo: str = SCREEN_REPO, *, timeout: float = 8.0):
    """The archive names actually present in ``repo``, or ``None``.

    ``None`` means "could not tell" -- offline, or the hub did not answer --
    and is DELIBERATELY different from an empty set. A caller that treated a
    failed lookup as "nothing is published" would grey out every row and leave
    the user with a picker that offers nothing and explains nothing.

    :param timeout: give up rather than hold a dialog open on a slow network.
    """
    try:
        from huggingface_hub import HfApi

        names = HfApi().list_repo_files(repo, repo_type="dataset",
                                        timeout=timeout)
    except TypeError:
        # Older huggingface_hub has no timeout on this call.
        try:
            from huggingface_hub import HfApi

            names = HfApi().list_repo_files(repo, repo_type="dataset")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not list %s", repo, exc_info=True)
            return None
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not list %s", repo, exc_info=True)
        return None
    return {name for name in names if name.endswith(".tar")}


def total_size(assets) -> int:
    """How many bytes a selection will cost."""
    return sum(int(a.bytes) for a in assets)


def human_size(count: int) -> str:
    """``1234567`` as ``1.2 MB``.

    Decimal units, matching what a download manager and a disk vendor both
    report -- a user comparing this figure with either should not have to
    know which of two conventions each of us picked.
    """
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"
