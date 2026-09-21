"""A sample of every dataset spaCR has a model for, one dialog away in Make Masks.

The maintainer, 2026-09-20: "for the generate masks modular, there should be multiple
datasets for the user to choose from, a small sample of each dataset a model is trained
on where I have the data on my huggingface."

Item 412 gave Make Masks ONE button and ONE dataset -- ten Toxoplasma vacuole fields.
This is that idea for the rest of the zoo: every published model was trained on a dataset
that is already on Hugging Face, and a user who wants to see what a model was taught
should be able to open ten of its fields without knowing a repository name.

NO NEW REPOSITORIES, AND NO ARCHIVES. Item 412's set is a purpose-built repo with a
single tar. The training datasets are not: they are 556 to 6,062 files of `images/` and
`masks/`, and nobody is downloading 3,029 fields to look at ten. Each sample is fetched
FILE BY FILE with `hf_hub_download`, which is why :data:`MASK_DATASETS` carries the two
folder names per repo rather than assuming them -- `cross-channel-toxoplasma-from-cellmask`
calls its masks `masks_pv`, and assuming `masks` would have given ten images and no
labels with no error to explain it.

WHICH FIELDS. A random draw in which every field of the dataset has the same chance,
at the maintainer's instruction of 2026-09-21 -- the first N by name are the first plate
of the first domain, which is not the dataset. SEEDED by the dataset key, so the draw is
the same on every machine: a sample that changes between two people's machines is a
sample nobody can talk about -- "the third field looks wrong" has to mean the same field
for both of them. A dataset with a quota draws that way within each domain.

THE LAYOUT IS MAKE MASKS' OWN: images at the top of the folder, masks in `masks/`
beneath them, which is what `curation_queue.detect_layout` calls `nested` and what the
maintainer's rule of 2026-09-20 puts everywhere. So a sample opens for EDITING, with the
published masks as the drafts -- deliberately unlike item 412's set, which hides its
truth in `ground_truth_masks/` so the fields open raw. Those are two different jobs: 412
is "try segmenting this", and this is "look at what the model was taught".

THE WELL-DETECTOR DATASET IS NOT HERE. `toxoplasma-plaque-well-detector-dataset` has
`images/` and `labels/`, and those labels are YOLO bounding boxes. A box is not a mask
and Make Masks would open every field blank.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout)

from .i18n import tr

LOG = logging.getLogger(__name__)

#: How many fields a sample holds. Ten is item 412's number and there is no reason to
#: differ; it is enough to see a domain and small enough to fetch over a hotel wifi.
SAMPLE_SIZE = 10

#: How samples are drawn, in the cache folder's name. ``random1``: a seeded
#: random draw over the whole dataset, or within each domain of a quota
#: (2026-09-21). Samples cached under the first-N rule sit under other names
#: and are not reopened.
SAMPLE_RULE = "random1"


@dataclass(frozen=True)
class MaskDataset:
    """One published dataset, and how to take a sample of it.

    :ivar key: the stable name this is stored and tested under.
    :ivar title: what the picker shows.
    :ivar repo: the Hugging Face dataset repository.
    :ivar images: the folder inside it holding the fields.
    :ivar masks: the folder holding their labels. NOT assumed to be "masks".
    :ivar model: the model this dataset trained, for the line under the title.
    :ivar note: what a reader should know before opening it.
    :ivar apps: which modules offer this set. A dataset belongs to the module whose
        job it illustrates, and the plaque sets belong to two -- Make Masks, where a
        curator edits the masks, and Plaque Analysis, where the pipeline runs on them.
    :ivar quota: how many fields to take from each domain, as ``(domain, count)``,
        a domain being the ``<domain>__`` prefix of a file name. Empty means
        :data:`SAMPLE_SIZE` fields from the whole set.
    """

    key: str
    title: str
    repo: str
    images: str
    masks: str
    model: str
    note: str = ""
    apps: Tuple[str, ...] = ("mask",)
    quota: Tuple[Tuple[str, int], ...] = ()

    @property
    def size(self) -> int:
        """How many fields a complete sample of this dataset holds."""
        return sum(n for _d, n in self.quota) if self.quota else SAMPLE_SIZE


MASK_DATASETS: Tuple[MaskDataset, ...] = (
    MaskDataset(
        key="toxoplasma_pv",
        title="Toxoplasma parasitophorous vacuoles",
        repo="einarolafsson/toxoplasma-pv-segmentation-dataset",
        images="images", masks="masks",
        model="cpsam_v2_toxo (PV segmentation)",
        note="556 merged fluorescence fields, curated twice in September 2026."),
    MaskDataset(
        key="toxoplasma_plaque",
        title="Toxoplasma plaque assays",
        repo="einarolafsson/toxoplasma-plaque-dataset",
        images="images", masks="masks",
        model="cpsam_plaque_r5 (plaque segmentation)",
        note="20 of the 488 curated v5 fields that trained cpsam_plaque_r5: 8 patrick, "
             "4 bigbean, 4 malnio, 4 literature. The objects are plaques, not cells.",
        apps=("mask", "analyze_plaques"),
        quota=(("patrick", 8), ("bigbean", 4), ("malnio", 4), ("literature", 4))),
    MaskDataset(
        key="plaque_figures",
        title="Plaque assay figures, whole plates",
        repo="einarolafsson/toxoplasma-plaque-well-detector-dataset",
        images="images", masks="",
        model="yolo well detector, then cpsam_plaque",
        note="Whole figures, no masks: this is what the Plaque Analysis pipeline "
             "takes as input -- find the wells, read the text, then segment.",
        apps=("analyze_plaques",)),
    MaskDataset(
        key="cell_from_hoechst",
        title="Cross-channel: cell from Hoechst",
        repo="einarolafsson/cross-channel-cell-from-hoechst",
        images="images", masks="masks",
        model="cross-channel-cell-from-hoechst",
        note="A nuclear stain in, a whole-cell mask out."),
    MaskDataset(
        key="nuclei_from_cellmask",
        title="Cross-channel: nuclei from CellMask",
        repo="einarolafsson/cross-channel-nuclei-from-cellmask",
        images="images", masks="masks",
        model="cross-channel-nuclei-from-cellmask",
        note="A whole-cell stain in, nuclei out."),
    MaskDataset(
        key="toxoplasma_from_cellmask",
        title="Cross-channel: Toxoplasma from CellMask",
        repo="einarolafsson/cross-channel-toxoplasma-from-cellmask",
        images="images", masks="masks_pv",
        model="cross-channel-toxoplasma-from-cellmask",
        note="Its masks folder is masks_pv, not masks."),
)

DATASETS_BY_KEY: Dict[str, MaskDataset] = {d.key: d for d in MASK_DATASETS}


def datasets_for(app_key: str) -> Tuple[MaskDataset, ...]:
    """The sets a given module offers.

    :param app_key: the module, e.g. ``mask`` or ``analyze_plaques``.
    :returns: its datasets, in registry order.
    """
    return tuple(d for d in MASK_DATASETS if app_key in d.apps)


def examples_root() -> Path:
    """Where spaCR keeps downloaded example data.

    The same ``~/.cache/spacr/example_data`` that item 412's set unpacks beside, so a
    user who clears one clears both and there is one place to look.

    :returns: the folder. It is not created here.
    """
    return Path.home() / ".cache" / "spacr" / "example_data"


def sample_folder(root, dataset: MaskDataset) -> Path:
    """Where ``dataset``'s sample is cached.

    :param root: the folder example data is kept in.
    :param dataset: the dataset.
    :returns: ``<root>/mask_datasets/<key>_<rule>``, the rule naming how the
        sample was drawn, so a copy drawn under an older rule is never opened
        as if it were this one.
    """
    return Path(root) / "mask_datasets" / f"{dataset.key}_{SAMPLE_RULE}"


def is_present(folder, expected: int = SAMPLE_SIZE) -> bool:
    """Whether a complete sample is already unpacked in ``folder``.

    A half-downloaded sample reads as ABSENT, which is the answer that fetches the rest
    of it rather than opening a folder with four fields in it and saying nothing.

    :param folder: the sample folder.
    :param expected: how many pairs a complete sample has.
    :returns: whether that many image/mask pairs are there.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False
    images = [p for p in sorted(folder.iterdir())
              if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")]
    masks = folder / "masks"
    if not masks.is_dir():
        return len(images) >= expected
    paired = [p for p in images if (masks / f"{p.stem}.tif").is_file()
              or (masks / p.name).is_file()]
    return len(paired) >= expected


def _random_pick(stems: List[str], count: int, seed: str) -> List[str]:
    """``count`` stems drawn at random, each with the same chance, reproducibly.

    The maintainer, 2026-09-21: "make sure each is a random selection of the
    dataset so everything in the dataset has the same chance to be included."
    The first N by name are the first plate of the first domain, which is not
    the dataset. The draw is seeded by ``seed`` -- the dataset key and the
    domain -- so two machines, and two openings on one, get the same sample.

    :param stems: every candidate stem.
    :param count: how many to take.
    :param seed: what the draw is seeded with.
    :returns: the picked stems, sorted.
    """
    import random

    pool = sorted(stems)
    if count >= len(pool):
        return pool
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return sorted(random.Random(int(digest[:16], 16)).sample(pool, count))


def choose_sample(dataset: MaskDataset, listing: List[str],
                  size: int = SAMPLE_SIZE) -> List[Tuple[str, str]]:
    """Pick which files a sample holds, as ``(image path, mask path)`` in the repo.

    A field is only taken when BOTH its image and its mask are in the listing, by stem.
    Pairing on the stem rather than the full name is what lets a repo store
    ``images/x.tif`` beside ``masks/x.tif`` or ``masks_pv/x.tif`` and still pair.

    A DATASET WITH NO MASKS FOLDER -- ``masks=""`` -- yields images alone, with an empty
    mask path. That is the plaque FIGURES set: the Plaque Analysis pipeline takes a
    figure and finds the wells itself, so there is nothing to pair and demanding a pair
    would return an empty sample and no reason why.

    :param dataset: which dataset, for its two folder names.
    :param listing: every path in the repository.
    :param size: how many pairs to take. Ignored when the dataset has a
        :attr:`MaskDataset.quota`, which says how many per domain instead.
    :returns: up to ``size`` pairs drawn by :func:`_random_pick`, sorted, the
        same on every machine.
    """
    prefix_i = f"{dataset.images}/"
    images = {Path(p).stem: p for p in listing if p.startswith(prefix_i)}
    if not dataset.masks:
        return [(images[stem], "") for stem in
                _random_pick(list(images), size, dataset.key)]
    prefix_m = f"{dataset.masks}/"
    masks = {Path(p).stem: p for p in listing if p.startswith(prefix_m)}
    both = sorted(set(images) & set(masks))
    if dataset.quota:
        picked: List[str] = []
        for domain, count in dataset.quota:
            picked += _random_pick([s for s in both if s.startswith(f"{domain}__")],
                                   count, f"{dataset.key}/{domain}")
        return [(images[stem], masks[stem]) for stem in sorted(picked)]
    return [(images[stem], masks[stem]) for stem in
            _random_pick(both, size, dataset.key)]


class _SampleWorker(QObject):
    """Fetch one sample, file by file, off the GUI thread.

    The shared archive worker cannot be reused: these repositories publish no archive,
    and streaming one tar is a different job from fetching twenty small files. What is
    kept the same is the SIGNAL SHAPE, so the existing progress dialog drives this
    without knowing which kind of worker it has.
    """

    progress = Signal(str, int, int)
    info = Signal(str)
    finished = Signal(bool, str, str, str)

    def __init__(self, dataset: MaskDataset, dest: Path, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.dest = Path(dest)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception as exc:                                  # noqa: BLE001
            self.finished.emit(False, "", "", f"huggingface_hub is missing: {exc}")
            return
        try:
            listing = HfApi().list_repo_files(self.dataset.repo, repo_type="dataset")
        except Exception as exc:                                  # noqa: BLE001
            self.finished.emit(False, "", "", str(exc))
            return
        pairs = choose_sample(self.dataset, list(listing))
        if not pairs:
            self.finished.emit(
                False, "", "",
                f"{self.dataset.repo} has no image/mask pairs under "
                f"{self.dataset.images}/ and {self.dataset.masks}/")
            return
        masks_dir = self.dest / "masks"
        (masks_dir if self.dataset.masks else self.dest).mkdir(
            parents=True, exist_ok=True)
        for i, (image, mask) in enumerate(pairs, 1):
            if self._cancelled:
                self.finished.emit(False, "", "", "cancelled")
                return
            self.progress.emit(Path(image).name, i, len(pairs))
            try:
                wanted = [(image, self.dest / Path(image).name)]
                if mask:
                    wanted.append((mask, masks_dir / Path(mask).name))
                for remote, local in wanted:
                    if local.is_file():
                        continue
                    got = hf_hub_download(self.dataset.repo, remote,
                                          repo_type="dataset")
                    local.write_bytes(Path(got).read_bytes())
            except Exception as exc:                              # noqa: BLE001
                self.finished.emit(False, "", "", f"{Path(image).name}: {exc}")
                return
        self.finished.emit(True, str(self.dest), "", "")


class DatasetPicker(QDialog):
    """The list the maintainer asked for: one row per dataset, with what it is.

    A dialog rather than a dropdown on the toolbar, because each row needs two lines --
    a title a user recognises and the model it trained -- and a dropdown gives one.
    """

    def __init__(self, parent=None, datasets=MASK_DATASETS) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Open a sample of a training dataset"))
        self.setMinimumWidth(520)
        self._datasets = tuple(datasets)
        column = QVBoxLayout(self)
        blurb = QLabel(tr(
            "Ten fields of the dataset a published model was trained on, with the "
            "masks it was taught. They open for editing, so what you see is what the "
            "model saw."), self)
        blurb.setWordWrap(True)
        column.addWidget(blurb)
        self._list = QListWidget(self)
        for dataset in self._datasets:
            item = QListWidgetItem(f"{dataset.title}\n{dataset.model}", self._list)
            item.setData(Qt.UserRole, dataset.key)
            if dataset.note:
                item.setToolTip(dataset.note)
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(lambda _i: self.accept())
        column.addWidget(self._list, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel,
                                   parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

    def chosen(self) -> Optional[MaskDataset]:
        """The dataset that is selected, or None."""
        item = self._list.currentItem()
        if item is None:
            return None
        return DATASETS_BY_KEY.get(item.data(Qt.UserRole))


def install_dataset_button(screen, app_key: str = "mask", use=None):
    """Build Make Masks' "Training datasets…" button, wired to ``screen``.

    It sits beside item 412's "Load test data…" rather than replacing it. The two
    answer different questions: that one gives raw fields to segment, this one gives
    fields WITH the masks a published model was trained on.

    :param screen: the Make Masks screen the sample opens in.
    :returns: the button, for the caller to place.
    """
    from PySide6.QtWidgets import QPushButton

    button = QPushButton(tr("Training datasets…"), screen)
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(tr(
        "Open a sample of the dataset a published model was trained on, with its "
        "masks, and edit them here. One entry per model in the zoo. Cached after the "
        "first download."))
    button.clicked.connect(
        lambda _checked=False: open_a_training_dataset(screen, app_key=app_key,
                                                       use=use))
    screen._btn_training_datasets = button
    return button


def _say(screen, text: str) -> None:
    """Put ``text`` on the screen's status line, if it has one."""
    label = getattr(screen, "_status_label", None)
    if label is not None:
        label.setText(text)


def open_a_training_dataset(screen, *, pick=None, fetch=None, root=None,
                            app_key: str = "mask", use=None) -> bool:
    """Ask which dataset, fetch a sample if it is not cached, and open it.

    A cached sample opens at once with no request. Otherwise the fetch runs on a worker
    thread behind the shared progress dialog and the folder opens when it lands.

    :param screen: the Make Masks screen to open in.
    :param pick: replaces the dialog, for tests. Called as ``pick(screen)`` and returns
        a :class:`MaskDataset` or None.
    :param fetch: replaces the download, for tests. Called as
        ``fetch(screen, dataset, folder, on_done)``.
    :param root: where samples are cached; defaults to the example-data folder.
    :param app_key: which module is asking, so the picker offers its sets.
    :param use: what to DO with the folder, as ``use(folder) -> bool``. Make Masks
        opens it in the editor; Plaque Analysis points its ``src`` at it instead. The
        default calls ``screen._open_folder``, which is Make Masks' own.
    :returns: whether a folder was opened synchronously. A download that has to run
        returns False and opens later, which is what a caller can check.
    """
    dataset = (pick or (lambda s: _ask_which(s, app_key)))(screen)
    if dataset is None:
        return False
    if root is None:
        root = examples_root()
    folder = sample_folder(root, dataset)
    take = use or (lambda path: screen._open_folder(str(path)))
    if is_present(folder, dataset.size):
        _say(screen, tr("Opening {name}").format(name=dataset.title))
        return bool(take(folder))

    _say(screen, tr("Downloading {count} fields of {name}…").format(
        count=dataset.size, name=dataset.title))
    button = getattr(screen, "_btn_training_datasets", None)
    if button is not None:
        button.setEnabled(False)

    def done(result, error) -> None:
        if button is not None:
            button.setEnabled(True)
        if result is None or error:
            _say(screen, tr("Could not fetch {name}: {why}").format(
                name=dataset.title, why=error or tr("unknown error")))
            LOG.warning("training dataset %s failed: %s", dataset.key, error)
            return
        if not take(folder):
            _say(screen, tr("Downloaded {name}, but the folder would not open")
                 .format(name=dataset.title))
            return
        _say(screen, tr("{name}: {count} fields and the masks the model was trained on")
             .format(name=dataset.title, count=dataset.size))

    (fetch or _fetch_sample)(screen, dataset, folder, done)
    return False


def _ask_which(screen, app_key: str = "mask") -> Optional[MaskDataset]:
    """Show the picker and return what was chosen, or None."""
    dialog = DatasetPicker(screen, datasets_for(app_key))
    if dialog.exec() != QDialog.Accepted:
        return None
    return dialog.chosen()


def _fetch_sample(screen, dataset: MaskDataset, folder: Path, on_done) -> None:
    """Run :class:`_SampleWorker` behind the shared progress dialog."""
    from .hf_download import download_toxo_mito_demo

    download_toxo_mito_demo(
        screen, Path(folder), on_done,
        worker_factory=lambda dest: _SampleWorker(dataset, dest),
        title=tr("Downloading {name}").format(name=dataset.title))
