"""Cellpose training and domain-specific analysis pipeline entry points."""

import seaborn as sns
import os, random, sqlite3, re, time, shutil, itertools
import pandas as pd
import numpy as np
import torch

from skimage.measure import regionprops, label
from skimage.transform import resize as sk_resize, rotate
from skimage.exposure import rescale_intensity

import cellpose
from cellpose import models as cp_models
from cellpose import train as train_cp
from cellpose import io as cp_io
from cellpose.metrics import aggregated_jaccard_index
from cellpose.metrics import average_precision

try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from math import pi
from scipy.stats import chi2_contingency

from sklearn.metrics import mean_absolute_error
from skimage.measure import label as sklabel
import matplotlib.pyplot as plt
from natsort import natsorted

from torch.utils.data import Dataset

from . import schema


#: How many image/label pairs :func:`train_cellpose` previews before training.
#: The preview is a sanity check on the data, not the dataset itself, and
#: :func:`plot_cellpose_batch` allocates 4 figure-inches per image.
_TRAIN_PREVIEW_N = 8


def _cellpose_use_gpu() -> bool:
    """Return whether Cellpose can use CUDA, falling back safely to CPU."""
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"Warning: CUDA probe failed; Cellpose will use CPU: {exc}")
        return False


class CellposeLazyDataset(Dataset):
    """Lazy image/label dataset for Cellpose training and inference.

    Loads paired image and label tiffs on demand, optionally normalizing,
    augmenting (8-fold rotations/flips), and resizing to a target size.

    :param image_files: paths to input image tiffs.
    :param label_files: paths to matching label tiffs (same length as ``image_files``).
    :param settings: dict with keys ``normalize``, ``percentiles``, ``target_size``.
    :param randomize: shuffle the image/label pairing order. Default ``True``.
    :param augment: enable 8-fold augmentation (dataset length x8). Default ``False``.
    :raises ValueError: when image/label lists differ in length or are empty.
    """
    def __init__(
        self,
        image_files,
        label_files,
        settings,
        randomize: bool = True,
        augment: bool = False,
    ):
        if len(image_files) != len(label_files):
            raise ValueError(
                "image_files and label_files must have the same length."
            )
        if len(image_files) == 0:
            raise ValueError("image_files and label_files cannot be empty.")

        pairs = list(zip(map(str, image_files), map(str, label_files)))
        if randomize:
            random.shuffle(pairs)

        self.image_files = [p[0] for p in pairs]
        self.label_files = [p[1] for p in pairs]
        self.normalize = bool(settings.get("normalize", True))
        self.percentiles = settings.get("percentiles", (2, 99))
        self.target_size = int(settings["target_size"])
        self.augment = bool(augment)
        self._n_augments = 8 if self.augment else 1

    def __len__(self):
        return len(self.image_files) * self._n_augments

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return image.mean(axis=-1)
        return image

    @staticmethod
    def _scale_to_unit_interval(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        max_value = float(image.max()) if image.size else 0.0
        if max_value > 1.0:
            image = image / max_value
        return image

    @staticmethod
    def _apply_augmentation(image: np.ndarray, label: np.ndarray, aug_idx: int):
        if aug_idx == 1:
            return (
                rotate(image, 90, resize=False, preserve_range=True),
                rotate(label, 90, resize=False, preserve_range=True),
            )
        if aug_idx == 2:
            return (
                rotate(image, 180, resize=False, preserve_range=True),
                rotate(label, 180, resize=False, preserve_range=True),
            )
        if aug_idx == 3:
            return (
                rotate(image, 270, resize=False, preserve_range=True),
                rotate(label, 270, resize=False, preserve_range=True),
            )
        if aug_idx == 4:
            return np.fliplr(image), np.fliplr(label)
        if aug_idx == 5:
            return np.flipud(image), np.flipud(label)
        if aug_idx == 6:
            return (
                np.fliplr(rotate(image, 90, resize=False, preserve_range=True)),
                np.fliplr(rotate(label, 90, resize=False, preserve_range=True)),
            )
        if aug_idx == 7:
            return (
                np.flipud(rotate(image, 90, resize=False, preserve_range=True)),
                np.flipud(rotate(label, 90, resize=False, preserve_range=True)),
            )
        return image, label

    def __getitem__(self, idx):
        base_idx = idx // self._n_augments
        aug_idx = idx % self._n_augments

        image = cp_io.imread(self.image_files[base_idx])
        label = cp_io.imread(self.label_files[base_idx])

        image = self._to_grayscale(image)
        image = self._scale_to_unit_interval(image)

        if self.normalize:
            lower_p, upper_p = np.percentile(image, self.percentiles)
            if upper_p > lower_p:
                image = rescale_intensity(
                    image,
                    in_range=(lower_p, upper_p),
                    out_range=(0, 1),
                )

        image, label = self._apply_augmentation(image, label, aug_idx)

        target_shape = (self.target_size, self.target_size)
        image = sk_resize(
            image,
            target_shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        label = sk_resize(
            label,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint16)

        return image, label

def train_cellpose(settings):
    """Fine-tune the Cellpose-SAM (``cpsam``) segmentation model from images and paired masks.

    :param settings: dict of training settings; see
        ``get_train_cellpose_default_settings`` for keys including ``src``,
        ``model_name``, ``target_size``, ``n_epochs``, ``batch_size``,
        ``learning_rate``, ``weight_decay``, and ``augment``.
    :returns: None. Saves the trained model under ``<src>/models/cellpose_model``,
        named ``<model_name>_cpsam_e<n_epochs>_X<w>_Y<h>.CP_model``.
    """
    from .settings import get_train_cellpose_default_settings
    from .utils import save_settings

    settings = get_train_cellpose_default_settings(settings)
    img_src = os.path.join(settings['src'], 'train', 'images')
    mask_src = os.path.join(settings['src'], 'train', 'masks')
    target_size = settings['target_size']

    # `_cyto_` was a Cellpose-3 leftover: it named the cyto model this
    # function used to fine-tune. It fine-tunes 'cpsam' (below) and has done
    # since the Cellpose 4 port, so the old infix stamped 'cyto' onto a
    # CPSAM checkpoint and a user reading the filename was told the wrong
    # architecture. New checkpoints say cpsam.
    #
    # Names written before this change keep working: nothing parses the
    # infix. spacr.model_zoo recognises a Cellpose checkpoint by its
    # ``.CP_model`` / ``.CPmodel`` SUFFIX (model_zoo.CELLPOSE_SUFFIXES) or by
    # the folder it sits in, and _resolve_cellpose_pretrained loads any
    # existing path as given -- so `foo_cyto_e500_X1120_Y1120.CP_model` on
    # disk still resolves, still loads, and still versions.
    model_name = f"{settings['model_name']}_cpsam_e{settings['n_epochs']}_X{target_size}_Y{target_size}.CP_model"
    model_save_path = os.path.join(settings['src'], 'models', 'cellpose_model')
    os.makedirs(model_save_path, exist_ok=True)

    save_settings(settings, name=model_name)

    model = cp_models.CellposeModel(
        gpu=_cellpose_use_gpu(), pretrained_model='cpsam'
    )

    #train_image_files = sorted([os.path.join(img_src, f) for f in os.listdir(img_src) if f.endswith('.tif')])
    #train_label_files = sorted([os.path.join(mask_src, f) for f in os.listdir(mask_src) if f.endswith('.tif')])
    
    image_filenames = set(f for f in os.listdir(img_src) if f.endswith('.tif'))
    label_filenames = set(f for f in os.listdir(mask_src) if f.endswith('.tif'))

    # Only keep files that are present in both folders
    matched_filenames = sorted(image_filenames & label_filenames)

    train_image_files = [os.path.join(img_src, f) for f in matched_filenames]
    train_label_files = [os.path.join(mask_src, f) for f in matched_filenames]

    train_dataset = CellposeLazyDataset(train_image_files, train_label_files, settings, randomize=True, augment=settings['augment'])

    n_aug = 8 if settings['augment'] else 1
    max_base_images = len(train_dataset) // n_aug if settings['augment'] else len(train_dataset)

    # EVERY annotated field is training data. ``batch_size`` is the
    # optimizer's minibatch size and is passed straight to train_seg below —
    # it is not, and never was meant to be, a cap on the dataset.
    #
    # This used to read ``n_base = min(settings['batch_size'], max_base_images)``
    # followed by ``unique_base_indices[:n_base]``, so a user who annotated
    # 300 fields and left ``batch_size`` at its default of 8 trained on 8
    # images (2.7% of their work) and was told nothing.
    #
    # ``max_train_images`` is an opt-in ceiling for machines that cannot hold
    # the whole set in RAM (the images are materialised as float32 arrays
    # before train_seg sees them). Unset/None means "use everything".
    max_train_images = settings.get('max_train_images')
    if max_train_images is not None and int(max_train_images) > 0:
        n_base = min(int(max_train_images), max_base_images)
    else:
        n_base = max_base_images

    unique_base_indices = list(range(max_base_images))
    random.shuffle(unique_base_indices)
    selected_indices = unique_base_indices[:n_base]

    if n_base < max_base_images:
        print(f"max_train_images={max_train_images}: training on {n_base} of "
              f"{max_base_images} annotated images.")

    images, labels = [], []
    for idx in selected_indices:
        for aug_idx in range(n_aug):
            i = idx * n_aug + aug_idx if settings['augment'] else idx
            img, lbl = train_dataset[i]
            images.append(img)
            labels.append(lbl)
    try:
        # Preview a handful only: plot_cellpose_batch lays out one column per
        # image at 4 inches each, so handing it a full 300-image training set
        # asks matplotlib for a 100-foot-wide figure.
        plot_cellpose_batch(images[:_TRAIN_PREVIEW_N], labels[:_TRAIN_PREVIEW_N])
    except Exception:
        print(f"could not print batch images")

    print(f"Training model on {len(images)} patches from {n_base} annotated "
          f"images (augment={bool(settings['augment'])}, x{n_aug}) for "
          f"{settings['n_epochs']} epochs, minibatch {settings['batch_size']}")

    # Cellpose 4.x (SAM era) dropped the ``channels`` kwarg from
    # train_seg — models are channel-agnostic now and take a
    # ``channel_axis`` instead (None = greyscale / already-stacked).
    train_cp.train_seg(model.net,
                       train_data=images,
                       train_labels=labels,
                       channel_axis=None,
                       save_path=model_save_path,
                       n_epochs=settings['n_epochs'],
                       batch_size=settings['batch_size'],
                       learning_rate=settings['learning_rate'],
                       weight_decay=settings['weight_decay'],
                       model_name=model_name,
                       save_every=max(1, (settings['n_epochs'] // 10)),
                       rescale=False)

    print(f"Model saved at: {model_save_path}/{model_name}")
    
def test_cellpose_model(settings):
    """Evaluate a Cellpose model on a labelled test set and report per-image metrics.

    Computes Jaccard, object counts, mean object area, precision, recall,
    F1 and accuracy for each image and writes a summary CSV.

    :param settings: dict of test settings; see
        ``get_default_test_cellpose_model_settings`` for keys including
        ``src``, ``model_path``, ``batch_size``, ``FT``, ``CP_probability``,
        and ``save``.
    :returns: None. Writes ``test_results.csv`` in ``<src>/results`` when ``save`` is set.
    """
    from .utils import save_settings, print_progress
    from .settings import get_default_test_cellpose_model_settings

    def plot_cellpose_resilts(i, j, results_dir, img, lbl, pred, flow):
        """Render one 5-panel diagnostic (image / label / pred / flow) for a Cellpose result.

        :param i: outer image index used in the output filename.
        :param j: inner batch index used in the output filename.
        :param results_dir: folder where the composite PNG is written.
        :param img: source image array.
        :param lbl: ground-truth label array.
        :param pred: predicted mask array.
        :param flow: Cellpose flow field.
        """
        from . plot import generate_mask_random_cmap
        fig, axs = plt.subplots(1, 5, figsize=(16, 4), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        cmap_lbl = generate_mask_random_cmap(lbl)
        cmap_pred = generate_mask_random_cmap(pred)

        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(lbl, cmap=cmap_lbl, interpolation='nearest')
        axs[1].set_title('True Mask')
        axs[1].axis('off')

        axs[2].imshow(pred, cmap=cmap_pred, interpolation='nearest')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')
        
        axs[3].imshow(flow[2], cmap='gray')
        axs[3].set_title('Cell Probability')
        axs[3].axis('off')

        axs[4].imshow(flow[0], cmap='gray')
        axs[4].set_title('Flows')
        axs[4].axis('off')

        save_path = os.path.join(results_dir, f"cellpose_result_{i+j:03d}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
    settings = get_default_test_cellpose_model_settings(settings)
        
    save_settings(settings, name='test_cellpose_model')
    test_image_folder = os.path.join(settings['src'], 'test', 'images')
    test_label_folder = os.path.join(settings['src'], 'test', 'masks')
    results_dir = os.path.join(settings['src'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")

    image_filenames = set(f for f in os.listdir(test_image_folder) if f.endswith('.tif'))
    label_filenames = set(f for f in os.listdir(test_label_folder) if f.endswith('.tif'))

    # Only keep files that are present in both folders
    matched_filenames = sorted(image_filenames & label_filenames)

    test_image_files = [os.path.join(test_image_folder, f) for f in matched_filenames]
    test_label_files = [os.path.join(test_label_folder, f) for f in matched_filenames]

    print(f"Found {len(test_image_files)} images and {len(test_label_files)} masks")

    test_dataset = CellposeLazyDataset(test_image_files, test_label_files, settings, randomize=False, augment=False)

    model = cp_models.CellposeModel(
        gpu=_cellpose_use_gpu(), pretrained_model=settings['model_path']
    )

    batch_size = settings['batch_size']
    scores = []
    names = []
    time_ls = []
    # These per-image metric lists used to be re-initialised INSIDE the
    # batch loop, while names/scores accumulated across batches — so the
    # df_results build below raised "All arrays must be of the same
    # length" as soon as there was more than one batch. They belong here,
    # next to names/scores, so every image contributes exactly one row.
    n_objects_true_ls = []
    n_objects_pred_ls = []
    mean_area_true_ls = []
    mean_area_pred_ls = []
    tp_ls, fp_ls, fn_ls = [], [], []
    precision_ls, recall_ls, f1_ls, accuracy_ls = [], [], [], []

    # test_image_folder is a path STRING (os.path.join), so len() of it
    # measured the number of characters in the path, not the number of
    # images — the progress line reported a nonsense total.
    files_to_process = len(test_image_files)

    for i in range(0, len(test_dataset), batch_size):
        start = time.time()
        batch = [test_dataset[j] for j in range(i, min(i + batch_size, len(test_dataset)))]
        images, labels = zip(*batch)

        # Cellpose 4.x dropped ``interp`` and ``tile`` from eval; the
        # tiling behaviour is now controlled by ``tile_overlap`` alone.
        masks_pred, flows, _ = model.eval(x=list(images),
                                          channels=[0, 0],
                                          normalize=False,
                                          diameter=30,
                                          flow_threshold=settings['FT'],
                                          cellprob_threshold=settings['CP_probability'],
                                          rescale=None,
                                          resample=True,
                                          anisotropy=None,
                                          min_size=5,
                                          augment=True,
                                          tile_overlap=0.2,
                                          bsize=224)

        for j, (img, lbl, pred, flow) in enumerate(zip(images, labels, masks_pred, flows)):
            # Cellpose 4 returns one AJI value per mask as a 1-D ndarray;
            # older releases returned a scalar. Normalise both contracts
            # without averaging across images (this loop records one row per
            # image).
            aji = np.asarray(
                aggregated_jaccard_index([lbl], [pred]),
                dtype=float,
            ).reshape(-1)
            score = float(aji[0]) if aji.size else float("nan")
            fname = os.path.basename(test_label_files[i + j])
            scores.append(score)
            names.append(fname)

            # Label masks
            lbl_lab = label(lbl)
            pred_lab = label(pred)

            # Count objects
            n_true = lbl_lab.max()
            n_pred = pred_lab.max()
            n_objects_true_ls.append(n_true)
            n_objects_pred_ls.append(n_pred)

            # Mean object size (area)
            area_true = [p.area for p in regionprops(lbl_lab)]
            area_pred = [p.area for p in regionprops(pred_lab)]

            mean_area_true = np.mean(area_true) if area_true else 0
            mean_area_pred = np.mean(area_pred) if area_pred else 0
            mean_area_true_ls.append(mean_area_true)
            mean_area_pred_ls.append(mean_area_pred)
            
            # Compute object-level TP, FP, FN
            ap, tp, fp, fn = average_precision([lbl], [pred], threshold=[0.5])
            tp, fp, fn = int(tp[0, 0]), int(fp[0, 0]), int(fn[0, 0])
            tp_ls.append(tp)
            fp_ls.append(fp)
            fn_ls.append(fn)

            # Precision, Recall, F1, Accuracy
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            precision_ls.append(prec)
            recall_ls.append(rec)
            f1_ls.append(f1)
            accuracy_ls.append(acc)

            # This block used to be duplicated verbatim, so every
            # diagnostic figure was rendered and savefig'd twice to the
            # same cellpose_result_{i+j:03d}.png path.
            if settings['save']:
                plot_cellpose_resilts(i, j, results_dir, img, lbl, pred, flow)

        stop = time.time()
        duration = stop-start
        # i already steps by batch_size, i.e. it IS the dataset index of
        # the first image of this batch — (i+1)*batch_size overshot the
        # number of images actually processed on every batch after the
        # first. min() clamps the final, partial batch.
        files_processed = min(i + batch_size, len(test_dataset))
        time_ls.append(duration)
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=None, batch_size=batch_size, operation_type="test custom cellpose model")

    df_results = pd.DataFrame({
        'label_image': names,
        'Jaccard': scores,
        'n_objects_true': n_objects_true_ls,
        'n_objects_pred': n_objects_pred_ls,
        'mean_area_true': mean_area_true_ls,
        'mean_area_pred': mean_area_pred_ls,
        'TP': tp_ls,
        'FP': fp_ls,
        'FN': fn_ls,
        'Precision': precision_ls,
        'Recall': recall_ls,
        'F1': f1_ls,
        'Accuracy': accuracy_ls
    })
    
    df_results['n_error'] = abs(df_results['n_objects_pred'] - df_results['n_objects_true'])

    print(f"Average true objects/image: {df_results['n_objects_true'].mean():.2f}")
    print(f"Average predicted objects/image: {df_results['n_objects_pred'].mean():.2f}")
    print(f"Mean object area (true): {df_results['mean_area_true'].mean():.2f} px")
    print(f"Mean object area (pred): {df_results['mean_area_pred'].mean():.2f} px")
    print(f"Average Jaccard score: {df_results['Jaccard'].mean():.4f}")
    
    print(f"Average Precision: {df_results['Precision'].mean():.3f}")
    print(f"Average Recall: {df_results['Recall'].mean():.3f}")
    print(f"Average F1-score: {df_results['F1'].mean():.3f}")
    print(f"Average Accuracy: {df_results['Accuracy'].mean():.3f}")

    display(df_results)

    if settings['save']:
        df_results.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        
def apply_cellpose_model(settings):
    """Run a Cellpose model over a folder of images and export per-object measurements.

    Optionally masks predictions to a central circle, then records per-object
    area to ``measurements.csv`` and a per-image summary to ``summary.csv``.

    :param settings: dict of inference settings; see
        ``get_default_apply_cellpose_model_settings`` for keys including
        ``src``, ``model_path``, ``batch_size``, ``FT``, ``CP_probability``,
        ``circularize`` and ``save``.
    :returns: None. Writes result CSVs under ``<src>/results``.
    """
    from .settings import get_default_apply_cellpose_model_settings
    from .utils import save_settings, print_progress

    def plot_cellpose_result(i, j, results_dir, img, pred, flow):
        """Render a 4-panel diagnostic (image / pred / flow) for one Cellpose apply result.

        :param i: outer image index used in the output filename.
        :param j: inner batch index used in the output filename.
        :param results_dir: folder where the composite PNG is written.
        :param img: source image array.
        :param pred: predicted mask array.
        :param flow: Cellpose flow field.
        """
        from .plot import generate_mask_random_cmap
        
        fig, axs = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        cmap_pred = generate_mask_random_cmap(pred)

        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(pred, cmap=cmap_pred, interpolation='nearest')
        axs[1].set_title('Predicted Mask')
        axs[1].axis('off')
        
        axs[2].imshow(flow[2], cmap='gray')
        axs[2].set_title('Cell Probability')
        axs[2].axis('off')
        
        axs[3].imshow(flow[0], cmap='gray')
        axs[3].set_title('Flows')
        axs[3].axis('off')

        save_path = os.path.join(results_dir, f"cellpose_result_{i + j:03d}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
    settings = get_default_apply_cellpose_model_settings(settings)
    save_settings(settings, name='apply_cellpose_model')

    image_folder = os.path.join(settings['src'])
    results_dir = os.path.join(settings['src'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')])
    print(f"Found {len(image_files)} images")

    dummy_labels = [image_files[0]] * len(image_files)
    dataset = CellposeLazyDataset(image_files, dummy_labels, settings, randomize=False, augment=False)

    model = cp_models.CellposeModel(
        gpu=_cellpose_use_gpu(), pretrained_model=settings['model_path']
    )
    batch_size = settings['batch_size']
    measurements = []
    
    files_to_process = len(image_files)
    time_ls = []

    for i in range(0, len(dataset), batch_size):
        start = time.time() 
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        images, _ = zip(*batch)
        
        X = list(images)
        
        print(settings['CP_probability'])
        # Cellpose 4.x dropped ``interp`` and ``tile`` from eval; the
        # tiling behaviour is now controlled by ``tile_overlap`` alone.
        masks_pred, flows, _ = model.eval(x=list(images),
                                          channels=[0, 0],
                                          normalize=False,
                                          diameter=30,
                                          flow_threshold=settings['FT'],
                                          cellprob_threshold=settings['CP_probability'],
                                          rescale=None,
                                          resample=True,
                                          anisotropy=None,
                                          min_size=5,
                                          augment=True,
                                          tile_overlap=0.2,
                                          bsize=224)
        
        for j, (img, pred, flow) in enumerate(zip(images, masks_pred, flows)):
            fname = os.path.basename(image_files[i + j])

            if settings.get('circularize', False):
                h, w = pred.shape
                Y, X = np.ogrid[:h, :w]
                center_x, center_y = w / 2, h / 2
                radius = min(center_x, center_y)
                circular_mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
                pred = pred * circular_mask

            if settings['save']:
                plot_cellpose_result(i, j, results_dir, img, pred, flow)

            props = regionprops(sklabel(pred))
            for k, prop in enumerate(props):
                measurements.append({
                    'image': fname,
                    'object_id': k + 1,
                    'area': prop.area
                })
                
        stop = time.time()            
        duration = stop-start
        files_processed = (i+1) * batch_size
        time_ls.append(duration)
        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=None, batch_size=batch_size, operation_type="apply custom cellpose model")


        # Write after each batch. The columns must be declared: when a
        # batch finds no objects (blank field, aggressive CP_probability,
        # or circularize=True zeroing every peripheral object)
        # `measurements` is still [] and pd.DataFrame([]) has NO columns,
        # so the groupby below died with KeyError('image') and left
        # measurements.csv as a bare newline that pd.read_csv rejects.
        df_measurements = pd.DataFrame(measurements, columns=['image', 'object_id', 'area'])
        df_measurements.to_csv(os.path.join(results_dir, 'measurements.csv'), index=False)
        print("Saved object counts and areas to measurements.csv")

        df_summary = df_measurements.groupby('image').agg(
            object_count=('object_id', 'count'),
            average_area=('area', 'mean')
        ).reset_index()
        df_summary.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
        print("Saved object count and average area to summary.csv")

def plot_cellpose_batch(images, labels):
    """Display a two-row grid of images and their paired label masks.

    :param images: iterable of 2D grayscale image arrays.
    :param labels: iterable of matching integer label arrays.
    :returns: None.
    """
    from .plot import generate_mask_random_cmap

    cmap_lbl = generate_mask_random_cmap(labels)
    batch_size = len(images)
    # squeeze=False keeps axs 2-D for every batch size; with the default
    # squeeze=True a single-image batch collapsed to a 1-D array and the
    # axs[0, i] indexing below raised IndexError.
    fig, axs = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8), squeeze=False)
    for i in range(batch_size):
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')
        axs[1, i].imshow(labels[i], cmap=cmap_lbl, interpolation='nearest')
        axs[1, i].set_title(f'Label {i+1}')
        axs[1, i].axis('off')
    plt.show()

def analyze_percent_positive(settings):
    """Annotate objects above a threshold and summarise positive fractions per well.

    Merges measurements from ``measurements.db``, thresholds on a chosen
    feature column, then joins the resulting well-level counts against
    ``rename_log.csv`` to recover human-readable plate/well identifiers.

    :param settings: dict of settings; see
        ``default_settings_analyze_percent_positive`` for keys including
        ``src``, ``tables``, ``value_col``, ``threshold`` and ``filter_1``.
    :returns: DataFrame of annotated per-well positive/negative counts and fractions.
    """
    from . import schema
    from .io import _read_and_merge_data
    from .utils import save_settings
    from .settings import default_settings_analyze_percent_positive

    settings = default_settings_analyze_percent_positive(settings)
    
    def translate_well_in_df(csv_loc):
        """Return a dataframe read from ``csv_loc`` with ``plateID`` / ``well`` columns split out of ``Renamed TIFF``.

        :param csv_loc: path to a CSV containing a ``Renamed TIFF`` column.
        :returns: :class:`pandas.DataFrame` with parsed ``plateID`` and ``well`` columns.
        """
        # Load and extract metadata
        df = pd.read_csv(csv_loc)
        df[['plateID', 'well']] = df['Renamed TIFF'].str.replace('.tif', '', regex=False).str.split('_', expand=True)[[0, 1]]
        df['plate_well'] = df['plateID'] + '_' + df['well']

        # Retain one row per plate_well
        df_2 = df.drop_duplicates(subset='plate_well').copy()

        # Translate well to row and column. Through spacr.schema, so that a
        # lowercase well, a 1536-plate row ('AA01') and a separator-bearing
        # one ('A-01') get the same rowID here as they do in measurements.db.
        # The hand-rolled version used string.ascii_uppercase.index, which
        # raised ValueError on all three and took the whole CSV with it.
        wells = df_2['well'].map(lambda w: schema.parse_well(w))
        df_2['rowID'] = wells.map(lambda rc: rc[0])
        df_2['column_name'] = wells.map(lambda rc: rc[1])

        # Optional: add prcf ID (plate_row_column_field)
        df_2['fieldID'] = schema.field_id(1)  # default or extract from filename if needed
        df_2['prc'] = 'p' + df_2['plateID'].str.extract(r'(\d+)')[0] + '_' + df_2['rowID'] + '_' + df_2['column_name']

        return df_2
    
    def annotate_and_summarize(df, value_col, condition_col, well_col, threshold, annotation_col='annotation'):
        """Annotate rows as ``above``/``below`` a threshold and summarise per condition and well.

        :param df: measurements DataFrame to annotate in place.
        :param value_col: column whose values are compared to ``threshold``.
        :param condition_col: experimental condition column used for grouping.
        :param well_col: well identifier column used for grouping.
        :param threshold: numeric cutoff; values above become ``above``.
        :param annotation_col: name of the new annotation column. Default ``'annotation'``.
        :returns: tuple ``(df, summary_df)`` with the annotated rows and a per-(condition, well) counts/fractions table.
        """
        # Annotate
        df[annotation_col] = np.where(df[value_col] > threshold, 'above', 'below')

        # Count per condition and well
        count_df = df.groupby([condition_col, well_col, annotation_col]).size().unstack(fill_value=0)

        # Calculate total and fractions
        count_df['total'] = count_df.sum(axis=1)
        count_df['fraction_above'] = count_df.get('above', 0) / count_df['total']
        count_df['fraction_below'] = count_df.get('below', 0) / count_df['total']

        return df, count_df.reset_index()
    
    save_settings(settings, name='analyze_percent_positive', show=False)
    
    df, _ = _read_and_merge_data(locs=[settings['src']+'/measurements/measurements.db'], 
                             tables=settings['tables'], 
                             verbose=True, 
                             nuclei_limit=None, 
                             pathogen_limit=None)

    df['condition'] = 'none'
    
    if not settings['filter_1'] is None:
        df = df[df[settings['filter_1'][0]]>settings['filter_1'][1]]
    
    condition_col = 'condition'
    well_col = 'prc'
    
    df, count_df = annotate_and_summarize(df, settings['value_col'], condition_col, well_col, settings['threshold'], annotation_col='annotation')
    count_df[['plateID', 'rowID', 'column_name']] = count_df['prc'].str.split('_', expand=True)
    
    csv_loc = os.path.join(settings['src'], 'rename_log.csv')
    csv_out_loc = os.path.join(settings['src'], 'result.csv')
    translate_df = translate_well_in_df(csv_loc)
    
    merged = pd.merge(count_df, translate_df, on=['rowID', 'column_name'], how='inner')

    # 'plateID_y' is the plate parsed from rename_log.csv's 'Renamed TIFF'.
    # This used to read 'plate_y', a leftover from before the
    # plate -> plateID rename: neither frame carries a 'plate' column any
    # more, so pandas never synthesises a 'plate_y' suffix and the
    # selection always raised KeyError "['plate_y'] not in index".
    merged = merged[['plateID_y', 'well', 'plate_well','fieldID','rowID','column_name','prc_x','Original File','Renamed TIFF','above','below','fraction_above','fraction_below']]
    merged[[f'part{i}' for i in range(merged['Original File'].str.count('_').max() + 1)]] = merged['Original File'].str.split('_', expand=True)
    merged.to_csv(csv_out_loc, index=False)
    display(merged)
    return merged

def analyze_recruitment(settings):
    """Quantify recruitment of a fluorescent marker to the pathogenic vacuole and produce per-PV / per-well summaries.

    Reads the merged cell/nucleus/pathogen/cytoplasm feature tables from
    a spacr ``measurements.db``, annotates each row with cell type /
    pathogen / treatment based on plate metadata, filters objects by
    size and intensity, computes the pathogen-to-cytoplasm mean-intensity
    ratio for ``channel_of_interest``, groups by well and writes both
    ``cells.csv`` and ``wells.csv`` alongside recruitment plots.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.get_analyze_recruitment_default_settings`.
        Key entries:

        - ``src`` — folder containing ``measurements/measurements.db``
          (or the DB path directly).
        - ``cell_types`` / ``cell_plate_metadata`` — labels + row/col
          metadata that map wells to cell lines.
        - ``pathogen_types`` / ``pathogen_plate_metadata``.
        - ``treatments`` / ``treatment_plate_metadata``.
        - ``channel_of_interest`` — intensity channel for the ratio.
        - ``cell_chann_dim`` / ``nucleus_chann_dim`` /
          ``pathogen_chann_dim`` — mask channel dims.
        - ``cell_size_range``, ``nucleus_size_range``,
          ``pathogen_size_range`` — ``[min, max]`` px area filters.
        - ``*_intensity_range``, ``target_intensity_min``.
        - ``cells_per_well`` — minimum well count to keep.
        - ``plot``, ``plot_control``, ``plot_nr``, ``figuresize``.

    :returns: List ``[cells, wells]`` — the per-PV and per-well
        recruitment DataFrames, also written to CSV under ``src``.

    Example:
        .. code-block:: python

            from spacr.submodules import analyze_recruitment
            settings = {
                'src': '/data/plate01',
                'cell_types': ['HeLa'], 'cell_plate_metadata': ['c2-c11'],
                'pathogen_types': ['tgme49'], 'pathogen_plate_metadata': ['c2-c11'],
                'treatments': ['dmso','drug'], 'treatment_plate_metadata': [['r1'],['r2']],
                'channel_of_interest': 3,
            }
            cells_df, wells_df = analyze_recruitment(settings)

    See Also:
        :func:`analyze_plaques` — plaque-count/size assay.
        :func:`spacr.ml.generate_ml_scores` — feature-based classifier
        as an alternative to recruitment ratios.
    """
    
    from .io import _read_and_merge_data, _results_to_csv
    from .plot import plot_image_mask_overlay, _plot_controls, _plot_recruitment
    from .utils import _object_filter, annotate_conditions, _calculate_recruitment, _group_by_well, save_settings
    from .settings import get_analyze_recruitment_default_settings

    settings = get_analyze_recruitment_default_settings(settings=settings)
    
    if settings['src'].endswith('/measurements.db'):
        src_orig = settings['src']
        settings['src'] = os.path.dirname(settings['src'])
        if settings['src'].endswith('/measurements'):
            # The db already lives in the canonical <plate>/measurements/
            # folder, so src must go one more level up to the plate. The
            # old code only skipped the move here and left src pointing at
            # the measurements folder, which made the read below build
            # <plate>/measurements/measurements/measurements.db.
            settings['src'] = os.path.dirname(settings['src'])
        else:
            src_mes = os.path.join(settings['src'], 'measurements')
            if not os.path.exists(src_mes):
                os.makedirs(src_mes)
                shutil.move(src_orig, os.path.join(src_mes, 'measurements.db'))

    save_settings(settings, name='recruitment')

    print(f"Cell(s): {settings['cell_types']}, in {settings['cell_plate_metadata']}")
    print(f"Pathogen(s): {settings['pathogen_types']}, in {settings['pathogen_plate_metadata']}")
    print(f"Treatment(s): {settings['treatments']}, in {settings['treatment_plate_metadata']}")
    
    mask_chans=[settings['nucleus_chann_dim'], settings['pathogen_chann_dim'], settings['cell_chann_dim']]
    
    sns.color_palette("mako", as_cmap=True)
    print(f"channel:{settings['channel_of_interest']} = {settings['target']}")
    
    df, _ = _read_and_merge_data(locs=[settings['src']+'/measurements/measurements.db'], 
                                 tables=['cell', 'nucleus', 'pathogen','cytoplasm'], 
                                 verbose=True, 
                                 nuclei_limit=settings['nuclei_limit'], 
                                 pathogen_limit=settings['pathogen_limit'])
        
    df = annotate_conditions(df, 
                             cells=settings['cell_types'], 
                             cell_loc=settings['cell_plate_metadata'], 
                             pathogens=settings['pathogen_types'],
                             pathogen_loc=settings['pathogen_plate_metadata'],
                             treatments=settings['treatments'], 
                             treatment_loc=settings['treatment_plate_metadata'])
      
    df = df.dropna(subset=['condition'])
    print(f'After dropping non-annotated wells: {len(df)} rows')

    files = df['file_name'].tolist()
    print(f'found: {len(files)} files')

    files = [item + '.npy' for item in files]
    random.shuffle(files)

    _max = 10**100
    if settings['cell_size_range'] is None:
        settings['cell_size_range'] = [0,_max]
    if settings['nucleus_size_range'] is None:
        settings['nucleus_size_range'] = [0,_max]
    if settings['pathogen_size_range'] is None:
        settings['pathogen_size_range'] = [0,_max]

    if settings['plot']:
        merged_path = os.path.join(settings['src'],'merged')
        if os.path.exists(merged_path):
            try:
                for idx, file in enumerate(os.listdir(merged_path)):
                    file_path = os.path.join(merged_path,file)
                    if idx <= settings['plot_nr']:
                        plot_image_mask_overlay(file_path, 
                                                settings['channel_dims'],
                                                settings['cell_chann_dim'],
                                                settings['nucleus_chann_dim'],
                                                settings['pathogen_chann_dim'],
                                                figuresize=10,
                                                normalize=True,
                                                thickness=3,
                                                save_pdf=True)
            except Exception as e:
                print(f'Failed to plot images with outlines, Error: {e}')
        
    if not settings['cell_chann_dim'] is None:
        df = _object_filter(df, 'cell', settings['cell_size_range'], settings['cell_intensity_range'], mask_chans, 0)
        if settings['target_intensity_min'] is not None and settings['target_intensity_min'] != 0:
            df = df[df[f"cell_channel_{settings['channel_of_interest']}_percentile_95"] > settings['target_intensity_min']]
            print(f"After channel {settings['channel_of_interest']} filtration", len(df))
    if not settings['nucleus_chann_dim'] is None:
        df = _object_filter(df, 'nucleus', settings['nucleus_size_range'], settings['nucleus_intensity_range'], mask_chans, 1)
    if not settings['pathogen_chann_dim'] is None:
        df = _object_filter(df, 'pathogen', settings['pathogen_size_range'], settings['pathogen_intensity_range'], mask_chans, 2)
       
    df['recruitment'] = df[f"pathogen_channel_{settings['channel_of_interest']}_mean_intensity"]/df[f"cytoplasm_channel_{settings['channel_of_interest']}_mean_intensity"]
    
    for chan in settings['channel_dims']:
        df = _calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    
    df_well = _group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= settings['cells_per_well']]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f"After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {settings['cells_per_well']}")
    
    if settings['plot_control']:
        _plot_controls(df, mask_chans, settings['channel_of_interest'], figuresize=5)

    print(f'PV level: {len(df)} rows')
    _plot_recruitment(df, 'by PV', settings['channel_of_interest'], columns=[], figuresize=settings['figuresize'])
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df_well, 'by well', settings['channel_of_interest'], columns=[], figuresize=settings['figuresize'])
    cells,wells = _results_to_csv(settings['src'], df, df_well)

    return [cells,wells]

def analyze_plaques(settings):
    """Segment host-cell plaques with a bundled Cellpose model and summarize per-image counts and areas.

    Downloads (if needed) the bundled ``toxo_plaque_cyto_e25000`` model,
    runs Cellpose over every ``.tif`` under ``src``, then computes
    per-image plaque count + mean/stddev area and writes a
    ``plaques_analysis.db`` (tables: ``summary``, ``stats``,
    ``details``) alongside the masks.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.get_analyze_plaque_settings`. Key entries:

        - ``src`` — folder containing plaque images.
        - ``masks`` — if truthy, run segmentation before analysis; if
          falsy, expect masks already in ``<src>/masks``.
        - Standard Cellpose knobs (``diameter``, ``flow_threshold``,
          ``cellprob_threshold``, ``resample``, etc.) forwarded to
          :func:`spacr.spacr_cellpose.identify_masks_finetune`.

    :returns: None. Writes ``<src>/masks/plaques_analysis.db``.

    Example:
        .. code-block:: python

            from spacr.submodules import analyze_plaques
            analyze_plaques({'src': '/data/plaque_assay', 'masks': True})

    See Also:
        :func:`analyze_recruitment` — intensity-ratio phenotype
        instead of plaque counts.
    """
    from .spacr_cellpose import identify_masks_finetune
    from .settings import get_analyze_plaque_settings
    from .utils import save_settings, download_models
    #from spacr import __file__ as spacr_path
    spacr_path = os.path.join(os.path.dirname(__file__), '__init__.py')

    download_models()
    package_dir = os.path.dirname(spacr_path)
    models_dir = os.path.join(package_dir, 'resources', 'models', 'cp')
    model_path = os.path.join(models_dir, 'toxo_plaque_cyto_e25000_X1120_Y1120.CP_model')
    settings['custom_model'] = model_path
    print('custom_model',settings['custom_model'])

    settings = get_analyze_plaque_settings(settings)
    save_settings(settings, name='analyze_plaques', show=True)
    settings['dst'] = os.path.join(settings['src'], 'masks')

    if settings['masks']:
        identify_masks_finetune(settings)
        folder = settings['dst']
    else:
        folder = settings['dst']

    summary_data = []
    details_data = []
    stats_data = []
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if filepath.endswith('.tif') and os.path.isfile(filepath):
            print(f"Analyzing: {filepath}")
            image = cellpose.io.imread(filepath)
            labeled_image = label(image)
            regions = regionprops(labeled_image)
            
            object_count = len(regions)
            sizes = [region.area for region in regions]
            average_size = np.mean(sizes) if sizes else 0
            std_dev_size = np.std(sizes) if sizes else 0
            
            summary_data.append({'file': filename, 'object_count': object_count, 'average_size': average_size})
            stats_data.append({'file': filename, 'plaque_count': object_count, 'average_size': average_size, 'std_dev_size': std_dev_size})
            for size in sizes:
                details_data.append({'file': filename, 'plaque_size': size})
    
    # Convert lists to pandas DataFrames
    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(details_data)
    stats_df = pd.DataFrame(stats_data)
    
    # Save DataFrames to a SQLite database
    db_name = os.path.join(folder, 'plaques_analysis.db')
    conn = sqlite3.connect(db_name)
    
    summary_df.to_sql('summary', conn, if_exists='replace', index=False)
    details_df.to_sql('details', conn, if_exists='replace', index=False)
    stats_df.to_sql('stats', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"Analysis completed and saved to database '{db_name}'.")

def count_phenotypes(settings):
    """Count unique phenotype annotations per plate/row/column and export to CSV.

    :param settings: dict with ``src`` (pointing at a measurements folder or
        ``measurements.db``) and ``annotation_column`` (the column of interest
        in the ``png_list`` table).
    :returns: None. Writes ``phenotype_counts.csv`` next to the database.
    """
    from .io import _read_db

    if not settings['src'].endswith('/measurements/measurements.db'):
        settings['src'] = os.path.join(settings['src'], 'measurements/measurements.db')

    # _read_db's signature is (db_loc, tables) and it returns a LIST of
    # DataFrames (one per requested table) — the previous call used a
    # non-existent `loc=` kwarg and treated the result as a single
    # DataFrame, so count_phenotypes crashed for every caller.
    df = _read_db(settings['src'], tables=['png_list'])[0]

    unique_values_count = df[settings['annotation_column']].nunique(dropna=True)
    print(f"Unique values in {settings['annotation_column']} (excluding NaN): {unique_values_count}")

    # Count unique values in 'value' column, grouped by 'plateID', 'rowID', 'columnID'
    grouped_unique_count = df.groupby(['plateID', 'rowID', 'columnID'])[settings['annotation_column']].nunique(dropna=True).reset_index(name='unique_count')
    display(grouped_unique_count)

    # Group by plate, row, and column, then count the occurrences of each unique value
    grouped_counts = df.groupby(['plateID', 'rowID', 'columnID', 'value']).size().reset_index(name='count')

    # Pivot the DataFrame so that unique values are columns and their counts are in the rows
    pivot_df = grouped_counts.pivot_table(index=['plateID', 'rowID', 'columnID'], columns='value', values='count', fill_value=0)

    # Flatten the multi-level columns
    pivot_df.columns = [f"value_{int(col)}" for col in pivot_df.columns]

    # Reset the index so that plate, row, and column form a combined index
    pivot_df.index = pivot_df.index.map(lambda x: f"{x[0]}_{x[1]}_{x[2]}")

    # Save the pivoted counts next to the measurements database. The
    # previous revision first did os.makedirs(os.path.join('src',
    # 'results')) — a hard-coded RELATIVE path whose value was discarded
    # on the very next line, so its only effect was littering the
    # caller's cwd with a stray ./src/results directory.
    output_dir = os.path.dirname(settings['src'])
    output_path = os.path.join(output_dir, 'phenotype_counts.csv')

    pivot_df.to_csv(output_path)

    return

def compare_reads_to_scores(reads_csv, scores_csv, empirical_dict=None,
                            pc_grna='TGGT1_220950_1', nc_grna='TGGT1_233460_4',
                            y_columns=None,
                            column='columnID', value='c3', plate=None, save_paths=None):
    """Compare sequencing read fractions to classifier score fractions across wells.

    Loads paired reads and scores tables (single files or matched lists),
    computes per-well class-1 and gRNA fractions, joins them with an
    empirical row-to-mixture dictionary, and plots the fractions against
    the positive- and negative-control fractions.

    :param reads_csv: path (or list of paths) to per-gRNA read count CSVs.
    :param scores_csv: path (or list of paths) to per-object classifier score CSVs.
    :param empirical_dict: mapping of ``rowID`` to ``(pc_units, nc_units)`` mixture; a 16-row default is used when ``None``.
    :param pc_grna: positive-control gRNA name. Default ``'TGGT1_220950_1'``.
    :param nc_grna: negative-control gRNA name. Default ``'TGGT1_233460_4'``.
    :param y_columns: columns to plot on the y axis; a sensible default is used when ``None``.
    :param column: column used to select a subset of wells. Default ``'columnID'``.
    :param value: value in ``column`` to keep. Default ``'c3'``.
    :param plate: plate ID to stamp when a single pair of CSVs is given.
    :param save_paths: two-element list of PDF output paths (pc plot, nc plot).
    :returns: two matplotlib figures ``[fig_pc, fig_nc]``.
    """
    if empirical_dict is None:
        empirical_dict = {'r1':(90,10),'r2':(90,10),'r3':(80,20),'r4':(80,20),'r5':(70,30),'r6':(70,30),'r7':(60,40),'r8':(60,40),'r9':(50,50),'r10':(50,50),'r11':(40,60),'r12':(40,60),'r13':(30,70),'r14':(30,70),'r15':(20,80),'r16':(20,80)}
    if y_columns is None:
        y_columns = ['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction']
    if save_paths is None:
        # save_paths is declared with a None default but was indexed
        # unconditionally below, so the documented minimal call raised
        # TypeError. plot_line already treats save_path=None as
        # "don't save", so normalise to the two-element form here.
        save_paths = [None, None]
    def calculate_well_score_fractions(df, class_columns='cv_predictions'):
        """Aggregate per-object classifier predictions into per-well class fractions.

        :param df: measurements dataframe with a ``prc`` well id and a
            classifier prediction column.
        :param class_columns: name of the prediction column to summarise.
        :returns: dataframe keyed by ``prc`` with one fraction column per class.
        """
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = df['plateID'] + '_' + df['rowID'] + '_' + df['columnID']
        else:
            raise ValueError("Cannot find 'plateID', 'rowID', or 'columnID' in df.columns")
        prc_summary = df.groupby(['plateID', 'rowID', 'columnID', 'prc']).size().reset_index(name='total_rows')
        well_counts = (df.groupby(['plateID', 'rowID', 'columnID', 'prc', class_columns])
                       .size()
                       .unstack(fill_value=0)
                       .reset_index()
                       .rename(columns={0: 'class_0', 1: 'class_1'}))
        # unstack(fill_value=0) only materialises columns for class labels
        # that occur SOMEWHERE in the frame, so a scores table where every
        # object got the same call yields just one class column and the
        # fractions below raised KeyError. Backfill (never reindex — that
        # would drop unexpected label columns that pass through today).
        for _cls in ('class_0', 'class_1'):
            if _cls not in well_counts.columns:
                well_counts[_cls] = 0
        summary_df = pd.merge(prc_summary, well_counts, on=['plateID', 'rowID', 'columnID', 'prc'], how='left')
        summary_df['class_0_fraction'] = summary_df['class_0'] / summary_df['total_rows']
        summary_df['class_1_fraction'] = summary_df['class_1'] / summary_df['total_rows']
        return summary_df
        
    def plot_line(df, x_column, y_columns, group_column=None, xlabel=None, ylabel=None,
                  title=None, figsize=(10, 6), save_path=None, theme='deep'):
        """Plot one line per y-column (or per ``group_column`` value) against ``x_column``.

        :param df: DataFrame containing the x and y columns.
        :param x_column: column used for the x axis.
        :param y_columns: str or list of columns to plot as lines.
        :param group_column: optional hue column when ``y_columns`` is a single column.
        :param xlabel: x-axis label; falls back to ``x_column``.
        :param ylabel: y-axis label; falls back to ``'Value'``.
        :param title: plot title; falls back to ``'Line Plot'``.
        :param figsize: figure size in inches. Default ``(10, 6)``.
        :param save_path: optional PDF path to save the figure.
        :param theme: Seaborn palette name. Default ``'deep'``.
        :returns: the created matplotlib Figure.
        """

        def _set_theme(theme):
            """Return a reordered Seaborn palette for consistent line coloring."""

            def __set_reordered_theme(theme='deep', order=None, n_colors=100, show_theme=False):
                """Return a Seaborn palette optionally reordered by index list ``order``."""
                palette = sns.color_palette(theme, n_colors)
                if order:
                    reordered_palette = [palette[i] for i in order]
                else:
                    reordered_palette = palette
                if show_theme:
                    sns.palplot(reordered_palette)
                    plt.show()
                return reordered_palette

            integer_list = list(range(1, 81))
            color_order = [7, 9, 4, 0, 3, 6, 2] + integer_list
            sns_palette = __set_reordered_theme(theme, color_order, 100)
            return sns_palette

        sns_palette = _set_theme(theme)

        # Sort the DataFrame based on the x_column
        df = df.loc[natsorted(df.index, key=lambda x: df.loc[x, x_column])]
        
        fig, ax = plt.subplots(figsize=figsize)

        # Handle multiple y-columns, each as a separate line
        if isinstance(y_columns, list):
            for idx, y_col in enumerate(y_columns):
                sns.lineplot(
                    data=df, x=x_column, y=y_col, ax=ax, label=y_col, 
                    color=sns_palette[idx % len(sns_palette)], linewidth=1
                )
        else:
            sns.lineplot(
                data=df, x=x_column, y=y_columns, hue=group_column, ax=ax, 
                palette=sns_palette, linewidth=2
            )

        # Set axis labels and title
        ax.set_xlabel(xlabel if xlabel else x_column)
        ax.set_ylabel(ylabel if ylabel else 'Value')
        ax.set_title(title if title else 'Line Plot')

        # Remove top and right spines
        sns.despine(ax=ax)

        # Ensure legend only appears when needed and place it to the right
        if group_column or isinstance(y_columns, list):
            ax.legend(title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()
        return fig
    
    def calculate_grna_fraction_ratio(df, grna1='TGGT1_220950_1', grna2='TGGT1_233460_4'):
        """Compute the per-well read-fraction ratio between two gRNAs.

        :param df: dataframe with ``prc``, ``grna_name``, and ``count`` columns.
        :param grna1: numerator gRNA.
        :param grna2: denominator gRNA.
        :returns: dataframe with one ratio value per ``prc``.
        """
        # Filter relevant grna_names within each prc and group them
        grouped = df[df['grna_name'].isin([grna1, grna2])] \
            .groupby(['prc', 'grna_name']) \
            .agg({'fraction': 'sum', 'count': 'sum'}) \
            .unstack(fill_value=0)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped['fraction_ratio'] = grouped[f'fraction_{grna1}'] / grouped[f'fraction_{grna2}']
        grouped = grouped.assign(
            fraction_ratio=lambda x: x['fraction_ratio'].replace([float('inf'), -float('inf')], 0)
        ).fillna({'fraction_ratio': 0})
        grouped = grouped.rename(columns={
            f'count_{grna1}': f'{grna1}_count',
            f'count_{grna2}': f'{grna2}_count'
        })
        result = grouped.reset_index()[['prc', f'{grna1}_count', f'{grna2}_count', 'fraction_ratio']]
        result['total_reads'] = result[f'{grna1}_count'] + result[f'{grna2}_count']
        result[f'{grna1}_fraction'] = result[f'{grna1}_count'] / result['total_reads']
        result[f'{grna2}_fraction'] = result[f'{grna2}_count'] / result['total_reads']
        return result

    def calculate_well_read_fraction(df, count_column='count'):
        """Compute the per-well fraction of reads for each gRNA.

        :param df: dataframe with ``plateID``/``rowID``/``columnID`` (or ``prc``),
            ``grna_name``, and a read count column.
        :param count_column: name of the read-count column.
        :returns: dataframe with a ``fraction`` column per ``(prc, grna_name)``.
        """
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = df['plateID'] + '_' + df['rowID'] + '_' + df['columnID']
        else:
            raise ValueError("Cannot find plate, row or column in df.columns")
        grouped_df = df.groupby('prc')[count_column].sum().reset_index()
        grouped_df = grouped_df.rename(columns={count_column: 'total_counts'})
        df = pd.merge(df, grouped_df, on='prc')
        df['fraction'] = df['count'] / df['total_counts']
        return df
    
    if isinstance(reads_csv, list):
        if len(reads_csv) == len(scores_csv):
            reads_ls = []
            scores_ls = []
            for i, reads_csv_temp in enumerate(reads_csv):
                reads_df_temp = pd.read_csv(reads_csv_temp)
                scores_df_temp = pd.read_csv(scores_csv[i])
                reads_df_temp['plateID'] = f"plate{i+1}"
                scores_df_temp['plateID'] = f"plate{i+1}"
                
                if 'column' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'column': 'columnID'})
                if 'column_name' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'column_name': 'columnID'})
                # The reads-side row fixup used to test for 'row' but
                # rename 'row_name', a pandas no-op, so neither legacy
                # spelling was ever repaired. The "canonical not already
                # present" guard mirrors utils' alias table and is
                # load-bearing: without it a frame carrying both spellings
                # ends up with two 'rowID' columns and dies later with
                # "cannot reindex on an axis with duplicate labels".
                if 'row' in reads_df_temp.columns and 'rowID' not in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'row': 'rowID'})
                if 'row_name' in reads_df_temp.columns and 'rowID' not in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'row_name': 'rowID'})
                if 'row_name' in scores_df_temp.columns:
                    scores_df_temp = scores_df_temp.rename(columns={'row_name': 'rowID'})
                    
                reads_ls.append(reads_df_temp)
                scores_ls.append(scores_df_temp)
                    
            reads_df = pd.concat(reads_ls, axis=0)
            scores_df = pd.concat(scores_ls, axis=0)
            print(f"Reads: {len(reads_df)} Scores: {len(scores_df)}")
        else:
            # This branch used to only print: control then fell through to
            # calculate_well_read_fraction(reads_df) with reads_df never
            # bound, so the validation message was followed by a confusing
            # UnboundLocalError. Raise so the branch actually terminates.
            raise ValueError("reads_csv and scores_csv must contain the same number of elements if reads_csv is a list")
    else:
        reads_df = pd.read_csv(reads_csv)
        scores_df = pd.read_csv(scores_csv)
        if plate != None:
            reads_df['plateID'] = plate
            scores_df['plateID'] = plate
        
    reads_df = calculate_well_read_fraction(reads_df)
    scores_df = calculate_well_score_fractions(scores_df)
    reads_col_df = reads_df[reads_df[column]==value]
    scores_col_df = scores_df[scores_df[column]==value]
    
    reads_col_df = calculate_grna_fraction_ratio(reads_col_df, grna1=pc_grna, grna2=nc_grna)
    df = pd.merge(reads_col_df, scores_col_df, on='prc')
    
    df_emp = pd.DataFrame([(key, val[0], val[1], val[0] / (val[0] + val[1]), val[1] / (val[0] + val[1])) for key, val in empirical_dict.items()],columns=['key', 'value1', 'value2', 'pc_fraction', 'nc_fraction'])
    
    df = pd.merge(df, df_emp, left_on='rowID', right_on='key')
    
    # `if any in y_columns not in df.columns` was a chained comparison,
    # i.e. `(any in y_columns) and (y_columns not in df.columns)`, which
    # is False for every realistic input — the guard was dead and an
    # unknown y column reached seaborn as a cryptic ValueError instead.
    # plot_line's else-branch also accepts a scalar column name and a bare
    # y *vector* (Series/array), so only list/tuple/str forms name columns
    # — iterating a Series here would test its VALUES against df.columns
    # and bail out on a perfectly good call.
    if isinstance(y_columns, str):
        _y_cols = [y_columns]
    elif isinstance(y_columns, (list, tuple)):
        _y_cols = list(y_columns)
    else:
        _y_cols = []
    if any(col not in df.columns for col in _y_cols):
        print(f"columns in dataframe:")
        for col in df.columns:
            print(col)
        return
    display(df)
    fig_1 = plot_line(df, x_column = 'pc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[0])
    fig_2 = plot_line(df, x_column = 'nc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[1])
    
    return [fig_1, fig_2]

def interpret_vision_model(settings=None):
    """Explain a spacr vision-model score by ranking which morphology / intensity features drive it.

    Joins the per-object CNN predictions (``score_column``) with the
    morphology + intensity measurements from
    :func:`spacr.measure.measure_crop`, expands cross-compartment
    feature ratios (e.g. ``nucleus_cell_area``), then runs random-forest
    feature importance, permutation importance and (optionally) SHAP on
    the top features. Also groups importance by compartment and by
    channel so you can answer "is my classifier looking at the
    pathogen or at the cell?".

    :param settings: Settings dict. Key entries:

        - ``src`` — folder containing ``measurements/measurements.db``
          with both feature and score tables.
        - ``tables`` — DB tables to merge, e.g.
          ``['cell','nucleus','pathogen','cytoplasm']``.
        - ``channels`` — intensity channels included in the feature
          space (e.g. ``[0,1,2,3]``).
        - ``score_column`` — column holding per-object CNN scores.
        - ``top_features`` — cap on features shown / SHAP-explained.
        - ``feature_importance`` / ``permutation_importance`` /
          ``shap`` — toggle each explainer.
        - ``shap_sample`` — subsample size for SHAP.
        - ``nuclei_limit`` / ``pathogen_limit`` — object-count caps in
          the read/merge step.
        - ``n_jobs``, ``save``.

    :returns: Dict of DataFrames keyed by analysis name
        (``'feature_importance'``, ``'permutation_importance'``,
        ``'shap'``, ``'compartment_importance'``,
        ``'channel_importance'``, ...).

    Example:
        .. code-block:: python

            from spacr.submodules import interpret_vision_model
            results = interpret_vision_model({
                'src': '/data/plate01',
                'score_column': 'pred',
                'channels': [0,1,2,3],
                'top_features': 30, 'shap': True,
            })

    See Also:
        :func:`spacr.deep_spacr.deep_spacr` — trains the model whose
        scores this function interprets.
    """
    if settings is None:
        settings = {}
    from .io import _read_and_merge_data

    def generate_comparison_columns(df, compartments=None):
        """Add cross-compartment feature ratios (e.g. nucleus/cell) as new columns.

        :param df: measurements DataFrame; columns prefixed with each compartment.
        :param compartments: compartment prefixes to compare. Defaults to
            ``['cell', 'nucleus', 'pathogen', 'cytoplasm']``.
        :returns: tuple ``(df, comparison_dict)`` with the expanded DataFrame and a mapping of source columns to their derived ratio partners.
        """
        if compartments is None:
            compartments = ['cell', 'nucleus', 'pathogen', 'cytoplasm']
        comparison_dict = {}

        # Get columns by compartment
        compartment_columns = {comp: [col for col in df.columns if col.startswith(comp)] for comp in compartments}

        for comp0, comp0_columns in compartment_columns.items():
            for comp0_col in comp0_columns:
                related_cols = []
                base_col_name = comp0_col.replace(comp0, '')  # Base feature name without compartment prefix

                # Look for matching columns in other compartments
                for prefix, prefix_columns in compartment_columns.items():
                    if prefix == comp0:  # Skip same-compartment comparisons
                        continue
                    # Check if related column exists in other compartment
                    related_col = prefix + base_col_name
                    if related_col in df.columns:
                        related_cols.append(related_col)
                        new_col_name = f"{prefix}_{comp0}{base_col_name}"  # Format: prefix_comp0_base

                        # Calculate ratio and handle infinite or NaN values
                        ratio = (
                            pd.to_numeric(df[related_col], errors='coerce')
                            / pd.to_numeric(df[comp0_col], errors='coerce')
                        )
                        df[new_col_name] = ratio.replace(
                            [np.inf, -np.inf], np.nan).fillna(0.0)

                # Generate all-to-all comparisons
                if related_cols:
                    comparison_dict[comp0_col] = related_cols
                    for i, rel_col_1 in enumerate(related_cols):
                        for rel_col_2 in related_cols[i + 1:]:
                            # Create a new column name for each pairwise comparison
                            comp1, comp2 = rel_col_1.split('_')[0], rel_col_2.split('_')[0]
                            new_col_name_all = f"{comp1}_{comp2}{base_col_name}"

                            # Calculate pairwise ratio and handle infinite or NaN values
                            ratio = (
                                pd.to_numeric(df[rel_col_1], errors='coerce')
                                / pd.to_numeric(df[rel_col_2], errors='coerce')
                            )
                            df[new_col_name_all] = ratio.replace(
                                [np.inf, -np.inf], np.nan).fillna(0.0)

        return df, comparison_dict

    def group_feature_class(df, feature_groups=None, name='compartment', include_all=False):
        """Sum feature importance by compartment or channel group.

        :param df: DataFrame with columns ``feature`` and ``importance``.
        :param feature_groups: substrings identifying each group (compartments or channels).
        :param name: name of the grouping column to create. Default ``'compartment'``.
        :param include_all: append an ``all`` row summing across groups. Default ``False``.
        :returns: DataFrame of summed importance per group.
        """
        if feature_groups is None:
            feature_groups = ['cell', 'cytoplasm', 'nucleus', 'pathogen']
        # spacr settings identify channels by integer id ([0, 1, 2, 3]),
        # but the feature columns spell them 'channel_<n>'. The groups are
        # fed straight to re.search below, which raises "first argument
        # must be string or compiled pattern" on an int, so the documented
        # channels=[0,1,2,3] crashed. String groups are left untouched
        # (they are deliberately treated as regex patterns).
        feature_groups = [g if isinstance(g, str) else f'channel_{g}'
                          for g in feature_groups]
        def find_feature_class(feature, compartments):
            """Return the compartment(s) whose name matches ``feature``."""
            matches = [compartment for compartment in compartments if re.search(compartment, feature)]
            if len(matches) > 1:
                return '-'.join(matches)
            elif matches:
                return matches[0]
            else:
                return None

        df[name] = df['feature'].apply(lambda x: find_feature_class(x, feature_groups))

        if name == 'channel':
            df['channel'] = df['channel'].fillna('morphology')

        # Create new DataFrame with summed importance for each compartment and channel
        importance_sum = df.groupby(name)['importance'].sum().reset_index(name=f'{name}_importance_sum')
        
        if include_all:
            total_compartment_importance = importance_sum[f'{name}_importance_sum'].sum()
            importance_sum = pd.concat(
                [importance_sum,
                 pd.DataFrame(
                     [{name: 'all', f'{name}_importance_sum': total_compartment_importance}])]
                , ignore_index=True)

        return importance_sum

    # Function to create radar plot for individual and combined values
    def create_extended_radar_plot(values, labels, title):
        """Render a polar radar plot of ``values`` against ``labels``.

        :param values: numeric values per axis (one per label).
        :param labels: axis labels.
        :param title: plot title.
        """
        values = list(values) + [values[0]]  # Close the loop for radar chart
        angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.show()

    def extract_compartment_channel(feature_name):
        """Split ``feature_name`` into ``(compartment, channel)`` by the leading underscore token.

        :param feature_name: measurement feature key, e.g. ``"cell_ch0_mean"``.
        :returns: two-tuple ``(compartment, channel)`` — either may be ``None``.
        """
        # Identify compartment as the first part before an underscore
        compartment = feature_name.split('_')[0]
        
        if compartment == 'cells':
            compartment = 'cell'

        # Identify channels based on substring presence
        channels = []
        if 'channel_0' in feature_name:
            channels.append('channel_0')
        if 'channel_1' in feature_name:
            channels.append('channel_1')
        if 'channel_2' in feature_name:
            channels.append('channel_2')
        if 'channel_3' in feature_name:
            channels.append('channel_3')

        # If multiple channels are found, join them with a '+'
        if channels:
            channel = ' + '.join(channels)
        else:
            channel = 'morphology'  # Use 'morphology' if no channel identifier is found

        return (compartment, channel)

    def read_and_preprocess_data(settings):
        """Load the measurements DB pointed at by ``settings`` and return the merged dataframe.

        :param settings: settings dict; must contain ``src`` (folder holding
            ``measurements/measurements.db``).
        :returns: dataframe of merged object measurements.
        """
        df, _ = _read_and_merge_data(
            locs=[settings['src']+'/measurements/measurements.db'], 
            tables=settings['tables'], 
            verbose=True, 
            nuclei_limit=settings['nuclei_limit'], 
            pathogen_limit=settings['pathogen_limit']
        )
                
        df, _dict = generate_comparison_columns(df, compartments=['cell', 'nucleus', 'pathogen', 'cytoplasm'])
        print(f"Expanded dataframe to {len(df.columns)} columns with relative features")
        scores_df = pd.read_csv(settings['scores'])

        # Clean and align columns for merging
        df['object_label'] = df['object_label'].str.replace('o', '')

        if 'rowID' not in scores_df.columns:
            if 'row' in scores_df.columns:
                scores_df['rowID'] = scores_df['row']
            if 'row_name' in scores_df.columns:
                scores_df['rowID'] = scores_df['row_name']

        if 'columnID' not in scores_df.columns:
            # Ordered so the more specific 'column_name' wins, mirroring the
            # row branch above where 'row_name' wins over 'row'. The old
            # order let a junk 'column' column override a good
            # 'column_name'; that was invisible while the merge below still
            # keyed on 'column_name', but now silently merges to zero rows.
            if 'column' in scores_df.columns:
                scores_df['columnID'] = scores_df['column']
            if 'column_name' in scores_df.columns:
                scores_df['columnID'] = scores_df['column_name']

        if 'object_label' not in scores_df.columns:
            scores_df['object_label'] = scores_df['object']

        # Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type
        df['object_label'] = df['object_label'].str.replace('o', '').astype(str)

        # Ensure 'object_label' in scores_df is also a string
        scores_df['object_label'] = scores_df['object'].astype(str)

        # The merge below used to key on the legacy 'column_name', but
        # io._read_and_merge_data normalises every spelling to 'columnID'
        # before it returns, so the merge raised KeyError
        # "['column_name'] not in index" against any real measurements.db.
        # Key on 'columnID' (matching the alias fixup above and the
        # spacr.ml twin) while still accepting the legacy spelling, which
        # older CSVs and hand-built frames in the wild still carry.
        if 'columnID' not in df.columns and 'column_name' in df.columns:
            df['columnID'] = df['column_name']

        # Ensure all join columns have the same data type in both DataFrames
        df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']] = df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']].astype(str)
        scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']] = scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']].astype(str)

        # Select only the necessary columns from scores_df for merging
        scores_df = scores_df[['plateID', 'rowID', 'columnID', 'fieldID', 'object_label', settings['score_column']]]

        # Now merge DataFrames
        merged_df = pd.merge(df, scores_df, on=['plateID', 'rowID', 'columnID', 'fieldID', 'object_label'], how='inner')

        # Select measurements by schema role, not every numeric column.
        # Object labels and acquisition provenance are numeric in many
        # databases but must never be learned by the classifier.
        X = schema.model_feature_frame(
            merged_df,
            exclude=[settings['score_column']],
        )
        y = merged_df[settings['score_column']]

        return X, y, merged_df
    
    X, y, merged_df = read_and_preprocess_data(settings)
    
    output = {}
    
    # Step 1: Feature Importance using Random Forest
    # The outer guard used to read `feature_importance or feature_importance`
    # — the same key OR'd with itself — so the forest was never fitted
    # unless feature importance was explicitly requested. Permutation
    # importance then hit UnboundLocalError on `model`, and SHAP on
    # `feature_importance_df`, even though the docstring documents the
    # three explainers as independent toggles. The forest and the
    # importance frame are shared by all three; only the reporting,
    # grouping and output writes belong to feature_importance itself.
    if settings['feature_importance'] or settings['permutation_importance'] or settings['shap']:
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X, y)

        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        if settings['feature_importance']:
            print(f"Feature Importance ...")
            top_feature_importance_df = feature_importance_df.head(settings['top_features'])

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(top_feature_importance_df['feature'], top_feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f"Top {settings['top_features']} Features - Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()

            output['feature_importance'] = feature_importance_df
            fi_compartment_df = group_feature_class(feature_importance_df, feature_groups=settings['tables'], name='compartment', include_all=settings['include_all'])
            fi_channel_df = group_feature_class(feature_importance_df, feature_groups=settings['channels'], name='channel', include_all=settings['include_all'])

            output['feature_importance_compartment'] = fi_compartment_df
            output['feature_importance_channel'] = fi_channel_df

    # Step 2: Permutation Importance
    if settings['permutation_importance']:
        print(f"Permutation Importance ...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=settings['n_jobs'])
        perm_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean})
        perm_importance_df = perm_importance_df.sort_values(by='importance', ascending=False)
        top_perm_importance_df = perm_importance_df.head(settings['top_features'])

        # Plot Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(top_perm_importance_df['feature'], top_perm_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f"Top {settings['top_features']} Features - Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()
            
        output['permutation_importance'] = perm_importance_df
    
    # Step 3: SHAP Analysis
    if settings['shap']:
        import shap

        print(f"SHAP Analysis ...")

        # Select top N features based on Random Forest importance and fit the model on these features only
        top_features = feature_importance_df.head(settings['top_features'])['feature']
        X_top = X[top_features]

        # Refit the model on this subset of features
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X_top, y)

        # Sample a smaller subset of rows to speed up SHAP
        if settings['shap_sample']:
            # int(len/100) floors to 0 for any experiment with fewer than
            # 100 objects, which handed shap an empty background AND an
            # empty matrix to explain -> IndexError. Clamp to at least one
            # row; for >=100 objects the clamp is a no-op.
            sample = max(1, min(int(len(X_top) / 100), len(X_top)))
            X_sample = X_top.sample(sample, random_state=42)
        else:
            X_sample = X_top

        # Initialize SHAP explainer with the same subset of features
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample, max_evals=1500)

        # Plot SHAP summary for the selected sample and top features
        shap.summary_plot(shap_values, X_sample, max_display=settings['top_features'])

        # Convert SHAP values to a DataFrame for easier manipulation
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        
        # Apply the function to create MultiIndex columns with compartment and channel
        shap_df.columns = pd.MultiIndex.from_tuples(
            [extract_compartment_channel(feat) for feat in shap_df.columns], 
            names=['compartment', 'channel']
        )
        
        output['shap'] = shap_df
        
    if settings['save']:
        dst = os.path.join(settings['src'], 'results')
        os.makedirs(dst, exist_ok=True)
        for key, df in output.items(): 
            save_path = os.path.join(dst, f"{key}.csv")
            df.to_csv(save_path)
            print(f"Saved {save_path}")
        
    return output


# Backward compatibility for the misspelling published in earlier releases.
interperate_vision_model = interpret_vision_model


def analyze_endodyogeny(settings):
    """Bin pathogen *size* by log2 doublings and test the bin proportions per group.

    This is the **size-proxy** replication readout, not a parasite count.
    Read that sentence twice before quoting a number from it:

    * The rows come from :func:`spacr.io._read_and_merge_data`, which collapses
      the per-object ``pathogen`` table onto the **host cell** (``prcfo`` is
      built from ``cell_id``). ``pathogen_area`` on each row is therefore the
      *sum* of the areas of every pathogen object inside that host cell — one
      host cell carrying two parasitophorous vacuoles contributes a single row
      holding the combined area of both.
    * ``area ** 1.5`` is a 2-D-to-3-D size proxy, not a measured volume.
    * Nothing here counts parasites. A bin is a doubling of *area-derived
      size*, which tracks parasites-per-vacuole only while the pathogen mask
      segments whole vacuoles and each host cell holds exactly one.

    Keep using it when the pathogen channel gives you fused rosettes that
    cannot be resolved into single parasites. When the individual parasites
    *are* resolvable, :func:`analyze_replication` counts them and reports the
    parasites-per-vacuole distribution directly, which is the readout an
    endodyogeny experiment is actually after.

    :param settings: dict of endodyogeny settings; see
        ``set_analyze_endodyogeny_defaults`` for keys including ``src``,
        ``tables``, ``compartment``, ``min_area_bin``, ``max_area``,
        ``max_bins``, ``um_per_px``, ``group_column``, ``level`` and ``save``.
    :returns: dict with ``data`` (binned DataFrame) and ``chi_squared`` (results DataFrame).

    Example:
        .. code-block:: python

            from spacr.submodules import analyze_endodyogeny
            out = analyze_endodyogeny({'src': '/data/plate1', 'save': True})

    See Also:
        :func:`analyze_replication` — counts parasites per vacuole instead of
        inferring replication from object size.
    """
    from .utils import annotate_conditions, save_settings
    from .io import _read_and_merge_data
    from .settings import set_analyze_endodyogeny_defaults
    from .plot import plot_proportion_stacked_bars

    def _calculate_volume_bins(df, compartment='pathogen', min_area_bin=500, max_bins=None, verbose=False):
        """Assign each row to a log2 volume-doubling bin and return the ordered categories."""
        area_column = f'{compartment}_area'
        volume_column = f'{compartment}_volume'
        bin_column = f'{compartment}_volume_bin'

        df[volume_column] = df[area_column] ** 1.5
        min_volume_bin = min_area_bin ** 1.5
        max_volume = df[volume_column].max()

        if max_volume <= min_volume_bin:
            raise ValueError(
                f"Max volume ({max_volume:.2f}) is not greater than "
                f"min_volume_bin ({min_volume_bin:.2f}). Check min_area_bin or data."
            )

        n_edges = int(np.ceil(np.log2(max_volume / min_volume_bin))) + 1
        bins = [min_volume_bin * (2 ** i) for i in range(n_edges)]
        bins = sorted(set(bins))

        # Ensure the last edge exceeds the data maximum so nothing is clipped
        if bins[-1] <= max_volume:
            bins.append(bins[-1] * 2)

        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]

        if verbose:
            print('Volume bins:', bins)
            print('Volume bin labels:', bin_labels)

        # Cut into bins; values outside the range become NaN
        df[bin_column] = pd.cut(
            df[volume_column], bins=bins, labels=bin_labels, right=False
        )
        df['bin_index'] = pd.cut(
            df[volume_column], bins=bins, labels=range(1, len(bins)), right=False
        )

        # Coerce to float so NaN is preserved (int would raise)
        df['bin_index'] = pd.to_numeric(df['bin_index'], errors='coerce')

        # Drop rows that fell outside all bins
        before = len(df)
        df = df.dropna(subset=['bin_index']).copy()
        if verbose and len(df) < before:
            print(f"Dropped {before - len(df)} rows outside volume bin range")
        df['bin_index'] = df['bin_index'].astype(int)

        # Cap at max_bins
        if max_bins is not None and max_bins < len(bin_labels):
            df.loc[df['bin_index'] > max_bins, 'bin_index'] = max_bins
            capped_labels = bin_labels[:max_bins - 1] + [f">{bins[max_bins - 1]:.2f}"]
        else:
            capped_labels = bin_labels

        # Build the authoritative ordered mapping and apply it
        index_to_label = {i + 1: label for i, label in enumerate(capped_labels)}
        df[bin_column] = df['bin_index'].map(index_to_label)

        # Convert to an ordered categorical so order is never ambiguous
        ordered_categories = [index_to_label[k] for k in sorted(index_to_label.keys())]
        df[bin_column] = pd.Categorical(
            df[bin_column], categories=ordered_categories, ordered=True
        )

        if verbose:
            print(df[[volume_column, bin_column, 'bin_index']].head(20))

        return df, ordered_categories

    # ------------------------------------------------------------------
    settings = set_analyze_endodyogeny_defaults(settings)
    save_settings(settings, name='analyze_endodyogeny', show=True)
    output = {}

    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]

    locs = [os.path.join(s, 'measurements/measurements.db') for s in settings['src']]

    if 'png_list' not in settings['tables']:
        settings['tables'] = settings['tables'] + ['png_list']

    df, _ = _read_and_merge_data(
        locs,
        tables=settings['tables'],
        verbose=settings['verbose'],
        nuclei_limit=settings['nuclei_limit'],
        pathogen_limit=settings['pathogen_limit'],
        change_plate=settings['change_plate']
    )

    area_column = f"{settings['compartment']}_area"
    # Local, not settings['min_area_bin']: the um scaling below is an internal
    # unit change, and writing it back would mutate the caller's dict (and make
    # a second call with the same dict scale the threshold twice).
    min_area_bin = settings['min_area_bin']

    if settings['um_per_px'] is not None:
        df[area_column] = df[area_column] * (settings['um_per_px'] ** 2)
        min_area_bin = min_area_bin * (settings['um_per_px'] ** 2)

    df = df[df[area_column] >= min_area_bin].copy()

    df = df[df[area_column] <= settings['max_area']].copy()

    df = annotate_conditions(
        df=df,
        cells=settings['cell_types'],
        cell_loc=settings['cell_plate_metadata'],
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'],
        treatment_loc=settings['treatment_plate_metadata']
    )

    if settings['group_by_class']:
        df['new_condition'] = (
            df['condition'].astype(str) + df[settings['class_column']].astype(str)
        )
        settings['group_column'] = 'new_condition'

    # This guard used to sit AFTER the dropna below. pandas' dropna raises
    # a bare KeyError for exactly the condition tested here, so the
    # informative "Available columns" message was unreachable dead code.
    if settings['group_column'] not in df.columns:
        available = ', '.join(df.columns.tolist())
        raise KeyError(
            f"'{settings['group_column']}' not found in DataFrame. "
            f"Available columns: {available}"
        )

    df = df.dropna(subset=[settings['group_column']])

    df, ordered_bin_labels = _calculate_volume_bins(
        df,
        settings['compartment'],
        min_area_bin,
        settings['max_bins'],
        settings['verbose']
    )

    output['data'] = df

    prc_column = 'plate' if settings['level'] == 'plate' else 'prc'

    bin_column = f"{settings['compartment']}_volume_bin"

    # Remove categories that have zero observations across the entire dataset
    # so the contingency table passed to chi2_contingency has no all-zero columns
    df[bin_column] = df[bin_column].cat.remove_unused_categories()
    ordered_bin_labels = df[bin_column].cat.categories.tolist()

    results_df, pairwise_results_df, fig = plot_proportion_stacked_bars(
        settings, df, settings['group_column'],
        bin_column=bin_column, prc_column=prc_column,
        level=settings['level'], cmap=settings['cmap']
    )

    # Use the authoritative ordered list (no sorting, no dtype check needed)
    legend_labels = [
        f"{i}: {label}" for i, label in enumerate(ordered_bin_labels, start=1)
    ]

    volume_unit = "px\u00b3" if settings['um_per_px'] is None else "\u00b5m\u00b3"
    plt.legend(
        legend_labels,
        title=f'Volume Range ({volume_unit})',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.ylim(0, 1)

    output['chi_squared'] = results_df

    if settings['save']:
        output_dir = os.path.join(settings['src'][0], 'results', 'analyze_endodyogeny')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'chi_squared_results.pdf'), dpi=300, bbox_inches='tight')
        df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
        results_df.to_csv(os.path.join(output_dir, 'chi_squared_results.csv'), index=False)
        pairwise_results_df.to_csv(os.path.join(output_dir, 'chi_squared_pairwise_results.csv'), index=False)
        print(f"Chi-squared results saved to {output_dir}")

    plt.show()

    return output


# ===========================================================================
# The field key both object assays are built on
# ===========================================================================

def _ensure_field_key(df, source='the parasite table', verbose=False):
    """Give ``df`` a ``prcf`` that identifies **one field at one timepoint**.

    ``prcf`` is the unit of observation for both object assays: the replication
    assay builds every vacuole id out of it, and the invasion assay computes one
    outside-stain threshold per ``prcf``. On a plain screen it is
    ``plate_row_column_field``; on a **timelapse** it is
    ``plate_row_column_field_TIME``, which is what
    :func:`spacr.utils._map_wells` — the writer that put ``prcf`` into the
    measurements database — actually writes.

    Both assays used to rebuild the four-token form whenever the column was
    missing (which is exactly what ``change_plate=True`` arranges, since it
    drops the database's own ``prcf`` so the relabelled plate does not
    disagree with it). A four-token key on a timelapse names a *stack*, not a
    frame, and both assays then silently fold every timepoint of a field into
    one observation:

    * replication — the spatial clustering scopes on ``(prcf, cell_id)``, so
      the same host cell photographed at t1/t2/t3 became one group. A real
      2-well x 1-field x 3-frame x 2-cell database (one parasite per cell,
      12 vacuoles of 1) came out as **4 vacuoles of 3 parasites**, every one of
      them in the ``non_power_of_two`` bucket — the assay reported 100 %
      segmentation error and zero singly-infected vacuoles.
    * invasion — one Otsu cut was computed across all frames. With the stain
      level drifting between frames (the ordinary reason the threshold is
      per-field in the first place), a 36-parasite well whose true efficiency
      is **0.500** was reported as **0.944**, and 6 field rows collapsed to 2.

    Repair-on-read, the contract :func:`spacr.utils.rename_columns_in_db`
    established: a stored ``prcf`` that is *provably* this frame's own
    time-blind key — it equals the four-token build character for character —
    is a key written before this was fixed, and gets the timepoint appended.
    A ``prcf`` that differs in any other way (a renamed plate, an imported
    table, a key from :mod:`spacr.foreign`) is left exactly as the caller
    supplied it, and a database with no timepoint column is not touched at
    all.

    :param df: Frame carrying ``plateID`` / ``rowID`` / ``columnID`` /
        ``fieldID`` and, on a timelapse, ``timeID`` (or the legacy
        ``time_id`` — either spelling is resolved through
        :func:`spacr.utils._time_column`).
    :param source: Name used in the repair message.
    :param verbose: Print the resolved key composition.
    :returns: ``df``, with ``prcf`` present and time-aware.
    """
    from .utils import _time_column

    time_column = _time_column(df.columns)
    blind = (df['plateID'].astype(str) + '_' + df['rowID'].astype(str)
             + '_' + df['columnID'].astype(str) + '_'
             + df['fieldID'].astype(str))
    keyed = blind if time_column is None else blind + '_' + df[time_column].astype(str)

    if 'prcf' not in df.columns:
        df['prcf'] = keyed
        if verbose:
            built = 'plate_row_column_field' if time_column is None else \
                f'plate_row_column_field_{time_column}'
            print(f"Built prcf for {source} as {built}.")
        return df

    if time_column is None:
        return df

    stale = df['prcf'].astype(str).eq(blind)
    if stale.any():
        print(f"Repaired {int(stale.sum())} time-blind prcf value(s) in "
              f"{source}: the table carries '{time_column}' but its prcf named "
              f"only plate/row/column/field, which merges every timepoint of a "
              f"field into one observation. The timepoint has been appended.")
        # .to_numpy(): assign positionally. `keyed` shares df's index, but a
        # frame whose index carries repeated labels would make .loc align on
        # the label and write the wrong rows.
        df.loc[stale, 'prcf'] = keyed[stale].to_numpy()
    return df


# ===========================================================================
# Replication assay (Toxoplasma endodyogeny) — parasites per vacuole
# ===========================================================================

def _set_analyze_replication_defaults(settings):
    """Fallback defaults for :func:`analyze_replication`.

    The canonical copy of every pipeline's defaults lives in
    :mod:`spacr.settings`; this one is used only while
    ``spacr.settings.set_analyze_replication_defaults`` does not exist yet, so
    the assay is runnable from the API before the GUI knobs are registered.
    Once :mod:`spacr.settings` defines it, that version wins.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('parasite_table', 'pathogen')
    settings.setdefault('compartment', 'pathogen')
    settings.setdefault('vacuole_key', 'auto')
    settings.setdefault('vacuole_link_distance', None)
    settings.setdefault('vacuole_link_factor', 1.5)
    settings.setdefault('parasite_count_column', None)
    settings.setdefault('min_parasite_area', 0)
    settings.setdefault('max_parasite_area', None)
    settings.setdefault('max_parasites_per_vacuole', 16)
    settings.setdefault('require_host_cell', True)
    settings.setdefault('seed_wells_from_cells', True)
    settings.setdefault('non_power_of_two_warn', 0.2)
    settings.setdefault('cell_types', ['Hela'])
    settings.setdefault('cell_plate_metadata', None)
    settings.setdefault('pathogen_types', ['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata', [['c1'], ['c2']])
    settings.setdefault('treatments', None)
    settings.setdefault('treatment_plate_metadata', None)
    settings.setdefault('group_column', 'condition')
    settings.setdefault('level', 'object')
    settings.setdefault('change_plate', False)
    settings.setdefault('cmap', 'viridis')
    settings.setdefault('save', True)
    settings.setdefault('verbose', False)
    return settings


def _replication_bucket_order(max_power=16):
    """Return the ordered parasites-per-vacuole bucket labels.

    Powers of two from 1 up to ``max_power``, then a ``'>max_power'`` bucket
    for larger powers of two, then ``'non_power_of_two'`` last. The first
    entries are the biological ladder (1 -> 2 -> 4 -> 8 -> 16 doublings);
    ``non_power_of_two`` sits last because it is *off* that ordinal scale, not
    at the top of it.

    :param max_power: Largest explicitly named power of two. Default 16.
    :returns: list of bucket labels in order.
    """
    labels = []
    p = 1
    while p <= max_power:
        labels.append(str(p))
        p *= 2
    labels.append(f'>{max_power}')
    labels.append('non_power_of_two')
    return labels


def _replication_bucket(n, max_power=16):
    """Map a parasite count onto its parasites-per-vacuole bucket label.

    *Toxoplasma gondii* divides by endodyogeny — two daughters inside one
    mother — so a vacuole holds 1, 2, 4, 8, 16 ... parasites. Anything else
    (3, 5, 6, 7, ...) is a segmentation error, an asynchronous vacuole, or two
    vacuoles fused by the mask, and it is reported in its own bucket rather
    than rounded into a neighbour.

    :param n: Parasite count for one vacuole.
    :param max_power: Largest explicitly named power of two. Default 16.
    :returns: bucket label string.
    """
    n = int(n)
    if n < 1 or (n & (n - 1)) != 0:
        return 'non_power_of_two'
    if n > max_power:
        return f'>{max_power}'
    return str(n)


def _find_centroid_columns(df, compartment='pathogen'):
    """Locate a pair of centroid columns for ``compartment`` in a raw object table.

    Tries, in order, the plain morphology centroid, an unqualified weighted
    centroid, and finally the per-channel weighted centroid written by
    :func:`spacr.measure._intensity_measurements` (lowest channel index wins,
    so the choice is deterministic).

    :param df: Raw per-object DataFrame.
    :param compartment: Object prefix, e.g. ``'pathogen'``. Default ``'pathogen'``.
    :returns: ``(y_column, x_column)`` or ``None`` when no pair is present.
    """
    for base in (f'{compartment}_centroid', f'{compartment}_centroid_weighted'):
        if f'{base}-0' in df.columns and f'{base}-1' in df.columns:
            return f'{base}-0', f'{base}-1'

    pattern = re.compile(
        rf'^{re.escape(compartment)}_channel_(\d+)_centroid_weighted-0$'
    )
    channels = []
    for column in df.columns:
        match = pattern.match(str(column))
        if match and str(column).replace('-0', '-1') in df.columns:
            channels.append(int(match.group(1)))
    if channels:
        channel = min(channels)
        base = f'{compartment}_channel_{channel}_centroid_weighted'
        return f'{base}-0', f'{base}-1'
    return None


def _derive_vacuole_link_distance(df, compartment='pathogen', link_factor=1.5):
    """Return the centroid distance below which two parasites share a vacuole.

    Derived from the parasites themselves rather than hard-coded: parasites in
    one rosette sit roughly one parasite-diameter apart, separate vacuoles in
    the same host cell are several diameters apart. Uses the median
    ``equivalent_diameter_area`` when present, otherwise the diameter of a disc
    with the median object area.

    :param df: Raw per-parasite DataFrame.
    :param compartment: Object prefix. Default ``'pathogen'``.
    :param link_factor: Multiplier applied to the median diameter. Default 1.5.
    :returns: float distance in the units of the centroid columns (pixels).
    :raises ValueError: when neither a diameter nor an area column is present.
    """
    diameter_column = f'{compartment}_equivalent_diameter_area'
    area_column = f'{compartment}_area'

    if diameter_column in df.columns and df[diameter_column].notna().any():
        diameter = float(np.nanmedian(df[diameter_column].to_numpy(dtype=float)))
    elif area_column in df.columns and df[area_column].notna().any():
        median_area = float(np.nanmedian(df[area_column].to_numpy(dtype=float)))
        diameter = 2.0 * np.sqrt(median_area / np.pi)
    else:
        raise ValueError(
            f"Cannot derive a vacuole link distance: neither "
            f"'{diameter_column}' nor '{area_column}' is in the table. Set "
            f"'vacuole_link_distance' explicitly."
        )
    return float(diameter) * float(link_factor)


def _assign_vacuole_ids(df, compartment='pathogen', vacuole_key='auto',
                        link_distance=None, link_factor=1.5, verbose=False):
    """Attach a ``vacuole_id`` to every parasite row and report how it was derived.

    The counting unit of a replication assay is the parasitophorous vacuole.
    It is *not* the host cell — one host cell routinely carries several
    vacuoles, and grouping on ``cell_id`` silently reports their combined
    parasite count as a single, plausible-looking, wrong number.

    Resolution order for ``vacuole_key='auto'``:

    1. an explicit ``vacuole_id`` / ``<compartment>_vacuole_id`` column, if the
       segmentation produced one;
    2. ``'spatial'`` — single-linkage clustering of parasite centroids inside
       each (field, host cell), which separates two rosettes sharing a host;
    3. ``'cell_id'`` — one vacuole per infected host cell. Approximate, and
       announced as such.
    4. ``'object'`` — one vacuole per pathogen object, used only when there is
       no host-cell column and no centroids to cluster on.

    :param df: Raw per-parasite DataFrame; needs ``prcf`` and usually ``cell_id``.
    :param compartment: Object prefix. Default ``'pathogen'``.
    :param vacuole_key: ``'auto'``, ``'spatial'``, ``'cell_id'``, ``'object'``
        or the name of a column holding a vacuole identifier.
    :param link_distance: Centroid distance threshold for ``'spatial'``;
        ``None`` derives it from the parasite sizes.
    :param link_factor: Multiplier used by that derivation. Default 1.5.
    :param verbose: Print the resolved key and threshold.
    :returns: ``(df, resolved_key, link_distance_used)``.
    :raises KeyError: when an explicitly named key is not a column.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    df = df.copy()
    has_cell = 'cell_id' in df.columns
    centroid_columns = _find_centroid_columns(df, compartment)

    explicit_columns = [
        column for column in ('vacuole_id', f'{compartment}_vacuole_id')
        if column in df.columns
    ]

    if vacuole_key == 'auto':
        if explicit_columns:
            vacuole_key = explicit_columns[0]
        elif centroid_columns is not None and has_cell:
            vacuole_key = 'spatial'
        elif has_cell:
            vacuole_key = 'cell_id'
        else:
            vacuole_key = 'object'

    if vacuole_key not in ('spatial', 'cell_id', 'object') and vacuole_key not in df.columns:
        raise KeyError(
            f"vacuole_key '{vacuole_key}' is not a column of the parasite "
            f"table. Available columns: {', '.join(map(str, df.columns))}"
        )

    if vacuole_key == 'object':
        print(
            "WARNING: no host-cell column and no centroids — every pathogen "
            "object is being treated as its own vacuole. Parasites-per-vacuole "
            "is only meaningful here if the pathogen mask segments whole "
            "vacuoles and a parasite count column is supplied."
        )
        df['vacuole_id'] = (
            df['prcf'].astype(str) + '_o' + df['object_label'].astype(str)
        )
        return df, vacuole_key, None

    if vacuole_key == 'cell_id':
        print(
            "WARNING: grouping parasites by host cell. A host cell carrying "
            "two vacuoles will be reported as ONE vacuole holding their "
            "combined parasite count. Provide centroids (vacuole_key="
            "'spatial') or a vacuole column for a per-vacuole readout."
        )
        df['vacuole_id'] = (
            df['prcf'].astype(str) + '_c' + df['cell_id'].astype(str)
        )
        return df, vacuole_key, None

    if vacuole_key != 'spatial':
        df['vacuole_id'] = (
            df['prcf'].astype(str) + '_v' + df[vacuole_key].astype(str)
        )
        return df, vacuole_key, None

    # -- spatial ----------------------------------------------------------
    if centroid_columns is None:
        raise KeyError(
            f"vacuole_key='spatial' needs centroid columns for "
            f"'{compartment}' and none were found. Measure with "
            f"intensity features enabled, or set vacuole_key='cell_id'."
        )
    if link_distance is None:
        link_distance = _derive_vacuole_link_distance(df, compartment, link_factor)
    link_distance = float(link_distance)

    y_column, x_column = centroid_columns
    scope_columns = ['prcf', 'cell_id'] if has_cell else ['prcf']

    labels = pd.Series(index=df.index, dtype=object)
    for scope, group in df.groupby(scope_columns, dropna=False, sort=False):
        scope_tag = '_'.join(str(part) for part in np.atleast_1d(scope))
        coordinates = group[[y_column, x_column]].to_numpy(dtype=float)
        finite = np.isfinite(coordinates).all(axis=1)

        clusters = np.zeros(len(group), dtype=int)
        if finite.sum() >= 2:
            linked = linkage(coordinates[finite], method='single')
            clusters[finite] = fcluster(linked, t=link_distance,
                                        criterion='distance')
        elif finite.sum() == 1:
            clusters[finite] = 1
        # Objects with a non-finite centroid cannot be clustered; each becomes
        # its own vacuole rather than silently joining cluster 0 together.
        next_id = clusters.max() + 1 if len(clusters) else 1
        for position in np.flatnonzero(~finite):
            clusters[position] = next_id
            next_id += 1

        labels.loc[group.index] = [
            f'{scope_tag}_v{cluster}' for cluster in clusters
        ]

    df['vacuole_id'] = labels
    if verbose:
        print(f"vacuole_key='spatial', link distance {link_distance:.2f} px, "
              f"{df['vacuole_id'].nunique()} vacuoles from {len(df)} parasites")
    return df, vacuole_key, link_distance


def _replication_well_distribution(vacuoles, group_column, buckets,
                                   non_power_of_two_warn=0.2, wells=None):
    """Summarize the parasites-per-vacuole distribution for every well.

    One row per (group, well). Reports the bucket fractions, the median, and a
    mean paired with the fraction of vacuoles that mean was computed from, so a
    mean taken over a minority of trustworthy vacuoles cannot be quoted without
    that context.

    :param vacuoles: Per-vacuole DataFrame from :func:`analyze_replication`.
    :param group_column: Condition column carried onto each well row.
    :param buckets: Ordered bucket labels from :func:`_replication_bucket_order`.
    :param non_power_of_two_warn: ``non_power_of_two`` fraction above which the
        well is flagged. Default 0.2.
    :param wells: Optional DataFrame of ``(plateID, rowID, columnID, prc,
        group_column)`` rows to seed the output with, so wells that contain
        host cells but no vacuoles appear with zeros instead of vanishing.
    :returns: per-well DataFrame.
    """
    bucket_columns = {bucket: _bucket_column_suffix(bucket) for bucket in buckets}

    identity = ['plateID', 'rowID', 'columnID', 'prc', group_column]
    rows = []

    seeded = {}
    if wells is not None and len(wells) > 0:
        for record in wells[identity].drop_duplicates().to_dict('records'):
            seeded[(record['prc'], record[group_column])] = record

    if len(vacuoles) > 0:
        for key, group in vacuoles.groupby(['prc', group_column],
                                           dropna=False, sort=False):
            record = {column: group[column].iloc[0] for column in identity}
            seeded[key] = record

    for key, record in seeded.items():
        prc, group_value = key
        subset = vacuoles[(vacuoles['prc'] == prc)
                          & (vacuoles[group_column] == group_value)]
        n_vacuoles = int(len(subset))
        row = dict(record)
        row['n_vacuoles'] = n_vacuoles
        row['n_parasites'] = int(subset['n_parasites'].sum()) if n_vacuoles else 0

        for bucket in buckets:
            suffix = bucket_columns[bucket]
            count = int((subset['replication_bucket'] == bucket).sum()) if n_vacuoles else 0
            row[f'n_{suffix}'] = count
            # No vacuoles is a real, reportable state (an uninfected well), not
            # a divide-by-zero: every fraction is 0.0 and n_vacuoles says why.
            row[f'frac_{suffix}'] = (count / n_vacuoles) if n_vacuoles else 0.0

        row['non_power_of_two_fraction'] = row['frac_non_power_of_two']
        row['qc_flag_non_power_of_two'] = bool(
            row['non_power_of_two_fraction'] > non_power_of_two_warn
        )

        if n_vacuoles:
            row['median_parasites_per_vacuole'] = float(
                np.median(subset['n_parasites'].to_numpy(dtype=float))
            )
            on_ladder = subset[subset['is_power_of_two']]
            row['n_power_of_two'] = int(len(on_ladder))
            if len(on_ladder):
                row['median_doublings'] = float(
                    np.median(on_ladder['doublings'].to_numpy(dtype=float))
                )
                row['mean_parasites_per_vacuole'] = float(
                    on_ladder['n_parasites'].mean()
                )
            else:
                row['median_doublings'] = 0.0
                row['mean_parasites_per_vacuole'] = 0.0
            row['mean_fraction_of_vacuoles'] = len(on_ladder) / n_vacuoles
        else:
            row['median_parasites_per_vacuole'] = 0.0
            row['n_power_of_two'] = 0
            row['median_doublings'] = 0.0
            row['mean_parasites_per_vacuole'] = 0.0
            row['mean_fraction_of_vacuoles'] = 0.0

        rows.append(row)

    columns = identity + ['n_vacuoles', 'n_parasites']
    for bucket in buckets:
        columns += [f'n_{bucket_columns[bucket]}', f'frac_{bucket_columns[bucket]}']
    columns += ['non_power_of_two_fraction', 'qc_flag_non_power_of_two',
                'median_parasites_per_vacuole', 'n_power_of_two',
                'median_doublings', 'mean_parasites_per_vacuole',
                'mean_fraction_of_vacuoles']

    return pd.DataFrame(rows, columns=columns)


def _bucket_column_suffix(bucket):
    """Turn a bucket label into a column-name-safe suffix (``'>16'`` -> ``'gt16'``)."""
    return str(bucket).replace('>', 'gt')


def _replication_summary(vacuoles, group_column, buckets,
                         non_power_of_two_warn=0.2):
    """Collapse the per-vacuole table to one row per experimental group.

    :param vacuoles: Per-vacuole DataFrame.
    :param group_column: Condition column.
    :param buckets: Ordered bucket labels.
    :param non_power_of_two_warn: QC threshold on the non-power-of-two fraction.
    :returns: per-group DataFrame.
    """
    rows = []
    for group_value, subset in vacuoles.groupby(group_column, dropna=False,
                                                sort=False):
        n_vacuoles = int(len(subset))
        row = {group_column: group_value,
               'n_wells': int(subset['prc'].nunique()),
               'n_vacuoles': n_vacuoles,
               'n_parasites': int(subset['n_parasites'].sum())}
        for bucket in buckets:
            suffix = _bucket_column_suffix(bucket)
            count = int((subset['replication_bucket'] == bucket).sum())
            row[f'n_{suffix}'] = count
            row[f'frac_{suffix}'] = (count / n_vacuoles) if n_vacuoles else 0.0

        row['non_power_of_two_fraction'] = row['frac_non_power_of_two']
        row['qc_flag_non_power_of_two'] = bool(
            row['non_power_of_two_fraction'] > non_power_of_two_warn
        )
        row['median_parasites_per_vacuole'] = (
            float(np.median(subset['n_parasites'].to_numpy(dtype=float)))
            if n_vacuoles else 0.0
        )
        on_ladder = subset[subset['is_power_of_two']]
        row['n_power_of_two'] = int(len(on_ladder))
        row['median_doublings'] = (
            float(np.median(on_ladder['doublings'].to_numpy(dtype=float)))
            if len(on_ladder) else 0.0
        )
        row['mean_parasites_per_vacuole'] = (
            float(on_ladder['n_parasites'].mean()) if len(on_ladder) else 0.0
        )
        row['mean_fraction_of_vacuoles'] = (
            len(on_ladder) / n_vacuoles if n_vacuoles else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _replication_compare_conditions(vacuoles, group_column, buckets,
                                    verbose=False):
    """Compare the parasites-per-vacuole distribution between every pair of groups.

    The primary test is a **Mann-Whitney U (Wilcoxon rank-sum) test on the
    doubling index** ``log2(n_parasites)``, restricted to vacuoles that sit on
    the power-of-two ladder. Reasons, in order of importance:

    * The outcome is an *ordered discrete* class (1, 2, 4, 8, 16), so a test
      must use the ordering. A plain chi-squared over the buckets throws it
      away — swap the 2 and 8 columns and the chi-squared is unchanged, while
      the biology is reversed. Mann-Whitney tests exactly the alternative that
      matters: one condition is stochastically shifted toward fewer (or more)
      divisions.
    * A t-test on the raw counts is wrong twice over. The counts are not
      interval-scaled — 8 -> 16 is one division, the same single division as
      1 -> 2 — so their arithmetic mean is dominated by the tail. And the
      distribution is discrete and multimodal by construction, so the normal
      approximation a t-test rests on never holds.
    * Ranks handle the heavy ties that a five-value scale produces; scipy's
      normal approximation applies the tie correction.

    A chi-squared over the full bucket table (including ``non_power_of_two``)
    is reported alongside as an omnibus "does anything differ" check, and the
    rank-biserial correlation gives an effect size that a p-value cannot.

    :param vacuoles: Per-vacuole DataFrame.
    :param group_column: Condition column.
    :param buckets: Ordered bucket labels.
    :param verbose: Print the resulting table.
    :returns: DataFrame with one row per group pair; empty (with the full
        column set) when there are fewer than two groups.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    from .sp_stats import choose_p_adjust_method

    columns = ['group1', 'group2', 'test', 'n1', 'n2',
               'n1_power_of_two', 'n2_power_of_two',
               'median_doublings_1', 'median_doublings_2',
               'u_statistic', 'p_value', 'rank_biserial',
               'chi_squared_stat', 'chi_squared_p_value',
               'non_power_of_two_fraction_1', 'non_power_of_two_fraction_2',
               'p_value_adj', 'adj']

    groups = list(pd.unique(vacuoles[group_column].dropna()))
    if len(groups) < 2:
        return pd.DataFrame(columns=columns)

    counts = (
        vacuoles.groupby([group_column, 'replication_bucket'], observed=False)
        .size().unstack(fill_value=0)
    )

    results = []
    for group1, group2 in itertools.combinations(groups, 2):
        left = vacuoles[vacuoles[group_column] == group1]
        right = vacuoles[vacuoles[group_column] == group2]
        left_ladder = left.loc[left['is_power_of_two'], 'doublings'].to_numpy(dtype=float)
        right_ladder = right.loc[right['is_power_of_two'], 'doublings'].to_numpy(dtype=float)

        if len(left_ladder) and len(right_ladder):
            if np.all(left_ladder == left_ladder[0]) and np.all(
                right_ladder == left_ladder[0]
            ):
                # SciPy 1.17 reports NaN when the pooled ranked outcome has
                # zero variance. The two distributions are exactly identical,
                # so the defined no-difference result is U=n1*n2/2, p=1.
                statistic = len(left_ladder) * len(right_ladder) / 2.0
                p_value = 1.0
            else:
                statistic, p_value = mannwhitneyu(
                    left_ladder, right_ladder, alternative='two-sided')
            # Rank-biserial correlation: +1 means every vacuole in group1 is
            # further along the ladder than every vacuole in group2.
            rank_biserial = 2.0 * statistic / (len(left_ladder) * len(right_ladder)) - 1.0
        else:
            statistic, p_value, rank_biserial = np.nan, np.nan, np.nan

        # Dropping the buckets neither group occupies is what keeps scipy from
        # rejecting the table over an all-zero column; every group has at
        # least one vacuole, so no row can be empty.
        pair_counts = counts.loc[[group1, group2]]
        pair_counts = pair_counts.loc[:, pair_counts.sum(axis=0) > 0]
        chi2, chi2_p, _, _ = chi2_contingency(pair_counts.to_numpy())

        results.append({
            'group1': group1,
            'group2': group2,
            'test': 'Mann-Whitney U on log2(parasites per vacuole)',
            'n1': int(len(left)),
            'n2': int(len(right)),
            'n1_power_of_two': int(len(left_ladder)),
            'n2_power_of_two': int(len(right_ladder)),
            'median_doublings_1': float(np.median(left_ladder)) if len(left_ladder) else np.nan,
            'median_doublings_2': float(np.median(right_ladder)) if len(right_ladder) else np.nan,
            'u_statistic': statistic,
            'p_value': p_value,
            'rank_biserial': rank_biserial,
            'chi_squared_stat': chi2,
            'chi_squared_p_value': chi2_p,
            'non_power_of_two_fraction_1': (
                float((~left['is_power_of_two']).mean()) if len(left) else 0.0),
            'non_power_of_two_fraction_2': (
                float((~right['is_power_of_two']).mean()) if len(right) else 0.0),
        })

    results_df = pd.DataFrame(results)
    method = choose_p_adjust_method(len(groups),
                                    float(counts.sum(axis=1).mean()))
    finite = results_df['p_value'].notna()
    results_df['p_value_adj'] = np.nan
    if finite.any():
        results_df.loc[finite, 'p_value_adj'] = multipletests(
            results_df.loc[finite, 'p_value'].to_numpy(dtype=float),
            method=method
        )[1]
    results_df['adj'] = method

    results_df = results_df[columns]
    if verbose:
        print("\nParasites-per-vacuole comparisons:")
        print(results_df.to_string(index=False))
    return results_df


def _chi_pairwise_is_safe(counts):
    """True when every pair of rows of ``counts`` can go through ``chi_pairwise``.

    :func:`spacr.sp_stats.chi_pairwise` slices the contingency table two rows
    at a time and hands each slice to ``scipy.stats.chi2_contingency``, which
    refuses a sub-table holding an all-zero row or column, and it divides by
    the number of comparisons, so a single group raises ``ZeroDivisionError``.
    Both cases are routine for a sparse per-well table (two wells whose
    vacuoles share no bucket), so callers check before delegating instead of
    crashing halfway through drawing a figure.

    :param counts: Contingency-table DataFrame or array indexed by group.
    :returns: bool.
    """
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return False
    for first, second in itertools.combinations(range(values.shape[0]), 2):
        pair = values[[first, second], :]
        if np.any(pair.sum(axis=0) == 0) or np.any(pair.sum(axis=1) == 0):
            return False
    return True


def _replication_stacked_bars(settings, vacuoles, group_column, prc_column,
                              level, cmap, title):
    """Draw stacked bucket-proportion bars, reusing the shared plot helper.

    Delegates to :func:`spacr.plot.plot_proportion_stacked_bars` whenever its
    contingency table is well formed — see :func:`_chi_pairwise_is_safe` for
    what "well formed" costs. When it is not (one group, or a sparse per-well
    table), the bars are drawn here and the statistics come back empty: the
    figure is descriptive either way, and the tests that matter live in
    :func:`_replication_compare_conditions`, which handles sparsity itself.

    :param settings: Settings dict (``verbose`` is read by the helper).
    :param vacuoles: Per-vacuole DataFrame with a ``replication_bucket`` column.
    :param group_column: Column forming the bar axis.
    :param prc_column: Per-well identifier used when ``level`` aggregates.
    :param level: ``'object'``, ``'well'`` or ``'plateID'``.
    :param cmap: Matplotlib colormap name.
    :param title: Axes title.
    :returns: ``(results_df, pairwise_df, fig)``.
    """
    from .plot import plot_proportion_stacked_bars

    working = vacuoles.copy()
    working['replication_bucket'] = (
        working['replication_bucket'].cat.remove_unused_categories()
    )
    counts = working.groupby([group_column, 'replication_bucket'],
                             observed=True).size().unstack(fill_value=0)

    if _chi_pairwise_is_safe(counts):
        results_df, pairwise_df, fig = plot_proportion_stacked_bars(
            settings, working, group_column, bin_column='replication_bucket',
            prc_column=prc_column, level=level, cmap=cmap
        )
    else:
        proportions = counts.div(counts.sum(axis=1), axis=0)
        axes = proportions.plot(kind='bar', stacked=True, colormap=cmap,
                                figsize=(12, 8))
        axes.set_xlabel('Group')
        axes.set_ylabel('Proportion')
        fig = plt.gcf()
        results_df = pd.DataFrame({'chi_squared_stat': [np.nan],
                                   'p_value': [np.nan],
                                   'degrees_of_freedom': [np.nan]})
        pairwise_df = pd.DataFrame(columns=['Group 1', 'Group 2', 'Test Name',
                                            'p-value', 'p-value_adj', 'adj'])

    axes = fig.axes[0]
    axes.set_title(title)
    axes.set_ylim(0, 1)
    axes.legend(title='Parasites per vacuole', bbox_to_anchor=(1.05, 1),
                loc='upper left')
    return results_df, pairwise_df, fig


def analyze_replication(settings):
    """Replication assay: count parasites per vacuole and compare the distributions.

    *Toxoplasma gondii* replicates by endodyogeny, two daughters forming inside
    a mother, so a parasitophorous vacuole holds 1, 2, 4, 8 or 16 parasites —
    a power of two. The readout of a replication assay is therefore the
    **distribution** of parasites-per-vacuole across a well, not a mean: a mean
    of 3.2 cannot distinguish "everything at 3-ish", which is biologically
    impossible, from a healthy mix of 2s and 4s. A drug that slows replication
    moves mass from the 8 and 4 buckets down into 2 and 1, and only the
    distribution shows that.

    **The counting unit is the vacuole.** Not the parasite, and emphatically
    not the host cell — one host cell routinely carries several vacuoles, so
    grouping on ``cell_id`` reports their combined count as a single vacuole
    and produces a plausible but meaningless number. See
    :func:`_assign_vacuole_ids` for how the vacuole is derived and what each
    ``vacuole_key`` costs you.

    Rosettes of 3, 5, 6 or 7 are counted into an explicit ``non_power_of_two``
    bucket that is always reported and never folded into a neighbouring bucket.
    That bucket is the assay's own quality control: a well where 30% of
    vacuoles are off the power-of-two ladder has a segmentation problem, and
    its replication number should not be trusted.

    Statistics: the two-condition comparison is a Mann-Whitney U test on the
    doubling index ``log2(n_parasites)``, with a chi-squared omnibus test
    alongside it. :func:`_replication_compare_conditions` explains why, and why
    a t-test on the raw counts is the wrong instrument.

    :param settings: dict of replication settings; see
        ``set_analyze_replication_defaults``. Key entries:

        - ``src`` — plate directory (or list of them) holding
          ``measurements/measurements.db``.
        - ``parasite_table`` / ``compartment`` — table and column prefix
          holding one row per segmented parasite. Default ``'pathogen'``.
        - ``vacuole_key`` — how parasite rows are grouped into vacuoles
          (``'auto'``, ``'spatial'``, ``'cell_id'``, ``'object'``, or a column
          name).
        - ``vacuole_link_distance`` / ``vacuole_link_factor`` — the spatial
          clustering threshold, or the multiplier used to derive it.
        - ``min_parasite_area`` / ``max_parasite_area`` — debris and
          merged-clump filters applied before counting.
        - ``max_parasites_per_vacuole`` — largest named power-of-two bucket.
        - ``non_power_of_two_warn`` — QC flag threshold.
        - ``cell_types`` / ``pathogen_types`` / ``treatments`` and their
          ``*_plate_metadata`` well maps, plus ``group_column`` and ``level``.
        - ``save`` — write the CSVs and figures under
          ``<src>/results/analyze_replication``.

    :returns: dict with ``vacuoles`` (per-vacuole counts), ``wells`` (per-well
        distribution), ``summary`` (per-condition distribution),
        ``comparisons`` (pairwise ordered tests), ``chi_squared`` /
        ``chi_squared_pairwise`` (omnibus proportion tests), ``figures`` and
        ``vacuole_key`` (the grouping actually used).
    :raises ValueError: when the parasite table holds no usable rows.

    Example:
        .. code-block:: python

            from spacr.submodules import analyze_replication
            out = analyze_replication({
                'src': '/data/plate1',
                'pathogen_types': ['dmso', 'pyrimethamine'],
                'pathogen_plate_metadata': [['c1'], ['c2']],
            })
            print(out['summary'][['condition', 'frac_1', 'frac_2', 'frac_4',
                                  'frac_8', 'frac_non_power_of_two']])

    See Also:
        :func:`analyze_endodyogeny` — the size-proxy version, for fused
        rosettes that cannot be resolved into single parasites.
    """
    from .utils import annotate_conditions, save_settings
    from .io import _read_db
    from . import settings as settings_module

    # spacr.settings owns every pipeline's defaults and wins wherever it
    # defines one; the local copy runs afterwards purely as a gap-filler, so
    # the assay is callable before the GUI knobs are registered. Both use
    # setdefault, so running settings.py first makes its values authoritative.
    apply_defaults = getattr(settings_module, 'set_analyze_replication_defaults',
                             None)
    if apply_defaults is not None:
        settings = apply_defaults(settings)
    settings = _set_analyze_replication_defaults(settings)
    save_settings(settings, name='analyze_replication', show=settings['verbose'])

    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]

    compartment = settings['compartment']
    parasite_table = settings['parasite_table']
    buckets = _replication_bucket_order(settings['max_parasites_per_vacuole'])

    # ---- read one row per segmented parasite ----------------------------
    # Deliberately NOT _read_and_merge_data: that helper collapses the
    # pathogen table onto the host cell (prcfo is built from cell_id), which
    # destroys the per-vacuole identity this assay is built on.
    parasite_frames, cell_frames = [], []
    for index, source in enumerate(settings['src']):
        location = os.path.join(source, 'measurements/measurements.db')
        frame = _read_db(location, [parasite_table])[0]
        if settings['change_plate']:
            # prcf carries the ORIGINAL plate name and is what the vacuole id
            # is built from, so relabelling plateID alone would let two plates
            # that share a well/field/cell collapse into one vacuole.
            frame['plateID'] = f'plate{index + 1}'
            frame = frame.drop(columns=['prcf'], errors='ignore')
        parasite_frames.append(frame)
        if settings['seed_wells_from_cells']:
            try:
                cell_frame = _read_db(location, ['cell'])[0]
            except ValueError:
                cell_frame = None
            if cell_frame is not None:
                if settings['change_plate']:
                    cell_frame['plateID'] = f'plate{index + 1}'
                cell_frames.append(cell_frame)

    df = pd.concat(parasite_frames, axis=0, ignore_index=True)

    for column in ('plateID', 'rowID', 'columnID', 'fieldID'):
        if column not in df.columns:
            raise ValueError(
                f"Table '{parasite_table}' has no '{column}' column; it does "
                f"not look like a spacr measurements table."
            )
    # The timepoint is part of this key on a timelapse; see _ensure_field_key.
    # Every vacuole id below is built from prcf, so a time-blind one merges the
    # same host cell across all of its frames into a single vacuole.
    df = _ensure_field_key(df, source=f"table '{parasite_table}'",
                           verbose=settings['verbose'])
    df['prc'] = (df['plateID'].astype(str) + '_' + df['rowID'].astype(str)
                 + '_' + df['columnID'].astype(str))

    # ---- object filters --------------------------------------------------
    area_column = f'{compartment}_area'
    if area_column in df.columns:
        if settings['min_parasite_area']:
            df = df[df[area_column] >= settings['min_parasite_area']]
        if settings['max_parasite_area'] is not None:
            df = df[df[area_column] <= settings['max_parasite_area']]

    if 'cell_id' in df.columns:
        # 0 / NaN means the object overlapped no host cell — an extracellular
        # parasite, which has no vacuole and cannot enter a replication count.
        host = pd.to_numeric(df['cell_id'], errors='coerce')
        if settings['require_host_cell']:
            df = df[host.notna() & (host != 0)]
        df = df.copy()
        df['cell_id'] = host.fillna(0).astype(int)

    df = df.copy()
    if len(df) == 0:
        raise ValueError(
            f"No parasite objects left in '{parasite_table}' after filtering. "
            f"Check min_parasite_area / max_parasite_area / require_host_cell."
        )

    df = annotate_conditions(
        df=df,
        cells=settings['cell_types'],
        cell_loc=settings['cell_plate_metadata'],
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'],
        treatment_loc=settings['treatment_plate_metadata'],
    )

    group_column = settings['group_column']
    if group_column not in df.columns:
        raise KeyError(
            f"'{group_column}' not found in the parasite table. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    df = df.dropna(subset=[group_column])
    if len(df) == 0:
        raise ValueError(
            f"Every parasite row has an empty '{group_column}'. Check the "
            f"cell_plate_metadata / pathogen_plate_metadata / "
            f"treatment_plate_metadata well maps."
        )

    # ---- vacuole assignment ---------------------------------------------
    df, vacuole_key_used, link_distance = _assign_vacuole_ids(
        df,
        compartment=compartment,
        vacuole_key=settings['vacuole_key'],
        link_distance=settings['vacuole_link_distance'],
        link_factor=settings['vacuole_link_factor'],
        verbose=settings['verbose'],
    )

    if settings['parasite_count_column'] is not None:
        count_column = settings['parasite_count_column']
        if count_column not in df.columns:
            raise KeyError(
                f"parasite_count_column '{count_column}' is not a column of "
                f"'{parasite_table}'."
            )
        counts = df.groupby('vacuole_id', sort=False)[count_column].max()
    else:
        counts = df.groupby('vacuole_id', sort=False)['vacuole_id'].size()

    identity_columns = ['plateID', 'rowID', 'columnID', 'fieldID', 'prc',
                        'prcf', group_column]
    if 'cell_id' in df.columns:
        identity_columns.append('cell_id')
    identity_columns = [c for c in dict.fromkeys(identity_columns) if c in df.columns]

    vacuoles = df.groupby('vacuole_id', sort=False)[identity_columns].first()
    vacuoles['n_parasites'] = counts.astype(int)
    if area_column in df.columns:
        vacuoles['total_parasite_area'] = df.groupby('vacuole_id',
                                                     sort=False)[area_column].sum()
    vacuoles = vacuoles.reset_index()

    vacuoles['replication_bucket'] = pd.Categorical(
        [_replication_bucket(n, settings['max_parasites_per_vacuole'])
         for n in vacuoles['n_parasites']],
        categories=buckets, ordered=True
    )
    vacuoles['is_power_of_two'] = (
        vacuoles['replication_bucket'].astype(str) != 'non_power_of_two'
    )
    vacuoles['doublings'] = np.where(
        vacuoles['is_power_of_two'],
        np.log2(vacuoles['n_parasites'].to_numpy(dtype=float)),
        np.nan
    )

    # ---- wells that hold host cells but no vacuoles ----------------------
    seed_wells = None
    if cell_frames:
        cells = pd.concat(cell_frames, axis=0, ignore_index=True)
        cells['prc'] = (cells['plateID'].astype(str) + '_'
                        + cells['rowID'].astype(str) + '_'
                        + cells['columnID'].astype(str))
        cells = annotate_conditions(
            df=cells,
            cells=settings['cell_types'],
            cell_loc=settings['cell_plate_metadata'],
            pathogens=settings['pathogen_types'],
            pathogen_loc=settings['pathogen_plate_metadata'],
            treatments=settings['treatments'],
            treatment_loc=settings['treatment_plate_metadata'],
        )
        if group_column in cells.columns:
            seed_wells = cells.dropna(subset=[group_column])[
                ['plateID', 'rowID', 'columnID', 'prc', group_column]
            ].drop_duplicates()

    wells = _replication_well_distribution(
        vacuoles, group_column, buckets,
        non_power_of_two_warn=settings['non_power_of_two_warn'],
        wells=seed_wells,
    )
    summary = _replication_summary(
        vacuoles, group_column, buckets,
        non_power_of_two_warn=settings['non_power_of_two_warn'],
    )
    comparisons = _replication_compare_conditions(
        vacuoles, group_column, buckets, verbose=settings['verbose']
    )

    # ---- figures ---------------------------------------------------------
    prc_column = 'plateID' if settings['level'] == 'plate' else 'prc'

    _, _, well_fig = _replication_stacked_bars(
        settings, vacuoles, group_column='prc', prc_column='prc',
        level='object', cmap=settings['cmap'],
        title='Parasites per vacuole — per well',
    )
    results_df, pairwise_df, group_fig = _replication_stacked_bars(
        settings, vacuoles, group_column=group_column, prc_column=prc_column,
        level=settings['level'], cmap=settings['cmap'],
        title='Parasites per vacuole — by condition',
    )

    output = {
        'vacuoles': vacuoles,
        'wells': wells,
        'summary': summary,
        'comparisons': comparisons,
        'chi_squared': results_df,
        'chi_squared_pairwise': pairwise_df,
        'vacuole_key': vacuole_key_used,
        'vacuole_link_distance': link_distance,
        'figures': {'per_well': well_fig, 'by_condition': group_fig},
    }

    if settings['save']:
        output_dir = os.path.join(settings['src'][0], 'results',
                                  'analyze_replication')
        os.makedirs(output_dir, exist_ok=True)
        vacuoles.to_csv(os.path.join(output_dir, 'vacuole_counts.csv'), index=False)
        wells.to_csv(os.path.join(output_dir, 'well_distribution.csv'), index=False)
        summary.to_csv(os.path.join(output_dir, 'condition_summary.csv'), index=False)
        comparisons.to_csv(os.path.join(output_dir, 'condition_comparisons.csv'), index=False)
        results_df.to_csv(os.path.join(output_dir, 'chi_squared_results.csv'), index=False)
        pairwise_df.to_csv(os.path.join(output_dir, 'chi_squared_pairwise_results.csv'), index=False)
        well_fig.savefig(os.path.join(output_dir, 'parasites_per_vacuole_per_well.pdf'),
                         dpi=300, bbox_inches='tight')
        group_fig.savefig(os.path.join(output_dir, 'parasites_per_vacuole_by_condition.pdf'),
                          dpi=300, bbox_inches='tight')
        print(f"Replication assay results saved to {output_dir}")

    if settings['verbose']:
        flagged = wells.loc[wells['qc_flag_non_power_of_two'], 'prc'].tolist()
        if flagged:
            print(f"QC: {len(flagged)} well(s) above the non_power_of_two "
                  f"threshold ({settings['non_power_of_two_warn']:.0%}): "
                  f"{', '.join(map(str, flagged))}")

    plt.show()
    # The figures stay usable (savefig works on a closed figure); closing them
    # keeps a batch run over many plates from accumulating open figures.
    plt.close(well_fig)
    plt.close(group_fig)

    return output


# ===========================================================================
# Invasion assay (Toxoplasma) — two-colour outside/inside stain
# ===========================================================================

def _set_analyze_invasion_defaults(settings):
    """Fallback defaults for :func:`analyze_invasion`.

    The canonical copy of every pipeline's defaults lives in
    :mod:`spacr.settings`; this one is used only while
    ``spacr.settings.set_analyze_invasion_defaults`` does not exist yet, so the
    assay is runnable from the API before the GUI knobs are registered. Once
    :mod:`spacr.settings` defines it, that version wins.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('parasite_table', 'pathogen')
    settings.setdefault('compartment', 'pathogen')
    settings.setdefault('outside_channel', 1)
    settings.setdefault('total_channel', 0)
    settings.setdefault('intensity_statistic', 'auto')
    settings.setdefault('background_correction', 'none')
    settings.setdefault('outside_threshold_method', 'otsu')
    settings.setdefault('outside_threshold', None)
    settings.setdefault('control_wells', None)
    settings.setdefault('control_quantile', 0.99)
    settings.setdefault('min_control_objects', 10)
    settings.setdefault('min_objects_for_threshold', 10)
    settings.setdefault('min_objects_for_bimodality', 30)
    settings.setdefault('bimodality_cutoff', 5.0 / 9.0)
    settings.setdefault('threshold_agreement_tolerance', 0.5)
    settings.setdefault('threshold_sensitivity', 0.25)
    settings.setdefault('inflation_warn', 0.05)
    settings.setdefault('min_parasites_per_well', 50)
    settings.setdefault('min_parasite_area', 0)
    settings.setdefault('max_parasite_area', None)
    settings.setdefault('min_total_intensity', None)
    settings.setdefault('extracellular_class', 'attached')
    settings.setdefault('seed_wells_from_cells', True)
    settings.setdefault('cell_types', ['Hela'])
    settings.setdefault('cell_plate_metadata', None)
    settings.setdefault('pathogen_types', ['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata', [['c1'], ['c2']])
    settings.setdefault('treatments', None)
    settings.setdefault('treatment_plate_metadata', None)
    settings.setdefault('group_column', 'condition')
    settings.setdefault('level', 'object')
    settings.setdefault('change_plate', False)
    settings.setdefault('qc_plot_max_panels', 12)
    settings.setdefault('cmap', 'viridis')
    settings.setdefault('save', True)
    settings.setdefault('verbose', False)
    return settings


# Per-object statistics of the outside-stain channel, in the naming
# :func:`spacr.measure._intensity_measurements` actually writes.
#
# Careful with the word "outside": measure.py's ``<object>_channel_<n>_outside_*``
# columns are the intensity of a five-pixel ring *outside the object's own
# mask* (:func:`spacr.measure._outside_intensity`) in whatever channel is
# named. They are a local background estimate, and they have nothing to do
# with the outside/inside *stain* of this assay. The assay's outside stain is
# a channel, selected with ``outside_channel``; the statistics below read the
# parasite's own pixels in that channel.
_INVASION_STATISTIC_TEMPLATES = {
    'periphery_95': '{compartment}_channel_{channel}_periphery_percentile_95',
    'periphery_85': '{compartment}_channel_{channel}_periphery_percentile_85',
    'periphery_mean': '{compartment}_channel_{channel}_periphery_mean',
    'percentile_95': '{compartment}_channel_{channel}_percentile_95',
    'percentile_85': '{compartment}_channel_{channel}_percentile_85',
    'max': '{compartment}_channel_{channel}_max_intensity',
    'mean': '{compartment}_channel_{channel}_mean_intensity',
    'median': '{compartment}_channel_{channel}_median_intensity',
    'integrated': '{compartment}_channel_{channel}_integrated_intensity',
}

# Resolution order for intensity_statistic='auto'. The order is the argument
# in :func:`_resolve_invasion_intensity_column`.
_INVASION_STATISTIC_AUTO_ORDER = ('periphery_95', 'percentile_95', 'mean')
_INVASION_LEGACY_STATISTIC_TEMPLATES = {
    'periphery_95': '{compartment}_channel_{channel}_periphery_95_percentile',
    'periphery_85': '{compartment}_channel_{channel}_periphery_85_percentile',
}

_INVASION_CLASSES = ['attached', 'invaded']


def _resolve_invasion_intensity_column(df, compartment, channel,
                                       statistic='auto', verbose=False):
    """Pick the per-object outside-channel statistic and say which one it is.

    **An outside stain is a rim stain**, and that fact decides this entirely.
    The antibody binds the parasite surface before permeabilisation, so the
    signal lives on the object's boundary while the object's interior stays at
    background. Three consequences, in the order that matters:

    * ``mean_intensity`` averages the rim over the *whole* object. Signal
      scales with the perimeter and the denominator with the area, so the mean
      of a fixed surface stain falls roughly as ``1/radius``: a bigger parasite
      reads dimmer than a smaller one that is stained identically. That is a
      size-dependent bias pushing objects *below* the threshold, which is the
      exact direction that manufactures false "invaded" calls. It is the last
      resort, and choosing it prints a warning.
    * ``max_intensity`` is one pixel, so one hot pixel or cosmic ray sets it.
    * ``percentile_95`` samples the brightest 5% of the object's pixels —
      which is where a rim sits — over enough pixels to be stable. It is the
      right default when nothing better exists.
    * ``periphery_percentile_95`` (:func:`spacr.measure._periphery_intensity`)
      is measured *only* on the object's boundary ring, so it does not depend
      on the object's area at all. When measure_crop wrote it, it wins.

    :param df: Raw per-parasite DataFrame.
    :param compartment: Object prefix, e.g. ``'pathogen'``.
    :param channel: Index of the outside-stain channel.
    :param statistic: ``'auto'``, a key of
        :data:`_INVASION_STATISTIC_TEMPLATES`, or a literal column name.
    :param verbose: Print the resolved column.
    :returns: ``(column_name, statistic_name)``.
    :raises KeyError: when the requested statistic is not in the table.
    """
    def _template(name):
        return _INVASION_STATISTIC_TEMPLATES[name].format(
            compartment=compartment, channel=channel)

    def _candidates(name):
        yield _template(name)
        legacy = _INVASION_LEGACY_STATISTIC_TEMPLATES.get(name)
        if legacy is not None:
            yield legacy.format(compartment=compartment, channel=channel)

    if statistic == 'auto':
        for name in _INVASION_STATISTIC_AUTO_ORDER:
            column = next(
                (
                    candidate
                    for candidate in _candidates(name)
                    if candidate in df.columns and df[candidate].notna().any()
                ),
                None,
            )
            if column is not None:
                if name == 'mean':
                    print(
                        "WARNING: falling back to the object MEAN of the "
                        f"outside channel ('{column}'). An outside stain is a "
                        "rim stain, so the mean is diluted by the parasite's "
                        "unstained interior and large parasites read dimmer "
                        "than small ones stained identically — a bias toward "
                        "calling outside parasites invaded. Re-run measure "
                        "with intensity features so percentile_95 / "
                            "periphery_percentile_95 exist."
                    )
                if verbose:
                    print(f"Outside-stain statistic: '{column}' ({name})")
                return column, name
        raise KeyError(
            f"No usable outside-channel statistic for compartment "
            f"'{compartment}' channel {channel}. Tried "
            + ', '.join(_template(n) for n in _INVASION_STATISTIC_AUTO_ORDER)
            + ". Check 'outside_channel', or name a column with "
              "'intensity_statistic'."
        )

    if statistic in _INVASION_STATISTIC_TEMPLATES:
        column = next(
            (candidate for candidate in _candidates(statistic)
             if candidate in df.columns),
            None,
        )
        if column is None:
            raise KeyError(
                f"intensity_statistic '{statistic}' resolves to column "
                f"'{_template(statistic)}', which is not in the parasite table."
            )
        if verbose:
            print(f"Outside-stain statistic: '{column}' ({statistic})")
        return column, statistic

    if statistic in df.columns:
        if verbose:
            print(f"Outside-stain statistic: '{statistic}' (custom column)")
        return statistic, 'custom'

    raise KeyError(
        f"intensity_statistic '{statistic}' is neither one of "
        f"{sorted(_INVASION_STATISTIC_TEMPLATES)} nor a column of the "
        f"parasite table."
    )


def _resolve_invasion_background_column(df, compartment, channel,
                                        background='none'):
    """Locate the per-object local-background column, or ``None``.

    ``'auto'`` uses ``<compartment>_channel_<n>_outside_percentile_50`` — the
    median of the five-pixel ring *outside* the parasite mask in the outside
    stain's channel — which removes a per-field background offset without a
    flat-field image.

    It is off by default on purpose. A brightly stained *attached* parasite
    carries an antibody halo that reaches into that same ring, so subtracting
    the ring suppresses exactly the objects the assay must keep above the
    threshold. Turn it on when the background varies more than the halo
    bleeds, and check the per-field thresholds afterwards.

    :param df: Raw per-parasite DataFrame.
    :param compartment: Object prefix.
    :param channel: Outside-stain channel index.
    :param background: ``'none'``/``None``/``False``, ``'auto'``, or a column name.
    :returns: column name or ``None``.
    :raises KeyError: when a named column is not in the table.
    """
    if background in (None, False, 'none', 'None', ''):
        return None
    if background == 'auto':
        # Measurement columns are canonicalised on database read. Keep the
        # legacy spelling as a fallback for direct DataFrame callers and old
        # databases that have not passed through that normalisation yet.
        for suffix in (
            'outside_percentile_50',
            'outside_50_percentile',
            'outside_mean',
        ):
            column = f'{compartment}_channel_{channel}_{suffix}'
            if column in df.columns and df[column].notna().any():
                return column
        print(
            "WARNING: background_correction='auto' found no "
            f"'{compartment}_channel_{channel}_outside_*' column; continuing "
            "with raw intensities."
        )
        return None
    if background in df.columns:
        return background
    raise KeyError(
        f"background_correction '{background}' is not a column of the "
        f"parasite table."
    )


def _bimodality_coefficient(values, min_objects=30):
    """Sarle's bimodality coefficient ``(skew**2 + 1) / kurtosis``, uncorrected.

    This is the assay's own answer to "are there two populations here at all?".
    A perfect two-point mixture returns exactly 1.0 at any sample size and any
    mixing ratio — 90/10 scores the same as 50/50 — while a single normal
    population returns about 1/3. The conventional cutoff is 5/9.

    The *small-sample-corrected* form usually quoted as Sarle's coefficient
    divides by ``kurtosis + 3(n-1)**2/((n-2)(n-3))``. It is not used here
    because that correction cannot reach 5/9 below roughly fifteen objects: a
    field holding a dozen parasites split perfectly into two populations would
    score 0.35 and be reported as unimodal. The uncorrected form has the
    opposite failure — on a genuinely unimodal sample it exceeds 5/9 about 45%
    of the time at n=10 and 15% at n=20 — so it is simply refused below
    ``min_objects``, which is the honest answer rather than a confident wrong
    one.

    :param values: 1-D array of the outside-channel statistic.
    :param min_objects: Below this many finite values, return NaN rather than
        a number the sample cannot support. Default 30, where a unimodal
        sample false-passes about 5% of the time.
    :returns: float coefficient, or NaN when it cannot be computed.
    """
    from scipy.stats import kurtosis, skew

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < max(4, int(min_objects)):
        return float('nan')
    if np.ptp(values) == 0:
        # One value repeated is one population, but skew/kurtosis are 0/0
        # there; say "no evidence of two populations" explicitly.
        return 0.0
    g1 = float(skew(values, bias=True))
    g2 = float(kurtosis(values, fisher=True, bias=True))
    denominator = g2 + 3.0
    if not np.isfinite(denominator) or denominator <= 0:
        return float('nan')
    return float((g1 ** 2 + 1.0) / denominator)


def _invasion_centre_threshold(values, threshold):
    """Move ``threshold`` to the middle of the gap it opens, keeping the split identical.

    skimage's threshold functions histogram their input and return the centre
    of a bin, so on a clean two-population field the returned value lands on
    the *upper edge of the dim population* rather than in the empty space
    between the two. The split is right and the placement is an artefact of
    the 256-bin histogram, but the placement is what the sensitivity bracket
    in :func:`_invasion_threshold_span` perturbs: a threshold sitting on top
    of the dim population reclassifies that whole population the moment it is
    nudged down, and the assay would report every clean field as
    threshold-sensitive.

    Recentring is exact rather than cosmetic. Everything at or below the
    original threshold stays below the new one and everything above stays
    above — the midpoint of ``(max below, min above)`` lies strictly between
    them — so the classification is untouched and only the margin changes,
    to the largest margin the data allow.

    :param values: 1-D array the threshold was derived from.
    :param threshold: Threshold returned by the chosen method.
    :returns: float, recentred where possible and unchanged otherwise.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(threshold) or values.size == 0:
        return float(threshold)
    below = values[values <= threshold]
    above = values[values > threshold]
    if below.size == 0 or above.size == 0:
        return float(threshold)
    return float((below.max() + above.min()) / 2.0)


def _invasion_threshold(values, method='otsu'):
    """Derive an outside-channel cut from the data alone.

    Every method here is a histogram/valley method that returns a value inside
    the gap between two populations. None of them can tell you whether that
    gap exists — that is what :func:`_bimodality_coefficient` is for, and why
    a threshold is never reported without it.

    The chosen cut is recentred in its own gap by
    :func:`_invasion_centre_threshold`, which changes no classification and
    makes the margin the widest the data support.

    :param values: 1-D array of the outside-channel statistic.
    :param method: ``'otsu'``, ``'triangle'``, ``'li'``, ``'yen'`` or ``'mean'``.
    :returns: float threshold, or NaN when the values cannot support one.
    :raises ValueError: for an unknown method.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if method == 'mean':
        if values.size == 0:
            return float('nan')
        return _invasion_centre_threshold(values, float(values.mean()))

    from skimage.filters import (threshold_li, threshold_otsu,
                                 threshold_triangle, threshold_yen)
    functions = {'otsu': threshold_otsu, 'triangle': threshold_triangle,
                 'li': threshold_li, 'yen': threshold_yen}
    if method not in functions:
        raise ValueError(
            f"outside_threshold_method '{method}' is not one of "
            f"{sorted(list(functions) + ['mean'])}."
        )
    if values.size < 2 or np.unique(values).size < 2:
        return float('nan')
    try:
        threshold = float(functions[method](values))
    except (ValueError, RuntimeError):
        return float('nan')
    return _invasion_centre_threshold(values, threshold)


def _invasion_relative_difference(used, reference):
    """Scale-free distance between two thresholds, in ``[0, 2]``.

    Symmetric in its arguments and defined when either is negative (which a
    background-corrected threshold can be), so it never divides by a value
    that happens to sit near zero.

    :param used: Threshold actually applied.
    :param reference: Threshold it is being judged against.
    :returns: float, or NaN when either input is not finite.
    """
    if not np.isfinite(used) or not np.isfinite(reference):
        return float('nan')
    scale = max(abs(float(used)), abs(float(reference)))
    if scale == 0:
        return 0.0
    return float(abs(float(used) - float(reference)) / scale)


def _invasion_threshold_span(threshold, values, sensitivity):
    """Return the thresholds ``sensitivity`` either side of ``threshold``.

    Reclassifying at these two values is what turns the assay's central
    asymmetry into a number. Raising the outside-channel threshold can only
    move objects from *attached* to *invaded*, so invasion efficiency is
    monotonically non-decreasing in the threshold — and only the upward move
    is dangerous. Lowering it can merely deflate the efficiency, which is the
    conservative direction and never manufactures a result, which is why the
    QC flag built from this pair (``qc_flag_threshold_inflates``) watches the
    high side alone while the low side is reported for context.

    :param threshold: Threshold actually used.
    :param values: The field's outside-channel statistics, used to set a scale
        when the threshold itself is zero.
    :param sensitivity: Relative perturbation, e.g. 0.25 for +/-25%.
    :returns: ``(low, high)`` floats, or ``(nan, nan)``.
    """
    if not np.isfinite(threshold):
        return float('nan'), float('nan')
    scale = abs(float(threshold))
    if scale == 0:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        scale = float(np.std(finite)) if finite.size else 0.0
    delta = float(sensitivity) * scale
    return float(threshold) - delta, float(threshold) + delta


def _invasion_control_mask(df, control_wells):
    """Boolean mask selecting the staining-control wells named in ``control_wells``.

    Accepts, per entry, a plate-row-column key (``'plate1_r1_c12'``), a
    row-column well (``'r1_c12'``), a whole row (``'r1'``) or a whole column
    (``'c12'``) — the same vocabulary the ``*_plate_metadata`` well maps use.

    :param df: Parasite DataFrame carrying ``prc``, ``rowID`` and ``columnID``.
    :param control_wells: str or iterable of str, or None.
    :returns: boolean Series aligned to ``df``.
    """
    mask = pd.Series(False, index=df.index)
    if control_wells is None:
        return mask
    if isinstance(control_wells, str):
        control_wells = [control_wells]
    if len(control_wells) == 0:
        return mask

    prc = df['prc'].astype(str)
    rows = df['rowID'].astype(str)
    columns = df['columnID'].astype(str)
    wells = rows + '_' + columns
    for spec in control_wells:
        spec = str(spec)
        mask |= (prc == spec) | (wells == spec) | (rows == spec) | (columns == spec)
    return mask


def _invasion_field_thresholds(df, value_column, settings, control_thresholds):
    """Resolve the outside-channel threshold for every field, and say where it came from.

    **The threshold is per field, not per plate.** Illumination and antibody
    penetration vary field to field; a single plate-wide cut turns an
    illumination gradient into an invasion gradient, because the dim corner of
    the plate loses its outside signal first and its parasites are then all
    scored as invaded. The one exception is a control-derived threshold, which
    is global by construction — the controls are separate wells — and which is
    therefore cross-checked against each field's own automatic threshold by
    the ``qc_flag_threshold_disagrees`` column rather than trusted blindly.

    Resolution order, per field:

    1. ``outside_threshold`` when the caller fixed one (source ``'fixed'``).
    2. the control-derived cut for that plate (source ``'control'``).
    3. the field's own automatic threshold (source ``'field'``), falling back
       to the well's (``'well'``) and then the plate's (``'plate'``) when the
       field holds fewer than ``min_objects_for_threshold`` objects — Otsu on
       four parasites is not a threshold.
    4. nothing (source ``'none'``): the objects are left unclassified rather
       than split on a number that does not exist.

    :param df: Per-parasite DataFrame with ``prcf``, ``prc``, ``plateID``.
    :param value_column: Column holding the outside-channel statistic.
    :param settings: Resolved settings dict.
    :param control_thresholds: ``{plateID: threshold}`` from the control wells.
    :returns: DataFrame with one row per field.
    """
    method = settings['outside_threshold_method']
    floor = int(settings['min_objects_for_threshold'])
    fixed = settings['outside_threshold']
    cutoff = float(settings['bimodality_cutoff'])
    min_bimodal = int(settings['min_objects_for_bimodality'])
    tolerance = float(settings['threshold_agreement_tolerance'])

    def _auto(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < floor:
            return float('nan')
        return _invasion_threshold(values, method)

    plate_auto = {str(key): _auto(group[value_column])
                  for key, group in df.groupby('plateID', sort=False)}
    well_auto = {str(key): _auto(group[value_column])
                 for key, group in df.groupby('prc', sort=False)}

    identity = ['plateID', 'rowID', 'columnID', 'fieldID', 'prc', 'prcf']
    rows = []
    for _, group in df.groupby('prcf', sort=False):
        values = group[value_column].to_numpy(dtype=float)
        record = {column: group[column].iloc[0] for column in identity}

        automatic = _auto(values)
        automatic_source = 'field'
        if not np.isfinite(automatic):
            automatic = well_auto.get(str(record['prc']), float('nan'))
            automatic_source = 'well'
        if not np.isfinite(automatic):
            automatic = plate_auto.get(str(record['plateID']), float('nan'))
            automatic_source = 'plate'
        if not np.isfinite(automatic):
            automatic_source = 'none'

        control = control_thresholds.get(str(record['plateID']), float('nan'))

        if fixed is not None:
            threshold, source = float(fixed), 'fixed'
        elif np.isfinite(control):
            threshold, source = float(control), 'control'
        else:
            threshold, source = automatic, automatic_source

        # A control-derived cut is the honest negative distribution, so it is
        # what an automatic cut should be judged against when it exists.
        reference = control if np.isfinite(control) else automatic

        low, high = _invasion_threshold_span(
            threshold, values, settings['threshold_sensitivity'])
        coefficient = _bimodality_coefficient(values, min_bimodal)
        difference = _invasion_relative_difference(threshold, reference)

        record.update({
            'n_objects': int(len(group)),
            'threshold': float(threshold),
            'threshold_source': source,
            'threshold_low': low,
            'threshold_high': high,
            'automatic_threshold': float(automatic),
            'automatic_source': automatic_source,
            'control_threshold': float(control),
            'reference_threshold': float(reference),
            'threshold_relative_difference': difference,
            'bimodality_coefficient': coefficient,
            'qc_flag_unimodal': bool(not (coefficient > cutoff)),
            'qc_flag_threshold_disagrees': bool(
                np.isfinite(difference) and difference > tolerance),
            'qc_flag_no_threshold': bool(not np.isfinite(threshold)),
        })
        rows.append(record)

    return pd.DataFrame(rows)


def _invasion_classify(df, fields, value_column, extracellular_class):
    """Attach ``invasion_class`` and the threshold that produced it to every parasite.

    An object is called **outside/attached** when its outside-channel
    statistic is strictly greater than its field's threshold, and
    **inside/invaded** when it is not. Note which way round the evidence runs:
    *attached* is a positive observation, *invaded* is the absence of one. A
    parasite that is genuinely outside but stained weakly — poor antibody
    penetration, a focal plane away from its equator, photobleaching, a
    low-expressing strain — falls below the threshold and is scored invaded,
    so every failure of the outside stain inflates invasion efficiency and
    none of them deflate it.

    Objects that overlap no host cell cannot have invaded anything, and
    ``extracellular_class`` decides what happens to them: ``'attached'``
    scores them attached whatever the stain says (the default, and the
    biologically literal reading), ``'exclude'`` drops them before the caller
    gets here, and ``'classify'`` leaves them to the stain, which is what you
    want when the cell mask is the unreliable part.

    :param df: Per-parasite DataFrame with ``prcf`` and ``no_host_cell``.
    :param fields: Per-field threshold table from :func:`_invasion_field_thresholds`.
    :param value_column: Column holding the outside-channel statistic.
    :param extracellular_class: ``'attached'``, ``'classify'`` (``'exclude'``
        is applied by the caller).
    :returns: DataFrame copy with the classification columns added.
    """
    from .io import _report_fan_out

    columns = ['prcf', 'threshold', 'threshold_source', 'threshold_low',
               'threshold_high', 'automatic_threshold', 'reference_threshold',
               'bimodality_coefficient']
    merged = df.merge(fields[columns], on='prcf', how='left')
    # ``fields`` comes out of a groupby on prcf so it holds one row per key and
    # this join cannot grow. Checked anyway, with io's own helper: if a caller
    # ever hands in a field table assembled some other way, a duplicated prcf
    # would silently duplicate every parasite and inflate n_total.
    _report_fan_out(df, merged, ['prcf'], left_name='parasite',
                    right_name='the field threshold table')
    df = merged

    values = df[value_column].to_numpy(dtype=float)
    thresholds = df['threshold'].to_numpy(dtype=float)
    usable = np.isfinite(values) & np.isfinite(thresholds)

    is_outside = values > thresholds
    outside_low = values > df['threshold_low'].to_numpy(dtype=float)
    outside_high = values > df['threshold_high'].to_numpy(dtype=float)

    if extracellular_class == 'attached':
        forced = df['no_host_cell'].to_numpy(dtype=bool)
        is_outside = is_outside | forced
        outside_low = outside_low | forced
        outside_high = outside_high | forced
        usable = usable | forced

    df['is_outside'] = np.where(usable, is_outside, np.nan)
    df['invasion_class'] = np.where(
        ~usable, 'unclassified', np.where(is_outside, 'attached', 'invaded'))
    df['invasion_class'] = pd.Categorical(
        df['invasion_class'],
        categories=_INVASION_CLASSES + ['unclassified'], ordered=False)
    # The two sensitivity columns exist so a reader can see how much of the
    # reported efficiency is the threshold rather than the biology.
    df['is_outside_low_threshold'] = np.where(usable, outside_low, np.nan)
    df['is_outside_high_threshold'] = np.where(usable, outside_high, np.nan)
    return df


def _invasion_efficiency(n_invaded, n_total):
    """``n_invaded / n_total``, or NaN when the well scored nothing.

    NaN rather than 0.0 on purpose: a well with no classified parasites has
    not observed zero invasion, it has observed nothing, and 0.0 would read as
    a result in every downstream mean and plot.

    :param n_invaded: Parasites scored invaded.
    :param n_total: Parasites scored at all (attached + invaded).
    :returns: float efficiency or NaN.
    """
    n_total = int(n_total)
    if n_total <= 0:
        return float('nan')
    return float(n_invaded) / float(n_total)


def _finite_median(values):
    """Return the median of finite values, or NaN when none are available."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float('nan')


def _invasion_well_table(parasites, fields, group_column, settings,
                         seed_wells=None):
    """Summarize invasion per well, with the denominator and the QC in the same row.

    Invasion efficiency is a proportion and it is quoted here **with
    ``n_total``**, because 90% from ten parasites and 90% from four thousand
    are not the same result and nothing downstream can tell them apart from
    the ratio alone. Four QC columns say when the ratio should not be quoted
    at all:

    * ``qc_flag_low_total`` — fewer than ``min_parasites_per_well`` scored
      parasites. At n=50 the 95% interval on a proportion near a half is still
      about +/-14 percentage points, which is wider than most real effects.
    * ``qc_flag_unimodal`` — the well's outside-channel distribution shows no
      two populations, so the threshold splits one population at an arbitrary
      place. Note that an all-invaded well is legitimately unimodal and will
      flag: it carries no internal evidence that its own threshold is right.
    * ``qc_flag_threshold_disagrees`` — the threshold applied sits further
      than ``threshold_agreement_tolerance`` from the reference (the
      control-derived cut when controls exist, otherwise the field's own
      automatic cut).
    * ``qc_flag_threshold_inflates`` — raising the threshold by
      ``threshold_sensitivity`` would add more than ``inflation_warn`` to this
      well's efficiency, so the threshold is sitting inside the data rather
      than in a gap. Only the upward move is watched: lowering a threshold can
      only turn invaded back into attached, which is the safe direction.

    :param parasites: Classified per-parasite DataFrame.
    :param fields: Per-field threshold table.
    :param group_column: Condition column carried onto each well row.
    :param settings: Resolved settings dict.
    :param seed_wells: Optional ``(plateID, rowID, columnID, prc,
        group_column)`` rows so wells holding host cells but no parasites
        appear with a zero denominator instead of vanishing.
    :returns: per-well DataFrame.
    """
    identity = ['plateID', 'rowID', 'columnID', 'prc', group_column]
    cutoff = float(settings['bimodality_cutoff'])
    min_bimodal = int(settings['min_objects_for_bimodality'])
    tolerance = float(settings['threshold_agreement_tolerance'])
    warn = float(settings['inflation_warn'])
    minimum = int(settings['min_parasites_per_well'])

    seeded = {}
    if seed_wells is not None and len(seed_wells) > 0:
        for record in seed_wells[identity].drop_duplicates().to_dict('records'):
            seeded[(record['prc'], record[group_column])] = record
    if len(parasites) > 0:
        for key, group in parasites.groupby(['prc', group_column],
                                            dropna=False, sort=False):
            seeded[key] = {column: group[column].iloc[0] for column in identity}

    field_index = fields.set_index('prcf') if len(fields) else fields

    rows = []
    for key, record in seeded.items():
        prc, group_value = key
        subset = parasites[(parasites['prc'] == prc)
                           & (parasites[group_column] == group_value)]
        classes = subset['invasion_class'].astype(str)
        n_attached = int((classes == 'attached').sum())
        n_invaded = int((classes == 'invaded').sum())
        n_total = n_attached + n_invaded

        row = dict(record)
        row['n_objects'] = int(len(subset))
        row['n_attached'] = n_attached
        row['n_invaded'] = n_invaded
        row['n_total'] = n_total
        row['n_unclassified'] = int(len(subset)) - n_total
        row['n_no_host_cell'] = (int(subset['no_host_cell'].sum())
                                 if len(subset) else 0)
        row['n_fields'] = int(subset['prcf'].nunique()) if len(subset) else 0
        row['invasion_efficiency'] = _invasion_efficiency(n_invaded, n_total)

        low = int((subset['is_outside_low_threshold'] == 0).sum()) if len(subset) else 0
        high = int((subset['is_outside_high_threshold'] == 0).sum()) if len(subset) else 0
        row['invasion_efficiency_low_threshold'] = _invasion_efficiency(low, n_total)
        row['invasion_efficiency_high_threshold'] = _invasion_efficiency(high, n_total)

        if len(subset):
            row['outside_intensity_median'] = _finite_median(
                subset['outside_intensity'])
            row['bimodality_coefficient'] = _bimodality_coefficient(
                subset['outside_intensity'].to_numpy(dtype=float), min_bimodal)
            row['threshold_median'] = _finite_median(subset['threshold'])
            row['reference_threshold_median'] = _finite_median(
                subset['reference_threshold'])
            sources = sorted(set(subset['threshold_source'].astype(str)))
            row['threshold_source'] = sources[0] if len(sources) == 1 else 'mixed'
        else:
            row['outside_intensity_median'] = float('nan')
            row['bimodality_coefficient'] = float('nan')
            row['threshold_median'] = float('nan')
            row['reference_threshold_median'] = float('nan')
            row['threshold_source'] = 'none'

        row['threshold_relative_difference'] = _invasion_relative_difference(
            row['threshold_median'], row['reference_threshold_median'])

        if len(field_index) and len(subset):
            prcfs = [p for p in subset['prcf'].unique() if p in field_index.index]
            row['n_fields_unimodal'] = int(
                field_index.loc[prcfs, 'qc_flag_unimodal'].sum()) if prcfs else 0
        else:
            row['n_fields_unimodal'] = 0

        # Only the upward move counts. Raising the threshold turns attached
        # into invaded and inflates the efficiency; lowering it can only do
        # the opposite, which is the direction that never invents a result.
        inflation = (row['invasion_efficiency_high_threshold']
                     - row['invasion_efficiency'])
        row['invasion_efficiency_inflation'] = inflation

        row['qc_flag_low_total'] = bool(n_total < minimum)
        row['qc_flag_unimodal'] = bool(
            not (row['bimodality_coefficient'] > cutoff))
        row['qc_flag_threshold_disagrees'] = bool(
            np.isfinite(row['threshold_relative_difference'])
            and row['threshold_relative_difference'] > tolerance)
        row['qc_flag_threshold_inflates'] = bool(
            np.isfinite(inflation) and inflation > warn)
        flags = [name.replace('qc_flag_', '') for name in
                 ('qc_flag_low_total', 'qc_flag_unimodal',
                  'qc_flag_threshold_disagrees', 'qc_flag_threshold_inflates')
                 if row[name]]
        row['qc_flags'] = ';'.join(flags)
        row['qc_pass'] = not flags
        rows.append(row)

    columns = identity + [
        'n_objects', 'n_attached', 'n_invaded', 'n_total', 'n_unclassified',
        'n_no_host_cell', 'n_fields', 'invasion_efficiency',
        'invasion_efficiency_low_threshold', 'invasion_efficiency_high_threshold',
        'invasion_efficiency_inflation', 'outside_intensity_median',
        'bimodality_coefficient', 'threshold_median', 'threshold_source',
        'reference_threshold_median', 'threshold_relative_difference',
        'n_fields_unimodal', 'qc_flag_low_total', 'qc_flag_unimodal',
        'qc_flag_threshold_disagrees', 'qc_flag_threshold_inflates',
        'qc_flags', 'qc_pass']
    return pd.DataFrame(rows, columns=columns)


def _invasion_summary(wells, group_column):
    """Collapse the per-well table to one row per experimental condition.

    Two efficiencies are reported and they answer different questions.
    ``invasion_efficiency`` is the mean of the per-well efficiencies — the
    well is the unit of replication, so this is the number to quote, and it
    comes with an SD, an SEM and ``n_wells`` to go with it.
    ``invasion_efficiency_pooled`` pools every parasite in the condition; it
    is the number a chi-squared on raw counts is implicitly about, and it is
    here so the two can be compared rather than confused.

    :param wells: Per-well DataFrame.
    :param group_column: Condition column.
    :returns: per-condition DataFrame.
    """
    rows = []
    for group_value, subset in wells.groupby(group_column, dropna=False,
                                             sort=False):
        efficiencies = subset['invasion_efficiency'].to_numpy(dtype=float)
        efficiencies = efficiencies[np.isfinite(efficiencies)]
        n_attached = int(subset['n_attached'].sum())
        n_invaded = int(subset['n_invaded'].sum())
        n_total = n_attached + n_invaded
        rows.append({
            group_column: group_value,
            'n_wells': int(len(subset)),
            'n_wells_scored': int(len(efficiencies)),
            'n_wells_flagged': int((~subset['qc_pass']).sum()),
            'n_attached': n_attached,
            'n_invaded': n_invaded,
            'n_total': n_total,
            'n_objects': int(subset['n_objects'].sum()),
            'invasion_efficiency': (float(efficiencies.mean())
                                    if efficiencies.size else float('nan')),
            'invasion_efficiency_median': (float(np.median(efficiencies))
                                           if efficiencies.size else float('nan')),
            'invasion_efficiency_sd': (float(efficiencies.std(ddof=1))
                                       if efficiencies.size > 1 else float('nan')),
            'invasion_efficiency_sem': (
                float(efficiencies.std(ddof=1) / np.sqrt(efficiencies.size))
                if efficiencies.size > 1 else float('nan')),
            'invasion_efficiency_pooled': _invasion_efficiency(n_invaded, n_total),
            'n_wells_low_total': int(subset['qc_flag_low_total'].sum()),
            'n_wells_unimodal': int(subset['qc_flag_unimodal'].sum()),
        })
    return pd.DataFrame(rows)


def _invasion_compare_conditions(wells, group_column, min_wells=2,
                                 verbose=False):
    """Compare invasion efficiency between conditions, **using the well as the unit**.

    Parasites inside one well are not independent observations. They share a
    coverslip, a field of antibody, a focal plane, a monolayer and a
    multiplicity of infection, so the well is the unit of replication and the
    number of wells is the sample size. A chi-squared on pooled parasite
    counts treats four thousand parasites in three wells as four thousand
    independent draws; its standard error is therefore too small by roughly
    the square root of the number of parasites per well, and it will call
    almost any pair of conditions significantly different — including two
    halves of the same plate.

    So the reported test is a **Mann-Whitney U on the per-well invasion
    efficiencies**, ``n1`` and ``n2`` being wells and not parasites. It is
    rank-based, so it needs no normality assumption on a bounded proportion,
    and it matches the ordered comparison used by
    :func:`_replication_compare_conditions`. Three wells against three wells
    cannot reach p < 0.05 two-sided, and that is the honest answer rather than
    a defect.

    ``pooled_chi_squared_p_value`` is computed on the pooled parasite counts
    and reported alongside **only** so the inflation is visible. It is not the
    result. The per-well counts travel in the ``wells`` table for anyone who
    wants to weight the wells or fit a mixed model.

    :param wells: Per-well DataFrame with ``invasion_efficiency``,
        ``n_attached`` and ``n_invaded``.
    :param group_column: Condition column.
    :param min_wells: Wells required per side before the test is run. Default 2.
    :param verbose: Print the resulting table.
    :returns: DataFrame with one row per condition pair; empty (with the full
        column set) when there are fewer than two conditions.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    from .sp_stats import choose_p_adjust_method

    columns = ['group1', 'group2', 'test', 'unit_of_replication',
               'n_wells_1', 'n_wells_2', 'n_parasites_1', 'n_parasites_2',
               'mean_efficiency_1', 'mean_efficiency_2',
               'median_efficiency_1', 'median_efficiency_2',
               'efficiency_difference', 'u_statistic', 'p_value',
               'rank_biserial', 'pooled_efficiency_1', 'pooled_efficiency_2',
               'pooled_chi_squared_stat', 'pooled_chi_squared_p_value',
               'n_wells_flagged_1', 'n_wells_flagged_2',
               'p_value_adj', 'adj']

    groups = list(pd.unique(wells[group_column].dropna()))
    if len(groups) < 2:
        return pd.DataFrame(columns=columns)

    results = []
    for group1, group2 in itertools.combinations(groups, 2):
        left = wells[wells[group_column] == group1]
        right = wells[wells[group_column] == group2]
        left_efficiency = left['invasion_efficiency'].to_numpy(dtype=float)
        right_efficiency = right['invasion_efficiency'].to_numpy(dtype=float)
        left_efficiency = left_efficiency[np.isfinite(left_efficiency)]
        right_efficiency = right_efficiency[np.isfinite(right_efficiency)]

        if len(left_efficiency) >= min_wells and len(right_efficiency) >= min_wells:
            statistic, p_value = mannwhitneyu(left_efficiency, right_efficiency,
                                              alternative='two-sided')
            rank_biserial = (2.0 * statistic
                             / (len(left_efficiency) * len(right_efficiency)) - 1.0)
        else:
            statistic, p_value, rank_biserial = np.nan, np.nan, np.nan

        attached1, invaded1 = int(left['n_attached'].sum()), int(left['n_invaded'].sum())
        attached2, invaded2 = int(right['n_attached'].sum()), int(right['n_invaded'].sum())
        table = np.array([[invaded1, attached1], [invaded2, attached2]],
                         dtype=float)
        if np.all(table.sum(axis=0) > 0) and np.all(table.sum(axis=1) > 0):
            chi2, chi2_p, _, _ = chi2_contingency(table)
        else:
            chi2, chi2_p = np.nan, np.nan

        mean1 = float(left_efficiency.mean()) if len(left_efficiency) else np.nan
        mean2 = float(right_efficiency.mean()) if len(right_efficiency) else np.nan

        results.append({
            'group1': group1,
            'group2': group2,
            'test': 'Mann-Whitney U on per-well invasion efficiency',
            'unit_of_replication': 'well',
            'n_wells_1': int(len(left_efficiency)),
            'n_wells_2': int(len(right_efficiency)),
            'n_parasites_1': attached1 + invaded1,
            'n_parasites_2': attached2 + invaded2,
            'mean_efficiency_1': mean1,
            'mean_efficiency_2': mean2,
            'median_efficiency_1': (float(np.median(left_efficiency))
                                    if len(left_efficiency) else np.nan),
            'median_efficiency_2': (float(np.median(right_efficiency))
                                    if len(right_efficiency) else np.nan),
            'efficiency_difference': mean1 - mean2,
            'u_statistic': statistic,
            'p_value': p_value,
            'rank_biserial': rank_biserial,
            'pooled_efficiency_1': _invasion_efficiency(invaded1,
                                                        invaded1 + attached1),
            'pooled_efficiency_2': _invasion_efficiency(invaded2,
                                                        invaded2 + attached2),
            'pooled_chi_squared_stat': chi2,
            'pooled_chi_squared_p_value': chi2_p,
            'n_wells_flagged_1': int((~left['qc_pass']).sum()),
            'n_wells_flagged_2': int((~right['qc_pass']).sum()),
        })

    results_df = pd.DataFrame(results)
    method = choose_p_adjust_method(
        len(groups), float(wells['n_total'].mean()) if len(wells) else 0.0)
    finite = results_df['p_value'].notna()
    results_df['p_value_adj'] = np.nan
    if finite.any():
        results_df.loc[finite, 'p_value_adj'] = multipletests(
            results_df.loc[finite, 'p_value'].to_numpy(dtype=float),
            method=method
        )[1]
    results_df['adj'] = method

    results_df = results_df[columns]
    if verbose:
        print("\nInvasion efficiency comparisons (unit of replication: well):")
        print(results_df.to_string(index=False))
    return results_df


def _invasion_stacked_bars(settings, parasites, group_column, prc_column,
                           level, cmap, title, denominators=None):
    """Draw stacked attached/invaded proportion bars, reusing the shared plot helper.

    Delegates to :func:`spacr.plot.plot_proportion_stacked_bars` whenever its
    contingency table is well formed — see :func:`_chi_pairwise_is_safe` — and
    draws the bars here when it is not, so a single-condition run or a well
    where every parasite landed in one class still produces a figure instead
    of dying inside the helper.

    Every bar is annotated with its denominator, because a proportion without
    one is not a result.

    :param settings: Settings dict (``verbose`` is read by the helper).
    :param parasites: Classified per-parasite DataFrame, unclassified rows dropped.
    :param group_column: Column forming the bar axis.
    :param prc_column: Per-well identifier used when ``level`` aggregates.
    :param level: ``'object'``, ``'well'`` or ``'plateID'``.
    :param cmap: Matplotlib colormap name.
    :param title: Axes title.
    :param denominators: ``{bar label: n}`` written above each bar.
    :returns: ``(results_df, pairwise_df, fig)``.
    """
    from .plot import plot_proportion_stacked_bars

    working = parasites.copy()
    working['invasion_class'] = (
        working['invasion_class'].cat.remove_unused_categories())
    counts = working.groupby([group_column, 'invasion_class'],
                             observed=True).size().unstack(fill_value=0)

    if counts.size == 0:
        # Nothing was classifiable anywhere — no threshold existed. Say that
        # on the axes rather than dying inside pandas' bar plot, because the
        # unclassified count in the well table is the real answer here.
        fig, axes = plt.subplots(figsize=(12, 8))
        axes.text(0.5, 0.5, 'No parasite could be classified:\nno usable '
                            'outside-stain threshold', ha='center',
                  va='center', transform=axes.transAxes)
        axes.set_xlabel('Group')
        axes.set_ylabel('Proportion')
        axes.set_title(title)
        axes.set_ylim(0, 1.15)
        results_df = pd.DataFrame({'chi_squared_stat': [np.nan],
                                   'p_value': [np.nan],
                                   'degrees_of_freedom': [np.nan]})
        pairwise_df = pd.DataFrame(columns=['Group 1', 'Group 2', 'Test Name',
                                            'p-value', 'p-value_adj', 'adj'])
        return results_df, pairwise_df, fig

    if _chi_pairwise_is_safe(counts):
        results_df, pairwise_df, fig = plot_proportion_stacked_bars(
            settings, working, group_column, bin_column='invasion_class',
            prc_column=prc_column, level=level, cmap=cmap
        )
    else:
        proportions = counts.div(counts.sum(axis=1), axis=0)
        axes = proportions.plot(kind='bar', stacked=True, colormap=cmap,
                                figsize=(12, 8))
        axes.set_xlabel('Group')
        axes.set_ylabel('Proportion')
        fig = plt.gcf()
        results_df = pd.DataFrame({'chi_squared_stat': [np.nan],
                                   'p_value': [np.nan],
                                   'degrees_of_freedom': [np.nan]})
        pairwise_df = pd.DataFrame(columns=['Group 1', 'Group 2', 'Test Name',
                                            'p-value', 'p-value_adj', 'adj'])

    axes = fig.axes[0]
    axes.set_title(title)
    axes.set_ylim(0, 1.15)
    axes.legend(title='Parasite class', bbox_to_anchor=(1.05, 1),
                loc='upper left')

    if denominators:
        for position, tick_label in enumerate(axes.get_xticklabels()):
            total = denominators.get(tick_label.get_text())
            if total is None:
                continue
            axes.text(position, 1.02, f'n={int(total)}', ha='center',
                      va='bottom', fontsize=8, rotation=90)
    return results_df, pairwise_df, fig


def _invasion_threshold_panels(parasites, wells, max_panels=12, cmap='viridis'):
    """Histogram the outside-channel signal per well with the threshold drawn on it.

    This is the figure that lets a reader disagree with the classification.
    Each panel is one well: the distribution the threshold was taken from, a
    solid line at the threshold applied, a dashed line at the reference it was
    judged against, and the well's bimodality coefficient in the title so a
    single smear of signal is visible as a single smear rather than as a
    confident efficiency.

    :param parasites: Classified per-parasite DataFrame.
    :param wells: Per-well DataFrame.
    :param max_panels: Largest number of wells drawn, taken in sorted well
        order so a 384-well plate does not produce a 384-panel figure.
    :param cmap: Matplotlib colormap the histogram bars are drawn from.
    :returns: matplotlib Figure.
    """
    try:
        face = plt.get_cmap(cmap)(0.5)
    except ValueError:
        face = '0.6'
    order = sorted(str(value) for value in wells['prc'].unique())
    truncated = len(order) > int(max_panels)
    order = order[:int(max_panels)]

    n_panels = max(1, len(order))
    n_columns = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns,
                             figsize=(4.0 * n_columns, 3.0 * n_rows),
                             squeeze=False)
    flat = axes.ravel()

    lookup = wells.set_index(wells['prc'].astype(str))
    for index, prc in enumerate(order):
        axis = flat[index]
        subset = parasites[parasites['prc'].astype(str) == prc]
        values = subset['outside_intensity'].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            axis.hist(values, bins=min(40, max(5, values.size // 3)),
                      color=face, edgecolor='none')
        row = lookup.loc[prc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        threshold = float(row['threshold_median'])
        reference = float(row['reference_threshold_median'])
        if np.isfinite(threshold):
            axis.axvline(threshold, color='crimson', linewidth=1.5,
                         label=f"threshold ({row['threshold_source']})")
        if np.isfinite(reference) and reference != threshold:
            axis.axvline(reference, color='steelblue', linewidth=1.2,
                         linestyle='--', label='reference')
        coefficient = float(row['bimodality_coefficient'])
        axis.set_title(
            f"{prc}\nn={int(row['n_total'])}  BC="
            + ('n/a' if not np.isfinite(coefficient) else f'{coefficient:.2f}'),
            fontsize=9)
        axis.set_xlabel('Outside-channel signal')
        axis.set_ylabel('Parasites')
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=7)

    for index in range(len(order), len(flat)):
        flat[index].axis('off')
    if truncated:
        fig.suptitle(f'Outside-stain thresholds (first {len(order)} wells)')
    else:
        fig.suptitle('Outside-stain thresholds')
    fig.tight_layout()
    return fig


def analyze_invasion(settings):
    """Invasion assay: score every parasite attached or invaded and report efficiency per well.

    The red/green invasion assay stains twice. Before permeabilisation an
    antibody reaches only the parasites still **outside** the host cell, so
    those are positive in both channels; the cells are then permeabilised and
    a second antibody stains **all** parasites, so a parasite positive only in
    the post-permeabilisation channel was inside. Hence:

    * **attached / outside** = present in the outside-stain channel;
    * **invaded / inside** = *absent* from the outside-stain channel.

    Read that asymmetry carefully, because the whole design follows from it.
    "Inside" is defined by an absence, and absence is the unreliable
    direction. Poor antibody penetration, a focal plane off the parasite's
    equator, photobleaching, a low-expressing parasite — every one of them
    removes outside signal from a parasite that is genuinely outside, and
    every one of them therefore *inflates* invasion efficiency. Nothing
    plausible pushes the error the other way. The threshold on the outside
    channel is the single number the assay rests on, so it is derived from the
    data, reported per field in ``fields``, cross-checked against a
    control-derived cut when one exists, and bracketed by a sensitivity pair
    that says how much of the answer is the threshold.

    Three design decisions worth stating outright:

    * **The threshold is per field.** Illumination and staining vary field to
      field, and a plate-wide cut turns an illumination gradient into an
      invasion gradient. See :func:`_invasion_field_thresholds`.
    * **Controls beat any automatic method.** ``control_wells`` names wells
      whose parasites are known to carry no outside stain; the threshold is
      then a high quantile of that honest negative distribution
      (``control_quantile``), the control wells are excluded from the results,
      and ``threshold_source`` says ``'control'`` so the report cannot be
      mistaken for an automatic run.
    * **A threshold without two populations is arbitrary.** Otsu will happily
      split a single smear of signal down the middle and return a confident
      number. ``bimodality_coefficient`` and ``qc_flag_unimodal`` say when
      that has happened, per field and per well, instead of letting it pass
      silently. See :func:`_bimodality_coefficient`.

    ``invasion_efficiency = n_invaded / (n_invaded + n_attached)`` and it is
    always reported next to ``n_total``: 90% from ten parasites and 90% from
    four thousand are not the same result. A well that scored nothing gets
    NaN, not 0.0.

    **Statistics use the well as the unit of replication.** Parasites within a
    well share a coverslip, an antibody bath and a focal plane, so they are
    not independent; the reported test is a Mann-Whitney U on the per-well
    efficiencies. A pooled-parasite chi-squared is reported beside it purely
    so its inflation is visible. See :func:`_invasion_compare_conditions`.

    :param settings: dict of invasion settings; see
        ``set_analyze_invasion_defaults``. Key entries:

        - ``src`` — plate directory (or list) holding
          ``measurements/measurements.db``.
        - ``parasite_table`` / ``compartment`` — table and column prefix with
          one row per segmented parasite. Default ``'pathogen'``.
        - ``outside_channel`` / ``total_channel`` — the pre- and
          post-permeabilisation stain channels.
        - ``intensity_statistic`` — which per-object statistic of the outside
          channel to threshold; ``'auto'`` prefers the boundary-restricted
          one. See :func:`_resolve_invasion_intensity_column`.
        - ``background_correction`` — optional per-object local background.
        - ``outside_threshold_method`` / ``outside_threshold`` — automatic method, or
          a fixed cut that overrides it.
        - ``control_wells`` / ``control_quantile`` / ``min_control_objects``.
        - ``min_objects_for_threshold`` / ``min_objects_for_bimodality`` /
          ``bimodality_cutoff`` / ``threshold_agreement_tolerance`` /
          ``threshold_sensitivity`` / ``inflation_warn`` /
          ``min_parasites_per_well`` — the QC thresholds.
        - ``extracellular_class`` — how parasites with no host cell are scored.
        - ``cell_types`` / ``pathogen_types`` / ``treatments`` and their
          ``*_plate_metadata`` well maps, plus ``group_column`` and ``level``.
        - ``save`` — write the CSVs and figures under
          ``<src>/results/analyze_invasion``.

    :returns: dict with ``parasites`` (per-object classification), ``fields``
        (per-field thresholds and QC), ``wells`` (per-well efficiency,
        denominators and QC flags), ``summary`` (per condition),
        ``comparisons`` (per-well statistics), ``chi_squared`` /
        ``chi_squared_pairwise`` (the shared proportion-bar omnibus tests),
        ``controls`` (the control-well objects, if any),
        ``control_thresholds``, ``intensity_column``, ``intensity_statistic``
        and ``figures``.
    :raises ValueError: when the parasite table holds no usable rows.
    :raises KeyError: when the requested statistic or group column is absent.

    Example:
        .. code-block:: python

            from spacr.submodules import analyze_invasion
            out = analyze_invasion({
                'src': '/data/plate1',
                'outside_channel': 1,
                'total_channel': 0,
                'control_wells': ['c12'],
                'pathogen_types': ['dmso', 'inhibitor'],
                'pathogen_plate_metadata': [['c1'], ['c2']],
            })
            print(out['wells'][['prc', 'n_total', 'invasion_efficiency',
                                'qc_flags']])

    See Also:
        :func:`analyze_replication` — the parasites-per-vacuole assay, whose
        table reading, condition annotation and output layout this follows.
    """
    from .utils import annotate_conditions, save_settings
    from .io import _read_db
    from . import settings as settings_module

    # spacr.settings owns every pipeline's defaults and wins wherever it
    # defines one; the local copy runs afterwards purely as a gap-filler, so
    # the assay is callable before the GUI knobs are registered. Both use
    # setdefault, so running settings.py first makes its values authoritative.
    apply_defaults = getattr(settings_module, 'set_analyze_invasion_defaults',
                             None)
    if apply_defaults is not None:
        settings = apply_defaults(settings)
    settings = _set_analyze_invasion_defaults(settings)
    save_settings(settings, name='analyze_invasion', show=settings['verbose'])

    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]

    compartment = settings['compartment']
    parasite_table = settings['parasite_table']
    group_column = settings['group_column']

    if settings['extracellular_class'] not in ('attached', 'exclude',
                                               'classify'):
        raise ValueError(
            "extracellular_class must be 'attached', 'exclude' or 'classify', "
            f"got {settings['extracellular_class']!r}."
        )

    # ---- read one row per segmented parasite ----------------------------
    # Deliberately NOT _read_and_merge_data: that helper collapses the
    # pathogen table onto the host cell (prcfo is built from cell_id), which
    # would sum several parasites' outside-stain intensities into one row and
    # destroy the per-parasite call this assay exists to make.
    parasite_frames, cell_frames = [], []
    for index, source in enumerate(settings['src']):
        location = os.path.join(source, 'measurements/measurements.db')
        frame = _read_db(location, [parasite_table])[0]
        if settings['change_plate']:
            # prcf carries the ORIGINAL plate name and is what the per-field
            # threshold is keyed on, so relabelling plateID alone would let
            # two plates that share a well/field pool their fields.
            frame['plateID'] = f'plate{index + 1}'
            frame = frame.drop(columns=['prcf'], errors='ignore')
        parasite_frames.append(frame)
        if settings['seed_wells_from_cells']:
            try:
                cell_frame = _read_db(location, ['cell'])[0]
            except ValueError:
                cell_frame = None
            if cell_frame is not None:
                if settings['change_plate']:
                    cell_frame['plateID'] = f'plate{index + 1}'
                cell_frames.append(cell_frame)

    df = pd.concat(parasite_frames, axis=0, ignore_index=True)

    for column in ('plateID', 'rowID', 'columnID', 'fieldID'):
        if column not in df.columns:
            raise ValueError(
                f"Table '{parasite_table}' has no '{column}' column; it does "
                f"not look like a spacr measurements table."
            )
    # The timepoint is part of this key on a timelapse; see _ensure_field_key.
    # One outside-stain threshold is computed per prcf, so a time-blind one
    # cuts every frame of a field on a single number.
    df = _ensure_field_key(df, source=f"table '{parasite_table}'",
                           verbose=settings['verbose'])
    df['prc'] = (df['plateID'].astype(str) + '_' + df['rowID'].astype(str)
                 + '_' + df['columnID'].astype(str))

    # ---- object filters --------------------------------------------------
    area_column = f'{compartment}_area'
    if area_column in df.columns:
        if settings['min_parasite_area']:
            df = df[df[area_column] >= settings['min_parasite_area']]
        if settings['max_parasite_area'] is not None:
            df = df[df[area_column] <= settings['max_parasite_area']]

    if settings['min_total_intensity'] is not None:
        total_column = (f"{compartment}_channel_{settings['total_channel']}"
                        f"_mean_intensity")
        if total_column not in df.columns:
            raise KeyError(
                f"min_total_intensity needs '{total_column}', which is not in "
                f"the parasite table. Check 'total_channel'."
            )
        df = df[pd.to_numeric(df[total_column], errors='coerce')
                >= settings['min_total_intensity']]

    df = df.copy()
    if len(df) == 0:
        raise ValueError(
            f"No parasite objects left in '{parasite_table}' after filtering. "
            f"Check min_parasite_area / max_parasite_area / min_total_intensity."
        )

    # ---- the outside-channel signal --------------------------------------
    value_column, statistic_name = _resolve_invasion_intensity_column(
        df, compartment, settings['outside_channel'],
        settings['intensity_statistic'], verbose=settings['verbose'])
    background_column = _resolve_invasion_background_column(
        df, compartment, settings['outside_channel'],
        settings['background_correction'])

    df['outside_intensity_raw'] = pd.to_numeric(df[value_column],
                                                errors='coerce')
    if background_column is None:
        df['outside_background'] = 0.0
    else:
        df['outside_background'] = pd.to_numeric(df[background_column],
                                                 errors='coerce').fillna(0.0)
    df['outside_intensity'] = (df['outside_intensity_raw']
                               - df['outside_background'])

    # ---- staining controls ----------------------------------------------
    # Split them off before conditions are annotated: a no-primary or
    # no-permeabilisation control is a staining control, not an experimental
    # condition, so it has no entry in the well maps and must not appear in
    # any efficiency.
    control_mask = _invasion_control_mask(df, settings['control_wells'])
    controls = df[control_mask].copy()
    df = df[~control_mask].copy()
    if len(df) == 0:
        raise ValueError(
            "Every parasite row fell inside 'control_wells'; there is nothing "
            "left to score."
        )

    control_thresholds = {}
    if len(controls) > 0:
        for plate, subset in controls.groupby('plateID', sort=False):
            values = subset['outside_intensity'].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size >= int(settings['min_control_objects']):
                control_thresholds[str(plate)] = float(
                    np.quantile(values, float(settings['control_quantile'])))
            else:
                print(
                    f"WARNING: control wells on plate {plate} hold only "
                    f"{values.size} object(s), below min_control_objects="
                    f"{settings['min_control_objects']}; falling back to the "
                    f"automatic per-field threshold for that plate."
                )
        if control_thresholds:
            print(
                "Outside-stain threshold taken from the control wells "
                f"(quantile {settings['control_quantile']:.3g} of the negative "
                f"distribution): "
                + ', '.join(f'{plate}={value:.4g}'
                            for plate, value in control_thresholds.items())
            )
    if settings['outside_threshold'] is not None and control_thresholds:
        print(
            "NOTE: 'outside_threshold' is set, so the fixed value is used and "
            "the control-derived cut becomes the reference the QC judges it "
            "against."
        )

    # ---- host cell -------------------------------------------------------
    if 'cell_id' in df.columns:
        host = pd.to_numeric(df['cell_id'], errors='coerce')
        df['cell_id'] = host.fillna(0).astype(int)
        df['no_host_cell'] = (~host.notna()) | (host == 0)
    else:
        # No cell mask at all: nothing is known about host association, so
        # nothing is forced and the stain decides every call.
        df['no_host_cell'] = False
    if settings['extracellular_class'] == 'exclude':
        df = df[~df['no_host_cell']].copy()
        if len(df) == 0:
            raise ValueError(
                "extracellular_class='exclude' removed every parasite: none "
                "of them overlap a host cell."
            )

    # ---- conditions ------------------------------------------------------
    df = annotate_conditions(
        df=df,
        cells=settings['cell_types'],
        cell_loc=settings['cell_plate_metadata'],
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'],
        treatment_loc=settings['treatment_plate_metadata'],
    )
    if group_column not in df.columns:
        raise KeyError(
            f"'{group_column}' not found in the parasite table. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    df = df.dropna(subset=[group_column])
    if len(df) == 0:
        raise ValueError(
            f"Every parasite row has an empty '{group_column}'. Check the "
            f"cell_plate_metadata / pathogen_plate_metadata / "
            f"treatment_plate_metadata well maps."
        )

    # ---- thresholds and classification -----------------------------------
    fields = _invasion_field_thresholds(df, 'outside_intensity', settings,
                                        control_thresholds)
    parasites = _invasion_classify(df, fields, 'outside_intensity',
                                   settings['extracellular_class'])

    field_classes = parasites.groupby('prcf', sort=False)['invasion_class']
    field_counts = field_classes.value_counts().unstack(fill_value=0)
    for name in _INVASION_CLASSES:
        if name not in field_counts.columns:
            field_counts[name] = 0
    fields = fields.merge(
        field_counts[_INVASION_CLASSES].rename(
            columns={'attached': 'n_attached', 'invaded': 'n_invaded'}
        ).reset_index(), on='prcf', how='left')
    fields[['n_attached', 'n_invaded']] = (
        fields[['n_attached', 'n_invaded']].fillna(0).astype(int))
    fields['n_total'] = fields['n_attached'] + fields['n_invaded']
    fields['invasion_efficiency'] = [
        _invasion_efficiency(invaded, total)
        for invaded, total in zip(fields['n_invaded'], fields['n_total'])
    ]
    field_group = parasites.groupby('prcf', sort=False)[group_column].first()
    fields[group_column] = fields['prcf'].map(field_group)

    # ---- wells that hold host cells but no parasites ----------------------
    seed_wells = None
    if cell_frames:
        cells = pd.concat(cell_frames, axis=0, ignore_index=True)
        cells['prc'] = (cells['plateID'].astype(str) + '_'
                        + cells['rowID'].astype(str) + '_'
                        + cells['columnID'].astype(str))
        cells = cells[~_invasion_control_mask(cells, settings['control_wells'])]
        cells = annotate_conditions(
            df=cells,
            cells=settings['cell_types'],
            cell_loc=settings['cell_plate_metadata'],
            pathogens=settings['pathogen_types'],
            pathogen_loc=settings['pathogen_plate_metadata'],
            treatments=settings['treatments'],
            treatment_loc=settings['treatment_plate_metadata'],
        )
        if group_column in cells.columns:
            seed_wells = cells.dropna(subset=[group_column])[
                ['plateID', 'rowID', 'columnID', 'prc', group_column]
            ].drop_duplicates()

    wells = _invasion_well_table(parasites, fields, group_column, settings,
                                 seed_wells=seed_wells)
    summary = _invasion_summary(wells, group_column)
    comparisons = _invasion_compare_conditions(wells, group_column,
                                               verbose=settings['verbose'])

    # ---- figures ---------------------------------------------------------
    prc_column = 'plateID' if settings['level'] == 'plate' else 'prc'
    scored = parasites[parasites['invasion_class'].astype(str)
                       != 'unclassified'].copy()

    well_totals = dict(zip(wells['prc'].astype(str), wells['n_total']))
    condition_totals = dict(zip(summary[group_column].astype(str),
                                summary['n_total']))

    _, _, well_fig = _invasion_stacked_bars(
        settings, scored, group_column='prc', prc_column='prc',
        level='object', cmap=settings['cmap'],
        title='Invasion — per well', denominators=well_totals,
    )
    results_df, pairwise_df, group_fig = _invasion_stacked_bars(
        settings, scored, group_column=group_column, prc_column=prc_column,
        level=settings['level'], cmap=settings['cmap'],
        title='Invasion — by condition', denominators=condition_totals,
    )
    threshold_fig = _invasion_threshold_panels(
        parasites, wells, max_panels=settings['qc_plot_max_panels'],
        cmap=settings['cmap'])

    output = {
        'parasites': parasites,
        'fields': fields,
        'wells': wells,
        'summary': summary,
        'comparisons': comparisons,
        'chi_squared': results_df,
        'chi_squared_pairwise': pairwise_df,
        'controls': controls,
        'control_thresholds': control_thresholds,
        'intensity_column': value_column,
        'intensity_statistic': statistic_name,
        'figures': {'per_well': well_fig, 'by_condition': group_fig,
                    'thresholds': threshold_fig},
    }

    if settings['save']:
        output_dir = os.path.join(settings['src'][0], 'results',
                                  'analyze_invasion')
        os.makedirs(output_dir, exist_ok=True)
        parasites.to_csv(os.path.join(output_dir, 'parasite_calls.csv'),
                         index=False)
        fields.to_csv(os.path.join(output_dir, 'field_thresholds.csv'),
                      index=False)
        wells.to_csv(os.path.join(output_dir, 'well_invasion.csv'), index=False)
        summary.to_csv(os.path.join(output_dir, 'condition_summary.csv'),
                       index=False)
        comparisons.to_csv(os.path.join(output_dir, 'condition_comparisons.csv'),
                           index=False)
        results_df.to_csv(os.path.join(output_dir, 'chi_squared_results.csv'),
                          index=False)
        pairwise_df.to_csv(
            os.path.join(output_dir, 'chi_squared_pairwise_results.csv'),
            index=False)
        well_fig.savefig(os.path.join(output_dir, 'invasion_per_well.pdf'),
                         dpi=300, bbox_inches='tight')
        group_fig.savefig(os.path.join(output_dir, 'invasion_by_condition.pdf'),
                          dpi=300, bbox_inches='tight')
        threshold_fig.savefig(
            os.path.join(output_dir, 'outside_stain_thresholds.pdf'),
            dpi=300, bbox_inches='tight')
        print(f"Invasion assay results saved to {output_dir}")

    if settings['verbose']:
        print(f"Outside-stain statistic: '{value_column}' ({statistic_name})")
        print("Per-field thresholds:")
        print(fields[['prcf', 'n_total', 'threshold', 'threshold_source',
                      'automatic_threshold', 'bimodality_coefficient',
                      'invasion_efficiency']].to_string(index=False))
        flagged = wells.loc[~wells['qc_pass'], ['prc', 'n_total', 'qc_flags']]
        if len(flagged):
            print(f"QC: {len(flagged)} well(s) flagged:")
            print(flagged.to_string(index=False))

    plt.show()
    # The figures stay usable (savefig works on a closed figure); closing them
    # keeps a batch run over many plates from accumulating open figures.
    for figure in (well_fig, group_fig, threshold_fig):
        plt.close(figure)

    return output


def analyze_class_proportion(settings):
    """Test whether classifier class proportions differ between experimental groups.

    Runs chi-squared and pairwise tests on the class column, plots stacked
    bars and a plate heatmap, and follows up with normality, Levene, and
    posthoc statistical tests.

    :param settings: dict of settings; see
        ``set_analyze_class_proportion_defaults`` for keys including ``src``,
        ``tables``, ``class_column``, ``group_column``, ``level`` and ``save``.
    :returns: dict with ``data`` (annotated DataFrame) and ``chi_squared`` (results DataFrame).
    """
    from .utils import annotate_conditions, save_settings
    from .io import _read_and_merge_data
    from .settings import set_analyze_class_proportion_defaults
    from .plot import plot_plates, plot_proportion_stacked_bars
    from .sp_stats import perform_normality_tests, perform_levene_test, perform_statistical_tests, perform_posthoc_tests
    
    settings = set_analyze_class_proportion_defaults(settings)
    save_settings(settings, name='analyze_class_proportion', show=True)
    output = {}

    # Process data
    if not isinstance(settings['src'], list):
        settings['src'] = [settings['src']]
    
    locs = []
    for s in settings['src']:
        loc = os.path.join(s, 'measurements/measurements.db')
        locs.append(loc)
        
    if 'png_list' not in settings['tables']:
        settings['tables'] = settings['tables'] + ['png_list']
            
    df, _ = _read_and_merge_data(
        locs, 
        tables=settings['tables'], 
        verbose=settings['verbose'], 
        nuclei_limit=settings['nuclei_limit'], 
        pathogen_limit=settings['pathogen_limit']
    )
        
    df = annotate_conditions(
        df=df, 
        cells=settings['cell_types'], 
        cell_loc=settings['cell_plate_metadata'], 
        pathogens=settings['pathogen_types'],
        pathogen_loc=settings['pathogen_plate_metadata'],
        treatments=settings['treatments'], 
        treatment_loc=settings['treatment_plate_metadata']
    )
    
    if settings['group_column'] not in df.columns:
        print(f"{settings['group_column']} not found in DataFrame, please choose from:")
        for col in df.columns:
            print(col)
    
    df[settings['class_column']] = df[settings['class_column']].fillna(0)
    output['data'] = df
    
    # Perform chi-squared test and plot
    results_df, pairwise_results, fig = plot_proportion_stacked_bars(settings, df, settings['group_column'], bin_column=settings['class_column'], level=settings['level'])
    
    output['chi_squared'] = results_df
    
    if settings['save']:
        output_dir = os.path.join(settings['src'][0], 'results', 'analyze_class_proportion')
        os.makedirs(output_dir, exist_ok=True)
        output_path_chi = os.path.join(output_dir, 'class_chi_squared_results.csv')
        output_path_chi_pairwise = os.path.join(output_dir, 'class_frequency_test.csv')
        output_path_data = os.path.join(output_dir, 'class_chi_squared_data.csv')
        output_path_fig = os.path.join(output_dir, 'class_chi_squared.pdf')
        fig.savefig(output_path_fig, dpi=300, bbox_inches='tight')
        results_df.to_csv(output_path_chi, index=False)
        pairwise_results.to_csv(output_path_chi_pairwise, index=False)
        df.to_csv(output_path_data, index=False)
        print(f"Chi-squared results saved to {output_path_chi}")
        print(f"Annotated data saved to {output_path_data}")

    plt.show()
    
    fig2 = plot_plates(df, variable=settings['class_column'], grouping='mean', min_max='allq', cmap='viridis', min_count=0, verbose=True, dst=None)
    if settings['save']:
        output_path_fig2 = os.path.join(output_dir, 'class_heatmap.pdf')
        fig2.savefig(output_path_fig2, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Perform normality, variance, and statistical tests
    is_normal, normality_results = perform_normality_tests(df, settings['group_column'], [settings['class_column']])
    variance_stat, variance_p = perform_levene_test(df, settings['group_column'], settings['class_column'])

    print(f"Levene's test statistic: {variance_stat:.4f}, p-value: {variance_p:.4e}")
    variance_results = {
        'Test Statistic': variance_stat,
        'p-value': variance_p,
        'Test Name': "Levene's Test"
    }

    test_results = perform_statistical_tests(df, settings['group_column'], [settings['class_column']])
    posthoc_results = perform_posthoc_tests(
        df, settings['group_column'], settings['class_column'], is_normal=is_normal
    )

    # Save additional results
    if settings['save']:
        pd.DataFrame(normality_results).to_csv(os.path.join(output_dir, 'normality_results.csv'), index=False)
        pd.DataFrame([variance_results]).to_csv(os.path.join(output_dir, 'variance_results.csv'), index=False)
        pd.DataFrame(test_results).to_csv(os.path.join(output_dir, 'statistical_test_results.csv'), index=False)
        pd.DataFrame(posthoc_results).to_csv(os.path.join(output_dir, 'posthoc_results.csv'), index=False)
        print("Statistical analysis results saved.")

    return output

def generate_score_heatmap(settings):
    """Combine multiple classifier score CSVs into a per-well heatmap and MAE table.

    Aggregates per-object scores across score CSVs, merges with a
    cross-validation score and a reads-derived fraction column, plots a
    multi-channel heatmap, and computes per-channel mean absolute error
    against the empirical fraction.

    :param settings: dict of settings including ``folders``, ``csv_name``,
        ``data_column``, ``csv``, ``cv_csv``, ``data_column_cv``,
        ``plateID``, ``columnID``, ``control_sgrnas``, ``fraction_grna``,
        ``cmap`` and ``dst``.
    :returns: merged DataFrame joining reads, classifier scores and CV scores per well.
    """

    def group_cv_score(csv, plate=1, column='c3', data_column='pred'):
        """Aggregate a CV predictions CSV to a per-(plate, row, column) mean."""
        
        df = pd.read_csv(csv)
        if 'columnID' in df.columns:
            df = df[df['columnID']==column]
        elif 'column' in df.columns:
            df['columnID'] = df['column']
            df = df[df['columnID']==column]
        if not plate is None:
            df['plateID'] = f"plate{plate}"
        grouped_df = df.groupby(['plateID', 'rowID', 'columnID'])[data_column].mean().reset_index()
        grouped_df['prc'] = grouped_df['plateID'].astype(str) + '_' + grouped_df['rowID'].astype(str) + '_' + grouped_df['columnID'].astype(str)
        return grouped_df

    def calculate_fraction_mixed_condition(csv, plate=1, column='c3', control_sgrnas = None):
        """Return per-well read fractions restricted to the given control sgRNAs."""
        if control_sgrnas is None:
            control_sgrnas = ['TGGT1_220950_1', 'TGGT1_233460_4']
        df = pd.read_csv(csv)
        # This helper was left half-way through the column_name -> columnID
        # rename: it grouped by 'columnID' but filtered and merged on
        # 'column_name', a key the grouped frame can never carry, so every
        # call died with KeyError('column_name'). Key on 'columnID' like
        # every sibling helper here, accepting the legacy spelling that
        # older reads CSVs still use.
        if 'columnID' not in df.columns and 'column_name' in df.columns:
            df = df.rename(columns={'column_name': 'columnID'})
        df = df[df['columnID']==column]
        # `plate` is a plate NUMBER, not a column name, so `plate not in
        # df.columns` was always True and the CSV's own plateID was always
        # overwritten -- stamping the literal "plateNone" when plate is None.
        # The prc keys then matched nothing downstream and the heatmap came
        # back empty with no error. Guard the way both sibling helpers do.
        if plate is not None:
            df['plateID'] = f"plate{plate}"
        df = df[df['grna_name'].str.match(f'^{control_sgrnas[0]}$|^{control_sgrnas[1]}$')]
        grouped_df = df.groupby(['plateID', 'rowID', 'columnID'])['count'].sum().reset_index()
        grouped_df = grouped_df.rename(columns={'count': 'total_count'})
        merged_df = pd.merge(df, grouped_df, on=['plateID', 'rowID', 'columnID'])
        merged_df['fraction'] = merged_df['count'] / merged_df['total_count']
        merged_df['prc'] = merged_df['plateID'].astype(str) + '_' + merged_df['rowID'].astype(str) + '_' + merged_df['columnID'].astype(str)
        return merged_df

    def plot_multi_channel_heatmap(df, column='c3', cmap='coolwarm'):
        """Plot a per-well heatmap with each classifier channel as a column.

        :param df: DataFrame with score columns keyed by channel.
        :param column: value in ``columnID`` used to filter rows. Default ``'c3'``.
        :param cmap: matplotlib/seaborn colormap. Default ``'coolwarm'``.
        :returns: the matplotlib Figure.
        """
        # Copy first: this assignment used to mutate the CALLER's frame,
        # so the temporary sort column survived in merged_df (the drop
        # below only affects the local slice) and leaked into the returned
        # frame, the saved *_data.csv and the MAE table as a bogus channel.
        df = df.copy()

        # Extract row number and convert to integer for sorting
        df['row_num'] = df['rowID'].str.extract(r'(\d+)').astype(int)

        # Filter and sort by plate, row, and column
        df = df[df['columnID'] == column]
        df = df.sort_values(by=['plateID', 'row_num', 'columnID'])

        # Drop temporary 'row_num' column after sorting
        df = df.drop('row_num', axis=1)

        # Create a new column combining plate, row, and column for the index
        df['plate_row_col'] = df['plateID'] + '-' + df['rowID'] + '-' + df['columnID']

        # Set 'plate_row_col' as the index
        df.set_index('plate_row_col', inplace=True)

        # Extract only numeric data for the heatmap
        heatmap_data = df.select_dtypes(include=[float, int])

        # Plot heatmap with square boxes, no annotations, and 'viridis' colormap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            cbar=True,
            square=True,
            annot=False
        )

        plt.title("Heatmap of Prediction Scores for All Channels")
        plt.xlabel("Channels")
        plt.ylabel("Plate-Row-Column")
        plt.tight_layout()

        # Save the figure object and return it
        fig = plt.gcf()
        plt.show()

        return fig


    def combine_classification_scores(folders, csv_name, data_column, plate=1, column='c3'):
        """Merge one ``data_column`` per sub-folder into a wide per-well DataFrame."""
        # Ensure `folders` is a list
        if isinstance(folders, str):
            folders = [folders]

        ls = []  # Initialize ls to store found CSV file paths

        # Iterate over the provided folders
        for folder in folders:
            sub_folders = os.listdir(folder)  # Get sub-folder list
            for sub_folder in sub_folders:  # Iterate through sub-folders
                path = os.path.join(folder, sub_folder)  # Join the full path

                if os.path.isdir(path):  # Check if it’s a directory
                    csv = os.path.join(path, csv_name)  # Join path to the CSV file
                    if os.path.exists(csv):  # If CSV exists, add to list
                        ls.append(csv)
                    else:
                        print(f'No such file: {csv}')

        # Initialize combined DataFrame
        combined_df = None
        print(f'Found {len(ls)} CSV files')

        # Loop through all collected CSV files and process them
        for csv_file in ls:
            df = pd.read_csv(csv_file)  # Read CSV into DataFrame
            df = df[df['columnID']==column]
            if not plate is None:
                df['plateID'] = f"plate{plate}"
            # Group the data by 'plateID', 'rowID', and 'columnID'
            grouped_df = df.groupby(['plateID', 'rowID', 'columnID'])[data_column].mean().reset_index()
            # Use the CSV filename to create a new column name
            folder_name = os.path.dirname(csv_file).replace(".csv", "")
            new_column_name = os.path.basename(f"{folder_name}_{data_column}")
            print(new_column_name)
            grouped_df = grouped_df.rename(columns={data_column: new_column_name})

            # Merge into the combined DataFrame
            if combined_df is None:
                combined_df = grouped_df
            else:
                combined_df = pd.merge(combined_df, grouped_df, on=['plateID', 'rowID', 'columnID'], how='outer')
        combined_df['prc'] = combined_df['plateID'].astype(str) + '_' + combined_df['rowID'].astype(str) + '_' + combined_df['columnID'].astype(str)
        return combined_df
    
    def calculate_mae(df):
        """Return the per-channel, per-row MAE between predictions and the ``fraction`` column."""
        # Extract numeric columns excluding 'fraction' and 'prc'
        channels = df.drop(columns=['fraction', 'prc']).select_dtypes(include=[float, int])

        mae_data = []

        # Compute MAE for each channel with 'fraction' for all rows
        for column in channels.columns:
            for index, row in df.iterrows():
                mae = mean_absolute_error([row['fraction']], [row[column]])
                mae_data.append({'Channel': column, 'MAE': mae, 'Row': row['prc']})

        # Convert the list of dictionaries to a DataFrame
        mae_df = pd.DataFrame(mae_data)
        return mae_df

    result_df = combine_classification_scores(settings['folders'], settings['csv_name'], settings['data_column'], settings['plateID'], settings['columnID'], )
    df = calculate_fraction_mixed_condition(settings['csv'], settings['plateID'], settings['columnID'], settings['control_sgrnas'])
    df = df[df['grna_name']==settings['fraction_grna']]
    fraction_df = df[['fraction', 'prc']]
    merged_df = pd.merge(fraction_df, result_df, on=['prc'])
    cv_df = group_cv_score(settings['cv_csv'], settings['plateID'], settings['columnID'], settings['data_column_cv'])
    cv_df = cv_df[[settings['data_column_cv'], 'prc']]
    merged_df = pd.merge(merged_df, cv_df, on=['prc'])
    
    fig = plot_multi_channel_heatmap(merged_df, settings['columnID'], settings['cmap'])
    # The guard used to test for 'row_number' while the helper adds
    # 'row_num', so it never fired for the column it meant to drop and
    # would KeyError on a frame that genuinely carries a 'row_number'
    # data column. With the copy in the helper this is now a no-op kept
    # as cheap defence. The matching mae_df guard was deleted: calculate_mae
    # only ever emits Channel/MAE/Row, so it was dead and, if it had ever
    # fired, would have dropped a differently-named column.
    if 'row_num' in merged_df.columns:
        merged_df = merged_df.drop('row_num', axis=1)
    mae_df = calculate_mae(merged_df)

    if not settings['dst'] is None:
        mae_dst = os.path.join(settings['dst'], f"mae_scores_comparison_plate_{settings['plateID']}.csv")
        merged_dst = os.path.join(settings['dst'], f"scores_comparison_plate_{settings['plateID']}_data.csv")
        heatmap_save = os.path.join(settings['dst'], f"scores_comparison_plate_{settings['plateID']}.pdf")
        mae_df.to_csv(mae_dst, index=False)
        merged_df.to_csv(merged_dst, index=False)
        fig.savefig(heatmap_save, format='pdf', dpi=600, bbox_inches='tight')
    return merged_df

def post_regression_analysis(csv_file, grna_dict, grna_list, save=False):
    """Compute gRNA correlation and propagate fixed effect sizes across correlated gRNAs.

    :param csv_file: CSV with columns ``grna``, ``fraction`` and ``prc``.
    :param grna_dict: mapping of anchor ``grna`` names to their fixed effect sizes.
    :param grna_list: gRNAs to include in the correlation matrix.
    :param save: persist correlation matrix, effect sizes and plots. Default ``False``.
    :returns: None. Displays plots and optionally writes results to ``<csv_dir>/post_regression_analysis_results``.
    """

    def _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save=False):
        """Return and plot the pivoted per-well gRNA fraction correlation matrix."""
        # Filter the DataFrame to include only rows with gRNAs in the list
        filtered_df = df[df['grna'].isin(grna_list)]

        # Pivot the data to create a prc-by-gRNA matrix, using fractions as values
        pivot_df = filtered_df.pivot_table(index='prc', columns='grna', values='fraction', aggfunc='sum').fillna(0)

        # Compute the correlation matrix
        correlation_matrix = pivot_df.corr()
        
        if save:
            # Save the correlation matrix
            correlation_matrix.to_csv(os.path.join(save_folder, 'correlation_matrix.csv'))
        
        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
        plt.title('gRNA Correlation Matrix')
        plt.xlabel('gRNAs')
        plt.ylabel('gRNAs')
        plt.tight_layout()
        
        if save:
            correlation_fig_path = os.path.join(save_folder, 'correlation_matrix_heatmap.pdf')
            plt.savefig(correlation_fig_path, dpi=300)
        
        plt.show()

        return correlation_matrix

    def _compute_effect_sizes(correlation_matrix, grna_dict, save_folder, save=False):
        """Return per-gRNA effect sizes propagated from anchor gRNAs via the correlation matrix."""
        # Ensure the matrix is symmetric and normalize values to 0-1
        corr_matrix = correlation_matrix.copy()
        corr_matrix = (corr_matrix - corr_matrix.min().min()) / (corr_matrix.max().max() - corr_matrix.min().min())

        # Initialize the effect sizes with dtype float
        effect_sizes = pd.Series(0.0, index=corr_matrix.index)

        # Set the effect sizes for the specified gRNAs
        for grna, size in grna_dict.items():
            effect_sizes[grna] = size

        # Propagate the effect sizes
        for grna in corr_matrix.index:
            if grna not in grna_dict:
                # Weighted sum of correlations with the fixed gRNAs
                effect_sizes[grna] = np.dot(corr_matrix.loc[grna], effect_sizes) / np.sum(corr_matrix.loc[grna])
        
        if save:
            # Save the effect sizes
            effect_sizes.to_csv(os.path.join(save_folder, 'effect_sizes.csv'))

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=effect_sizes.index,
            y=effect_sizes.values,
            hue=effect_sizes.index,
            palette="viridis",
            legend=False,
        )

        #for i, val in enumerate(effect_sizes.values):
        #    plt.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
        plt.title("Effect Sizes of gRNAs")
        plt.xlabel("gRNAs")
        plt.ylabel("Effect Size")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            effect_sizes_fig_path = os.path.join(save_folder, 'effect_sizes_barplot.pdf')
            plt.savefig(effect_sizes_fig_path, dpi=300)
        
        plt.show()

        return effect_sizes
    
    # Ensure the save folder exists
    save_folder = os.path.join(os.path.dirname(csv_file), 'post_regression_analysis_results')
    os.makedirs(save_folder, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Perform analysis
    correlation_matrix = _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save)
    _compute_effect_sizes(correlation_matrix, grna_dict, save_folder, save)
