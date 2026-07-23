import seaborn as sns
import os, random, sqlite3, re, shap, string, time, shutil
import pandas as pd
import numpy as np

from skimage.measure import regionprops, label
from skimage.transform import resize as sk_resize, rotate
from skimage.exposure import rescale_intensity

import cellpose
from cellpose import models as cp_models
from cellpose import train as train_cp
from cellpose import models as cp_models
from cellpose import io as cp_io
from cellpose import train as train_cp
from cellpose.metrics import aggregated_jaccard_index
from cellpose.metrics import average_precision

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from math import pi
from scipy.stats import chi2_contingency

from sklearn.metrics import mean_absolute_error
from skimage.measure import regionprops, label as sklabel 
import matplotlib.pyplot as plt
from natsort import natsorted

from torch.utils.data import Dataset

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

class CellposeLazyDataset_v1(Dataset):
    """Legacy lazy image/label dataset for Cellpose (kept for backwards compatibility).

    Superseded by :class:`CellposeLazyDataset`; retained so older training
    scripts continue to import a working class.

    :param image_files: paths to input image tiffs.
    :param label_files: paths to matching label tiffs.
    :param settings: dict with keys ``normalize``, ``percentiles``, ``target_size``.
    :param randomize: shuffle pairs at construction. Default ``True``.
    :param augment: enable 8-fold augmentation. Default ``False``.
    """
    def __init__(self, image_files, label_files, settings, randomize=True, augment=False):
        combined = list(zip(image_files, label_files))
        if randomize:
            random.shuffle(combined)
        self.image_files, self.label_files = zip(*combined)
        self.normalize = settings['normalize']
        self.percentiles = settings.get('percentiles', [2, 99])
        self.target_size = settings['target_size']
        self.augment = augment

    def __len__(self):
        return len(self.image_files) * (8 if self.augment else 1)

    def apply_augmentation(self, image, label, aug_idx):
        """Return the ``aug_idx``-th rotation/flip augmentation of ``image``/``label``."""
        if aug_idx == 1:
            return rotate(image, 90, resize=False, preserve_range=True), rotate(label, 90, resize=False, preserve_range=True)
        elif aug_idx == 2:
            return rotate(image, 180, resize=False, preserve_range=True), rotate(label, 180, resize=False, preserve_range=True)
        elif aug_idx == 3:
            return rotate(image, 270, resize=False, preserve_range=True), rotate(label, 270, resize=False, preserve_range=True)
        elif aug_idx == 4:
            return np.fliplr(image), np.fliplr(label)
        elif aug_idx == 5:
            return np.flipud(image), np.flipud(label)
        elif aug_idx == 6:
            return np.fliplr(rotate(image, 90, resize=False, preserve_range=True)), np.fliplr(rotate(label, 90, resize=False, preserve_range=True))
        elif aug_idx == 7:
            return np.flipud(rotate(image, 90, resize=False, preserve_range=True)), np.flipud(rotate(label, 90, resize=False, preserve_range=True))
        return image, label

    def __getitem__(self, idx):
        base_idx = idx // 8 if self.augment else idx
        aug_idx = idx % 8 if self.augment else 0

        image = cp_io.imread(self.image_files[base_idx])
        label = cp_io.imread(self.label_files[base_idx])

        if image.ndim == 3:
            image = image.mean(axis=-1)

        if image.max() > 1:
            image = image / image.max()

        if self.normalize:
            lower_p, upper_p = np.percentile(image, self.percentiles)
            image = rescale_intensity(image, in_range=(lower_p, upper_p), out_range=(0, 1))

        image, label = self.apply_augmentation(image, label, aug_idx)

        image_shape = (self.target_size, self.target_size)
        image = sk_resize(image, image_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
        label = sk_resize(label, image_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        return image, label

def train_cellpose(settings):
    """Train a Cellpose ``cyto`` segmentation model from images and paired masks.

    :param settings: dict of training settings; see
        ``get_train_cellpose_default_settings`` for keys including ``src``,
        ``model_name``, ``target_size``, ``n_epochs``, ``batch_size``,
        ``learning_rate``, ``weight_decay``, and ``augment``.
    :returns: None. Saves the trained model under ``<src>/models/cellpose_model``.
    """
    from .settings import get_train_cellpose_default_settings
    from .utils import save_settings
    
    settings = get_train_cellpose_default_settings(settings)
    img_src = os.path.join(settings['src'], 'train', 'images')
    mask_src = os.path.join(settings['src'], 'train', 'masks')
    target_size = settings['target_size']

    model_name = f"{settings['model_name']}_cyto_e{settings['n_epochs']}_X{target_size}_Y{target_size}.CP_model"
    model_save_path = os.path.join(settings['src'], 'models', 'cellpose_model')
    os.makedirs(model_save_path, exist_ok=True)

    save_settings(settings, name=model_name)

    model = cp_models.CellposeModel(gpu=True, model_type='cyto', diam_mean=30, pretrained_model='cyto')
    cp_channels = [0, 0]

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
    n_base = min(settings['batch_size'], max_base_images)

    unique_base_indices = list(range(max_base_images))
    random.shuffle(unique_base_indices)
    selected_indices = unique_base_indices[:n_base]

    images, labels = [], []
    for idx in selected_indices:
        for aug_idx in range(n_aug):
            i = idx * n_aug + aug_idx if settings['augment'] else idx
            img, lbl = train_dataset[i]
            images.append(img)
            labels.append(lbl)
    try:
        plot_cellpose_batch(images, labels)
    except Exception:
        print(f"could not print batch images")
        
    print(f"Training model with {len(images)} ber patch for {settings['n_epochs']} Epochs")

    train_cp.train_seg(model.net,
                       train_data=images,
                       train_labels=labels,
                       channels=cp_channels,
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

    model = cp_models.CellposeModel(gpu=True, pretrained_model=settings['model_path'])

    batch_size = settings['batch_size']
    scores = []
    names = []
    time_ls = []
    
    files_to_process = len(test_image_folder)

    for i in range(0, len(test_dataset), batch_size):
        start = time.time()
        batch = [test_dataset[j] for j in range(i, min(i + batch_size, len(test_dataset)))]
        images, labels = zip(*batch)

        masks_pred, flows, _ = model.eval(x=list(images),
                                          channels=[0, 0],
                                          normalize=False,
                                          diameter=30,
                                          flow_threshold=settings['FT'],
                                          cellprob_threshold=settings['CP_probability'],
                                          rescale=None,     
                                          resample=True,
                                          interp=True,
                                          anisotropy=None,
                                          min_size=5,         
                                          augment=True,
                                          tile=True,
                                          tile_overlap=0.2,
                                          bsize=224)
        
        n_objects_true_ls = []
        n_objects_pred_ls = []
        mean_area_true_ls = []
        mean_area_pred_ls = []
        tp_ls, fp_ls, fn_ls = [], [], []
        precision_ls, recall_ls, f1_ls, accuracy_ls = [], [], [], []

        for j, (img, lbl, pred, flow) in enumerate(zip(images, labels, masks_pred, flows)):
            score = float(aggregated_jaccard_index([lbl], [pred]))
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

            if settings['save']:
                plot_cellpose_resilts(i, j, results_dir, img, lbl, pred, flow)

            if settings['save']:
                plot_cellpose_resilts(i,j,results_dir, img, lbl, pred, flow)
                
        stop = time.time()
        duration = stop-start
        files_processed = (i+1) * batch_size
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

    model = cp_models.CellposeModel(gpu=True, pretrained_model=settings['model_path'])
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
        masks_pred, flows, _ = model.eval(x=list(images),
                                          channels=[0, 0],
                                          normalize=False,
                                          diameter=30,
                                          flow_threshold=settings['FT'],
                                          cellprob_threshold=settings['CP_probability'],
                                          rescale=None,     
                                          resample=True,
                                          interp=True,
                                          anisotropy=None,
                                          min_size=5,         
                                          augment=True,
                                          tile=True,
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


        # Write after each batch
        df_measurements = pd.DataFrame(measurements)
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
    fig, axs = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
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

        # Translate well to row and column
        df_2['rowID'] = 'r' + df_2['well'].str[0].map(lambda x: str(string.ascii_uppercase.index(x) + 1))
        df_2['column_name'] = 'c' + df_2['well'].str[1:].astype(int).astype(str)

        # Optional: add prcf ID (plate_row_column_field)
        df_2['fieldID'] = 'f1'  # default or extract from filename if needed
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

    merged = merged[['plate_y', 'well', 'plate_well','fieldID','rowID','column_name','prc_x','Original File','Renamed TIFF','above','below','fraction_above','fraction_below']]
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
        if not settings['src'].endswith('/measurements'):
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

    df = _read_db(loc=settings['src'], tables=['png_list'])

    unique_values_count = df[settings['annotation_column']].nunique(dropna=True)
    print(f"Unique values in {settings['annotation_column']} (excluding NaN): {unique_values_count}")

    # Count unique values in 'value' column, grouped by 'plateID', 'rowID', 'columnID'
    grouped_unique_count = df.groupby(['plateID', 'rowID', 'columnID'])[settings['annotation_column']].nunique(dropna=True).reset_index(name='unique_count')
    display(grouped_unique_count)

    save_path = os.path.join(settings['src'], 'phenotype_counts.csv')

    # Group by plate, row, and column, then count the occurrences of each unique value
    grouped_counts = df.groupby(['plateID', 'rowID', 'columnID', 'value']).size().reset_index(name='count')

    # Pivot the DataFrame so that unique values are columns and their counts are in the rows
    pivot_df = grouped_counts.pivot_table(index=['plateID', 'rowID', 'columnID'], columns='value', values='count', fill_value=0)

    # Flatten the multi-level columns
    pivot_df.columns = [f"value_{int(col)}" for col in pivot_df.columns]

    # Reset the index so that plate, row, and column form a combined index
    pivot_df.index = pivot_df.index.map(lambda x: f"{x[0]}_{x[1]}_{x[2]}")

    # Saving the DataFrame to a SQLite .db file
    output_dir = os.path.join('src', 'results')  # Replace 'src' with the actual base directory
    os.makedirs(output_dir, exist_ok=True)

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
                if 'row' in reads_df_temp.columns:
                    reads_df_temp = reads_df_temp.rename(columns={'row_name': 'rowID'})
                if 'row_name' in scores_df_temp.columns:
                    scores_df_temp = scores_df_temp.rename(columns={'row_name': 'rowID'})
                    
                reads_ls.append(reads_df_temp)
                scores_ls.append(scores_df_temp)
                    
            reads_df = pd.concat(reads_ls, axis=0)
            scores_df = pd.concat(scores_ls, axis=0)
            print(f"Reads: {len(reads_df)} Scores: {len(scores_df)}")
        else:
            print(f"reads_csv and scores_csv must contain the same number of elements if reads_csv is a list")
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
    
    if any in y_columns not in df.columns:
        print(f"columns in dataframe:")
        for col in df.columns:
            print(col)
        return
    display(df)
    fig_1 = plot_line(df, x_column = 'pc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[0])
    fig_2 = plot_line(df, x_column = 'nc_fraction', y_columns=y_columns, group_column=None, xlabel=None, ylabel='Fraction', title=None, figsize=(10, 6), save_path=save_paths[1])
    
    return [fig_1, fig_2]

def interperate_vision_model(settings=None):
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

            from spacr.submodules import interperate_vision_model
            results = interperate_vision_model({
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
                        df[new_col_name] = df[related_col] / df[comp0_col]
                        df[new_col_name].replace([float('inf'), -float('inf')], pd.NA, inplace=True)  # Replace inf values with NA
                        df[new_col_name].fillna(0, inplace=True)  # Replace NaN values with 0 for ease of further calculations

                # Generate all-to-all comparisons
                if related_cols:
                    comparison_dict[comp0_col] = related_cols
                    for i, rel_col_1 in enumerate(related_cols):
                        for rel_col_2 in related_cols[i + 1:]:
                            # Create a new column name for each pairwise comparison
                            comp1, comp2 = rel_col_1.split('_')[0], rel_col_2.split('_')[0]
                            new_col_name_all = f"{comp1}_{comp2}{base_col_name}"

                            # Calculate pairwise ratio and handle infinite or NaN values
                            df[new_col_name_all] = df[rel_col_1] / df[rel_col_2]
                            df[new_col_name_all].replace([float('inf'), -float('inf')], pd.NA, inplace=True)  # Replace inf with NA
                            df[new_col_name_all].fillna(0, inplace=True)  # Replace NaN with 0

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
        def find_feature_class(feature, compartments):
            """Return the compartment(s) whose name matches ``feature``."""
            matches = [compartment for compartment in compartments if re.search(compartment, feature)]
            if len(matches) > 1:
                return '-'.join(matches)
            elif matches:
                return matches[0]
            else:
                return None

        from .plot import spacrGraph

        df[name] = df['feature'].apply(lambda x: find_feature_class(x, feature_groups))

        if name == 'channel':
            df['channel'].fillna('morphology', inplace=True)

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
            if 'column_name' in scores_df.columns:
                scores_df['columnID'] = scores_df['column_name']
            if 'column' in scores_df.columns:
                scores_df['columnID'] = scores_df['column']

        if 'object_label' not in scores_df.columns:
            scores_df['object_label'] = scores_df['object']

        # Remove the 'o' prefix from 'object_label' in df, ensuring it is a string type
        df['object_label'] = df['object_label'].str.replace('o', '').astype(str)

        # Ensure 'object_label' in scores_df is also a string
        scores_df['object_label'] = scores_df['object'].astype(str)

        # Ensure all join columns have the same data type in both DataFrames
        df[['plateID', 'rowID', 'column_name', 'fieldID', 'object_label']] = df[['plateID', 'rowID', 'column_name', 'fieldID', 'object_label']].astype(str)
        scores_df[['plateID', 'rowID', 'column_name', 'fieldID', 'object_label']] = scores_df[['plateID', 'rowID', 'column_name', 'fieldID', 'object_label']].astype(str)

        # Select only the necessary columns from scores_df for merging
        scores_df = scores_df[['plateID', 'rowID', 'column_name', 'fieldID', 'object_label', settings['score_column']]]

        # Now merge DataFrames
        merged_df = pd.merge(df, scores_df, on=['plateID', 'rowID', 'column_name', 'fieldID', 'object_label'], how='inner')

        # Separate numerical features and the score column
        X = merged_df.select_dtypes(include='number').drop(columns=[settings['score_column']])
        y = merged_df[settings['score_column']]

        return X, y, merged_df
    
    X, y, merged_df = read_and_preprocess_data(settings)
    
    output = {}
    
    # Step 1: Feature Importance using Random Forest
    if settings['feature_importance'] or settings['feature_importance']:
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X, y)
        
        if settings['feature_importance']:
            print(f"Feature Importance ...")
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
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
        print(f"SHAP Analysis ...")

        # Select top N features based on Random Forest importance and fit the model on these features only
        top_features = feature_importance_df.head(settings['top_features'])['feature']
        X_top = X[top_features]

        # Refit the model on this subset of features
        model = RandomForestClassifier(random_state=42, n_jobs=settings['n_jobs'])
        model.fit(X_top, y)

        # Sample a smaller subset of rows to speed up SHAP
        if settings['shap_sample']:
            sample = int(len(X_top) / 100)
            X_sample = X_top.sample(min(sample, len(X_top)), random_state=42)
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
        
        # Aggregate SHAP values by compartment and channel
        compartment_mean = shap_df.abs().groupby(level='compartment', axis=1).mean().mean(axis=0)
        channel_mean = shap_df.abs().groupby(level='channel', axis=1).mean().mean(axis=0)

        # Calculate combined importance for each pair of compartments and channels
        combined_compartment = {}
        for i, comp1 in enumerate(compartment_mean.index):
            for comp2 in compartment_mean.index[i+1:]:
                combined_compartment[f"{comp1} + {comp2}"] = shap_df.loc[:, (comp1, slice(None))].abs().mean().mean() + \
                                                              shap_df.loc[:, (comp2, slice(None))].abs().mean().mean()
        
        combined_channel = {}
        for i, chan1 in enumerate(channel_mean.index):
            for chan2 in channel_mean.index[i+1:]:
                combined_channel[f"{chan1} + {chan2}"] = shap_df.loc[:, (slice(None), chan1)].abs().mean().mean() + \
                                                          shap_df.loc[:, (slice(None), chan2)].abs().mean().mean()

        # Prepare values and labels for radar charts
        all_compartment_importance = list(compartment_mean.values) + list(combined_compartment.values())
        all_compartment_labels = list(compartment_mean.index) + list(combined_compartment.keys())

        all_channel_importance = list(channel_mean.values) + list(combined_channel.values())
        all_channel_labels = list(channel_mean.index) + list(combined_channel.keys())

        # Create radar plots for compartments and channels
        #create_extended_radar_plot(all_compartment_importance, all_compartment_labels, "SHAP Importance by Compartment (Individual and Combined)")
        #create_extended_radar_plot(all_channel_importance, all_channel_labels, "SHAP Importance by Channel (Individual and Combined)")
        
        output['shap'] = shap_df
        
    if settings['save']:
        dst = os.path.join(settings['src'], 'results')
        os.makedirs(dst, exist_ok=True)
        for key, df in output.items(): 
            save_path = os.path.join(dst, f"{key}.csv")
            df.to_csv(save_path)
            print(f"Saved {save_path}")
        
    return output


def analyze_endodyogeny(settings):
    """Bin pathogen volumes by log2 doublings and test group proportions.

    Converts a compartment area to a volume, groups objects into doubling
    volume bins, then runs the shared chi-squared proportion plot per group.

    :param settings: dict of endodyogeny settings; see
        ``set_analyze_endodyogeny_defaults`` for keys including ``src``,
        ``tables``, ``compartment``, ``min_area_bin``, ``max_area``,
        ``max_bins``, ``um_per_px``, ``group_column``, ``level`` and ``save``.
    :returns: dict with ``data`` (binned DataFrame) and ``chi_squared`` (results DataFrame).
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

    if settings['um_per_px'] is not None:
        df[f"{settings['compartment']}_area"] = (
            df[f"{settings['compartment']}_area"] * (settings['um_per_px'] ** 2)
        )
        settings['min_area_bin'] = settings['min_area_bin'] * (settings['um_per_px'] ** 2)

    df = df[df[f"{settings['compartment']}_area"] >= settings['min_area_bin']].copy()
    
    df = df[df[f"{settings['compartment']}_area"] <= settings['max_area']].copy()

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

    df = df.dropna(subset=[settings['group_column']])

    if settings['group_column'] not in df.columns:
        available = ', '.join(df.columns.tolist())
        raise KeyError(
            f"'{settings['group_column']}' not found in DataFrame. "
            f"Available columns: {available}"
        )

    df, ordered_bin_labels = _calculate_volume_bins(
        df,
        settings['compartment'],
        settings['min_area_bin'],
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
        df = df[df['column_name']==column]
        if plate not in df.columns:
            df['plateID'] = f"plate{plate}"
        df = df[df['grna_name'].str.match(f'^{control_sgrnas[0]}$|^{control_sgrnas[1]}$')]
        grouped_df = df.groupby(['plateID', 'rowID', 'columnID'])['count'].sum().reset_index()
        grouped_df = grouped_df.rename(columns={'count': 'total_count'})
        merged_df = pd.merge(df, grouped_df, on=['plateID', 'rowID', 'column_name'])
        merged_df['fraction'] = merged_df['count'] / merged_df['total_count']
        merged_df['prc'] = merged_df['plateID'].astype(str) + '_' + merged_df['rowID'].astype(str) + '_' + merged_df['column_name'].astype(str)
        return merged_df

    def plot_multi_channel_heatmap(df, column='c3', cmap='coolwarm'):
        """Plot a per-well heatmap with each classifier channel as a column.

        :param df: DataFrame with score columns keyed by channel.
        :param column: value in ``columnID`` used to filter rows. Default ``'c3'``.
        :param cmap: matplotlib/seaborn colormap. Default ``'coolwarm'``.
        :returns: the matplotlib Figure.
        """
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
    if 'row_number' in merged_df.columns:
        merged_df = merged_df.drop('row_num', axis=1)
    mae_df = calculate_mae(merged_df)
    if 'row_number' in mae_df.columns:
        mae_df = mae_df.drop('row_num', axis=1)
        
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
        sns.barplot(x=effect_sizes.index, y=effect_sizes.values, palette="viridis", hue=None, legend=False)

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
    effect_sizes = _compute_effect_sizes(correlation_matrix, grna_dict, save_folder, save)
