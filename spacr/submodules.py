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
    """
    A PyTorch Dataset for lazy loading and optional augmentation of images and segmentation masks
    for training Cellpose-based models.

    Images and labels are loaded from file paths on-the-fly, optionally normalized and resized,
    and can be augmented with basic geometric transformations.

    Args:
        image_files (list of str): List of file paths to image files.
        label_files (list of str): List of file paths to corresponding label files.
        settings (dict): Dictionary containing dataset settings:
            - 'normalize' (bool): Whether to apply percentile-based intensity normalization.
            - 'percentiles' (list of int): Two-element list specifying lower and upper percentiles (default: [2, 99]).
            - 'target_size' (int): Desired output image size (height and width).
        randomize (bool, optional): Whether to shuffle the dataset order. Defaults to True.
        augment (bool, optional): Whether to apply 8-way geometric data augmentation. Defaults to False.
    """
    def __init__(self, image_files, label_files, settings, randomize=True, augment=False):
        """
        Initialize the CellposeLazyDataset.

        Args:
            image_files (list of str): Paths to input image files.
            label_files (list of str): Paths to corresponding label files.
            settings (dict): Configuration dictionary with keys:
                - 'normalize' (bool)
                - 'percentiles' (list of int)
                - 'target_size' (int)
            randomize (bool, optional): Shuffle file order. Defaults to True.
            augment (bool, optional): Enable 8-fold augmentation. Defaults to False.
        """
        combined = list(zip(image_files, label_files))
        if randomize:
            random.shuffle(combined)
        self.image_files, self.label_files = zip(*combined)
        self.normalize = settings['normalize']
        self.percentiles = settings.get('percentiles', [2, 99])
        self.target_size = settings['target_size']
        self.augment = augment

    def __len__(self):
        """
        Return the number of samples in the dataset.

        If augmentation is enabled, each sample contributes 8 variants.

        Returns:
            int: Total number of samples (augmented if applicable).
        """
        return len(self.image_files) * (8 if self.augment else 1)

    def apply_augmentation(self, image, label, aug_idx):
        """
        Apply one of 8 geometric augmentations to an image-label pair.

        Augmentations include rotations (90°, 180°, 270°) and horizontal/vertical flips.

        Args:
            image (ndarray): Input image array.
            label (ndarray): Corresponding label array.
            aug_idx (int): Index from 0 to 7 specifying the augmentation to apply.

        Returns:
            tuple: Augmented (image, label) pair.
        """
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
        """
        Retrieve a sample by index, optionally applying augmentation and preprocessing.

        Loads the image and label, normalizes intensity if specified, applies augmentation,
        and resizes to the target shape.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - image (np.ndarray): Preprocessed image, shape (target_size, target_size), dtype float32.
                - label (np.ndarray): Resized label mask, shape (target_size, target_size), dtype uint8.
        """
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
    """
    Train a Cellpose model on custom images and masks using specified settings.

    This function prepares training data from `train/images` and `train/masks` subfolders within 
    the provided `settings['src']` directory. It constructs a model name based on training parameters, 
    initializes the Cellpose model, and trains it using the specified number of epochs and hyperparameters.

    The dataset can be augmented up to 8-fold, and training images and masks are matched by filename. 
    Training progress is visualized (if possible), and the model is saved in `models/cellpose_model`.

    Args:
        settings (dict): Dictionary with the following required keys:
            - 'src' (str): Root directory containing `train/images`, `train/masks`, and `models`.
            - 'target_size' (int): Side length to which images/masks are resized.
            - 'model_name' (str): Base name of the model.
            - 'n_epochs' (int): Number of training epochs.
            - 'batch_size' (int): Number of images to train per batch.
            - 'learning_rate' (float): Learning rate for training.
            - 'weight_decay' (float): Weight decay (L2 regularization).
            - 'augment' (bool): Whether to use 8-fold data augmentation.

    Side Effects:
        - Trains a model and saves it to `models/cellpose_model/`.
        - Writes training settings using `save_settings()`.
        - Optionally visualizes a training batch using `plot_cellpose_batch()`.
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
    except:
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
    """
    Evaluate a pretrained Cellpose model on a test dataset and report segmentation performance.

    This function loads test images and ground-truth masks from a specified directory structure,
    applies the Cellpose model to predict masks, and compares them to the ground truth using
    object-level metrics and Jaccard index. Results are saved and visualized if specified.

    Args:
        settings (dict): Dictionary with the following keys:
            - 'src' (str): Root directory containing `test/images` and `test/masks`.
            - 'model_path' (str): Path to the pretrained Cellpose model.
            - 'batch_size' (int): Number of images per inference batch.
            - 'FT' (float): Flow threshold for Cellpose segmentation.
            - 'CP_probability' (float): Cell probability threshold for segmentation.
            - 'target_size' (int): Size to which input images are resized.
            - 'save' (bool): Whether to save segmentation visualizations and results.

    Side Effects:
        - Loads and processes image and mask files from `test/images` and `test/masks`.
        - Evaluates segmentation and computes the following metrics per image:
            * Jaccard index
            * Number of predicted and ground-truth objects
            * Mean object area (true and predicted)
            * True positives, false positives, false negatives
            * Precision, recall, F1-score, accuracy
        - Saves figures showing input, true mask, predicted mask, probability map, and flow.
        - Optionally saves results to `results/test_results.csv`.
        - Prints performance summary to stdout.
    """
    from .utils import save_settings, print_progress
    from .settings import get_default_test_cellpose_model_settings
    
    def plot_cellpose_resilts(i, j, results_dir, img, lbl, pred, flow):
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
    """
    Apply a pretrained Cellpose model to a folder of images and extract object-level measurements.

    This function processes all `.tif` images in the specified source directory using a pretrained 
    Cellpose model, generates segmentation masks, and computes region properties (e.g., area) for each 
    detected object. It optionally applies circular masking to limit the field of view and saves results 
    and figures if enabled.

    Args:
        settings (dict): Dictionary with the following required keys:
            - 'src' (str): Directory containing the input `.tif` images.
            - 'model_path' (str): Path to the pretrained Cellpose model file.
            - 'batch_size' (int): Number of images processed per batch.
            - 'FT' (float): Flow threshold for Cellpose segmentation.
            - 'CP_probability' (float): Cell probability threshold for segmentation.
            - 'target_size' (int): Resize target for input images.
            - 'save' (bool): If True, saves visualizations and CSV results.
            - 'circularize' (bool, optional): If True, applies a circular mask to each prediction.

    Side Effects:
        - Generates and saves segmented mask visualizations (if `save=True`) to `results/`.
        - Extracts per-object measurements (area) and saves:
            * `results/measurements.csv`: one row per object.
            * `results/summary.csv`: average object area and count per image.
        - Displays progress and timing for each batch.
    """
    from .settings import get_default_apply_cellpose_model_settings
    from .utils import save_settings, print_progress
    
    def plot_cellpose_result(i, j, results_dir, img, pred, flow):
        
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
    """
    Display a batch of input images and their corresponding label masks.

    This function plots two rows of subplots:
        - Top row: grayscale input images.
        - Bottom row: corresponding label masks with randomly colored regions.

    Args:
        images (list of np.ndarray): List of 2D grayscale input images.
        labels (list of np.ndarray): List of 2D label masks corresponding to the images.
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
    """
    Analyze the fraction of objects above a threshold per well and annotate them accordingly.

    This function loads merged object-level measurements from a SQLite database, applies optional
    filtering, and annotates each object as 'above' or 'below' a specified threshold on a value column.
    It then summarizes the number and fraction of above-threshold objects per condition and well.
    Results are merged with well metadata and saved as a CSV.

    Args:
        settings (dict): Dictionary with the following required keys:
            - 'src' (str): Root directory containing `measurements.db` and `rename_log.csv`.
            - 'tables' (list of str): Table names to extract from the SQLite database.
            - 'value_col' (str): Column to apply the threshold to.
            - 'threshold' (float): Threshold value to classify objects.
            - 'filter_1' (tuple or None): Optional filter in the form (column, min_value), or None.

    Returns:
        pd.DataFrame: Merged DataFrame with annotation summary and metadata, also saved to `result.csv`.

    Side Effects:
        - Reads measurement data from `measurements/measurements.db`.
        - Reads well metadata from `rename_log.csv`.
        - Writes annotated results to `result.csv` in the same directory.
        - Displays the final DataFrame in notebook or console.
    """
    from .io import _read_and_merge_data
    from .utils import save_settings
    from .settings import default_settings_analyze_percent_positive
    
    settings = default_settings_analyze_percent_positive(settings)
    
    def translate_well_in_df(csv_loc):
        """
        Extract and translate well metadata from a CSV file containing renamed TIFF filenames.

        This function parses the 'Renamed TIFF' column to extract plate and well information,
        generates plate-well identifiers, and translates well positions to row and column IDs.
        It also constructs a unique `prc` (plate_row_column) identifier for each well.

        Args:
            csv_loc (str): Path to a CSV file containing a 'Renamed TIFF' column.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - 'plateID': extracted plate identifier
                - 'well': extracted well identifier (e.g., A01)
                - 'plate_well': combined plate and well string
                - 'rowID': converted row index (e.g., 'r1' for row A)
                - 'column_name': converted column index (e.g., 'c1' for column 01)
                - 'fieldID': field identifier (default: 'f1')
                - 'prc': combined identifier in the form 'p<plate>_r<row>_c<column>'
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
        """
        Annotate and summarize a DataFrame based on a threshold.

        Parameters:
        - df: pandas.DataFrame
        - value_col: str, column name to apply threshold on
        - condition_col: str, column name for experimental condition
        - well_col: str, column name for wells
        - threshold: float, threshold value for annotation
        - annotation_col: str, name of the new annotation column

        Returns:
        - df: annotated DataFrame
        - summary_df: DataFrame with counts and fractions per condition and well
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
    """
    Analyze host protein recruitment to the pathogen vacuole across experimental conditions.

    This function loads object-level measurements, applies size and intensity-based filtering, computes
    recruitment scores as the ratio of pathogen-to-cytoplasm intensities, and aggregates results at the
    cell and well level. Recruitment is annotated and visualized, and filtered results are saved.

    Args:
        settings (dict): Dictionary with the following required keys:
            - 'src' (str): Path to the root folder or `measurements.db` file.
            - 'cell_types', 'pathogen_types', 'treatments' (list of str): Experimental conditions to annotate.
            - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata sources.
            - 'channel_of_interest' (int): Channel index for recruitment quantification.
            - 'channel_dims' (list of int): Channels to calculate recruitment for.
            - 'cell_chann_dim', 'nucleus_chann_dim', 'pathogen_chann_dim' (int or None): Mask channel assignments.
            - 'cell_size_range', 'nucleus_size_range', 'pathogen_size_range' (list): Min/max object size filters.
            - 'cell_intensity_range', etc. (list): Intensity-based object filters.
            - 'target_intensity_min' (float): Minimum required target channel intensity in cell.
            - 'nuclei_limit', 'pathogen_limit' (int or None): Max number of nuclei/pathogens to load.
            - 'cells_per_well' (int): Minimum number of cells per well to retain.
            - 'plot' (bool): Whether to plot cell outlines on merged images.
            - 'plot_control' (bool): Whether to plot control comparisons.
            - 'plot_nr' (int): Maximum number of outlines to plot.
            - 'figuresize' (float): Size of recruitment plots in inches.

    Returns:
        list: [cell-level DataFrame, well-level DataFrame] after filtering and recruitment analysis.

    Side Effects:
        - Reads from and writes to files in the source directory.
        - Plots recruitment distributions and controls.
        - Saves `cell_level_results.csv` and `well_level_results.csv` via `_results_to_csv`.
        - Displays status messages and summary statistics.
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
        if not settings['target_intensity_min'] is None or not settings['target_intensity_min'] is 0:
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
    """
    Analyze plaque-like structures from microscopy images using a pretrained Cellpose model.

    This function applies a custom-trained Cellpose model to identify and segment plaques in `.tif` images.
    It calculates object-level statistics including object count and area, and saves the results into a 
    SQLite database with three tables: `summary`, `details`, and `stats`.

    Args:
        settings (dict): Dictionary with the following keys:
            - 'src' (str): Source folder containing input `.tif` images.
            - 'masks' (bool): If True, (re)generate segmentation masks using Cellpose.
            - 'custom_model' (str, optional): Will be set internally to the plaque model path.
            - Any keys required by `get_analyze_plaque_settings()` or `identify_masks_finetune()`.

    Side Effects:
        - Downloads pretrained plaque segmentation model if not available.
        - Saves generated masks (if `masks=True`) to `src/masks/`.
        - Analyzes all `.tif` masks in the folder.
        - Creates and saves the following tables to `plaques_analysis.db`:
            * `summary`: file name, number of plaques, average size
            * `details`: file name, individual plaque sizes
            * `stats`: file name, plaque count, average size, and standard deviation
        - Prints progress and completion messages.
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
    """
    Count and summarize annotated phenotypes in a measurement database.

    This function reads annotation data from the `measurements.db` database and performs the following:
      - Counts the number of unique phenotype labels in the specified annotation column.
      - Computes how many unique labels occur per well (defined by plateID, rowID, columnID).
      - Computes the count of each phenotype value per well.
      - Outputs a CSV file with the per-well phenotype counts.

    Args:
        settings (dict): Dictionary with the following keys:
            - 'src' (str): Path to a measurement database or directory containing it.
            - 'annotation_column' (str): Column name containing the phenotype annotations.

    Returns:
        None

    Side Effects:
        - Displays number of unique phenotype values.
        - Displays a table of unique value counts per well.
        - Saves a CSV file `phenotype_counts.csv` summarizing value counts per well
          (one row per well, one column per phenotype).
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

def compare_reads_to_scores(reads_csv, scores_csv, empirical_dict={'r1':(90,10),'r2':(90,10),'r3':(80,20),'r4':(80,20),'r5':(70,30),'r6':(70,30),'r7':(60,40),'r8':(60,40),'r9':(50,50),'r10':(50,50),'r11':(40,60),'r12':(40,60),'r13':(30,70),'r14':(30,70),'r15':(20,80),'r16':(20,80)},
                            pc_grna='TGGT1_220950_1', nc_grna='TGGT1_233460_4', 
                            y_columns=['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction'], 
                            column='columnID', value='c3', plate=None, save_paths=None):
    """
    Compare Cellpose-based phenotypic classification scores with sequencing read distributions
    from positive and negative control gRNAs across plate wells.

    This function merges phenotype classification scores and read count data, calculates gRNA
    fractions and class predictions per well, and overlays empirical expectations for controls
    to assess model calibration. It generates and saves line plots for selected y-axis columns.

    Args:
        reads_csv (str or list of str): Path(s) to CSV file(s) containing read counts and gRNA names.
        scores_csv (str or list of str): Path(s) to CSV file(s) with classification results per object.
        empirical_dict (dict): Mapping of rowID to (positive, negative) control counts.
        pc_grna (str): Name of the positive control gRNA.
        nc_grna (str): Name of the negative control gRNA.
        y_columns (list of str): Columns in the merged dataframe to plot on y-axis.
        column (str): Column to subset on (e.g., 'columnID').
        value (str): Value to filter the column on (e.g., 'c3').
        plate (str, optional): Plate name to assign if not present in the CSVs.
        save_paths (list of str, optional): List of two paths to save the plots as PDF.

    Returns:
        list: Two matplotlib Figure objects, one for pc_fraction and one for nc_fraction x-axis plots.

    Side Effects:
        - Reads and merges input CSVs.
        - Computes gRNA read fractions and classification fractions per well.
        - Merges empirical expectations based on rowID.
        - Displays merged DataFrame.
        - Saves two line plots (if `save_paths` is specified).
    """
    
    def calculate_well_score_fractions(df, class_columns='cv_predictions'):
        """
        Calculate the fraction of predicted classes (e.g., class_0 and class_1) per well.

        This function groups a DataFrame by plate, row, and column IDs to compute the total number
        of objects and the number of objects predicted in each class per well. It returns a summary
        DataFrame that includes the absolute and relative frequencies of class 0 and class 1 predictions.

        Args:
            df (pd.DataFrame): Input DataFrame containing at least:
                - 'plateID', 'rowID', 'columnID': to define wells
                - A column specified by `class_columns` with predicted class labels (e.g., 0, 1)
            class_columns (str): Name of the column containing predicted class labels. Defaults to 'cv_predictions'.

        Returns:
            pd.DataFrame: Summary DataFrame with columns:
                - 'plateID', 'rowID', 'columnID', 'prc'
                - 'total_rows': total number of entries per well
                - 'class_0': number of class 0 predictions
                - 'class_1': number of class 1 predictions
                - 'class_0_fraction': class_0 / total_rows
                - 'class_1_fraction': class_1 / total_rows

        Raises:
            ValueError: If required columns ('plateID', 'rowID', 'columnID') are missing in `df`.
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
        """
        Create a customizable line plot for one or more y-columns against a common x-axis.

        This function supports grouped or ungrouped line plots and includes options for custom axis
        labels, title, figure size, and color themes. When multiple y-columns are specified, each is 
        plotted as an individual line. Optionally saves the plot as a high-resolution PDF.

        Args:
            df (pd.DataFrame): Input DataFrame containing the plotting data.
            x_column (str): Column name to use for the x-axis.
            y_columns (str or list of str): Column(s) to use for the y-axis. If a list, each column is plotted as a separate line.
            group_column (str, optional): Column name for grouping lines (only used if `y_columns` is a single string).
            xlabel (str, optional): Label for the x-axis. Defaults to `x_column` name.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Value'.
            title (str, optional): Title of the plot. Defaults to 'Line Plot'.
            figsize (tuple, optional): Figure size in inches. Defaults to (10, 6).
            save_path (str, optional): If provided, saves the figure as a PDF to this path.
            theme (str, optional): Seaborn color palette theme. Defaults to 'deep'.

        Returns:
            matplotlib.figure.Figure: The generated figure object.

        Side Effects:
            - Displays the plot.
            - Saves the figure as a PDF if `save_path` is specified.
        """
        def _set_theme(theme):
            """Set the Seaborn theme and reorder colors if necessary."""

            def __set_reordered_theme(theme='deep', order=None, n_colors=100, show_theme=False):
                """Set and reorder the Seaborn color palette."""
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
        """
        Calculate the ratio and fractional abundance of two gRNAs per well.

        This function filters the input DataFrame for two specified gRNAs, aggregates their read
        fractions and counts per well (`prc`), and computes:
        - the fraction ratio (grna1 / grna2),
        - total read count per well,
        - individual gRNA read fractions per well.

        Args:
            df (pd.DataFrame): Input DataFrame with columns ['prc', 'grna_name', 'fraction', 'count'].
            grna1 (str): Name of the positive control or reference gRNA.
            grna2 (str): Name of the negative control or comparison gRNA.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'prc': well identifier (plate_row_column)
                - '<grna1>_count': total read count of grna1 per well
                - '<grna2>_count': total read count of grna2 per well
                - 'fraction_ratio': grna1 fraction / grna2 fraction
                - 'total_reads': total reads of both gRNAs
                - '<grna1>_fraction': grna1 reads / total_reads
                - '<grna2>_fraction': grna2 reads / total_reads
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
        """
        Calculate the fractional read count for each gRNA within a well.

        This function computes the total number of reads per well (identified by plateID, rowID, columnID),
        and then computes the fraction of reads contributed by each gRNA relative to the total reads
        in that well.

        Args:
            df (pd.DataFrame): DataFrame containing at least the columns:
                - 'plateID', 'rowID', 'columnID' (to define wells)
                - `count_column` (default: 'count'), the read count per gRNA instance
            count_column (str): Name of the column holding raw read counts. Defaults to 'count'.

        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - 'prc': combined plate_row_column identifier
                - 'total_counts': total reads per well
                - 'fraction': count / total_counts for each row

        Raises:
            ValueError: If any of 'plateID', 'rowID', or 'columnID' are missing in the input DataFrame.
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

def interperate_vision_model(settings={}):
    """
    Interpret a vision-based machine learning model using multiple feature attribution strategies.

    This function reads merged object-level data and model scores, engineers relative features, and applies:
      - Random Forest feature importance
      - Permutation importance
      - SHAP (SHapley Additive exPlanations) values

    The model interpretation is performed across cellular compartments and imaging channels, and
    results can be visualized or saved for downstream analysis.

    Args:
        settings (dict): Dictionary of configuration parameters including:
            - 'src' (str): Path to the data directory.
            - 'scores' (str): Path to the score CSV file.
            - 'tables' (list of str): Feature group keywords for compartmental grouping.
            - 'channels' (list of str): Feature group keywords for channel grouping.
            - 'score_column' (str): Column name for predicted class scores.
            - 'nuclei_limit' (int): Optional filter on number of nuclei to load.
            - 'pathogen_limit' (int): Optional filter on number of pathogens to load.
            - 'feature_importance' (bool): Whether to compute Random Forest feature importances.
            - 'permutation_importance' (bool): Whether to compute permutation-based importances.
            - 'shap' (bool): Whether to compute SHAP values.
            - 'shap_sample' (bool): Whether to subsample SHAP input for speed.
            - 'top_features' (int): Number of top features to plot.
            - 'include_all' (bool): Whether to include a row summing all feature groups.
            - 'n_jobs' (int): Number of parallel jobs to use for model training or SHAP.
            - 'save' (bool): Whether to save output CSV files.

    Returns:
        dict: A dictionary of interpretation outputs with the following possible keys:
            - 'feature_importance': Raw feature importances (Random Forest)
            - 'feature_importance_compartment': Grouped by compartment
            - 'feature_importance_channel': Grouped by imaging channel
            - 'permutation_importance': Permutation importance values
            - 'shap': SHAP values for each object-feature pair

    Side Effects:
        - Displays importance plots.
        - Optionally saves CSVs to `<src>/results/`.
        - Merges and preprocesses input data, including relative feature construction.
    """

    from .io import _read_and_merge_data

    def generate_comparison_columns(df, compartments=['cell', 'nucleus', 'pathogen', 'cytoplasm']):

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

    def group_feature_class(df, feature_groups=['cell', 'cytoplasm', 'nucleus', 'pathogen'], name='compartment', include_all=False):
        """
        Group features by compartment or channel and sum their importance scores.

        This function identifies the compartment or channel associated with each feature based on
        string matching, and computes the total importance of each group. Optionally, it includes
        a row summing the total importance across all groups.

        Args:
            df (pd.DataFrame): DataFrame with at least the columns:
                - 'feature': feature names to classify
                - 'importance': importance values associated with each feature
            feature_groups (list of str): Keywords used to assign features to groups (e.g., compartments or channels).
            name (str): Column name to assign group classification (e.g., 'compartment' or 'channel').
            include_all (bool): If True, include a summary row for total importance across all groups.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - `name` (e.g., 'compartment', 'channel')
                - `<name>_importance_sum`: total importance per group
        """
        # Function to determine compartment based on multiple matches
        def find_feature_class(feature, compartments):
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
        """
        Create a radar (spider) plot to visualize multivariate values on a circular axis.

        This function plots a closed polygon where each axis represents a different category
        (e.g., feature group), allowing for intuitive comparison of relative magnitudes.

        Args:
            values (list or array-like): List of values to plot. Should match the length of `labels`.
            labels (list of str): Labels for each axis in the radar chart.
            title (str): Title of the plot.

        Returns:
            None

        Side Effects:
            - Displays the radar chart using matplotlib.
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
        """
        Extract the compartment and imaging channel from a feature name.

        This function parses a feature name string to identify:
        - The compartment (first substring before an underscore, with 'cells' mapped to 'cell')
        - The imaging channel(s), based on the presence of substrings like 'channel_0', etc.

        Args:
            feature_name (str): The name of the feature to parse.

        Returns:
            tuple:
                - compartment (str): The identified compartment (e.g., 'cell', 'nucleus').
                - channel (str): Channel label or 'morphology' if no channel found.
                                If multiple channels are present, they are joined by ' + '.
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
        """
        Load, preprocess, and merge measurement data and classification scores.

        This function:
        - Reads merged object-level measurements from a SQLite database.
        - Adds relative comparison features between compartments (e.g., nucleus/cytoplasm).
        - Loads classification scores and aligns them with the measurement data.
        - Ensures consistent formatting of join keys (plateID, rowID, etc.).
        - Merges the two datasets on object identifiers and returns the feature matrix and labels.

        Args:
            settings (dict): Dictionary with required keys:
                - 'src' (str): Base path to the dataset directory.
                - 'tables' (list of str): Tables to load from the database.
                - 'nuclei_limit' (int or None): Optional limit on number of nuclei to load.
                - 'pathogen_limit' (int or None): Optional limit on number of pathogens to load.
                - 'scores' (str): Path to the CSV file with prediction/classification scores.
                - 'score_column' (str): Column name in scores CSV used as prediction target.

        Returns:
            tuple:
                - X (pd.DataFrame): Numerical feature matrix (excluding the score column).
                - y (pd.Series): Target values corresponding to the score column.
                - merged_df (pd.DataFrame): Full merged DataFrame including both features and scores.

        Side Effects:
            - Prints the number of columns in the final feature-expanded DataFrame.
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

def _plot_proportion_stacked_bars(settings, df, group_column, bin_column, prc_column='prc', level='object'):
    """
    Generate a stacked bar plot of bin proportions by group and perform a chi-squared test for association.

    Depending on the `level` parameter, this function can:
      - Plot mean ± SD of binned proportions across wells (`level='well'`).
      - Plot object-level proportions per group (`level='object'`).

    The function also calculates and prints the chi-squared test statistic and p-value
    for independence between group and bin category distributions.

    Args:
        settings (dict): Dictionary with settings, must include:
            - 'um_per_px' (float or None): Micron-per-pixel scaling factor. If None, volume is in px³.
        df (pd.DataFrame): DataFrame containing binned volume information.
        group_column (str): Column name indicating group assignment (e.g., treatment, genotype).
        bin_column (str): Column name for the binned volume categories.
        prc_column (str): Column name specifying well ID. Only used if level='well'. Default is 'prc'.
        level (str): Either 'object' (default) or 'well', determines plot granularity.

    Returns:
        tuple:
            - chi2 (float): Chi-squared test statistic from raw data.
            - p (float): p-value from the chi-squared test.
            - dof (int): Degrees of freedom from the test.
            - expected (ndarray): Expected frequencies under independence.
            - raw_counts (pd.DataFrame): Contingency table used for chi-squared test.
            - fig (matplotlib.figure.Figure): The figure object containing the plot.

    Side Effects:
        - Displays a stacked bar plot using matplotlib.
        - Prints the chi-squared test result.
    """
    # Always calculate chi-squared on raw data
    raw_counts = df.groupby([group_column, bin_column]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(raw_counts)
    print(f"Chi-squared test statistic (raw data): {chi2:.4f}")
    print(f"p-value (raw data): {p:.4e}")

    # Extract bin labels and indices for formatting the legend in the correct order
    bin_labels = df[bin_column].cat.categories if pd.api.types.is_categorical_dtype(df[bin_column]) else sorted(df[bin_column].unique())
    bin_indices = range(1, len(bin_labels) + 1)
    legend_labels = [f"{index}: {label}" for index, label in zip(bin_indices, bin_labels)]

    # Plot based on level setting
    if level == 'well':
        # Aggregate by well for mean ± SD visualization
        well_proportions = (
            df.groupby([group_column, prc_column, bin_column])
            .size()
            .groupby(level=[0, 1])
            .apply(lambda x: x / x.sum())
            .unstack(fill_value=0)
        )
        mean_proportions = well_proportions.groupby(group_column).mean()
        std_proportions = well_proportions.groupby(group_column).std()

        ax = mean_proportions.plot(
            kind='bar', stacked=True, yerr=std_proportions, capsize=5, colormap='viridis', figsize=(12, 8)
        )
        plt.title('Proportion of Volume Bins by Group (Mean ± SD across wells)')
    else:
        # Object-level plotting without aggregation
        group_counts = df.groupby([group_column, bin_column]).size()
        group_totals = group_counts.groupby(level=0).sum()
        proportions = group_counts / group_totals
        proportion_df = proportions.unstack(fill_value=0)

        ax = proportion_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
        plt.title('Proportion of Volume Bins by Group')

    plt.xlabel('Group')
    plt.ylabel('Proportion')

    # Update legend with formatted labels, maintaining correct order
    volume_unit = "px³" if settings['um_per_px'] is None else "µm³"
    plt.legend(legend_labels, title=f'Volume Range ({volume_unit})', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    fig = plt.gcf() 
    return chi2, p, dof, expected, raw_counts, fig

def analyze_endodyogeny(settings):
    """
    Analyze intracellular pathogen replication (endodyogeny) based on compartment volume measurements.

    This function bins single-cell or single-object volume estimates (e.g., pathogen area^1.5) into
    discrete categories, annotates experimental conditions, and visualizes the volume distribution
    across groups as a stacked bar plot. It also performs a chi-squared test to assess whether the 
    volume distribution is significantly different between groups.

    Args:
        settings (dict): Configuration dictionary with required keys:
            - 'src' (str or list): Path(s) to directories containing measurements.db.
            - 'tables' (list of str): Tables to extract from the SQLite database.
            - 'compartment' (str): Compartment to analyze (e.g., 'pathogen').
            - 'min_area_bin' (float): Minimum 2D area threshold to consider (in px² or µm²).
            - 'max_bins' (int or None): Maximum number of volume bins to display.
            - 'group_column' (str): Column defining experimental groups (e.g., genotype or treatment).
            - 'cell_types', 'pathogen_types', 'treatments' (list): Experimental groups for annotation.
            - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata source paths.
            - 'nuclei_limit', 'pathogen_limit' (int or None): Max number of objects to load per type.
            - 'change_plate' (bool): Whether to modify plate names for uniqueness.
            - 'verbose' (bool): Whether to print debug output.
            - 'um_per_px' (float or None): Pixel-to-micrometer conversion (if None, volume remains in px³).
            - 'level' (str): Either 'object' or 'well' for aggregation level.
            - 'cmap' (str): Colormap to use in the plot.
            - 'save' (bool): Whether to save results and figures.

    Returns:
        dict: A dictionary with:
            - 'data': Annotated and binned DataFrame.
            - 'chi_squared': Chi-squared test summary DataFrame.

    Side Effects:
        - Displays a stacked bar plot showing volume bin proportions by group.
        - Performs chi-squared test and prints p-value.
        - Saves data and plots to `results/analyze_endodyogeny/` if `save=True`.
    """
    from .utils import annotate_conditions, save_settings
    from .io import _read_and_merge_data
    from .settings import set_analyze_endodyogeny_defaults
    from .plot import plot_proportion_stacked_bars

    def _calculate_volume_bins(df, compartment='pathogen', min_area_bin=500, max_bins=None, verbose=False):
        area_column = f'{compartment}_area'
        df[f'{compartment}_volume'] = df[area_column] ** 1.5
        min_volume_bin = min_area_bin ** 1.5
        max_volume = df[f'{compartment}_volume'].max()

        # Generate bin edges as floats, and filter out any duplicate edges
        bins = [min_volume_bin * (2 ** i) for i in range(int(np.ceil(np.log2(max_volume / min_volume_bin)) + 1))]
        bins = sorted(set(bins))  # Ensure bin edges are unique

        # Create bin labels as ranges with decimal precision for float values (e.g., "500.0-1000.0")
        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
        if verbose:
            print('Volume bins:', bins)
            print('Volume bin labels:', bin_labels)

        # Apply the bins to create a new column with the binned labels
        df[f'{compartment}_volume_bin'] = pd.cut(df[f'{compartment}_volume'], bins=bins, labels=bin_labels, right=False)
        
        # Create a bin index column (numeric version of bins)
        df['bin_index'] = pd.cut(df[f'{compartment}_volume'], bins=bins, labels=range(1, len(bins)), right=False).astype(int)

        # Adjust bin indices and labels based on max_bins
        if max_bins is not None:
            df.loc[df['bin_index'] > max_bins, 'bin_index'] = max_bins
            
            # Update bin labels to reflect capped bins
            bin_labels = bin_labels[:max_bins - 1] + [f">{bins[max_bins - 1]:.2f}"]
            df[f'{compartment}_volume_bin'] = df['bin_index'].map(
                {i + 1: label for i, label in enumerate(bin_labels)}
            )

        if verbose:
            print(df[[f'{compartment}_volume', f'{compartment}_volume_bin', 'bin_index']].head())

        return df

    settings = set_analyze_endodyogeny_defaults(settings)
    save_settings(settings, name='analyze_endodyogeny', show=True)
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
        pathogen_limit=settings['pathogen_limit'],
        change_plate=settings['change_plate']
    )
    
    if not settings['um_per_px'] is None:
        df[f"{settings['compartment']}_area"] = df[f"{settings['compartment']}_area"] * (settings['um_per_px'] ** 2)
        settings['min_area_bin'] = settings['min_area_bin'] * (settings['um_per_px'] ** 2)
    
    df = df[df[f"{settings['compartment']}_area"] >= settings['min_area_bin']]
    
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
    
    df = df.dropna(subset=[settings['group_column']])
    df = _calculate_volume_bins(df, settings['compartment'], settings['min_area_bin'], settings['max_bins'], settings['verbose'])
    output['data'] = df
    
    
    if settings['level'] == 'plateID':
        prc_column = 'plateID'
    else:
        prc_column = 'prc'
    
    # Perform chi-squared test and plot
    results_df, pairwise_results_df, fig = plot_proportion_stacked_bars(settings, df, settings['group_column'], bin_column=f"{settings['compartment']}_volume_bin", prc_column=prc_column, level=settings['level'], cmap=settings['cmap'])
    
    # Extract bin labels and indices for formatting the legend in the correct order
    bin_labels = df[f"{settings['compartment']}_volume_bin"].cat.categories if pd.api.types.is_categorical_dtype(df[f"{settings['compartment']}_volume_bin"]) else sorted(df[f"{settings['compartment']}_volume_bin"].unique())
    bin_indices = range(1, len(bin_labels) + 1)
    legend_labels = [f"{index}: {label}" for index, label in zip(bin_indices, bin_labels)]
    
    # Update legend with formatted labels, maintaining correct order
    volume_unit = "px³" if settings['um_per_px'] is None else "µm³"
    plt.legend(legend_labels, title=f'Volume Range ({volume_unit})', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    
    output['chi_squared'] = results_df

    if settings['save']:
        # Save DataFrame to CSV
        output_dir = os.path.join(settings['src'][0], 'results', 'analyze_endodyogeny')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'chi_squared_results.csv')
        output_path_data = os.path.join(output_dir, 'data.csv')
        output_path_pairwise = os.path.join(output_dir, 'chi_squared_results.csv')
        output_path_fig = os.path.join(output_dir, 'chi_squared_results.pdf')
        fig.savefig(output_path_fig, dpi=300, bbox_inches='tight')
        results_df.to_csv(output_path, index=False)
        df.to_csv(output_path_data, index=False)
        pairwise_results_df.to_csv(output_path_pairwise, index=False)
        print(f"Chi-squared results saved to {output_path}")
        
    plt.show()     

    return output

def analyze_class_proportion(settings):
    """
    Analyze class frequency distributions across experimental groups and perform statistical tests.

    This function compares the proportion of classification outcomes (e.g., phenotypic classes)
    across specified experimental groups. It generates stacked bar plots, performs chi-squared
    and parametric/non-parametric statistical tests, and saves annotated data and results.

    Args:
        settings (dict): Dictionary of parameters, including:
            - 'src' (str or list): Path(s) to measurement directories.
            - 'tables' (list): Tables to load from measurements.db.
            - 'class_column' (str): Column indicating class or category.
            - 'group_column' (str): Column indicating grouping variable (e.g., genotype or treatment).
            - 'cell_types', 'pathogen_types', 'treatments' (list): Experimental group values.
            - 'cell_plate_metadata', 'pathogen_plate_metadata', 'treatment_plate_metadata' (str): Metadata paths.
            - 'nuclei_limit', 'pathogen_limit' (int or None): Optional load limits.
            - 'verbose' (bool): Whether to print debug information.
            - 'level' (str): Aggregation level for chi-squared test ('object' or 'well').
            - 'save' (bool): Whether to save results and figures.

    Returns:
        dict: A dictionary containing:
            - 'data': Annotated DataFrame.
            - 'chi_squared': Chi-squared summary results.

    Side Effects:
        - Annotates DataFrame with experimental conditions.
        - Performs chi-squared test on class distribution and plots stacked bars.
        - Plots class heatmap across plates.
        - Performs and prints:
            * Shapiro-Wilk normality test
            * Levene’s test for equal variances
            * ANOVA or Kruskal-Wallis test
            * Post hoc pairwise comparisons
        - Optionally saves all results and figures in a `results/analyze_class_proportion/` subfolder.
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
    """
    Generate a heatmap comparing predicted classification scores from multiple models
    with empirical gRNA-based fraction measurements across wells.

    This function:
      - Aggregates prediction scores from subfolders.
      - Calculates the gRNA-based empirical fraction from sequencing data.
      - Merges predictions with empirical scores and computes MAE (mean absolute error).
      - Displays and optionally saves a multi-channel heatmap and the corresponding MAE results.

    Args:
        settings (dict): Configuration dictionary with the following keys:
            - 'folders' (list or str): Folder(s) containing model output subfolders with CSVs.
            - 'csv_name' (str): Name of the CSV file to search for in each model output subfolder.
            - 'data_column' (str): Column in CSVs containing the prediction scores.
            - 'cv_csv' (str): CSV with cross-validated model scores.
            - 'data_column_cv' (str): Column in `cv_csv` with scores to merge.
            - 'csv' (str): CSV with gRNA counts used for computing empirical fractions.
            - 'control_sgrnas' (list): List of two control gRNA names used to compute the fraction.
            - 'fraction_grna' (str): One of the control gRNAs to extract the empirical fraction from.
            - 'plateID' (int or str): Plate ID used for labeling and filtering.
            - 'columnID' (str): Column ID to filter on (e.g., 'c3').
            - 'cmap' (str): Colormap to use for the heatmap.
            - 'dst' (str or None): If provided, save outputs to this directory.

    Returns:
        pd.DataFrame: Merged DataFrame containing all prediction and empirical score columns.

    Side Effects:
        - Displays a multi-channel heatmap of prediction scores.
        - Computes and prints MAE per well per channel.
        - Saves merged data, MAE scores, and the heatmap as PDF/CSV if `dst` is set.
    """
    def group_cv_score(csv, plate=1, column='c3', data_column='pred'):
        """
        Group cross-validation prediction scores by plate, row, and column.

        This function reads a CSV file containing predictions, filters by a specific column (e.g., 'c3'),
        assigns a plate ID, and computes the mean prediction score per well.

        Args:
            csv (str): Path to the CSV file containing predictions.
            plate (int or str, optional): Plate identifier to assign. Defaults to 1.
            column (str, optional): Column ID to filter by (e.g., 'c3'). Defaults to 'c3'.
            data_column (str, optional): Name of the column with prediction scores. Defaults to 'pred'.

        Returns:
            pd.DataFrame: Aggregated DataFrame with mean scores and a unique 'prc' well identifier.
                        Columns: ['plateID', 'rowID', 'columnID', data_column, 'prc']
        """
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

    def calculate_fraction_mixed_condition(csv, plate=1, column='c3', control_sgrnas = ['TGGT1_220950_1', 'TGGT1_233460_4']):
        """
        Calculate the fraction of control sgRNAs in each well for a mixed condition.

        This function filters a CSV containing sgRNA counts for a specified column and control sgRNAs,
        computes the total count per well, and calculates the fraction of each sgRNA within that well.
        A unique identifier ('prc') is added for each well.

        Args:
            csv (str): Path to the input CSV file containing sgRNA data.
            plate (int or str, optional): Plate identifier to assign if not present. Defaults to 1.
            column (str, optional): Column name to filter by (e.g., 'c3'). Defaults to 'c3'.
            control_sgrnas (list of str, optional): List of two control sgRNA names to include in the calculation.
                                                    Defaults to ['TGGT1_220950_1', 'TGGT1_233460_4'].

        Returns:
            pd.DataFrame: DataFrame with fraction per sgRNA in each well and a unique 'prc' well identifier.
                        Includes columns: ['plateID', 'rowID', 'column_name', 'grna_name', 'count',
                                            'total_count', 'fraction', 'prc']
        """
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
        """
        Plot a heatmap with multiple channels as columns.

        Parameters:
        - df: DataFrame with scores for different channels.
        - column: Column to filter by (default is 'c3').
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
        """
        Combine classification scores from multiple CSV files across folders.

        This function traverses given directories, locates specified CSV files within subfolders,
        filters by a target column, aggregates prediction scores by well position, and merges
        the results into a single DataFrame with each file contributing a new column of scores.

        Args:
            folders (str or list of str): Path(s) to parent folders containing subfolders with CSV files.
            csv_name (str): Name of the CSV file to locate in each subfolder.
            data_column (str): Name of the column in each CSV containing prediction scores to aggregate.
            plate (int or str, optional): Identifier for the plate to add as 'plateID'. Defaults to 1.
            column (str, optional): Column name to filter data by (e.g., 'c3'). Defaults to 'c3'.

        Returns:
            pd.DataFrame: Combined DataFrame with mean prediction scores per well from all found CSV files.
                        Includes a 'prc' column for well-level identifiers and one score column per source.
        """
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
        """
        Calculate mean absolute error (MAE) between prediction scores and reference fractions.

        This function computes the MAE between each numeric prediction column and the
        reference 'fraction' column on a per-row basis. The results are returned as a 
        long-form DataFrame with one row per (channel, row) pair.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'fraction' column as ground truth,
                            a 'prc' column as row identifier, and one or more numeric 
                            prediction columns.

        Returns:
            pd.DataFrame: DataFrame with columns ['Channel', 'MAE', 'Row'], reporting the 
                        MAE between each prediction channel and the fraction per row.
        """
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
    """
    Perform post-regression analysis to assess gRNA correlation and infer relative effect sizes.

    This function loads a CSV file containing gRNA abundance data (with columns including 
    'grna', 'fraction', and 'prc'), filters it to the specified gRNAs of interest, computes 
    a correlation matrix, visualizes it, and estimates inferred effect sizes for all gRNAs 
    based on a subset of known fixed effect sizes.

    Args:
        csv_file (str): Path to the CSV file containing gRNA data with columns ['grna', 'fraction', 'prc'].
        grna_dict (dict): Dictionary mapping selected gRNAs to their known effect sizes.
        grna_list (list): List of gRNAs to include in correlation and effect size inference.
        save (bool, optional): Whether to save outputs including figures and CSVs. Defaults to False.

    Returns:
        None. Saves plots and results if `save=True`.
    """
    def _analyze_and_visualize_grna_correlation(df, grna_list, save_folder, save=False):
        """
        Analyze and visualize the correlation matrix of gRNAs based on their fractions and overlap.

        Parameters:
        df (pd.DataFrame): DataFrame with columns ['grna', 'fraction', 'prc'].
        grna_list (list): List of gRNAs to include in the correlation analysis.
        save_folder (str): Path to the folder where figures and data will be saved.

        Returns:
        pd.DataFrame: Correlation matrix of the gRNAs.
        """
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
        """
        Compute and visualize the effect sizes of gRNAs given fixed effect sizes for a subset of gRNAs.

        Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix of gRNAs.
        grna_dict (dict): Dictionary of gRNAs with fixed effect sizes {grna_name: effect_size}.
        save_folder (str): Path to the folder where figures and data will be saved.

        Returns:
        pd.Series: Effect sizes of all gRNAs.
        """
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
