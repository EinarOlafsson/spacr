"""PyTorch dataset generation, classification, inference, and attribution."""

import os, torch, time, gc, datetime, logging
torch.backends.cudnn.benchmark = True
import numpy as np
import pandas as pd
from torch.optim import Adagrad, AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from multiprocessing import cpu_count
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix, f1_score
    

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Fail-loud accounting: a cross-validation fold that dies must not be
# averaged away silently, and an optional plot that fails must still be
# visible somewhere other than /dev/null.
from .errors import RunLedger
from .plot import save_figure  # every kept figure goes through the format/DPI preference
# One seed reaching Python, NumPy and Torch (CPU + CUDA) rather than only
# the split helpers. See spacr.runctx.
from .runctx import resolve_seed, seed_everything, seed_worker, torch_generator
from .torch_artifacts import (
    load_model_artifact,
    restore_training_state,
    save_model_artifact,
)


def _empty_device_cache() -> None:
    """Release accelerator caches without touching unavailable backends."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_inference_model(model_path, device):
    """Load current or legacy model artifacts and disable training-only recomputation."""
    model, metadata = load_model_artifact(model_path, map_location=device)
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
    return model.to(device).eval(), metadata


def _probability_columns(logits):
    """Convert binary or multiclass logits into scalar scores and columns."""
    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
        score = torch.sigmoid(logits.reshape(-1))
        return score, (score >= 0.5).long(), {}
    probabilities = torch.softmax(logits, dim=1)
    if probabilities.size(1) == 2:
        return probabilities[:, 1], probabilities.argmax(dim=1), {}
    score, predicted = probabilities.max(dim=1)
    columns = {
        f"prob_class_{index}": probabilities[:, index]
        for index in range(probabilities.size(1))
    }
    return score, predicted, columns


def _unpack_supervised_batch(batch):
    """Accept loaders yielding ``(images, labels)`` or an additional metadata item."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            "A supervised data loader must yield at least (images, labels).")
    return batch[0], batch[1]

def apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, n_jobs=10):
    """
    Apply a trained PyTorch model to images in a directory.

    The function loads a saved model, builds a dataset from the input images,
    runs batched inference, and saves prediction scores to a CSV file.

    :param src: Path to the input image directory or collection of image paths.
    :type src: str or sequence
    :param model_path: Path to the saved PyTorch model.
    :type model_path: str
    :param image_size: Final square crop size used before inference.
    :type image_size: int
    :param batch_size: Number of images processed per batch.
    :type batch_size: int
    :param normalize: Whether to normalize the image channels using mean 0.5
        and standard deviation 0.5.
    :type normalize: bool
    :param n_jobs: Number of worker processes used by the DataLoader.
    :type n_jobs: int
    :return: DataFrame with image paths and prediction scores.
    :rtype: pandas.DataFrame

    The returned DataFrame always contains the columns ``path`` and ``pred``.
    A single-logit head is converted with ``torch.sigmoid``, so ``pred`` is the
    positive-class probability; a multi-logit head is converted with
    ``torch.softmax``, so ``pred`` is the probability of class 1 for two
    classes and the winning class's confidence for more. With more than two
    classes the frame also carries ``predicted_label`` and one
    ``prob_class_<i>`` column per class. Results are also written to a CSV file
    derived from ``model_path`` and the current date.
    """
    from .io import NoClassDataset
    from .utils import print_progress
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size))])
    
    model, _ = _load_inference_model(model_path, device)

    print(model)
    
    print(f'Loading dataset in {src} with {len(src)} images')
    dataset = NoClassDataset(data_dir=src, transform=transform, shuffle=False,
                             load_to_memory=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=n_jobs,
                             pin_memory=(device.type == "cuda"))
    print(f'Loaded {len(src)} images')
    
    result_loc = os.path.splitext(model_path)[0]+datetime.date.today().strftime('%y%m%d')+'_'+os.path.splitext(model_path)[1]+'_test_result.csv'
    print(f'Results wil be saved in: {result_loc}')
    
    prediction_pos_probs = []
    predicted_labels = []
    probability_columns = {}
    filenames_list = []
    time_ls = []
    with torch.inference_mode():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            start = time.time()
            images = batch_images.to(device=device, dtype=torch.float,
                                     non_blocking=(device.type == "cuda"))
            outputs = model(images)
            scores, labels, extra = _probability_columns(outputs)
            prediction_pos_probs.extend(scores.cpu().tolist())
            predicted_labels.extend(labels.cpu().tolist())
            for name, values in extra.items():
                probability_columns.setdefault(name, []).extend(
                    values.cpu().tolist())
            filenames_list.extend(filenames)
            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed = min(batch_idx * batch_size, len(dataset))
            files_to_process = len(dataset)
            print_progress(files_processed, files_to_process, n_jobs=n_jobs, time_ls=time_ls, batch_size=batch_size, operation_type="Generating predictions")

    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    if probability_columns:
        data['predicted_label'] = predicted_labels
        data.update(probability_columns)
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    _empty_device_cache()
    return df

def apply_model_to_tar(settings=None):
    """
    Apply a trained PyTorch model to images stored in a tar archive.

    The function loads a saved model, reads images from a tar-based dataset,
    performs batched inference, post-processes prediction scores, and saves the
    results to a CSV file.

    :param settings: Dictionary of inference settings. Expected keys include
        ``tar_path``, ``model_path``, ``image_size``, ``batch_size``,
        ``normalize``, ``n_jobs``, ``verbose``, and ``score_threshold``.
    :type settings: dict
    :return: DataFrame with processed prediction results.
    :rtype: pandas.DataFrame

    The returned DataFrame contains at least the columns ``path`` and ``pred``,
    plus the columns added by ``process_vision_results``. A single-logit head is
    converted with ``torch.sigmoid``; a multi-logit head with ``torch.softmax``,
    giving the probability of class 1 for two classes and the winning class's
    confidence for more. With more than two classes the frame also carries
    ``predicted_label`` and one ``prob_class_<i>`` column per class, and its
    ``cv_predictions`` column holds the predicted class index rather than a
    threshold on ``pred``.
    """
    if settings is None:
        settings = {}
    from .io import TarImageDataset
    from .utils import process_vision_results, print_progress

    tar_path = settings['tar_path']
    model_path = settings['model_path']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if settings['normalize']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(settings['image_size'], settings['image_size'])),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(settings['image_size'], settings['image_size'])),
        ])

    if settings['verbose']:
        print(f"Loading model from {model_path}")
        print(f"Loading dataset from {tar_path}")

    model, _ = _load_inference_model(settings['model_path'], device)

    dataset = TarImageDataset(tar_path, transform=transform)
    # A tar built from on-demand crops carries the crop-format marker, so say
    # which channel ordering the model is about to be shown. The pixels are
    # NOT re-ordered here: a model's weights are tied to the order it was
    # trained on, and quietly correcting a legacy archive at inference time
    # would invalidate every model trained before spaCR grew the marker.
    if getattr(dataset, 'crop_format', None) is not None:
        from .crops import CROP_FORMAT_RGB
        order = 'rgb' if dataset.crop_format == CROP_FORMAT_RGB else 'bgr (legacy)'
        print(f"Tar crop format {dataset.crop_format} ({order}); images are "
              f"scored in the order they are stored.")
    elif settings.get('verbose'):
        print("Tar carries no crop-format marker, so its channel order is "
              "whatever wrote it (crops written before spacr 341f446 are "
              "BGR). Rebuild it with spacr.io.generate_dataset for one.")
    data_loader = DataLoader(
        dataset,
        batch_size=settings['batch_size'],
        shuffle=False,
        num_workers=settings['n_jobs'],
        pin_memory=(device.type == 'cuda'),
    )

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    dataset_name = os.path.splitext(os.path.basename(settings['tar_path']))[0]
    date_name = datetime.date.today().strftime('%y%m%d')
    dst = os.path.dirname(tar_path)
    result_loc = f'{dst}/{date_name}_{dataset_name}_{model_name}_result.csv'

    if settings['verbose']:
        print(model)
        print(f'Generated dataset with {len(dataset)} images')
        print(f'Generating loader from {len(data_loader)} batches')
        print(f'Results wil be saved in: {result_loc}')
        print(f'Model is in eval mode')
        print(f'Model loaded to device')

    prediction_pos_probs = []
    predicted_labels = []
    probability_columns = {}
    filenames_list = []
    time_ls = []
    gc.collect()
    with torch.inference_mode():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            start = time.time()
            images = batch_images.to(device=device, dtype=torch.float,
                                     non_blocking=(device.type == "cuda"))
            outputs = model(images)
            scores, labels, extra = _probability_columns(outputs)
            prediction_pos_probs.extend(scores.cpu().tolist())
            predicted_labels.extend(labels.cpu().tolist())
            for name, values in extra.items():
                probability_columns.setdefault(name, []).extend(
                    values.cpu().tolist())
            filenames_list.extend(filenames)

            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed = batch_idx * settings['batch_size']
            files_to_process = len(data_loader) * settings['batch_size']
            print_progress(files_processed, files_to_process, n_jobs=settings['n_jobs'],
                           time_ls=time_ls, batch_size=settings['batch_size'], operation_type="Tar dataset")

    data = {'path': filenames_list, 'pred': prediction_pos_probs}
    if probability_columns:
        data['predicted_label'] = predicted_labels
        data.update(probability_columns)
    df = pd.DataFrame(data, index=None)
    df = process_vision_results(df, settings['score_threshold'])
    if probability_columns:
        # Multiclass predictions are class indices, not a binary threshold on
        # the winning class's confidence.
        df['cv_predictions'] = df['predicted_label'].astype(int)

    df.to_csv(result_loc, index=True, header=True, mode='w')
    print(f"Saved results to {result_loc}")
    _empty_device_cache()
    return df

def _to_numpy_labels(target: torch.Tensor) -> np.ndarray:
    """
    Convert targets to integer class ids:
    - if 1D float/bool tensor -> round and cast to int
    - if shape (N, C) one-hot -> argmax
    - else assume already (N,) int
    """
    t = target.detach().cpu()
    if t.ndim == 2 and t.size(1) > 1:
        return t.argmax(dim=1).numpy().astype(int)
    if t.dtype.is_floating_point:
        return t.round().numpy().astype(int)
    return t.numpy().astype(int)


def _binary_metrics(y_true: np.ndarray, pos_probs: np.ndarray) -> dict:
    """Metrics for binary classification."""
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    # Precision-Recall AUC
    if len(np.unique(y_true)) >= 2:
        precision, recall, thresholds = precision_recall_curve(y_true, pos_probs, pos_label=1)
        pr_auc = auc(recall, precision)
        # F1-optimal threshold (optional; we still report 0.5 preds below)
        thresholds = np.append(thresholds, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
        opt_idx = np.nanargmax(f1)
        opt_thr = float(thresholds[opt_idx])
    else:
        pr_auc = np.nan
        opt_thr = 0.5

    # Discrete preds at 0.5 threshold for stability/readability
    pred = (pos_probs >= 0.5).astype(int)

    # Accuracies
    acc = (pred == y_true).mean() if len(y_true) else np.nan
    neg_mask = y_true == 0
    pos_mask = y_true == 1
    acc_neg = (pred[neg_mask] == 0).mean() if neg_mask.any() else np.nan
    acc_pos = (pred[pos_mask] == 1).mean() if pos_mask.any() else np.nan

    return {
        "accuracy": float(acc),
        "neg_accuracy": float(acc_neg),
        "pos_accuracy": float(acc_pos),
        "prauc": float(pr_auc),
        "optimal_threshold": float(opt_thr),
        # train_model has always PRINTED f1_macro, but neither metric helper
        # returned it, so it read nan on every line and never reached the CSV.
        "f1_macro": (float(f1_score(y_true, pred, average='macro',
                                    zero_division=0))
                     if len(y_true) else float(np.nan)),
        # Binary reported its two class accuracies under names nothing else
        # understood, so every consumer that wanted "the per-class numbers"
        # had to branch on the head shape. Report the same two values under
        # the SAME key the multiclass path uses, so the live view, the
        # TensorBoard scalars and the model card are one code path.
        # neg/pos stay for backwards compatibility.
        "per_class_accuracy": [
            0.0 if not np.isfinite(acc_neg) else float(acc_neg),
            0.0 if not np.isfinite(acc_pos) else float(acc_pos),
        ],
        "class_support": [int(neg_mask.sum()), int(pos_mask.sum())],
        "num_classes": 2,
    }

def _multiclass_metrics(y_true: np.ndarray, prob_mat: np.ndarray) -> dict:
    """
    Metrics for multiclass (single-label):
    - overall accuracy
    - per-class accuracy (weighted by support)
    - macro average precision (one-vs-rest)
    """
    C = prob_mat.shape[1]
    if len(y_true) == 0:
        # scikit-learn 1.7 rejects empty arrays in confusion_matrix. An empty
        # validation split is still a valid evaluator result: its metrics are
        # undefined, its class schema is known, and no fabricated sample
        # should be introduced merely to make a dependency accept the call.
        return {
            "accuracy": float(np.nan),
            "neg_accuracy": float(np.nan),
            "pos_accuracy": float(np.nan),
            "prauc": float(np.nan),
            "optimal_threshold": float(np.nan),
            "f1_macro": float(np.nan),
            "per_class_accuracy": [0.0] * int(C),
            "class_support": [0] * int(C),
            "num_classes": int(C),
        }

    preds = prob_mat.argmax(axis=1)
    acc = (preds == y_true).mean() if len(y_true) else np.nan

    # Per-class (diagonal / row sum)
    cm = confusion_matrix(y_true, preds, labels=np.arange(prob_mat.shape[1]))
    # The old `cm.sum(axis=1, where=(rowsums != 0), initial=1)` looked like a
    # divide-by-zero guard but was neither: `initial` seeds np.add.reduce, so it
    # added 1 to *every* row sum (a perfect classifier scored diag/(rowsum+1)),
    # and the (C,) mask broadcasts over the LAST axis of the (C, C) matrix, so it
    # dropped columns instead of rows. Guard the row sums explicitly; classes with
    # no true support report 0.0.
    row_sums = cm.sum(axis=1)
    per_class_acc = np.where(row_sums > 0, np.diag(cm) / np.maximum(row_sums, 1), 0.0)
    # Average precision macro (one-vs-rest)
    # Build one-hot y_true
    y_true_oh = np.zeros((len(y_true), C), dtype=int)
    if len(y_true):
        y_true_oh[np.arange(len(y_true)), y_true] = 1
    try:
        ap_macro = average_precision_score(y_true_oh, prob_mat, average="macro")
    except Exception as e:
        # NaN is written straight into the metrics CSV, where it is
        # indistinguishable from "not computed". Say why, at least once.
        logging.getLogger('spacr.deep_spacr').error(
            'macro average-precision could not be computed (%s: %s); '
            'prauc will be NaN for this evaluation',
            type(e).__name__, e)
        ap_macro = np.nan

    # For compatibility with your logging keys:
    return {
        "accuracy": float(acc),
        "neg_accuracy": np.nan,  # not meaningful in multiclass
        "pos_accuracy": np.nan,  # not meaningful in multiclass
        "prauc": float(ap_macro),  # reuse key for macro-AP
        "optimal_threshold": np.nan,
        # Macro F1 is the metric that actually matters on an imbalanced screen:
        # accuracy is dominated by the majority class, and this weights every
        # class equally. It was printed but never computed.
        "f1_macro": (float(f1_score(y_true, preds, average='macro',
                                    zero_division=0))
                     if len(y_true) else float(np.nan)),
        "per_class_accuracy": per_class_acc.tolist(),
        # Support belongs beside the accuracy it was computed from: a class
        # at 0.40 over 500 objects is a broken classifier, the same 0.40 over
        # 5 objects is two mistakes. Without it, the per-class line invites
        # exactly the wrong reading.
        "class_support": [int(v) for v in row_sums],
        "num_classes": int(C),
    }

#: Prefix of the flat per-class accuracy columns written into ``train.csv`` /
#: ``validation.csv`` and into the TensorBoard scalar names. Flat because a
#: list in a DataFrame cell reaches the CSV as the string ``"[0.99, 0.4]"``,
#: which nothing can plot and nobody can grep.
PER_CLASS_ACC_PREFIX = 'acc_class_'


def class_labels(metrics, classes=None):
    """Names for the classes ``metrics`` describes, one per class.

    :param metrics: a dict from :func:`_binary_metrics` /
        :func:`_multiclass_metrics`.
    :param classes: the folder names training read the classes from, in
        head order, when they are known. Omitted, the names
        :func:`attach_per_class_columns` stamped into ``metrics`` are used —
        which is what lets the live plot and the model card name the classes
        without every helper having to be handed the list again.
    :returns: list of ``str``, length ``num_classes``. Falls back to
        ``class_0, class_1, …`` — never to an empty list, because the caller
        is about to index it per class.
    """
    per_class = list(metrics.get('per_class_accuracy') or [])
    count = int(metrics.get('num_classes') or len(per_class) or 0)
    names = [str(c) for c in (classes or metrics.get('class_names') or [])]
    if len(names) == count and count:
        return names
    return [f'class_{i}' for i in range(count)]


def per_class_accuracy(metrics, classes=None):
    """``[(name, accuracy, support), …]`` for one epoch's metrics.

    The single place that knows a binary head reports two classes and a
    multiclass head reports C, so nothing downstream branches on head shape.

    :param metrics: one epoch's metrics dict.
    :param classes: optional class names in head order.
    :returns: list of ``(name, float accuracy, int support)``; empty when the
        metrics carry no per-class breakdown at all.
    """
    accs = list(metrics.get('per_class_accuracy') or [])
    if not accs:
        return []
    names = class_labels(metrics, classes)
    support = list(metrics.get('class_support') or [])
    out = []
    for i, acc in enumerate(accs):
        name = names[i] if i < len(names) else f'class_{i}'
        n = int(support[i]) if i < len(support) else 0
        out.append((name, float(acc), n))
    return out


def attach_per_class_columns(metrics, classes=None):
    """Add flat ``acc_class_<name>`` keys to ``metrics``, in place.

    Called on every epoch dict before it reaches ``train.csv`` /
    ``validation.csv``. The key set is fixed by the head size, which does not
    change inside a run, so the appended CSV keeps one stable header — see
    :func:`spacr.io._save_progress`, which writes the header only for the
    first chunk.

    :param metrics: one epoch's metrics dict; mutated and returned.
    :param classes: optional class names in head order.
    """
    rows = per_class_accuracy(metrics, classes)
    if rows:
        # Stamped so the history is self-describing: everything downstream
        # (the live plot, the model card) can name the classes from one
        # epoch dict rather than needing the list threaded through it.
        metrics['class_names'] = [name for name, _, _ in rows]
    for name, acc, support in rows:
        metrics[f'{PER_CLASS_ACC_PREFIX}{name}'] = float(acc)
        metrics[f'n_{name}'] = int(support)
    return metrics


def format_per_class_accuracy(metrics, classes=None, prefix=''):
    """One line naming every class and how it actually did.

    A 96 % aggregate hiding a class at 40 % is the commonest way a
    classifier looks finished and is not, and the aggregate is the only
    number the epoch line used to print. The worst class is flagged
    explicitly so it does not have to be spotted in a row of numbers.

    :param metrics: one epoch's metrics dict.
    :param classes: optional class names in head order.
    :param prefix: text put in front of the line ("Train ", "Val ").
    :returns: the line, or ``''`` when there is no per-class breakdown.
    """
    rows = per_class_accuracy(metrics, classes)
    if not rows:
        return ''
    parts = [f"{name} {acc:.3f} (n={n})" for name, acc, n in rows]
    line = f"{prefix}per-class acc.: " + ', '.join(parts)
    finite = [(name, acc) for name, acc, _ in rows if np.isfinite(acc)]
    if len(finite) > 1:
        worst_name, worst_acc = min(finite, key=lambda t: t[1])
        best_acc = max(a for _, a in finite)
        if best_acc - worst_acc >= 0.10:
            line += (f"  <- WORST: {worst_name} at {worst_acc:.3f}, "
                     f"{best_acc - worst_acc:.3f} below the best class")
    return line


def evaluate_model_performance(model, loader, epoch, loss_type='auto',
                               loss_fn=None, num_classes=None):
    """Evaluate a binary or multiclass classifier and return metrics plus raw probs/labels.

    Head size is inferred from the first batch — a single-logit head is
    treated as binary (BCE + sigmoid), otherwise softmax + CE metrics
    apply. If ``loss_fn`` is None, one is constructed via ``build_loss``.

    :param model: PyTorch classifier.
    :param loader: DataLoader yielding ``(input, target, meta)`` batches.
    :param epoch: Current epoch (recorded in the returned dict).
    :param loss_type: Loss selection passed to ``build_loss`` when
        ``loss_fn`` is None. Default ``'auto'``.
    :param loss_fn: Optional callable ``(logits, target) -> Tensor``.
    :param num_classes: Class count when the loader is empty.
    :returns: ``(metrics_dict, [probs, labels])`` — metrics include
        ``loss``, ``epoch`` and ``Accuracy``. ``probs`` is shape
        ``(N,)`` for binary or ``(N, C)`` for multiclass.
    """
    from .utils import build_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    total_loss, total_samples = 0.0, 0
    all_labels = []
    prob_bucket = []
    head_dim = None  # infer from first batch
    binary_mode = None

    with torch.no_grad():
        for data, target, _ in loader:
            data = data.to(device)
            logits = model(data)

            # infer head size/mode once
            if head_dim is None:
                head_dim = logits.size(1) if (logits.ndim == 2) else 1
                binary_mode = (head_dim == 1)

            # ----- target normalization for loss/metrics -----
            if binary_mode:
                # BCE-style targets: float {0,1}, allow (N,) or (N,1)
                target = target.to(device).float()
                y_true_batch = (target.view(-1) > 0.5).long().detach().cpu().numpy()
            else:
                # CE-style: class indices (N,)
                if target.ndim == 2:
                    # handle one-hot inputs robustly
                    target = target.argmax(dim=1)
                target = target.to(device).long()
                y_true_batch = target.view(-1).detach().cpu().numpy()

            # ----- choose loss (prefer training's loss_fn if provided) -----
            local_loss_fn = loss_fn
            if local_loss_fn is None:
                # fallback: construct something reasonable matching the head
                local_loss_fn = build_loss(loss_type or 'auto',
                                           num_classes=head_dim,
                                           class_counts=None,
                                           label_smoothing=0.0,
                                           focal_gamma=2.0,
                                           focal_alpha=None,
                                           logit_adjust_tau=0.0)

            loss = local_loss_fn(logits, target)

            batch_size = data.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_labels.extend(y_true_batch.tolist())

            # ----- probabilities for metrics -----
            if binary_mode:
                probs = torch.sigmoid(logits.view(-1))
                prob_bucket.append(probs.detach().cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
                prob_bucket.append(probs.detach().cpu().numpy())

    # aggregate
    mean_loss = total_loss / max(1, total_samples)
    y_true = np.asarray(all_labels, dtype=int)

    if len(prob_bucket) == 0:
        # empty loader: synthesize empty array with correct rank
        if (num_classes or head_dim or 1) == 1:
            probs_np = np.empty((0,))
        else:
            c = num_classes if num_classes is not None else (head_dim if head_dim is not None else 2)
            probs_np = np.empty((0, c))
    else:
        probs_np = np.concatenate(prob_bucket, axis=0)

    # metrics (assumes _binary_metrics / _multiclass_metrics exist)
    if probs_np.ndim == 1:
        metrics = _binary_metrics(y_true, probs_np)
    else:
        metrics = _multiclass_metrics(y_true, probs_np)

    metrics["loss"] = float(mean_loss)
    metrics["epoch"] = int(epoch)
    metrics["Accuracy"] = metrics["accuracy"]
    return metrics, [probs_np, y_true.tolist()]

def test_model_core(model, loader, loader_name, epoch, loss_type):
    """
    Core test loop over ``loader``, compatible with binary & multiclass.

    :returns: the 4-tuple ``(metrics, probs, labels, results_df)``. ``metrics``
        is the summary dict, with ``loss``, ``epoch`` and ``Accuracy`` added.
        ``probs`` is shape ``(N,)`` for a single-logit head and ``(N, C)``
        otherwise. ``labels`` is the list of true class ids. ``results_df``
        holds one row per image with ``filename``, ``true_label``,
        ``predicted_label`` and either ``class_1_probability`` (single-logit
        head) or one ``prob_class_<k>`` column per class.
    """
    from .utils import calculate_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    probs_rows = []
    filenames = []

    with torch.no_grad():
        for data, target, batch_filenames in loader:
            data = data.to(device)
            target = target.to(device)

            logits = model(data)
            batch_size = data.size(0)
            loss = calculate_loss(logits, target, prefer_focal=True)
            #loss = calculate_loss(logits, target, loss_type=loss_type)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            # labels & filenames
            y_true = _to_numpy_labels(target)
            all_labels.extend(y_true)
            filenames.extend(list(batch_filenames))

            # probs
            if logits.ndim == 1 or logits.size(-1) == 1:
                probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
                probs_rows.append(probs.reshape(-1, 1))  # keep 2D for uniform handling
            else:
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs_rows.append(probs)

    mean_loss = total_loss / max(1, total_samples)
    y_true = np.asarray(all_labels, dtype=int)
    prob_mat = np.vstack(probs_rows) if probs_rows else np.empty((0, 1))
    C = prob_mat.shape[1]

    # metrics
    if C == 1:
        metrics = _binary_metrics(y_true, prob_mat.ravel())
    else:
        metrics = _multiclass_metrics(y_true, prob_mat)
    metrics["loss"] = float(mean_loss)
    metrics["epoch"] = int(epoch)
    metrics["Accuracy"] = metrics["accuracy"]

    # Build per-file results dataframe
    df_dict = {
        "filename": filenames,
        "true_label": y_true.tolist(),
        "predicted_label": prob_mat.argmax(1).tolist() if C > 1 else (prob_mat.ravel() >= 0.5).astype(int).tolist(),
    }
    if C == 1:
        df_dict["class_1_probability"] = prob_mat.ravel().tolist()
    else:
        # add one column per class probs: prob_class_0, prob_class_1, ...
        for k in range(C):
            df_dict[f"prob_class_{k}"] = prob_mat[:, k].tolist()

    results_df = pd.DataFrame(df_dict)

    return metrics, (prob_mat if C > 1 else prob_mat.ravel()), y_true.tolist(), results_df

def test_model_performance(loaders, model, loader_name_list, epoch, loss_type):
    """
    Evaluate ``model`` on a single loader and report the metrics as a frame.

    Thin wrapper around :func:`test_model_core`, kept for API compatibility.

    :returns: ``(summary_metrics_dataframe, per_file_results_dataframe)`` — the
        first is the one-row frame of summary metrics, the second holds one row
        per image.
    """
    data_dict, _, _, results_df = test_model_core(
        model=model,
        loader=loaders,
        loader_name=loader_name_list,
        epoch=epoch,
        loss_type=loss_type,
    )

    # The old function returned a DataFrame in 'result'; emulate that:
    result_df = pd.DataFrame([data_dict])
    return result_df, results_df

#: Scalar metrics worth aggregating across folds. ``Accuracy`` is a duplicate
#: of ``accuracy`` and ``epoch``/``num_classes`` are bookkeeping, so neither
#: belongs in a spread statistic.
CV_METRIC_KEYS = ('accuracy', 'loss', 'prauc', 'neg_accuracy', 'pos_accuracy')


def resolve_class_balance_loss(loss_type, class_balance, num_classes):
    """Translate ``class_balance='weighted_loss'`` into a concrete loss type.

    The weighted losses already exist in :func:`spacr.utils.build_loss`; this
    only steers to them, so nothing about the loss maths changes here.

    :param loss_type: the loss the user asked for.
    :param class_balance: one of ``spacr.io.CLASS_BALANCE_MODES``.
    :param num_classes: size of the classifier head.
    :returns: ``(loss_type, message)`` — ``message`` is '' when nothing changed.
    """
    #: Losses that already correct for class frequency on their own.
    reweighting = ('ce_weighted', 'logit_adjust_ce', 'la_ce')

    if class_balance in ('weighted_sampler', 'sqrt_weighted_sampler'):
        if loss_type in reweighting:
            # Both corrections multiply: the rare class ends up over-weighted
            # and the model swings to over-predicting it.
            return loss_type, (
                f"WARNING: class_balance={class_balance!r} resamples the train "
                f"loader while loss_type={loss_type!r} also reweights by class "
                f"frequency - the two corrections compound. Pick one: either "
                f"class_balance='none' with this loss, or a frequency-neutral "
                f"loss such as 'cross_entropy' with this sampler.")
        return loss_type, ''
    if class_balance != 'weighted_loss':
        return loss_type, ''
    if int(num_classes) < 2:
        return loss_type, (
            "class_balance='weighted_loss' needs a 2+ class head; a single-logit "
            f"head keeps loss_type={loss_type!r}. Use focal_alpha to weight the "
            "positive class instead.")
    if loss_type == 'ce_weighted':
        return loss_type, ("class_balance='weighted_loss': loss_type is already "
                           "'ce_weighted' (inverse-frequency class weights)")
    return 'ce_weighted', (
        f"class_balance='weighted_loss': loss_type {loss_type!r} -> 'ce_weighted' "
        f"(inverse-frequency class weights from the train-split counts)")


def summarize_cv_metrics(fold_df, metric_keys=None):
    """Reduce per-fold metrics to mean plus the spread around it.

    The spread is the whole point of k-fold: a single split can be lucky, and
    only the fold-to-fold standard deviation and range say by how much.

    :param fold_df: DataFrame with one row per fold.
    :param metric_keys: metric columns to summarise. Defaults to
        ``CV_METRIC_KEYS`` intersected with the columns present.
    :returns: DataFrame indexed by metric with ``n_folds``, ``mean``, ``std``,
        ``min``, ``max``, ``range`` and ``cv_percent`` columns.
    """
    if metric_keys is None:
        metric_keys = [k for k in CV_METRIC_KEYS if k in fold_df.columns]
    rows = []
    for key in metric_keys:
        vals = pd.to_numeric(fold_df[key], errors='coerce').dropna()
        if vals.empty:
            continue
        mean = float(vals.mean())
        # ddof=1: folds are a sample of the possible splits, not the population.
        std = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
        rows.append({
            'metric': key,
            'n_folds': int(len(vals)),
            'mean': mean,
            'std': std,
            'min': float(vals.min()),
            'max': float(vals.max()),
            'range': float(vals.max() - vals.min()),
            'cv_percent': (abs(std / mean) * 100.0) if (mean and std == std) else float('nan'),
        })
    return pd.DataFrame(rows)


def _print_cv_report(fold_df, summary_df, k):
    """Print the per-fold table and the fold-to-fold spread."""
    print(f"\n=== Cross-validation results ({k} folds) ===")
    print(fold_df.to_string(index=False))
    print("\n--- Fold-to-fold spread ---")
    if summary_df.empty:
        print("  no numeric metrics were produced by any fold")
        return
    print(summary_df.to_string(index=False))
    acc = summary_df[summary_df['metric'] == 'accuracy']
    if not acc.empty:
        row = acc.iloc[0]
        print(f"\n  accuracy across folds: {row['mean']:.4f} +/- {row['std']:.4f} "
              f"(sd), range {row['min']:.4f}-{row['max']:.4f}")
        print("  A single train/val split reports one number from this range; "
              "the spread is how lucky that number could have been.")


def _cross_validate_model(settings, num_classes):
    """Run k-fold cross-validation and report per-fold metrics plus their spread.

    Each fold trains a fresh model on its own train split and is scored on the
    held-out fold, which is never resampled. Folds are group-aware by default
    (see ``cv_group_by``), so crops from one well cannot appear on both sides.

    :param settings: the canonicalised train/test settings dict.
    :param num_classes: size of the classifier head.
    :returns: path to the written per-fold CSV, or None if no fold trained.
    """
    from sklearn.metrics import log_loss

    from .classifier_evaluation import (
        audit_cv_folds,
        audit_split_leakage,
        evaluate_predictions,
        nested_group_folds,
        normalize_probabilities,
        write_evaluation_bundle,
    )
    from .io import (
        dataset_filenames,
        dataset_labels,
        generate_cv_loaders,
        make_class_balance_sampler,
    )
    from .utils import augment_dataset

    src = settings['src']
    dst = settings['dst']
    k = int(settings.get('cross_validation_folds', 0) or 0)
    if settings.get('resume_checkpoint'):
        raise ValueError(
            "resume_checkpoint cannot be shared across cross-validation folds; "
            "resume an individual fold directly or start a fresh k-fold run.")

    fold_loaders, info = generate_cv_loaders(
        src,
        n_splits=k,
        mode='train',
        image_size=settings['image_size'],
        batch_size=settings['batch_size'],
        classes=settings['classes'],
        n_jobs=settings['n_jobs'],
        pin_memory=settings['pin_memory'],
        normalize=settings['normalize'],
        channels=settings['train_channels'],
        augment=settings['augment'],
        verbose=settings['verbose'],
        group_by=settings.get('cv_group_by', 'well'),
        class_balance=settings.get('class_balance', 'none'),
        seed=settings.get('random_seed', 42),
    )
    cv_partition_audit = audit_cv_folds(
        dataset_filenames(info['dataset']),
        info['folds'],
        labels=info['labels'],
        group_by=settings.get('cv_group_by', 'well'),
        hash_content=settings.get('leakage_hash_content', True),
        require_identity=settings.get('leakage_require_identity', True),
        raise_on_leakage=settings.get('evaluation_fail_on_leakage', True),
    )
    if not cv_partition_audit.passed:
        print(
            "WARNING: full CV partition leakage audit failed: "
            f"{cv_partition_audit.critical_levels}"
        )

    nested_inner = int(settings.get('nested_cv_inner_folds', 0) or 0)
    if nested_inner < 0 or nested_inner == 1:
        raise ValueError(
            f"nested_cv_inner_folds={nested_inner} is not valid; use 0 for "
            "ordinary grouped CV or at least 2 inner folds.")
    nested_layout = None
    if nested_inner >= 2:
        nested_layout = nested_group_folds(
            info['labels'],
            outer_splits=k,
            inner_splits=nested_inner,
            groups=info.get('groups'),
            seed=settings.get('random_seed', 42),
        )
        print(
            f"Nested CV enabled: {k} untouched outer folds x "
            f"{nested_inner} inner training folds."
        )

    def _fit_one(train_loader, validation_loader, destination):
        """Train one fold model with a validation set not used for final scoring."""
        return train_model(
            src=src,
            dst=destination,
            model_type=settings['model_type'],
            train_loaders=train_loader,
            epochs=settings['epochs'],
            learning_rate=settings['learning_rate'],
            init_weights=settings['init_weights'],
            weight_decay=settings['weight_decay'],
            amsgrad=settings['amsgrad'],
            optimizer_type=settings['optimizer_type'],
            use_checkpoint=settings['use_checkpoint'],
            dropout_rate=settings['dropout_rate'],
            n_jobs=settings['n_jobs'],
            val_loaders=validation_loader,
            test_loaders=None,
            intermedeate_save=settings['intermedeate_save'],
            schedule=settings['schedule'],
            loss_type=settings['loss_type'],
            label_smoothing=settings.get('label_smoothing', 0.1),
            focal_gamma=settings.get('focal_gamma', 2.0),
            focal_alpha=settings.get('focal_alpha'),
            logit_adjust_tau=settings.get('logit_adjust_tau', 1.0),
            gradient_accumulation=settings['gradient_accumulation'],
            gradient_accumulation_steps=settings[
                'gradient_accumulation_steps'],
            channels=settings['train_channels'],
            num_classes=num_classes,
            image_size=settings.get('image_size', 224),
            plot=settings.get('plot', False),
            tensorboard=settings.get('tensorboard', True),
            early_stopping_patience=settings.get(
                'early_stopping_patience', 0),
            custom_model_path=settings.get('custom_model_path') or None,
            preprocessing={
                'image_size': settings.get('image_size', 224),
                'normalize': settings.get('normalize', True),
                'channels': settings.get('train_channels'),
                'augment': settings.get('augment', False),
            },
            classes=list(settings.get('classes') or []),
        )

    def _metrics_for_probabilities(labels, probabilities):
        """Compute legacy fold metrics for an ensemble probability matrix."""
        labels = np.asarray(labels, dtype=int)
        normalized = normalize_probabilities(
            probabilities, n_classes=num_classes,
        )
        if num_classes == 2:
            metrics = _binary_metrics(labels, normalized[:, 1])
        else:
            metrics = _multiclass_metrics(labels, normalized)
        metrics['loss'] = float(log_loss(
            labels, normalized, labels=np.arange(num_classes),
        ))
        return metrics

    def _inner_loader(indices, *, training):
        """Build one inner loader from global indexes into the base dataset."""
        dataset = Subset(info['dataset'], list(indices))
        if training and settings.get('augment'):
            dataset = augment_dataset(
                dataset,
                is_grayscale=(len(settings.get('train_channels') or []) == 1),
            )
        sampler = None
        if training:
            sampler, _ = make_class_balance_sampler(
                dataset_labels(dataset),
                settings.get('class_balance', 'none'),
            )
        workers = max(0, int(settings.get('n_jobs', 0) or 0))
        # A shuffled loader with no generator draws its permutation from
        # torch's global RNG, and a worker inherits (fork) or loses (spawn)
        # the parent's stream -- so the inner folds were never reproducible
        # even with random_seed set. See spacr.runctx.seed_worker.
        return DataLoader(
            dataset,
            batch_size=settings['batch_size'],
            shuffle=bool(training and sampler is None),
            sampler=sampler,
            num_workers=workers,
            pin_memory=settings['pin_memory'],
            persistent_workers=(workers > 0),
            generator=torch_generator(stream='inner_cv'),
            worker_init_fn=seed_worker if workers > 0 else None,
        )

    rows = []
    oof_probabilities = []
    oof_labels = []
    oof_paths = []
    oof_folds = []
    leakage_reports = [cv_partition_audit]
    # A fold that does not train is dropped from the spread. Two dead folds
    # out of five used to produce a "5-fold CV" summary computed on three.
    ledger = RunLedger('cross_validation')
    for i, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
        fold_dst = os.path.join(dst, f'fold_{i}')
        os.makedirs(fold_dst, exist_ok=True)
        print(f"\n--- Fold {i}/{k} ---")
        train_paths = dataset_filenames(train_loader.dataset)
        validation_paths = dataset_filenames(val_loader.dataset)
        outer_leakage = audit_split_leakage(
            train_paths,
            validation_paths,
            group_by=settings.get('cv_group_by', 'well'),
            raise_on_leakage=settings.get(
                'evaluation_fail_on_leakage', True),
            split_name=f'outer_{i}',
            hash_content=False,
            require_identity=settings.get('leakage_require_identity', True),
        )
        leakage_reports.append(outer_leakage)
        if not outer_leakage.passed:
            print(
                f"WARNING: outer fold {i} leakage: "
                f"{outer_leakage.critical_levels}"
            )

        fold_model_path = None
        if nested_layout is None:
            model, fold_model_path = _fit_one(
                train_loader, val_loader, fold_dst,
            )
            if model is None:
                ledger.record_failure(
                    f'fold_{i}', stage='train',
                    exc=(
                        f"model_type {settings['model_type']!r} "
                        "could not be built"
                    ),
                )
                print(
                    f"Fold {i}: model_type "
                    f"{settings['model_type']!r} could not be built; "
                    "fold skipped."
                )
                continue
            metrics, payload = evaluate_model_performance(
                model,
                val_loader,
                epoch=1,
                loss_type='ce' if num_classes >= 2 else 'bce',
                num_classes=num_classes,
            )
            fold_probabilities = np.asarray(payload[0], dtype=float)
            fold_labels = np.asarray(payload[1], dtype=int)
        else:
            outer = nested_layout[i - 1]
            member_probabilities = []
            fold_labels = None
            for inner_index, (
                inner_train_indices,
                inner_validation_indices,
            ) in enumerate(outer['inner'], start=1):
                inner_train = _inner_loader(
                    inner_train_indices, training=True,
                )
                inner_validation = _inner_loader(
                    inner_validation_indices, training=False,
                )
                inner_leakage = audit_split_leakage(
                    dataset_filenames(inner_train.dataset),
                    dataset_filenames(inner_validation.dataset),
                    group_by=settings.get('cv_group_by', 'well'),
                    raise_on_leakage=settings.get(
                        'evaluation_fail_on_leakage', True),
                    split_name=f'outer_{i}_inner_{inner_index}',
                    hash_content=False,
                    require_identity=settings.get(
                        'leakage_require_identity', True),
                )
                leakage_reports.append(inner_leakage)
                inner_dst = os.path.join(
                    fold_dst, f'inner_{inner_index}',
                )
                os.makedirs(inner_dst, exist_ok=True)
                print(
                    f"  Inner fold {inner_index}/{nested_inner}: "
                    f"train={len(inner_train.dataset)}, "
                    f"validation={len(inner_validation.dataset)}"
                )
                model, _inner_model_path = _fit_one(
                    inner_train, inner_validation, inner_dst,
                )
                if model is None:
                    ledger.record_failure(
                        f'fold_{i}_inner_{inner_index}',
                        stage='train',
                        exc="inner model could not be built",
                    )
                    continue
                _inner_metrics, payload = evaluate_model_performance(
                    model,
                    val_loader,
                    epoch=1,
                    loss_type='ce' if num_classes >= 2 else 'bce',
                    num_classes=num_classes,
                )
                current_probabilities = np.asarray(
                    payload[0], dtype=float,
                )
                current_labels = np.asarray(payload[1], dtype=int)
                if fold_labels is None:
                    fold_labels = current_labels
                elif not np.array_equal(fold_labels, current_labels):
                    raise RuntimeError(
                        f"Outer fold {i} labels changed between inner "
                        "ensemble members."
                    )
                member_probabilities.append(current_probabilities)
            if not member_probabilities:
                ledger.record_failure(
                    f'fold_{i}', stage='train',
                    exc="every inner model failed",
                )
                print(f"Fold {i}: every inner model failed; fold skipped.")
                continue
            fold_probabilities = np.mean(
                np.stack(member_probabilities, axis=0),
                axis=0,
            )
            metrics = _metrics_for_probabilities(
                fold_labels, fold_probabilities,
            )

        if len(validation_paths) != len(fold_labels):
            raise RuntimeError(
                f"Fold {i} produced {len(fold_labels)} labels for "
                f"{len(validation_paths)} validation paths."
            )
        row = {'fold': i,
               'n_train': len(train_loader.dataset),
               'n_val': len(val_loader.dataset)}
        if fold_model_path:
            row['model_path'] = str(fold_model_path)
        for key in CV_METRIC_KEYS:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
        oof_probabilities.append(fold_probabilities)
        oof_labels.extend(fold_labels.tolist())
        oof_paths.extend(validation_paths)
        oof_folds.extend([i] * len(fold_labels))
        ledger.record_success(f'fold_{i}', stage='train')

    if not rows:
        ledger.finalize()
        print("Cross-validation produced no fold results.")
        return None

    fold_df = pd.DataFrame(rows)
    summary_df = summarize_cv_metrics(fold_df)
    _print_cv_report(fold_df, summary_df, k)

    time_now = datetime.date.today().strftime('%y%m%d')
    stem = f"{settings['model_type']}_time_{time_now}_cv{k}"
    folds_loc = os.path.join(dst, f"{stem}_per_fold.csv")
    summary_loc = os.path.join(dst, f"{stem}_spread.csv")
    split_loc = os.path.join(dst, f"{stem}_fold_composition.csv")
    fold_df.to_csv(folds_loc, index=False)
    summary_df.to_csv(summary_loc, index=False)
    info['fold_table'].to_csv(split_loc, index=False)
    if settings.get('classifier_evaluation', True):
        probabilities = np.concatenate(oof_probabilities, axis=0)
        calibration_method = settings.get(
            'evaluation_calibration', 'temperature',
        )
        if len(set(oof_folds)) < 2 and calibration_method == 'temperature':
            print(
                "Warning: fewer than two successful outer folds remain; "
                "temperature calibration is disabled."
            )
            calibration_method = 'none'
        evaluation = evaluate_predictions(
            oof_labels,
            probabilities,
            oof_paths,
            classes=settings.get('classes'),
            fold_ids=oof_folds,
            calibration_method=calibration_method,
            calibration_bins=settings.get('evaluation_bins', 10),
        )
        evaluation_manifest = write_evaluation_bundle(
            os.path.join(dst, 'evaluation'),
            evaluation,
            leakage_reports=leakage_reports,
        )
        settings['classifier_evaluation_path'] = str(evaluation_manifest)
        print(f"Classifier evaluation: {evaluation_manifest}")
    model_rows = fold_df[
        fold_df.get('model_path', pd.Series(index=fold_df.index,
                                            dtype=object)).notna()
    ]
    if not model_rows.empty:
        if 'accuracy' in model_rows:
            best_row = model_rows.loc[
                pd.to_numeric(model_rows['accuracy'],
                              errors='coerce').idxmax()]
        elif 'loss' in model_rows:
            best_row = model_rows.loc[
                pd.to_numeric(model_rows['loss'], errors='coerce').idxmin()]
        else:
            best_row = model_rows.iloc[0]
        settings['cv_best_model_path'] = str(best_row['model_path'])
        print(f"Best fold model:   {settings['cv_best_model_path']}")
    settings['cv_results_path'] = folds_loc
    print(f"\nPer-fold metrics: {folds_loc}")
    print(f"Fold spread:      {summary_loc}")
    print(f"Fold composition: {split_loc}")
    # The per-fold CSV is stamped so a reader can see it covers fewer folds
    # than requested, and a run in which most folds died aborts outright:
    # the "spread" of two surviving folds out of five is not a spread.
    ledger.finalize(artifact=folds_loc, threshold=0.5)
    return folds_loc


def train_test_model(settings):
    """Train a vision classifier on a spacr training dataset and/or evaluate it on the held-out ``test/`` split.

    Given a dataset folder ``src`` laid out as ``train/<class>/*.png`` and
    ``test/<class>/*.png`` (as produced by
    :func:`spacr.io.generate_dataset`), this function optionally trains a
    torchvision-style model (``model_type``), then optionally scores the
    test split and copies misclassified images into a review folder.
    Best-checkpoint selection is automatic when ``train=False`` and
    ``test=True``.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.get_train_test_model_settings`. Key entries:

        - ``src`` — dataset root containing ``train/`` and ``test/``.
        - ``model_type`` — e.g. ``'maxvit_t'``, ``'resnet50'``.
        - ``classes`` — list of class names, must match subfolder names.
        - ``epochs``, ``batch_size``, ``learning_rate``, ``weight_decay``.
        - ``image_size``, ``train_channels`` (e.g. ``['r','g','b']``),
          ``normalize`` (``[low, high]`` percentiles).
        - ``val_split`` — fraction pulled from ``train/`` for validation.
        - ``loss_type`` — ``'auto'``, ``'cross_entropy'``,
          ``'binary_cross_entropy_with_logits'``.
        - ``train`` / ``test`` — flip the two halves of the pipeline.
        - ``cross_validation_enabled`` / ``cross_validation_folds`` — train
          with k-fold cross-validation over ``train/`` instead of a single
          validation split (enabling it with fewer than 2 folds uses 5, and
          1 fold falls back to the single split); ``cv_group_by`` names the
          grouping level used for the folds and the leakage audits.
        - ``augment``, ``dropout_rate``, ``optimizer_type``,
          ``early_stopping_patience``, ``n_jobs``, ``pin_memory``.

    :returns: When ``train=True``, the path to the saved best model — or, in
        cross-validation mode, the path to the per-fold metrics CSV, since
        there is no single model; ``None`` if ``model_type`` could not be
        built. When ``train=False`` and ``test=True``, the path to the
        test-result CSV.
    :raises ValueError: if ``settings['classes']`` is missing or empty.

    Example:
        .. code-block:: python

            from spacr.deep_spacr import train_test_model
            settings = {
                'src': '/data/dataset_v1',
                'model_type': 'maxvit_t', 'classes': ['neg', 'pos'],
                'epochs': 25, 'batch_size': 32, 'learning_rate': 1e-4,
                'image_size': 224, 'train_channels': ['r','g','b'],
                'train': True, 'test': True,
            }
            model_path = train_test_model(settings)

    See Also:
        :func:`spacr.deep_spacr.deep_spacr` — full end-to-end training +
        activation-map pipeline.
        :func:`spacr.io.generate_dataset` — build the ``train/``/``test/``
        folder tree.
    """
    from .io import _copy_missclassified
    from .utils import pick_best_model, save_settings
    from .io import generate_loaders, CLASS_BALANCE_MODES
    from .settings import get_train_test_model_settings

    settings = get_train_test_model_settings(settings)

    # random_seed used to reach the split helpers below and nothing else:
    # torch's own initialisation -- weight init, dropout, the shuffle inside
    # every DataLoader -- was never seeded at all, so two "identical" runs
    # trained two different models. One call fixes Python, NumPy and Torch
    # (CPU and CUDA); what it still cannot promise is in
    # spacr.runctx.SeedReport.caveats, and cudnn.benchmark (set True at the
    # top of this module) is one of the things deterministic=True undoes.
    seed_everything(resolve_seed(settings),
                    deterministic=bool(settings.get('deterministic', False)))

    _empty_device_cache()
    gc.collect()

    src = settings['src']

    channels_str = ''.join(settings['train_channels'])
    dst = os.path.join(src, 'model', settings['model_type'], channels_str,
                       str(f"epochs_{settings['epochs']}"))
    os.makedirs(dst, exist_ok=True)
    settings['dst'] = dst

    num_classes = len(settings.get('classes', [])) if settings.get('classes') else 0
    if num_classes <= 0:
        raise ValueError("No classes provided in settings['classes'].")

    # Audit the permanent dataset boundary before a model sees a pixel. This
    # catches renamed byte-identical copies as well as plate/well/object and
    # exported-augmentation relationships.
    if settings.get('leakage_audit_train_test', True):
        from .classifier_evaluation import (
            audit_dataset_splits, write_leakage_audit,
        )
        train_dir = os.path.join(src, 'train')
        test_dir = os.path.join(src, 'test')
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            dataset_audit = audit_dataset_splits(
                src,
                group_by=settings.get('cv_group_by', 'well'),
                hash_content=settings.get('leakage_hash_content', True),
                require_identity=settings.get(
                    'leakage_require_identity', True),
                raise_on_leakage=settings.get(
                    'evaluation_fail_on_leakage', True),
            )
            audit_path = write_leakage_audit(
                os.path.join(dst, 'train_test_leakage_audit.json'),
                dataset_audit,
            )
            settings['train_test_leakage_audit_path'] = str(audit_path)
            print(
                f"Train/test leakage audit: {'PASS' if dataset_audit.passed else 'FAIL'} "
                f"({audit_path})"
            )

    if settings.get('loss_type') in (None, 'auto'):
        settings['loss_type'] = 'cross_entropy' if num_classes > 1 else 'binary_cross_entropy_with_logits'

    # Class-imbalance steering: 'weighted_loss' is expressed as a loss_type, the
    # sampler modes as a DataLoader sampler inside generate_loaders. Either way
    # the change is announced before the settings snapshot is written, so the
    # saved settings record what actually ran.
    class_balance = settings.get('class_balance', 'none')
    if class_balance not in CLASS_BALANCE_MODES:
        raise ValueError(
            f"class_balance {class_balance!r} is not one of {CLASS_BALANCE_MODES}")
    settings['loss_type'], balance_msg = resolve_class_balance_loss(
        settings['loss_type'], class_balance, num_classes)
    if balance_msg:
        print(balance_msg)

    cv_folds = int(settings.get('cross_validation_folds', 0) or 0)
    if settings.get('cross_validation_enabled') and cv_folds < 2:
        cv_folds = 5
        settings['cross_validation_folds'] = cv_folds
    if cv_folds == 1:
        print("cross_validation_folds=1 is not a cross-validation; falling back "
              "to the single train/validation split (val_split="
              f"{settings.get('val_split')}).")
        cv_folds = 0

    # This ladder used to sit inside an outer `if settings['train']:`, which made
    # the test-only arm unreachable (a test-only run snapshotted nothing), and the
    # `is True` comparisons also skipped the snapshot for truthy-but-not-True flags
    # such as train=1 coming from a scripted caller.
    if settings['train'] and settings['test']:
        save_settings(settings, name=f"train_test_{settings['model_type']}_{settings['epochs']}", show=True)
    elif settings['train']:
        save_settings(settings, name=f"train_{settings['model_type']}_{settings['epochs']}", show=True)
    elif settings['test']:
        save_settings(settings, name=f"test_{settings['model_type']}_{settings['epochs']}", show=True)

    # save_settings writes to <src>/settings/<name>.csv, and the name is keyed
    # on model_type and epochs alone -- so a second run of the same shape with
    # a different learning rate silently OVERWRITES the first run's snapshot,
    # and the first run's curves become unattributable. A copy inside dst is
    # per-run by construction, since dst already varies with the run.
    # spacr.train_compare.load_run prefers this one.
    try:
        pd.DataFrame(list(settings.items()), columns=['Key', 'Value']).to_csv(
            os.path.join(dst, 'settings.csv'), index=False)
    except Exception as e:
        print(f"Could not write the per-run settings snapshot to {dst}: {e}")

    model = None
    model_path = None
    cv_result_loc = None

    if settings['train'] and cv_folds >= 2:
        # k-fold replaces the single split entirely: every crop is validated
        # once, and the reported number is a mean with its fold-to-fold spread
        # rather than one draw from it.
        cv_result_loc = _cross_validate_model(settings, num_classes)

    elif settings['train']:
        train, val, _ = generate_loaders(
            src,
            mode='train',
            image_size=settings['image_size'],
            batch_size=settings['batch_size'],
            classes=settings['classes'],
            n_jobs=settings['n_jobs'],
            validation_split=settings['val_split'],
            pin_memory=settings['pin_memory'],
            normalize=settings['normalize'],
            channels=settings['train_channels'],
            augment=settings['augment'],
            verbose=settings['verbose'],
            class_balance=class_balance,
            seed=settings.get('random_seed', 42),
            group_by=settings.get('cv_group_by', 'well'),
        )

        if hasattr(train, 'dataset') and hasattr(val, 'dataset'):
            from .classifier_evaluation import (
                audit_split_leakage, write_leakage_audit,
            )
            from .io import dataset_filenames
            validation_audit = audit_split_leakage(
                dataset_filenames(train.dataset),
                dataset_filenames(val.dataset),
                group_by=settings.get('cv_group_by', 'well'),
                raise_on_leakage=settings.get(
                    'evaluation_fail_on_leakage', True),
                split_name='train_vs_validation',
                hash_content=settings.get('leakage_hash_content', True),
                require_identity=settings.get(
                    'leakage_require_identity', True),
            )
            validation_audit_path = write_leakage_audit(
                os.path.join(dst, 'train_validation_leakage_audit.json'),
                validation_audit,
            )
            settings['train_validation_leakage_audit_path'] = str(
                validation_audit_path
            )
            print(
                f"Train/validation leakage audit: "
                f"{'PASS' if validation_audit.passed else 'FAIL'} "
                f"({validation_audit_path})"
            )

        model, model_path = train_model(
            src=src,
            dst=settings['dst'],
            model_type=settings['model_type'],
            train_loaders=train,
            epochs=settings['epochs'],
            learning_rate=settings['learning_rate'],
            init_weights=settings['init_weights'],
            weight_decay=settings['weight_decay'],
            amsgrad=settings['amsgrad'],
            optimizer_type=settings['optimizer_type'],
            use_checkpoint=settings['use_checkpoint'],
            dropout_rate=settings['dropout_rate'],
            n_jobs=settings['n_jobs'],
            val_loaders=val,
            test_loaders=None,
            intermedeate_save=settings['intermedeate_save'],
            schedule=settings['schedule'],
            loss_type=settings['loss_type'],
            label_smoothing=settings.get('label_smoothing', 0.1),
            focal_gamma=settings.get('focal_gamma', 2.0),
            focal_alpha=settings.get('focal_alpha'),
            logit_adjust_tau=settings.get('logit_adjust_tau', 1.0),
            gradient_accumulation=settings['gradient_accumulation'],
            gradient_accumulation_steps=settings['gradient_accumulation_steps'],
            channels=settings['train_channels'],
            num_classes=num_classes,
            image_size=settings.get('image_size', 224),
            plot=settings.get('plot', False),
            tensorboard=settings.get('tensorboard', True),
            early_stopping_patience=settings.get('early_stopping_patience', 0),
            custom_model_path=settings.get('custom_model_path') or None,
            resume_checkpoint=settings.get('resume_checkpoint') or None,
            preprocessing={
                'image_size': settings.get('image_size', 224),
                'normalize': settings.get('normalize', True),
                'channels': settings.get('train_channels'),
                'augment': settings.get('augment', False),
            },
            classes=list(settings.get('classes') or []),
            settings=settings,
            split_rule=(
                f"{settings['val_split']:.0%} of train/ held out for "
                f"validation by generate_loaders, grouped by "
                f"{settings.get('cv_group_by', 'well')}; test/ is a separate "
                f"folder tree audited for leakage before training"
                if settings.get('val_split') else
                f"held out by generate_loaders, grouped by "
                f"{settings.get('cv_group_by', 'well')}"),
        )

        if model is None:
            # choose_model could not build model_type (e.g. a typo in settings).
            # Abort here rather than falling through into the test branch, where
            # pick_best_model would look for a checkpoint that was never written.
            print(f"Training aborted: model_type {settings['model_type']!r} could not be built.")
            return None

    if settings['test']:
        test, _, _ = generate_loaders(
            src,
            mode='test',
            image_size=settings['image_size'],
            batch_size=settings['batch_size'],
            classes=settings['classes'],
            n_jobs=settings['n_jobs'],
            validation_split=0.0,
            pin_memory=settings['pin_memory'],
            normalize=settings['normalize'],
            channels=settings['train_channels'],
            augment=False,
            verbose=settings['verbose']
        )

        if model_path and os.path.isfile(model_path):
            # Test the checkpoint selected by validation, not the final
            # in-memory epoch (which may already have overfit).
            print(f'Loading selected checkpoint for testing: {model_path}')
            model, _ = _load_inference_model(model_path, torch.device('cpu'))
        elif model is None:
            model_path = pick_best_model(src + '/model')
            print(f'Best model: {model_path}')
            model, _ = _load_inference_model(model_path, torch.device('cpu'))

        model_fldr = dst
        time_now = datetime.date.today().strftime('%y%m%d')
        result_loc = f"{model_fldr}/{settings['model_type']}_time_{time_now}_test_result.csv"
        acc_loc = f"{model_fldr}/{settings['model_type']}_time_{time_now}_test_acc.csv"
        print(f'Results will be saved in: {result_loc}')

        result, accuracy = test_model_performance(loaders=test,
                                                  model=model,
                                                  loader_name_list='test',
                                                  epoch=1,
                                                  loss_type=settings['loss_type'])

        result.to_csv(result_loc, index=True, header=True, mode='w')
        accuracy.to_csv(acc_loc, index=True, header=True, mode='w')
        _copy_missclassified(accuracy)

    _empty_device_cache()
    gc.collect()

    if settings['train']:
        # In k-fold mode there is no single "the model"; the per-fold metric
        # CSV is the artefact worth handing back.
        return cv_result_loc if cv_folds >= 2 else model_path
    if settings['test']:
        return result_loc

#: Colours the per-class accuracy panel cycles through. Deliberately not the
#: train/val blue and red used by the two aggregate panels, so a class line is
#: never mistaken for a split.
#: Curve colours: teal, blue, purple, grey first, as asked for, then a tail
#: for runs with more classes than that. Chosen to read on BOTH a light and a
#: dark background, because the figure itself is transparent now and spaCR
#: does not know which one is behind it -- a palette tuned for dark alone
#: disappears on the light theme.
_CLASS_CURVE_COLORS = ('#2aa198', '#4A9EFF', '#9b7fd4', '#8a8f98',
                       '#2ec27e', '#c061cb', '#e5a50a', '#62a0ea',
                       '#ed333b', '#ff7800')

#: The two series every training run has. Teal and blue, from the same list.
_TRAIN_CURVE_COLOR = _CLASS_CURVE_COLORS[1]     # blue
_VAL_CURVE_COLOR = _CLASS_CURVE_COLORS[0]       # teal


def _per_class_series(history, classes=None):
    """``(epochs, {class name: [accuracy per epoch]})`` out of an epoch history.

    Epochs whose metrics carry no per-class breakdown contribute ``nan``
    rather than being dropped, so the x-axis of the per-class panel stays
    aligned with the two panels beside it.

    :param history: list of per-epoch metrics dicts.
    :param classes: optional class names in head order.
    """
    names = []
    for entry in history:
        rows = per_class_accuracy(entry, classes)
        if len(rows) > len(names):
            names = [name for name, _, _ in rows]
    if not names:
        return [], {}
    epochs = [d.get('epoch', i + 1) for i, d in enumerate(history)]
    series = {name: [] for name in names}
    for entry in history:
        by_name = {name: acc for name, acc, _ in per_class_accuracy(entry, classes)}
        for name in names:
            series[name].append(float(by_name.get(name, float('nan'))))
    return epochs, series


def _plot_training_curves(train_hist, val_hist, total_epochs=None, figure=None,
                          classes=None):
    """Render or refresh live loss + accuracy + per-class curves for the run.

    Three panels, not two. The aggregate accuracy panel answers "is it
    learning"; the per-class panel answers "is it learning *all of it*",
    which is the question a 96 % aggregate hiding a class at 40 % gets wrong
    — and that failure is invisible for the whole run if the only live number
    is the mean. Per-class lines come from the validation history when there
    is one, because train accuracy on a minority class is the number least
    worth trusting.

    ``figure`` lets the GUI update one zoomable monitor in place instead of
    adding an epoch snapshot to the figure gallery every time.  ``plt.show``
    is captured by the Qt bridge and re-renders figures marked as live.

    The show is ``block=False``, and that is load-bearing rather than a
    style choice. This runs *inside* the epoch loop, so on any interpreter
    whose matplotlib backend is interactive — which is every machine with
    PySide6 installed, i.e. every spaCR install, because matplotlib then
    picks 'qtagg' — a blocking ``plt.show()`` enters the Qt main loop and
    never comes back. Training stops dead at the end of epoch 1 with no
    error and no output; measured on the classify demo, which hung for as
    long as it was left running with the whole stack parked in
    ``backend_qt.start_main_loop``. Inside the Qt GUI the bridge's
    ``_capture_show(*args, **kwargs)`` replaces ``plt.show`` entirely and
    ignores the keyword, so the GUI path is unchanged.
    """
    import matplotlib.pyplot as plt
    if not train_hist:
        return None
    tr_ep = [d.get('epoch', i + 1) for i, d in enumerate(train_hist)]
    tr_loss = [d.get('loss', float('nan')) for d in train_hist]
    tr_acc = [d.get('accuracy', float('nan')) for d in train_hist]

    # Prefer held-out per-class accuracy; fall back to train so a run without
    # a validation split still gets the panel rather than a blank third.
    class_hist = val_hist if val_hist else train_hist
    class_split = 'val' if val_hist else 'train'
    cls_ep, cls_series = _per_class_series(class_hist, classes)

    if figure is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        fig._spacr_live_update = True
        # Transparent from the start, so the container shows through and the
        # page opacity reaches the plot. The GUI restyles text and spines for
        # the active theme when it renders (figure_queue._style_figure_colors);
        # what matters here is that no opaque page is baked in, because a
        # white or black rectangle cannot be undone by restyling.
        fig.patch.set_alpha(0.0)
    else:
        fig = figure
        fig.clear()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        fig.patch.set_alpha(0.0)
    ax1.plot(tr_ep, tr_loss, marker='o', ms=3, color=_TRAIN_CURVE_COLOR,
             label='train')
    ax2.plot(tr_ep, tr_acc, marker='o', ms=3, color=_TRAIN_CURVE_COLOR,
             label='train')
    if val_hist:
        v_ep = [d.get('epoch') for d in val_hist]
        ax1.plot(v_ep, [d.get('loss', float('nan')) for d in val_hist],
                 marker='s', ms=3, color=_VAL_CURVE_COLOR, label='val')
        ax2.plot(v_ep, [d.get('accuracy', float('nan')) for d in val_hist],
                 marker='s', ms=3, color=_VAL_CURVE_COLOR, label='val')
    for axis in (ax1, ax2, ax3):
        axis.patch.set_alpha(0.0)
    ax1.set_title('Loss'); ax1.set_xlabel('epoch'); ax1.legend(loc='best')
    ax2.set_title('Accuracy'); ax2.set_xlabel('epoch')
    ax2.set_ylim(0, 1.02); ax2.legend(loc='best')

    if cls_series:
        for i, (name, values) in enumerate(cls_series.items()):
            ax3.plot(cls_ep, values, marker='o', ms=3,
                     color=_CLASS_CURVE_COLORS[i % len(_CLASS_CURVE_COLORS)],
                     label=str(name))
        ax3.set_title(f'Per-class accuracy ({class_split})')
        ax3.set_ylim(0, 1.02)
        ax3.legend(loc='best', fontsize='small')
        latest = {name: values[-1] for name, values in cls_series.items()
                  if values and np.isfinite(values[-1])}
        if len(latest) > 1:
            worst = min(latest, key=latest.get)
            ax3.set_xlabel(f'epoch — worst: {worst} at {latest[worst]:.3f}')
        else:
            ax3.set_xlabel('epoch')
    else:
        ax3.set_title('Per-class accuracy')
        ax3.set_xlabel('epoch')
        ax3.text(0.5, 0.5, 'no per-class metrics', ha='center', va='center',
                 transform=ax3.transAxes, fontsize='small', color='#888888')

    last = tr_ep[-1] if tr_ep else 0
    suffix = f' / {total_epochs}' if total_epochs else ''
    fig.suptitle(f'Training — epoch {last}{suffix}')
    plt.tight_layout()
    plt.show(block=False)
    return fig


def _open_tensorboard_writer(dst, enabled=True):
    """Create a PyTorch TensorBoard writer for a training run.

    TensorBoard is a declared dependency, but keeping the import guarded makes
    old editable environments fail soft and tells the user how to repair them.
    """
    log_dir = os.path.abspath(os.path.join(dst, 'tensorboard'))
    if not enabled:
        return None, log_dir
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "TensorBoard logging is unavailable. Install the current package "
            "dependencies (or `pip install tensorboard`) to enable it. "
            f"Details: {exc}"
        )
        return None, log_dir

    writer = SummaryWriter(log_dir=log_dir, flush_secs=5)
    print(f"TensorBoard log: {log_dir}")
    print(f"Open it with: tensorboard --logdir {log_dir}")
    return writer, log_dir


def _log_tensorboard_epoch(writer, train_dict, val_dict, epoch, classes=None):
    """Write the scalar metrics produced by one epoch and flush immediately.

    Per-class accuracy goes in as one scalar per class under
    ``accuracy_<class>/<split>``, so the class that is not learning shows up
    as a flat line beside the rising aggregate rather than being averaged
    into it.
    """
    if writer is None:
        return
    groups = (('train', train_dict), ('validation', val_dict))
    for split, metrics in groups:
        if not metrics:
            continue
        for metric in ('loss', 'accuracy', 'f1_macro'):
            value = metrics.get(metric)
            if value is not None:
                writer.add_scalar(f'{metric}/{split}', float(value), epoch)
        for name, acc, _support in per_class_accuracy(metrics, classes):
            if np.isfinite(acc):
                writer.add_scalar(f'accuracy_{name}/{split}', float(acc), epoch)
    lr = train_dict.get('lr')
    if lr is not None:
        writer.add_scalar('learning_rate', float(lr), epoch)
    writer.flush()


# ---------------------------------------------------------------------------
# Model cards — what a checkpoint was trained on, and how well it did
# ---------------------------------------------------------------------------

#: Written beside ``<model>.pth`` as ``<model>.card.json``. A sidecar rather
#: than a key inside the checkpoint, on purpose: the card has to be readable
#: without ``torch.load``, which is how a reviewer, a shell script and a
#: registry all get to it.
MODEL_CARD_SUFFIX = '.card.json'

#: The human-readable twin of the JSON, same stem.
MODEL_CARD_MD_SUFFIX = '.card.md'

#: The registry role a card is registered under. The weights themselves are
#: registered as :data:`spacr.ports.MODEL_WEIGHTS` by the run; the card is a
#: separate artifact so a checkpoint whose card is missing is visibly missing
#: rather than silently uncarded.
MODEL_CARD_ROLE = 'model-card'


def held_out_report(y_true, probs, classes=None):
    """Everything the card says about held-out performance, recomputable.

    The card must not be the only place a number exists — a card that
    reports 0.96 with nothing to check it against is a claim, not a record.
    So this returns the confusion matrix alongside every derived figure, and
    each figure is exactly the standard function of that matrix:
    ``accuracy = trace / total`` and
    ``per_class_accuracy[c] = M[c, c] / M[c, :].sum()``. Recomputing them
    from ``confusion_matrix`` must reproduce them exactly, and the test suite
    pins that.

    :param y_true: integer class ids, shape ``(N,)``.
    :param probs: ``(N,)`` positive-class probabilities for a single-logit
        head, or ``(N, C)`` softmax rows for a C-logit head. The same two
        shapes :func:`evaluate_model_performance` returns.
    :param classes: class names in head order, when known.
    :returns: dict with ``n``, ``num_classes``, ``classes``, ``accuracy``,
        ``f1_macro``, ``per_class_accuracy``, ``class_support``,
        ``predicted_support`` and ``confusion_matrix``.

    The implementation lives in :func:`spacr.active_learning.holdout_report`,
    which is torch-free, so a card written by a CNN round and a card written
    by an in-Annotate classical round carry the identical shape and are
    comparable side by side. Imported lazily: that module deliberately does
    not pull torch, and this one already has.
    """
    from .active_learning import holdout_report
    return holdout_report(y_true, probs, classes)


def dataset_class_balance(src, classes=None):
    """Count the images the classifier was trained on, per split, per class.

    Reads the ``train/<class>/`` and ``test/<class>/`` folder tree
    :func:`spacr.io.generate_dataset` writes, because that tree *is* the
    training set — a class balance quoted from the settings would describe
    what was requested rather than what ended up on disk.

    :param src: dataset root holding ``train/`` (and optionally ``test/``).
    :param classes: class names to count; discovered from the folder names
        when omitted.
    :returns: ``{split: {class: count}}`` for the splits that exist.
    """
    out = {}
    root = str(src or '')
    for split in ('train', 'test', 'val', 'validation'):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        names = [str(c) for c in classes] if classes else sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.'))
        counts = {}
        for name in names:
            class_dir = os.path.join(split_dir, name)
            if not os.path.isdir(class_dir):
                counts[name] = 0
                continue
            counts[name] = sum(
                1 for f in os.listdir(class_dir)
                if not f.startswith('.')
                and os.path.isfile(os.path.join(class_dir, f)))
        if counts:
            out[split] = counts
    return out


def _training_counts(balance):
    """``{class: n}`` for the split a card should judge balance on.

    ``balance`` arrives as ``{split: {class: n}}`` from
    :func:`dataset_class_balance`, but a caller with a single population
    (the in-Annotate rounds pass ``{'annotated': {...}}``) or a flat
    ``{class: n}`` is just as legitimate. Resolving the shape here means no
    caller has to know which one the note expects — the flat fallback used to
    be taken literally, so a dict of dicts reached ``int()`` and the whole
    card write failed, silently, with only a note to show for it.
    """
    if not isinstance(balance, dict) or not balance:
        return {}
    if isinstance(balance.get('train'), dict):
        return balance['train']
    nested = [v for v in balance.values() if isinstance(v, dict)]
    if nested:
        return nested[0] if len(nested) == 1 else {}
    return balance


def _imbalance_note(balance):
    """A sentence when one class dominates the training set, else ``''``."""
    counts = {}
    for key, value in (balance or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[key] = count
    total = sum(counts.values())
    if len(counts) < 2 or not total:
        return ''
    biggest = max(counts, key=counts.get)
    smallest = min(counts, key=counts.get)
    share = counts[biggest] / total
    if share < 0.7:
        return ''
    return (f"Class {biggest!r} is {share:.0%} of the training set and "
            f"{smallest!r} is {counts[smallest] / total:.0%}. A model that "
            f"always answered {biggest!r} would score {share:.0%} accuracy, "
            f"so read the per-class numbers, not the aggregate.")


def build_model_card(model_path, *, settings=None, classes=None,
                     split_rule='', held_out=None, train_metrics=None,
                     dataset_src=None, class_balance=None, module='train',
                     epochs=None, history=None, extra=None):
    """Assemble the record that travels with a checkpoint.

    Answers, for a ``.pth`` somebody finds in six months: what was it trained
    on, how was the held-out split drawn, how balanced were the classes, how
    did it do *per class*, which spaCR wrote it, under which settings, when.

    :param model_path: the checkpoint the card describes.
    :param settings: the run's settings; only the material ones are hashed
        and stored (:func:`spacr.artifacts.material_settings`).
    :param classes: class names in head order.
    :param split_rule: how the held-out set was drawn, in words — the field
        most often left implicit and most often the reason a number is wrong
        ("random 20 % of objects" leaks across wells; "grouped by well" does
        not).
    :param held_out: a :func:`held_out_report` dict.
    :param train_metrics: the training-split metrics of the same epoch.
    :param dataset_src: dataset root, for the class balance on disk.
    :param class_balance: override for the counted balance.
    :param module: producing module key for the registry.
    :param epochs: epochs requested, when known.
    :param history: per-epoch metrics, trimmed into the card as a curve.
    :param extra: anything else worth recording.
    :returns: a JSON-safe dict.
    """
    from .artifacts import material_settings, settings_hash
    from .version import get_version

    balance = class_balance if class_balance is not None else \
        dataset_class_balance(dataset_src, classes) if dataset_src else {}
    training_balance = _training_counts(balance)
    card = {
        'card_version': 1,
        'model_path': os.path.abspath(str(model_path)),
        'model_file': os.path.basename(str(model_path)),
        'module': str(module),
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'spacr_version': get_version(),
        'settings_hash': settings_hash(settings),
        'settings': material_settings(settings),
        'classes': [str(c) for c in (classes or [])],
        'epochs': (int(epochs) if epochs is not None else None),
        'training_set': {
            'src': (os.path.abspath(str(dataset_src)) if dataset_src else ''),
            'class_balance': balance,
            'n_train': sum(int(v) for v in training_balance.values()
                           if isinstance(v, (int, float))),
            'imbalance_note': _imbalance_note(training_balance),
        },
        'split_rule': str(split_rule or 'not recorded'),
        'held_out': dict(held_out or {}),
        'train_metrics': {
            k: v for k, v in (train_metrics or {}).items()
            if not isinstance(v, (dict, list, tuple))
            or k in ('per_class_accuracy', 'class_support')
        },
        'history': [
            {k: v for k, v in entry.items()
             if k in ('epoch', 'loss', 'accuracy', 'f1_macro',
                      'per_class_accuracy', 'class_support')}
            for entry in (history or [])
        ],
        'extra': dict(extra or {}),
    }
    if not card['split_rule'] or card['split_rule'] == 'not recorded':
        card.setdefault('warnings', []).append(
            "The split rule was not recorded, so nothing here says whether "
            "the held-out numbers are leakage-free.")
    if not card['held_out']:
        card.setdefault('warnings', []).append(
            "No held-out evaluation is attached: every number in this card "
            "describes data the model was fitted on.")
    return card


def format_model_card(card):
    """Render a card as Markdown — the version a human reads first."""
    lines = [f"# Model card — {card.get('model_file', '?')}", '']
    lines.append(f"* **spaCR version**: {card.get('spacr_version', '?')}")
    lines.append(f"* **Created (UTC)**: {card.get('created_utc', '?')}")
    lines.append(f"* **Settings hash**: `{card.get('settings_hash', '')}`")
    lines.append(f"* **Produced by**: `{card.get('module', '?')}`")
    classes = card.get('classes') or []
    if classes:
        lines.append(f"* **Classes** (head order): {', '.join(map(str, classes))}")
    lines.append('')

    training = card.get('training_set') or {}
    lines.append('## Training set')
    lines.append('')
    lines.append(f"Source: `{training.get('src') or 'not recorded'}`")
    lines.append('')
    balance = training.get('class_balance') or {}
    if balance:
        for split, counts in balance.items():
            total = sum(int(v) for v in counts.values()) or 1
            lines.append(f"**{split}** ({total} images)")
            lines.append('')
            lines.append('| class | n | share |')
            lines.append('| --- | ---: | ---: |')
            for name, n in counts.items():
                lines.append(f"| {name} | {int(n)} | {int(n) / total:.1%} |")
            lines.append('')
    if training.get('imbalance_note'):
        lines.append(f"> {training['imbalance_note']}")
        lines.append('')

    lines.append('## Split rule')
    lines.append('')
    lines.append(str(card.get('split_rule', 'not recorded')))
    lines.append('')

    held = card.get('held_out') or {}
    lines.append('## Held-out metrics')
    lines.append('')
    if not held:
        lines.append('_None recorded._')
        lines.append('')
    else:
        lines.append(f"n = {held.get('n', 0)} · accuracy "
                     f"{held.get('accuracy', float('nan')):.4f} · macro-F1 "
                     f"{held.get('f1_macro', float('nan')):.4f}")
        lines.append('')
        names = held.get('classes') or []
        lines.append('| class | accuracy | support |')
        lines.append('| --- | ---: | ---: |')
        for i, acc in enumerate(held.get('per_class_accuracy') or []):
            name = names[i] if i < len(names) else f'class_{i}'
            support = (held.get('class_support') or [0] * (i + 1))[i]
            lines.append(f"| {name} | {float(acc):.4f} | {int(support)} |")
        lines.append('')
        matrix = held.get('confusion_matrix') or []
        if matrix:
            lines.append('Confusion matrix (rows = true, columns = predicted):')
            lines.append('')
            header = ' | '.join(str(n) for n in names) or '?'
            lines.append(f"| true \\ pred | {header} |")
            lines.append('| --- |' + ' ---: |' * len(matrix[0]))
            for i, row in enumerate(matrix):
                name = names[i] if i < len(names) else f'class_{i}'
                lines.append(f"| {name} | " +
                             ' | '.join(str(int(v)) for v in row) + ' |')
            lines.append('')
            lines.append('Every figure above is a function of this matrix: '
                         '`accuracy = trace / total`, '
                         '`per-class[c] = M[c, c] / M[c, :].sum()`.')
            lines.append('')

    for warning in card.get('warnings') or []:
        lines.append(f"> **Warning** {warning}")
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def write_model_card(model_path, card, *, markdown=True):
    """Write ``card`` beside ``model_path``; return the JSON card's path.

    :param model_path: the checkpoint the card belongs to.
    :param card: a :func:`build_model_card` dict.
    :param markdown: also write the Markdown twin.
    :returns: absolute path of the ``.card.json``.
    """
    from .checkpoint import json_safe
    import json as _json

    stem = os.path.splitext(os.path.abspath(str(model_path)))[0]
    json_path = stem + MODEL_CARD_SUFFIX
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w') as handle:
        _json.dump(json_safe(card), handle, indent=2, sort_keys=True)
    if markdown:
        with open(stem + MODEL_CARD_MD_SUFFIX, 'w') as handle:
            handle.write(format_model_card(card))
    return json_path


def read_model_card(model_path):
    """The card beside ``model_path``, or ``None`` when there is not one."""
    import json as _json
    stem = os.path.splitext(os.path.abspath(str(model_path)))[0]
    json_path = stem + MODEL_CARD_SUFFIX
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as handle:
        return _json.load(handle)


def register_model_card(model_path, card, *, project=None, registry=None,
                        inputs=(), run_id=''):
    """Register the checkpoint and its card in the artifact registry.

    Registering the *weights* with the card as ``extra`` — rather than only
    dropping a JSON file next to them — is what makes the provenance real:
    the row carries the content fingerprint of the ``.pth``, the settings
    hash, the spaCR version and the ids of the artifacts it was derived from,
    so "is this model stale?" has an answer that a hand-written note cannot
    give.

    :param model_path: the checkpoint.
    :param card: a :func:`build_model_card` dict, stored as provenance.
    :param project: project root; defaults to the checkpoint's own tree.
    :param registry: an open :class:`spacr.artifacts.Registry`.
    :param inputs: upstream artifact ids or :class:`spacr.artifacts.Artifact`.
    :param run_id: the run this came out of.
    :returns: the stored :class:`spacr.artifacts.Artifact`, or ``None`` when
        the registry could not be written (a card on disk is still worth
        having, so this never raises the run down).
    """
    from . import artifacts as artifacts_module
    from .ports import MODEL_WEIGHTS

    root = project or os.path.dirname(os.path.abspath(str(model_path)))
    try:
        store = registry if registry is not None else \
            artifacts_module.open_registry(root)
        return store.register(
            module=str(card.get('module') or 'train'),
            kind=MODEL_WEIGHTS,
            role=MODEL_CARD_ROLE,
            path=model_path,
            project=root,
            settings=card.get('settings') or {},
            settings_digest=str(card.get('settings_hash') or ''),
            inputs=inputs,
            run_id=run_id,
            extra=card,
        )
    except Exception as exc:
        print(f"Model card written but not registered ({type(exc).__name__}: "
              f"{exc}). The card is still on disk beside the weights.")
        return None


def model_card(model_path, *, registry=None, project=None, inputs=(),
               run_id='', **card_kwargs):
    """Build, write and register a card for ``model_path`` in one call.

    :returns: ``(card, card_path, artifact_or_None)``.
    """
    card = build_model_card(model_path, **card_kwargs)
    card_path = write_model_card(model_path, card)
    artifact = register_model_card(model_path, card, project=project,
                                   registry=registry, inputs=inputs,
                                   run_id=run_id)
    if artifact is not None:
        card['artifact_id'] = artifact.artifact_id
        write_model_card(model_path, card)
    return card, card_path, artifact


def train_model(src,dst, model_type, train_loaders, epochs=100, learning_rate=0.0001,
                weight_decay=0.05, amsgrad=False, optimizer_type='adamw',
                use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None,
                test_loaders=None, init_weights='imagenet', intermedeate_save=None,
                chan_dict=None, schedule=None, loss_type='auto',
                gradient_accumulation=False, gradient_accumulation_steps=4,
                channels=None, verbose=False, num_classes=2,
                image_size=224, plot=False, tensorboard=True,
                # add early stopping parameters
                early_stopping_patience=0,  # 0 = disabled; e.g. 20 = stop after 20 epochs without val improvement
                custom_model_path=None, resume_checkpoint=None,
                preprocessing=None, classes=None,
                label_smoothing=0.1, focal_gamma=2.0, focal_alpha=None,
                logit_adjust_tau=1.0,
                settings=None, split_rule='', write_card=True,
                ):
    """
    Train a classifier and return it together with its checkpoint path.

    Supports 2-class and >2-class heads via CrossEntropy and a single-logit
    head via BCE; the loss itself is built by :func:`spacr.utils.build_loss`.

    :param src: Dataset root. When ``<src>/train`` exists, its subfolder names
        become the class list and override ``classes``.
    :param dst: Output folder for checkpoints, progress CSVs and TensorBoard
        logs.
    :param model_type: Architecture name passed to
        :func:`spacr.utils.choose_model`.
    :param train_loaders: DataLoader yielding ``(data, target, filenames)``.
    :param epochs: Final epoch number to train through. Default ``100``.
    :param learning_rate: Optimizer learning rate. Default ``0.0001``.
    :param weight_decay: Optimizer weight decay. Default ``0.05``.
    :param amsgrad: AMSGrad variant, honoured by ``adamw`` and ``adam`` only.
    :param optimizer_type: One of ``'adamw'``, ``'adam'``, ``'adamax'``,
        ``'adagrad'``, ``'adadelta'``, ``'asgd'``, ``'sgd'``, ``'rmsprop'``,
        ``'nadam'``, ``'radam'``. Default ``'adamw'``.
    :param use_checkpoint: Enable gradient checkpointing in the architecture.
    :param dropout_rate: Dropout rate passed to the architecture.
    :param n_jobs: Unused; accepted for call-site compatibility.
    :param val_loaders: Validation loader driving best-checkpoint selection and
        early stopping. Without one, training accuracy is used instead.
    :param test_loaders: Only its batch count is printed; it is not evaluated
        here.
    :param init_weights: Pretrained-weight selection, ignored (passed as
        ``False``) when ``custom_model_path`` or ``resume_checkpoint`` is set.
    :param intermedeate_save: Accuracy thresholds that trigger intermediate
        checkpoints, forwarded to :func:`spacr.io._save_model`.
    :param chan_dict: Unused; accepted for call-site compatibility.
    :param schedule: ``None``, ``'step_lr'``, ``'reduce_lr_on_plateau'``,
        ``'cosine'``, ``'cosine_warm_restarts'``, ``'exponential'`` or
        ``'linear'``.
    :param loss_type: Loss identifier passed to
        :func:`spacr.utils.build_loss`; ``'auto'`` picks one from the head.
    :param gradient_accumulation: Accumulate gradients over several batches.
    :param gradient_accumulation_steps: Batches per optimizer step when
        ``gradient_accumulation`` is on. Default ``4``.
    :param channels: Channel names recorded with the checkpoint. Default
        ``['r', 'g', 'b']``.
    :param verbose: Verbose architecture construction.
    :param num_classes: Size of the classifier head; ``1`` selects the binary
        BCE path. Default ``2``.
    :param image_size: Square input size in pixels. Default ``224``.
    :param plot: Refresh a live training-curve figure after every epoch.
    :param tensorboard: Write TensorBoard event files into ``dst``.
    :param early_stopping_patience: Number of epochs with no val improvement
        before stopping. Set to 0 to disable (original behavior).
    :param custom_model_path: Checkpoint whose weights are fine-tuned.
    :param resume_checkpoint: spaCR training artifact whose weights, optimizer,
        scheduler and epoch counter are restored.
    :param preprocessing: Preprocessing description stored in the checkpoint.
    :param classes: Class names, used when ``<src>/train`` does not exist.
    :param label_smoothing: Label-smoothing epsilon for the smoothed losses.
        Default ``0.1``.
    :param focal_gamma: Focal-loss focusing parameter. Default ``2.0``.
    :param focal_alpha: Focal-loss class-balancing factor.
    :param logit_adjust_tau: Strength of the logit adjustment; ``0`` disables
        it. Default ``1.0``.
    :param settings: the run's settings dict, recorded (hashed) in the model card.
    :param split_rule: how the held-out set was drawn, in words, for the card.
    :param write_card: write ``<model>.card.json`` beside the weights and
        register it as an artifact. On by default: an uncarded checkpoint is a
        file nobody can audit six months later.
    :returns: ``(model, model_path)`` — the trained model and the best
        checkpoint, which on a resumed run where no epoch beat the restored
        best metric is ``resume_checkpoint`` itself, or ``(None, None)`` when
        ``model_type`` could not be built.
    :raises ValueError: on an unknown ``optimizer_type``, a checkpoint whose
        class count differs from ``num_classes``, a ``resume_checkpoint``
        carrying no optimizer state, or a checkpoint that already completed
        ``epochs``.
    :raises FileNotFoundError: if ``custom_model_path`` or
        ``resume_checkpoint`` does not exist.
    """


    if channels is None:
        channels = ['r', 'g', 'b']
    from .io import _save_model, _save_progress
    from .utils import choose_model, suggest_training_changes, build_loss, estimate_class_counts

    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders) if val_loaders else 0}')
    if test_loaders is not None:
        print(f'Test batches:{len(test_loaders)}')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using {device} for Torch')

    head_dim = max(1, int(num_classes))

    #counts = estimate_class_counts(train_loaders, head_dim) if head_dim >= 2 else None

    train_data_dir = os.path.join(src, 'train')

    if os.path.isdir(train_data_dir):
        # The folder names in ImageFolder's sorted order ARE the head order,
        # so they win over whatever the caller passed.
        classes = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d)) and not d.startswith('.')])
    elif not classes:
        # ...but with no folder tree to read (a tar-backed dataset, a caller
        # supplying its own loaders), the caller's list is the only class
        # naming there is. The old `else: classes = None` threw it away, so
        # the checkpoint and every per-class report came out as
        # class_0/class_1 even when the names were passed in.
        classes = None

    counts = estimate_class_counts(train_loaders, head_dim, src=train_data_dir, classes=classes) if (head_dim >= 2 and classes) else None

    loss_fn = build_loss(
        loss_type=loss_type,
        num_classes=head_dim,
        class_counts=counts,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        logit_adjust_tau=logit_adjust_tau
    )

    initialization_path = resume_checkpoint or custom_model_path
    model = choose_model(model_type, device,
                         False if initialization_path else init_weights,
                         dropout_rate,
                         use_checkpoint, verbose=verbose, num_classes=head_dim,
                         height=image_size, width=image_size)
    if model is None:
        print(f'Model {model_type} not found')
        # Match the 2-tuple arity of the success path below. A bare `return` made
        # the caller's `model, model_path = train_model(...)` raise
        # "cannot unpack non-iterable NoneType object", burying this message.
        return None, None

    resume_payload = None
    if initialization_path:
        if not os.path.isfile(initialization_path):
            raise FileNotFoundError(
                f"Training checkpoint does not exist: {initialization_path}")
        loaded_model, loaded_payload = load_model_artifact(
            initialization_path, map_location=device, model=model)
        loaded_classes = int(getattr(loaded_model, 'num_classes', head_dim))
        if loaded_classes != head_dim:
            raise ValueError(
                f"Checkpoint has {loaded_classes} output classes but this run "
                f"requests {head_dim}. Use matching classes or train a new head.")
        model = loaded_model
        if resume_checkpoint:
            if loaded_payload.get('optimizer_state_dict') is None:
                raise ValueError(
                    "resume_checkpoint is not a resumable spaCR training "
                    "artifact (optimizer state is missing). Use "
                    "custom_model_path to fine-tune its weights instead.")
            resume_payload = loaded_payload
            print(f"Resuming training state from {resume_checkpoint}")
        else:
            print(f"Fine-tuning model weights from {custom_model_path}")

    print(f'Loading Model to {device}...')
    model.to(device)

    import torch.optim as _optim
    ot = str(optimizer_type).lower()
    if ot == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                          weight_decay=weight_decay, amsgrad=amsgrad)
    elif ot == 'adam':
        optimizer = _optim.Adam(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999), weight_decay=weight_decay,
                                amsgrad=amsgrad)
    elif ot == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=learning_rate, eps=1e-8,
                            weight_decay=weight_decay)
    elif ot == 'sgd':
        optimizer = _optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                               nesterov=True, weight_decay=weight_decay)
    elif ot == 'rmsprop':
        optimizer = _optim.RMSprop(model.parameters(), lr=learning_rate,
                                   momentum=0.9, weight_decay=weight_decay)
    elif ot == 'nadam':
        optimizer = _optim.NAdam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    elif ot == 'radam':
        optimizer = _optim.RAdam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    elif ot == 'adamax':
        optimizer = _optim.Adamax(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    elif ot == 'adadelta':
        optimizer = _optim.Adadelta(model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)
    elif ot == 'asgd':
        optimizer = _optim.ASGD(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type!r}. Choose from: "
            "adamw, adam, adamax, adagrad, adadelta, asgd, sgd, rmsprop, "
            "nadam, radam.")

    if schedule == 'step_lr':
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs / 5)), gamma=0.75)
    elif schedule == 'reduce_lr_on_plateau':
        # `verbose` was deprecated in torch 2.2 and removed in 2.5; passing it
        # made this documented schedule raise TypeError before the first batch.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10)
    elif schedule == 'cosine':
        # FIX: new option — cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif schedule == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, int(epochs / 5)), eta_min=1e-7)
    elif schedule == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95)
    elif schedule == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1,
            total_iters=max(1, epochs))
    else:
        scheduler = None

    start_epoch = 1
    best_val_acc = -1.0
    best_model_path = None
    epochs_without_improvement = 0
    if resume_payload is not None:
        training_state = restore_training_state(
            resume_payload, optimizer=optimizer, scheduler=scheduler)
        start_epoch = int(training_state.get('epoch') or 0) + 1
        restored_best = training_state.get('best_metric')
        if restored_best is not None:
            best_val_acc = float(restored_best)
        epochs_without_improvement = int(
            training_state.get('epochs_without_improvement') or 0)
        best_model_path = os.path.abspath(resume_checkpoint)
        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {start_epoch - 1}, but "
                f"epochs={epochs}. Increase epochs to continue training.")

    accumulated_train_dicts, accumulated_val_dicts, accumulated_test_dicts = [], [], []
    # Full per-epoch history kept for the live training plot (the accumulators
    # above get consumed/cleared by _save_progress each epoch).
    live_train_hist, live_val_hist = [], []
    live_figure = None
    # (epoch, metrics, [probs, labels]) of the epoch whose weights became the
    # best checkpoint — the ONLY held-out evaluation that describes the file
    # the model card is written beside. Using the last epoch's numbers for a
    # checkpoint saved five epochs earlier is the quiet way a card lies.
    held_out_raw = None
    # Kept separate from any training ledger: a failed live plot says nothing
    # about whether the weights are trustworthy.
    _curve_ledger = RunLedger('train_model:live_curves')
    tensorboard_writer, _ = _open_tensorboard_writer(dst, tensorboard)

    print('Training ...')
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        start_time = time.time()

        if gradient_accumulation:
            optimizer.zero_grad(set_to_none=True)

        # record total number of batches so we can detect leftover gradients
        n_batches = len(train_loaders)

        for batch_idx, (data, target, filenames) in enumerate(train_loaders, start=1):
            data = data.to(device)
            logits = model(data)

            is_multiclass = (logits.ndim == 2 and logits.size(1) >= 2)

            if is_multiclass:
                if target.ndim == 2:
                    target = target.argmax(dim=1)
                target = target.to(device).long()
                if not (logits.ndim == 2 and logits.size(1) == head_dim):
                    raise RuntimeError(
                        f"Expected logits (N,{head_dim}) for CE, got {tuple(logits.shape)}")
            else:
                target = target.to(device).float()

            loss = loss_fn(logits, target)

            if gradient_accumulation:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (not gradient_accumulation) or (batch_idx % gradient_accumulation_steps == 0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # flush leftover accumulated gradients at the end of the epoch
        if gradient_accumulation and (n_batches % gradient_accumulation_steps != 0):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Epoch end: evaluate
        train_time = time.time() - start_time
        loop_loss_type = 'ce' if head_dim >= 2 else 'bce'
        train_dict, _ = evaluate_model_performance(
            model, train_loaders, epoch,
            loss_type=loop_loss_type,
            loss_fn=loss_fn,
            num_classes=head_dim
        )
        train_dict['train_time'] = train_time
        # The schedule moves the learning rate every epoch and nothing
        # recorded it, so "why did the curve bend at epoch 30" was
        # unanswerable from the run folder alone.
        train_dict['lr'] = float(optimizer.param_groups[0]['lr'])
        attach_per_class_columns(train_dict, classes)
        accumulated_train_dicts.append(train_dict)

        # initialize val_dict to None so the variable always exists for _save_model
        val_dict = None

        is_best = False
        if val_loaders is not None and len(val_loaders) > 0:
            val_dict, val_raw = evaluate_model_performance(
                model, val_loaders, epoch,
                loss_type=loop_loss_type,
                loss_fn=loss_fn,
                num_classes=head_dim
            )
            attach_per_class_columns(val_dict, classes)
            accumulated_val_dicts.append(val_dict)
            if schedule == 'reduce_lr_on_plateau':
                scheduler.step(val_dict['loss'])

            print(f"Progress: {train_dict.get('epoch', epoch)}/{epochs}, operation_type: Training, "
                  f"Train Loss: {train_dict.get('loss', float('nan')):.3f}, "
                  f"Val Loss: {val_dict.get('loss', float('nan')):.3f}, "
                  f"Train acc.: {train_dict.get('accuracy', float('nan')):.3f}, "
                  f"Val acc.: {val_dict.get('accuracy', float('nan')):.3f}, "
                  f"Train F1(macro): {train_dict.get('f1_macro', float('nan')):.3f}, "
                  f"Val F1(macro): {val_dict.get('f1_macro', float('nan')):.3f}")
            # The aggregate above is the number that hides a dead class.
            # Print the breakdown on its own line, every epoch, held-out
            # first — not once at the end, by which point the run is over.
            class_line = format_per_class_accuracy(val_dict, classes, 'Val ')
            if class_line:
                print(f"  {class_line}")

            # track best validation accuracy for early stopping and best-model selection
            current_val_acc = val_dict.get('accuracy', 0.0)
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                is_best = True
                epochs_without_improvement = 0
                held_out_raw = (epoch, val_dict, val_raw)
            else:
                epochs_without_improvement += 1
        else:
            print(f"Progress: {train_dict.get('epoch', epoch)}/{epochs}, operation_type: Training, "
                  f"Train Loss: {train_dict.get('loss', float('nan')):.3f}, "
                  f"Train acc.: {train_dict.get('accuracy', float('nan')):.3f}, "
                  f"Train F1(macro): {train_dict.get('f1_macro', float('nan')):.3f}")
            class_line = format_per_class_accuracy(train_dict, classes, 'Train ')
            if class_line:
                print(f"  {class_line}")
            current_train_acc = train_dict.get('accuracy', 0.0)
            if current_train_acc > best_val_acc:
                best_val_acc = current_train_acc
                is_best = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        try:
            _log_tensorboard_epoch(
                tensorboard_writer, train_dict, val_dict, epoch, classes)
        except Exception as exc:
            print(f"TensorBoard logging disabled after an error: {exc}")
            try:
                tensorboard_writer.close()
            except Exception:
                pass
            tensorboard_writer = None

        # Live training curves — follow loss/accuracy in real time in the GUI
        # when plot is enabled. Each epoch refreshes the same figure (the GUI
        # bridge captures plt.show and routes it to the figure view).
        # Accumulated unconditionally: the accumulators above are consumed
        # and cleared by _save_progress every epoch, so this is the only
        # in-memory record of the run, and the model card's curve needs it
        # whether or not anyone asked for a live plot.
        live_train_hist.append(train_dict)
        if val_dict is not None:
            live_val_hist.append(val_dict)
        if plot:
            # Cosmetic: a live curve that fails to render must not kill the
            # training run. It must not be *invisible* either — the bare
            # `pass` here hid a broken plot for the whole run.
            with _curve_ledger.item(f'epoch_{epoch}', stage='live_curves'):
                # Class names ride along inside the epoch dicts (see
                # attach_per_class_columns), so this call site keeps the
                # signature every existing caller and stub already has.
                live_figure = _plot_training_curves(
                    live_train_hist, live_val_hist, epochs, live_figure)

        if scheduler and schedule in (
                'step_lr', 'cosine', 'cosine_warm_restarts',
                'exponential', 'linear'):
            # FIX: also step cosine scheduler here
            scheduler.step()

        # Save rolling CSVs
        if accumulated_train_dicts and accumulated_val_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_train_dicts),
                           pd.DataFrame(accumulated_val_dicts))
            accumulated_train_dicts, accumulated_val_dicts = [], []
        elif accumulated_train_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_train_dicts), None)
            accumulated_train_dicts = []
        elif accumulated_test_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_test_dicts), None)
            accumulated_test_dicts = []

        # pass val_dict to _save_model so checkpoint decisions use validation accuracy
        will_stop = (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        )
        model_path = _save_model(model, model_type, train_dict, dst, epoch, epochs,
                                 intermedeate_save=intermedeate_save,
                                 channels=channels,
                                 val_dict=val_dict,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 best_metric=best_val_acc,
                                 is_best=is_best,
                                 epochs_without_improvement=epochs_without_improvement,
                                 preprocessing=preprocessing,
                                 classes=classes,
                                 force_last=will_stop)

        # track the best model path based on validation accuracy
        if model_path is not None and is_best:
            best_model_path = model_path
        elif model_path is not None and best_model_path is None:
            best_model_path = model_path

        # early stopping — break if val hasn't improved for `patience` epochs
        if will_stop:
            print(f"\nEarly stopping at epoch {epoch}: no val improvement for "
                  f"{early_stopping_patience} epochs. Best val acc: {best_val_acc:.4f}")
            break

        # Periodic suggestions (every 25 epochs and final epoch)
        if (epoch % 25 == 0) or (epoch == epochs):
            try:
                report = suggest_training_changes(dst)
                print("== Summary ==")
                for k, v in report["summary"].items():
                    print(f"{k}: {v}")
                print("\n== Flags ==")
                print(", ".join(report["flags"]) or "none")
                print("\n== Suggestions ==")
                for i, s in enumerate(report["suggestions"], 1):
                    print(f"{i}. {s}")
            except Exception as e:
                print(f"[suggest_training_changes] Skipped at epoch {epoch}: {e}")

    # Not stamped and not fatal — the training artifacts are unaffected —
    # but a run where every live plot failed now says so instead of ending
    # with a silently empty figure pane.
    _curve_ledger.finalize()
    if tensorboard_writer is not None:
        tensorboard_writer.close()

    # return best_model_path if available, otherwise fall back to last model_path
    final_path = best_model_path if best_model_path is not None else model_path

    if write_card and final_path:
        # A card that fails to write must not lose the weights that were
        # just trained for six hours, but it must also not fail silently —
        # an uncarded checkpoint that nobody noticed is the state this
        # feature exists to end.
        try:
            held_epoch, held_metrics, held_raw = (
                held_out_raw if held_out_raw is not None else (None, None, None))
            held = (held_out_report(held_raw[1], held_raw[0], classes)
                    if held_raw is not None else {})
            if held:
                held['epoch'] = int(held_epoch)
                held['selected_by'] = 'best validation accuracy'
            rule = split_rule or (
                f"validation split held out by generate_loaders "
                f"(val_split={settings.get('val_split')}, grouped by "
                f"{settings.get('cv_group_by', 'well')})"
                if isinstance(settings, dict) and settings.get('val_split')
                else '')
            card, card_path, _artifact = model_card(
                final_path,
                settings=settings,
                classes=classes,
                split_rule=rule,
                held_out=held,
                train_metrics=(held_metrics if held_metrics is not None
                               else train_dict),
                dataset_src=src,
                module='train',
                epochs=epochs,
                history=live_val_hist or live_train_hist,
                extra={'model_type': model_type, 'channels': list(channels),
                       'image_size': image_size, 'loss_type': loss_type,
                       'optimizer_type': optimizer_type, 'schedule': schedule,
                       'best_metric': float(best_val_acc)},
            )
            print(f"Model card: {card_path}")
        except Exception as exc:
            print(f"Could not write the model card for {final_path} "
                  f"({type(exc).__name__}: {exc}). The weights are unaffected.")

    return model, final_path

def generate_activation_map(settings):
    """Generate saliency or Grad-CAM activation maps for every image in a tar dataset.

    Loads the model, iterates the dataset, computes the requested map
    type per batch, saves per-image maps into class/plate/well folders,
    optionally plots batch grids, computes activation-image correlations,
    and pushes both maps and correlations into the measurement database.

    :param settings: Settings dict — see
        ``settings.get_default_generate_activation_map_settings`` for
        keys (``dataset``, ``model_path``, ``cam_type``, ``target_layer``,
        ``image_size``, ``batch_size``, ``channels``, ``normalize``,
        ``save``, ``plot``, ``correlation``, ...).
    :returns: None
    """
    from .utils import SaliencyMapGenerator, GradCAMGenerator, SelectChannels, activation_maps_to_database, activation_correlations_to_database
    from .utils import print_progress, save_settings, calculate_activation_correlations
    from .attribution import ATTRIBUTION_METHODS, AttributionMapGenerator, methods_by_family
    from .io import TarImageDataset
    from .settings import get_default_generate_activation_map_settings

    _empty_device_cache()
    gc.collect()
    
    plt.clf()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    source_folder = os.path.dirname(os.path.dirname(settings['dataset']))
    settings['src'] = source_folder
    settings = get_default_generate_activation_map_settings(settings)
    save_settings(settings, name=f"{settings['cam_type']}_settings", show=False)
    
    if settings['model_type'] == 'maxvit' and settings['target_layer'] == None:
        settings['target_layer'] = 'base_model.blocks.3.layers.1.layers.MBconv.layers.conv_b'
    if settings['cam_type'] in ['saliency_image', 'saliency_channel']:
        settings['target_layer'] = None

    # Anything outside the four legacy names is one of the methods registered
    # in spacr.attribution (Grad-CAM++, Score-CAM, XGrad-CAM, Layer-CAM,
    # Eigen-CAM, guided backprop, input x gradient, DeepLIFT, integrated
    # gradients, occlusion, feature ablation, attention rollout). They run
    # through the same batch loop via AttributionMapGenerator, which exposes
    # the same compute_*_and_predictions / plot_activation_grid calls the two
    # legacy generators do.
    _LEGACY_CAM_TYPES = ('gradcam', 'gradcam_pp', 'saliency_image',
                         'saliency_channel')
    cam_type = settings['cam_type']
    use_attribution = cam_type not in _LEGACY_CAM_TYPES
    if use_attribution and cam_type not in ATTRIBUTION_METHODS:
        raise ValueError(
            f"unknown cam_type {cam_type!r}. Legacy names: "
            f"{list(_LEGACY_CAM_TYPES)}. Registered attribution methods by "
            f"family — " + ", ".join(f"{fam}: {names}" for fam, names
                                     in methods_by_family().items()))
    settings.setdefault('smoothgrad_samples', 0)
    settings.setdefault('smoothgrad_sigma', 0.15)

    # Set number of jobs for loading
    n_jobs = settings['n_jobs']
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 4)

    # Set transforms for images. The Normalize step has to be appended
    # conditionally: an inline `... if normalize_input else None` put a literal
    # None into the Compose list, so normalize_input=False raised
    # "TypeError: 'NoneType' object is not callable" on the first image.
    transform_steps = [
        transforms.ToTensor(),
        transforms.CenterCrop(size=(settings['image_size'], settings['image_size'])),
    ]
    if settings['normalize_input']:
        transform_steps.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform_steps.append(SelectChannels(settings['channels']))
    transform = transforms.Compose(transform_steps)

    # Handle dataset path
    if not os.path.exists(settings['dataset']):
        print(f"Dataset not found at {settings['dataset']}")
        return

    # Load the model
    model, _ = _load_inference_model(settings['model_path'], device)
    model.to(device)
    model.eval()

    # Create directory for saving activation maps if it does not exist
    dataset_dir = os.path.dirname(settings['dataset'])
    dataset_name = os.path.splitext(os.path.basename(settings['dataset']))[0]
    save_dir = os.path.join(dataset_dir, dataset_name, settings['cam_type'])
    batch_grid_fldr = os.path.join(save_dir, 'batch_grids')
    
    if settings['save']:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Activation maps will be saved in: {save_dir}")
        
    if settings['plot']:
        os.makedirs(batch_grid_fldr, exist_ok=True)
        print(f"Batch grid maps will be saved in: {batch_grid_fldr}")
    
    # Load dataset
    dataset = TarImageDataset(settings['dataset'], transform=transform)
    # Seeded generator + worker init: which images land in the activation-map
    # batches is otherwise a different sample every run.
    data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=settings['shuffle'], num_workers=n_jobs, pin_memory=True,
                             generator=torch_generator(stream='activation_maps'),
                             worker_init_fn=seed_worker if n_jobs else None)

    # Initialize generator based on cam_type
    if use_attribution:
        cam_generator = AttributionMapGenerator(
            model, method=cam_type, target_layer=settings['target_layer'],
            model_type=settings.get('model_type'),
            smoothgrad_samples=settings['smoothgrad_samples'],
            smoothgrad_sigma=settings['smoothgrad_sigma'])
    elif settings['cam_type'] in ['gradcam', 'gradcam_pp']:
        cam_generator = GradCAMGenerator(model, target_layer=settings['target_layer'], cam_type=settings['cam_type'])
    elif settings['cam_type'] in ['saliency_image', 'saliency_channel']:
        cam_generator = SaliencyMapGenerator(model)

    time_ls = []
    for batch_idx, (inputs, filenames) in enumerate(data_loader):
        start = time.time()
        img_paths = []
        inputs = inputs.to(device)

        # Compute activation maps and predictions
        if use_attribution:
            activation_maps, predicted_classes = cam_generator.compute_maps_and_predictions(inputs)
        elif settings['cam_type'] in ['gradcam', 'gradcam_pp']:
            activation_maps, predicted_classes = cam_generator.compute_gradcam_and_predictions(inputs)
        elif settings['cam_type'] in ['saliency_image', 'saliency_channel']:
            activation_maps, predicted_classes = cam_generator.compute_saliency_and_predictions(inputs)

        # Move activation maps to CPU
        activation_maps = activation_maps.cpu()

        # Sum saliency maps for 'saliency_image' type
        if settings['cam_type'] == 'saliency_image':
            summed_activation_maps = []
            for i in range(activation_maps.size(0)):
                activation_map = activation_maps[i]                
                #print(f"1: {activation_map.shape}")
                activation_map_sum = activation_map.sum(dim=0, keepdim=False)
                #print(f"2: {activation_map.shape}")
                activation_map_sum = np.squeeze(activation_map_sum, axis=0)
                #print(f"3: {activation_map_sum.shape}")
                summed_activation_maps.append(activation_map_sum)
            activation_maps = torch.stack(summed_activation_maps)

        # For plotting
        if settings['plot']:
            fig = cam_generator.plot_activation_grid(inputs, activation_maps, predicted_classes, overlay=settings['overlay'], normalize=settings['normalize'])
            pdf_save_path = os.path.join(batch_grid_fldr,f"batch_{batch_idx}_grid.pdf")
            pdf_save_path = save_figure(fig, pdf_save_path)
            print(f"Saved batch grid to {pdf_save_path}")
            #plt.show()
            display(fig)
                    
        for i in range(inputs.size(0)):
            activation_map = activation_maps[i].detach().numpy()

            # A flat map (e.g. a Grad-CAM fully suppressed by its F.relu, which
            # happens whenever the target layer has collapsed to 1x1) has
            # max == min, so the unguarded min-max rescale below used to produce
            # 0/0 -> all-NaN and then an undefined NaN -> uint8 cast. `rng > 0` is
            # also False for NaN, so a map that arrives already NaN is absorbed too.
            if use_attribution or settings['cam_type'] in ['saliency_image', 'gradcam', 'gradcam_pp']:
                # Every spacr.attribution method returns a single (H, W) map,
                # so it takes the same greyscale path the summed saliency and
                # the CAMs already took.
                #activation_map = activation_map.sum(axis=0)
                lo = activation_map.min()
                rng = activation_map.max() - lo
                activation_map = (activation_map - lo) / rng if rng > 0 else np.zeros_like(activation_map)
                activation_map = (activation_map * 255).astype(np.uint8)
                activation_image = Image.fromarray(activation_map, mode='L')

            elif settings['cam_type'] == 'saliency_channel':
                # Handle each channel separately and save as RGB
                rgb_activation_map = np.zeros((activation_map.shape[1], activation_map.shape[2], 3), dtype=np.uint8)
                for c in range(min(activation_map.shape[0], 3)):  # Limit to 3 channels for RGB
                    channel_map = activation_map[c]
                    lo = channel_map.min()
                    rng = channel_map.max() - lo
                    channel_map = (channel_map - lo) / rng if rng > 0 else np.zeros_like(channel_map)
                    rgb_activation_map[:, :, c] = (channel_map * 255).astype(np.uint8)
                activation_image = Image.fromarray(rgb_activation_map, mode='RGB')

            # Save activation maps
            class_pred = predicted_classes[i].item()
            parts = filenames[i].split('_')
            plate = parts[0]
            well = parts[1]
            save_class_dir = os.path.join(save_dir, f'class_{class_pred}', str(plate), str(well))
            os.makedirs(save_class_dir, exist_ok=True)
            save_path = os.path.join(save_class_dir, f'{filenames[i]}')
            if settings['save']:
                activation_image.save(save_path)
            img_paths.append(save_path)
        
        if settings['save']:
            activation_maps_to_database(img_paths, source_folder, settings)
            
        if settings['correlation']:
            df = calculate_activation_correlations(inputs, activation_maps, filenames, manders_thresholds=settings['manders_thresholds'])
            if settings['plot']:
                display(df)
            if settings['save']:
                activation_correlations_to_database(df, img_paths, source_folder, settings)

        stop = time.time()
        duration = stop - start
        time_ls.append(duration)
        files_processed = batch_idx * settings['batch_size']
        files_to_process = len(data_loader) * settings['batch_size']
        print_progress(files_processed, files_to_process, n_jobs=n_jobs, time_ls=time_ls, batch_size=settings['batch_size'], operation_type="Generating Activation Maps")

    _empty_device_cache()
    gc.collect()
    print("Activation map generation complete.")

def analyze_activation_maps(model, images, methods=None, *, masks=None,
                            target=None, target_layer=None, model_type=None,
                            n_steps=12, baseline='blur', sanity_check=True,
                            sanity_threshold=0.5, verbose=True):
    """Attribute images several ways and report whether any of it is trustworthy.

    Grad-CAM and a saliency map always render. Nothing about the picture says
    whether it describes what the model uses, so this runs the four checks that
    do (see :mod:`spacr.attribution` for why each is limited):

    * **deletion / insertion AUC** — remove, or add, the pixels each map ranks
      highest and track the class probability. A flat deletion curve means the
      map ranked pixels the model does not use.
    * **pointing game** — does the map's peak land inside the object mask?
      Scored only when ``masks`` is given; spaCR's ``merged/*.npy`` carries the
      label planes.
    * **model-randomisation sanity check** (Adebayo et al. 2018) — randomise the
      weights layer by layer and attribute again. A method whose map survives
      that is an edge detector, and this is the one check that catches it.
    * **agreement** — rank correlation between the methods. Disagreement is
      strong evidence that no single map should be quoted alone.

    :param model: the trained classifier.
    :param images: one image tensor, or a sequence of them, ``(C, H, W)``.
    :param methods: method names from
        :data:`spacr.attribution.ATTRIBUTION_METHODS`; defaults to one
        representative of each family.
    :param masks: optional per-image boolean object masks for the pointing game.
    :param target: class index to explain; defaults to each image's prediction.
    :param target_layer: CAM target layer, or None for the last convolution.
    :param model_type: architecture name, used to make errors readable.
    :param n_steps: steps in the deletion / insertion curves.
    :param baseline: what removed pixels become — ``'blur'``, ``'zero'``,
        ``'mean'`` or ``'uniform'``.
    :param sanity_check: run the randomisation check (on the first image).
    :param sanity_threshold: rank correlation below which a method passes.
    :param verbose: print the per-method verdicts.
    :returns: dict with ``table`` (a DataFrame, one row per method × image),
        ``attributions``, ``agreement``, ``sanity`` and ``notes``.
    """
    import pandas as pd

    from .attribution import (NOT_AN_EXPLANATION, compare_methods,
                              faithfulness, method_agreement,
                              randomization_sanity_check)

    if isinstance(images, torch.Tensor) and images.ndim == 3:
        images = [images]
    images = list(images)
    if not images:
        raise ValueError(
            "analyze_activation_maps needs at least one image; nothing was "
            "given, so there is nothing to attribute.")
    methods = list(methods or ['gradcam', 'saliency', 'integrated_gradients',
                               'occlusion'])
    masks = list(masks) if masks is not None else None

    rows = []
    per_image = []
    for i, image in enumerate(images):
        atts = compare_methods(model, image, methods, target=target,
                               layer=target_layer, model_type=model_type)
        per_image.append(atts)
        mask = masks[i] if masks is not None and i < len(masks) else None
        for att in atts:
            failed = any(n.startswith('FAILED:') for n in att.notes)
            row = {'image': i, 'method': att.method, 'family': att.family,
                   'backend': att.backend, 'target': att.target,
                   'predicted': att.predicted, 'flat': att.is_flat(),
                   'failed': failed}
            if not failed:
                scores = faithfulness(model, image, att, target=att.target,
                                      n_steps=n_steps, baseline=baseline,
                                      mask=mask)
                row.update({'deletion_auc': scores['deletion_auc'],
                            'insertion_auc': scores['insertion_auc'],
                            'pointing_game': scores['pointing_game']})
            else:
                row.update({'deletion_auc': float('nan'),
                            'insertion_auc': float('nan'),
                            'pointing_game': None})
            rows.append(row)

    agreement = method_agreement([a for a in per_image[0]
                                  if not a.is_flat()]) \
        if sum(1 for a in per_image[0] if not a.is_flat()) >= 2 else None

    sanity = {}
    if sanity_check:
        for name in methods:
            try:
                sanity[name] = randomization_sanity_check(
                    model, images[0], name, target=target, layer=target_layer,
                    model_type=model_type, threshold=sanity_threshold)
            except Exception as exc:
                sanity[name] = f"{type(exc).__name__}: {exc}"

    table = pd.DataFrame(rows)
    if not table.empty and 'deletion_auc' in table.columns:
        table = table.sort_values(['image', 'deletion_auc'],
                                  na_position='last').reset_index(drop=True)

    notes = [NOT_AN_EXPLANATION]
    if masks is None:
        notes.append(
            "No object masks were given, so the pointing game was not scored. "
            "spaCR's merged/*.npy files carry the label planes if you want it.")
    if not sanity_check:
        notes.append(
            "The model-randomisation sanity check was skipped. Without it, a "
            "method that returns the same map for a randomised model is "
            "indistinguishable here from one that does not.")

    if verbose:
        print(NOT_AN_EXPLANATION)
        for name, check in sanity.items():
            print(check if isinstance(check, str) else check.verdict())
        if agreement is not None:
            print(agreement.verdict())

    return {'table': table, 'attributions': per_image,
            'agreement': agreement, 'sanity': sanity, 'notes': notes}


def visualize_classes(model, dtype, class_names, **kwargs):
    """Show one synthesised class-visualisation image for class 0 and class 1.

    The loop is hard-coded to the first two classes, so a model with more than
    two classes has only those two visualised.

    :param model: Trained classifier.
    :param dtype: Tensor dtype used for optimisation.
    :param class_names: Ordered class names; at least two are required, since
        indices 0 and 1 are both looked up.
    :param kwargs: Extra keyword arguments forwarded to
        ``utils.class_visualization``.
    :returns: None
    """
    from .utils import class_visualization

    for target_y in range(2):  # Assuming binary classification
        print(f"Visualizing class: {class_names[target_y]}")
        visualization = class_visualization(target_y, model, dtype, **kwargs)
        plt.imshow(visualization)
        plt.title(f"Class {class_names[target_y]} Visualization")
        plt.axis('off')
        plt.show()

def visualize_integrated_gradients(src, model_path, target_label_idx=0, image_size=224, channels=None, normalize=True, save_integrated_grads=False, save_dir='integrated_grads'):
    """Compute and plot Integrated Gradients maps for every PNG under ``src``.

    :param src: Folder of PNG images.
    :param model_path: Path to the trained model checkpoint.
    :param target_label_idx: Target class index for the attribution.
        Default ``0``.
    :param image_size: Square input size in pixels. Default ``224``.
    :param channels: Channel subset to keep. Default ``[1, 2, 3]``.
    :param normalize: Apply per-channel normalisation. Default ``True``.
    :param save_integrated_grads: If True, save each map as PNG.
        Default ``False``.
    :param save_dir: Output folder for saved maps. Default
        ``'integrated_grads'``.
    :returns: None
    """
    if channels is None:
        channels = [1,2,3]
    from .utils import IntegratedGradients, preprocess_image

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, _ = _load_inference_model(model_path, device)
    model.to(device)
    integrated_gradients = IntegratedGradients(model)

    if save_integrated_grads and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = []
    filenames = []
    for file in os.listdir(src):
        if not file.endswith('.png'):
            continue
        image_path = os.path.join(src, file)
        image, input_tensor = preprocess_image(image_path, normalize=normalize, image_size=image_size, channels=channels)
        images.append(image)
        filenames.append(file)

        input_tensor = input_tensor.to(device)
        integrated_grads = integrated_gradients.generate_integrated_gradients(input_tensor, target_label_idx)
        integrated_grads = np.mean(integrated_grads, axis=1).squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title("Original Image")
        ax[1].imshow(integrated_grads, cmap='hot')
        ax[1].axis('off')
        ax[1].set_title("Integrated Gradients")
        # Same trap as in visualize_smooth_grad: `image` is the unresized original
        # while the attribution map is image_size square, so the blend below only
        # broadcast when the source PNG happened to be image_size square.
        overlay = np.array(image.resize((image_size, image_size)))
        overlay = overlay / overlay.max()
        integrated_grads_rgb = np.stack([integrated_grads] * 3, axis=-1)  # Convert saliency map to RGB
        overlay = (overlay * 0.5 + integrated_grads_rgb * 0.5).clip(0, 1)
        ax[2].imshow(overlay)
        ax[2].axis('off')
        ax[2].set_title("Overlay")
        plt.show()

        if save_integrated_grads:
            os.makedirs(save_dir, exist_ok=True)
            integrated_grads_image = Image.fromarray((integrated_grads * 255).astype(np.uint8))
            integrated_grads_image.save(os.path.join(save_dir, f'integrated_grads_{file}'))

class SmoothGrad:
    """SmoothGrad attribution: average gradients over noisy copies of the input.

    :param model: PyTorch classifier used for gradient computation.
    :param n_samples: Number of noisy samples to average over. Default ``50``.
    :param stdev_spread: Noise standard deviation as a fraction of the
        input's dynamic range. Default ``0.15``.
    """

    def __init__(self, model, n_samples=50, stdev_spread=0.15):
        """Store the model and noise parameters."""
        self.model = model
        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def compute_smooth_grad(self, input_tensor, target_class):
        """Return the averaged gradient map for ``target_class`` given ``input_tensor``.

        :param input_tensor: Input tensor to attribute (single sample or batch).
        :param target_class: Class index whose logit is differentiated.
        :returns: Tensor of the same shape as ``input_tensor`` holding
            the averaged gradients.
        """
        self.model.eval()
        stdev = self.stdev_spread * (input_tensor.max() - input_tensor.min())
        total_gradients = torch.zeros_like(input_tensor)
        
        for i in range(self.n_samples):
            noise = torch.normal(mean=0, std=stdev, size=input_tensor.shape).to(input_tensor.device)
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_()
            output = self.model(noisy_input)
            self.model.zero_grad()
            # Back-propagate the whole target column, not just row 0: with
            # `output[0, target_class]` autograd only populated row 0 of .grad, so
            # a batched input silently came back with all-zero attributions for
            # every sample after the first. Identical for a single sample.
            output[:, target_class].sum().backward()
            total_gradients += noisy_input.grad

        avg_gradients = total_gradients / self.n_samples
        return avg_gradients.abs()

def visualize_smooth_grad(src, model_path, target_label_idx, image_size=224, channels=None, normalize=True, save_smooth_grad=False, save_dir='smooth_grad'):
    """Compute and plot SmoothGrad maps for every PNG under ``src``.

    :param src: Folder of PNG images.
    :param model_path: Path to the trained model checkpoint.
    :param target_label_idx: Target class index for the attribution.
    :param image_size: Square input size in pixels. Default ``224``.
    :param channels: Channel subset to keep. Default ``[1, 2, 3]``.
    :param normalize: Apply per-channel normalisation. Default ``True``.
    :param save_smooth_grad: If True, save each map as PNG. Default ``False``.
    :param save_dir: Output folder for saved maps. Default ``'smooth_grad'``.
    :returns: None
    """
    if channels is None:
        channels = [1,2,3]
    from .utils import preprocess_image

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, _ = _load_inference_model(model_path, device)
    model.to(device)
    smooth_grad = SmoothGrad(model)

    if save_smooth_grad and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = []
    filenames = []
    for file in os.listdir(src):
        if not file.endswith('.png'):
            continue
        image_path = os.path.join(src, file)
        image, input_tensor = preprocess_image(image_path, normalize=normalize, image_size=image_size, channels=channels)
        images.append(image)
        filenames.append(file)

        input_tensor = input_tensor.to(device)
        smooth_grad_map = smooth_grad.compute_smooth_grad(input_tensor, target_label_idx)
        smooth_grad_map = np.mean(smooth_grad_map.cpu().data.numpy(), axis=1).squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title("Original Image")
        ax[1].imshow(smooth_grad_map, cmap='hot')
        ax[1].axis('off')
        ax[1].set_title("SmoothGrad")
        # preprocess_image returns the UNRESIZED PIL image next to the resized
        # tensor, so blending np.array(image) with the image_size-sized map raised
        # a broadcast ValueError for any source PNG that is not image_size square.
        # Blend at the resolution the model actually saw (a no-op copy when they
        # already match); ax[0] still shows the full-resolution original.
        overlay = np.array(image.resize((image_size, image_size)))
        overlay = overlay / overlay.max()
        smooth_grad_map_rgb = np.stack([smooth_grad_map] * 3, axis=-1)  # Convert smooth grad map to RGB
        overlay = (overlay * 0.5 + smooth_grad_map_rgb * 0.5).clip(0, 1)
        ax[2].imshow(overlay)
        ax[2].axis('off')
        ax[2].set_title("Overlay")
        plt.show()

        if save_smooth_grad:
            os.makedirs(save_dir, exist_ok=True)
            smooth_grad_image = Image.fromarray((smooth_grad_map * 255).astype(np.uint8))
            smooth_grad_image.save(os.path.join(save_dir, f'smooth_grad_{file}'))
            
def save_top_class_examples(df, tar_path, dst, n=20, classes=None):
    """Extract the ``n`` most confident images per class from a tar into class-labelled folders.

    For binary classification, class 0 keeps the lowest ``pred`` scores
    and class 1 keeps the highest. Multiclass output is ranked using the
    corresponding ``prob_class_<index>`` column.

    :param df: DataFrame with columns ``path`` (tar member name) and
        ``pred`` (probability). Multiclass results also contain
        ``predicted_label`` and one ``prob_class_<index>`` column per class.
    :param tar_path: Tar archive containing the images.
    :param dst: Output root; ``dst/class_<label>/`` subfolders are
        created.
    :param n: Number of images to keep per class. Default ``20``.
    :param classes: Optional display labels, in model-output order. When
        omitted, multiclass labels are inferred from the probability columns;
        binary output defaults to ``[0, 1]``.
    :returns: ``dst`` — for chaining.
    """
    import os
    import re
    import tarfile

    if 'path' not in df.columns or 'pred' not in df.columns:
        raise ValueError(
            "Top-example export requires prediction columns 'path' and 'pred'.")
    if int(n) < 1:
        raise ValueError("n must be at least 1 when exporting top examples.")
    n = int(n)

    probability_columns = []
    for column in df.columns:
        match = re.fullmatch(r'prob_class_(\d+)', str(column))
        if match:
            probability_columns.append((int(match.group(1)), column))
    probability_columns.sort()

    # Build each folder's selection once. ``classes`` contains human-readable
    # labels, whereas the probability-column suffix is the model-output index.
    selections = []
    if probability_columns:
        labels = list(classes) if classes is not None else [
            index for index, _column in probability_columns
        ]
        if len(labels) != len(probability_columns):
            raise ValueError(
                f"Received {len(labels)} class labels for "
                f"{len(probability_columns)} model outputs.")
        for label, (_index, probability_column) in zip(
                labels, probability_columns):
            selections.append(
                (label, df.nlargest(n, probability_column)))
    else:
        labels = list(classes) if classes is not None else [0, 1]
        if len(labels) != 2:
            raise ValueError(
                "More than two class labels require multiclass probability "
                "columns named prob_class_0, prob_class_1, ... .")
        selections = [
            (labels[0], df.nsmallest(n, 'pred')),
            (labels[1], df.nlargest(n, 'pred')),
        ]

    # Build a lookup: tar member name → list of destination paths. An image can
    # legitimately appear at both binary extremes in a one-row result.
    member_destinations = {}
    for label, top in selections:
        safe_label = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label)).strip('_')
        safe_label = safe_label or 'unnamed'
        cls_dir = os.path.join(dst, f'class_{safe_label}')
        os.makedirs(cls_dir, exist_ok=True)
        for _, row in top.iterrows():
            fname = os.path.basename(row['path'])
            dest_file = os.path.join(cls_dir, fname)
            member_destinations.setdefault(row['path'], []).append(dest_file)

    # -- single pass through the tar: extract only the members we need --
    extracted = 0
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name in member_destinations:
                source = tar.extractfile(member)
                if source is None:
                    continue
                img_bytes = source.read()
                for dest_file in member_destinations[member.name]:
                    with open(dest_file, 'wb') as f:
                        f.write(img_bytes)
                    extracted += 1

    print(f"Saved {extracted} top-confidence example images to {dst}")
    return dst

def merge_predictions_into_db(df, db_path, table='png_list', pred_col='pred',
                               class_col='cv_predictions'):
    """Write per-image prediction scores back into a spacr SQLite database.

    Thin wrapper over :func:`spacr.predictions.merge_cv_predictions`, which is
    the one merge path Classify (CV) and Classify (ML) share. It used to be
    implemented here, keyed on ``basename(png_path)``, and collapsed repeated
    basenames with a plain ``dict`` assignment -- so a run over two source
    folders whose plates were both called ``plate1`` scored one plate with the
    other one's predictions and said nothing. The replacement keys on
    ``prcfo``, refuses a key that arrives with two different values instead of
    letting the last one win, counts every row it could not place, and runs in
    one transaction. See :mod:`spacr.predictions`.

    :param df: DataFrame with columns ``path``, ``pred`` and
        ``cv_predictions``.
    :param db_path: SQLite database file.
    :param table: Target table. Default ``'png_list'``.
    :param pred_col: Database column for the probability. Default ``'pred'``.
    :param class_col: Database column for the class label. Default
        ``'cv_predictions'``.
    :returns: Number of DB rows updated, or ``None`` if the database is
        missing.

    See Also:
        :func:`spacr.predictions.merge_prediction_results` -- the shared
        implementation, and the full :class:`~spacr.predictions.MergeReport`.
    """
    from .predictions import merge_cv_predictions

    report = merge_cv_predictions(
        df, db_path, table=table, score_col=pred_col, class_col=class_col,
        score_source=pred_col, class_source=class_col)
    if report is None:
        return None
    return report.matched_rows


def deep_spacr(settings=None):
    """Run the full spacr deep-learning pipeline: build dataset, train, apply model, merge predictions into the measurements DB.

    High-level driver that chains :func:`spacr.io.generate_training_dataset`
    -> :func:`train_test_model` -> :func:`spacr.io.generate_dataset` (tar)
    -> :func:`apply_model_to_tar` -> :func:`save_top_class_examples` ->
    :func:`merge_predictions_into_db`. Each stage is toggled by a flag
    in ``settings`` so the same call can (re)train from scratch, only
    apply a saved model, or only merge predictions.

    :param settings: Settings dict; canonicalized via
        :func:`spacr.settings.deep_spacr_defaults`. Key flags/inputs:

        - ``src`` — root folder(s) containing per-object PNGs from
          :func:`spacr.measure.measure_crop`.
        - ``generate_training_dataset`` — build ``train/``/``test/``
          splits via annotation rules before training.
        - ``train`` / ``test`` — pass-through to :func:`train_test_model`.
        - ``apply_model_to_dataset`` — run inference on a tar of PNGs.
        - ``model_path`` — pretrained checkpoint to reuse when
          ``train=False``.
        - ``tar_path`` — pre-built dataset tar; regenerated if missing.
        - ``n_top_examples`` — how many top-confidence images per class
          to copy into ``top_examples/``.
        - ``crop_source`` — ``'auto'`` | ``'png'`` | ``'merged'``, passed
          straight through to :func:`spacr.io.generate_training_dataset`
          and :func:`spacr.io.generate_dataset`. ``'merged'`` builds both
          the training split and the inference tar by cutting each crop out
          of ``merged/*.npy`` through :mod:`spacr.crops`, so neither needs a
          pre-generated PNG folder and neither can be built from a stale one.
        - Plus every key consumed by :func:`train_test_model`,
          :func:`spacr.io.generate_training_dataset`, and
          :func:`spacr.io.generate_dataset`.

    :returns: None. Writes model checkpoints, ``DL_model_settings.csv``,
        a dataset tar, ``top_examples/`` and updates the
        ``measurements.db`` in-place.

    Example:
        .. code-block:: python

            from spacr.deep_spacr import deep_spacr
            settings = {
                'src': '/data/plate01',
                'generate_training_dataset': True,
                'train': True, 'test': True,
                'apply_model_to_dataset': True,
                'model_type': 'maxvit_t', 'classes': ['neg','pos'],
                'epochs': 25, 'batch_size': 32,
            }
            deep_spacr(settings)

    See Also:
        :func:`train_test_model` — training-only entry point.
        :func:`spacr.io.generate_training_dataset` — labeling from
        annotation rules.
        :func:`apply_model_to_tar` — inference on a packed dataset.
    """
    if settings is None:
        settings = {}
    import os
    # local imports kept inside to avoid import cycles on some setups
    from .settings import deep_spacr_defaults
    from .io import generate_training_dataset, generate_dataset
    from .utils import save_settings

    # 1) expand defaults (now supports things like metadata_rules, annotation_columns, measurement_rules, etc.)
    settings = deep_spacr_defaults(settings)
    src_before = settings.get('src')

    # persist a snapshot of the config for reproducibility
    save_settings(settings, name='DL_model')

    # 2) dataset generation (train/test)
    if settings.get('train') or settings.get('test'):
        if settings.get('generate_training_dataset'):
            print("Generating train and test datasets ...")
            train_path, test_path = generate_training_dataset(settings)
            print(f'Generated Train set: {train_path}')
            print(f'Generated Test set: {test_path}')
            
            if train_path:
                settings['src'] = os.path.dirname(train_path)
            else:
                print("Training dataset generation failed; skipping model training step.")
                return  # or raise RuntimeError if you prefer hard fail
            
            # point training to the newly created train folder by default
            settings['src'] = os.path.dirname(train_path)
        elif isinstance(settings.get('src'), (list, tuple)):
            training_sources = [
                str(path) for path in settings['src'] if str(path).strip()]
            if len(training_sources) != 1:
                raise ValueError(
                    "Training from an existing split needs exactly one dataset "
                    "root containing train/<class>/ and test/<class>/. To use "
                    "multiple plate folders, enable Generate training dataset "
                    "so Classify first combines them into training_all.")
            settings['src'] = training_sources[0]

        print("Training model ...")
        training_result = train_test_model(settings)
        if settings.get('train'):
            cv_best = settings.get('cv_best_model_path')
            if cv_best:
                settings['cv_results_path'] = training_result
                settings['model_path'] = cv_best
            else:
                settings['model_path'] = training_result
        # restore original src (so later steps like apply can use the user’s dataset if needed)
        settings['src'] = src_before
        
    # 3) build the full, unlabelled inference dataset independently of model
    # application when requested. Applying a model still creates it on demand,
    # preserving the previous one-switch workflow.
    tar_path = settings.get('tar_path')
    needs_tar = settings.get('generate_full_dataset') or settings.get(
        'apply_model_to_dataset')
    if needs_tar and (
            not tar_path or not os.path.isabs(tar_path)
            or not os.path.exists(tar_path)):
        print("Generating full dataset tar ...")
        tar_path = generate_dataset(settings)
        if not tar_path or not os.path.isfile(tar_path):
            raise RuntimeError(
                "Full dataset generation did not produce a readable tar file.")
        settings['tar_path'] = tar_path

    # 4) apply model to the full dataset/tar
    if settings.get('apply_model_to_dataset'):

        model_path = settings.get('model_path')
        if model_path and os.path.exists(model_path):
            # -- run inference and get the results DataFrame --
            df = apply_model_to_tar(settings)

            # -- NEW: save the top-N most confident images per class --
            # dst sits next to the tar file, in a subfolder called 'top_examples'
            examples_dst = os.path.join(os.path.dirname(tar_path), 'top_examples')
            n_examples = settings.get('n_top_examples', 20)
            save_top_class_examples(
                df, tar_path, examples_dst, n=n_examples,
                classes=settings.get('classes'))

            # -- NEW: merge predictions back into the measurements database --
            # settings['src'] can be a string or list; use the first entry
            src_list = settings['src'] if isinstance(settings['src'], list) else [settings['src']]
            for src in src_list:
                db_path = os.path.join(src, 'measurements', 'measurements.db')
                merge_predictions_into_db(df, db_path)

        else:
            print(f"Model path {model_path} not found; skipping model application.")
            
def model_knowledge_transfer(teacher_paths, student_save_path, data_loader, device='cpu', student_model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, alpha=0.5, temperature=2.0, lr=1e-4, epochs=10):
    """Distil an ensemble of teacher models into a single student TorchModel.

    :param teacher_paths: Paths to the teacher checkpoints (each either
        a saved ``TorchModel`` or a state dict).
    :param student_save_path: Destination for the trained student; the
        suffix ``_KD.pth`` is appended.
    :param data_loader: Training DataLoader used during distillation.
    :param device: Torch device string. Default ``'cpu'``.
    :param student_model_name: TorchModel architecture name for the
        student. Default ``'maxvit_t'``.
    :param pretrained: Whether the student uses pretrained weights.
    :param dropout_rate: Optional dropout rate for the student.
    :param use_checkpoint: Whether to enable gradient checkpointing.
    :param alpha: Weight on the true-label loss vs. distillation loss.
        Default ``0.5``.
    :param temperature: Softmax temperature for distillation. Default ``2.0``.
    :param lr: Adam learning rate. Default ``1e-4``.
    :param epochs: Training epochs. Default ``10``.
    :returns: The trained student model.
    :raises ValueError: on unsupported checkpoint types.
    """
    from .utils import TorchModel

    if not teacher_paths:
        raise ValueError("teacher_paths must contain at least one model.")
    if data_loader is None or len(data_loader) == 0:
        raise ValueError("data_loader must contain at least one training batch.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if float(temperature) <= 0:
        raise ValueError("temperature must be greater than zero.")
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1.")

    # Adjust filename to reflect knowledge-distillation if desired
    if student_save_path.endswith('.pth'):
        base, ext = os.path.splitext(student_save_path)
    else:
        base = student_save_path
    student_save_path = base + '_KD.pth'

    # -- 1. Load teacher models --
    teachers = []
    print("Loading teacher models:")
    for path in teacher_paths:
        print(f"  Loading teacher: {path}")
        try:
            teacher, _ = load_model_artifact(path, map_location=device)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported checkpoint type at {path}: {exc}") from exc
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(teacher)

    teacher_classes = {
        int(getattr(teacher, 'num_classes',
                    getattr(getattr(teacher, 'spacr_classifier', None),
                            'out_features', 1)))
        for teacher in teachers
    }
    if len(teacher_classes) != 1:
        raise ValueError(
            f"All teachers must have the same output size; found "
            f"{sorted(teacher_classes)}.")
    num_classes = teacher_classes.pop()

    # -- 2. Initialize the student TorchModel --
    student_model = TorchModel(
        model_name=student_model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_checkpoint=use_checkpoint,
        num_classes=num_classes,
    ).to(device)

    # You could load a partial checkpoint into the student here if desired.

    # -- 3. Optimizer --
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # Distillation training loop
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for batch in data_loader:
            images, labels = _unpack_supervised_batch(batch)
            images, labels = images.to(device), labels.to(device)
            if labels.ndim == 2 and labels.size(1) > 1:
                labels = labels.argmax(dim=1)

            # Forward pass student
            logits_s = student_model(images)         # shape: (B, num_classes)
            logits_s_temp = logits_s / temperature   # scale by T

            # Distillation from teachers
            with torch.no_grad():
                # We'll average teacher probabilities
                teacher_probs_list = []
                for tm in teachers:
                    logits_t = tm(images) / temperature
                    if num_classes == 1:
                        positive = torch.sigmoid(logits_t.reshape(-1))
                        teacher_probs_list.append(
                            torch.stack((1.0 - positive, positive), dim=1))
                    else:
                        teacher_probs_list.append(F.softmax(logits_t, dim=1))
                # average them
                teacher_probs_ensemble = torch.mean(torch.stack(teacher_probs_list), dim=0)

            # Student probabilities (log-softmax)
            if num_classes == 1:
                flat = logits_s_temp.reshape(-1)
                student_log_probs = torch.stack(
                    (F.logsigmoid(-flat), F.logsigmoid(flat)), dim=1)
            else:
                student_log_probs = F.log_softmax(logits_s_temp, dim=1)

            # Distillation loss => KLDiv
            loss_distill = F.kl_div(
                student_log_probs,
                teacher_probs_ensemble,
                reduction='batchmean'
            ) * (temperature ** 2)

            # Real label loss => cross-entropy
            # We can compute this on the raw logits or scaled. Typically raw logits is standard:
            if num_classes == 1:
                loss_ce = F.binary_cross_entropy_with_logits(
                    logits_s.reshape(-1), labels.float().reshape(-1))
            else:
                loss_ce = F.cross_entropy(logits_s, labels.long().reshape(-1))

            # Weighted sum
            loss = alpha * loss_ce + (1 - alpha) * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    save_model_artifact(
        student_model, student_save_path,
        optimizer=optimizer, epoch=epochs,
        metrics={'distillation_loss': float(avg_loss)},
        artifact_role='knowledge_distillation')
    print(f"Knowledge-distilled student saved to: {student_save_path}")

    return student_model
            
def model_fusion(model_paths,save_path,device='cpu',model_name='maxvit_t',pretrained=True,dropout_rate=None,use_checkpoint=False,aggregator='mean'):
    """Fuse the weights of several identically-shaped model checkpoints into one.

    :param model_paths: Paths to source checkpoints (dicts or ``TorchModel`` objects).
    :param save_path: Base output path; suffix ``_<aggregator>.pth`` is
        appended.
    :param device: Torch device string. Default ``'cpu'``.
    :param model_name: TorchModel architecture name for the fused model.
    :param pretrained: Whether pretrained weights are expected.
    :param dropout_rate: Optional dropout rate.
    :param use_checkpoint: Whether to enable gradient checkpointing.
    :param aggregator: Reduction over stacked weights — one of
        ``'mean'``, ``'geomean'``, ``'median'``, ``'sum'``, ``'max'``,
        ``'min'``. Default ``'mean'``.
    :returns: The fused ``TorchModel``.
    :raises ValueError: on unsupported ``aggregator``, mismatched state
        dict keys, or unsupported checkpoint types.
    """
    if not model_paths:
        raise ValueError("model_paths must contain at least one checkpoint.")
    if save_path.endswith('.pth'):
        save_path_part1, ext = os.path.splitext(save_path)
    else:
        save_path_part1 = save_path
    
    save_path = save_path_part1 + f'_{aggregator}.pth'

    valid_aggregators = {'mean', 'geomean', 'median', 'sum', 'max', 'min'}
    if aggregator not in valid_aggregators:
        raise ValueError(f"Invalid aggregator '{aggregator}'. "
                         f"Must be one of {valid_aggregators}.")

    # --- 1. Load the first checkpoint to figure out architecture & hyperparams ---
    print(f"Loading the first model from: {model_paths[0]} to derive architecture")
    try:
        fused_model, first_metadata = load_model_artifact(
            model_paths[0], map_location=device)
    except ValueError as exc:
        raise ValueError(
            "Unsupported checkpoint format. Must be a spaCR artifact, "
            "legacy state dict, or TorchModel instance.") from exc
    fused_model = fused_model.to(device)
    state_dicts = [fused_model.state_dict()]

    # --- 2. Load the rest of the checkpoints ---
    for path in model_paths[1:]:
        print(f"Loading model from: {path}")
        try:
            loaded, _ = load_model_artifact(path, map_location=device)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Unsupported checkpoint format in {path}; it must be a "
                "spaCR artifact, dict or TorchModel with an identical "
                "architecture.") from exc
        state_dicts.append(loaded.state_dict())

    # --- 3. Verify all state dicts have the same keys ---
    fused_sd = fused_model.state_dict()
    for sd in state_dicts:
        if fused_sd.keys() != sd.keys() or any(
                fused_sd[key].shape != sd[key].shape for key in fused_sd):
            raise ValueError("All models must have identical architecture/state_dict keys.")

    # --- 4. Define aggregator logic ---
    def combine_tensors(tensor_list, mode='mean'):
        """Given a list of Tensors, combine them using the chosen aggregator."""
        # stack along new dimension => shape (num_models, *tensor.shape)
        first = tensor_list[0]
        if not first.is_floating_point() and not first.is_complex():
            # Counters such as BatchNorm.num_batches_tracked are state, not
            # learnable weights. Combining them numerically corrupts their
            # meaning, so retain the first compatible model's value.
            return first.clone()
        stacked = torch.stack(
            [tensor.to(device=first.device, dtype=torch.float64)
             for tensor in tensor_list], dim=0)

        if mode == 'mean':
            combined = stacked.mean(dim=0)
        elif mode == 'geomean':
            # Neural weights are signed. Use a signed geometric mean of
            # magnitudes and preserve the sign of the arithmetic mean.
            zero = (stacked == 0).any(dim=0)
            magnitude = torch.exp(
                torch.log(stacked.abs().clamp_min(torch.finfo(stacked.dtype).tiny))
                .mean(dim=0))
            combined = torch.sign(stacked.mean(dim=0)) * magnitude
            combined = torch.where(zero, torch.zeros_like(combined), combined)
        elif mode == 'median':
            combined = stacked.median(dim=0).values
        elif mode == 'sum':
            combined = stacked.sum(dim=0)
        elif mode == 'max':
            combined = stacked.max(dim=0).values
        elif mode == 'min':
            combined = stacked.min(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregator: {mode}")
        return combined.to(dtype=first.dtype)

    # --- 5. Combine the weights ---
    for key in fused_sd.keys():
        # gather all versions of this tensor
        all_tensors = [sd[key] for sd in state_dicts]
        fused_sd[key] = combine_tensors(all_tensors, mode=aggregator)

    # Load combined weights into the fused model
    fused_model.load_state_dict(fused_sd)

    save_model_artifact(
        fused_model, save_path,
        metrics={'aggregator': str(aggregator),
                 'source_models': len(model_paths)},
        artifact_role='model_fusion')
    print(f"Fused model (aggregator='{aggregator}') saved to: {save_path}")

    return fused_model

def annotate_filter_vision(settings):
    """Annotate and filter vision-model score CSVs, then optionally drop training images.

    For every ``src`` CSV the plate metadata is corrected and annotated, rows
    whose ``filter_column`` value lies between ``lower_threshold`` and
    ``upper_threshold`` are dropped, and the result is written to
    ``<src>_annotated_filtered.csv``. Only afterwards, and only when
    ``remove_train`` is set, are rows matching a PNG under the sibling
    ``datasets/training/train`` folders removed and the same CSV rewritten.

    :param settings: Settings dict with ``src`` (path or list of paths); the
        ``annotate_conditions`` keys ``cells``, ``cell_loc``, ``pathogens``,
        ``pathogen_loc``, ``treatments`` and ``treatment_loc``; the filter keys
        ``filter_column`` (``None``, or a name absent from the frame, leaves it
        unfiltered), ``upper_threshold`` and ``lower_threshold``; and
        ``remove_train``.
    :returns: None
    """
    from .utils import annotate_conditions, correct_metadata

    def filter_csv_by_png(csv_file):
        """Return a DataFrame with rows matching any PNG in the sibling ``training/train`` folders removed.

        :param csv_file: Path to the score CSV.
        :returns: Filtered DataFrame.
        """
        # Split the path to identify the datasets folder and build the training folder path.
        # Unpacking the split into two names raised a bare "not enough values to
        # unpack" for any CSV outside a '.../datasets/...' tree; say what is wrong
        # instead, since remove_train cannot locate the training images without it.
        marker = os.sep + "datasets" + os.sep
        if marker not in csv_file:
            raise ValueError(
                f"remove_train=True needs the score CSV to sit inside a "
                f"'...{marker}...' folder so the training images can be found, "
                f"but got: {csv_file}")
        before_datasets = csv_file.split(marker, 1)[0]
        train_fldr = os.path.join(before_datasets, 'datasets', 'training', 'train')

        # Paths for train/nc and train/pc
        nc_folder = os.path.join(train_fldr, 'nc')
        pc_folder = os.path.join(train_fldr, 'pc')

        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Collect PNG filenames from train/nc and train/pc
        png_files = set()
        for folder in [nc_folder, pc_folder]:
            if os.path.exists(folder):  # Ensure the folder exists
                png_files.update({file for file in os.listdir(folder) if file.endswith(".png")})

        # Filter the DataFrame by excluding rows where filenames match PNG files
        filtered_df = df[~df['path'].isin(png_files)]

        return filtered_df
    
    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]
    
    for src in settings['src']:
        ann_src, ext = os.path.splitext(src)
        output_csv = ann_src+'_annotated_filtered.csv'
        print(output_csv)

        df = pd.read_csv(src)
        
        df = correct_metadata(df)
            
        df = annotate_conditions(df, 
                            cells=settings['cells'],
                            cell_loc=settings['cell_loc'],
                            pathogens=settings['pathogens'],
                            pathogen_loc=settings['pathogen_loc'],
                            treatments=settings['treatments'],
                            treatment_loc=settings['treatment_loc'])
        
        if not settings['filter_column'] is None:
            if settings['filter_column'] in df.columns:
                filtered_df = df[(df[settings['filter_column']] > settings['upper_threshold']) | (df[settings['filter_column']] < settings['lower_threshold'])]
                print(f'Filtered DataFrame with {len(df)} rows to {len(filtered_df)} rows.')
            else:
                print(f"{settings['filter_column']} not in DataFrame columns.")
                filtered_df = df
        else:
            filtered_df = df
                
        filtered_df.to_csv(output_csv, index=False)
        
        if settings['remove_train']:
            df = filter_csv_by_png(output_csv)
            df.to_csv(output_csv, index=False)
