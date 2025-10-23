import os, torch, time, gc, datetime
torch.backends.cudnn.benchmark = True
import numpy as np
import pandas as pd
from torch.optim import Adagrad, AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve
from IPython.display import display
from multiprocessing import cpu_count
import torch.optim as optim

from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix
    

from torchvision import transforms
from torch.utils.data import DataLoader

def apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, n_jobs=10):
    
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
    
    model = torch.load(model_path)
    print(model)
    
    print(f'Loading dataset in {src} with {len(src)} images')
    dataset = NoClassDataset(data_dir=src, transform=transform, shuffle=True, load_to_memory=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
    print(f'Loaded {len(src)} images')
    
    result_loc = os.path.splitext(model_path)[0]+datetime.date.today().strftime('%y%m%d')+'_'+os.path.splitext(model_path)[1]+'_test_result.csv'
    print(f'Results wil be saved in: {result_loc}')
    
    model.eval()
    model = model.to(device)
    prediction_pos_probs = []
    filenames_list = []
    time_ls = []
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            start = time.time()
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed = batch_idx*batch_size
            files_to_process = len(data_loader)
            print_progress(files_processed, files_to_process, n_jobs=n_jobs, time_ls=time_ls, batch_size=batch_size, operation_type="Generating predictions")

    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df.to_csv(result_loc, index=True, header=True, mode='w')
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    return df

def apply_model_to_tar(settings={}):
    
    from .io import TarImageDataset
    from .utils import process_vision_results, print_progress

    #if os.path.exists(settings['dataset']):
    #    tar_path = settings['dataset']
    #else:
    #    tar_path = os.path.join(settings['src'], 'datasets', settings['dataset'])
    tar_path = settings['tar_path']
    model_path = settings['model_path']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if settings['normalize']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(settings['image_size'], settings['image_size'])),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(settings['image_size'], settings['image_size']))])
    
    if settings['verbose']:
        print(f"Loading model from {model_path}")
        print(f"Loading dataset from {tar_path}")
        
    model = torch.load(settings['model_path'])
    
    dataset = TarImageDataset(tar_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True, num_workers=settings['n_jobs'], pin_memory=True)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    #dataset_name = os.path.splitext(os.path.basename(settings['dataset']))[0] 
    dataset_name = os.path.splitext(os.path.basename(settings['tar_path']))[0] 
    date_name = datetime.date.today().strftime('%y%m%d')
    dst = os.path.dirname(tar_path)
    result_loc = f'{dst}/{date_name}_{dataset_name}_{model_name}_result.csv'

    model.eval()
    model = model.to(device)
    
    if settings['verbose']:
        print(model)
        print(f'Generated dataset with {len(dataset)} images')
        print(f'Generating loader from {len(data_loader)} batches')
        print(f'Results wil be saved in: {result_loc}')
        print(f'Model is in eval mode')
        print(f'Model loaded to device')
        
    prediction_pos_probs = []
    filenames_list = []
    time_ls = []
    gc.collect()
    with torch.no_grad():
        for batch_idx, (batch_images, filenames) in enumerate(data_loader, start=1):
            start = time.time()
            images = batch_images.to(torch.float).to(device)
            outputs = model(images)
            batch_prediction_pos_prob = torch.sigmoid(outputs).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            filenames_list.extend(filenames)
            stop = time.time()
            duration = stop - start
            time_ls.append(duration)
            files_processed = batch_idx*settings['batch_size']
            files_to_process = len(data_loader)*settings['batch_size']
            print_progress(files_processed, files_to_process, n_jobs=settings['n_jobs'], time_ls=time_ls, batch_size=settings['batch_size'], operation_type="Tar dataset")

    data = {'path':filenames_list, 'pred':prediction_pos_probs}
    df = pd.DataFrame(data, index=None)
    df = process_vision_results(df, settings['score_threshold'])

    df.to_csv(result_loc, index=True, header=True, mode='w')
    print(f"Saved results to {result_loc}")
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
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
    }

def _multiclass_metrics(y_true: np.ndarray, prob_mat: np.ndarray) -> dict:
    """
    Metrics for multiclass (single-label):
    - overall accuracy
    - per-class accuracy (weighted by support)
    - macro average precision (one-vs-rest)
    """
    preds = prob_mat.argmax(axis=1)
    acc = (preds == y_true).mean() if len(y_true) else np.nan

    # Per-class (diagonal / row sum)
    cm = confusion_matrix(y_true, preds, labels=np.arange(prob_mat.shape[1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1, where=(cm.sum(axis=1) != 0), initial=1)
    # Average precision macro (one-vs-rest)
    # Build one-hot y_true
    C = prob_mat.shape[1]
    y_true_oh = np.zeros((len(y_true), C), dtype=int)
    if len(y_true):
        y_true_oh[np.arange(len(y_true)), y_true] = 1
    try:
        ap_macro = average_precision_score(y_true_oh, prob_mat, average="macro")
    except Exception:
        ap_macro = np.nan

    # For compatibility with your logging keys:
    return {
        "accuracy": float(acc),
        "neg_accuracy": np.nan,  # not meaningful in multiclass
        "pos_accuracy": np.nan,  # not meaningful in multiclass
        "prauc": float(ap_macro),  # reuse key for macro-AP
        "optimal_threshold": np.nan,
        "per_class_accuracy": per_class_acc.tolist(),
        "num_classes": int(C),
    }

def evaluate_model_performance(model, loader, epoch, loss_type):
    """
    Evaluates performance for binary or multiclass models.

    Returns:
        data_dict (dict): metrics + loss + epoch
        [prediction_probs, all_labels] where:
           - binary: prediction_probs is shape (N,) (positive class probabilities)
           - multiclass: prediction_probs is shape (N, C) (softmax probabilities)
    """
    # lazy import to use your loss selection
    from .utils import calculate_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    # For binary we store a vector; for multiclass we store rows we’ll stack
    prob_bucket = []

    with torch.no_grad():
        for data, target, _ in loader:
            data = data.to(device)
            target = target.to(device)
            logits = model(data)

            # accumulate loss correctly (sum over batch; normalize later)
            batch_size = data.size(0)
            loss = calculate_loss(logits, target, prefer_focal=True)
            #loss = calculate_loss(logits, target, loss_type=loss_type)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            # collect labels (as class ids)
            all_labels.extend(_to_numpy_labels(target))

            # probs
            if logits.ndim == 1 or logits.size(-1) == 1:
                # binary
                probs = torch.sigmoid(logits.view(-1))
                prob_bucket.append(probs.detach().cpu().numpy())
            else:
                # multiclass
                probs = torch.softmax(logits, dim=1)
                prob_bucket.append(probs.detach().cpu().numpy())

    mean_loss = total_loss / max(1, total_samples)
    y_true = np.asarray(all_labels, dtype=int)

    # stack probabilities
    probs_np = np.concatenate(prob_bucket, axis=0) if len(prob_bucket) else (
        np.empty((0,)) if (logits.ndim == 1 or logits.size(-1) == 1) else np.empty((0, logits.size(-1)))
    )

    # metrics
    if probs_np.ndim == 1:
        metrics = _binary_metrics(y_true, probs_np)
    else:
        metrics = _multiclass_metrics(y_true, probs_np)

    metrics["loss"] = float(mean_loss)
    metrics["epoch"] = int(epoch)
    # Backward-compat key casing
    metrics["Accuracy"] = metrics["accuracy"]

    return metrics, [probs_np, y_true.tolist()]

def test_model_core(model, loader, loader_name, epoch, loss_type):
    """
    Core test loop returning both summary metrics and a row-per-image dataframe,
    compatible with binary & multiclass.
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
    Wrapper kept for API compatibility with your caller.
    Returns (summary_metrics_dataframe, per_file_results_dataframe)
    """
    start_time = time.time()

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

def train_test_model(settings):
    from .io import _copy_missclassified
    from .utils import pick_best_model, save_settings
    from .io import generate_loaders
    from .settings import get_train_test_model_settings

    settings = get_train_test_model_settings(settings)

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()

    src = settings['src']

    channels_str = ''.join(settings['train_channels'])
    dst = os.path.join(src,'model', settings['model_type'], channels_str, str(f"epochs_{settings['epochs']}"))
    os.makedirs(dst, exist_ok=True)
    settings['dst'] = dst

    # NEW: number of classes
    num_classes = len(settings.get('classes', [])) if settings.get('classes') else 0
    if num_classes <= 0:
        raise ValueError("No classes provided in settings['classes'].")

    # NEW: pick loss automatically if desired
    if settings.get('loss_type') in (None, 'auto'):
        # Multiclass => cross entropy, Binary (1 class folder would be odd; 2 => CE works well)
        settings['loss_type'] = 'cross_entropy' if num_classes > 1 else 'binary_cross_entropy_with_logits'

    if settings['train']:
        if settings['train'] and settings['test']:
            save_settings(settings, name=f"train_test_{settings['model_type']}_{settings['epochs']}", show=True)
        elif settings['train'] is True:
            save_settings(settings, name=f"train_{settings['model_type']}_{settings['epochs']}", show=True)
        elif settings['test'] is True:
            save_settings(settings, name=f"test_{settings['model_type']}_{settings['epochs']}", show=True)

    if settings['train']:
        train, val, _  = generate_loaders(
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
            verbose=settings['verbose']
        )

        model, model_path = train_model(
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
            gradient_accumulation=settings['gradient_accumulation'],
            gradient_accumulation_steps=settings['gradient_accumulation_steps'],
            channels=settings['train_channels'],
            num_classes=num_classes  # <-- NEW
        )

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

        if model is None:
            model_path = pick_best_model(src+'/model')
            print(f'Best model: {model_path}')
            model = torch.load(model_path, map_location=lambda storage, loc: storage)

        model_fldr = dst
        time_now = datetime.date.today().strftime('%y%m%d')
        result_loc = f"{model_fldr}/{settings['model_type']}_time_{time_now}_test_result.csv"
        acc_loc = f"{model_fldr}/{settings['model_type']}_time_{time_now}_test_acc.csv"
        print(f'Results wil be saved in: {result_loc}')

        result, accuracy = test_model_performance(loaders=test,
                                                  model=model,
                                                  loader_name_list='test',
                                                  epoch=1,
                                                  loss_type=settings['loss_type'])

        result.to_csv(result_loc, index=True, header=True, mode='w')
        accuracy.to_csv(acc_loc, index=True, header=True, mode='w')
        _copy_missclassified(accuracy)

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()

    if settings['train']:
        return model_path
    if settings['test']:
        return result_loc
    
def train_model(dst, model_type, train_loaders, epochs=100, learning_rate=0.0001,
                weight_decay=0.05, amsgrad=False, optimizer_type='adamw',
                use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None,
                test_loaders=None, init_weights='imagenet', intermedeate_save=None,
                chan_dict=None, schedule=None, loss_type='binary_cross_entropy_with_logits',
                gradient_accumulation=False, gradient_accumulation_steps=4,
                channels=['r','g','b'], verbose=False, num_classes=2):
    """
    Trains a model (now supports multiclass).
    """
    from .io import _save_model, _save_progress
    from .utils import calculate_loss, choose_model

    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders) if val_loaders else 0}')
    if test_loaders is not None:
        print(f'Test batches:{len(test_loaders)}')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using {device} for Torch')

    model = choose_model(model_type, device, init_weights, dropout_rate,
                         use_checkpoint, verbose=verbose, num_classes=num_classes)  # <-- pass num_classes
    if model is None:
        print(f'Model {model_type} not found')
        return

    print(f'Loading Model to {device}...')
    model.to(device)

    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                          weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer_type == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    # schedulers
    if schedule == 'step_lr':
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs/5)), gamma=0.75)
    elif schedule == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.1, patience=10, verbose=True)
    else:
        scheduler = None

    time_ls = []
    accumulated_train_dicts, accumulated_val_dicts, accumulated_test_dicts = [], [], []

    print(f'Training ...')
    for epoch in range(1, epochs+1):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        if gradient_accumulation:
            optimizer.zero_grad()

        for batch_idx, (data, target, filenames) in enumerate(train_loaders, start=1):
            data = data.to(device)

            # --- target dtype per loss ---
            if loss_type in ('cross_entropy', 'ce'):
                # integer class indices [0..C-1]
                target = target.to(device).long()
            else:
                # BCE-style: float targets; if your model outputs (N,1), squeeze for convenience
                target = target.to(device).float()

            output = model(data)

            # Optional shape checks for clarity
            if loss_type in ('cross_entropy', 'ce'):
                # expect (N, C)
                if output.ndim == 1:
                    output = output.unsqueeze(1)
                assert output.ndim == 2 and output.size(1) == num_classes, \
                    f"Cross-entropy requires (N,{num_classes}) logits, got {tuple(output.shape)}"
            else:
                # BCE with logits: allow (N,) or (N,1)
                if output.ndim == 2 and output.size(1) == 1:
                    output = output.squeeze(1)

            loss = calculate_loss(output, target, prefer_focal=True)
            #loss = calculate_loss(output, target, loss_type=loss_type)

            if gradient_accumulation:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if not gradient_accumulation or (batch_idx % gradient_accumulation_steps == 0):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        # Epoch end: evaluate
        train_time = time.time() - start_time
        train_dict, _ = evaluate_model_performance(model, train_loaders, epoch, loss_type=loss_type)
        train_dict['train_time'] = train_time
        accumulated_train_dicts.append(train_dict)

        if val_loaders is not None and len(val_loaders) > 0:
            val_dict, _ = evaluate_model_performance(model, val_loaders, epoch, loss_type=loss_type)
            accumulated_val_dicts.append(val_dict)
            if schedule == 'reduce_lr_on_plateau':
                scheduler.step(val_dict['loss'])

            # Robust printing: fall back if binary-only keys aren’t present
            print(f"Progress: {train_dict.get('epoch', epoch)}/{epochs}, operation_type: Training, "
                  f"Train Loss: {train_dict.get('loss', float('nan')):.3f}, "
                  f"Val Loss: {val_dict.get('loss', float('nan')):.3f}, "
                  f"Train acc.: {train_dict.get('accuracy', float('nan')):.3f}, "
                  f"Val acc.: {val_dict.get('accuracy', float('nan')):.3f}, "
                  f"Train F1(macro): {train_dict.get('f1_macro', float('nan')):.3f}, "
                  f"Val F1(macro): {val_dict.get('f1_macro', float('nan')):.3f}")
        else:
            print(f"Progress: {train_dict.get('epoch', epoch)}/{epochs}, operation_type: Training, "
                  f"Train Loss: {train_dict.get('loss', float('nan')):.3f}, "
                  f"Train acc.: {train_dict.get('accuracy', float('nan')):.3f}, "
                  f"Train F1(macro): {train_dict.get('f1_macro', float('nan')):.3f}")

        if scheduler and schedule == 'step_lr':
            scheduler.step()

        # Save rolling CSVs
        if accumulated_train_dicts and accumulated_val_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_train_dicts), pd.DataFrame(accumulated_val_dicts))
            accumulated_train_dicts, accumulated_val_dicts = [], []
        elif accumulated_train_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_train_dicts), None)
            accumulated_train_dicts = []
        elif accumulated_test_dicts:
            _save_progress(dst, pd.DataFrame(accumulated_test_dicts), None)
            accumulated_test_dicts = []

        # Save checkpoints
        model_path = _save_model(model, model_type, train_dict, dst, epoch, epochs,
                        intermedeate_save=[0.99,0.98,0.95,0.94], channels=channels)

    return model, model_path#os.path.join(dst, 'best.pt')  # or whatever _save_model returns    

def generate_activation_map(settings):
    
    from .utils import SaliencyMapGenerator, GradCAMGenerator, SelectChannels, activation_maps_to_database, activation_correlations_to_database
    from .utils import print_progress, save_settings, calculate_activation_correlations
    from .io import TarImageDataset
    from .settings import get_default_generate_activation_map_settings
    
    torch.cuda.empty_cache()
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
    
    # Set number of jobs for loading
    n_jobs = settings['n_jobs']
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 4)

    # Set transforms for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(size=(settings['image_size'], settings['image_size'])),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if settings['normalize_input'] else None,
        SelectChannels(settings['channels'])
    ])

    # Handle dataset path
    if not os.path.exists(settings['dataset']):
        print(f"Dataset not found at {settings['dataset']}")
        return

    # Load the model
    model = torch.load(settings['model_path'])
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
    data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=settings['shuffle'], num_workers=n_jobs, pin_memory=True)

    # Initialize generator based on cam_type
    if settings['cam_type'] in ['gradcam', 'gradcam_pp']:
        cam_generator = GradCAMGenerator(model, target_layer=settings['target_layer'], cam_type=settings['cam_type'])
    elif settings['cam_type'] in ['saliency_image', 'saliency_channel']:
        cam_generator = SaliencyMapGenerator(model)
        
    time_ls = []
    for batch_idx, (inputs, filenames) in enumerate(data_loader):
        start = time.time()
        img_paths = []
        inputs = inputs.to(device)

        # Compute activation maps and predictions
        if settings['cam_type'] in ['gradcam', 'gradcam_pp']:
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
            fig.savefig(pdf_save_path, format='pdf')
            print(f"Saved batch grid to {pdf_save_path}")
            #plt.show()
            display(fig)
                    
        for i in range(inputs.size(0)):
            activation_map = activation_maps[i].detach().numpy()

            if settings['cam_type'] in ['saliency_image', 'gradcam', 'gradcam_pp']:
                #activation_map = activation_map.sum(axis=0) 
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
                activation_map = (activation_map * 255).astype(np.uint8)
                activation_image = Image.fromarray(activation_map, mode='L')

            elif settings['cam_type'] == 'saliency_channel':
                # Handle each channel separately and save as RGB
                rgb_activation_map = np.zeros((activation_map.shape[1], activation_map.shape[2], 3), dtype=np.uint8)
                for c in range(min(activation_map.shape[0], 3)):  # Limit to 3 channels for RGB
                    channel_map = activation_map[c]
                    channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min())
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

    torch.cuda.empty_cache()
    gc.collect()
    print("Activation map generation complete.")

def visualize_classes(model, dtype, class_names, **kwargs):

    from .utils import class_visualization

    for target_y in range(2):  # Assuming binary classification
        print(f"Visualizing class: {class_names[target_y]}")
        visualization = class_visualization(target_y, model, dtype, **kwargs)
        plt.imshow(visualization)
        plt.title(f"Class {class_names[target_y]} Visualization")
        plt.axis('off')
        plt.show()

def visualize_integrated_gradients(src, model_path, target_label_idx=0, image_size=224, channels=[1,2,3], normalize=True, save_integrated_grads=False, save_dir='integrated_grads'):

    from .utils import IntegratedGradients, preprocess_image

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(model_path)
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
        overlay = np.array(image)
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
    def __init__(self, model, n_samples=50, stdev_spread=0.15):
        self.model = model
        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def compute_smooth_grad(self, input_tensor, target_class):
        self.model.eval()
        stdev = self.stdev_spread * (input_tensor.max() - input_tensor.min())
        total_gradients = torch.zeros_like(input_tensor)
        
        for i in range(self.n_samples):
            noise = torch.normal(mean=0, std=stdev, size=input_tensor.shape).to(input_tensor.device)
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_()
            output = self.model(noisy_input)
            self.model.zero_grad()
            output[0, target_class].backward()
            total_gradients += noisy_input.grad

        avg_gradients = total_gradients / self.n_samples
        return avg_gradients.abs()

def visualize_smooth_grad(src, model_path, target_label_idx, image_size=224, channels=[1,2,3], normalize=True, save_smooth_grad=False, save_dir='smooth_grad'):

    from .utils import preprocess_image

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(model_path)
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
        overlay = np.array(image)
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

def deep_spacr_v1(settings={}):
    from .settings import deep_spacr_defaults
    from .io import generate_training_dataset, generate_dataset
    from .utils import save_settings
    
    settings = deep_spacr_defaults(settings)
    src = settings['src']

    save_settings(settings, name='DL_model')
    
    if settings['train'] or settings['test']:
        if settings['generate_training_dataset']:
            print(f"Generating train and test datasets ...")
            train_path, test_path = generate_training_dataset(settings)
            print(f'Generated Train set: {train_path}')
            print(f'Generated Test set: {test_path}')
            settings['src'] = os.path.dirname(train_path)
    
    if settings['train_DL_model']:
        print(f"Training model ...")
        model_path = train_test_model(settings)
        settings['model_path'] = model_path
        settings['src'] = src
        
    if settings['apply_model_to_dataset']:
        if not settings['tar_path'] and os.path.isabs(settings['tar_path']) and os.path.exists(settings['tar_path']):
            print(f"{settings['tar_path']} not found generating dataset ...")
            tar_path = generate_dataset(settings)
            settings['tar_path'] = tar_path
            
        if os.path.exists(settings['model_path']):
            apply_model_to_tar(settings)
            
def deep_spacr(settings={}):
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

    # 3) train DL model
    if settings.get('train_DL_model'):
        print("Training model ...")
        model_path = train_test_model(settings)  # must exist in your codebase
        settings['model_path'] = model_path
        # restore original src (so later steps like apply can use the user’s dataset if needed)
        settings['src'] = src_before

    # 4) apply model to dataset/tar
    if settings.get('apply_model_to_dataset'):
        tar_path = settings.get('tar_path')

        # if tar_path missing OR invalid, (re)generate it
        if not tar_path or not os.path.isabs(tar_path) or not os.path.exists(tar_path):
            print("tar_path not valid/found; generating dataset tar ...")
            tar_path = generate_dataset(settings)
            settings['tar_path'] = tar_path

        model_path = settings.get('model_path')
        if model_path and os.path.exists(model_path):
            apply_model_to_tar(settings)  # must exist in your codebase
        else:
            print("Model path missing or not found; skipping model application.")
            
def model_knowledge_transfer(teacher_paths, student_save_path, data_loader, device='cpu', student_model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, alpha=0.5, temperature=2.0, lr=1e-4, epochs=10):

    from .utils import TorchModel

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
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, TorchModel):
            teacher = ckpt.to(device)
        elif isinstance(ckpt, dict):
            # If it's a dict with 'model' inside
            # We might need to check if it has 'model_name', etc. 
            # But let's keep it simple: same architecture as the student
            teacher = TorchModel(
                model_name=ckpt.get('model_name', student_model_name),
                pretrained=ckpt.get('pretrained', pretrained),
                dropout_rate=ckpt.get('dropout_rate', dropout_rate),
                use_checkpoint=ckpt.get('use_checkpoint', use_checkpoint)
            ).to(device)
            teacher.load_state_dict(ckpt['model'])
        else:
            raise ValueError(f"Unsupported checkpoint type at {path} (must be TorchModel or dict).")

        teacher.eval()  # For consistent batchnorm, dropout
        teachers.append(teacher)

    # -- 2. Initialize the student TorchModel --
    student_model = TorchModel(
        model_name=student_model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_checkpoint=use_checkpoint
    ).to(device)

    # You could load a partial checkpoint into the student here if desired.

    # -- 3. Optimizer --
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # Distillation training loop
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass student
            logits_s = student_model(images)         # shape: (B, num_classes)
            logits_s_temp = logits_s / temperature   # scale by T

            # Distillation from teachers
            with torch.no_grad():
                # We'll average teacher probabilities
                teacher_probs_list = []
                for tm in teachers:
                    logits_t = tm(images) / temperature
                    # convert to probabilities
                    teacher_probs_list.append(F.softmax(logits_t, dim=1))
                # average them
                teacher_probs_ensemble = torch.mean(torch.stack(teacher_probs_list), dim=0)

            # Student probabilities (log-softmax)
            student_log_probs = F.log_softmax(logits_s_temp, dim=1)

            # Distillation loss => KLDiv
            loss_distill = F.kl_div(
                student_log_probs,
                teacher_probs_ensemble,
                reduction='batchmean'
            ) * (temperature ** 2)

            # Real label loss => cross-entropy
            # We can compute this on the raw logits or scaled. Typically raw logits is standard:
            loss_ce = F.cross_entropy(logits_s, labels)

            # Weighted sum
            loss = alpha * loss_ce + (1 - alpha) * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # -- 4. Save final student as a TorchModel --
    torch.save(student_model, student_save_path)
    print(f"Knowledge-distilled student saved to: {student_save_path}")

    return student_model
            
def model_fusion(model_paths,save_path,device='cpu',model_name='maxvit_t',pretrained=True,dropout_rate=None,use_checkpoint=False,aggregator='mean'):

    from .utils import TorchModel
    
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
    first_ckpt = torch.load(model_paths[0], map_location=device)

    if isinstance(first_ckpt, dict):
        # It's a dict with state_dict + possibly metadata
        # Use any stored metadata if present
        model_name = first_ckpt.get('model_name', model_name)
        pretrained = first_ckpt.get('pretrained', pretrained)
        dropout_rate = first_ckpt.get('dropout_rate', dropout_rate)
        use_checkpoint = first_ckpt.get('use_checkpoint', use_checkpoint)

        # Initialize the fused model
        fused_model = TorchModel(
            model_name=model_name,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_checkpoint=use_checkpoint
        ).to(device)

        # We'll collect state dicts in a list
        state_dicts = [first_ckpt['model']]  # the actual weights
    elif isinstance(first_ckpt, TorchModel):
        # The checkpoint is directly a TorchModel instance
        fused_model = first_ckpt.to(device)
        state_dicts = [fused_model.state_dict()]
    else:
        raise ValueError("Unsupported checkpoint format. Must be a dict or a TorchModel instance.")

    # --- 2. Load the rest of the checkpoints ---
    for path in model_paths[1:]:
        print(f"Loading model from: {path}")
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict):
            state_dicts.append(ckpt['model'])  # Just the state dict portion
        elif isinstance(ckpt, TorchModel):
            state_dicts.append(ckpt.state_dict())
        else:
            raise ValueError(f"Unsupported checkpoint format in {path} (must be dict or TorchModel).")

    # --- 3. Verify all state dicts have the same keys ---
    fused_sd = fused_model.state_dict()
    for sd in state_dicts:
        if fused_sd.keys() != sd.keys():
            raise ValueError("All models must have identical architecture/state_dict keys.")

    # --- 4. Define aggregator logic ---
    def combine_tensors(tensor_list, mode='mean'):
        """Given a list of Tensors, combine them using the chosen aggregator."""
        # stack along new dimension => shape (num_models, *tensor.shape)
        stacked = torch.stack(tensor_list, dim=0).float()

        if mode == 'mean':
            return stacked.mean(dim=0)
        elif mode == 'geomean':
            # geometric mean = exp(mean(log(tensor))) 
            # caution: requires all > 0
            return torch.exp(torch.log(stacked).mean(dim=0))
        elif mode == 'median':
            return stacked.median(dim=0).values
        elif mode == 'sum':
            return stacked.sum(dim=0)
        elif mode == 'max':
            return stacked.max(dim=0).values
        elif mode == 'min':
            return stacked.min(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregator: {mode}")

    # --- 5. Combine the weights ---
    for key in fused_sd.keys():
        # gather all versions of this tensor
        all_tensors = [sd[key] for sd in state_dicts]
        fused_sd[key] = combine_tensors(all_tensors, mode=aggregator)

    # Load combined weights into the fused model
    fused_model.load_state_dict(fused_sd)

    # --- 6. Save the entire TorchModel object ---
    torch.save(fused_model, save_path)
    print(f"Fused model (aggregator='{aggregator}') saved as a full TorchModel to: {save_path}")

    return fused_model

def annotate_filter_vision(settings):
    
    from .utils import annotate_conditions, correct_metadata
    
    def filter_csv_by_png(csv_file):
        """
        Filters a DataFrame by removing rows that match PNG filenames in a folder.

        Parameters:
            csv_file (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        # Split the path to identify the datasets folder and build the training folder path
        before_datasets, after_datasets = csv_file.split(os.sep + "datasets" + os.sep, 1)
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
