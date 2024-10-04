import os, torch, time, gc, datetime
torch.backends.cudnn.benchmark = True

import numpy as np
import pandas as pd
from torch.optim import Adagrad, AdamW
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve

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
        print(f"Loading model from {settings['model_path']}")
        print(f"Loading dataset from {settings['tar_path']}")
        
    model = torch.load(settings['model_path'])
    
    dataset = TarImageDataset(settings['tar_path'], transform=transform)
    data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True, num_workers=settings['n_jobs'], pin_memory=True)
    
    model_name = os.path.splitext(os.path.basename(settings['model_path']))[0] 
    dataset_name = os.path.splitext(os.path.basename(settings['tar_path']))[0]  
    date_name = datetime.date.today().strftime('%y%m%d')
    dst = os.path.dirname(settings['tar_path'])
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

def evaluate_model_performance(model, loader, epoch, loss_type):
    """
    Evaluates the performance of a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader to evaluate the model on.
        loader_name (str): The name of the data loader.
        epoch (int): The current epoch number.
        loss_type (str): The type of loss function to use.

    Returns:
        data_df (pandas.DataFrame): The classification metrics data as a DataFrame.
        prediction_pos_probs (list): The positive class probabilities for each prediction.
        all_labels (list): The true labels for each prediction.
    """
    
    from .utils import calculate_loss
    
    def classification_metrics(all_labels, prediction_pos_probs):
        """
        Calculate classification metrics for binary classification.

        Parameters:
        - all_labels (list): List of true labels.
        - prediction_pos_probs (list): List of predicted positive probabilities.
        - loader_name (str): Name of the data loader.

        Returns:
        - data_df (DataFrame): DataFrame containing the calculated metrics.
        """

        if len(all_labels) != len(prediction_pos_probs):
            raise ValueError(f"all_labels ({len(all_labels)}) and pred_labels ({len(prediction_pos_probs)}) have different lengths")

        unique_labels = np.unique(all_labels)
        if len(unique_labels) >= 2:
            pr_labels = np.array(all_labels).astype(int)
            precision, recall, thresholds = precision_recall_curve(pr_labels, prediction_pos_probs, pos_label=1)
            pr_auc = auc(recall, precision)
            thresholds = np.append(thresholds, 0.0)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_idx = np.nanargmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            pred_labels = [int(p > 0.5) for p in prediction_pos_probs]
        if len(unique_labels) < 2:
            optimal_threshold = 0.5
            pred_labels = [int(p > optimal_threshold) for p in prediction_pos_probs]
            pr_auc = np.nan
        data = {'label': all_labels, 'pred': pred_labels}
        df = pd.DataFrame(data)
        pc_df = df[df['label'] == 1.0]
        nc_df = df[df['label'] == 0.0]
        correct = df[df['label'] == df['pred']]
        acc_all = len(correct) / len(df)
        if len(pc_df) > 0:
            correct_pc = pc_df[pc_df['label'] == pc_df['pred']]
            acc_pc = len(correct_pc) / len(pc_df)
        else:
            acc_pc = np.nan
        if len(nc_df) > 0:
            correct_nc = nc_df[nc_df['label'] == nc_df['pred']]
            acc_nc = len(correct_nc) / len(nc_df)
        else:
            acc_nc = np.nan
        data_dict = {'accuracy': acc_all, 'neg_accuracy': acc_nc, 'pos_accuracy': acc_pc, 'prauc':pr_auc, 'optimal_threshold':optimal_threshold}
        return data_dict
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0
    correct = 0
    total_samples = 0
    prediction_pos_probs = []
    all_labels = []
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(loader, start=1):
            start_time = time.time()
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            loss = calculate_loss(output, target, loss_type=loss_type)
            loss += loss.item()
            total_samples += data.size(0)
            pred = torch.where(output >= 0.5,
                               torch.Tensor([1.0]).to(device).float(),
                               torch.Tensor([0.0]).to(device).float())
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_prediction_pos_prob = torch.sigmoid(output).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            mean_loss = loss / total_samples
            acc = correct / total_samples
            end_time = time.time()
            test_time = end_time - start_time
            #print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx+1}/{len(loader)} loss: {mean_loss:.5f} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    
    loss /= len(loader)
    data_dict = classification_metrics(all_labels, prediction_pos_probs)
    data_dict['loss'] = loss
    data_dict['epoch'] = epoch
    data_dict['Accuracy'] = acc
    
    return data_dict, [prediction_pos_probs, all_labels]

def test_model_core(model, loader, loader_name, epoch, loss_type):
    
    from .utils import calculate_loss, classification_metrics
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0
    correct = 0
    total_samples = 0
    prediction_pos_probs = []
    all_labels = []
    filenames = []
    true_targets = []
    predicted_outputs = []

    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (data, target, filename) in enumerate(loader, start=1):  # Assuming loader provides filenames
            start_time = time.time()
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            loss = calculate_loss(output, target, loss_type=loss_type)
            loss += loss.item()
            total_samples += data.size(0)
            pred = torch.where(output >= 0.5,
                               torch.Tensor([1.0]).to(device).float(),
                               torch.Tensor([0.0]).to(device).float())
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_prediction_pos_prob = torch.sigmoid(output).cpu().numpy()
            prediction_pos_probs.extend(batch_prediction_pos_prob.tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            
            # Storing intermediate results in lists
            true_targets.extend(target.cpu().numpy().tolist())
            predicted_outputs.extend(pred.cpu().numpy().tolist())
            filenames.extend(filename)
            
            mean_loss = loss / total_samples
            acc = correct / total_samples
            end_time = time.time()
            test_time = end_time - start_time
            #print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx}/{len(loader)} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    
    # Constructing the DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': true_targets,
        'predicted_label': predicted_outputs,
        'class_1_probability':prediction_pos_probs})

    loss /= len(loader)
    data_df = classification_metrics(all_labels, prediction_pos_probs, loss, epoch)
    return data_df, prediction_pos_probs, all_labels, results_df

def test_model_performance(loaders, model, loader_name_list, epoch, loss_type):
    """
    Test the performance of a model on given data loaders.

    Args:
        loaders (list): List of data loaders.
        model: The model to be tested.
        loader_name_list (list): List of names for the data loaders.
        epoch (int): The current epoch.
        loss_type: The type of loss function.

    Returns:
        tuple: A tuple containing the test results and the results dataframe.
    """
    start_time = time.time()
    df_list = []

    result, prediction_pos_probs, all_labels, results_df = test_model_core(model, loaders, loader_name_list, epoch, loss_type)

    return result, results_df

def train_test_model(settings):
    
    from .io import _save_settings, _copy_missclassified
    from .utils import pick_best_model
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
    settings['src'] = src
    settings['dst'] = dst
    
    if settings['custom_model']:
        model = torch.load(settings['custom_model_path'])
    
    if settings['train']:
        _save_settings(settings, src)

    if settings['train']:
        train, val, train_fig  = generate_loaders(src, 
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
                                                  verbose=settings['verbose'])
        
        #train_batch_1_figure = os.path.join(dst, 'batch_1.pdf')
        #train_fig.savefig(train_batch_1_figure, format='pdf', dpi=300)
    
    if settings['train']:
        model, model_path = train_model(dst = settings['dst'],
                                        model_type=settings['model_type'],
                                        train_loaders = train, 
                                        epochs = settings['epochs'], 
                                        learning_rate = settings['learning_rate'],
                                        init_weights = settings['init_weights'],
                                        weight_decay = settings['weight_decay'], 
                                        amsgrad = settings['amsgrad'], 
                                        optimizer_type = settings['optimizer_type'], 
                                        use_checkpoint = settings['use_checkpoint'], 
                                        dropout_rate = settings['dropout_rate'], 
                                        n_jobs = settings['n_jobs'], 
                                        val_loaders = val, 
                                        test_loaders = None, 
                                        intermedeate_save = settings['intermedeate_save'],
                                        schedule = settings['schedule'],
                                        loss_type=settings['loss_type'], 
                                        gradient_accumulation=settings['gradient_accumulation'], 
                                        gradient_accumulation_steps=settings['gradient_accumulation_steps'],
                                        channels=settings['train_channels'])
        
    if settings['test']:
        test, _, train_fig = generate_loaders(src, 
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
                                              verbose=settings['verbose'])
        
        if model == None:
            model_path = pick_best_model(src+'/model')
            print(f'Best model: {model_path}')

            model = torch.load(model_path, map_location=lambda storage, loc: storage)

            model_type = settings['model_type']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(type(model))
            print(model)
        
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
    
def train_model(dst, model_type, train_loaders, epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule = None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4, channels=['r','g','b'], verbose=False):
    """
    Trains a model using the specified parameters.

    Args:
        dst (str): The destination path to save the model and results.
        model_type (str): The type of model to train.
        train_loaders (list): A list of training data loaders.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.0001.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.05.
        amsgrad (bool, optional): Whether to use AMSGrad for the optimizer. Defaults to False.
        optimizer_type (str, optional): The type of optimizer to use. Defaults to 'adamw'.
        use_checkpoint (bool, optional): Whether to use checkpointing during training. Defaults to False.
        dropout_rate (float, optional): The dropout rate for the model. Defaults to 0.
        n_jobs (int, optional): The number of n_jobs for data loading. Defaults to 20.
        val_loaders (list, optional): A list of validation data loaders. Defaults to None.
        test_loaders (list, optional): A list of test data loaders. Defaults to None.
        init_weights (str, optional): The initialization weights for the model. Defaults to 'imagenet'.
        intermedeate_save (list, optional): The intermediate save thresholds. Defaults to None.
        chan_dict (dict, optional): The channel dictionary. Defaults to None.
        schedule (str, optional): The learning rate schedule. Defaults to None.
        loss_type (str, optional): The loss function type. Defaults to 'binary_cross_entropy_with_logits'.
        gradient_accumulation (bool, optional): Whether to use gradient accumulation. Defaults to False.
        gradient_accumulation_steps (int, optional): The number of steps for gradient accumulation. Defaults to 4.

    Returns:
        None
    """    
    
    from .io import _save_model, _save_progress
    from .utils import calculate_loss, choose_model
    
    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders)}')
    
    if test_loaders != None:
        print(f'Test batches:{len(test_loaders)}')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f'Using {device} for Torch')
    
    kwargs = {'n_jobs': n_jobs, 'pin_memory': True} if use_cuda else {}
    
    model = choose_model(model_type, device, init_weights, dropout_rate, use_checkpoint, verbose=verbose)
    
    
    if model is None:
        print(f'Model {model_type} not found')
        return

    print(f'Loading Model to {device}...')
    model.to(device)

    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=amsgrad)
    
    if optimizer_type == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    
    if schedule == 'step_lr':
        StepLR_step_size = int(epochs/5)
        StepLR_gamma = 0.75
        scheduler = StepLR(optimizer, step_size=StepLR_step_size, gamma=StepLR_gamma)
    elif schedule == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = None

    time_ls = []

    # Initialize lists to accumulate results
    accumulated_train_dicts = []
    accumulated_val_dicts = []
    accumulated_test_dicts = []

    print(f'Training ...')
    for epoch in range(1, epochs+1):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        # Initialize gradients if using gradient accumulation
        if gradient_accumulation:
            optimizer.zero_grad()

        for batch_idx, (data, target, filenames) in enumerate(train_loaders, start=1):
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss = calculate_loss(output, target, loss_type=loss_type)
            
            # Normalize loss if using gradient accumulation
            if gradient_accumulation:
                loss /= gradient_accumulation_steps
            running_loss += loss.item() * gradient_accumulation_steps  # correct the running_loss
            loss.backward()

            # Step optimizer if not using gradient accumulation or every gradient_accumulation_steps
            if not gradient_accumulation or (batch_idx % gradient_accumulation_steps == 0):
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = running_loss / batch_idx
            batch_size = len(train_loaders)
            duration = time.time() - start_time
            time_ls.append(duration)
            #print(f'Progress: {batch_idx}/{batch_size}, operation_type: DL-Batch, Epoch {epoch}/{epochs}, Loss {avg_loss}, Time {duration}')
            
        end_time = time.time()
        train_time = end_time - start_time
        train_dict, _ = evaluate_model_performance(model, train_loaders, epoch, loss_type=loss_type)
        train_dict['train_time'] = train_time
        accumulated_train_dicts.append(train_dict)
        
        if val_loaders != None:
            val_dict, _ = evaluate_model_performance(model, val_loaders, epoch, loss_type=loss_type)
            accumulated_val_dicts.append(val_dict)
            
            if schedule == 'reduce_lr_on_plateau':
                val_loss = val_dict['loss']

            print(f"Progress: {train_dict['epoch']}/{epochs}, operation_type: Training, Train Loss: {train_dict['loss']:.3f}, Val Loss: {val_dict['loss']:.3f}, Train acc.: {train_dict['accuracy']:.3f}, Val acc.: {val_dict['accuracy']:.3f}, Train NC acc.: {train_dict['neg_accuracy']:.3f}, Val NC acc.: {val_dict['neg_accuracy']:.3f}, Train PC acc.: {train_dict['pos_accuracy']:.3f}, Val PC acc.: {val_dict['pos_accuracy']:.3f}, Train PRAUC: {train_dict['prauc']:.3f}, Val PRAUC: {val_dict['prauc']:.3f}")
       
        else:
            print(f"Progress: {train_dict['epoch']}/{epochs}, operation_type: Training, Train Loss: {train_dict['loss']:.3f}, Train acc.: {train_dict['accuracy']:.3f}, Train NC acc.: {train_dict['neg_accuracy']:.3f}, Train PC acc.: {train_dict['pos_accuracy']:.3f}, Train PRAUC: {train_dict['prauc']:.3f}")
        if test_loaders != None:
            test_dict, _ = evaluate_model_performance(model, test_loaders, epoch, loss_type=loss_type)
            accumulated_test_dicts.append(test_dict)
            print(f"Progress: {test_dict['epoch']}/{epochs}, operation_type: Training, Train Loss: {test_dict['loss']:.3f}, Train acc.: {test_dict['accuracy']:.3f}, Train NC acc.: {test_dict['neg_accuracy']:.3f}, Train PC acc.: {test_dict['pos_accuracy']:.3f}, Train PRAUC: {test_dict['prauc']:.3f}")

        if scheduler:
            if schedule == 'reduce_lr_on_plateau':
                scheduler.step(val_loss)
            if schedule == 'step_lr':
                scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            if accumulated_train_dicts:
                train_df = pd.DataFrame(accumulated_train_dicts)
                _save_progress(dst, train_df, result_type='train')
                
            if accumulated_val_dicts:
                val_df = pd.DataFrame(accumulated_val_dicts)
                _save_progress(dst, val_df,result_type='validation')
                
            if accumulated_test_dicts:
                val_df = pd.DataFrame(accumulated_test_dicts)
                _save_progress(dst, val_df, result_type='test')
                
        batch_size = len(train_loaders)
        duration = time.time() - start_time
        time_ls.append(duration)
        
        model_path = _save_model(model, model_type, train_dict, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94], channels=channels)
            
    return model, model_path

def visualize_saliency_map(src, model_type='maxvit', model_path='', image_size=224, channels=[1,2,3], normalize=True, class_names=None, save_saliency=False, save_dir='saliency_maps'):

    from spacr.utils import SaliencyMapGenerator, preprocess_image

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the entire model object
    model = torch.load(model_path)
    model.to(device)

    # Create directory for saving saliency maps if it does not exist
    if save_saliency and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Collect all images and their tensors
    images = []
    input_tensors = []
    filenames = []
    for file in os.listdir(src):
        if not file.endswith('.png'):
            continue
        image_path = os.path.join(src, file)
        image, input_tensor = preprocess_image(image_path, normalize=normalize, image_size=image_size, channels=channels)
        images.append(image)
        input_tensors.append(input_tensor)
        filenames.append(file)

    input_tensors = torch.cat(input_tensors).to(device)
    class_labels = torch.zeros(input_tensors.size(0), dtype=torch.long).to(device)  # Replace with actual class labels if available

    # Generate saliency maps
    cam_generator = SaliencyMapGenerator(model)
    saliency_maps = cam_generator.compute_saliency_maps(input_tensors, class_labels)

    # Convert saliency maps to numpy arrays
    saliency_maps = saliency_maps.cpu().numpy()

    N = len(images)

    dst = os.path.join(src, 'saliency_maps')

    for i in range(N):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(images[i])
        axes[0].axis('off')
        if class_names:
            axes[0].set_title(f"Class: {class_names[class_labels[i].item()]}")

        # Saliency Map
        axes[1].imshow(saliency_maps[i, 0], cmap='hot')
        axes[1].axis('off')
        axes[1].set_title("Saliency Map")

        # Overlay
        overlay = np.array(images[i])
        overlay = overlay / overlay.max()
        saliency_map_rgb = np.stack([saliency_maps[i, 0]] * 3, axis=-1)  # Convert saliency map to RGB
        overlay = (overlay * 0.5 + saliency_map_rgb * 0.5).clip(0, 1)
        axes[2].imshow(overlay)
        axes[2].axis('off')
        axes[2].set_title("Overlay")

        plt.tight_layout()
        plt.show()

        # Save the saliency map if required
        if save_saliency:
            os.makedirs(dst, exist_ok=True)
            saliency_image = Image.fromarray((saliency_maps[i, 0] * 255).astype(np.uint8))
            saliency_image.save(os.path.join(dst, f'saliency_{filenames[i]}'))

def visualize_grad_cam(src, model_path, target_layers=None, image_size=224, channels=[1, 2, 3], normalize=True, class_names=None, save_cam=False, save_dir='grad_cam'):

    from spacr.utils import GradCAM, preprocess_image, show_cam_on_image, recommend_target_layers

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = torch.load(model_path)
    model.to(device)
    
    # If no target layers provided, recommend a target layer
    if target_layers is None:
        target_layers, all_layers = recommend_target_layers(model)
        print(f"No target layer provided. Using recommended layer: {target_layers[0]}")
        print("All possible target layers:")
        for layer in all_layers:
            print(layer)
    
    grad_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    if save_cam and not os.path.exists(save_dir):
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
        cam = grad_cam(input_tensor)
        cam_image = show_cam_on_image(np.array(image) / 255.0, cam)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title("Original Image")
        ax[1].imshow(cam_image)
        ax[1].axis('off')
        ax[1].set_title("Grad-CAM")
        plt.show()

        if save_cam:
            cam_pil = Image.fromarray(cam_image)
            cam_pil.save(os.path.join(save_dir, f'grad_cam_{file}'))

def visualize_classes(model, dtype, class_names, **kwargs):

    from spacr.utils import class_visualization

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

def deep_spacr(settings={}):
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
