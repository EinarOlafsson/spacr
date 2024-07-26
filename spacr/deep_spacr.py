import os, torch, time, gc, datetime
import numpy as np
import pandas as pd
from torch.optim import Adagrad, AdamW
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image

from .logger import log_function_call
from .utils import close_multiprocessing_processes, reset_mp
#reset_mp()
#close_multiprocessing_processes()

def evaluate_model_core(model, loader, loader_name, epoch, loss_type):
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
    
    from .utils import calculate_loss, classification_metrics
    
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
            print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx+1}/{len(loader)} loss: {mean_loss:.5f} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    loss /= len(loader)
    data_df = classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch)
    return data_df, prediction_pos_probs, all_labels

def evaluate_model_performance(loaders, model, loader_name_list, epoch, train_mode, loss_type):
    """
    Evaluate the performance of a model on given data loaders.

    Args:
        loaders (list): List of data loaders.
        model: The model to evaluate.
        loader_name_list (list): List of names for the data loaders.
        epoch (int): The current epoch.
        train_mode (str): The training mode ('erm' or 'irm').
        loss_type: The type of loss function.

    Returns:
        tuple: A tuple containing the evaluation result and the time taken for evaluation.
    """
    start_time = time.time()
    df_list = []
    if train_mode == 'erm':
        result, _, _ = evaluate_model_core(model, loaders, loader_name_list, epoch, loss_type)
    if train_mode == 'irm':
        for loader_index in range(0, len(loaders)):
            loader = loaders[loader_index]
            loader_name = loader_name_list[loader_index]
            data_df, _, _ = evaluate_model_core(model, loader, loader_name, epoch, loss_type)
            torch.cuda.empty_cache()
            df_list.append(data_df)
        result = pd.concat(df_list)
        nc_mean = result['neg_accuracy'].mean(skipna=True)
        pc_mean = result['pos_accuracy'].mean(skipna=True)
        tot_mean = result['accuracy'].mean(skipna=True)
        loss_mean = result['loss'].mean(skipna=True)
        prauc_mean = result['prauc'].mean(skipna=True)
        data_mean = {'accuracy': tot_mean, 'neg_accuracy': nc_mean, 'pos_accuracy': pc_mean, 'loss': loss_mean, 'prauc': prauc_mean}
        result = pd.concat([pd.DataFrame(result), pd.DataFrame(data_mean, index=[str(epoch)+'_mean'])])
    end_time = time.time()
    test_time = end_time - start_time
    return result, test_time

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
            print(f'\rTest: epoch: {epoch} Accuracy: {acc:.5f} batch: {batch_idx}/{len(loader)} loss: {mean_loss:.5f} time {test_time:.5f}', end='\r', flush=True)
    
    # Constructing the DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': true_targets,
        'predicted_label': predicted_outputs,
        'class_1_probability':prediction_pos_probs})

    loss /= len(loader)
    data_df = classification_metrics(all_labels, prediction_pos_probs, loader_name, loss, epoch)
    return data_df, prediction_pos_probs, all_labels, results_df

def test_model_performance(loaders, model, loader_name_list, epoch, train_mode, loss_type):
    """
    Test the performance of a model on given data loaders.

    Args:
        loaders (list): List of data loaders.
        model: The model to be tested.
        loader_name_list (list): List of names for the data loaders.
        epoch (int): The current epoch.
        train_mode (str): The training mode ('erm' or 'irm').
        loss_type: The type of loss function.

    Returns:
        tuple: A tuple containing the test results and the results dataframe.
    """
    start_time = time.time()
    df_list = []
    if train_mode == 'erm':
        result, prediction_pos_probs, all_labels, results_df = test_model_core(model, loaders, loader_name_list, epoch, loss_type)
    if train_mode == 'irm':
        for loader_index in range(0, len(loaders)):
            loader = loaders[loader_index]
            loader_name = loader_name_list[loader_index]
            data_df, prediction_pos_probs, all_labels, results_df = test_model_core(model, loader, loader_name, epoch, loss_type)
            torch.cuda.empty_cache()
            df_list.append(data_df)
        result = pd.concat(df_list)
        nc_mean = result['neg_accuracy'].mean(skipna=True)
        pc_mean = result['pos_accuracy'].mean(skipna=True)
        tot_mean = result['accuracy'].mean(skipna=True)
        loss_mean = result['loss'].mean(skipna=True)
        prauc_mean = result['prauc'].mean(skipna=True)
        data_mean = {'accuracy': tot_mean, 'neg_accuracy': nc_mean, 'pos_accuracy': pc_mean, 'loss': loss_mean, 'prauc': prauc_mean}
        result = pd.concat([pd.DataFrame(result), pd.DataFrame(data_mean, index=[str(epoch)+'_mean'])])
    end_time = time.time()
    test_time = end_time - start_time
    return result, results_df

def train_test_model(src, settings, custom_model=False, custom_model_path=None):
    
    from .io import _save_settings, _copy_missclassified
    from .utils import pick_best_model
    from .core import generate_loaders
    from .settings import set_default_train_test_model

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()

    settings = set_default_train_test_model(settings)
    channels_str = ''.join(settings['channels'])
    dst = os.path.join(src,'model', settings['model_type'], channels_str, str(f"epochs_{settings['epochs']}"))
    os.makedirs(dst, exist_ok=True)
    settings['src'] = src
    settings['dst'] = dst
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(dst,'train_test_model_settings.csv')
    settings_df.to_csv(settings_csv, index=False)
    
    if custom_model:
        model = torch.load(custom_model_path)
    
    if settings['train']:
        _save_settings(settings, src)

    if settings['train']:
        train, val, plate_names, train_fig  = generate_loaders(src, 
                                                    train_mode=settings['train_mode'], 
                                                    mode='train', 
                                                    image_size=settings['image_size'],
                                                    batch_size=settings['batch_size'], 
                                                    classes=settings['classes'], 
                                                    n_jobs=settings['n_jobs'],
                                                    validation_split=settings['val_split'],
                                                    pin_memory=settings['pin_memory'],
                                                    normalize=settings['normalize'],
                                                    channels=settings['channels'],
                                                    augment=settings['augment'],
                                                    verbose=settings['verbose'])
        
        train_batch_1_figure = os.path.join(dst, 'batch_1.pdf')
        train_fig.savefig(train_batch_1_figure, format='pdf', dpi=600)
    
    if settings['train']:
        model = train_model(dst = settings['dst'],
                            model_type=settings['model_type'],
                            train_loaders = train, 
                            train_loader_names = plate_names, 
                            train_mode = settings['train_mode'], 
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
                            channels=settings['channels'])
        
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        gc.collect()
        
    if settings['test']:
        test, _, plate_names_test, train_fig = generate_loaders(src, 
                                                     train_mode=settings['train_mode'], 
                                                     mode='test', 
                                                     image_size=settings['image_size'],
                                                     batch_size=settings['batch_size'], 
                                                     classes=settings['classes'], 
                                                     n_jobs=settings['n_jobs'],
                                                     validation_split=0.0,
                                                     pin_memory=settings['pin_memory'],
                                                     normalize=settings['normalize'],
                                                     channels=settings['channels'],
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
                                                  train_mode=settings['train_mode'],
                                                  loss_type=settings['loss_type'])
        
        result.to_csv(result_loc, index=True, header=True, mode='w')
        accuracy.to_csv(acc_loc, index=True, header=True, mode='w')
        _copy_missclassified(accuracy)

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()
    
def train_model(dst, model_type, train_loaders, train_loader_names, train_mode='erm', epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule = None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4, channels=['r','g','b']):
    """
    Trains a model using the specified parameters.

    Args:
        dst (str): The destination path to save the model and results.
        model_type (str): The type of model to train.
        train_loaders (list): A list of training data loaders.
        train_loader_names (list): A list of names for the training data loaders.
        train_mode (str, optional): The training mode. Defaults to 'erm'.
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
    from .utils import compute_irm_penalty, calculate_loss, choose_model
    
    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders)}')
    
    if test_loaders != None:
        print(f'Test batches:{len(test_loaders)}')
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'n_jobs': n_jobs, 'pin_memory': True} if use_cuda else {}
    
    for idx, (images, labels, filenames) in enumerate(train_loaders):
        batch, chans, height, width = images.shape
        break

    model = choose_model(model_type, device, init_weights, dropout_rate, use_checkpoint)

    if model is None:
        print(f'Model {model_type} not found')
        return

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

    if train_mode == 'erm':
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
                print(f'\rTrain: epoch: {epoch} batch: {batch_idx}/{len(train_loaders)} avg_loss: {avg_loss:.5f} time: {(time.time()-start_time):.5f}', end='\r', flush=True)

            end_time = time.time()
            train_time = end_time - start_time
            train_metrics = {'epoch':epoch,'loss':loss.cpu().item(), 'train_time':train_time}
            train_metrics_df = pd.DataFrame(train_metrics, index=[epoch])
            train_names = 'train'
            results_df, train_test_time = evaluate_model_performance(train_loaders, model, train_names, epoch, train_mode='erm', loss_type=loss_type)
            train_metrics_df['train_test_time'] = train_test_time
            if val_loaders != None:
                val_names = 'val'
                result, val_time = evaluate_model_performance(val_loaders, model, val_names, epoch, train_mode='erm', loss_type=loss_type)
                
                if schedule == 'reduce_lr_on_plateau':
                    val_loss = result['loss']
                
                results_df = pd.concat([results_df, result])
                train_metrics_df['val_time'] = val_time
            if test_loaders != None:
                test_names = 'test'
                result, test_test_time = evaluate_model_performance(test_loaders, model, test_names, epoch, train_mode='erm', loss_type=loss_type)
                results_df = pd.concat([results_df, result])
                test_time = (train_test_time+val_time+test_test_time)/3
                train_metrics_df['test_time'] = test_time
            
            if scheduler:
                if schedule == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                if schedule == 'step_lr':
                    scheduler.step()
            
            _save_progress(dst, results_df, train_metrics_df, epoch, epochs)
            clear_output(wait=True)
            display(results_df)
            _save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94], channels=channels)
            
    if train_mode == 'irm':
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
        phi = torch.nn.Parameter (torch.ones(4,1))
        for epoch in range(1, epochs):
            model.train()
            penalty_factor = epoch * 1e-5
            epoch_names = [str(epoch) + '_' + item for item in train_loader_names]
            loader_erm_loss_list = []
            total_erm_loss_mean = 0
            for loader_index in range(0, len(train_loaders)):
                start_time = time.time()
                loader = train_loaders[loader_index]
                loader_erm_loss_mean = 0
                batch_count = 0
                batch_erm_loss_list = []
                for batch_idx, (data, target, filenames) in enumerate(loader, start=1):
                    optimizer.zero_grad()
                    data, target = data.to(device), target.to(device).float()
                    
                    output = model(data)
                    erm_loss = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
                    
                    batch_erm_loss_list.append(erm_loss.mean())
                    print(f'\repoch: {epoch} loader: {loader_index} batch: {batch_idx+1}/{len(loader)}', end='\r', flush=True)
                loader_erm_loss_mean = torch.stack(batch_erm_loss_list).mean()
                loader_erm_loss_list.append(loader_erm_loss_mean)
            total_erm_loss_mean = torch.stack(loader_erm_loss_list).mean()
            irm_loss = compute_irm_penalty(loader_erm_loss_list, dummy_w, device)
            
            (total_erm_loss_mean + penalty_factor * irm_loss).backward()
            optimizer.step()
            
            end_time = time.time()
            train_time = end_time - start_time
            
            train_metrics = {'epoch': epoch, 'irm_loss': irm_loss, 'erm_loss': total_erm_loss_mean, 'penalty_factor': penalty_factor, 'train_time': train_time}
            #train_metrics = {'epoch':epoch,'irm_loss':irm_loss.cpu().item(),'erm_loss':total_erm_loss_mean.cpu().item(),'penalty_factor':penalty_factor, 'train_time':train_time}
            train_metrics_df = pd.DataFrame(train_metrics, index=[epoch])
            print(f'\rTrain: epoch: {epoch} loader: {loader_index} batch: {batch_idx+1}/{len(loader)} irm_loss: {irm_loss:.5f} mean_erm_loss: {total_erm_loss_mean:.5f} train time {train_time:.5f}', end='\r', flush=True)            
            
            train_names = [item + '_train' for item in train_loader_names]
            results_df, train_test_time = evaluate_model_performance(train_loaders, model, train_names, epoch, train_mode='irm', loss_type=loss_type)
            train_metrics_df['train_test_time'] = train_test_time
            
            if val_loaders != None:
                val_names = [item + '_val' for item in train_loader_names]
                result, val_time = evaluate_model_performance(val_loaders, model, val_names, epoch, train_mode='irm', loss_type=loss_type)
                
                if schedule == 'reduce_lr_on_plateau':
                    val_loss = result['loss']
                
                results_df = pd.concat([results_df, result])
                train_metrics_df['val_time'] = val_time
            
            if test_loaders != None:
                test_names = [item + '_test' for item in train_loader_names] #test_loader_names?
                result, test_test_time = evaluate_model_performance(test_loaders, model, test_names, epoch, train_mode='irm', loss_type=loss_type)
                results_df = pd.concat([results_df, result])
                train_metrics_df['test_test_time'] = test_test_time
                
            if scheduler:
                if schedule == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                if schedule == 'step_lr':
                    scheduler.step()
            
            clear_output(wait=True)
            display(results_df)
            _save_progress(dst, results_df, train_metrics_df, epoch, epochs)
            _save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94])
            print(f'Saved model: {dst}')
    return model

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

# Usage
#src = '/path/to/images'
#model_path = '/path/to/model.pth'
#target_label_idx = 0  # Change this to the target class index
#visualize_smooth_grad(src, model_path, target_label_idx)
