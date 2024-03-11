import os, torch, time, gc, datetime
import pandas as pd
from torch.optim import Adagrad
from torch.optim import AdamW
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from IPython.display import display, clear_output

from .logger import log_function_call

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
            #data, target = data.to(torch.float).to(device), target.to(device).float()
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
    
    from .io import save_settings, _copy_missclassified
    from .utils import pick_best_model, test_model_performance
    from .core import generate_loaders
    
    settings['src'] = src
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','train_test_model_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)
    
    if custom_model:
        model = torch.load(custom_model_path)
    
    if settings['train']:
        save_settings(settings, src)
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()
    dst = os.path.join(src,'model')
    os.makedirs(dst, exist_ok=True)
    settings['src'] = src
    settings['dst'] = dst
    if settings['train']:
        train, val, plate_names  = generate_loaders(src, 
                                                    train_mode=settings['train_mode'], 
                                                    mode='train', 
                                                    image_size=settings['image_size'],
                                                    batch_size=settings['batch_size'], 
                                                    classes=settings['classes'], 
                                                    num_workers=settings['num_workers'],
                                                    validation_split=settings['val_split'],
                                                    pin_memory=settings['pin_memory'],
                                                    normalize=settings['normalize'],
                                                    verbose=settings['verbose']) 

    if settings['test']:
        test, _, plate_names_test = generate_loaders(src, 
                                   train_mode=settings['train_mode'], 
                                   mode='test', 
                                   image_size=settings['image_size'],
                                   batch_size=settings['batch_size'], 
                                   classes=settings['classes'], 
                                   num_workers=settings['num_workers'],
                                   validation_split=0.0,
                                   pin_memory=settings['pin_memory'],
                                   normalize=settings['normalize'],
                                   verbose=settings['verbose'])
        if model == None:
            model_path = pick_best_model(src+'/model')
            print(f'Best model: {model_path}')

            model = torch.load(model_path, map_location=lambda storage, loc: storage)

            model_type = settings['model_type']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(type(model))
            print(model)
        
        model_fldr = os.path.join(src,'model')
        time_now = datetime.date.today().strftime('%y%m%d')
        result_loc = f'{model_fldr}/{model_type}_time_{time_now}_result.csv'
        acc_loc = f'{model_fldr}/{model_type}_time_{time_now}_acc.csv'
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
    else:
        test = None
    
    if settings['train']:
        train_model(dst = settings['dst'],
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
                    num_workers = settings['num_workers'], 
                    val_loaders = val, 
                    test_loaders = test, 
                    intermedeate_save = settings['intermedeate_save'],
                    schedule = settings['schedule'],
                    loss_type=settings['loss_type'], 
                    gradient_accumulation=settings['gradient_accumulation'], 
                    gradient_accumulation_steps=settings['gradient_accumulation_steps'])

    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    gc.collect()
    
def train_model(dst, model_type, train_loaders, train_loader_names, train_mode='erm', epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, num_workers=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule = None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4):
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
        num_workers (int, optional): The number of workers for data loading. Defaults to 20.
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
    
    from .io import save_model, save_progress
    from .utils import evaluate_model_performance, compute_irm_penalty, calculate_loss, choose_model
    
    print(f'Train batches:{len(train_loaders)}, Validation batches:{len(val_loaders)}')
    
    if test_loaders != None:
        print(f'Test batches:{len(test_loaders)}')
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    for idx, (images, labels, filenames) in enumerate(train_loaders):
        batch, channels, height, width = images.shape
        break

    model = choose_model(model_type, device, init_weights, dropout_rate, use_checkpoint)
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
            
            save_progress(dst, results_df, train_metrics_df)
            clear_output(wait=True)
            display(results_df)
            save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94])
            
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
            save_progress(dst, results_df, train_metrics_df)
            save_model(model, model_type, results_df, dst, epoch, epochs, intermedeate_save=[0.99,0.98,0.95,0.94])
            print(f'Saved model: {dst}')
    return