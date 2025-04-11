spacr.deep_spacr
================

.. py:module:: spacr.deep_spacr






Module Contents
---------------

.. py:function:: apply_model(src, model_path, image_size=224, batch_size=64, normalize=True, n_jobs=10)

.. py:function:: apply_model_to_tar(settings={})

.. py:function:: evaluate_model_performance(model, loader, epoch, loss_type)

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


.. py:function:: test_model_core(model, loader, loader_name, epoch, loss_type)

.. py:function:: test_model_performance(loaders, model, loader_name_list, epoch, loss_type)

   Test the performance of a model on given data loaders.

   Args:
       loaders (list): List of data loaders.
       model: The model to be tested.
       loader_name_list (list): List of names for the data loaders.
       epoch (int): The current epoch.
       loss_type: The type of loss function.

   Returns:
       tuple: A tuple containing the test results and the results dataframe.


.. py:function:: train_test_model(settings)

.. py:function:: train_model(dst, model_type, train_loaders, epochs=100, learning_rate=0.0001, weight_decay=0.05, amsgrad=False, optimizer_type='adamw', use_checkpoint=False, dropout_rate=0, n_jobs=20, val_loaders=None, test_loaders=None, init_weights='imagenet', intermedeate_save=None, chan_dict=None, schedule=None, loss_type='binary_cross_entropy_with_logits', gradient_accumulation=False, gradient_accumulation_steps=4, channels=['r', 'g', 'b'], verbose=False)

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


.. py:function:: generate_activation_map(settings)

.. py:function:: visualize_classes(model, dtype, class_names, **kwargs)

.. py:function:: visualize_integrated_gradients(src, model_path, target_label_idx=0, image_size=224, channels=[1, 2, 3], normalize=True, save_integrated_grads=False, save_dir='integrated_grads')

.. py:class:: SmoothGrad(model, n_samples=50, stdev_spread=0.15)

   .. py:attribute:: model


   .. py:attribute:: n_samples
      :value: 50



   .. py:attribute:: stdev_spread
      :value: 0.15



   .. py:method:: compute_smooth_grad(input_tensor, target_class)


.. py:function:: visualize_smooth_grad(src, model_path, target_label_idx, image_size=224, channels=[1, 2, 3], normalize=True, save_smooth_grad=False, save_dir='smooth_grad')

.. py:function:: deep_spacr(settings={})

.. py:function:: model_knowledge_transfer(teacher_paths, student_save_path, data_loader, device='cpu', student_model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, alpha=0.5, temperature=2.0, lr=0.0001, epochs=10)

.. py:function:: model_fusion(model_paths, save_path, device='cpu', model_name='maxvit_t', pretrained=True, dropout_rate=None, use_checkpoint=False, aggregator='mean')

.. py:function:: annotate_filter_vision(settings)

