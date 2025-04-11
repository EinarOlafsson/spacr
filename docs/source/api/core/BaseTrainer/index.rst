core.BaseTrainer
================

.. py:module:: core.BaseTrainer




Module Contents
---------------

.. py:class:: BaseTrainer(model, dataloaders, optimizer, scheduler=None, criterion=None, num_epochs=100, device='cuda:0', no_valid=False, valid_frequency=1, amp=False, algo_params=None)

   Abstract base class for trainer implementations


   .. py:attribute:: model


   .. py:attribute:: dataloaders


   .. py:attribute:: optimizer


   .. py:attribute:: scheduler
      :value: None



   .. py:attribute:: criterion
      :value: None



   .. py:attribute:: num_epochs
      :value: 100



   .. py:attribute:: no_valid
      :value: False



   .. py:attribute:: valid_frequency
      :value: 1



   .. py:attribute:: device
      :value: 'cuda:0'



   .. py:attribute:: amp
      :value: False



   .. py:attribute:: best_weights
      :value: None



   .. py:attribute:: best_f1_score
      :value: 0.1



   .. py:attribute:: scaler
      :value: None



   .. py:attribute:: loss_metric


   .. py:attribute:: f1_metric


   .. py:attribute:: post_pred


   .. py:attribute:: post_gt


   .. py:method:: train()

      Train the model



