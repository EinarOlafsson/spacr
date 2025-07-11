core.MEDIAR.Trainer
===================

.. py:module:: core.MEDIAR.Trainer




Module Contents
---------------

.. py:class:: Trainer(model, dataloaders, optimizer, scheduler=None, criterion=None, num_epochs=100, device='cuda:0', no_valid=False, valid_frequency=1, amp=False, algo_params=None)

   Bases: :py:obj:`core.BaseTrainer.BaseTrainer`


   Abstract base class for trainer implementations


   .. py:attribute:: mse_loss


   .. py:attribute:: bce_loss


   .. py:method:: mediar_criterion(outputs, labels_onehot_flows)

      loss function between true labels and prediction outputs



