core.Baseline.Trainer
=====================

.. py:module:: core.Baseline.Trainer




Module Contents
---------------

.. py:class:: Trainer(model, dataloaders, optimizer, scheduler=None, criterion=None, num_epochs=100, device='cuda:0', no_valid=False, valid_frequency=1, amp=False, algo_params=None)

   Bases: :py:obj:`core.BaseTrainer.BaseTrainer`


   Abstract base class for trainer implementations


   .. py:attribute:: criterion


