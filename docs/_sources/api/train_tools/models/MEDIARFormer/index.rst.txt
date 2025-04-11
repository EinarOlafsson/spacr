train_tools.models.MEDIARFormer
===============================

.. py:module:: train_tools.models.MEDIARFormer




Module Contents
---------------

.. py:class:: MEDIARFormer(encoder_name='mit_b5', encoder_weights='imagenet', decoder_channels=(1024, 512, 256, 128, 64), decoder_pab_channels=256, in_channels=3, classes=3)

   Bases: :py:obj:`segmentation_models_pytorch.MAnet`


   MEDIAR-Former Model


   .. py:attribute:: segmentation_head
      :value: None



   .. py:attribute:: cellprob_head


   .. py:attribute:: gradflow_head


   .. py:method:: forward(x)

      Forward pass through the network



