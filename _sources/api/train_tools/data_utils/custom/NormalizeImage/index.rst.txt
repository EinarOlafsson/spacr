train_tools.data_utils.custom.NormalizeImage
============================================

.. py:module:: train_tools.data_utils.custom.NormalizeImage




Module Contents
---------------

.. py:class:: CustomNormalizeImage(percentiles=[0, 99.5], channel_wise=False)

   Bases: :py:obj:`monai.transforms.transform.Transform`


   Normalize the image.


   .. py:attribute:: channel_wise
      :value: False



.. py:class:: CustomNormalizeImaged(keys: monai.config.KeysCollection, percentiles=[1, 99], channel_wise: bool = False, allow_missing_keys: bool = False)

   Bases: :py:obj:`monai.transforms.compose.MapTransform`


   Dictionary-based wrapper of NormalizeImage


   .. py:attribute:: normalizer


