train_tools.data_utils.datasetter
=================================

.. py:module:: train_tools.data_utils.datasetter




Module Contents
---------------

.. py:function:: get_dataloaders_labeled(root, mapping_file, mapping_file_tuning, join_mapping_file=None, valid_portion=0.0, batch_size=8, amplified=False, relabel=False)

   Set DataLoaders for labeled datasets.

   Args:
       root (str): root directory
       mapping_file (str): json file for mapping dataset
       valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
       batch_size (int, optional): batch size. Defaults to 8.
       shuffle (bool, optional): shuffles dataloader. Defaults to True.
       num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.

   Returns:
       dict: dictionary of data loaders.


.. py:function:: get_dataloaders_public(root, mapping_file, valid_portion=0.0, batch_size=8)

   Set DataLoaders for labeled datasets.

   Args:
       root (str): root directory
       mapping_file (str): json file for mapping dataset
       valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
       batch_size (int, optional): batch size. Defaults to 8.
       shuffle (bool, optional): shuffles dataloader. Defaults to True.

   Returns:
       dict: dictionary of data loaders.


.. py:function:: get_dataloaders_unlabeled(root, mapping_file, batch_size=8, shuffle=True, num_workers=5)

   Set dataloaders for unlabeled dataset.


