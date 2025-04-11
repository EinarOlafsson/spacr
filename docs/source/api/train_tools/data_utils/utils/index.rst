train_tools.data_utils.utils
============================

.. py:module:: train_tools.data_utils.utils




Module Contents
---------------

.. py:function:: split_train_valid(data_dicts, valid_portion=0.1)

   Split train/validata data according to the given proportion


.. py:function:: path_decoder(root, mapping_file, no_label=False, unlabeled=False)

   Decode img/label file paths from root & mapping directory.

   Args:
       root (str):
       mapping_file (str): json file containing image & label file paths.
       no_label (bool, optional): whether to include "label" key. Defaults to False.

   Returns:
       list: list of dictionary. (ex. [{"img": img_path, "label": label_path}, ...])


