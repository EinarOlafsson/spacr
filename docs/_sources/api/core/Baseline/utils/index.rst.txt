core.Baseline.utils
===================

.. py:module:: core.Baseline.utils

.. autoapi-nested-parse::

   Adapted from the following references:
   [1] https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/model_training_3class.py





Module Contents
---------------

.. py:function:: identify_instances_from_classmap(class_map, cell_class=1, threshold=0.5, from_logits=True)

   Identification of cell instances from the class map


.. py:function:: create_interior_onehot(inst_maps)

   interior : (H,W), np.uint8
       three-class map, values: 0,1,2
       0: background
       1: interior
       2: boundary


