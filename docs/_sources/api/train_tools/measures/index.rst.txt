train_tools.measures
====================

.. py:module:: train_tools.measures

.. autoapi-nested-parse::

   Adapted from the following references:
   [1] https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/compute_metric.py
   [2] https://github.com/stardist/stardist/blob/master/stardist/matching.py





Module Contents
---------------

.. py:function:: evaluate_f1_score_cellseg(masks_true, masks_pred, threshold=0.5)

   Get confusion elements for cell segmentation results.
   Boundary pixels are not considered during evaluation.


.. py:function:: evaluate_f1_score(tp, fp, fn)

   Evaluate F1-score for the given confusion elements


