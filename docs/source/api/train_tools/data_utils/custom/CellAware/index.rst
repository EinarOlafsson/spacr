train_tools.data_utils.custom.CellAware
=======================================

.. py:module:: train_tools.data_utils.custom.CellAware




Module Contents
---------------

.. py:class:: BoundaryExclusion(keys=['label'], allow_missing_keys=False)

   Bases: :py:obj:`monai.transforms.compose.MapTransform`


   Map the cell boundary pixel labels to the background class (0).


.. py:class:: IntensityDiversification(keys=['img'], change_cell_ratio=0.4, scale_factors=[0, 0.7], allow_missing_keys=False)

   Bases: :py:obj:`monai.transforms.compose.MapTransform`


   Randomly rescale the intensity of cell pixels.


   .. py:attribute:: change_cell_ratio
      :value: 0.4



   .. py:attribute:: randscale_intensity


