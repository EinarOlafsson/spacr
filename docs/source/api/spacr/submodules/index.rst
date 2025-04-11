spacr.submodules
================

.. py:module:: spacr.submodules






Module Contents
---------------

.. py:class:: CellposeLazyDataset(image_files, label_files, settings, randomize=True, augment=False)

   Bases: :py:obj:`torch.utils.data.Dataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:attribute:: normalize


   .. py:attribute:: percentiles


   .. py:attribute:: target_size


   .. py:attribute:: augment
      :value: False



   .. py:method:: apply_augmentation(image, label, aug_idx)


.. py:function:: train_cellpose(settings)

.. py:function:: test_cellpose_model(settings)

.. py:function:: apply_cellpose_model(settings)

.. py:function:: plot_cellpose_batch(images, labels)

.. py:function:: analyze_percent_positive(settings)

.. py:function:: analyze_recruitment(settings)

   Analyze recruitment data by grouping the DataFrame by well coordinates and plotting controls and recruitment data.

   Parameters:
   settings (dict): settings.

   Returns:
   None


.. py:function:: analyze_plaques(settings)

.. py:function:: count_phenotypes(settings)

.. py:function:: compare_reads_to_scores(reads_csv, scores_csv, empirical_dict={'r1': (90, 10), 'r2': (90, 10), 'r3': (80, 20), 'r4': (80, 20), 'r5': (70, 30), 'r6': (70, 30), 'r7': (60, 40), 'r8': (60, 40), 'r9': (50, 50), 'r10': (50, 50), 'r11': (40, 60), 'r12': (40, 60), 'r13': (30, 70), 'r14': (30, 70), 'r15': (20, 80), 'r16': (20, 80)}, pc_grna='TGGT1_220950_1', nc_grna='TGGT1_233460_4', y_columns=['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction'], column='columnID', value='c3', plate=None, save_paths=None)

.. py:function:: interperate_vision_model(settings={})

.. py:function:: analyze_endodyogeny(settings)

.. py:function:: analyze_class_proportion(settings)

.. py:function:: generate_score_heatmap(settings)

.. py:function:: post_regression_analysis(csv_file, grna_dict, grna_list, save=False)

