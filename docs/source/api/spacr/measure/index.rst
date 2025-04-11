spacr.measure
=============

.. py:module:: spacr.measure




Module Contents
---------------

.. py:function:: get_components(cell_mask, nucleus_mask, pathogen_mask)

   Get the components (nucleus and pathogens) for each cell in the given masks.

   Args:
       cell_mask (ndarray): Binary mask of cell labels.
       nucleus_mask (ndarray): Binary mask of nucleus labels.
       pathogen_mask (ndarray): Binary mask of pathogen labels.

   Returns:
       tuple: A tuple containing two dataframes - nucleus_df and pathogen_df.
           nucleus_df (DataFrame): Dataframe with columns 'cell_id' and 'nucleus',
               representing the mapping of each cell to its nucleus.
           pathogen_df (DataFrame): Dataframe with columns 'cell_id' and 'pathogen',
               representing the mapping of each cell to its pathogens.


.. py:function:: save_and_add_image_to_grid(png_channels, img_path, grid, plot=False)

   Add an image to a grid and save it as PNG.

   Args:
       png_channels (ndarray): The array representing the image channels.
       img_path (str): The path to save the image as PNG.
       grid (list): The grid of images to be plotted later.

   Returns:
       grid (list): Updated grid with the new image added.


.. py:function:: img_list_to_grid(grid, titles=None)

   Plot a grid of images with optional titles.

   Args:
       grid (list): List of images to be plotted.
       titles (list): List of titles for the images.

   Returns:
       fig (Figure): The matplotlib figure object containing the image grid.


.. py:function:: measure_crop(settings)

   Measure the crop of an image based on the provided settings.

   Args:
       settings (dict): The settings for measuring the crop.

   Returns:
       None


.. py:function:: process_meassure_crop_results(partial_results, settings)

   Process the results, display, and optionally save the figures.

   Args:
       partial_results (list): List of partial results.
       settings (dict): Settings dictionary.
       save_figures (bool): Flag to save figures or not.


.. py:function:: generate_cellpose_train_set(folders, dst, min_objects=5)

.. py:function:: get_object_counts(src)

