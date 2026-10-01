Train a Cellpose model
======================

Use your corrected image masks to fine-tune Cellpose-SAM, then apply the saved
model to another image folder. From **Home → Tools → Make Masks**, open
**Cellpose Workbench** and choose **Train**.

To follow along, download the
`six example image/mask pairs <tutorials/examples/Cellpose_training_images_masks.zip>`__
and extract them. Set Source to the extracted ``training`` folder. Review the
supplied cell masks in Make Masks, then follow the settings and Run steps below.
The archive's ``apply`` folder holds three more images from wells that are not
in the training set; use it to try the trained checkpoint in Apply.

Prepare images and masks
------------------------

Place training images in one folder and their masks in its ``masks`` subfolder::

   training/
       field_01.tif
       field_02.tif
       masks/
           field_01.tif
           field_02.tif

Each mask must have the same height and width as its image. Use integer object
labels: zero for background and a different positive number for each object.
A mask filename can match its image or add ``_masks`` before the extension,
such as ``field_01_masks.tif``. Keep one matching mask per image.

Use Make Masks to inspect and correct object boundaries before training.
Keep separate fields for validation. You can reuse a legacy project containing
``train/images`` and ``train/masks`` by selecting that project as Source.

Choose inputs and settings
--------------------------

#. Set **Source** (``src``) to the training image folder.
#. Leave ``mask_src`` empty to use ``<src>/masks``, or select another mask folder.
#. For validation, set ``test_src`` to a separate image folder. Its masks belong
   in ``<test_src>/masks`` unless you select ``test_mask_src``.
#. Choose ``base_model``: ``cpsam`` starts from Cellpose-SAM; a model-zoo key
   or checkpoint path continues training an existing model.
#. Enter ``model_name`` as a filename prefix. Use ``save_path`` to choose an
   output folder, or leave it empty to save under the training project.

For multichannel images, ``channels`` selects up to three zero-based channel
indices. For example, ``[0, 2]`` selects the first and third channels.
Leave ``channel_axis`` empty for automatic detection, or set it explicitly
when the image layout is ambiguous. Training accepts two-dimensional images
with optional channels.

.. list-table:: Main training controls
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Default
     - What it controls
   * - ``n_epochs``
     - 100
     - Number of training epochs.
   * - ``batch_size``
     - 1
     - Images per training minibatch. Larger batches require more memory.
   * - ``learning_rate``
     - 0.00001
     - Size of the model's training updates.
   * - ``weight_decay``
     - 0.1
     - Strength of weight regularization during training.
   * - ``normalize`` / ``percentiles``
     - On / [1, 99]
     - Percentile normalization of each image channel.
   * - ``min_train_masks``
     - 5
     - Minimum labelled objects in an image included in training.
   * - ``scale_range``
     - 0.5
     - Variation in image scale during training augmentation.
   * - ``save_every`` / ``save_each``
     - 100 / Off
     - Checkpoint interval and whether periodic checkpoints get separate names.

For a smaller initial run, ``max_train_images`` limits the loaded training
images. ``nimg_per_epoch`` and ``nimg_test_per_epoch`` limit how many images
are used per epoch. Leave these limits empty to use all available images.

Run and use the output
----------------------

Click **Run** and follow progress in the console. When training finishes,
the console reports the saved checkpoint path. With the default output
location, checkpoints are under ``<src>/models/cellpose_model/models``.
With a chosen ``save_path``, they are under ``<save_path>/models``.

Open **Apply**, select a separate image folder, such as the example's ``apply``
folder, and check that **Custom model** points to the checkpoint you want.
Turn on **Save**, which is off by default. Inspect a preview before processing
the folder. Apply writes label masks into that image folder's ``masks`` subfolder.
Use separate labelled images to evaluate segmentation before applying the
model to an entire experiment.

The Python entry point :func:`spacr.submodules.train_cellpose` accepts the same
settings and returns the checkpoint path, training losses and validation
losses. Continue with :doc:`Make Masks <make_masks>` to inspect or edit masks,
or :ref:`Mask <workflow-module-mask>` to use the model in an image pipeline.
