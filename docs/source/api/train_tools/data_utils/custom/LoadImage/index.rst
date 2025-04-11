train_tools.data_utils.custom.LoadImage
=======================================

.. py:module:: train_tools.data_utils.custom.LoadImage




Module Contents
---------------

.. py:class:: CustomLoadImage(reader=None, image_only: bool = False, dtype: monai.config.DtypeLike = np.float32, ensure_channel_first: bool = False, *args, **kwargs)

   Bases: :py:obj:`monai.transforms.LoadImage`


   Load image file or files from provided path based on reader.
   If reader is not specified, this class automatically chooses readers
   based on the supported suffixes and in the following order:

       - User-specified reader at runtime when calling this loader.
       - User-specified reader in the constructor of `LoadImage`.
       - Readers from the last to the first in the registered list.
       - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
         (npz, npy -> NumpyReader), (nrrd -> NrrdReader), (DICOM file -> ITKReader).

   [!Caution] This overriding replaces the original ITK with Custom UnifiedITKReader.


   .. py:attribute:: readers
      :value: []



.. py:class:: CustomLoadImaged(keys: monai.config.KeysCollection, reader: Optional[Union[monai.data.image_reader.ImageReader, str]] = None, dtype: monai.config.DtypeLike = np.float32, meta_keys: Optional[monai.config.KeysCollection] = None, meta_key_postfix: str = DEFAULT_POST_FIX, overwriting: bool = False, image_only: bool = False, ensure_channel_first: bool = False, simple_keys=False, allow_missing_keys: bool = False, *args, **kwargs)

   Bases: :py:obj:`monai.transforms.LoadImaged`


   Dictionary-based wrapper of `CustomLoadImage`.


   .. py:attribute:: meta_keys


   .. py:attribute:: meta_key_postfix


   .. py:attribute:: overwriting
      :value: False



