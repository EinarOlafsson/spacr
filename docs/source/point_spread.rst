Process images with a point-spread function
===========================================

Use :mod:`spacr.point_spread` from Python to convolve an image with a
point-spread function (PSF), or to deconvolve it with Richardson–Lucy.
Prepare a two-dimensional image or three-dimensional volume and a measured
PSF sampled at the same pixel or voxel spacing. Supply intensity images;
keep object-label masks separate.

Load the image and PSF
----------------------

The example below reads a single-channel image and a measured PSF from TIFF
files. Replace the example spacing of 0.11 µm with your calibrated Y and X
pixel spacing. A three-dimensional input uses Z, Y and X spacing in that
order. The PSF must have odd spatial dimensions, a central origin and
nonnegative signal after background removal.

.. code-block:: python

   import json
   from pathlib import Path
   import tifffile
   from spacr.point_spread import load_psf, apply_psf

   spacing = (0.11, 0.11)
   image = tifffile.imread("cell_channel.tif")
   kernel = load_psf("measured_psf.tif", sampling_um=spacing)

Choose the operation and save the result
----------------------------------------

Set ``operation="deconvolve"`` to perform Richardson–Lucy deconvolution.
Start with a modest iteration count and compare the result with the input;
more iterations can amplify noise. ``operation="convolve"`` instead blurs
the image with the PSF. It does not use the iteration count.

.. code-block:: python

   result = apply_psf(
       image,
       kernel,
       operation="deconvolve",
       image_sampling_um=spacing,
       iterations=10,
   )
   tifffile.imwrite("cell_channel_deconvolved.tif", result.image)
   Path("cell_channel_deconvolved.json").write_text(
       json.dumps(result.provenance, indent=2), encoding="utf-8"
   )

The output preserves the image shape and uses float32 values in the original
intensity units. Inspect its range before converting it to an integer image.
Keep the JSON record with the result; it contains the kernel identity,
sampling and processing settings.

For multichannel images, set ``channel_axis`` explicitly: ``-1`` selects a
final channel dimension, while ``0`` selects a first channel dimension.
Each intensity channel is processed independently. Image and PSF sampling
must match; the function rejects mismatched spacing.

Use a calculated approximation
------------------------------

If your workflow calls for a Gaussian approximation, construct it with
:func:`spacr.point_spread.gaussian_psf`. Supply the full width at half maximum
in micrometres and the image spacing. These are example values to replace
with the values appropriate to your analysis:

.. code-block:: python

   from spacr.point_spread import gaussian_psf

   kernel = gaussian_psf(
       fwhm_um=(0.30, 0.30), sampling_um=spacing, ndim=2
   )

Pass this kernel to ``apply_psf`` as above. A Gaussian kernel is an
approximation; retain that distinction when comparing it with a measured
PSF. See :func:`spacr.point_spread.apply_psf` for cancellation, progress
callbacks and the complete parameter reference.
