## spaCR 1.5.1.8

### Upgrading

- **Measure now writes raw crops.** Crop PNGs keep the intensities and bit depth of your images, usually 16-bit. Only the pixels outside the object are set to 0. Earlier versions stretched every crop to its own brightest and darkest pixels, which hid intensity differences between cells. The new setting `crop_intensity` controls this. `'auto'`, the default, writes raw crops unless `normalize` is set to a percentile pair such as `[1, 99]`. `'stretch'` writes the old stretched crops. Inside spaCR nothing looks different: Annotate, the montages, Classify and models you have already trained show raw crops with the old stretch, pixel for pixel. Only other programs see the raw values.
- **Viewing raw crops in Fiji.** A raw crop opened in Fiji (ImageJ) can look black, because its values use only a small part of the 16-bit range. Choose Image > Adjust > Brightness/Contrast and press Auto, or use Process > Enhance Contrast for display only. These change the view, not the file. To compare cells, set the same display range for every crop with Set in the Brightness/Contrast window, and do not apply it to the pixels.
- **Crops are flat-field corrected when illumination correction is on.** With `illumination_correction` enabled, Measure now cuts crops from the corrected field, so the crops match the measured intensities. Your raw images on disk are not changed. Crop folders record which illumination model was used.
- **Classifier checkpoints record their input transform.** Every new model stores how its crops were prepared: crop size, resize, channels, mean and standard deviation, mask mode and, where used, the crop scales and flat-field model. Testing a model refuses to run when the crops would be prepared differently. The message lists the differences. Set `preprocessing_mismatch` to `'warn'` to continue anyway. Models saved before this release load and run as before, with a warning that they carry no full record.

### New

- `model_type='custom'` builds spaCR's small CustomCellClassifier, trained from scratch, for any number of channels. Its outputs are now plain logits.
- Classify shows `input_statistics`. `'auto'` uses a pretrained backbone's own mean and standard deviation, and dataset-level statistics for a model trained from scratch.
- Test reports list `majority_class_accuracy`, the accuracy of always predicting the largest class, as a control.

### Alpha

- `crop_scaling='fixed'` trains on raw crops. Each plate and channel is divided by one fixed scale taken from the training crops, then clipped, resized and normalised on the GPU. Augmentation uses flips, rotations, small shifts and a small per-channel gain. It needs raw crops.
- `control_shuffled_labels` trains a shuffled-label control model.
