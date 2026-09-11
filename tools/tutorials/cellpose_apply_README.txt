Apply Cellpose: three real crops and the recorded stock-model settings
====================================================================

Use a newly extracted copy of images/ for this demonstration. It contains
three unchanged 512 x 512 single-channel uint16 TIFF crops from real downloaded
spaCR microscopy fields. See source_manifest.json for identities and SHA-256.
These are examples, not independent segmentation ground truth.

Open Make Masks -> Cellpose Workbench -> Apply. Set src to the absolute path
of your extracted images directory, not the historical workstation path in
recorded_settings.json. Use the actual settings search to find the controls.
The JSON is a reference, not a claim that GUI JSON import was demonstrated.

Recorded choices: model_name cpsam, custom_model None, channels [0], normalize
True, percentiles [2,99], diameter 30, CP_prob 0, flow_threshold 0.4, batch_size
1, save True, verbose True. Rescale, resample, resize, fill_in, invert and
remove_background were False. The other values are in recorded_settings.json.
These are demonstration choices, not optimal settings for other experiments.
The stock CPSAM checkpoint is not bundled; the recorded SHA-256 is
e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2.
No custom model from the Train lesson is selected.

Run generated one same-named TIFF per input in images/masks and three genuine
GUI figures. The recorded counts for cell_pair_01/02/03 were 9, 94 and 46.
The console named an RTX 3090 using CUDA. This is not a speed comparison with
CPU, MPS or another GPU. Figures are produced on this route without a separate
Plot control. Original, external foreground outlines and a single channel of
the flow visualization were checked against the actual arrays. Red outlines
do not trace every touching instance boundary; flow colours are not physical
motion or accuracy. Independent direct-model calls reproduced all saved pixels.

Live preview uses a DIFFERENT inference/preprocessing route in this build.
For cell_pair_02.tif it produces 95 objects rather than the batch's 94, even
after choosing channel 0 and displayed percentiles 2/99. The independent check
matches raw-image Cellpose defaults, not the batch's explicit per-field 2/99
stretch followed by normalize=False and resample=False. Do not promise that
the preview or its propagated controls reproduce the saved batch mask.

In Live settings, a minimum cell area of 1838 pixels changes the preview from
95 to 47 objects without rerunning Cellpose. Returning it to zero restores
every label pixel. This changes only the demonstrated preview, not the TIFFs
already saved. Close the dialog to see the result. The native preview card is
still narrow: wheel zoom gives a cropped detail view on both canvases, not a
larger card or a full-field view. Filtering resets zoom; reapply wheel zoom if
needed. No application layout or preprocessing defect was fixed for the video.

No provider request, human annotation review or biological accuracy claim.
Keep the source images, settings, model identity and output masks together,
and evaluate independently before using a model for scientific conclusions.
