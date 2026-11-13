# Classify images with a neural network

**Purpose.** Train a convolutional image classifier from labelled object crops and apply it to the experimental dataset.

**Recommended use.** Use when phenotype classes are visually distinguishable but are not adequately represented by predefined measurements.

**Primary outputs.** A trained model, held-out performance metrics, and per-object class probabilities or scores.

**Desktop route.** Classify → Computer Vision

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.deep_spacr.deep_spacr`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.deep_spacr)

```python
deep_spacr(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.deep_spacr.deep_spacr`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.deep_spacr)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`classes`** *(conditionally required)* — (dict) - Class definitions in the form class name -&gt; {column, value}; for example, 'pc' might be {'column': 'columnID', 'value': 'c3'}. In the Classes editor, select a column and assign names to its distinct values; definitions may span several columns. One row may instead define a random complement containing objects not assigned by any other class, sampled to match the largest class. This setting defines which objects belong to each class; class_folder_names defines the training subfolders. A pre-split settings file contains a plain list and is converted when read. Default {}.
- **`class_folder_names`** *(conditionally required)* — (list of str) - Ordered training folder names. Each must exactly match a subfolder under src/train and src/test; its position becomes the integer label, and the list length sets the classifier-head width. Training raises FileNotFoundError with missing and available folders when a name is absent. Generate Training Dataset replaces this list with the folders it wrote. This setting identifies crop locations; 'classes' defines their semantic labels. Default ['nc','pc'].
- **`train`** *(conditionally required)* — (bool) - Run the training stage. Disable this setting to apply an existing model_path to a dataset without retraining, as when scoring a new plate with a previously trained model. Default True.
- **`apply_model_to_dataset`** *(conditionally required)* — (bool) - After training (or straight away when reusing a saved model_path), pack the object PNGs into a tar, run inference over it, copy the n_top_examples most confident images per class into top_examples/, and merge the per-object scores back into measurements.db. Turn it off to only train and evaluate a model without scoring the screen. Default True.
- **`dataset`** *(conditionally required)* — (str) - Path to the .tar archive of single-object PNG crops produced by generate_dataset, which the activation-map step opens with TarImageDataset. The plate folder is inferred two levels above it and CAM outputs are written next to it under &lt;tar_name&gt;/&lt;cam_type&gt;/. Provide an absolute or directory-qualified path rather than a filename alone. Default ''.
- **`model_path`** *(conditionally required)* — (str) - Path to a trained spaCR classifier saved as a whole PyTorch object (loaded with torch.load(weights_only=False), not a state_dict). Used when applying a model to a dataset tar and when generating activation maps. deep_spacr overwrites it with the freshly trained model whenever train is True, so set it only to score with an existing model. Default ''.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Classifier (1 settings)</summary>

#### Classifier

- **`learning_rate`** *(optional)* — (float) - Initial optimizer step size. Values that are too high may prevent convergence; values that are too low may slow convergence or converge to a suboptimal solution. A value near 1e-3 is commonly used for training from random initialization, while 1e-4 to 1e-5 is appropriate for fine-tuning ImageNet weights (init_weights=True). The selected schedule modifies this initial value during training. Default 0.001. Default: `0.001`.

</details>

<details>
<summary>Plate Sources &amp; Workflow (7 settings)</summary>

#### Plate Sources & Workflow

- **`experiment`** *(optional)* — (str) - Free-text run label. Its real effect is naming the exported PNG dataset tar as &lt;YYMMDD&gt;_&lt;experiment&gt;.tar (a random-numbered variant is used if that name already exists), so give each screen a distinct value to avoid confusing dataset tars. It is also passed to the measurement-database writer but not stored there. Default 'experiment' (the barcode pipeline uses 'experiment_1' and a foreign import uses 'foreign_import'). Default: `'experiment'`.
- **`generate_training_dataset`** *(optional)* — (bool) - Rebuild the train/ and test/ PNG folders from the object crops before training, using the dataset_mode rules (annotation_column labels, metadata rules or measurement rules) and splitting off test_split of the images. Turn it off to reuse an existing split; it is only consulted when train or test is True, and a failed build aborts training. Default True. Default: `True`.
- **`test`** *(optional)* — (bool) - In classifier training, run the held-out evaluation pass, either with train or alone to score an existing model. In the sequencing barcode mapper, process only the first read chunk and print a preview so the regex and barcode CSVs can be validated quickly. Default False. Default: `False`.
- **`generate_full_dataset`** *(optional)* — (bool) - Build the full unlabelled inference dataset tar from every selected plate independently of training or model application. Apply model to dataset also creates it automatically when needed. API: spacr.io.generate_dataset. Default False. Default: `False`.
- **`tar_path`** *(optional)* — (str) - Existing full-dataset tar to reuse for inference. Leave blank to generate one beneath the first plate's datasets folder. Multiple selected plates are combined into one tar. API: spacr.deep_spacr.apply_model_to_tar. Default empty. Default: `''`.
- **`channel_of_interest`** *(optional)* — (int, list, or str) - Measurements available to the model. Specify one channel to train on that channel alone, multiple channels to use their combination of measurements, or 'shape' to use outline-derived features. An empty value includes every measurement. A colocalisation feature is associated with both measured channels, so selecting either channel also includes their shared colocalisation features. This setting also selects the channel used for recruitment measurements. Default 3 in machine-learning steps and 1 or 2 elsewhere. Default: `3`.
- **`tables`** *(optional)* — (list) - Measurement tables read from each plate's database and merged into one analysis frame. Supported tables are registered object roles, including organelles, and png_list. For tabular Classify, choose a nonempty list of distinct supported tables with at least one object-role table. Default ['cell', 'nucleus', 'pathogen', 'cytoplasm']. Default: `['cell', 'nucleus', 'pathogen', 'cytoplasm']`.

</details>

<details>
<summary>Labels &amp; Classes (5 settings)</summary>

#### Labels & Classes

- **`dataset_mode`** *(optional)* — (str) - How training classes are defined: 'metadata' splits crops by well metadata, 'annotation' by the values in one or more annotation columns of png_list. Either way the classes themselves are set in the Classes editor, which names a column and a value per class. A settings file written before the 'measurement' basis was removed still loads: it is read as 'annotation', which is what its threshold rules resolved to after writing their label column. Any other value aborts and returns no dataset. Default 'metadata'. Default: `'metadata'`.
- **`balance_to_smallest`** *(optional)* — (bool) - Downsample every generated training class to the size of the smallest class before writing train/test folders. This removes the dataset prior but discards majority examples; disable it and use class_balance during training to retain all images. Default True. Default: `True`.
- **`test_split`** *(optional)* — (float) - Fraction of the generated crops held out as the test set, between 0 and 1. The split respects the grouping level chosen elsewhere, so crops from one well do not straddle it and the score is not inflated by the model recognising the well. Raising it buys a steadier estimate and costs training data. Default 0.1. Default: `0.1`.
- **`val_split`** *(optional)* — (float) - Fraction of src/train randomly held out as a validation set in each run (0.1 = 10 percent). The validation score controls checkpoint selection, early stopping and live training curves; at 0 there is no validation loader, so checkpointing uses training accuracy and may favour memorisation. Increase it on small datasets for a less variable estimate. When a grouping level is set, complete groups are held out, so the realised fraction is quantised and may differ substantially from the requested value; both values are reported. Default 0.1. Default: `0.1`.
- **`sample`** *(optional)* — (int, list or None) - Randomly draw this many PNG crops from the database when building the dataset tar instead of using all of them; a list uses its first element, and values above the total are clamped. Use it to build a quick trial dataset or to cap a huge screen. None uses every crop, shuffled. Default None. Default: `None`.

</details>

<details>
<summary>Computer Vision — Images &amp; Cropping (14 settings)</summary>

#### Computer Vision — Images & Cropping

- **`image_source`** *(optional)* — (str) - Source of classification images. 'load_images' reads previously exported object crops; 'stream_images' generates crops from merged image and mask arrays during training. Streaming avoids creating a separate export for each object, channel, and crop-shape combination. Default 'load_images'. Default: `'load_images'`.
- **`stream_method`** *(optional)* — (str) - Method used to locate objects for streamed crops. 'column' uses coordinates stored in the object table and requires object_array and channel_arrays. 'array' uses labelled objects in the same object_array plane and additionally requires bounding_box. Settings that do not apply to the selected method are ignored. Default 'column'. Default: `'column'`.
- **`object_array`** *(optional)* — (str) - On-demand crops: which object the crops are cut around - 'cell', 'nucleus', 'pathogen', 'cytoplasm' or 'organelle'. Its mask plane in merged/*.npy is what defines each object's extent. Default 'cell'. Default: `'cell'`.
- **`channel_arrays`** *(optional)* — (list[int]) - Zero-based intensity-plane indices included in each streamed image, in output-channel order. This setting applies to both stream methods; changing the order changes the channel mapping presented to the model. Default [0, 1, 2]. Default: `[0, 1, 2]`.
- **`bounding_box`** *(optional)* — (bool) - Crop geometry used when stream_method is 'array'. True retains the rectangular region enclosing each labelled object, including local background and neighbouring signal. False retains only pixels within the object mask and sets surrounding pixels to zero. Default True. Default: `True`.
- **`crop_shape`** *(optional)* — (str) - 'bounding_box' cuts the smallest rectangle containing the object; 'object' masks everything outside it away. Database-sourced crops can only be bounding boxes. Default 'bounding_box'. Default: `'bounding_box'`.
- **`train_channels`** *(optional)* — (list) - Which colour planes of each object crop the classifier sees, chosen from 'r', 'g' and 'b'. Fewer channels means a smaller input tensor and a model that cannot use the dropped stain, so drop a channel only when it carries no signal for your phenotype. The joined letters also become part of the saved model's filename. Default ['r', 'g', 'b']. Default: `['r', 'g', 'b']`.
- **`image_size`** *(optional)* — (int) - Side length in pixels of the center crop taken from each object PNG before model input. Images are cropped rather than rescaled, so larger values add zero padding and smaller values discard peripheral object pixels. This value also defines the backbone input resolution and must match the crop size used to generate the dataset. Default 224. Default: `224`.
- **`augment`** *(optional)* — (bool) - Expand the training split eightfold by adding four 90-degree rotations of each crop and their horizontal reflections; validation and test splits are not augmented. Enable this setting when few annotated objects are available and validation accuracy is below training accuracy. The expanded set is materialised in RAM, requiring approximately eight times the memory and epoch duration. Default False. Default: `False`.
- **`crop_scaling`** *(optional)* — (str) - How raw crops become model input. 'legacy' decodes each crop to 8 bits with the per-crop stretch, as before. 'fixed' divides each plate and channel of the raw crops by one fixed scale (the 99th percentile of object pixels in up to 2,048 training crops), clips at crop_scale_clip, resizes to image_size and normalises with input_statistics on the GPU; it keeps intensity differences between cells and needs Measure crop_intensity 'raw'. With augment: flips, 90-degree rotations, shifts of up to 4 px and a per-channel gain within 10%. Every step is saved in the checkpoint. Default 'legacy'. Default: `'legacy'`.
- **`crop_scale_clip`** *(optional)* — (float) - Upper clip after the fixed crop scale, so a value of 1.5 keeps signal up to 1.5 times the 99th percentile. Used only when crop_scaling is 'fixed'. Default 1.5. Default: `1.5`.
- **`crop_scale_file`** *(optional)* — (str or None) - Scales file for crop_scaling 'fixed'. An existing file is reused, so several models can share one scale. None writes crop_scales.json in the dataset folder. Its md5 is saved in the checkpoint. Default None. Default: `None`.
- **`input_statistics`** *(optional)* — (str) - Mean and standard deviation applied to the model input. 'symmetric' is spaCR's historical 0.5/0.5, mapping [0,1] to [-1,1]. 'imagenet' and 'clip' use those backbones' statistics; with crop_scaling 'fixed', 'imagenet' takes the mean, std and interpolation of the model's torchvision weights. 'dataset' uses per-channel training statistics. 'custom' uses input_mean and input_std. 'auto' picks 'imagenet' for a pretrained backbone, else 'dataset'. Legacy training, and 'auto' when scoring or drawing activation maps, use 'symmetric'. Default 'auto' in Classify, 'symmetric' elsewhere. Default: `'auto'`.
- **`preprocessing_mismatch`** *(optional)* — (str) - What testing a model does when the crops would be prepared differently from training, for example another image_size, channel selection, mask mode or scales file. 'refuse' stops with a message that lists the differences. 'warn' prints them and continues. Models saved before November 2026 have no full record, so they are tested with the current settings and a warning. Default 'refuse'. Default: `'refuse'`.

</details>

<details>
<summary>Computer Vision — Model &amp; Regularization (8 settings)</summary>

#### Computer Vision — Model & Regularization

- **`model_type`** *(optional)* — (str) - Backbone architecture for the single-object image classifier: any TorchVision classification model name (resnet50, maxvit_t, densenet121, ...). An unrecognized name does not fail during initial validation: choose_model reports 'Invalid model_type' and returns None, after which training fails. The special name 'custom' builds spaCR's small CustomCellClassifier, which is trained from scratch and accepts any channel count. Larger backbones require more memory and generally need more labeled crops than smaller backbones. Default 'maxvit_t'. Default: `'maxvit_t'`.
- **`custom_model_path`** *(optional)* — (str) - Path to a trained classifier artifact whose model weights initialize a new fine-tuning run. The optimizer and epoch start fresh. Leave empty to initialize from ImageNet or random weights according to init_weights. Default ''. Default: `''`.
- **`resume_checkpoint`** *(optional)* — (str) - Path to a spaCR training artifact to continue exactly: restores model, optimizer, scheduler, epoch, best score and random-generator state. Use custom_model_path instead when only the weights should be reused. Default ''. Default: `''`.
- **`init_weights`** *(optional)* — (bool) - Start the backbone from ImageNet-pretrained weights instead of random initialisation; the spaCR classifier head added on top is randomly initialised either way. Leave it on - transfer learning converges in far fewer epochs on the small annotated sets typical here. Turn it off only to train from scratch on a very large dataset, or to measure how much pretraining contributes. Default True. Default: `True`.
- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.
- **`dropout_rate`** *(optional)* — (float) - Dropout probability (0-1) written into every existing Dropout layer of the backbone and applied to a Dropout inserted before the final linear classifier; 0 or None removes dropout entirely. Raise it (0.2-0.5) when training accuracy runs well ahead of validation accuracy; lower it when the model underfits and training loss stalls high. Default 0.1. Default: `0.1`.
- **`weight_decay`** *(optional)* — (float) - L2 penalty applied to the weights on every optimizer step (AdamW applies it decoupled from the gradient). Raise it, toward 1e-3 to 1e-2, when validation loss climbs while training loss keeps falling; lower it toward 0 when the model cannot fit the training set at all. Every supported optimizer honours it. Default 0.00001. Default: `1e-05`.
- **`use_checkpoint`** *(optional)* — (bool) - Run the backbone's forward pass through torch.utils.checkpoint: intermediate activations are discarded and recomputed during the backward pass, trading extra compute for a large drop in activation memory. Enable when a bigger batch_size or image_size gives CUDA out-of-memory; disable for the fastest epochs when VRAM is not the constraint. Default True. Default: `True`.

</details>

<details>
<summary>Computer Vision — Training &amp; Loss (16 settings)</summary>

#### Computer Vision — Training & Loss

- **`epochs`** *(optional)* — (int) - Number of full passes over the training set. It also sets the learning-rate schedule horizon - cosine anneals over exactly this many epochs and step_lr drops every epochs/5 - so changing it rescales the schedule. A checkpoint is always written on the final epoch and every 100th. Raise it for small datasets and use early_stopping_patience to cut runs short. Default 100. Default: `100`.
- **`optimizer_type`** *(optional)* — (str) - PyTorch optimizer used by deep_spacr.train_model: 'adamw', 'adam', 'adamax', 'sgd', 'rmsprop', 'nadam', 'radam', 'adagrad', 'adadelta' or 'asgd'. AdamW is the default for fine-tuning; SGD can generalise better but usually needs more epochs. amsgrad applies only to Adam/AdamW. API: spacr.deep_spacr.train_model(optimizer_type=...). Default 'adamw'. Default: `'adamw'`.
- **`schedule`** *(optional)* — (str) - Learning-rate scheduler used by spacr.deep_spacr.train_model: 'cosine', 'cosine_warm_restarts', 'reduce_lr_on_plateau', 'step_lr', 'exponential', 'linear', or 'none'. Plateau reacts to validation loss; cosine and linear use the epoch budget; warm restarts periodically raise the rate to escape a narrow minimum. API: train_model(schedule=...). Default 'cosine'. Default: `'cosine'`.
- **`amsgrad`** *(optional)* — (bool) - Use the AMSGrad variant of Adam/AdamW, which keeps a running maximum of past squared gradients instead of their decaying average so the effective step size never grows back. Enable when training loss oscillates or stops converging with plain Adam; it costs a little speed and memory. Only honoured by optimizer_type 'adam' and 'adamw' - ignored by sgd, rmsprop, nadam, radam and adagrad. Default True. Default: `True`.
- **`loss_type`** *(optional)* — (str) - Loss used to train the classifier. For a head with two or more classes: 'focal_loss' (down-weights easy examples), 'cross_entropy', 'label_smoothing' (epsilon 0.1), 'ce_weighted' (inverse-frequency class weights), 'logit_adjust_ce' and 'asl'. 'binary_cross_entropy_with_logits' is valid only for a single-logit head and raises otherwise. Use a weighted or focal loss for imbalanced classes. Default 'focal_loss'. Merged Classifier starts at 'auto', which resolves to cross_entropy for a multi-class head and binary_cross_entropy_with_logits for a single-logit head. Default: `'auto'`.
- **`class_balance`** *(optional)* — (str) - Correction for imbalance among training classes. 'none' preserves sampling and reports class counts, their ratio and a recommendation. 'weighted_sampler' samples classes approximately equally using 1/n. 'sqrt_weighted_sampler' uses 1/sqrt(n), reducing the risk that repeated sampling of a very small class causes memorization. 'weighted_loss' preserves sampling and weights the loss. Use one of the latter three when the reported ratio exceeds approximately 3:1. Default 'none'. Default: `'none'`.
- **`label_smoothing`** *(optional)* — (float) - Epsilon passed to cross-entropy when loss_type is label_smoothing: each target retains 1 - eps of its probability mass and distributes the remainder across the other classes. Increase it, typically within 0.05-0.2, when predicted probabilities are overconfident or annotations are noisy; 0 disables smoothing. Ignored by other loss types. Default 0.1. Default: `0.1`.
- **`focal_gamma`** *(optional)* — (float) - Focusing exponent in the focal-loss weight (1 - p_t)^gamma, applied only when loss_type is focal. 0 reduces it to cross-entropy; increasing it, typically within 1-5, reduces the contribution of confidently classified crops and increases the relative contribution of difficult samples. Default 2.0. Increase when class imbalance causes training to be dominated by easy examples. Default: `2.0`.
- **`focal_alpha`** *(optional)* — (float) - Class-balancing weight for focal loss (read only when loss_type resolves to focal). In the single-logit binary path it scales positives by alpha and negatives by 1-alpha, so raise it toward 1 to emphasise a rare positive class; with two or more output classes a plain float scales the whole loss uniformly. Default None (no alpha weighting). Default: `None`.
- **`logit_adjust_tau`** *(optional)* — (float) - Strength of the Menon-et-al. logit adjustment: tau * log(class prior) is added to the logits during training, pulling decisions toward rare classes. Only used when loss_type resolves to logit_adjust_ce, which 'auto' picks when the smallest class is under 10% of the data. Higher tau corrects harder; 0 disables. Default 1.0. Default: `1.0`.
- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `64`.
- **`mixed_precision`** *(optional)* — (bool) - Enable automatic mixed precision for supported accelerator operations during training. On a modern graphics card it can make training roughly twice as fast while using less memory, although the gain depends on the model and hardware. Floating-point differences can affect exact reproducibility, so compare runs only when they use the same precision mode. Unsupported hardware disables this mode with an explicit message. Default False. Default: `False`.
- **`train_metrics_full_pass`** *(optional)* — (bool) - How each epoch's training loss and accuracy are measured. Off, they are accumulated from the training batches as the model learns, in the same mode and precision, so no extra pass over the data is needed; the weights change during the epoch and dropout and augmentation are on. On, the whole training set is evaluated again after every epoch in evaluation mode and full precision, as earlier versions did, at the cost of one more pass. Validation metrics, best-checkpoint choice and early stopping do not depend on it. Default False. Default: `False`.
- **`gradient_accumulation_steps`** *(optional)* — (int) - How many batches are summed per optimizer step when gradient_accumulation is on; the loss is divided by this value so gradient magnitude stays comparable. Effective batch size = batch_size x this. Raise it (4-16) to emulate a larger batch on limited VRAM, at the cost of fewer weight updates per epoch. Ignored when gradient_accumulation is False. Default 4. Default: `4`.
- **`early_stopping_patience`** *(optional)* — (int) - Stop training after this many consecutive epochs in which validation accuracy fails to beat the best value so far; the best checkpoint is still kept. 0 (default) disables it and always runs the full 'epochs' budget. Set 10-20 on long runs to cut wasted epochs once the model plateaus. Default: `0`.
- **`control_shuffled_labels`** *(optional)* — (bool) - Train a shuffled-label control. The labels of the training split are permuted with random_seed, while validation and test labels stay true. A model that learns real phenotype should beat this control clearly. The run is saved in a separate folder whose name ends in _shuffled_labels. Test reports also list majority_class_accuracy, the accuracy of always predicting the largest class. Default False. Default: `False`.

</details>

<details>
<summary>Computer Vision — Test-time augmentation (7 settings)</summary>

#### Computer Vision — Test-time augmentation

- **`tta_enabled`** *(optional)* — (bool) - Score selected rotated or reflected versions of each phenotype crop during inference. Choose transforms in this category; the original view is always included. Saves original predictions, mean probabilities, standard deviations, agreement and review flags. Keep disabled when orientation carries biological meaning. Does not change training augmentation. Default False. Default: `False`.
- **`tta_rotations`** *(optional)* — (bool) - With test-time augmentation enabled, include 90, 180 and 270 degree rotations. Uses exact pixel rotations without interpolation. Default False. Default: `False`.
- **`tta_horizontal_flip`** *(optional)* — (bool) - With test-time augmentation enabled, include horizontal reflections of the selected rotations. Each distinct reflected view is classified before aggregation; equivalent orientations are evaluated once. Disable when horizontal direction has biological meaning in the assay. Default False. Default: `False`.
- **`tta_vertical_flip`** *(optional)* — (bool) - With test-time augmentation enabled, include vertical reflections. Equivalent orientations are evaluated once; all rotations and both flips produce eight distinct views. Default False. Default: `False`.
- **`tta_aggregation`** *(optional)* — (str) - Combine test-time views using probability_mean or majority_vote. Voting ties prefer the higher mean probability, then the lower class index. pred retains the mean probability; predicted_label and cv_predictions retain the selected label. Default probability_mean. Default: `'probability_mean'`.
- **`tta_min_agreement`** *(optional)* — (float) - Flag an object for review when fewer than this fraction of orientation predictions agree with the selected label. Agreement measures orientation stability, not calibrated confidence. Range 0-1; default 0.75. Default: `0.75`.
- **`tta_max_std`** *(optional)* — (float) - Flag an object when its positive-class probability (binary) or selected-class probability (multiclass) has a population standard deviation above this value across orientations. Range 0-1; default 0.15. Default: `0.15`.

</details>

<details>
<summary>Machine Learning — Held-out Evaluation α (4 settings)</summary>

#### Machine Learning — Held-out Evaluation α

- **`train_eval_split`** *(optional)* — (str) - Held-out evaluation of the classifier, as in the machine-learning module: leave_one_well_out trains on every well but one and tests on that one, in turn; leave_one_plate_out does the same per plate; random uses the stratified cross_validation_folds folds. No crop of a test well or plate is seen in training. The pooled and per-fold accuracy, balanced accuracy, macro F1 and AUROC go to ml_eval/metrics.csv in the model folder, beside the controls of train_eval_controls. Turns cross-validation on. none keeps the usual split. Default none. Default: `'none'`.
- **`train_eval_controls`** *(optional)* — (str) - Controls reported beside the classifier when train_eval_split is set, comma-separated. majority always predicts the commonest class; shuffled runs the whole cross-validation again with the training labels permuted within each fold, train_eval_shuffle_repeats times, and averages it. Blank reports no control. Default majority, shuffled. Default: `'majority, shuffled'`.
- **`train_eval_shuffle_seed`** *(optional)* — (int) - Seed of the first shuffled-label control run of train_eval_split; the next runs use the following seeds. Default 0. Default: `0`.
- **`train_eval_shuffle_repeats`** *(optional)* — (int) - How many times the shuffled-label control of train_eval_split retrains every fold; its row is the mean. Each repeat costs as much as the classifier itself. Default 1. Default: `1`.

</details>

<details>
<summary>Evaluation (17 settings)</summary>

#### Evaluation

- **`cross_validation_enabled`** *(optional)* — (bool) - Enable k-fold validation for Classify. If cross_validation_folds is 0 or 1, enabling this uses 5 folds. Use cv_group_by='plate' to hold out whole plates, or 'well'/'field' for within-plate validation without leaking related crops between training and validation. Default False. Default: `False`.
- **`cross_validation_folds`** *(optional)* — (int) - Number of k-fold splits used to train the vision classifier instead of one val_split holdout. 0 (the default) or 1 uses one random split; 2 or more trains a separate model per fold, evaluates each model on its held-out fold, and reports the mean, fold-to-fold standard deviation, and range. These statistics quantify sensitivity to the data partition. Runtime is approximately proportional to k. Distinct from 'cross_validation', which controls the feature-based (ml) classifier. Default: `0`.
- **`cv_group_by`** *(optional)* — (str) - Train/test independence: 'cell', 'field', 'well' (default), or 'plate'. Cell can place sibling crops from one well on both sides; field narrows but does not close that leak; well matches the usual experimental assignment unit; plate holds out a complete batch. Whole groups make the requested fraction approximate, so runs report held-out groups and cells. Legacy 'none'/'off' alias 'cell'. Crop identities come from spaCR's plate_well_field_object.png names; unverifiable grouped designs are refused rather than silently randomized. Default: `'well'`.
- **`cv_group_column`** *(optional)* — (str) - Leave-one-group-out cross-validation by a metadata column, for example compound: with cross-validation on, every distinct value of the column is held out once, so there is one fold per value and cross_validation_folds is not used. plate, well and field are read from the crop names; any other column is looked up by well in well_metadata_path. cv_group_by still names the identity that may never cross a fold, so keep it at well or field unless every plate holds one group. Cannot be combined with nested_cv_inner_folds. Blank keeps the stratified folds. Default blank. Default: `''`.
- **`cv_exclude_batch_column`** *(optional)* — (str) - Not-same-compound-or-batch rule for cross-validation: the column naming each crop's batch, such as plate or an imaging week. Each fold then trains without every crop from a batch among its held-out crops, so a held-out compound is never scored by a model that saw its plate or week. plate, well and field come from the crop names, other columns from well_metadata_path. The fold-composition table lists the excluded batches and any class left without training crops. Not combinable with nested_cv_inner_folds. Default blank. Default: `''`.
- **`well_metadata_path`** *(optional)* — (str) - Plate-map table (CSV, Excel or Parquet) with one row per well: plateID, the well (wellID such as A01, or rowID and columnID) and metadata columns such as compound, concentration or imaging batch. Classify reads the columns named in cv_group_column, cv_exclude_batch_column and prediction_treatment_column from it, joined to each crop by plate and well. A crop whose well is missing is refused for fold building. Default blank. Default: `''`.
- **`prediction_treatment_column`** *(optional)* — (str or list) - Plate-map column, or columns, naming each well's treatment, for example compound, or compound and concentration, read by well from well_metadata_path. Classify always writes &lt;stem&gt;_per_well.csv (objects, predicted-class counts, mean class probabilities); with this set it also writes &lt;stem&gt;_per_treatment.csv, where each class probability is the mean over the treatment's wells and the predicted class is the most probable. Cross-validation adds the true label and prints treatment accuracy. Default blank. Default: `''`.
- **`holdout_plate`** *(optional)* — (str | list | None) - Train without this plate and score on it. None -- the default -- splits within the available data. Cross-validation can otherwise learn plate-associated variation rather than the phenotype while retaining apparently strong metrics. Performance on a held-out plate directly evaluates generalization across plates. The run is rejected if either split would omit a class. Default: `None`.
- **`nested_cv_inner_folds`** *(optional)* — (int) - Number of inner grouped folds used inside every outer CV fold. 0 (default) keeps the faster ordinary grouped CV; 2 or more trains one inner model per fold, uses inner validation for early stopping/model selection, ensembles those models, and evaluates only once on the untouched outer fold. Runtime is approximately outer_folds x inner_folds training runs, but the outer score is not reused for tuning. API: spacr.classifier_evaluation.nested_group_folds. Default: `0`.
- **`score_threshold`** *(optional)* — (float) - Probability cutoff (0-1) applied to the model's positive-class score when deriving binary predictions. With test-time augmentation, this threshold labels each view before agreement and voting are calculated. Probability scores remain available. Default 0.5. Default: `0.5`.
- **`classifier_evaluation`** *(optional)* — (bool) - Build the Classifier Evaluation workbench bundle from out-of-fold predictions after Classify (CV): sample-level predictions, confusion matrices, reliability curves, calibrated probabilities, per-plate metrics, leakage reports and a manifest. It requires cross_validation_folds &gt;= 2; a single train/validation split cannot produce unbiased out-of-fold diagnostics. Default True. API: spacr.classifier_evaluation.evaluate_predictions. Default: `True`.
- **`evaluation_calibration`** *(optional)* — (str) - Probability calibration written to the evaluation bundle. 'temperature' cross-fits one scalar temperature per held-out fold using all other out-of-fold predictions, so a sample never fits its own calibrator; 'none' retains raw softmax probabilities. Calibration changes reported probabilities, not the saved model weights. Default 'temperature'. API: spacr.classifier_evaluation.cross_calibrate_probabilities. Default: `'temperature'`.
- **`evaluation_bins`** *(optional)* — (int) - Number of equal-width probability bins in reliability curves and expected calibration error. Values around 10 balance resolution against noise; use fewer bins for small validation sets and more only when every class has many hundreds of out-of-fold samples. Minimum 2, default 10. API: spacr.classifier_evaluation.calibration_table. Default: `10`.
- **`evaluation_fail_on_leakage`** *(optional)* — (bool) - Stop Classify (CV) before fitting a fold when the same object, augmentation family, or protected cv_group_by identity appears in both train and validation. False records the problem and continues, which is useful only for diagnosing a legacy dataset because its performance estimate remains invalid. Default True. API: spacr.classifier_evaluation.audit_split_leakage. Default: `True`.
- **`leakage_audit_train_test`** *(optional)* — (bool) - Audit the permanent train/ and test/ boundary before any classifier fit. Checks plate/well/field/object lineage, exported augmentation families and (when enabled) byte-identical renamed copies. Default True. API: spacr.classifier_evaluation.audit_dataset_splits. Default: `True`.
- **`leakage_hash_content`** *(optional)* — (bool) - SHA-256 hash classifier images during leakage audits so an identical crop copied or renamed across train/test or CV boundaries is still detected. Reads files in 1 MiB chunks and never decodes pixels. Default True. API: spacr.classifier_evaluation.audit_cv_folds. Default: `True`.
- **`leakage_require_identity`** *(optional)* — (bool) - Treat filenames that do not encode the protected cv_group_by identity, and files that cannot be hashed, as a failed audit rather than an advisory warning. Default True because independence cannot be claimed when lineage is unknown. API: spacr.classifier_evaluation.audit_split_leakage. Default: `True`.

</details>

<details>
<summary>Results &amp; figures (4 settings)</summary>

#### Results & figures

- **`n_top_examples`** *(optional)* — (int) - Number of highest-confidence images saved per predicted class after full-dataset inference. This gives a quick visual check of class meaning and common errors. Default 20. Default: `20`.
- **`plot`** *(optional)* — (bool) - Render and save quality-control figures during the pipeline, including channel montages, Cellpose mask overlays, filtration comparisons, and crop grids. Figure generation increases runtime and memory use, particularly for complete plates. test_mode enables this setting automatically. Default False. Merged Classifier and Recruitment both start with plotting enabled so their diagnostic figures are produced on the first run. Default: `True`.
- **`tensorboard`** *(optional)* — (bool) - Write PyTorch loss, accuracy, macro-F1, and learning-rate events to dst/tensorboard while the vision model trains. Run tensorboard --logdir &lt;path&gt; with that directory to open an interactive dashboard and compare runs. The in-app loss and accuracy monitor is controlled separately by plot. Default True. Default: `True`.
- **`intermedeate_save`** *(optional)* — (bool, sequence of float, or None) - Control archival model snapshots on improving epochs. True or None uses validation-accuracy thresholds 0.99, 0.98, 0.95 and 0.94; False disables archival snapshots; and a sequence supplies custom thresholds. Best-model and last-model checkpoints remain enabled independently, so False does not remove those recovery artifacts. Default True. Default: `True`.

</details>

<details>
<summary>Runtime &amp; Reliability (7 settings)</summary>

#### Runtime & Reliability

- **`random_seed`** *(optional)* — (int) - Reproducibility seed shared by labelled train/test splitting, train/validation splitting, and grouped cross-validation folds. Keep it fixed to reproduce a run; vary it to evaluate sensitivity to the sampled partition. Default 42. Default: `42`.
- **`n_jobs`** *(optional)* — (int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None. Default: `max(1, (__import__('os').cpu_count() or 1) - 4)`.
- **`ram_guard`** *(optional)* — (bool) - Keep worker pools from filling RAM. Before any module starts its workers, spaCR estimates each worker's memory from one input (a field, mask, crop batch or table) and lowers n_jobs to the count that leaves 12.5% of RAM free, printing a warning; the GUI asks first. Off keeps your n_jobs, and Measure fields still wait while free RAM is below that reserve. Default True. Default: `True`.
- **`pin_memory`** *(optional)* — (bool) - Decode and hold the entire train/test image set in RAM up front (loaded in parallel across all cores) and hand batches to the GPU from page-locked memory. Enable when the dataset fits comfortably in RAM and disk I/O is the bottleneck; disable for large datasets or it will exhaust memory before the first epoch even starts. Default False. Default: `False`.
- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.
- **`strict_errors`** *(optional)* — (bool or None) - Error-handling policy for recoverable steps. Off records failures in the run ledger and final summary while continuing with successful items. On raises immediately for setup or configuration errors such as unreadable paths, missing columns or inaccessible databases, preventing partial batch results from invalid inputs. Per-item failures such as one corrupt image remain recoverable under either policy. None defers to $SPACR_STRICT_ERRORS. Default None. Default: `None`.
- **`max_failure_rate`** *(optional)* — (float or None) - Fraction of failed items above which the run aborts. For example, 0.2 aborts after more than 20% of items fail. The failure ledger is written to the artifact before the abort. None disables rate-based abortion; failures remain counted and reported, and incomplete artifacts are marked partial. Default None. Default: `None`.

</details>

<details>
<summary>GPU Memory α (4 settings)</summary>

#### GPU Memory α

- **`gpu_memory_fraction`** *(optional)* — (float or None) - Hard ceiling on the share of each GPU's memory this run may use, from 0 to 1, applied with PyTorch's per-process memory fraction. 0.25 lets the run take at most a quarter of the card, so several jobs can share it. Blank sets no ceiling and the run sizes its batches to the memory that is free. Default None. Default: `None`.
- **`gpu_memory_margin_gb`** *(optional)* — (float) - GPU memory, in GB, kept free for other jobs on the card. Before each batch spaCR reads the free GPU memory and, when the batch is estimated not to fit with this much to spare, shrinks the batch or waits for memory to be freed. Default 1.5. Default: `1.5`.
- **`gpu_adaptive_batch`** *(optional)* — (bool) - Size GPU batches to the memory that is free on a shared card. Segmentation, measurement, OPS and classifier runs shrink a batch that would not fit, halve it and retry the same item after an out-of-memory error, fall back to the CPU when batch size 1 does not fit, and grow the batch again after clean items; no item is dropped. spaCR processes share a GPU ledger in the user cache folder, so a new job counts what running jobs will take. Counts appear in the run log. Off runs every batch at the requested size. Default True. Default: `True`.
- **`gpu_fallback_policy`** *(optional)* — (str) - What a GPU item does when it does not fit in GPU memory even at batch size 1. per_item runs that item on the CPU and records its compute path (gpu or cpu) per item: Measure writes a compute_path column on its object rows when measure_gpu is on, the run report says whether the run stayed on one path, and a warning appears when more than 3 % of items fell back. pause never falls back: the item waits for GPU memory to be freed, so every result comes from the GPU path, at the cost of waiting while the card is full. Default per_item. Default: `'per_item'`.

</details>

<details>
<summary>Additional settings (6 settings)</summary>

#### Additional settings

- **`annotation_column`** *(optional)* — (str) - Integer column in the png_list table that stores manual class labels. The Annotate app adds it with ALTER TABLE if it is absent and writes labels to it. This column provides the reference labels when dataset_mode is 'annotation' and is the fallback when annotation_columns is unset. Supplying it while dataset_mode is unset also selects annotation mode for compatibility with older settings files. Default None. Default: `'test'`.
- **`class_metadata`** *(optional)* — (list of lists) - One inner list per training class, holding the metadata values that select that class's objects, for example [['c1'],['c2']] for a two-class run keyed on column. Order fixes the class indices the model learns, so reordering the inner lists relabels the whole training set. Values that occur in no row make the generator select nothing and stop. Default [['c1'], ['c2']]. Default: `[['c1'], ['c2']]`.
- **`coordinate_columns`** *(optional)* — (list or None) - Database columns used to locate an object for on-demand cropping. Two columns may give a centroid row and column; one column may identify a labeled object whose mask supplies its extent. Coordinate-backed routes support only bounding-box crops. None uses the configured mask plane instead. Default None. Merged Classifier derives one identifier from object_array, initially ['cell_id'], so the single-column form is valid rather than an incomplete coordinate pair. Default: `['cell_id']`.
- **`crop_source`** *(optional)* — (str) - Select where image crops come from. Viewers use 'png' (LOAD IMAGES) for exported crops in data/ or 'merged' (STREAM IMAGES) to cut from merged/*.npy using the measurements database. These viewer modes correspond to training's 'load_images' and 'stream_images' sources. spaCR reports any fallback, and controls that do not apply to the selected source are disabled. Default 'png' in viewers and 'load_images' in training. Image UMAP starts at 'auto', preferring exported PNGs when available and otherwise streaming from merged arrays. Default: `'load_images'`.
- **`file_metadata`** *(optional)* — (str, list, or None) - Substring filter applied to png_path when retrieving crops from the database. Only paths containing the supplied string are included; a list matches any entry rather than requiring all entries. Use this setting to restrict a dataset to one plate, well, or object type, for example 'plate1_' or 'cell_png'. None includes every crop. Default None. Default: `None`.
- **`file_type`** *(optional)* — (str) - Image format of the pre-generated crops, as a file extension: 'png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp' or 'npy'. It filters by format only; path_string selects the object type, so the pair can select, for example, every nucleus crop in any format or every TIFF of any object. A legacy value of the form '&lt;object&gt;_png' is still accepted and read as its extension, so an old settings file keeps working. Blank accepts any format. Default 'cell_png', which is read as 'png'. Default: `'cell_png'`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A trained model, held-out performance metrics, and per-object class probabilities or scores.

## Custom classification stages

- [`spacr.io.generate_training_dataset`](https://einarolafsson.github.io/spacr/api/spacr/io/index.html#spacr.io.generate_training_dataset)
- [`spacr.deep_spacr.train_test_model`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.train_test_model)
- [`spacr.io.generate_dataset`](https://einarolafsson.github.io/spacr/api/spacr/io/index.html#spacr.io.generate_dataset)
- [`spacr.deep_spacr.apply_model_to_tar`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.apply_model_to_tar)
