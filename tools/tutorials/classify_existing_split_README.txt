Classify: explicit existing-split tutorial example
================================================

This is a bounded teaching example, NOT a general-purpose dataset splitter.
It was recorded with spaCR 1.5.0.6. Follow the Annotate tutorial and choose
Load test data, then Load (not Stream). Retain the downloaded plate1 project,
including measurements/measurements.db, its image crops and format markers.
Do not replace the source annotations or edit the original database.

Activate your spaCR Python environment, extract this ZIP, and run:

  python prepare_classify_split.py --source /path/to/plate1 --destination /path/to/new_cv_example

Substitute your actual paths. The destination must not exist and must be
outside the source project. The helper refuses unknown/missing labels,
unexpected wells, changed storage formats and non-canonical identities.
Its path-prefix mapping targets this specific downloaded example, not any
arbitrary measurements database. If your download differs, stop and inspect
the discrepancy instead of removing the checks.

The helper uses the real png_list metadata to select eight existing label-1
and eight existing label-2 crops per well, ranked by a fixed SHA-256 rule
before fitting. It copies 64 original RGB224x224 PNGs unchanged, but names
the COPIES using their database prcfo identities. The application's legacy
filename parser is not repaired by this workaround. Format markers and an
input manifest accompany the copied files. No model is started by the helper.

Actual training-folder wells: plate1/r12/c1 and plate1/r5/c2.
Actual test-folder wells:     plate1/r12/c2 and plate1/r5/c1.
The deliberately balanced selection does not preserve population prevalence.
Existing labels are preserved, not independently biologically validated.

Open Home -> Classify -> Computer Vision (Torch).
Choose the NEW dataset root containing train/ and test/, not a class folder.
Use these explicit settings for the recorded bounded run:

  generate_training_dataset: False
  classifier_family: cv
  dataset_mode: annotation
  classes: {'infected_1': {'column': 'infected', 'value': 1}, 'infected_2': {'column': 'infected', 'value': 2}}
  model_type: resnet18
  epochs: 1
  batch_size: 8
  image_size: 128
  n_jobs: 0
  init_weights: False
  augment: False
  tensorboard: False
  train: True
  test: True
  plot: True
  val_split: 0.5
  test_split: 0.5
  cv_group_by: well
  random_seed: 42
  train_channels: ['r', 'g', 'b']
  generate_full_dataset: False
  apply_model_to_dataset: False

Keep every leakage check enabled. Do not replace these with a recipe that
regenerates a dataset from the original legacy filenames. The source choice
and actual loaded settings must agree before Run.

The 32 training-folder crops become 16 training and 16 validation crops in
separate wells. The 32 test crops remain held out. All four wells share one
plate: this is not validation on a new experiment. In the recording every
test image was predicted as class zero, giving 50% accuracy, exactly the
majority baseline, and macro F1 of 1/3. These settings did not yield a useful
model. Class folders infected_1 and infected_2 retain database labels 1 and 2;
the model indices are 0 and 1, respectively.

Keep model/resnet18/rgb/epochs_1 with its model card, checkpoints, leakage
audits, prediction/result CSV files, figures and tutorial_input_manifest.json
at the dataset root. The CSV named *_test_acc.csv contains prediction rows;
*_test_result.csv contains the metric summary. Review-image outputs are not
additional independent input samples. Recheck identities and true labels,
not just reported accuracy. Do not claim calibration or generalization from
this tiny demonstration. Later runs need new preserved output directories.

API route: spacr.classify.classify -> spacr.deep_spacr.train_test_model.
Related tutorials: Annotate, tabular Classify, Classifier Evaluation,
Training Runs, Activation Maps and the API introduction.
