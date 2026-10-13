TRAIN CELLPOSE: AN EXPLICIT, BOUNDED PYTHON API EXAMPLE
=====================================================

This is a tutorial workaround, not a repaired GUI and not a recommended model.
The recorded Train form drops the required source path during CSV import and
leaves the requested model name as new_model. Other numerical controls import.
Do not treat that partial import as a ready-to-run training configuration.

Unzip this example into a new folder. In the installed spaCR environment, from
that folder, run:

  python train_cellpose_example.py --source data --destination ../my_new_cellpose_demo

The destination must not exist and must be outside data. The example requires
a CUDA GPU; the recorded device was an NVIDIA GeForce RTX 3090. The helper caps
its CUDA allocation at 65% of device memory and uses two CPU threads. Run it
with your system-memory monitor; the recorded workstation used a process-tree
guard at 100 GiB total RAM in use. That absolute workstation limit is NOT a
safe default for a smaller computer. Do not launch competing training jobs.

The helper calls the installed spacr.submodules.train_cellpose API. A read-only
Python profiler observes the real Cellpose call and checks the actual input
arrays, figure arrays and returned checkpoint; it does not replace the model,
optimizer or training function. It saves the real Matplotlib figure as an
opaque PNG. The tutorial opens it in an external viewer, not spaCR Figures.

DATA AND LIMITATIONS
--------------------
The six pairs are exact 512 x 512 crops from the real downloadable tutorial
fields. Original recorded settings identify channel 1 as cell intensity and
plane 4 as cell masks; plane 6 is pathogen, not cell. Matching filenames in
data/train/images and data/train/masks preserve each pair. source_manifest.json
contains original field identities, crop origins and hashes. Its absolute
historical source_root is provenance only; it is not needed to run this bundle.

The existing instance labels have not been independently reviewed. Objects cut
by crop boundaries may remain. This demonstrates software mechanics only; it
does not create expert annotations, a held-out set or biological validation.
Review labels and design an independent evaluation before production training.

This fixed demonstration uses all six pairs, two epochs, minibatch 1, target
size 512, learning_rate=1e-5 and weight_decay=1e-5. Python, NumPy and PyTorch are
seeded at 19 on entry; Cellpose's own internal randomisation still occurs.
The numerical settings differ from the old one-epoch example. They are not
claimed optimal. recorded_settings.json is a record, not a portable GUI import.

normalize=False here controls spaCR's dataset preprocessing: the input image
is still scaled by its maximum. Cellpose's own training normalisation and
random rotation/resize remain enabled. augment=False disables spaCR's extra
eightfold materialisation, not Cellpose's internal training augmentation.
The input target is 512, while the actual CPSAM training patch is 256.

The measured warm-up rates were 0 and 0.0000011111111111111112. A one-epoch run
would stop after the zero-rate warm-up; two epochs let us verify actual weight
updates. All six pairs reached Cellpose and remained in its training set.
All 3,145,728 input image/label values and twelve actual preview arrays matched
their source pairs. In the recorded checkpoint, 345 of 347 trainable parameter
tensors changed; this count is of tensors, not individual scalar weights.

Training losses were 1.496740683913231 and 1.5215517828861873: loss rose slightly.
The zero test-loss slots are placeholders because no test data were supplied.
Neither a saved file, changed weights nor these training losses establish a
useful model, a better segmentation, convergence or held-out accuracy.

OUTPUTS
-------
The new folder holds private unchanged image/mask copies, actual_training_pairs.png,
requested_settings.json, saved spaCR settings and training_checks.json. The
checkpoint path in training_checks.json was checked against the actual file:

  models/cellpose_model/models/tutorial_cells_two_epoch_API_demo_cpsam_e2_X512_Y512.CP_model

The generic spaCR console summary omits the second models directory. Do not
copy that summary as a verified file path: use the checked result path above.
The checkpoint was loaded with torch.load(weights_only=True) for tensor checks;
that is not a held-out inference test. The checkpoint itself is not in this ZIP.
Keep it with its settings, source manifest and checks. Existing outputs are
never overwritten. The Apply tutorial is separate; opening its tab alone does
not test or validate a newly trained model. No AI request or provider response
was used in this example.
