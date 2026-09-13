EMBEDDINGS — REAL API EXAMPLE FOR spaCR 1.5.0.7
===========================================

The Home screen now includes Embeddings. In 1.5.0.7 its standalone screen
does not provide a crop picker; Embed remains disabled without a programmatic
input. This example uses the supported Python API, NOT a repaired GUI.

Activate your spaCR environment. Install the optional dependency if needed:

    python -m pip install "spacr[embeddings]==1.5.0.7"

The recording uses timm 1.0.29 in an isolated tutorial environment. ResNet18
weights are downloaded on first use (or read from their existing cache).
The default ResNet18 is ImageNet-supervised: it is NOT a cell-trained,
self-supervised encoder merely because this module is called Embeddings.

Load Annotate's real example dataset using its Test data control, as shown in
the Annotate tutorial. Keep the original downloaded project. The recording's
16 crops are the first 16 sorted PNGs directly in:

    plate1/data/single_nucleus/uninfected/plate1_E02/cell_png

These are existing 224 x 224 PNG crops, not generated training data. No labels
or database are read by this demonstration. Folder names are NOT used as a
prediction target. The sample is deliberately small and is not representative
of a screen. Locate the equivalent folder in your downloaded project and run:

    python embeddings_example.py --crops /path/to/cell_png --output run_per_channel
    python embeddings_example.py --crops /path/to/cell_png --output run_project --policy project

Both commands use CPU and batches of four. The first calls the pretrained
encoder once per stored channel, in batches, giving 16 x 1536 values. The
second encodes the three-slot image together, giving 16 x 512. It changes the
representation; it is not a faster interchangeable copy of the first result.
All saved inputs have the same shape. The helper does not resize or resegment.

API: spacr.crops.read_crop_png, spacr.embeddings.EmbeddingSpec,
spacr.embeddings.embed_array, EmbeddingResult.to_frame, encoder_entry.
embed_array applies its per-channel 99th-percentile scaling over the provided
stack. Changing the set of crops can therefore change scaling. Match inputs,
normalization, crop size, channel policy, weights and package versions when
comparing results; the short specification fingerprint is not a weights hash.

In the default policy, emb_c0_* means PNG slot 0, NOT a verified stain name.
PNG display slots can differ from original acquisition channels. Check the
export's channel mapping before biological interpretation. The helper uses
filenames as example object_id values and does NOT claim they equal a
measurements.db join key. Construct and validate that mapping before joining.

Each NEW output directory contains vectors.npy, vectors.csv, input_crops.png,
pca_coordinates.npy, pca.png and run.json. The helper reopens every saved
matrix value and ordered row/column identity, checks the original image hashes,
and records weights SHA256, settings and dependency versions. Existing output
directories are refused. A run that raises an error is not complete: retain
it for diagnosis and choose a new output directory after correcting the cause.

View the real input montage and exploratory PCA. Numbered points correspond
to CSV rows and montage numbers. A single embedding dimension is not a
phenotype; two-dimensional separation in this small sample is not validation.
No classifier is fitted, no gene retrieval is measured, and no claim of
outperforming the hand-measured panel is made. Keep the measured features.

CPU works for this bounded example. --device cuda or --device mps is an
explicit request only for a supported device/build; no accelerator run is
claimed by this recording. This engine's automatic choice is CUDA if available,
otherwise CPU, not the application's general accelerator resolver. The example
does not send images or prompts to spaCR AI or any language-model provider.
