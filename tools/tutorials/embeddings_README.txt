EMBEDDINGS — LOAD CROPS AND SAVE FEATURE VECTORS
==============================================

Install the optional encoder dependency in your spaCR environment:

    python -m pip install "spacr[embeddings]"

With a checkout, install its matching extra using pip install -e '.[embeddings]'.
The first encoder run downloads pretrained weights if they are not cached.
The default ResNet18 uses ImageNet weights. Choose an encoder appropriate for
your images and evaluate the resulting features for your downstream task.

EXAMPLE INPUT

Download Annotate's example through its Test data control. Locate:

    plate1/data/single_nucleus/uninfected/plate1_E02/cell_png

The helper selects the first sixteen PNGs in sorted filename order. These
example images are 224 x 224 pixels with three stored channels. To follow the
GUI video exactly, copy those sixteen images into a new folder.

FROM HOME

Open Embeddings in Data. Choose crop folder under Crops from, enter the folder
path and press Enter, or use Browse. Set At most to 16 and click Load crops.
Choose resnet18, Batch 4 and Per channel, then click Embed. Select Project to
three and run again to compare the policies.

Per channel produces 16 x 1536 values for this example. Project to three
produces 16 x 512. These are different feature representations. Keep the
policy, backbone, weights and normalization consistent when comparing runs.
The preview shows a few dimensions of up to fifty objects. The current screen
has no export button; use the helper below to save complete matrices.

SAVE WITH PYTHON

Extract this ZIP and run these commands in the environment containing spaCR:

    python embeddings_example.py --crops /path/to/cell_png --output run_per_channel
    python embeddings_example.py --crops /path/to/cell_png --output run_project --policy project

Use a NEW output folder for each run. Both commands use CPU and batches of
four. --device cuda or --device mps requests a supported accelerator explicitly.
The helper reads existing pixels without resizing or segmenting them.

Each output folder contains vectors.npy, vectors.csv, input_crops.png,
pca_coordinates.npy, pca.png and run.json. CSV rows include the source filenames;
the montage and plots number the same objects. Keep run.json with the vectors.
Use embed_array and EmbeddingSpec directly when integrating your own code.

Channel numbers refer to stored PNG slots. Check the crop export's mapping to
stains, and establish object join keys before combining vectors with database
measurements. embed_array estimates channel scaling from the supplied stack;
changing that stack can change the scale. Use matching normalization when
comparing datasets. PCA plots are exploratory; use the full vectors for
analysis, and evaluate separation on data relevant to your biological question.
