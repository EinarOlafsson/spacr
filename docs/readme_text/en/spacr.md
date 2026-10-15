# spaCR

<strong>Spatial phenotype analysis of CRISPR screens.</strong>

spaCR segments and measures single cells in microscopy images, integrates per-object phenotypes with sequencing-derived guide abundance, and estimates which genes are associated with phenotypic changes. Starting from plate images and FASTQ reads, it produces per-object measurements, trained classifiers, per-guide and per-gene effect estimates, and a ranked hit list.

The segmentation, measurement, annotation and classification modules also run without a sequencing arm.

Make Masks corrects segmentation masks and annotates independent, class-labelled rectangles with the <strong>Box</strong> tool for YOLO export. Boxes keep their own labels and history without changing source images or masks.

See the [feature guide](<../../source/features.rst>) for each tool.

Images, masks and crops are files on disk. Measurements, annotations and model predictions are tables in the project's ` measurements.db ` SQLite database.

Runs as a desktop application or headlessly on a workstation, server or cluster.
