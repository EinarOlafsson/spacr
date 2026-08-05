"use strict";

// YouTube video IDs for the 4K cut of each lesson, keyed by lesson id.
//
// The site itself plays a 1440p copy: the 4K masters are ~681 MiB and the
// published documentation has a size budget (tools/docs_media_budget.py).
// YouTube carries the full-resolution version instead, and each lesson links
// out to it.
//
// A lesson whose value is "" simply shows no link -- an unfilled entry is not
// an error, and a missing lesson key is not either. Paste the 11-character
// video id, not the whole URL:
//
//     "07_mask": "dQw4w9WgXcQ",
//
// Upload sources live outside the repo, in
// /media/carruthers/mnt3/claude/tutorials/youtube_4k.
window.SPACR_YOUTUBE_LINKS = Object.freeze({
  "01_pypi_github": "",                // PyPI, GitHub, and conda-forge
  "02_conda_install": "",              // Installation with Conda
  "03_pip_install": "",                // Installation with pip
  "04_platform_installers": "",        // Platform installers
  "05_home": "",                       // Home screen and navigation
  "06_api": "",                        // Python API and headless workflows
  "07_mask": "",                       // Mask
  "08_measure": "",                    // Measure
  "09_annotate": "",                   // Annotate
  "10_classify_cv": "",                // Classify (CV)
  "11_classify_ml": "",                // Classify (ML)
  "12_map_barcodes": "",               // Map Barcodes
  "13_regression": "",                 // Regression
  "14_make_masks": "",                 // Make Masks
  "15_image_umap": "",                 // Image UMAP
  "16_activation": "",                 // Activation Maps
  "17_timelapse": "",                  // Timelapse
  "18_motility": "",                   // Motility Assay
  "19_train_cellpose": "",             // Train Cellpose
  "20_cellpose_masks": "",             // Cellpose Masks
  "21_model_compare": "",              // Model Compare
  "22_model_zoo": "",                  // Model Zoo
  "23_agreement": "",                  // Annotator Agreement
  "24_plaque": "",                     // Plaque Assay
  "25_recruitment": "",                // Recruitment
  "26_invasion": "",                   // Invasion Assay
  "27_replication": "",                // Replication Assay
  "28_training_runs": "",              // Training Runs
  "29_report": "",                     // Report
  "30_plate_queue": "",                // Plate Queue
  "31_external_masks": "",             // External Masks
  "32_align_stitch": "",               // Align and Stitch
  "33_plate_viewer": "",               // Plate Viewer
  "34_database": "",                   // Database Browser
  "35_converter": "",                  // Format Converter
  "36_import": "",                     // Import Project
  "37_batch": "",                      // Batch Runner
  "38_distributed_jobs": "",           // Distributed Jobs
  "39_classifier_evaluation": "",      // Classifier Evaluation
  "40_run_history": "",                // Run History
});
