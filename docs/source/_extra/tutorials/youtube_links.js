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
//     "07_mask": "AAAAAAAAAAA",
//
// Upload sources live outside the repo, in
// /media/carruthers/mnt3/claude/tutorials/youtube_4k.
window.SPACR_YOUTUBE_LINKS = Object.freeze({
  "01_pypi_github": "Hkmghk8jwkI",                // PyPI, GitHub, and conda-forge
  "02_conda_install": "qfs4sm201uo",              // Installation with Conda
  "03_pip_install": "fP_E6ishElw",                // Installation with pip
  "04_platform_installers": "X0CJaYhWg3I",        // Platform installers
  "05_home": "hDCeEPHVA-Y",                       // Home screen and navigation
  "06_api": "MCsjYtLfLZU",                        // Python API and headless workflows
  "07_mask": "nlErufRyrtU",                       // Mask
  "08_measure": "kXdRsQI60Vc",                    // Measure
  "09_annotate": "4VoAwo9YVjA",                   // Annotate
  "10_classify_cv": "u69A67fZuEs",                // Classify (CV)
  "11_classify_ml": "qW4xNH-Q12E",                // Classify (ML)
  "12_map_barcodes": "yEb2HrNIKEc",               // Map Barcodes
  "13_regression": "Emj9jQA0Tnc",                 // Regression
  "14_make_masks": "VZBgGTgPk0M",                 // Make Masks
  "15_image_umap": "h909hIIPass",                 // Image UMAP
  "16_activation": "vDBqzRsUZvM",                 // Activation Maps
  "17_timelapse": "Cw8S8-co3a0",                  // Timelapse
  "18_motility": "L8giEuAj1mM",                   // Motility Assay
  "19_train_cellpose": "psMRmQUk-0w",             // Train Cellpose
  "20_cellpose_masks": "WLIfSPzgQXU",             // Cellpose Masks
  "21_model_compare": "AD7zmlmPBcc",              // Model Compare
  "22_model_zoo": "YWEqJBbcAlQ",                  // Model Zoo
  "23_agreement": "-jRxAJ4CTWU",                  // Annotator Agreement
  "24_plaque": "Rccu9sPEw_Q",                     // Plaque Assay
  "25_recruitment": "FcqlhMWjZ04",                // Recruitment
  "26_invasion": "ByOEWH1GHl8",                   // Invasion Assay
  "27_replication": "wIGASmZ5Sgk",                // Replication Assay
  "28_training_runs": "4bgQFqWtppA",              // Training Runs
  "29_report": "fuKXbKLjprQ",                     // Report
  "30_plate_queue": "QLQFfwEIQQU",                // Plate Queue
  "31_external_masks": "i0q32gQ39cQ",             // External Masks
  "32_align_stitch": "MMoW1w93uVI",               // Align and Stitch
  "33_plate_viewer": "XzaHXNFPrOE",               // Plate Viewer
  "34_database": "bkjMZRbWlnU",                   // Database Browser
  "35_converter": "8gRXINFQQ7U",                  // Format Converter
  "36_import": "rQilrniNbMg",                     // Import Project
  "37_batch": "I-Lg1Rqi-UQ",                      // Batch Runner
  "38_distributed_jobs": "aF63i0KHXxk",           // Distributed Jobs
  "39_classifier_evaluation": "ZusbvugRNlA",      // Classifier Evaluation
  "40_run_history": "jG2ERpPRhmc",                // Run History
});
