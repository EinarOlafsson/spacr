Item 899 reference fixture. gbp_plate77_features.csv is the GBP plate77 annotation selection
(500 vacuoles with the rim-enrichment features of gbp_rim.py, selection.csv of the BMC reanalysis
annotation set) with a fixed synthetic label column `annotate`: 3 where the vacuole touches the
border, else 1 where rim_enrichment >= 3.34541888153259 (the null p99 rim threshold of
gbp_threshold.json), else 2. reference_classifier_metrics.csv is what the reference script
train_eval_recruitment.py (feature_route) wrote for these labels: rows 1-3 are the
leave-one-well-out gradient boosting, the majority control and the shuffled control (mean of 20);
row 0 is the HRMAlexNet literature row the script carries.
