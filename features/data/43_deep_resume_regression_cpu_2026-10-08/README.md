# Resolved-checkpoint regression cohort

A source-exact CPU cohort at `94f9995b896adfc7ff4ec19e5ae689fcf64545a9` checks the `train_model` resumed-checkpoint path repair alongside pre-existing training, best-checkpoint, artifact roundtrip, atomic save and model-card contracts. All 44 tests passed under a 4 GiB cap with CUDA hidden. The complete command, source blob and raw log digest are in `receipt.json`; `focused.log.gz` contains the unedited terminal output. This bounded result is not a full CI or coverage verdict.
