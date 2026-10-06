# F567 tracked-field annotation editor CPU receipt

The alpha Timelapse preview now opens an existing tracker CSV and its matching image sequence, displays original zero-based frames and real track IDs, and writes `field`, `track_id`, `frame`, `event`, and `object` rows accepted by `_event_read_annotations`. The UI never derives IDs from the capped preview. It loads frames lazily with a six-frame cache, refuses tracks without finite positions, requires explicit sequence confirmation, preserves annotations for other fields and objects, and validates a staged CSV before atomic replacement. It refuses publication if the tracks or annotation table changed after loading; Cancel never writes an annotation table. A single track observation has one event class. GUI-written rows bind to `tracker_backend` and `track_source_sha256`; the detector refuses a mismatch with its selected tracker before model fitting. Legacy annotations without provenance retain their prior reader behavior, while the GUI requires an explicit confirmation of their current tracker before it writes source-bound rows.

The source is the six-commit chain from `bf7873dec6859ba75622c532216cd36f183472ec` through `ac198b6b6861d5c4d0873ddb2c3bb0c95905b6d0`. Source Git blob IDs, coverage counts, focused command, and artifact digests are in `receipt.json`; the exact two-module coverage report is archived as `coverage.json.gz`. `reproduce.py` checks both exact source blobs, reruns the bounded branch-coverage cohort, and compares added executable lines and added-origin branch arcs to the frozen base. Run it from the repository root with CUDA hidden and the normal memory cap:

```sh
CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg tools/run_capped.sh 4G python features/data/567_event_annotation_editor_2026-10-06/reproduce.py
```

This proves GUI/table integration only. It does not prove pretrained video inference, annotated Toxoplasma egress/invasion accuracy, or GPU acceptance; F567 remains OPEN. Generated API, translations, help, tutorials, and runtime catalogs require their normal owner refresh after source integration.
