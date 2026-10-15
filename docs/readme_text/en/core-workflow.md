### Core workflow

The primary workflow comprises six modules:

- <strong>Mask</strong> segments cells, nuclei, pathogens and organelles with Cellpose.
- <strong>Measure</strong> writes morphology, intensity, texture, spatial and colocalization features, together with object crops, to SQLite.
- <strong>Annotate</strong> labels crops in a keyboard-driven grid and supports active-learning queues.
- <strong>Classify</strong> trains image or measurement-based models and records held-out performance with each checkpoint.
- <strong>Map Barcodes</strong> maps FASTQ reads to wells and gRNAs, with abundance, collision and coverage QC.
- <strong>Regression</strong> estimates guide, gene, condition and control effects with model families suited to continuous, fractional and count responses.
