### Kern-Workflow

Der primäre Arbeitsablauf umfasst sechs Module:

- <strong>Mask</strong> segmentiert Zellen, Zellkerne, Pathogene und Organellen mit Cellpose.
- <strong>Measure</strong> schreibt Morphologie-, Intensitäts-, Textur-, räumliche und Kolokalisationsmerkmale sowie Objektausschnitte nach SQLite.
- <strong>Annotate</strong> beschriftet Objektausschnitte in einem tastaturgesteuerten Raster und unterstützt Active-Learning-Warteschlangen.
- <strong>Classify</strong> trainiert bild- oder messwertbasierte Modelle und speichert mit jedem Checkpoint die Leistung auf zurückgehaltenen Daten.
- <strong>Map Barcodes</strong> ordnet FASTQ-Reads Wells und gRNAs zu und liefert QC für Häufigkeit, Kollisionen und Abdeckung.
- <strong>Regression</strong> schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit Modellfamilien für kontinuierliche Werte, Anteile und Zähldaten.
