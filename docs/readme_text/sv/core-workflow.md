### Kärnarbetsflöde

Det primära arbetsflödet består av sex moduler:

- <strong>Mask</strong> segmenterar celler, cellkärnor, patogener och organeller med Cellpose.
- <strong>Measure</strong> skriver morfologiska, intensitets-, textur-, rumsliga och kolokaliseringsmått samt objektutsnitt till SQLite.
- <strong>Annotate</strong> märker objektutsnitt i ett tangentbordsstyrt rutnät och stöder köer för aktiv inlärning.
- <strong>Classify</strong> tränar bild- eller mätningsbaserade modeller och sparar prestandan på undanhållna data med varje kontrollpunkt.
- <strong>Map Barcodes</strong> kopplar FASTQ-läsningar till brunnar och gRNA:er och rapporterar QC för förekomst, kollisioner och täckning.
- <strong>Regression</strong> skattar effekter för guider, gener, betingelser och kontroller med modellfamiljer för kontinuerliga data, andelar och antal.
