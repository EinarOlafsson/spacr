### Flux de travail principal

Le flux de travail principal comprend six modules :

- <strong>Mask</strong> segmente les cellules, les noyaux, les agents pathogènes et les organites avec Cellpose.
- <strong>Measure</strong> enregistre dans SQLite les caractéristiques morphologiques, d’intensité, de texture, spatiales et de colocalisation, ainsi que les vignettes des objets.
- <strong>Annotate</strong> annote les vignettes dans une grille pilotée au clavier et prend en charge les files d’apprentissage actif.
- <strong>Classify</strong> entraîne des modèles fondés sur les images ou les mesures et enregistre, avec chaque checkpoint, les performances sur les données réservées.
- <strong>Map Barcodes</strong> associe les lectures FASTQ aux puits et aux gRNA, avec un contrôle qualité de l’abondance, des collisions et de la couverture.
- <strong>Regression</strong> estime les effets des guides, des gènes, des conditions et des contrôles avec des familles de modèles adaptées aux réponses continues, fractionnelles et de comptage.
