# spaCR

<strong>Analyse spatiale des phénotypes de criblages CRISPR.</strong>

spaCR segmente et mesure les cellules individuelles dans des images de microscopie, intègre les phénotypes par objet à l’abondance des guides dérivée du séquençage et estime quels gènes sont associés aux changements phénotypiques. À partir d’images de plaques et de lectures FASTQ, il produit des mesures par objet, des classificateurs entraînés, des estimations d’effet par guide et par gène, ainsi qu’une liste de résultats classée.

Les modules de segmentation, de mesure, d'annotation et de classification fonctionnent également sans bras de séquençage.

Make Masks corrige les masques de segmentation et annote des rectangles indépendants avec des étiquettes de classe grâce à l’outil <strong>Box</strong> pour l’export YOLO. Les rectangles conservent leurs propres étiquettes et leur historique sans modifier les images ni les masques sources.

Chaque outil est décrit dans le [guide des fonctionnalités](<../../source/features.rst>).

Les images, masques et vignettes sont des fichiers sur disque. Les mesures, annotations et prédictions du modèle sont des tables de la base de données SQLite ` measurements.db ` du projet.

Fonctionne comme une application de bureau ou sans interface graphique sur un poste de travail, un serveur ou un cluster.
