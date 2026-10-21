## Bibliothèque de modèles

spaCR inclut un catalogue de modèles entraînés et les télécharge à la demande. Ouvrez <strong>Model Zoo</strong> depuis l’écran d’accueil pour les parcourir et les installer, ou indiquez une clé dans un fichier de paramètres, par exemple ` pathogen_model: toxoplasma_pv_v1 `. Le modèle est téléchargé et sa somme de contrôle est vérifiée lors de sa première utilisation. Chaque entrée publiée comporte une somme SHA-256. Les entrées qui en sont dépourvues sont refusées, car un fichier de modèle incomplet ou remplacé ne pourrait sinon pas être distingué de l’original.
