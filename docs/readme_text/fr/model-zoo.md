## Bibliothèque de modèles

spaCR envoie un catalogue de modèles formés et les récupère sur demande. Ouvrez <strong>Modèle Zoo</strong> depuis l'écran d'accueil pour les parcourir et les installer, ou nommez une clé dans un fichier de paramètres -- ` pathogen_model: toxoplasma_pv_v1 ` -- et le modèle est téléchargé et vérifié la première fois qu'il est nécessaire. Chaque entrée publiée porte un SHA-256; une entrée sans un est refusée plutôt que installée, parce qu'un point de contrôle tronqué ou substitué ne peut pas être dit du vrai.
