spaCR prend en charge Python <strong>3.9 à 3.14</strong>, à l'exception de Python 3.14.1, que torchvision exclut. Linux est recommandé pour les flux de travail CUDA et ROCm les plus lourds; macOS et Windows sont également pris en charge, et tous deux utilisent leurs GPUs — macOS via Metal, qui couvre Apple Silicon et les cartes AMD des Mac Intel, et Windows via CUDA ou DirectML.

L’installation standard inclut l’interface de bureau Qt. Sur un serveur, un cluster ou un exécuteur CI, lancez les pipelines en ligne de commande sans l’ouvrir :
