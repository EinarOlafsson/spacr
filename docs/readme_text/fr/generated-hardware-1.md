| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU ; ajustement PCA CPU, projection GPU / CPU ; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU ; ajustement PCA CPU, projection GPU / CPU ; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 pris en charge (stable)   🟣 implémenté (bêta)   🔴 uniquement sur CPU

Auto: True 3D dans Make Masks essaie les backends disponibles dans cet ordre : Linux : Vulkan → OpenGL → CPU ; Windows : Vulkan → Direct3D 12 → OpenGL → CPU ; macOS : Metal → Vulkan → OpenGL → CPU. Vulkan, Metal et Direct3D 12 utilisent wgpu. Si un backend GPU échoue ou est indisponible, le suivant est essayé ; le rendu sur CPU reste disponible.

wgpu est installé par défaut avec Python 3.10 ou ultérieur sous Windows (x64, ARM64), macOS (Intel, Apple silicon) et Linux (x86-64, ARM64, glibc 2.28 ou ultérieure). Python 3.9, les autres processeurs, Linux avec une ancienne glibc et les installations sans wgpu utilisent OpenGL, puis le CPU. Les cellules 3D violettes indiquent un support GPU implémenté, en attente de validation sur un véritable écran.

L’exécution sur CPU est prise en charge et peut être plus lente. Lors d’une mesure sur un Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB), Cellpose-SAM a pris 444.5 s sur CPU et 3.2 s avec Metal pour une image de 256×256, avec torch 2.2.2, Python 3.12.11 et macOS 26.5.2. Il s’agit d’un exemple historique, pas d’une garantie de vitesse. L’accélération GPU nécessite un backend utilisable et des bibliothèques compatibles. Consultez la [mesure matérielle enregistrée](<../../../features/feature_list/319_use_non_nvidia_accelerators_where_they_exist.txt>).
