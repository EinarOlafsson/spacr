| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; ajuste de PCA CPU, proyección GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; ajuste de PCA CPU, proyección GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 compatible (estable)   🟣 implementado (beta)   🔴 solo compatible con CPU

Auto: True 3D de Make Masks prueba los backends disponibles en este orden: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal y Direct3D 12 usan wgpu. Si un backend de GPU falla o no está disponible, se prueba el siguiente; el renderizado en CPU sigue disponible.

wgpu se instala de forma predeterminada con Python 3.10 o posterior en Windows (x64, ARM64), macOS (Intel, Apple silicon) y Linux (x86-64, ARM64, glibc 2.28 o posterior). Python 3.9, otros procesadores, Linux con glibc anterior y las instalaciones sin wgpu usan OpenGL y después CPU. Las celdas moradas de 3D indican soporte de GPU implementado, pendiente de validación en una pantalla real.

Se admite la ejecución en CPU, que puede ser más lenta. En una medición en un Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB), Cellpose-SAM tardó 444.5 s en CPU y 3.2 s con Metal para una imagen de 256×256, con torch 2.2.2, Python 3.12.11 y macOS 26.5.2. Es un ejemplo histórico, no una garantía de velocidad. La aceleración por GPU requiere un backend utilizable y bibliotecas compatibles. Consulte la [medición de hardware registrada](<../../../features/feature_list/319_use_non_nvidia_accelerators_where_they_exist.txt>).
