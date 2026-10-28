| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA anpassning CPU, projektion GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA anpassning CPU, projektion GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 stöds (stabilt)   🟣 implementerat (beta)   🔴 endast CPU-stöd

Auto: True 3D i Make Masks provar tillgängliga renderingsmotorer i ordning: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal och Direct3D 12 använder wgpu. Om en GPU-motor saknas eller misslyckas provas nästa; rendering på CPU finns alltid kvar.

wgpu installeras som standard med Python 3.10 eller senare på Windows (x64, ARM64), macOS (Intel, Apple silicon) och Linux (x86-64, ARM64, glibc 2.28 eller senare). Python 3.9, andra processorer, Linux med äldre glibc och installationer utan wgpu använder OpenGL, sedan CPU. De lila 3D-cellerna anger implementerat GPU-stöd som väntar på verifiering på en riktig skärm.

Körning på CPU stöds och kan vara långsammare. Vid en mätning på en Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB) tog Cellpose-SAM 444.5 s på CPU och 3.2 s på Metal för en bild på 256×256, med torch 2.2.2, Python 3.12.11 och macOS 26.5.2. Detta är ett historiskt exempel, ingen hastighetsgaranti. GPU-acceleration kräver en fungerande backend och kompatibla bibliotek. Se den [dokumenterade hårdvarumätningen](<../../../features/feature_list/319_use_non_nvidia_accelerators_where_they_exist.txt>).
