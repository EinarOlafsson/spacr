| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA fitting CPU, projection GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA fitting CPU, projection GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

Auto: True 3D in Make Masks tries each available backend in order: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal and Direct3D 12 use wgpu. A failed or unavailable GPU backend falls back to the next backend; CPU rendering remains available.

wgpu is installed by default with Python 3.10 or later on Windows (x64, ARM64), macOS (Intel, Apple silicon) and Linux (x86-64, ARM64, glibc 2.28 or later). Python 3.9, other processors, Linux with older glibc and installations made without wgpu use OpenGL, then CPU. The purple 3D cells indicate implemented GPU support awaiting real-display acceptance.

CPU execution is supported and can be slower. In one measurement on an Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB), Cellpose-SAM took 444.5 s on the CPU and 3.2 s on Metal for one 256×256 image, using torch 2.2.2, Python 3.12.11 and macOS 26.5.2. This is a historical example, not a speed guarantee. GPU acceleration requires a usable backend and compatible libraries. See the [recorded hardware measurement](<../../../features/new/319_use_non_nvidia_accelerators_where_they_exist.txt>).
