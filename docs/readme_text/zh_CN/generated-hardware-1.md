| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU；PCA 拟合 CPU，投影 GPU / CPU；🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU；PCA 拟合 CPU，投影 GPU / CPU；🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 支持（稳定）   🟣 已实现（beta）   🔴 仅支持 CPU

Auto: Make Masks 的 True 3D 按以下顺序尝试可用后端：Linux：Vulkan → OpenGL → CPU；Windows：Vulkan → Direct3D 12 → OpenGL → CPU；macOS：Metal → Vulkan → OpenGL → CPU。Vulkan、Metal 和 Direct3D 12 使用 wgpu。如果 GPU 后端不可用或失败，则尝试下一个后端；CPU 渲染始终可用。

在 Windows（x64、ARM64）、macOS（Intel、Apple silicon）和 Linux（x86-64、ARM64、glibc 2.28 或更高版本）上，Python 3.10 或更高版本默认安装 wgpu。Python 3.9、其他处理器、使用旧版 glibc 的 Linux，以及未安装 wgpu 的环境会先使用 OpenGL，再回退到 CPU。紫色的 3D 单元格表示已实现 GPU 支持，但仍待真实显示器验收。

支持 CPU 执行，但速度可能较慢。在 Intel iMac（Core i5-10500、AMD Radeon Pro 5300 4 GB）上的一次测量中，使用 torch 2.2.2、Python 3.12.11 和 macOS 26.5.2 时，Cellpose-SAM 处理一张 256×256 图像在 CPU 上耗时 444.5 s，在 Metal 上耗时 3.2 s。这是历史示例，并非速度保证。GPU 加速需要可用的后端和兼容的库。请参阅[已记录的硬件测量](<../../../features/feature_list/319_use_non_nvidia_accelerators_where_they_exist.txt>)。
