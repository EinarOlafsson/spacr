| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA 적합 CPU, 투영 GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA 적합 CPU, 투영 GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 지원 (안정)   🟣 구현됨 (베타)   🔴 CPU만 지원

Auto: Make Masks의 True 3D는 사용 가능한 백엔드를 다음 순서로 시도합니다. Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal, Direct3D 12는 wgpu를 사용합니다. GPU 백엔드를 사용할 수 없거나 실패하면 다음 백엔드를 시도하며 CPU 렌더링은 계속 사용할 수 있습니다.

Windows(x64, ARM64), macOS(Intel, Apple silicon), Linux(x86-64, ARM64, glibc 2.28 이상)에서 Python 3.10 이상은 wgpu를 기본으로 설치합니다. Python 3.9, 기타 프로세서, 이전 glibc를 사용하는 Linux 및 wgpu 없이 설치한 환경은 OpenGL을 사용한 뒤 CPU로 전환합니다. 보라색 3D 셀은 GPU 지원이 구현되었으나 실제 디스플레이에서의 검증이 남아 있음을 뜻합니다.

CPU 실행을 지원하며 속도가 느릴 수 있습니다. Intel iMac(Core i5-10500, AMD Radeon Pro 5300 4 GB)에서 torch 2.2.2, Python 3.12.11, macOS 26.5.2를 사용한 한 측정에서 Cellpose-SAM은 256×256 이미지 한 장에 CPU에서 444.5 s, Metal에서 3.2 s가 걸렸습니다. 이는 과거의 예이며 속도를 보장하지 않습니다. GPU 가속에는 사용 가능한 백엔드와 호환 라이브러리가 필요합니다. [기록된 하드웨어 측정](<../../../features/new/319_use_non_nvidia_accelerators_where_they_exist.txt>) 을 참고하세요.
