| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 suportado (estável)   🟣 implementado (beta)   🔴 apenas suporte a CPU

Auto: O True 3D do Make Masks testa os backends disponíveis nesta ordem: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal e Direct3D 12 usam wgpu. Se um backend de GPU falhar ou não estiver disponível, o próximo é testado; a renderização na CPU continua disponível.

wgpu é instalado por padrão com Python 3.10 ou posterior no Windows (x64, ARM64), macOS (Intel, Apple silicon) e Linux (x86-64, ARM64, glibc 2.28 ou posterior). Python 3.9, outros processadores, Linux com glibc anterior e instalações sem wgpu usam OpenGL, depois CPU. As células roxas de 3D indicam suporte de GPU implementado, aguardando validação em uma tela real.

A execução na CPU é compatível e pode ser mais lenta. Em uma medição em um Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB), o Cellpose-SAM levou 444.5 s na CPU e 3.2 s com Metal para uma imagem de 256×256, usando torch 2.2.2, Python 3.12.11 e macOS 26.5.2. Este é um exemplo histórico, não uma garantia de velocidade. A aceleração por GPU requer um backend utilizável e bibliotecas compatíveis. Consulte a [medição de hardware registrada](<../../../features/new/319_use_non_nvidia_accelerators_where_they_exist.txt>).
