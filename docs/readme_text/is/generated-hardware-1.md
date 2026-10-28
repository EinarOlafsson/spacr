| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA aðlögun CPU, vörpun GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA aðlögun CPU, vörpun GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 með stuðningi (stöðugt)   🟣 útfært (beta)   🔴 aðeins CPU-stuðningur

Auto: True 3D í Make Masks reynir tiltæka bakenda í þessari röð: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal og Direct3D 12 nota wgpu. Ef GPU-bakendi er ekki tiltækur eða bregst er næsti reyndur; teiknun á CPU er áfram tiltæk.

wgpu er sjálfgefið sett upp með Python 3.10 eða nýrra á Windows (x64, ARM64), macOS (Intel, Apple silicon) og Linux (x86-64, ARM64, glibc 2.28 eða nýrra). Python 3.9, aðrir örgjörvar, Linux með eldra glibc og uppsetningar án wgpu nota OpenGL og síðan CPU. Fjólubláu 3D-reitirnir tákna útfærðan GPU-stuðning sem bíður samþykktar á raunverulegum skjá.

Keyrsla á CPU er studd og getur verið hægari. Í einni mælingu á Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB) tók Cellpose-SAM 444.5 s á CPU og 3.2 s með Metal fyrir eina 256×256 mynd, með torch 2.2.2, Python 3.12.11 og macOS 26.5.2. Þetta er sögulegt dæmi, ekki hraðaloforð. GPU-hröðun krefst nothæfs bakenda og samhæfra safna. Sjá [skráða vélbúnaðarmælingu](<../../../features/feature_list/319_use_non_nvidia_accelerators_where_they_exist.txt>).
