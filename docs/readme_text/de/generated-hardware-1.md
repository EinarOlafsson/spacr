| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA Anpassung CPU, Projektion GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🟣 KMeans GPU; PCA Anpassung CPU, Projektion GPU / CPU; 🔴 UMAP / t-SNE / DBSCAN CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 unterstützt (stabil)   🟣 implementiert (Beta)   🔴 nur CPU-Unterstützung

Auto: True 3D in Make Masks versucht die verfügbaren Backends der Reihe nach: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU. Vulkan, Metal und Direct3D 12 verwenden wgpu. Ist ein GPU-Backend nicht verfügbar oder schlägt es fehl, folgt das nächste; CPU-Rendering bleibt verfügbar.

wgpu wird standardmäßig mit Python 3.10 oder neuer unter Windows (x64, ARM64), macOS (Intel, Apple silicon) und Linux (x86-64, ARM64, glibc 2.28 oder neuer) installiert. Python 3.9, andere Prozessoren, Linux mit älterer glibc und Installationen ohne wgpu verwenden OpenGL, dann CPU. Die violetten 3D-Zellen kennzeichnen implementierte GPU-Unterstützung, deren Abnahme an einem echten Bildschirm noch aussteht.

Die Ausführung auf der CPU wird unterstützt und kann langsamer sein. Bei einer Messung auf einem Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB) benötigte Cellpose-SAM 444.5 s auf der CPU und 3.2 s mit Metal für ein Bild mit 256×256 Pixeln, mit torch 2.2.2, Python 3.12.11 und macOS 26.5.2. Dies ist ein historisches Beispiel, keine Geschwindigkeitsgarantie. GPU-Beschleunigung erfordert ein nutzbares Backend und kompatible Bibliotheken. Siehe die [aufgezeichnete Hardwaremessung](<../../../features/new/319_use_non_nvidia_accelerators_where_they_exist.txt>).
