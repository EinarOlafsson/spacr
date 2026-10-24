.. list-table::
   :header-rows: 1
   :widths: 30 17 15 23 15

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
     - 3D
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
     - 🟣 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
     - 🟣 GPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🟣 KMeans GPU ; ajustement PCA CPU, projection GPU / CPU ; 🔴 UMAP / t-SNE / DBSCAN CPU
     - 🟣 GPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🟣 KMeans GPU ; ajustement PCA CPU, projection GPU / CPU ; 🔴 UMAP / t-SNE / DBSCAN CPU
     - 🟣 GPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
     - 🟣 GPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 pris en charge (stable)   🟣 implémenté (bêta)   🔴 uniquement sur CPU
