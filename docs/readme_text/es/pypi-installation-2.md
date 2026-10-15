spaCR admite Python <strong>3.9 through 3.14</strong>, salvo Python 3.14.1, que torchvision excluye. Se recomienda Linux para los flujos de trabajo CUDA y ROCm más exigentes; macOS y Windows también son compatibles, y ambos usan sus GPU — macOS mediante Metal, que cubre Apple Silicon y las tarjetas AMD de los Mac con Intel, y Windows mediante CUDA o DirectML.

La instalación estándar incluye la interfaz de escritorio Qt. En un servidor, clúster o ejecutor de CI, ejecute los flujos de trabajo de línea de comandos sin abrirla:
