spaCR supports Python <strong>3.9 through 3.14</strong>, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML.

The standard installation includes the Qt desktop interface. For a server, cluster or CI runner, run the command-line pipelines without opening it:
