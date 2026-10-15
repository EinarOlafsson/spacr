spaCR unterstützt Python <strong>3.9 through 3.14</strong>, außer Python 3.14.1, das von torchvision ausgeschlossen wird. Linux wird für die anspruchsvollsten CUDA- und ROCm-Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt und nutzen beide ihre GPUs — macOS über Metal, das Apple Silicon und die AMD-Karten in Intel Macs abdeckt, und Windows über CUDA oder DirectML.

Die Standardinstallation enthält die Qt-Desktopoberfläche. Auf einem Server, Cluster oder CI-Runner können Sie die Befehlszeilen-Pipelines ausführen, ohne sie zu öffnen:
