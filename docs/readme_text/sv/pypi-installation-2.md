spaCR stöder Python <strong>3.9 through 3.14</strong>, utom Python 3.14.1, som torchvision utesluter. Linux rekommenderas för de tyngsta CUDA- och ROCm-arbetsflödena; macOS och Windows stöds också, och båda använder sina GPU:er — macOS via Metal, som täcker Apple Silicon och AMD-korten i Intel-Mac-datorer, och Windows via CUDA eller DirectML.

Standardinstallationen innehåller Qt-gränssnittet för skrivbordet. På en server, ett kluster eller en CI-körmiljö kan du köra kommandoradspipelines utan att öppna det:
