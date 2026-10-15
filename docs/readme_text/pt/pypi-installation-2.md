O spaCR oferece suporte ao Python <strong>3.9 through 3.14</strong>, exceto ao Python 3.14.1, que é excluído pelo torchvision. Recomenda-se Linux para os fluxos de trabalho mais pesados com CUDA e ROCm; macOS e Windows também são compatíveis, e ambos utilizam suas GPUs — macOS por meio do Metal, que abrange o Apple Silicon e as placas gráficas AMD dos Macs Intel, e Windows por meio de CUDA ou DirectML.

A instalação padrão inclui a interface de desktop Qt. Em um servidor, cluster ou executor de CI, execute os fluxos de trabalho de linha de comando sem abri-la:
