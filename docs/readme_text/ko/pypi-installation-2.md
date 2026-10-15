spaCR는 Python <strong>3.9 through 3.14</strong> 버전을 지원하며, torchvision이 제외하는 Python 3.14.1은 예외입니다. 가장 부하가 큰 CUDA 및 ROCm 워크플로에는 Linux를 권장합니다. macOS와 Windows도 지원되며, 두 플랫폼 모두 GPU를 사용합니다 — macOS는 Apple Silicon과 Intel Mac의 AMD 카드를 모두 포괄하는 Metal을 통해, Windows는 CUDA 또는 DirectML을 통해 GPU를 사용합니다.

표준 설치에는 Qt 데스크톱 인터페이스가 포함됩니다. 서버, 클러스터 또는 CI 실행기에서는 인터페이스를 열지 않고 명령줄 파이프라인을 실행하세요:
