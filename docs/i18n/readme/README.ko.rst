|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: 문서
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: 대화형 튜토리얼
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI 버전
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9~3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: 테스트 모음
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt 인터페이스
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub 소스 코드
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub 이슈
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm 비상업용 라이선스
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: 최신 설치 프로그램
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge 레시피

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

언어: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`번역 모델 정보 <../TRANSLATION_MODELS.md>`_

**CRISPR 스크리닝의 공간 표현형 분석.**

spaCR는 고함량 현미경 영상에서 단일 세포를 분할하고 측정하며, 각 세포를 전달받은 gRNA와 연결하고 어떤 유전자가 표현형을 바꾸었는지 보고합니다. 플레이트 영상과 FASTQ 리드를 입력하면 객체별 측정값, 학습된 분류기, 가이드별·유전자별 효과 크기와 우선순위가 지정된 후보 목록이 출력됩니다.

영상 기반 풀드 CRISPR 스크리닝에서는 이것이 전체 작업 흐름입니다. 고함량 현미경 데이터만 있고 스크리닝 실험은 없는 경우에도 분할, 측정, 주석 및 분류 단계를 독립적으로 실행할 수 있습니다.

영상, 마스크, 이미지 크롭, 측정값, 주석, 예측, 바코드 및 웰 식별자는 하나의 SQLite 프로젝트에 저장되므로 결과의 값을 그 출처 객체까지 추적할 수 있습니다.

spaCR를 데스크톱 애플리케이션으로 실행하거나 워크스테이션, 서버 또는 클러스터에서 그래픽 인터페이스 없이 실행할 수 있습니다. 두 방식 모두 동일한 모듈을 사용하며, 모듈이 지원하면 CUDA가 자동으로 활성화됩니다.


작업 흐름 개요
--------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR 작업 흐름 및 출력 구성
   :align: center

현미경 영상(TIFF, OME-TIFF, LIF, CZI, ND2)과 시퀀싱 리드(FASTQ)는 서로 보완적인 영상 분석 및 바코드 매핑 작업 흐름으로 들어갑니다. 그런 다음 객체 테이블, 이미지 크롭, 주석, 예측, 가이드 식별 정보, QC 결과 및 웰 단위 요약을 함께 분석합니다.


빠른 시작
-----------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR는 Python **3.9~3.14** 버전을 지원합니다(단, torchvision이 지원하지 않는 Python 3.14.1은 제외). Python 3.12에서 선택 가능한 과학 패키지가 가장 많습니다. CUDA 작업 흐름에는 Linux를 권장하며 macOS와 Windows도 지원합니다.


설치 세부 정보
--------------------

|Release| |PyPI| |CondaRecipe|

**(베타) 경량 데스크톱 설치 프로그램:**

.. spacr-installer-links-begin

|InstallerWindows| |InstallerMacOS| |InstallerLinux|

.. |InstallerWindows| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: 다운로드 SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (인텔 및 애플 실리콘): 다운로드 SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64비트 Linux: 다운로드 SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

경량 설치 프로그램 — conda 또는 기존 Python 환경 불필요
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

설치 과정에서 전용 Python 3.12 런타임, Qt, PyTorch, spaCR 및 과학 계산 종속성을 내려받으므로 conda나 기존 Python 설치가 필요하지 않습니다. 기본값은 휴대용 CPU 빌드이며, 별도 안내 없이 수 GB의 CUDA 라이브러리를 내려받는 일을 방지합니다. Windows에서는 NVIDIA 가속을 선택적 설치 구성 요소로 제공하고, Linux에서는 ``--torch-backend auto`` 옵션을 사용할 수 있으며, macOS의 표준 PyTorch wheel은 Apple MPS 가속을 유지합니다.

설치 프로그램의 도움말, 진행 상황 및 오류 메시지는 영어, 스웨덴어, 독일어, 스페인어, 중국어 간체, 포르투갈어, 힌디어, 한국어, 아이슬란드어 및 프랑스어의 10개 spaCR 언어로 운영 체제 언어를 따릅니다. 지원되지 않는 로캘에서는 영어를 사용합니다.

Linux에서 다운로드된 설치기를 열기 전에 실행할 수 있도록 하십시오.

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

macOS에서는 다운로드한 ``.pkg`` 파일을 여십시오. 현재 베타 설치 프로그램이 공증되지 않아 Gatekeeper에서 차단되면 **시스템 설정 → 개인정보 보호 및 보안** 을 열고 spaCR에 대해 **그래도 열기** 를 선택한 다음 패키지를 다시 실행하십시오.

설치 프로그램은 기존 설치를 교체하기 전에 spaCR, Qt, PyTorch 및 종속성의 일관성을 확인합니다. 따라서 업데이트가 중단되어도 이전의 정상 작동 환경은 그대로 유지됩니다. 진단 로그는 spaCR 전용 설치 디렉터리에 ``install.log`` 파일로 저장됩니다.

PyPI에서 데스크톱 애플리케이션 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

그래픽 인터페이스 없이 또는 서버에 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

최신 개발 브랜치
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda 환경
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

선택 기능
~~~~~~~~~~~~~~~~~~~~~

작업 흐름에 필요한 추가 패키지만 설치하십시오:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

설치할 수 있는 추가 패키지는 Python 버전에 따라 다릅니다. Python 3.13에서는 ultrack의 제약 조건이 ``spacr[all]`` 설치를 제한하고, TorchCAM의 NumPy 제약 조건이 ``attribution`` 추가 패키지를 제한합니다. 핵심 패키지와 Qt 애플리케이션은 영향을 받지 않습니다. Python 3.14에서는 btrack를 해당 추가 패키지로 설치할 수 있습니다. pylibCZIrw CZI 변환기는 선택 기능이며 아직 테스트되지 않았지만, czifile 기반 CZI 읽기는 계속 사용할 수 있습니다.

레거시 Tk 인터페이스는 여전히 ``spacr-legacy`` 로 설치되지만 더 이상 개발되지 않습니다.


명령줄 진입점
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

문제를 해결할 때 ``SPACR_LOG_LEVEL=DEBUG`` 를 설정하십시오. 순환 로그는 ``~/.spacr/logs/spacr.log`` 파일에 기록됩니다.


기능
--------

대부분의 스크리닝에서 사용하는 6개 모듈
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** 는 Cellpose를 사용하여 2D 이미지와 볼륨 또는 시계열 데이터에서 세포, 핵, 병원체 및 세포소기관을 분할합니다. 모델 목록은 코드에 고정하지 않고 설치된 Cellpose에서 읽으며, 실행 전에 이미지에서 객체 직경을 추정합니다. 마스크는 레이어 뷰어에서 직접 수정하거나 napari로 보내 편집한 뒤 다시 가져올 수 있습니다.

**Measure** 는 객체별 형태, 강도, 질감 및 공위치 특성을 이미지 크롭과 함께 프로젝트 데이터베이스에 기록합니다. 1.5.0.0의 조명 보정은 플레이트 자체에서 플랫 필드를 추정해 강도 특성을 측정하기 전에 보정하므로 플레이트 히트맵에서 가장자리 효과로 보이는 웰 위치 편향을 제거합니다. 분할 QC 배너는 Measure 실행 전에 마스크 상태를 쉬운 말로 설명하며 실행을 막지는 않습니다. 그린 다각형으로 측정 영역을 관심 영역에 제한할 수 있습니다.

**Annotate** 는 키보드로 조작하는 그리드에 이미지 크롭을 표시하고 레이블을 SQLite에 바로 저장합니다. 화면을 떠나지 않고 기존 레이블로 모델을 다시 학습하고, 불확실성에 따라 대기열을 재정렬하며, 학습 곡선을 확인하고, 추가 레이블이 모델을 더 이상 개선하지 않을 때 중단 시점을 알려 주는 능동 학습 순환을 지원합니다. 클래스, 웰 및 플레이트별 적용 범위와 각 라운드도 기록합니다.

**Classify** 는 주석이 있는 이미지 크롭에서 PyTorch CNN 및 트랜스포머를, 측정 테이블에서 전통적 모델 또는 부스팅 모델을 학습합니다. 이제 매 epoch마다 클래스별 정확도를 보존하고, 각 체크포인트에 데이터세트, 클래스 균형, 분할 규칙 및 홀드아웃 지표를 기록한 모델 카드를 만듭니다. 평가 화면에서는 혼동 행렬의 셀을 클릭해 해당 크롭을 열 수 있으며, 확신도가 높은 오류와 불확실한 예측을 구분해 보여 줍니다.

**Map Barcodes** 는 FASTQ 리드에서 행, 열 및 gRNA 바코드를 해독하고 웰에 가이드 식별 정보를 할당한 뒤 촬영된 세포와 연결합니다. Barcode QC는 고정 임계값을 쓰지 않고 사용자가 지정한 웰당 예상 gRNA 수 주변을 탐색하여 웰당 리드 수, 충돌률 및 미매핑 비율을 보고합니다.

**Regression** 은 혼합 모델, Logistic, Probit, Quantile, Beta, 준이항 분산 GLM, Lasso, Ridge, Elastic Net, Hinge 및 Horseshoe를 포함한 17개 모델군으로 가이드, 유전자, 조건 및 대조군 효과를 추정합니다. 출력은 계수 덤프가 아니라 순위와 주석이 포함된 히트 목록입니다.

1.5.0.0의 새로운 기능
~~~~~~~~~~~~~~~~~~~~~~

스크린을 시작하기 전에 Power / Design 모듈이 필요한 세포 수와 웰 수를 계산하며, 이때 시퀀싱 오류와 촬영된 세포가 너무 적은 웰의 누락률도 반영합니다. 실험 설계 도구는 플레이트, 대조군 및 반복군을 배치하고 그 레이아웃을 파이프라인용으로 내보냅니다. 실험 후에는 QC 대시보드가 분할, 플레이트, 주석자 간 일치도 및 데이터 누출 검사를 하나의 판정으로 종합하며, 배치 보정에는 ``center`` 및 ``zscore`` 외에 ComBat도 사용할 수 있습니다.

결과는 내보낸 뒤 다시 가져오지 않고 spaCR 안에서 바로 탐색합니다. Graph Builder에서 열을 x, y, 색상, 크기 및 면으로 끌어다 놓아 테이블을 그래프로 표현할 수 있습니다. 히스토그램이나 산점도에 그린 게이트는 필터가 되며, Feature Explorer는 클래스를 잘 구분하는 순으로 특징을 정렬합니다. 작은 다중 그래프, 용량-반응 적합, 관리도 및 견고한 이상치 탐지가 같은 축 엔진을 사용합니다. 한 보기에서 개체를 선택하면 모든 보기에서 동일한 개체가 선택되고, 선택 항목을 열면 해당 개체의 이미지 크롭이 표시됩니다. Layer Viewer는 이미지, 레이블, 점 및 도형을 층으로 쌓고, 직교 보기, 동기화된 비교 그리드, 그리고 세포에서 핵과 병원체로 이어지는 계보 트리를 제공합니다.

이제 각 실행을 명확하게 추적할 수 있습니다. 모든 실행에는 실행 ID, 시드 및 ``on_error`` 정책이 있으며, Mask, Measure, Classify 및 AnnData 내보내기는 생성한 항목을 아티팩트 레지스트리에 기록하므로 출력 파일에서 해당 파일을 만든 설정까지 거슬러 올라갈 수 있습니다. 모듈은 이전 단계가 실제로 기록한 출력을 열고, 파이프라인 그래프는 오래된 출력을 표시하며, 실행 비교는 두 실행의 설정, 객체 수 및 히트 목록 차이를 보여 줍니다. 모든 GUI 실행은 동일한 작업을 수행하는 Python 스크립트도 생성합니다. 측정값은 scanpy용 ``.h5ad`` 파일로 내보낼 수 있으며 OME-Zarr와 OMERO는 Python API에서 사용할 수 있습니다. 메서드 및 결과 내보내기는 실행의 구조화된 요약을 바탕으로 원고의 두 섹션을 작성합니다. 모델이 문장을 쓰지만 모든 숫자는 요약에서 가져오며, 요약에 없는 숫자가 포함된 초안은 거부됩니다. 설치에 문제가 있으면 ``spacr-doctor`` 도구가 실제로 실행 중인 spaCR, GPU 사용 가능 여부, spaCR이 호출하는 API와 Cellpose의 호환 여부, 프로젝트 데이터베이스와 설정의 유효성을 보고하고 실패한 검사마다 복사할 수 있는 해결 방법을 제시합니다.

다국어 데스크톱 인터페이스
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → 환경설정 → 언어** 메뉴에서 애플리케이션을 다시 시작하지 않고 영어, 스웨덴어, 독일어, 스페인어, 중국어 간체, 포르투갈어, 힌디어, 한국어, 아이슬란드어 또는 프랑스어로 전환할 수 있습니다. 선택한 언어는 저장되며 나중에 여는 화면에도 적용됩니다.

탐색 메뉴, 환경설정, AI 및 LIVE 컨트롤, 모듈 설명과 spaCR이 작성한 콘솔 알림은 선택한 언어로 표시됩니다. 워커 출력, 로그, 트레이스백, 경로, 데이터베이스 값, 주석, AI 응답, 측정값 및 저장된 결과는 번역하지 않으므로 과학적 출력은 표준 영어 형식을 유지합니다. 아직 해당 언어로 검토되지 않은 설정 도구 설명은 언어가 뒤섞인 설명이 되지 않도록 영어로 유지됩니다. `현지화 안내서 <https://einarolafsson.github.io/spacr/localization.html>`_ 에는 이 동작, 환경 변수 재정의 및 함께 번역되는 `상황별 도움말 <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ 이 설명되어 있습니다.

애니메이션 설정 안내
~~~~~~~~~~~~~~~~~~~~~~

94개의 짧은 애니메이션으로 143개 시각 설정이 이미지에 미치는 영향을 확인할 수 있습니다. 설정 위에 포인터를 놓고 도구 설명의 **애니메이션** 버튼을 클릭하면 텍스트 옆의 사각형 미리 보기가 재생되며, 다시 클릭하면 접힙니다. 애니메이션은 요청할 때만 재생되고 환경설정에서 완전히 끌 수도 있습니다. `갤러리 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ 에는 모든 애니메이션이 표시되고, `설정 애니메이션 레지스트리 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ 에는 각 애니메이션이 연결된 설정이 기록됩니다.

모듈 참조
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 모듈
     - 기능
     - 상태
     - 설명
   * - **데스크톱 환경**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - 안정
     - 열려 있거나 필요할 때 생성되는 화면을 10개 내장 언어로 즉시 다시 번역합니다.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - 안정
     - API URL은 그대로 유지하면서 모듈 요약과 설정 도움말 인터페이스를 현지화합니다.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - 안정
     - 사용자 또는 모델 콘텐츠를 변경하지 않고 AI 및 LIVE 컨트롤을 현지화합니다.
   * - |api-animations|_
     - |doc-animations|_
     - 안정
     - 설정 도구 설명에서 143개 시각 설정을 위한 94개 내장 애니메이션을 재생합니다.
   * - |api-selection|_
     - |api-linked-views|_
     - 알파
     - 테이블, 플레이트, 임베딩, 산점도 및 그래프 보기에서 하나의 객체 선택을 공유합니다.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - 알파
     - GPU, Cellpose API, 데이터베이스 및 설정을 진단하고 실패한 각 검사에 해결 방법을 제시합니다.
   * - **이미지 분석**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - 안정
     - 2D 이미지에서 세포, 핵, 병원체 및 세포소기관을 분할합니다.
   * - |api-mask|_
     - |api-mask-3d|_
     - 베타
     - 3D 볼륨 이미지와 4D 시계열을 분할합니다.
   * - |api-illumination|_
     - |api-flatfield|_
     - 알파
     - 플레이트에서 플랫 필드를 추정하고 강도를 측정하기 전에 보정합니다.
   * - |api-measure|_
     - |api-measure-2d|_
     - 안정
     - 형태, 강도, 질감 및 공위치를 측정하고 이미지 크롭을 저장합니다.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - 알파
     - Measure 실행 전에 분할 상태를 설명하되 실행을 차단하지는 않습니다.
   * - |api-timelapse|_
     - |api-tracking|_
     - 베타
     - IoU, Trackpy, btrack, Trackastra 또는 ultrack으로 객체를 추적하고 운동성을 정량화합니다.
   * - |api-layers|_
     - |api-layer-viewer|_
     - 알파
     - 직교 보기와 비교 그리드에서 이미지, 레이블, 점 및 도형 레이어를 겹쳐 표시합니다.
   * - |api-napari|_
     - |api-napari-curation|_
     - 알파
     - 마스크를 napari로 보내 수정한 뒤 다시 가져오며 모든 편집 내용을 기록합니다.
   * - **AI 및 표현형 분석**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - 안정
     - 키보드로 조작하는 그리드에서 이미지 크롭을 검토하고 주석을 SQLite에 저장합니다.
   * - |api-active-learning|_
     - |api-al-loop|_
     - 알파
     - Annotate 안에서 모델을 다시 학습하고 불확실성에 따라 순위를 조정하며 레이블링 중단 시점을 알려 줍니다.
   * - |api-classify|_
     - |api-classification|_
     - 안정
     - PyTorch CNN 및 트랜스포머 모델을 학습하고 적용합니다.
   * - |api-classify|_
     - |api-model-cards|_
     - 알파
     - 각 체크포인트에 데이터세트, 클래스 균형, 분할 규칙 및 홀드아웃 지표를 기록합니다.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - 알파
     - 혼동 행렬 셀에 해당하는 이미지 크롭을 열고 확신도가 높은 오류와 불확실한 항목을 구분해 표시합니다.
   * - |api-ml|_
     - |api-ml-models|_
     - 안정
     - 측정 테이블에서 해석 가능한 전통적 모델과 부스팅 모델을 학습합니다.
   * - |api-classify|_
     - |api-activation|_
     - 베타
     - Captum, SmoothGrad 및 TorchCAM으로 예측을 설명합니다.
   * - |api-umap|_
     - |api-embedding|_
     - 베타
     - 이미지 임베딩을 대화형으로 탐색하고 클러스터 레이블을 전파합니다.
   * - **시퀀싱 및 스크리닝 분석**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - 안정
     - FASTQ 리드에서 행, 열 및 gRNA 바코드를 매핑하고 촬영된 세포에 가이드를 할당합니다.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - 알파
     - 웰당 예상 gRNA 수를 기준으로 웰당 리드 수, 충돌률 및 미매핑 비율을 보고합니다.
   * - |api-regression|_
     - |api-regression-models|_
     - 안정
     - 17개 모델군으로 가이드, 유전자, 조건 및 대조군 효과를 추정합니다.
   * - |api-power|_
     - |api-power-design|_
     - 알파
     - 시퀀싱 오류와 웰 탈락을 반영하여 스크리닝에 필요한 세포 및 웰 수를 계산합니다.
   * - |api-graph|_
     - |api-graph-builder|_
     - 알파
     - 열을 x, y, 색상, 크기 및 패싯으로 끌어 놓아 플롯을 만듭니다.
   * - |api-artifacts|_
     - |api-provenance|_
     - 알파
     - Mask, Measure, Classify 및 내보내기 출력의 실행 ID, 시드 및 설정을 기록합니다.

.. |api-qt-app| replace:: **Qt 애플리케이션**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **10개 언어 현지화**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **현지화된 상황별 도움말**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI 지원 콘솔**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **설정 애니메이션 레지스트리**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **시각 설정 애니메이션**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **선택**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **연결된 선택**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **설치 진단**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D 마스크 생성**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D 및 4D 마스크 생성**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **조명 보정**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **플랫 필드 보정**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **객체 측정**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **분할 품질 관리**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **실행 전 평가**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **객체 추적**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **레이어**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **레이어 뷰어**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari 연동**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **마스크 교정**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **수동 주석**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **능동 학습**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **재학습 및 재정렬**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **이미지 분류**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **모델 카드**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **활성화 맵**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **혼동 행렬 상세 보기**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **머신 러닝**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **측정값 분류**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **대화형 임베딩**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **시퀀싱**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **바코드 매핑**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **바코드 품질 관리**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **웰 및 충돌 보고서**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **스크리닝 효과 추정**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **통계적 검정력 및 설계**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **아티팩트**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **실행 출처**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


데이터
------

참조 데이터세트
~~~~~~~~~~~~~~~~~~

- `전체 현미경 데이터세트: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `테스트 데이터 세트: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `시퀀싱 데이터: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `검정력 분석: spaCRPower <https://github.com/maomlab/spaCRPower>`_


기여 및 지원
------------------------

오류 보고서와 구체적인 기능 요청은 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ 에서 접수합니다. 오류를 보고할 때는 spaCR 버전, 운영 체제, Python 버전, 모듈 설정 및 관련 로그 일부를 포함해 주십시오. ``spacr-doctor`` 도구가 이 정보의 대부분을 자동으로 수집합니다.

라이선스
~~~~~~~~~

현재 개발 브랜치의 소스는 `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ 조건으로 사용할 수 있습니다. 상업적 사용에는 저작권자와의 별도 라이선스가 필요합니다. spaCR 1.4.9.9까지 배포된 버전은 해당 배포판에 포함된 MIT License로 계속 제공됩니다.

튜토리얼
~~~~~~~~~

`대화형 spaCR 튜토리얼 모음 <https://einarolafsson.github.io/spacr/tutorials/>`_ 에는 설치와 각 애플리케이션 작업 흐름을 설명하는 내레이션 및 자막 안내가 8개 언어로 제공됩니다.

spaCR 인용
~~~~~~~~~~~~

spaCR이 연구에 기여한다면 다음을 인용하십시오 :

Olafsson EB, *et al.* 풀드 이미지 기반 CRISPR 스크린에서 EAF1을 *T. gondii*\ 의 ESCRT 기능 탈취 조절 인자로 확인했습니다.

`BioRxiv 프리프린트 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `소프트웨어 아카이브 <https://doi.org/10.5281/zenodo.21343317>`_
