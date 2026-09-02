|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: 최신 설치 프로그램
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 버전

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

**CRISPR 스크리닝의 공간 표현형 분석.**

spaCR는 고함량 현미경 영상에서 단일 세포를 분할하고 측정하며, 객체별 표현형을 시퀀싱에서 산출한 가이드 풍부도와 통합하고, 어떤 유전자가 표현형 변화와 연관되는지 추정합니다. 플레이트 영상과 FASTQ 리드에서 시작하여 객체별 측정값, 학습된 분류기, 가이드별·유전자별 효과 추정치, 우선순위가 지정된 히트 목록을 생성합니다.

분류, 측정, 기록 및 분류 모듈은 또한 순서 팔없이 실행됩니다.

이미지, 마스크, 크롭, 측정, 기록, 예측, 바코드 및 잘 식별자는 하나의 SQLite 프로젝트에서 살고 있습니다.

데스크톱 응용 프로그램으로 실행되거나 워크 스테이션, 서버 또는 클러스터에서 헤드없이 실행됩니다.

하드웨어 지원
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


spaCR 설치
-------------

데스크톱 애플리케이션
~~~~~~~~~~~~~~~~~~~~~

The installers bundle their own Python. Conda is not required.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11용 spaCR 1.5.0.4 다운로드
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel 및 Apple Silicon)용 spaCR 1.5.0.4 다운로드
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64비트 Linux용 spaCR 1.5.0.4 다운로드
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: 이전 spaCR 설치 프로그램
   :target: ../../source/installers.rst

.. spacr-installer-links-end

첫 번째 세 개의 아이콘이 현재 버전을 다운로드합니다. spaCR 아이콘은 전체 설치기 아카이브를 열어줍니다. 설치기 링크와 버전 된 파일 이름은 버전 작업 흐름에 의해 업데이트됩니다; 이전 설치기는 동일한 버전기록에 남아 있습니다.

Linux에서는 다운로드한 파일에 실행 권한을 부여한 후 실행합니다:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

macOS에서는 ``.pkg``\ 를 여세요. 현재 베타는 공증되지 않았습니다. Gatekeeper가 차단하면 **시스템 설정 → 개인정보 보호 및 보안 → 그래도 열기**\ 를 선택하세요.

업데이트, 제거, 오프라인 설치 및 문제 해결 지침은 `설치 가이드 <../../source/installer_guide.rst>`_\ 를 참조하십시오.

PyPI 설치
~~~~~~~~~~~~~~~~~

PyPI 릴리스는 Conda 환경 안에서 pip로 spaCR를 설치하세요. Python 3.12에서 선택 가능한 과학 패키지의 범위가 가장 넓습니다:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML.

서버, 클러스터 또는 CI 실행 환경에서는 Qt를 제외합니다:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

선택적 통합 기능은 ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` 및 ``spacr[czi,nd2,lif]``\ 과 같이 별도로 설치합니다. 전체 추가 기능 목록과 Python 버전 호환성 표는 `설치 안내서 <../../source/installer_guide.rst>`_\ 를 참조하십시오.

conda-forge 설치
~~~~~~~~~~~~~~~~~~~~~~~~

공식 conda-forge 패키지는 활성 환경에 spaCR 및 데스크톱 실행에 필요한 종속성을 설치합니다:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

출처에서 설치하기
~~~~~~~~~~~~~~~~~~~

저장소를 클론하고 편집 가능한 모드에 설치하여 작업 복사본 *is* 설치된 패키지 및 편집이 다시 설치하지 않고 효력을 발휘합니다.::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

기본 지점은 ``nightly``입니다.특정 릴리스를 위해::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

나중에 변화를 끌어내기 위해, 클론 내부에서::

    git pull
    pip install -e .

두 번째 라인은 의존 또는 입력 포인트가 변경되면만 필요합니다; Python 코드는 그것없이 수집됩니다. ``spacr-doctor`` 명령이 끌고 나서 여전히 오래된 코드를 실행하는 경우, ``spacr``는 실제로 당신의 길에 있으며, 이는 일반적인 원인입니다.

출처에서 설치 (빛)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

전체 클론: 427 MB 코어 클론 : 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

스키프 ``docs/``, ``tests/``,Cellpose 체크 포인트, 아카이브 된 숫자 및 확장 번역 카탈로그.

옵션: ``--dir``, ``--branch`` (기본 ``main``), ``--with-tests``,``--with-docs``, ``--with-translations`` 및 ``--no-install``.

``packaging/source_install_excludes.txt``는 각각의 횡단 경로를 나열합니다.


명령줄 진입점
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

문제를 해결할 때 ``SPACR_LOG_LEVEL=DEBUG``\ 로 설정하세요. 순환 로그는 ``~/.spacr/logs/spacr.log``\ 에 기록됩니다.

``spacr-run --list``\ 는 그래픽 인터페이스 없이 실행할 수 있는 명령줄 진입점이 있는 모듈을 나열합니다. GUI에서만 제공되는 주석, 큐레이션, 비교 및 탐색 모듈은 목록에서 제외됩니다.


핵심 워크플로
-------------

기본 워크플로는 6개 모듈로 구성됩니다:

- **Mask** Cellpose로 세포, 핵, 병원체 및 세포소기관을 분할합니다.
- **Measure** 형태, 강도, 텍스처, 공간 및 공위치 특성과 객체 크롭을 SQLite에 저장합니다.
- **Annotate** 키보드로 조작하는 격자에서 크롭에 라벨을 지정하고 능동 학습 대기열을 지원합니다.
- **Classify** 이미지 또는 측정값 기반 모델을 학습하고 각 체크포인트에 홀드아웃 데이터 성능을 기록합니다.
- **Map Barcodes** FASTQ 리드를 웰과 gRNA에 매핑하고 풍부도, 충돌 및 커버리지 QC를 제공합니다.
- **Regression** 연속형, 비율형 및 계수형 반응에 적합한 모델 계열로 가이드, 유전자, 조건 및 대조군 효과를 추정합니다.

동일한 프로젝트에서 플레이트 설계, 검정력 추정, 배치 효과 보정, 세그멘테이션 품질 점검, 연결된 플롯과 크롭 탐색, AnnData 내보내기, 중단된 작업 재개 및 각 결과에 사용된 설정 기록도 수행할 수 있습니다.

spaCR 모듈
-------------

.. spacr-workflow-begin

|Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

|Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

|Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|

|Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Mask API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Measure API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Annotate API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Classify API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Map Barcodes API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Regression API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Import API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Run Compare API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Experiment Design API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Power / Design API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Dose–Response API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: QC API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Make Masks API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Align & Stitch API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Image UMAP API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Gate Editor API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Graph Builder API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Plaque Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Recruitment API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Invasion Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Replication Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

Every module spaCR ships, in the order the home screen lists them: the six pipeline modules first, then everything else. Select a tile to open that module's API page.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose-SAM runs here show the cell-probability map and the flow field beside the mask. See the `가이드 가이드 <../../source/features.rst>`_ for each tool.

**다른 자원**

- `인터랙티브 튜토리얼 <https://einarolafsson.github.io/spacr/tutorials/>`_ — 설치에서 히트 조사를 통해 73 개의 지시된 작업 흐름.
- `Python API 빠른 시작 <../../source/python_api.rst>`_ - 스크립트, 노트북 또는 클러스터에서 튜브를 실행하고 검증합니다.
- `특징 가이드 <../../source/features.rst>`_ - 능력, 성숙성 및 선택적 통합.
- `정리된 API 참조 <https://einarolafsson.github.io/spacr/api/index.html>`_ - 작업에 따라 지원되는 입력 포인트, 전체 모듈 참조 1 레벨 더 깊습니다.
- `언어 & 번역 가이드 <../../source/localization.rst>`_ - 인터페이스 언어, 컨텍스트 지원 및 과학 출력 정책.

언어 및 번역
~~~~~~~~~~~~~~~~~~~~~~

인터페이스는 탐색 및 환경 설정에서 10개 언어를 지원합니다. AI 및 LIVE 컨트롤, 모듈 설명과 검토된 상황별 도움말도 번역됩니다. 다시 시작하지 않고 **spaCR → 환경 설정 → 언어** 메뉴에서 언어를 변경할 수 있습니다. 로그, 경로, 데이터베이스 값과 측정값은 번역하지 않으며 과학적 출력은 표준 영어로 유지됩니다. `상황별 도움말 정책 <../../source/localization.rst#contextual-help>`_ 문서를 참조하세요.

애니메이션 설정 안내
~~~~~~~~~~~~~~~~~~~~~~~~~

시각적 설명이 있는 설정은 도구 설명에 **Animation** 컨트롤을 제공합니다. 다음 리소스를 살펴보세요: `설정 애니메이션 갤러리 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ 및 `설정 애니메이션 레지스트리 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

데이터
------

참조 데이터세트
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: BioStudies 현미경 데이터세트 열기
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Hugging Face 테스트 데이터세트 열기
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: NCBI 시퀀싱 데이터세트 열기
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: spaCRPower 열기
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: bioRxiv 사전 인쇄본 열기
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

모델 동물원
~~~~~~~~~~~

spaCR는 학습된 모델 카탈로그를 함께 제공하며 필요할 때 내려받습니다. 홈 화면에서 **Model Zoo**를 열어 모델을 살펴보고 설치하거나, 설정 파일에 키를 지정하면 -- ``pathogen_model: toxoplasma_pv_v1`` -- 처음 필요한 시점에 모델을 내려받고 체크섬을 검증합니다. 공개된 항목은 모두 SHA-256을 포함하며, 이것이 없는 항목은 설치하지 않고 거부합니다. 잘리거나 바꿔치기된 체크포인트는 진짜와 구별할 수 없기 때문입니다.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Key
     - Trained on
     - Measured performance and limits
   * - ``toxoplasma_pv_v1``
       (cpsam_v2_toxo_r2)
     - Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. 115 pairs (104 train / 11 test), 100 epochs, base cpsam_v2
     - F1 0.867 at IoU 0.5 against 0.713 for stock cpsam; AJI 0.808 against 0.426; accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
   * - ``toxoplasma_plaque_v1``
       (cpsam_plaque_r3)
     - Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set
     - F1 0.856 in-domain and 0.834 on the literature set, against 0.718 / 0.755 for round 1; round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
   * - ``toxoplasma_well_detector_v1``
       (yolo_welldetect_v3.pt)
     - Whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987; locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

.. spacr-model-zoo-end

위 수치는 공개 시점에 측정한 값이며, 한계도 함께 명시되어 있습니다. 모델은 측정한 작업에 유용할 뿐 모든 작업에 유용한 것은 아닙니다. ``toxoplasma_well_detector_v1``과 ``toxoplasma_plaque_v1``은 하나의 파이프라인을 이루는 두 부분입니다. 검출기가 웰을 찾고, 분할 모델이 그 안의 플라크를 찾으며, 웰 지름 덕분에 서로 다른 현미경 사이에서 면적을 비교할 수 있습니다.

모델은 각 작성자 본인의 Hugging Face 계정에 호스팅되므로, 모델을 기여한다고 해서 다른 사람의 계정에 쓰기 권한을 넘겨줄 필요가 없습니다. ``spacr.model_zoo``의 ``publish_model``이 업로드를 수행하고 추가할 카탈로그 항목을 출력합니다.


성능 진단
----------------------

하드웨어 보고서를 생성하여 성능 관련 이슈에 첨부하세요::

    python tools/spacr_hardware_report.py

``~/.spacr/reports``로 저장하고 경로를 인쇄합니다. ``--quick``는 더 긴 좌표를 스키; ``--out PATH``는 위치를 설정합니다.

Reads no project data. Times imports, numeric libraries, window construction and animation. Reports processor-architecture emulation (an x86_64 Python build on Apple Silicon) and NumPy's BLAS implementation.

명령선 참조
----------------------

Every command below is installed by ``pip install spacr``. All of them accept ``--help``.

신청을 시작하는 방법
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` modal 설정 화면을 스키, 그렇지 않으면 예상치 못한 작업을 차단합니다.

``spacr-qt`` 및 ``spacr-nightly``은 ``spacr``의 동화입니다.

spaCR 시작하지 않을 때
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor``은 각 실패에 대해 실행하는 명령이있는 한 줄을 인쇄합니다. ``spacr``이 경로에있는 것을보고합니다.

``safespacr``은 각 선호도를 기본으로 읽고 배경 화면, 애니메이션, 구두 로그인 및 프리로드를 강요합니다. 저장된 선호도가 출시를 깨면 사용합니다.

모듈을 끊임없이 실행하는 방법
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qt 없으며, 클러스터, 서버 및 CI를 위한 디스플레이가 없습니다.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate``은 실행하는 것과 동일한 설정을 읽고 실종되는 것, 반대되는 것 또는 아무것도 지적하지 않는 것을 보고합니다.

``spacr-run --list``는 헤드없는 입력 지점을 가진 모듈만 표시되며, 메모, 치유 및 탐험은 상호 작용하고 놓치지 않습니다.

나중에 경주를 검사합니다.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

각 라운드는 ``~/.spacr/runs``로 기록되며 설정, 해시된 입력, 출력, 경고, 버전 및 씨앗이 있습니다.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

데이터 검토 및 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

환경
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

회전 기록은 ``~/.spacr/logs/spacr.log``로 작성됩니다.이 파일을 오류 보고서에 붙여 넣으십시오.


기여 및 지원
------------------------

버그 보고와 범위가 명확한 기능 요청은 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_\ 를 통해 제출하세요. 오류를 보고할 때는 spaCR 버전, 운영 체제, Python 버전, 모듈 설정 및 관련 로그 일부를 포함하십시오. ``spacr-doctor``\ 가 이 정보의 대부분을 수집합니다. 성능 문제를 보고할 때는 하드웨어 보고서도 포함하십시오.

라이선스
~~~~~~~~~

spaCR is released under the `BSD 3 클래스 라이센스 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

If spaCR contributed to published work, a citation is appreciated and is not a condition of the licence — see `Citing spaCR`_ below.

튜토리얼
~~~~~~~~~

`대화형 spaCR 튜토리얼 라이브러리 <https://einarolafsson.github.io/spacr/tutorials/>`_\ 에는 설치 및 각 애플리케이션 워크플로를 설명하는 음성·자막 안내가 있으며, 8개 언어의 50개 음성으로 제작된 73개 강의가 포함되어 있습니다.

spaCR 인용
~~~~~~~~~~~~

spaCR가 연구에 기여했다면 다음을 인용해 주세요:

Olafsson EB, *et al.* 풀드 이미지 기반 CRISPR 스크린은 EAF1을 *T. gondii*\ 의 ESCRT 기능 탈취 조절 인자로 규명합니다.

`BioRxiv 프리프린트 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `소프트웨어 아카이브 <https://doi.org/10.5281/zenodo.21343316>`_

감사의 말
~~~~~~~~~~~~~~~

spaCR는 NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch 및 Qt를 비롯한 개방형 과학 소프트웨어를 기반으로 합니다. 다국어 문서와 인터페이스 카탈로그 작성에 사용된 모델은 `번역 모델 표기 <../TRANSLATION_MODELS.md>`_ 문서에서 확인할 수 있습니다.
