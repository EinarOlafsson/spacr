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
   :alt: BSD 3-Clause 라이선스
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

영상 기반 풀드 CRISPR 스크리닝에서 spaCR는 영상 분할부터 히트 우선순위 지정까지의 작업 흐름을 제공합니다. 시퀀싱 기반 스크리닝이 없는 고함량 현미경 연구에서는 분할, 측정, 주석, 분류 모듈을 독립적으로 사용할 수 있습니다.

영상, 마스크, 이미지 크롭, 측정값, 주석, 예측, 바코드 및 웰 식별자는 하나의 SQLite 프로젝트에 저장되므로 결과의 값을 그 출처 객체까지 추적할 수 있습니다.

spaCR를 데스크톱 애플리케이션으로 실행하거나 워크스테이션, 서버 또는 클러스터에서 그래픽 인터페이스 없이 실행할 수 있습니다. 두 방식 모두 동일한 모듈을 사용하며, 모듈이 지원하면 CUDA가 자동으로 활성화됩니다.


작업 흐름 개요
--------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Mask API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Measure API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Annotate API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Classify API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Map Barcodes API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Regression API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**데이터**

|App_foreign|\ |App_run_compare|\ |App_experiment_design|\ |App_power|\ |App_dose_response|\ |App_qc_dashboard|

**Tools**

|App_make_masks|\ |App_align|\ |App_umap|\ |App_gate_editor|\ |App_graph_builder|

**분석**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.583%
   :alt: Import API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.583%
   :alt: Run Compare API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.583%
   :alt: Experiment Design API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.583%
   :alt: Power / Design API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.583%
   :alt: Dose–Response API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.583%
   :alt: QC API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.583%
   :alt: Make Masks API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.583%
   :alt: Align & Stitch API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.583%
   :alt: Image UMAP API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.583%
   :alt: Gate Editor API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.583%
   :alt: Graph Builder API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.583%
   :alt: Plaque Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.583%
   :alt: Recruitment API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.583%
   :alt: Invasion Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.583%
   :alt: Replication Assay API 열기
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

작업 흐름 모듈을 선택하면 해당 API 페이지가 열립니다. 격자에는 나머지 모든 애플리케이션이 spaCR 홈 화면과 동일한 범주와 순서로 배치되어 있습니다.


spaCR 설치
-------------

데스크톱 애플리케이션
~~~~~~~~~~~~~~~~~~~~~

데스크톱 설치에는 개인 Python 환경이 포함되어 있으므로 콘다와 기존 Python 설치가 필요하지 않습니다.

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

conda-forge 설치
~~~~~~~~~~~~~~~~~~~~~~~~

공식 conda-forge 패키지는 활성 환경에 spaCR 및 데스크톱 실행에 필요한 종속성을 설치합니다:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

PyPI 설치
~~~~~~~~~~~~~~~~~

PyPI 릴리스는 Conda 환경 안에서 pip로 spaCR를 설치하세요. Python 3.12에서 선택 가능한 과학 패키지의 범위가 가장 넓습니다:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR는 Python **3.9~3.14** 버전을 지원하지만 torchvision이 제외하는 Python 3.14.1은 지원하지 않습니다. CUDA 워크플로에는 Linux를 권장하며 macOS와 Windows도 지원합니다.

서버, 클러스터 또는 CI 실행 환경에서는 Qt를 제외합니다:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

선택적 통합 기능은 ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` 및 ``spacr[czi,nd2,lif]``\ 과 같이 별도로 설치합니다. 전체 추가 기능 목록과 Python 버전 호환성 표는 `설치 안내서 <../../source/installer_guide.rst>`_\ 를 참조하십시오.

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


할 수 있는 일
---------------

기본 워크플로는 6개 모듈로 구성됩니다:

- **Mask** Cellpose로 세포, 핵, 병원체 및 세포소기관을 분할합니다.
- **Measure** 형태, 강도, 텍스처, 공간 및 공위치 특성과 객체 크롭을 SQLite에 저장합니다.
- **Annotate** 키보드로 조작하는 격자에서 크롭에 라벨을 지정하고 능동 학습 대기열을 지원합니다.
- **Classify** 이미지 또는 측정값 기반 모델을 학습하고 각 체크포인트에 홀드아웃 데이터 성능을 기록합니다.
- **Map Barcodes** FASTQ 리드를 웰과 gRNA에 매핑하고 풍부도, 충돌 및 커버리지 QC를 제공합니다.
- **Regression** 연속형, 비율형 및 계수형 반응에 적합한 모델 계열로 가이드, 유전자, 조건 및 대조군 효과를 추정합니다.

동일한 프로젝트에서 플레이트 설계, 검정력 추정, 배치 효과 보정, 세그멘테이션 품질 점검, 연결된 플롯과 크롭 탐색, AnnData 내보내기, 중단된 작업 재개 및 각 결과에 사용된 설정 기록도 수행할 수 있습니다.

호스트 화면에서 사용할 수 있는 모듈
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

20개 모듈은 별도의 Home 타일로 표시되지 않고 관련 호스트 화면에 통합되어 있습니다. 각 모듈은 호스트 화면의 상단 헤더에서 열리며 활성 프로젝트를 사용합니다. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP 및 Make Masks에서 이러한 통합 모듈을 제공합니다. 도움말과 API 문서는 계속 사용할 수 있으며, 파이프라인 진입점이 있는 모듈은 그래픽 인터페이스 없이도 실행할 수 있습니다. `기능 안내서 <../../source/features.rst>`_\ 에는 각 통합 모듈과 해당 호스트 화면이 나열되어 있습니다.

Make Masks
~~~~~~~~~~

Make Masks는 **Data** 아래에 있으며 세그멘테이션 마스크를 수동으로 수정하는 기능을 제공합니다. 상단 헤더에서 Cellpose 워크플로에도 접근할 수 있습니다. 캔버스에는 아홉 가지 도구가 있습니다: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** 및 **Recrop**. Draw는 자유 형식의 닫힌 윤곽선으로 채워진 레이블 하나를 생성합니다. Divide는 사용자가 지정한 선을 따라 합쳐진 객체를 분리하면서 다른 객체 레이블은 모두 보존합니다.

Recrop은 큐레이션을 위해 준비된 여러 객체 이미지에서 단일 객체 필드를 추출합니다. 한 객체 주위에 경계 상자를 지정하면 대응하는 이미지 및 마스크 영역을 새 필드로 기록하고, 해당 필드를 현재 필드 다음에 배치하며, 원래의 다중 객체 필드를 큐레이션 대기열에서 제거합니다. Recrop은 레이블 픽셀을 편집하는 대신 활성 필드를 바꿉니다.

Make Masks에서 Cellpose-SAM을 실행하면 마스크 옆에 두 가지 중간 출력인 **세포 확률 맵**\ 과 **흐름장**\ 이 표시됩니다. 마스크는 확률 맵의 임계값으로 정의되며, 흐름 일관성 검사는 계산된 흐름이 예측된 흐름장과 다른 객체를 제외할 수 있습니다. 잘못되거나 불완전한 마스크를 평가할 때 이 출력들을 확인하여 낮은 세포 확률과 일관되지 않은 흐름을 구분하십시오.

객체 및 설정
~~~~~~~~~~~~~~~~~~~~

spaCR는 세포, 핵 및 병원체 객체, 이 객체들의 마스크에서 파생되는 세포질, 그리고 0개에서 26개까지의 세포소기관 슬롯을 지원합니다. 각 세포소기관 슬롯에는 독립적인 채널, 직경, 형태 프리셋 및 검출 방법이 있습니다.

설정 패널은 적용되는 경우에만 컨트롤을 표시합니다. 설정된 개수를 초과하는 세포소기관 슬롯은 숨겨지고, 채널이 지정되지 않은 객체는 실행에서 제외되며, 형태별 컨트롤은 선택한 방법에 해당할 때만 표시됩니다. **3D**와 **Time** 스위치는 차원을 정의합니다. ``z_stack``\ 은 체적 설정을 활성화하고, ``timelapse``\ 는 추적 설정을 활성화하며, 두 스위치를 모두 활성화하면 4차원 설정이 표시됩니다.

다음 페이지를 선택하십시오 당신이 원하는 것에 따라 :

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

성능 진단
----------------------

하드웨어 보고서를 생성하여 성능 관련 이슈에 첨부하세요::

    python tools/spacr_hardware_report.py

이 명령은 보고서를 출력하고 ``~/.spacr/reports`` 아래에 사본을 저장하며, 마지막 줄에 저장된 경로를 표시합니다. ``--quick``\ 은 시간이 오래 걸리는 벤치마크를 생략하고, ``--out PATH``\ 는 다른 출력 위치를 지정합니다.

이 보고서는 프로젝트를 열거나 프로젝트 데이터를 읽지 않습니다. 가져오기 및 수치 라이브러리 실행 시간, 디스플레이 배율, 활성 환경 설정, 기본 창과 모듈 화면의 구성, 애니메이션 성능을 기록합니다. 보고서 파일이 생성되는 유일한 출력입니다.

또한 Apple Silicon에서 실행되는 x86_64 Python 빌드와 같은 프로세서 아키텍처 에뮬레이션과 NumPy가 사용하는 BLAS 구현을 확인합니다. 어느 쪽이든 성능에 상당한 영향을 줄 수 있습니다.

기여 및 지원
------------------------

버그 보고와 범위가 명확한 기능 요청은 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_\ 를 통해 제출하세요. 오류를 보고할 때는 spaCR 버전, 운영 체제, Python 버전, 모듈 설정 및 관련 로그 일부를 포함하십시오. ``spacr-doctor``\ 가 이 정보의 대부분을 수집합니다. 성능 문제를 보고할 때는 하드웨어 보고서도 포함하십시오.

라이선스
~~~~~~~~~

spaCR는 `BSD 3-Clause License <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ 에 따른 오픈 소스이며, CellProfiler·napari·Cellpose와 같은 라이선스입니다. 상업적 용도를 포함해 어떤 목적으로도 사용할 수 있습니다. 1.5.0.0부터 1.5.0.4까지의 릴리스는 PolyForm Noncommercial License 1.0.0을, 1.4.9.9까지의 버전은 MIT License를 따랐으며, 해당 릴리스는 함께 제공된 라이선스에 따라 계속 사용할 수 있습니다.

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
