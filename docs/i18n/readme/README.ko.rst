|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

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
   :alt: spaCR workflow and output organization
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

spaCR는 Python **3.9에서 3.14**까지 지원합니다 (그것은 Python 3.14.1을 제외하고, 그 후반 시각을 제한합니다). Python 3.12은 선택적 과학 패키지의 가장 광범위한 선택을 가지고 있습니다. Linux는 작업 흐름에 대한 권장됩니다. CUDA; macOS 및 Windows도 지원됩니다.


설치 세부 정보
--------------------

|Release| |PyPI| |CondaRecipe|

**(베타) Lightweight 데스크톱 설치기:**

.. spacr-installer-links-begin

* `Windows 10/11: 다운로드 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (인텔 및 애플 실리콘): 다운로드 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64비트 Linux: 다운로드 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

경량 설치 프로그램 — conda 또는 기존 Python 환경 불필요
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

설치자는 개인 Python 3.12 실행 시간, Qt, PyTorch, spaCR 및 설치 중 과학적 의존성을 다운로드, 그래서 콘다 또는 기존 Python가 필요하지 않습니다. 휴대용 CPU 건물은 기본으로, 설치를 유지하는 CUDA 라이브러리의 몇 개의 히가비트를 끌어 내지 않습니다. Windows는 NVIDIA 가속을 제공합니다 선택적 인 설치 구성 요소로, Linux는 ``--torch-backend auto``를 받아 들일 수 있으며, 표준 macOS PyTorch 바퀴는 Apple MPS 가속을 유지합니다.

설치 도움말, 진행 및 오류는 10 개의 spaCR 언어로 운영 체제 언어를 따르십시오 : 영어, 스웨덴어, 독일어, 스페인어, 간단한 중국어, 포르투갈어, 힌두교, 한국어, 아이슬란드어 및 프랑스어.

Linux에서 다운로드된 설치기를 열기 전에 실행할 수 있도록 하십시오.

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

macOS에서 다운로드된 ``.pkg``을 열어보세요.Gatekeeper가 현재 베타 설치기를 차단하면 노트북이 아닌 경우 **시스템 설정 → 개인 정보 보호 및 보안**을 엽니 다. **Open Anyway**를 선택하여 spaCR을 다시 실행합니다.

The installer validates spaCR, Qt, PyTorch and dependency consistency before replacing an older installation, so an interrupted update leaves the previous working environment in place. A diagnostic log is kept as ``install.log`` inside the private spaCR installation directory.

PyPI에서 데스크톱 애플리케이션 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

그래픽 인터페이스 없이 또는 서버에 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

작업 흐름의 필요에 대한 추가만 설치하십시오 :

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

어떤 엑스트라 해결은 Python 버전에 달려 있습니다. Python 3.13에서, 엑스트라 ``spacr[all]`` 및 TorchCAM의 NumPy 제한은 ``attribution`` 추가를 제한합니다. 핵심 패키지와 Qt 응용 프로그램은 영향을받지 않습니다. Python 3.14에서, btrack는 추가를 통해 사용할 수 있습니다. pylibCZIrw CZI 변환기는 선택적이고 테스트되지 않습니다. czifile 기반 CZI 읽기 여전히 사용할 수 있습니다.

유산 Tk 인터페이스는 여전히 ``spacr-legacy``로 설치되지만 더 이상 개발되지 않습니다.


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

문제 해결을 할 때 ``SPACR_LOG_LEVEL=DEBUG``를 설정합니다. 회전 기록은 ``~/.spacr/logs/spacr.log``로 작성됩니다.


기능
--------

대부분의 스크리닝에서 사용하는 6개 모듈
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**마스크** 세포, 핵, 병원 및 유기체의 Cellpose, 2D 이미지 및 볼륨 또는 시간 시리즈 데이터. 모델 목록은 설치된 Cellpose 대신 하드 코드, 그리고 개체 직경은 실행이 시작되기 전에 이미지에서 추정됩니다. 마스크는 레이어 시청자에서 수동으로 수정 될 수 있습니다, 또는 napari 및 뒤로 보낼 수 있습니다.

**메이저**는 프로젝트 데이터베이스에 개체별 형태, 강도, 구조 및 위치 기능을 작성합니다. 1.5.0.0에서 새로 : 조명 수정은 플레이트 자체에서 평면 필드를 추정하고 어떤 강도 기능이 취해지기 전에 분할되며, 플레이트 열지도가 옆 효과로 표시하는 균형을 제거합니다. QC 배너는 측정이 실행되기 전에 마스크가 어떻게 보이는지 평면 언어로 표시합니다.

**Annotate** 키보드가 주도된 네트워크에 작물을 표시하고 SQLite로 라벨을 작성합니다.이제는 활성 학습 루프를 닫습니다 : 화면을 떠나지 않고 라벨을 표시한 모델을 뒤집고, 불확실성으로 라벨을 다시 평가하고, 학습 곡선을 지켜보고, 더 많은 라벨이 모델을 변경하는 것을 멈추게 할 때 중단 판결을 얻습니다.

**Classify** 트렌즈 PyTorch CNNs 및 변압기에 기록 된 크롭, 그리고 측정 테이블에 고전 또는 강화 된 모델. 순수성은 이제 각 시대에 보존되지 않고, 각 체크 포인트는 그것의 데이터 세트, 클래스 균형, 분열 규칙 및 유지-out 메트릭을 기록하는 모델 카드를 얻습니다. 평가 화면에서, 혼란-마트릭스 세포는 질문입니다 : 그것을 클릭하여 그 작물을 열고, 확실히 잘못된 예측은 불확실한 것과 분리되어 있습니다.

**지도 바코드** FASTQ에서 라인, 열 및 gRNA 바코드를 분해하고, 냄비에 가이드 정체성을 부여하고, 그들을 이미지 세포에 연결합니다. 바코드 QC 보고서는 냄비, 충돌 속도 및 맵화되지 않은 부분에 따라 읽고, 당신이 기대하는 대신 견고한 한계에 gRNA의 수를 주위에 굴복한다.

**Regression** 가이드, 유전자, 상태 및 제어 효과를 17 모델 가족을 사용하여 추정, 혼합 모델, 물류 및 probit, 양, 베타, GLMs와 quasi-binomial 변형, lasso, ridge, elastique net, hinge 및 horeshoe.

1.5.0.0의 새로운 기능
~~~~~~~~~~~~~~~

스크린이 존재하기 전에, 전원 / 디자인 모듈은 그것이 필요로하는 셀의 수와 수의 덩어리를 응답, 순서 오류와 너무 얇게 묘사 된 덩어리에서 온 덩어리와 함께 가격. 실험 디자이너는 플레이트를 제거하고, 그것의 컨트롤과 그것의 복제 및 파이프 라인에 대한 배치를 수출합니다. 그 후, QC 다이블은 분할, 플레이트, 애노터 합의 및 유출을 하나의 판결에 검사하고, ComBat는 ``center`` 및 ``zscore`` 덩어리 수정에 사용할 수 있습니다.

결과는 수출 및 다시 수입 대신 탐색됩니다. 그래프 건축가는 x, y, 색상, 크기 및 측면에 열을 끌어 내는 테이블을 쌓습니다. 히스토그램이나 스캐터에 끌어 내는 문은 필터가됩니다. 특징 탐험가는 그들이 클래스를 얼마나 잘 분리하는지에 의해 특징을 정렬합니다. 작은 다이얼, 복용량 응답 칩, 제어 차트 및 강력한 외부 탐지 동일한 좌석 엔진을 사용합니다. 하나의 시선에서 개체를 선택하면 그들 모두에서 그들을 선택하고, 선택을 열면 그 개체가 나온 식물에 도달합니다. 레이어 시청자는 orthogonal 시선, 동기화 된 비교 그리드와 형태와 함께 이미지를, 라벨, 포인트와 모양을 쌓습니다.

Runs are now identifiable. Each carries one run id, one seed and an ``on_error`` policy; Mask, Measure, Classify and the AnnData export register what they wrote in an artifact registry, so an output file leads back to the settings that produced it. A module opens on what the previous step actually wrote, the pipeline graph marks which outputs are stale, run comparison diffs the settings, object counts and hit lists of two runs, and every GUI run emits the equivalent Python script. Measurements export to ``.h5ad`` for scanpy; OME-Zarr and OMERO are available through the Python API. The methods-and-results exporter drafts those two manuscript sections from a structured digest of the run: the model writes the prose, but every number comes from the digest, and a draft containing a number the digest does not contain is rejected. 설치에 뭔가 잘못되었을 때 ``spacr-doctor`` 보고서 spaCR 실제로 실행되고 있는지, GPU가 사용 가능하다는지, Cellpose가 API spaCR 통화와 일치하는지, 프로젝트 데이터베이스 및 설정이 소리인지, 각 라인에 복사 가능한 수정이 있는지 여부.

다국어 데스크톱 인터페이스
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → 선호 → 언어**은 재시작 없이 실행 응용 프로그램을 영어, 스웨덴어, 독일어, 스페인어, 맨더리어 중국어, 포르투갈어, 힌두교, 한국어, 아이슬란드어 또는 프랑스어로 재시작합니다.

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. 노동자 출력, 로그, 트랙바이크, 경로, 데이터베이스 값, 기록, AI 응답, 측정 및 저장 결과는 결코 번역되지 않습니다, 그래서 과학 출력은 캐논 영어로 남아 있습니다. 도구 팁을 설정하는 것은 아직 영어로 언어로 남아있는 대신 혼합 언어 설명이되지 않습니다. `위치 가이드 <https://einarolafsson.github.io/spacr/localization.html>`_ 문서 행동, 환경이 과장하고, 그것과 함께 번역되는 `컨텍스트 지원 <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_.

애니메이션 설정 가이드
~~~~~~~~~~~~~~~~~~~~~~~~~

94 짧은 애니메이션은 143 시각 설정이 이미지에 무엇을 하는지 설명합니다. 설정을 옮기고 ** 애니메이션**를 클릭하여 텍스트 옆에 평면을 재생합니다. 다시 클릭하여 폴더를 꺼내십시오. 애니메이션은 요청할 때까지 꺼져 있으며, 선호도에서 끄는 수 있습니다. `갤러리 <https://einarolafsson.github.io/spacr/setting_animations.html>`_는 모두 표시되며, 설정이 각각에 속하는 `애니메이션 레지스트리 설정 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ 레코드.

모듈 참조
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Feature
     - State
     - Description
   * - **Desktop experience**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


데이터
----

참조 데이터세트
~~~~~~~~~~~~~~~~~~

- `전체 미생물 데이터 세트: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `테스트 데이터 세트: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `추적 데이터: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `전원 분석 : spaCRPower <https://github.com/maomlab/spaCRPower>`_


기여 및 지원
------------------------

오류 보고서 및 집중된 기능 요청은 `GitHub 문제 <https://github.com/EinarOlafsson/spacr/issues>`_를 통해 환영합니다. 실패를 보고할 때 spaCR 버전, 운영 체제, Python 버전, 모듈 설정 및 관련 로그 excerpt를 포함합니다. ``spacr-doctor``는 대부분을 귀하를 위해 수집합니다.

라이선스
~~~~~~~~~

현재 개발 지점은 `PolyForm 비상업적 라이센스 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ 아래에서 출처 이용 가능합니다. 상업용 사용은 저작권 소유자로부터 별도의 라이센스를 필요로합니다. spaCR 1.4.9.9을 통해 발행된 버전은 MIT 라이센스에 따라 계속 이용 가능합니다.

튜토리얼
~~~~~~~~~

`상호 작용 spaCR 도서관 <https://einarolafsson.github.io/spacr/tutorials/>`_에는 8개의 언어로 설명된 설치 및 각 응용 프로그램 작업 흐름의 흔적이 담겨 있습니다.

spaCR 인용
~~~~~~~~~~~~

spaCR이 연구에 기여한다면 다음을 인용하십시오 :

Olafsson EB, *et al.* 합성 이미지 기반 CRISPR 스크린은 EAF1을 *T. gondii* ESCRT 하위 변형의 모듈로 식별합니다.

`BioRxiv 프리프린트 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `소프트웨어 아카이브 <https://doi.org/10.5281/zenodo.21343317>`_
