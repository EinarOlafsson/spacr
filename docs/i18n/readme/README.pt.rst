|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentação
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriais interativos
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Versão no PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Suíte de testes
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Interface Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código-fonte no GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Problemas no GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licença PolyForm Noncommercial
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: DOI do Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores mais recentes
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: Receita do conda-forge

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

Idiomas: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.

Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.

Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.


Visão geral do fluxo de trabalho
--------------------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Abrir a API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Abrir a API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Abrir a API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Abrir a API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Abrir a API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Abrir a API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Data**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

|App_project_browser|

**Results & QC**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|\ |App_control_chart|

**Explore**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|\ |App_feature_dict|\ |App_trellis|\ |App_gate_editor|

|App_feature_explorer|\ |App_outliers|

**Assays**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Design**

|App_experiment_design|\ |App_power|\ |App_dose_response|

.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: Abrir a API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: Abrir a API de Format Converter
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: Abrir a API de Import Project
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: Abrir a API de External Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: Abrir a API de Plate Queue
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: Abrir a API de Batch Runner
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: Abrir a API de Distributed Jobs
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: Abrir a API de Database Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: Abrir a API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: Abrir a API de Data Manager
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_project_browser| image:: ../../../spacr/resources/icons/workflow/apps/project_browser.png
   :width: 19.9%
   :alt: Abrir a API de Project Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/project_browser/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: Abrir a API de Plate Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: Abrir a API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: Abrir a API de Training Runs
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: Abrir a API de Run History
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: Abrir a API de Report
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: Abrir a API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: Abrir a API de Investigate Hit
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_control_chart| image:: ../../../spacr/resources/icons/workflow/apps/control_chart.png
   :width: 19.9%
   :alt: Abrir a API de Control Charts
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/control_chart/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: Abrir a API de Pipeline Graph
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: Abrir a API de Prediction Profiler
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: Abrir a API de QC Dashboard
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: Abrir a API de Lineage
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: Abrir a API de Layer Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: Abrir a API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: Abrir a API de Tabulate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_feature_dict| image:: ../../../spacr/resources/icons/workflow/apps/feature_dict.png
   :width: 19.9%
   :alt: Abrir a API de Feature Dictionary
   :target: https://einarolafsson.github.io/spacr/api/spacr/feature_dict/index.html
   :align: middle
.. |App_trellis| image:: ../../../spacr/resources/icons/workflow/apps/trellis.png
   :width: 19.9%
   :alt: Abrir a API de Small Multiples
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/trellis/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 19.9%
   :alt: Abrir a API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_feature_explorer| image:: ../../../spacr/resources/icons/workflow/apps/feature_explorer.png
   :width: 19.9%
   :alt: Abrir a API de Feature Explorer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/feature_explorer/index.html
   :align: middle
.. |App_outliers| image:: ../../../spacr/resources/icons/workflow/apps/outliers.png
   :width: 19.9%
   :alt: Abrir a API de Outliers
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/outliers/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: Abrir a API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: Abrir a API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: Abrir a API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: Abrir a API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: Abrir a API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: Abrir a API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 19.9%
   :alt: Abrir a API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle

.. spacr-workflow-end

Selecione um módulo do fluxo de trabalho para abrir sua página da API. A grade contém todos os outros aplicativos, organizados nas mesmas categorias e na mesma ordem da tela inicial do spaCR.


Instalar o spaCR
----------------

Aplicativo para desktop
~~~~~~~~~~~~~~~~~~~~~~~

Os instaladores de desktop incluem um ambiente Python privado, portanto, não é necessária uma instalação  Python existente.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para macOS 11+ (Intel e Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para Linux de 64 bits
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores do spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Os três primeiros ícones baixam a versão atual. O ícone spaCR abre o arquivo completo do instalador. Os links do instaladores e os nomes dos arquivos versionados são atualizados pelo fluxo de trabalho da versão; os instaladores anteriores permanecem no mesmo arquivo de versão.

No Linux, torne o arquivo baixado executável e execute-o:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

No macOS, abra o arquivo ``.pkg``. A versão beta atual não é notarizada; se o Gatekeeper a bloquear, selecione **Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim**.

Veja as instruções `guia do instalador <../../source/installer_guide.rst>`_ para atualização, desinstalação, off-line e solução de problemas.

Instalação com Python
~~~~~~~~~~~~~~~~~~~~~

Python 3,12 tem a mais ampla escolha de pacotes científicos opcionais:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

O spaCR oferece suporte ao Python **3.9 a 3.14**, exceto ao Python 3.14.1, que é excluído pelo torchvision. Recomenda-se Linux para fluxos de trabalho com CUDA; macOS e Windows também são compatíveis.

Em um servidor, cluster ou executor de CI, omita o Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

As integrações opcionais são instaladas separadamente, por exemplo ``spacr[zarr]``,  ``spacr[omero]``,``spacr[napari]`` e ``spacr[czi,nd2,lif]``. Veja a tabela de compatibilidade  `guia de instalação <../../source/installer_guide.rst>`_ para os extras completos e ?Python-versão.

Comandos de linha de comando
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Os logs rotativos são gravados em  ``~/.spacr/logs/spacr.log``.


O que você pode fazer
---------------------

A maioria das triagens segue seis módulos:

- **Mask** segmenta células, núcleos, patógenos e organelas com Cellpose.
- **Measure** grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos.
- **Annotate** rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo.
- **Classify** treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados.
- **Map Barcodes** associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura.
- **Regression** estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.

O mesmo projeto também pode projetar placas, estimar a potência, corrigir efeitos de lote, inspecionar a qualidade da segmentação, explorar parcelas e recortes vinculadas, exportar AnnData, retomar o trabalho interrompido e registrar as configurações por trás de cada resultado.

Escolha a próxima página pelo que você quer fazer:

- `Tutoriais interativos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 guided workflows from installation por meio de hit investigation.
- `Python  API início rápido <../../source/python_api.rst>`_ — run and validate pipelines from scripts, notebooks or a cluster.
- `Guia de funcionalidades <../../source/features.rst>`_ capacidades, maturidade e integrações opcionais.
- `Referência API com curadoria <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry points by task, with the complete module reference one level deeper.
- `Guia de idiomas e tradução <../../source/localization.rst>`_ linguagens de interface, ajuda contextual e política de saída científica.

Idioma e tradução
~~~~~~~~~~~~~~~~~~~~~~

A interface oferece dez idiomas na navegação e nas preferências. Os controles AI e LIVE, as descrições dos módulos e a ajuda contextual revisada também são traduzidos. Altere o idioma em **spaCR → Preferências → Idioma** sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a `política de ajuda contextual <../../source/localization.rst#contextual-help>`_.

Guia animado de configurações
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As configurações com uma explicação visual oferecem um controle **Animation** na dica de ferramenta. Consulte a `galeria de animações de configurações <https://einarolafsson.github.io/spacr/setting_animations.html>`_ ou o `registro de animações de configurações <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Dados
-----

Conjuntos de dados de referência
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Abrir o conjunto de microscopia no BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Abrir o conjunto de teste no Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Abrir o conjunto de sequenciamento no NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Abrir o spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Abrir a pré-publicação no bioRxiv
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Contribuições e suporte
------------------------

Relatos de erros e solicitações de recursos bem definidos são bem-vindos no `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho relevante do log. O ``spacr-doctor`` coleta automaticamente a maior parte dessas informações.

Licença
~~~~~~~~~

O código-fonte da ramificação de desenvolvimento atual está disponível sob a `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. O uso comercial exige uma licença separada do titular dos direitos autorais. As versões publicadas até o spaCR 1.4.9.9 continuam disponíveis sob a licença MIT que acompanhava essas versões.

Tutoriais
~~~~~~~~~

A `biblioteca interativa de tutoriais do spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contém demonstrações narradas e legendadas da instalação e de cada fluxo de trabalho: 73 lições, com 50 vozes em oito idiomas.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se o spaCR contribuir para sua pesquisa, cite:

Olafsson EB, *et al.* Um pooled image-based  CRISPR screen identifica o EAF1 como um modulador *T. gondii* da subversão ESCRT.

`bioRxiv pré-impressão <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_  · ? `arquivo de software <https://doi.org/10.5281/zenodo.21343317>`_

Agradecimentos
~~~~~~~~~~~~~~~

O spaCR utiliza software científico aberto, incluindo NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch e Qt. Consulte a `atribuição dos modelos de tradução <../TRANSLATION_MODELS.md>`_ para ver os modelos usados na documentação multilíngue e nos catálogos da interface.
