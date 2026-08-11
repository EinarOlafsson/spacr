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

Idiomas: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`Informações sobre os modelos de tradução <../TRANSLATION_MODELS.md>`_>

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.

Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.

Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.


Visão geral do fluxo de trabalho
--------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Imagens de microscopia (TIFF, OME-TIFF, LIF, CZI, ND2) e leituras de sequenciamento (FASTQ) entram em fluxos complementares de análise de imagens e mapeamento de códigos de barras. Em seguida, tabelas de objetos, recortes, anotações, previsões, identidades de guia, resultados de QC e resumos por poço são analisados em conjunto.


Início rápido
-------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR suportes Python **3.9 até 3.14** (excepto Python 3.14.1, que a visão da tocha exclui). Python 3.12 tem a mais ampla escolha de pacotes científicos opcionais. Linux é recomendado para CUDA fluxos de trabalho; macOS e Windows também são apoiados.


Detalhes da instalação
----------------------

|Release| |PyPI| |CondaRecipe|

**>(beta) Instaladores de desktop leves:**>

.. spacr-installer-links-begin

* `Windows 10/11: download SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_>
* `macOS 11+ (Intel e Apple silicone): baixar SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_>
* `64-bit Linux: baixar SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_>

.. spacr-installer-links-end

Instaladores leves — não exigem conda nem uma instalação existente do Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O instalador baixa um tempo de execução privado Python 3,12, Qt, PyTorch, spaCR e as dependências científicas durante a instalação, portanto, nem o conta nem um Python existente são necessários. A compilação portátil CPU é o padrão, o que impede que a instalação puxe vários gigabytes de bibliotecas CUDA sem aviso prévio. Windows oferece aceleração NVIDIA como um componente de instalação opcional, Linux aceita ``--torch-backend auto``, e a roda padrão macOS PyTorch mantém a aceleração MPS da Apple.

Ajuda do instalador, progresso e erros seguem a linguagem do sistema operacional em todos os dez idiomas spaCR: inglês, sueco, alemão, espanhol, chinês simplificado, português, hindi, coreano, islandês e francês.

Em Linux, faça o executável do instalador baixado antes de abri-lo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Ligado macOS, abra o download ``.pkg``. Se o Gatekeeper bloquear o instalador beta atual porque não está notarizado, abra **Configurações do sistema  Privacidade e segurança**, escolha **Abra de qualquer maneira** para spaCR, em seguida, execute o pacote novamente.

O instalador valida a consistência spaCR, Qt, PyTorch e dependência antes de substituir uma instalação mais antiga, então uma atualização interrompida deixa o ambiente de trabalho anterior no lugar. Um log de diagnóstico é mantido como ``install.log``> dentro do diretório de instalação spaCR privado.

Aplicativo para desktop pelo PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Instalação sem interface gráfica ou em servidor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Ramificação de desenvolvimento mais recente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Ambientes conda
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Recursos opcionais
~~~~~~~~~~~~~~~~~~~~~

Instale apenas os extras que seu fluxo de trabalho precisa:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Quais extras resolvem depende da versão Python. Em Python 3.13, limites ultrack ``spacr[all]``> e a restrição NumPy da TorchCAM limitam o ``attribution``> extra; o pacote principal e o aplicativo Qt não são afetados. Em Python 3.14, a btrack está disponível através do seu extra. O conversor CZI pylibCZrw é opcional e não testado;

A interface Tk legada ainda está instalada como ``spacr-legacy``>, mas não está mais desenvolvida.


Comandos de linha de comando
----------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Defina ``SPACR_LOG_LEVEL=DEBUG``> ao solucionar problemas. Os logs rotativos são gravados em ``~/.spacr/logs/spacr.log``>.


Recursos
--------

Os seis módulos mais usados nas triagens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Máscara** segmentos células, núcleos, patógenos e organelas com Cellpose, em imagens 2D e em dados volumétricos ou de séries temporais. A lista de modelos é lida a partir do Cellpose em vez de codificado, e um diâmetro do objeto é estimado a partir das imagens antes da execução começa. Máscaras podem ser corrigidas à mão no visualizador de camada, ou enviado para napari e volta.

**Medida** escreve características de morfologia, intensidade, textura e localização por objeto para o banco de dados do projeto, juntamente com as recortes. Novo em 1.5.0.0: correção de iluminação estima o campo plano da própria placa e divide-o antes de qualquer característica de intensidade ser tomada, o que remove o viés de boa posição que os heatmaps de placa mostram como efeitos de borda. QC banner indica em linguagem simples como as máscaras se parecem antes de Measure é executado; informa, ele não bloqueia. Um polígono desenhado restringe a medição para uma região de interesse.

**Anotar** mostra recortes em uma grade orientada por teclado e escreve rótulos diretamente para SQLite. Agora fecha o ciclo de aprendizagem ativa: retreinar um modelo no que você rotulou sem sair da triagem, re-classificar a fila pela incerteza, assistir a curva de aprendizagem, e obter um veredicto de parada quando rótulos adicionais param de mudar o modelo. Cobertura é relatada por classe, por poço e por prato, e cada rodada é gravada.

**Classificar** comboios PyTorch CNNs e transformadores em recortes anotadas e modelos clássicos ou aprimorados em tabelas de medição. A precisão por classe agora é mantida em todas as épocas em vez de ser descartada, e cada ponto de verificação recebe um cartão de modelo registrando seu conjunto de dados, equilíbrio de classes, regra dividida e métricas suspensas. Na triagem de avaliação, uma célula de matriz de confusão é uma consulta: clique nela para abrir essas recortes, com previsões confiantemente erradas listadas além das incertas.

**Mapa códigos de barras** decodifica linha, coluna e gRNA códigos de barras do FASTQ lê, atribui identidades de guia a poços e os une a células imageadas. QC relatórios lê por poço, taxa de colisão e fração não mapeada, varrendo o número de gRNAs por poço que você diz que espera em vez de um limite fixo.

**Regressão** estima-se efeitos de guia, gene, condição e controle utilizando 17 famílias de modelos, incluindo modelos mistos, logísticos e probit, quantis, beta, GLMs com variância quase binomial, lasso, ridge, elásticos, dobradiça e ferradura. O resultado é uma lista de acertos classificada e anotada, em vez de um despejo de coeficiente.

Novidades na 1.5.0.0
~~~~~~~~~~~~~~~~~~~~

Antes de existir uma triagem, o módulo Power / Design responde quantas células e quantos poços ele precisa, com preço de erro de sequenciamento e com o dropout que vem de poços que foram fotografados muito finamente. Um designer de experimentos expõe a placa, seus controles e suas réplicas e exporta o layout para o gasoduto. Depois, um painel QC coleta as verificações de segmentação, placa, acordo de anotador e vazamento em um veredicto, e o ComBat está disponível ao lado de ``center`` e ``zscore`` para correção em lote.

Os resultados são explorados em vez de exportados e re-importados. Um construtor de gráficos traça uma tabela arrastando colunas para x, y, cor, tamanho e faceta. Portões desenhados em um histograma ou uma dispersão tornam-se filtros. Um explorador de recursos classifica os recursos por quão poço eles separam as classes. Pequenos múltiplos, ajustes de comparação de dose-resposta, gráficos de controle e detecção robusta de outliers usam o mesmo motor de eixo. Selecionar objetos em uma única vista os seleciona todos eles.

As execuções são agora identificáveis. Cada um carrega um run id, uma semente e uma política ``on_error``; Máscara, Medição, Classificar e o registro de exportação AnnData que eles escreveram em um registro de artefato, então um arquivo de saída leva de volta às configurações que o produziram. Um módulo abre o que a etapa anterior realmente escreveu, o gráfico de pipeline marca quais saídas estão obsoletas, a comparação de execução difere as configurações, a contagem de objetos e as listas de hits de duas execuções, e cada execução de GUI emite o script Python equivalente. As medições exportam para ``.h5ad`` para scanpy; OME-Zarr e OMERO estão disponíveis através do Python API. O exportador de métodos e resultados elabora essas duas seções de manuscrito a partir de um resumo estruturado da execução: o modelo escreve a prosa, mas cada número vem do resumo, e um rascunho contendo um número que o resumo não contém é rejeitado. Quando algo está errado com a instalação, ``spacr-doctor``> relata que spaCR está realmente em execução, se o GPU é utilizável, se Cellpose corresponde às chamadas API spaCR e se o banco de dados e as configurações do projeto são sólidas, com uma correção copyable em cada linha que não é um passe.

Interface multilíngue para desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR Preferências  Idioma** retraduz o aplicativo em execução para o inglês, sueco, alemão, espanhol, mandarim, português, hindi, coreano, islandês ou francês sem reiniciar. A escolha persiste e as triagens abertas mais tarde herdam.

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `guia de localização <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `Ajuda contextual <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

Orientação de cenário animado
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 animações curtas explicam o que 143 configurações visuais fazem em uma imagem. Passe o mouse em uma configuração e clique em **Animação** em sua dica de ferramenta para reproduzir o quadrado ao lado do texto; clique nele novamente para dobrá-lo. As animações estão desativadas até serem solicitadas e podem ser desativadas nas Preferências. O `galeria <https://einarolafsson.github.io/spacr/setting_animations.html>`_ mostra todos eles, e o `Configuração do registo de animação <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ registros aos quais cada um pertence.

Referência dos módulos
~~~~~~~~~~~~~~~~~~~~~~

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


Dados
-----

Conjuntos de dados de referência
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Conjunto de dados de microscopia completa: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_>
- `Testando o conjunto de dados: Abraçando a face toxo_mito> <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_>
- `Dados de sequenciamento: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_>
- `Análise de potência: spaCRPower <https://github.com/maomlab/spaCRPower>`_>


Contribuições e suporte
------------------------

Relatórios de bugs e solicitações de recursos focados são bem-vindos através de `GitHub Problemas <https://github.com/EinarOlafsson/spacr/issues>`_>. Ao relatar uma falha, inclua a versão  spaCR, o sistema operacional, a versão "Python", as configurações do módulo e o trecho de log relevante.  ``spacr-doctor``> coleta a maior parte disso para você.

Licença
~~~~~~~~~

O ramo de desenvolvimento atual está disponível na fonte sob o `PolyForm Licença não comercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_>. O uso comercial requer uma licença separada do detentor dos direitos autorais. As versões liberadas através do  spaCR 1.4.9.9 permanecem disponíveis sob a Licença MIT que acompanhou esses lançamentos.

Tutoriais
~~~~~~~~~

O `biblioteca tutorial interativa spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_> contém passos narrados e legendados de instalação e de cada fluxo de trabalho de aplicativo, em oito idiomas.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se spaCR contribuir para a sua pesquisa, cite:

Olafsson EB, *et al.* Uma triagem pooled image-based CRISPR identifica o EAF1 como um modulador *T. gondii* da subversão ESCRT.

`bioRxiv pré-impressão <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_>   `arquivo de software <https://doi.org/10.5281/zenodo.21343317>`_>
