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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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

`Informações sobre os modelos de tradução <../TRANSLATION_MODELS.md>`_

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.

Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.

Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.


Visão geral do fluxo de trabalho
--------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: Fluxo de trabalho e organização das saídas do spaCR
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

O spaCR oferece suporte ao Python **3.9 até 3.14** (exceto ao Python 3.14.1, que não é aceito pelo torchvision). O Python 3.12 oferece a maior seleção de pacotes científicos opcionais. Recomenda-se Linux para fluxos de trabalho CUDA; macOS e Windows também são compatíveis.


Detalhes da instalação
----------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Instaladores leves para desktop:**

.. spacr-installer-links-begin

|InstallerWindows| |InstallerMacOS| |InstallerLinux|

.. |InstallerWindows| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: baixar o SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel e Apple Silicon): baixar o SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux de 64 bits: baixar o SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

Instaladores leves — não exigem conda nem uma instalação existente do Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Durante a instalação, o instalador baixa um ambiente privado do Python 3.12, Qt, PyTorch, spaCR e as dependências científicas; portanto, não é necessário ter conda nem Python previamente instalado. A compilação portátil para CPU é o padrão, evitando o download sem aviso de vários gigabytes de bibliotecas CUDA. No Windows, a aceleração NVIDIA é um componente opcional; o Linux aceita ``--torch-backend auto``; e o wheel padrão do PyTorch para macOS mantém a aceleração Apple MPS.

A ajuda, o progresso e os erros do instalador acompanham o idioma do sistema operacional nos dez idiomas do spaCR: inglês, sueco, alemão, espanhol, chinês simplificado, português, hindi, coreano, islandês e francês. Localidades não compatíveis usam o inglês como idioma alternativo.

No Linux, torne o instalador baixado executável antes de abri-lo:

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

No macOS, abra o arquivo ``.pkg`` baixado. Se o Gatekeeper bloquear o instalador beta atual por ele não estar notarizado, abra **Ajustes do Sistema → Privacidade e Segurança**, escolha **Abrir Mesmo Assim** para o spaCR e execute o pacote novamente.

Antes de substituir uma instalação antiga, o instalador valida a consistência do spaCR, do Qt, do PyTorch e das dependências. Assim, se uma atualização for interrompida, o ambiente anterior continua funcionando. Um log de diagnóstico é mantido como ``install.log`` dentro do diretório privado de instalação do spaCR.

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

Os extras que podem ser instalados dependem da versão do Python. No Python 3.13, o ultrack limita ``spacr[all]`` e a restrição de NumPy do TorchCAM limita o extra ``attribution``; o pacote principal e o aplicativo Qt não são afetados. No Python 3.14, o btrack está disponível por meio de seu extra. O conversor CZI pylibCZIrw é opcional e não foi testado; a leitura de CZI baseada em czifile continua disponível.

A interface Tk legada ainda é instalada como ``spacr-legacy``, mas não recebe mais desenvolvimento.


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

Defina ``SPACR_LOG_LEVEL=DEBUG`` ao solucionar problemas. Os logs rotativos são gravados em ``~/.spacr/logs/spacr.log``.


Recursos
--------

Os seis módulos mais usados nas triagens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmenta células, núcleos, patógenos e organelas com o Cellpose em imagens 2D e em dados volumétricos ou de séries temporais. A lista de modelos é obtida do Cellpose instalado, em vez de ser fixada no código, e o diâmetro dos objetos é estimado a partir das imagens antes do início da execução. As máscaras podem ser corrigidas manualmente no visualizador de camadas ou enviadas ao napari e trazidas de volta.

**Measure** grava no banco de dados do projeto, junto com os recortes, características de morfologia, intensidade, textura e colocalização de cada objeto. Novo na versão 1.5.0.0: a correção de iluminação estima o campo plano na própria placa e o divide antes de medir qualquer característica de intensidade, removendo o viés de posição dos poços que aparece como efeito de borda nos mapas de calor. Antes da execução, um aviso de QC de segmentação descreve as máscaras em linguagem simples; ele informa, mas não bloqueia. Um polígono desenhado restringe a medição a uma região de interesse.

**Annotate** mostra recortes em uma grade controlada pelo teclado e grava os rótulos diretamente no SQLite. O ciclo de aprendizado ativo agora ocorre inteiro na mesma tela: treine novamente o modelo com o que já foi rotulado, reordene a fila pela incerteza, acompanhe a curva de aprendizado e receba um aviso para parar quando novos rótulos deixarem de alterar o modelo. A cobertura é informada por classe, poço e placa, e cada rodada é registrada.

**Classify** treina CNNs e transformers do PyTorch em recortes anotados, além de modelos clássicos ou de boosting em tabelas de medição. A acurácia de cada classe agora é preservada a cada época, e cada checkpoint recebe um cartão do modelo que registra conjunto de dados, equilíbrio de classes, regra de divisão e métricas de validação. Na tela de avaliação, uma célula da matriz de confusão funciona como consulta: clique nela para abrir os recortes correspondentes, separando previsões erradas de alta confiança das previsões incertas.

**Map Barcodes** decodifica códigos de barras de linha, coluna e gRNA das leituras FASTQ, atribui identidades de guia aos poços e as associa às células fotografadas. O QC de códigos de barras informa leituras por poço, taxa de colisão e fração não mapeada, examinando valores em torno do número esperado de gRNAs por poço informado pelo usuário, em vez de usar um limite fixo.

**Regression** estima efeitos de guia, gene, condição e controle usando 17 famílias de modelos, incluindo modelos mistos, logistic, probit, quantile, beta, GLMs com variância quase binomial, lasso, ridge, elastic net, hinge e horseshoe. O resultado é uma lista de hits classificada e anotada, não apenas um despejo de coeficientes.

Novidades na 1.5.0.0
~~~~~~~~~~~~~~~~~~~~

Antes mesmo de existir uma triagem, o módulo Power / Design calcula quantas células e quantos poços serão necessários, levando em conta erros de sequenciamento e a perda de dados causada por poços com poucas células fotografadas. O planejador de experimentos organiza a placa, os controles e as réplicas e exporta o layout para o pipeline. Depois, um painel de QC reúne as verificações de segmentação, placa, concordância entre anotadores e vazamento de dados em um único veredito; para correção de lote, ComBat fica disponível ao lado de ``center`` e ``zscore``.

Os resultados podem ser explorados diretamente, sem exportação e reimportação. O Graph Builder cria gráficos de uma tabela ao arrastar colunas para x, y, cor, tamanho e faceta. Gates desenhados em um histograma ou gráfico de dispersão tornam-se filtros. O Feature Explorer ordena as características conforme sua capacidade de separar as classes. Pequenos múltiplos, ajustes de dose–resposta, gráficos de controle e detecção robusta de outliers usam o mesmo mecanismo de eixos. Selecionar objetos em uma visualização os seleciona em todas; abrir a seleção mostra os recortes de origem. O Layer Viewer sobrepõe imagens, rótulos, pontos e formas, com vistas ortogonais, uma grade de comparação sincronizada e uma árvore de linhagem de célula para núcleo e patógeno.

Agora cada execução pode ser identificada e rastreada. Ela recebe um ID, uma semente e uma política ``on_error``; Mask, Measure, Classify e a exportação AnnData registram os arquivos gerados em um registro de artefatos, permitindo voltar de um arquivo de saída às configurações que o produziram. Cada módulo abre o que a etapa anterior realmente gravou; o gráfico do pipeline marca saídas obsoletas; a comparação de execuções mostra diferenças nas configurações, contagens de objetos e listas de hits; e toda execução na GUI gera o script Python equivalente. As medições são exportadas para ``.h5ad`` para uso no scanpy; OME-Zarr e OMERO estão disponíveis pela API Python. O exportador de métodos e resultados redige essas duas seções do manuscrito a partir de um resumo estruturado da execução: o modelo escreve a prosa, mas todos os números vêm do resumo, e um rascunho com um número ausente do resumo é rejeitado. Quando há um problema na instalação, ``spacr-doctor`` informa qual spaCR está em execução, se a GPU pode ser usada, se o Cellpose corresponde à API chamada pelo spaCR e se o banco de dados e as configurações do projeto são válidos, além de oferecer uma correção copiável para cada verificação que falhar.

Interface multilíngue para desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Preferências → Idioma** muda o aplicativo em execução para inglês, sueco, alemão, espanhol, chinês simplificado, português, hindi, coreano, islandês ou francês sem reiniciar. A escolha é preservada e também se aplica às telas abertas depois.

A navegação, as Preferências, os controles de AI e LIVE, as descrições dos módulos e os avisos de console produzidos pelo spaCR seguem o idioma selecionado. A saída dos processos, os logs, os rastreamentos de erro, os caminhos, os valores do banco de dados, as anotações, as respostas de AI, as medições e os resultados salvos nunca são traduzidos; assim, a saída científica permanece no inglês canônico. As dicas de configurações que ainda não foram revisadas em um idioma permanecem em inglês, evitando explicações em idiomas misturados. O `guia de localização <https://einarolafsson.github.io/spacr/localization.html>`_ documenta esse comportamento, a substituição por variável de ambiente e a `ajuda contextual <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ traduzida junto com a interface.

Guia animado de configurações
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 animações curtas mostram como 143 configurações visuais afetam uma imagem. Passe o cursor sobre uma configuração e clique em **Animação** na dica para reproduzir a prévia quadrada ao lado do texto; clique novamente para recolhê-la. As animações só são executadas quando solicitadas e podem ser desativadas nas Preferências. A `galeria <https://einarolafsson.github.io/spacr/setting_animations.html>`_ reúne todas elas, e o `registro de animações de configurações <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ informa a qual configuração cada animação pertence.

Referência dos módulos
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Módulo
     - Recurso
     - Estado
     - Descrição
   * - **Experiência no desktop**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Estável
     - Retraduz imediatamente as telas abertas ou criadas sob demanda entre os dez idiomas incluídos.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Estável
     - Localiza os resumos dos módulos e a interface de ajuda das configurações sem alterar as URLs da API.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Estável
     - Localiza os controles de AI e LIVE sem modificar o conteúdo do usuário ou do modelo.
   * - |api-animations|_
     - |doc-animations|_
     - Estável
     - Reproduz, a partir das dicas, 94 animações incluídas para 143 configurações visuais.
   * - |api-selection|_
     - |api-linked-views|_
     - Alfa
     - Compartilha uma seleção de objetos entre as vistas de tabela, placa, embedding, dispersão e gráfico.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alfa
     - Verifica GPU, API do Cellpose, banco de dados e configurações, com uma solução para cada teste que falhar.
   * - **Análise de imagens**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Estável
     - Segmenta células, núcleos, patógenos e organelas em imagens 2D.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segmenta imagens volumétricas e séries temporais 4D.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alfa
     - Estima o campo plano a partir da placa e o corrige antes da medição de intensidade.
   * - |api-measure|_
     - |api-measure-2d|_
     - Estável
     - Mede morfologia, intensidade, textura e colocalização e salva os recortes.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alfa
     - Descreve a qualidade da segmentação antes da execução de Measure, sem bloqueá-la.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Rastreia objetos com IoU, Trackpy, btrack, Trackastra ou ultrack e quantifica a motilidade.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alfa
     - Sobrepõe camadas de imagem, rótulo, ponto e forma, com vistas ortogonais e uma grade de comparação.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alfa
     - Envia uma máscara ao napari para correção, recupera-a e registra cada edição.
   * - **AI e fenotipagem**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Estável
     - Revisa recortes em uma grade controlada pelo teclado e salva as anotações no SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alfa
     - Retreina no Annotate, reordena por incerteza e indica quando a rotulagem pode terminar.
   * - |api-classify|_
     - |api-classification|_
     - Estável
     - Treina e aplica modelos CNN e transformer do PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alfa
     - Registra, junto a cada checkpoint, o conjunto de dados, o equilíbrio de classes, a regra de divisão e as métricas de validação.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alfa
     - Abre os recortes associados a uma célula da matriz de confusão e separa erros confiantes de casos incertos.
   * - |api-ml|_
     - |api-ml-models|_
     - Estável
     - Treina modelos clássicos e de boosting interpretáveis em tabelas de medições.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explica as previsões com Captum, SmoothGrad e TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explora embeddings de imagens de forma interativa e propaga rótulos de clusters.
   * - **Sequenciamento e análise de triagens**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Estável
     - Mapeia os códigos de barras de linha, coluna e gRNA das leituras FASTQ e atribui guias às células imageadas.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alfa
     - Informa leituras por poço, taxa de colisão e fração não mapeada conforme os gRNAs esperados por poço.
   * - |api-regression|_
     - |api-regression-models|_
     - Estável
     - Estima os efeitos de guia, gene, condição e controle com 17 famílias de modelos.
   * - |api-power|_
     - |api-power-design|_
     - Alfa
     - Calcula quantas células e quantos poços uma triagem exige, considerando erro de sequenciamento e perda de poços.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alfa
     - Cria um gráfico arrastando colunas para x, y, cor, tamanho e faceta.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alfa
     - Registra o ID da execução, a semente e as configurações que geraram as saídas de Mask, Measure, Classify e exportação.

.. |api-qt-app| replace:: **Aplicativo Qt**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Localização em dez idiomas**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Ajuda contextual localizada**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **Console assistido por AI**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Registro de animações de configurações**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Animações de configurações visuais**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Seleção**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Seleção vinculada**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Diagnóstico da instalação**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **Geração de máscaras 2D**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **Geração de máscaras 3D e 4D**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Iluminação**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Correção de campo plano**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Medições de objetos**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **QC de segmentação**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Veredito antes da execução**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Rastreamento de objetos**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Camadas**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Visualizador de camadas**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **Ponte com o napari**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Curadoria de máscaras**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Anotação manual**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Aprendizado ativo**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retreinar e reordenar**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Classificação de imagens**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Cartões de modelos**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Mapas de ativação**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Exploração da matriz de confusão**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Aprendizado de máquina**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Classificação de medições**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Embedding interativo**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequenciamento**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Mapeamento de códigos de barras**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **QC de códigos de barras**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Relatório de poços e colisões**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Estimativa dos efeitos da triagem**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Poder estatístico e planejamento**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artefatos**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Proveniência da execução**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Dados
-----

Conjuntos de dados de referência
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Conjunto de dados completo de microscopia: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Conjunto de dados de teste: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Dados de sequenciamento: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Análise de poder estatístico: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contribuições e suporte
------------------------

Relatos de bugs e solicitações objetivas de recursos são bem-vindos no `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho de log relevante. O ``spacr-doctor`` coleta automaticamente a maior parte dessas informações.

Licença
~~~~~~~~~

O código-fonte do ramo de desenvolvimento atual está disponível sob a `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. O uso comercial exige uma licença separada do detentor dos direitos autorais. As versões lançadas até o spaCR 1.4.9.9 continuam disponíveis sob a Licença MIT incluída nesses lançamentos.

Tutoriais
~~~~~~~~~

A `biblioteca interativa de tutoriais do spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ oferece, em oito idiomas, orientações narradas e legendadas sobre a instalação e cada fluxo de trabalho do aplicativo.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se spaCR contribuir para a sua pesquisa, cite:

Olafsson EB, *et al.* Uma triagem CRISPR agrupada e baseada em imagens identifica EAF1 como modulador da subversão do ESCRT em *T. gondii*.

`Preprint no bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Arquivo do software <https://doi.org/10.5281/zenodo.21343317>`_
