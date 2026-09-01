|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
   :alt: Licença BSD 3-Clause
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI do Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores mais recentes
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versão no conda-forge

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada.

Para triagens CRISPR agrupadas e baseadas em imagens, o spaCR fornece o fluxo de trabalho desde a segmentação de imagens até a priorização de resultados. Em estudos de microscopia de alto conteúdo sem triagens baseadas em sequenciamento, os módulos de segmentação, medição, anotação e classificação podem ser usados de forma independente.

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

**Dados**

|App_foreign|\ |App_run_compare|\ |App_experiment_design|\ |App_power|\ |App_dose_response|\ |App_qc_dashboard|

**Tools**

|App_make_masks|\ |App_align|\ |App_umap|\ |App_gate_editor|\ |App_graph_builder|

**Ensaios**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 15.466%
   :alt: Abrir a API de Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 15.466%
   :alt: Abrir a API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 15.466%
   :alt: Abrir a API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 15.466%
   :alt: Abrir a API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 15.466%
   :alt: Abrir a API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 15.466%
   :alt: Abrir a API de QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 15.466%
   :alt: Abrir a API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 15.466%
   :alt: Abrir a API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 15.466%
   :alt: Abrir a API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 15.466%
   :alt: Abrir a API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 15.466%
   :alt: Abrir a API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 15.466%
   :alt: Abrir a API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 15.466%
   :alt: Abrir a API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 15.466%
   :alt: Abrir a API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 15.466%
   :alt: Abrir a API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
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

Instalação com conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~

O pacote oficial do conda-forge instala o spaCR e suas dependências de desktop no ambiente ativo:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Instalação pelo PyPI
~~~~~~~~~~~~~~~~~~~~

Para usar a versão publicada no PyPI, instale o spaCR com pip em um ambiente Conda. O Python 3.12 oferece a maior variedade de pacotes científicos opcionais:

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

Defina ``SPACR_LOG_LEVEL=DEBUG`` durante a solução de problemas. Os logs rotativos são gravados em ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` lista os módulos com pontos de entrada de linha de comando para execução sem interface gráfica. Os módulos de anotação, curadoria, comparação e exploração disponíveis apenas na interface gráfica são omitidos.


Fluxo de trabalho principal
---------------------------

O fluxo de trabalho principal compreende seis módulos:

- **Mask** segmenta células, núcleos, patógenos e organelas com Cellpose.
- **Measure** grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos.
- **Annotate** rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo.
- **Classify** treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados.
- **Map Barcodes** associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura.
- **Regression** estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.

O mesmo projeto também pode ser usado para criar placas, estimar o poder estatístico, corrigir efeitos de lote, inspecionar a qualidade da segmentação, explorar gráficos e recortes vinculados, exportar AnnData, retomar processamentos interrompidos e registrar as configurações associadas a cada resultado.

Módulos disponíveis nas telas hospedeiras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vinte módulos são integrados a telas hospedeiras relacionadas, em vez de aparecerem como blocos separados na tela inicial. Cada módulo é aberto pelo cabeçalho de sua tela hospedeira e usa o projeto ativo. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP e Make Masks disponibilizam esses módulos integrados. A ajuda e a documentação da API continuam disponíveis, e os módulos com pontos de entrada de pipeline ainda podem ser executados sem interface gráfica. O `guia de recursos <../../source/features.rst>`_ lista cada módulo integrado e sua tela hospedeira.

Make Masks
~~~~~~~~~~

Make Masks aparece em **Data** e permite a correção manual de máscaras de segmentação. O cabeçalho também dá acesso aos fluxos de trabalho do Cellpose. A área de edição tem nove ferramentas: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** e **Recrop**. Draw cria um rótulo preenchido a partir de um contorno fechado desenhado à mão livre. Divide separa um objeto mesclado ao longo de uma linha definida pelo usuário e preserva todos os demais rótulos de objetos.

Recrop extrai um campo com um único objeto de uma imagem preparada que contém vários objetos. Uma caixa delimitadora ao redor de um objeto grava as regiões correspondentes da imagem e da máscara como um novo campo, agenda esse campo após o campo atual e remove da fila de curadoria o campo original com vários objetos. Recrop altera o campo ativo, não os pixels dos rótulos.

A execução do Cellpose-SAM pelo Make Masks exibe dois resultados intermediários ao lado da máscara: o **mapa de probabilidade celular** e o **campo de fluxo**. A máscara é definida por um limiar no mapa de probabilidade, e as verificações de consistência de fluxo podem rejeitar objetos cujos fluxos derivados diferem do campo previsto. Examine esses resultados para distinguir baixa probabilidade celular de fluxo inconsistente ao avaliar uma máscara incorreta ou incompleta.

**Outros recursos**

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

Diagnóstico de desempenho
-------------------------

Gere um relatório de hardware e anexe-o a uma issue relacionada ao desempenho::

    python tools/spacr_hardware_report.py

O comando imprime um relatório e salva uma cópia em ``~/.spacr/reports``; a última linha identifica o caminho do arquivo salvo. ``--quick`` omite as avaliações de desempenho mais longas, e ``--out PATH`` seleciona outro local de saída.

O relatório não abre nenhum projeto nem lê dados do projeto. Ele registra o tempo de importação e das bibliotecas numéricas, a escala da tela, as preferências ativas, a construção da janela principal e das telas dos módulos e o desempenho das animações. O arquivo do relatório é a única saída criada.

O relatório também identifica a emulação da arquitetura do processador, como uma compilação x86_64 do Python no Apple Silicon, e a implementação de BLAS usada pelo NumPy. Ambos os fatores podem afetar substancialmente o desempenho.

Contribuições e suporte
------------------------

Envie relatos de erros e solicitações de recursos bem delimitadas pelo `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho relevante do log. O ``spacr-doctor`` coleta a maior parte dessas informações; inclua o relatório de hardware ao relatar problemas de desempenho.

Licença
~~~~~~~~~

O spaCR é de código aberto sob a `BSD 3-Clause License <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_, a mesma licença do CellProfiler, do napari e do Cellpose. Pode ser usado para qualquer finalidade, inclusive comercial. As versões 1.5.0.0 a 1.5.0.4 usaram a PolyForm Noncommercial License 1.0.0 e as versões até 1.4.9.9 usaram a licença MIT; essas versões continuam disponíveis sob a licença que as acompanhava.

Tutoriais
~~~~~~~~~

A `biblioteca interativa de tutoriais do spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contém demonstrações narradas e legendadas da instalação e de cada fluxo de trabalho: 73 lições, com 50 vozes em oito idiomas.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se o spaCR contribuir para sua pesquisa, cite:

Olafsson EB, *et al.* Um pooled image-based  CRISPR screen identifica o EAF1 como um modulador *T. gondii* da subversão ESCRT.

`bioRxiv pré-impressão <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_  · ? `arquivo de software <https://doi.org/10.5281/zenodo.21343316>`_

Agradecimentos
~~~~~~~~~~~~~~~

O spaCR utiliza software científico aberto, incluindo NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch e Qt. Consulte a `atribuição dos modelos de tradução <../TRANSLATION_MODELS.md>`_ para ver os modelos usados na documentação multilíngue e nos catálogos da interface.
