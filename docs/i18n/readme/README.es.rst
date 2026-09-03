|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |Preprint| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentación
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriales interactivos
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Versión de PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Conjunto de pruebas
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Interfaz Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código fuente en GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Incidencias de GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licencia PolyForm Noncommercial
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: DOI de Zenodo
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Instaladores más recientes
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Versión en conda-forge
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: spaCR

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

Idiomas: `🌐 Español ▾ <README.md>`_

.. spacr-language-picker-end

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, integra los fenotipos por objeto con la abundancia de guías derivada de la secuenciación y estima qué genes están asociados con cambios fenotípicos. A partir de imágenes de placas y lecturas FASTQ, produce mediciones por objeto, clasificadores entrenados, estimaciones del efecto por guía y por gen y una lista ordenada de resultados.

Los módulos de segmentación, medición, anotación y clasificación también funcionan sin un brazo de secuenciación.

Imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pozo viven en un proyecto SQLite .

Se ejecuta como una aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster.

Soporte de hardware
~~~~~~~~~~~~~~~~~~~

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

soportado (estable)  implementado (beta) CPU soporte solamente

.. spacr-hardware-end


Instalar spaCR
--------------

Aplicación de escritorio
~~~~~~~~~~~~~~~~~~~~~~~~

Los instaladores agrupan sus propios Python. No se requiere Conda.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel y Apple silicon): descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux de 64 bits: descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores de spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Los tres primeros iconos descargan la versión actual. El icono spaCR abre el archivo completo del instalador. Los enlaces al instalador y los nombres de archivos versionados se actualizan por el flujo de trabajo de la versión; los instaladores anteriores permanecen en el mismo archivo de lanzamiento.

En Linux, marque el archivo descargado como ejecutable y ejecútelo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

En macOS, abra el archivo ``.pkg``. La beta actual no está notarizada; si Gatekeeper la bloquea, seleccione **Ajustes del Sistema → Privacidad y seguridad → Abrir igualmente**.

Consulte las instrucciones `Guía del instalador <../../source/installer_guide.rst>`_ para actualizar, desinstalar, offline y solucionar problemas.

Instalación desde PyPI
~~~~~~~~~~~~~~~~~~~~~~

Para la versión de PyPI, instale spaCR con pip dentro de un entorno Conda. Python 3.12 ofrece la mayor variedad de paquetes científicos opcionales:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML.

En un servidor, clúster o ejecutor de CI, omita Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` y ``spacr[czi,nd2,lif]``. See the `Guía de instalación <../../source/installer_guide.rst>`_ for the complete extras y Python-version compatibility table.

Instalación con conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~

El paquete oficial de conda-forge instala spaCR y sus dependencias de escritorio en el entorno activo:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Instalar desde el origen
~~~~~~~~~~~~~~~~~~~~~~~~

Clonar el repositorio e instalarlo en modo editable, por lo que su copia de trabajo *es* el paquete y las ediciones instaladas tienen efecto sin reinstalar::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

La rama predeterminada es ``nightly``. Para una versión específica::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Para tirar de los cambios posteriores, desde el interior del clon::

    git pull
    pip install -e .

La segunda línea sólo es necesaria cuando las dependencias o los puntos de entrada han cambiado; el código Python se recoge sin él. Si una orden todavía ejecuta código antiguo después de tirar, ``spacr-doctor`` informa que ``spacr`` está realmente en su ruta, que es la causa habitual.

Instalar desde la fuente (luz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clon completo: 427 MB. Clon central: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Saltar ``docs/``, ``tests/``, puntos de control Cellpose, cifras archivadas y los catálogos de traducción extendidos. El resultado es una compra normal.

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``, ``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` enumera cada ruta omitida.


Comandos de línea de comandos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Establezca ``SPACR_LOG_LEVEL=DEBUG`` durante la resolución de problemas. Los registros rotatorios se escriben en ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` enumera los módulos con puntos de entrada de línea de comandos para ejecutarse sin interfaz gráfica. Se omiten los módulos de anotación, curación, comparación y exploración disponibles únicamente en la interfaz gráfica.


Flujo de trabajo principal
--------------------------

El flujo de trabajo principal comprende seis módulos:

- **Mask** segmenta células, núcleos, patógenos y orgánulos con Cellpose.
- **Measure** guarda en SQLite características morfológicas, de intensidad, textura, espaciales y de colocalización, junto con recortes de objetos.
- **Annotate** etiqueta recortes en una cuadrícula controlada con el teclado y admite colas de aprendizaje activo.
- **Classify** entrena modelos basados en imágenes o mediciones y registra con cada punto de control el rendimiento en los datos reservados.
- **Map Barcodes** asigna las lecturas FASTQ a los pocillos y los gRNA, con controles de calidad de abundancia, colisiones y cobertura.
- **Regression** estima los efectos de guías, genes, condiciones y controles con familias de modelos adecuadas para respuestas continuas, fraccionarias y de recuento.

El mismo proyecto también puede diseñar placas, estimar la potencia estadística, corregir efectos de lote, inspeccionar la calidad de la segmentación, explorar gráficos e imágenes recortadas vinculados, exportar AnnData, reanudar procesos interrumpidos y registrar los ajustes asociados a cada resultado.

Módulos de spaCR
----------------

.. spacr-workflow-begin

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|
| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|
| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|
| |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Abrir la API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Abrir la API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Abrir la API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Abrir la API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Abrir la API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Abrir la API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Abrir la API de Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Abrir la API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Abrir la API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Abrir la API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Abrir la API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Abrir la API de QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Abrir la API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Abrir la API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Abrir la API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Abrir la API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Abrir la API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Abrir la API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Abrir la API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Abrir la API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Abrir la API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

Cada módulo spaCR envía, en el orden en que la pantalla de inicio los lista: los seis módulos de flujo de trabajo primero, luego todo lo demás. Seleccione una tesela para abrir la página API de ese módulo.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose-SAM se ejecuta aquí mostrar el mapa de probabilidad de celda y el campo de flujo al lado de la máscara. Vea el `guía de características <../../source/features.rst>`_ para cada herramienta.

**Otros recursos**

- `Tutoriales interactivos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 flujos de trabajo guiados desde la instalación hasta la investigación de impacto.
- `Inicio rápido Python API <../../source/python_api.rst>`_ — ejecutar y validar flujos de trabajo desde scripts, cuadernos o un clúster.
- `Guía de características <../../source/features.rst>`_ — capacidades, madurez e integraciones opcionales.
- `Referencia comisariada API <https://einarolafsson.github.io/spacr/api/index.html>`_ — puntos de entrada soportados por tarea, con el módulo completo de referencia un nivel más profundo.
- `Guía de idioma y traducción <../../source/localization.rst>`_ — lenguajes de interfaz, ayuda contextual y política de salida científica.

Idioma y traducción
~~~~~~~~~~~~~~~~~~~~~~

La interfaz admite diez idiomas en la navegación y las preferencias. Los controles AI y LIVE, las descripciones de los módulos y la ayuda contextual revisada también se traducen. Cambie el idioma en **spaCR → Preferencias → Idioma** sin reiniciar. Los registros, las rutas, los valores de la base de datos y las mediciones nunca se traducen; los resultados científicos permanecen en inglés canónico. Consulte la `política de ayuda contextual <../../source/localization.rst#contextual-help>`_.

Guía animada de ajustes
~~~~~~~~~~~~~~~~~~~~~~~~~

Los ajustes con una explicación visual incluyen un control **Animación** en su información emergente. Consulte la `galería de animaciones de ajustes <https://einarolafsson.github.io/spacr/setting_animations.html>`_ o el `registro de animaciones de ajustes <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Datos
-----

Conjuntos de datos de referencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Abrir el conjunto de microscopía en BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Abrir el conjunto de prueba en Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Abrir el conjunto de secuenciación en NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Abrir spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Abrir la prepublicación de bioRxiv
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Zoológico modelo
~~~~~~~~~~~~~~~~

spaCR envía un catálogo de modelos entrenados y los trae a pedido. Abra **Model Zoo** desde la pantalla de inicio para navegar e instalarlos, o nombre una clave en un archivo de configuración -- ``pathogen_model: toxoplasma_pv_v1`` -- y el modelo se descarga y comprueba la primera vez que es necesario. Cada entrada publicada lleva un SHA-256; una entrada sin uno se rechaza en lugar de instalarse, porque un puesto de control truncado o sustituido no se puede decir desde el real.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987

.. spacr-model-zoo-end

Cada figura de arriba se mide en imágenes que el modelo nunca vio en el entrenamiento.

**Precisión** es cuántos de los objetos reportados por un modelo son reales; **recordar** es cuantos de los verdaderos objetos que encontró. Fallan en direcciones opuestas: la mala precisión inventa placas, la mala memoria los echa en falta.

**F1** son los dos combinados, y se cita porque cada uno es trivialmente gamed -- reporta una placa inconfundible para la precisión casi perfecta, o cada mancha oscura para la memoria casi perfecta. Lo que preferirías perder depende del ensayo, y el conteo es generalmente mejor servido por sobrellamada: el modelo de placa fue aceptado con precisión 0,858 con memoria 0,811 sobre una ronda anterior en 0,939 y 0,631.

**IoU**, intersección sobre unión, es cuánto un objeto predicho y el real se superponen, dividido por el área que cubren juntos. Es la regla contra la que se leen los demás, así que una puntuación no significa nada sin su umbral: "F1 0.867 a IoU 0.5" cuenta una vacuole como se encuentra cuando los dos contornos están de acuerdo sobre la mitad de su área combinada.

**mAP50** and **mAP50-95** belong to the detector. The first asks whether the wells were found; the second repeats it across ten thresholds from 0.5 to 0.95, so it also asks how tightly each box is drawn. The gap between them is placement, not detection.

**Cross-validated**, con un **SD**, significa que la puntuación es la media de tres ejecuciones en diferentes divisiones y el SD es lo lejos que se alejaron. Una división puede tener suerte: la cifra de literatura de este modelo es 0,834 en una sola división de 19 pocillos y 0,806 en los tres.

Models are hosted on their author's own Hugging Face account, so contributing one does not mean handing write access to anyone else's. ``spacr.model_zoo``'s ``publish_model`` performs the upload and prints the catalogue row to add.


Diagnóstico del rendimiento
---------------------------

Genere un informe de hardware y adjúntelo a una incidencia relacionada con el rendimiento::

    python tools/spacr_hardware_report.py

Guarda en ``~/.spacr/reports`` e imprime la ruta. ``--quick`` omite los parámetros de referencia más largos; ``--out PATH`` establece la ubicación.

No lee datos del proyecto. Importación de tiempos, bibliotecas numéricas, construcción de ventanas y animación. Reporta la emulación del procesador-arquitectura (una construcción x86_64 Python en Apple Silicon) y la implementación BLAS de NumPy.

Referencia de la línea de órdenes
---------------------------------

Cada comando de abajo está instalado por ``pip install spacr``. Todos aceptan ``--help``.

Lanzamiento de la aplicación
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` omite la cribado de configuración modal, que de otro modo bloquearía un trabajo no vigilado.

``spacr-qt`` y ``spacr-nightly`` son alias de ``spacr``.

Cuando spaCR no se iniciará
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` imprime una línea por cheque, con un comando para ejecutar por cada fallo. También informa que ``spacr`` está en la ruta, que es lo que una vieja instalación editable sombras.

``safespacr`` lee cada preferencia como por defecto y fuerza el telón de fondo, animaciones, registro verboso y precargar. Utilícela cuando una preferencia guardada rompa el lanzamiento. No cambia nada de forma permanente.

Módulos de ejecución sin interfaz gráfica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No Qt, no display — para clusters, servidores e IC.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` lee los mismos ajustes que la ejecución haría e informa de lo que falta, contradictorio o apuntando a nada.

``spacr-run --list`` muestra sólo módulos con un punto de entrada sin interfaz gráfica; la anotación, curatela y exploración son interactivas y omitidas.

Inspeccionar una carrera después
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cada ejecución se lleva a cabo a ``~/.spacr/runs`` con sus ajustes, entradas de hashed, salidas, advertencias, versiones y semillas.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Datos de auditoría e instalación
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Entorno
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Los registros de rotación se escriben en ``~/.spacr/logs/spacr.log``. Adjuntar ese archivo a un informe de fallo.


Contribuciones y soporte
------------------------

Envíe informes de errores y solicitudes de funciones concretas mediante `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Al informar de un fallo, incluya la versión de spaCR, el sistema operativo, la versión de Python, los ajustes del módulo y el fragmento de registro pertinente. ``spacr-doctor`` recopila la mayor parte de esta información; incluya el informe de hardware cuando notifique problemas de rendimiento.

Licencia
~~~~~~~~~

spaCR se libera bajo el `Licencia de 3-clausura BSD <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Si spaCR contribuyó al trabajo publicado, una citación es apreciada y no es una condición de la licencia — véase `Citing spaCR`_ a continuación.

Tutoriales
~~~~~~~~~~

La `biblioteca interactiva de tutoriales de spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contiene recorridos narrados y subtitulados de la instalación y de cada flujo de trabajo: 73 lecciones con 50 voces en ocho idiomas.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · =`archivo de software <https://doi.org/10.5281/zenodo.21343316>`_

Agradecimientos
~~~~~~~~~~~~~~~

spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la `atribución de los modelos de traducción <../TRANSLATION_MODELS.md>`_ para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
