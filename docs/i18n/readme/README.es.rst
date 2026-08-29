|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI de Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores más recientes
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versión en conda-forge

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

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, integra los fenotipos por objeto con la abundancia de guías derivada de la secuenciación y estima qué genes están asociados con cambios fenotípicos. A partir de imágenes de placas y lecturas FASTQ, produce mediciones por objeto, clasificadores entrenados, estimaciones del efecto por guía y por gen y una lista ordenada de resultados.

Para los cribados CRISPR agrupados y basados en imágenes, spaCR proporciona el flujo de trabajo desde la segmentación de imágenes hasta la priorización de resultados. Para estudios de microscopía de alto contenido sin cribados basados en secuenciación, los módulos de segmentación, medición, anotación y clasificación pueden utilizarse de forma independiente.

Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.

Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.


Flujo de trabajo de un vistazo
------------------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Abrir la API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Abrir la API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Abrir la API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Abrir la API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Abrir la API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Abrir la API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Datos**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

|App_project_browser|

**Resultados y control de calidad**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|\ |App_control_chart|

**Explorar**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|\ |App_feature_dict|\ |App_trellis|\ |App_gate_editor|

|App_feature_explorer|\ |App_outliers|

**Ensayos**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Diseño**

|App_experiment_design|\ |App_power|\ |App_dose_response|

.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: Abrir la API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: Abrir la API de Format Converter
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: Abrir la API de Import Project
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: Abrir la API de External Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: Abrir la API de Plate Queue
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: Abrir la API de Batch Runner
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: Abrir la API de Distributed Jobs
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: Abrir la API de Database Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: Abrir la API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: Abrir la API de Data Manager
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_project_browser| image:: ../../../spacr/resources/icons/workflow/apps/project_browser.png
   :width: 19.9%
   :alt: Abrir la API de Project Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/project_browser/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: Abrir la API de Plate Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: Abrir la API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: Abrir la API de Training Runs
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: Abrir la API de Run History
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: Abrir la API de Report
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: Abrir la API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: Abrir la API de Investigate Hit
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_control_chart| image:: ../../../spacr/resources/icons/workflow/apps/control_chart.png
   :width: 19.9%
   :alt: Abrir la API de Control Charts
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/control_chart/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: Abrir la API de Pipeline Graph
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: Abrir la API de Prediction Profiler
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: Abrir la API de QC Dashboard
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: Abrir la API de Lineage
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: Abrir la API de Layer Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: Abrir la API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: Abrir la API de Tabulate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_feature_dict| image:: ../../../spacr/resources/icons/workflow/apps/feature_dict.png
   :width: 19.9%
   :alt: Abrir la API de Feature Dictionary
   :target: https://einarolafsson.github.io/spacr/api/spacr/feature_dict/index.html
   :align: middle
.. |App_trellis| image:: ../../../spacr/resources/icons/workflow/apps/trellis.png
   :width: 19.9%
   :alt: Abrir la API de Small Multiples
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/trellis/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 19.9%
   :alt: Abrir la API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_feature_explorer| image:: ../../../spacr/resources/icons/workflow/apps/feature_explorer.png
   :width: 19.9%
   :alt: Abrir la API de Feature Explorer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/feature_explorer/index.html
   :align: middle
.. |App_outliers| image:: ../../../spacr/resources/icons/workflow/apps/outliers.png
   :width: 19.9%
   :alt: Abrir la API de Outliers
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/outliers/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: Abrir la API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: Abrir la API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: Abrir la API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: Abrir la API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: Abrir la API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: Abrir la API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 19.9%
   :alt: Abrir la API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle

.. spacr-workflow-end

Seleccione un módulo del flujo de trabajo para abrir su página de API. La cuadrícula contiene las demás aplicaciones, organizadas en las mismas categorías y en el mismo orden que en la pantalla de inicio de spaCR.


Instalar spaCR
--------------

Aplicación de escritorio
~~~~~~~~~~~~~~~~~~~~~~~~

Los instaladores de escritorio incluyen un entorno privado Python, por lo que conda y una instalación existente Python no son necesarios.

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

Instalación con conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~

El paquete oficial de conda-forge instala spaCR y sus dependencias de escritorio en el entorno activo:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Instalación desde PyPI
~~~~~~~~~~~~~~~~~~~~~~

Para la versión de PyPI, instale spaCR con pip dentro de un entorno Conda. Python 3.12 ofrece la mayor variedad de paquetes científicos opcionales:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR admite Python **3.9 a 3.14**, salvo Python 3.14.1, que torchvision excluye. Se recomienda Linux para los flujos de trabajo con CUDA; macOS y Windows también son compatibles.

En un servidor, clúster o ejecutor de CI, omita Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` y ``spacr[czi,nd2,lif]``. See the `Guía de instalación <../../source/installer_guide.rst>`_ for the complete extras y Python-version compatibility table.

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


Qué puede hacer
---------------

El flujo de trabajo principal comprende seis módulos:

- **Mask** segmenta células, núcleos, patógenos y orgánulos con Cellpose.
- **Measure** guarda en SQLite características morfológicas, de intensidad, textura, espaciales y de colocalización, junto con recortes de objetos.
- **Annotate** etiqueta recortes en una cuadrícula controlada con el teclado y admite colas de aprendizaje activo.
- **Classify** entrena modelos basados en imágenes o mediciones y registra con cada punto de control el rendimiento en los datos reservados.
- **Map Barcodes** asigna las lecturas FASTQ a los pocillos y los gRNA, con controles de calidad de abundancia, colisiones y cobertura.
- **Regression** estima los efectos de guías, genes, condiciones y controles con familias de modelos adecuadas para respuestas continuas, fraccionarias y de recuento.

El mismo proyecto también puede diseñar placas, estimar la potencia estadística, corregir efectos de lote, inspeccionar la calidad de la segmentación, explorar gráficos e imágenes recortadas vinculados, exportar AnnData, reanudar procesos interrumpidos y registrar los ajustes asociados a cada resultado.

Módulos disponibles desde sus pantallas anfitrionas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Veinte módulos están integrados en pantallas anfitrionas relacionadas, en lugar de aparecer como mosaicos independientes en la pantalla de inicio. Cada módulo se abre desde la cabecera de su pantalla anfitriona y utiliza el proyecto activo. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP y Make Masks proporcionan estos módulos integrados. Su ayuda y documentación de la API siguen disponibles, y los módulos con puntos de entrada de canalización aún pueden ejecutarse sin interfaz gráfica. La `guía de funciones <../../source/features.rst>`_ enumera cada módulo integrado y su pantalla anfitriona.

Make Masks
~~~~~~~~~~

Make Masks aparece en **Data** y permite corregir manualmente las máscaras de segmentación. Su cabecera también da acceso a los flujos de trabajo de Cellpose. El lienzo tiene nueve herramientas: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** y **Recrop**. Draw crea una etiqueta rellena a partir de un contorno cerrado dibujado a mano alzada. Divide separa un objeto fusionado por una línea definida por el usuario y conserva las etiquetas de todos los demás objetos.

Recrop extrae un campo con un solo objeto de una imagen preparada que contiene varios objetos. Un cuadro delimitador alrededor de un objeto guarda las regiones correspondientes de la imagen y la máscara como un campo nuevo, programa ese campo después del actual y elimina de la cola de curación el campo original con varios objetos. Recrop cambia el campo activo, no los píxeles de las etiquetas.

Al ejecutar Cellpose-SAM desde Make Masks se muestran dos resultados intermedios junto a la máscara: el **mapa de probabilidad celular** y el **campo de flujo**. La máscara se define mediante un umbral sobre el mapa de probabilidad, y las comprobaciones de coherencia del flujo pueden rechazar objetos cuyos flujos derivados difieran del campo predicho. Examine estos resultados para distinguir una probabilidad celular baja de un flujo incoherente al evaluar una máscara incorrecta o incompleta.

Objetos y ajustes
~~~~~~~~~~~~~~~~~~~~

spaCR admite objetos de célula, núcleo y patógeno, un citoplasma derivado de sus máscaras y entre cero y veintiséis espacios para orgánulos. Cada espacio para orgánulos tiene un canal, diámetro, ajuste predefinido de morfología y método de detección independientes.

El panel de ajustes solo muestra los controles cuando son aplicables. Los espacios para orgánulos por encima de la cantidad configurada se ocultan, los objetos sin un canal asignado se excluyen de la ejecución y los controles específicos de la morfología solo se muestran para el método seleccionado. Los interruptores **3D** y **Time** definen la dimensionalidad: ``z_stack`` activa los ajustes volumétricos, ``timelapse`` activa los ajustes de seguimiento y los ajustes de cuatro dimensiones aparecen cuando ambos están activados.

Elija la siguiente página por lo que desea hacer:

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

Diagnóstico del rendimiento
---------------------------

Genere un informe de hardware y adjúntelo a una incidencia relacionada con el rendimiento::

    python tools/spacr_hardware_report.py

El comando imprime un informe y guarda una copia en ``~/.spacr/reports``; la última línea indica la ruta del archivo guardado. ``--quick`` omite las pruebas de rendimiento más largas y ``--out PATH`` selecciona otra ubicación de salida.

El informe no abre ningún proyecto ni lee datos del proyecto. Registra los tiempos de importación y de las bibliotecas numéricas, el escalado de pantalla, las preferencias activas, la construcción de la ventana principal y de las pantallas de los módulos, y el rendimiento de las animaciones. El archivo del informe es la única salida que crea.

El informe también identifica la emulación de la arquitectura del procesador, como una compilación x86_64 de Python en Apple Silicon, y la implementación de BLAS utilizada por NumPy. Ambos factores pueden afectar considerablemente al rendimiento.

Contribuciones y soporte
------------------------

Envíe informes de errores y solicitudes de funciones concretas mediante `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Al informar de un fallo, incluya la versión de spaCR, el sistema operativo, la versión de Python, los ajustes del módulo y el fragmento de registro pertinente. ``spacr-doctor`` recopila la mayor parte de esta información; incluya el informe de hardware cuando notifique problemas de rendimiento.

Licencia
~~~~~~~~~

El código fuente de la rama de desarrollo actual está disponible bajo la `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. El uso comercial requiere una licencia independiente del titular de los derechos de autor. Las versiones publicadas hasta spaCR 1.4.9.9 siguen disponibles bajo la licencia MIT que acompañaba a esas versiones.

Tutoriales
~~~~~~~~~~

La `biblioteca interactiva de tutoriales de spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contiene recorridos narrados y subtitulados de la instalación y de cada flujo de trabajo: 73 lecciones con 50 voces en ocho idiomas.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343316>`_

Agradecimientos
~~~~~~~~~~~~~~~

spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la `atribución de los modelos de traducción <../TRANSLATION_MODELS.md>`_ para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
