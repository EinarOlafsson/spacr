|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: DOI de Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores más recientes
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: Receta de conda-forge

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

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.

Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.

Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.

Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.


Flujo de trabajo de un vistazo
------------------------------

.. spacr-workflow-begin

|Workflow_mask| |Workflow_arrow| |Workflow_measure| |Workflow_arrow| |Workflow_annotate| |Workflow_arrow| |Workflow_classify_merged| |Workflow_arrow| |Workflow_map_barcodes| |Workflow_arrow| |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 132
   :alt: Open the Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 132
   :alt: Open the Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 132
   :alt: Open the Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 132
   :alt: Open the Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 132
   :alt: Open the Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 132
   :alt: Open the Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 20
   :align: middle

**More core tools**

|App_timelapse| |App_motility| |App_classify| |App_ml_analyze| |App_curate|

**Data**

|App_align| |App_convert| |App_foreign| |App_external_masks| |App_queue|

|App_batch| |App_distributed_jobs| |App_db_browser| |App_illumination| |App_data_manager|

**Segmentation models**

|App_make_masks| |App_train_cellpose| |App_cellpose_masks| |App_model_compare| |App_model_zoo|

**Results & QC**

|App_plate_view| |App_agreement| |App_umap| |App_activation| |App_train_compare|

|App_classifier_evaluation| |App_run_history| |App_report| |App_barcode_qc| |App_hit_list|

|App_methods_export| |App_volcano_explorer| |App_parameter_sweep| |App_run_compare| |App_explain_cv|

|App_investigate_hit|

**Explore**

|App_pipeline_graph| |App_profiler| |App_qc_dashboard| |App_image_scatter| |App_lineage|

|App_layer_viewer| |App_graph_builder| |App_anndata_export| |App_pca| |App_tabulate|

**Toxoplasma**

|App_analyze_plaques| |App_recruitment| |App_invasion| |App_replication|

**Design**

|App_experiment_design| |App_power|

.. |App_timelapse| image:: ../../../spacr/resources/icons/workflow/apps/timelapse.png
   :width: 183
   :alt: Open the Timelapse API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_motility| image:: ../../../spacr/resources/icons/workflow/apps/motility.png
   :width: 183
   :alt: Open the Motility Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html
   :align: middle
.. |App_classify| image:: ../../../spacr/resources/icons/workflow/apps/classify.png
   :width: 183
   :alt: Open the Classify (CV) API
   :target: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html
   :align: middle
.. |App_ml_analyze| image:: ../../../spacr/resources/icons/workflow/apps/ml_analyze.png
   :width: 183
   :alt: Open the Classify (ML) API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |App_curate| image:: ../../../spacr/resources/icons/workflow/apps/curate.png
   :width: 183
   :alt: Open the Curate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/curate/index.html
   :align: middle
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 183
   :alt: Open the Align & Stitch API
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 183
   :alt: Open the Format Converter API
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 183
   :alt: Open the Import Project API
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 183
   :alt: Open the External Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 183
   :alt: Open the Plate Queue API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 183
   :alt: Open the Batch Runner API
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 183
   :alt: Open the Distributed Jobs API
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 183
   :alt: Open the Database Browser API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_illumination| image:: ../../../spacr/resources/icons/workflow/apps/illumination.png
   :width: 183
   :alt: Open the Illumination API
   :target: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 183
   :alt: Open the Data Manager API
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 183
   :alt: Open the Make Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_train_cellpose| image:: ../../../spacr/resources/icons/workflow/apps/train_cellpose.png
   :width: 183
   :alt: Open the Train Cellpose API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_cellpose_masks| image:: ../../../spacr/resources/icons/workflow/apps/cellpose_masks.png
   :width: 183
   :alt: Open the Cellpose Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html
   :align: middle
.. |App_model_compare| image:: ../../../spacr/resources/icons/workflow/apps/model_compare.png
   :width: 183
   :alt: Open the Model Compare API
   :target: https://einarolafsson.github.io/spacr/api/spacr/model_compare/index.html
   :align: middle
.. |App_model_zoo| image:: ../../../spacr/resources/icons/workflow/apps/model_zoo.png
   :width: 183
   :alt: Open the Model Zoo API
   :target: https://einarolafsson.github.io/spacr/api/spacr/model_zoo/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 183
   :alt: Open the Plate Viewer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_agreement| image:: ../../../spacr/resources/icons/workflow/apps/agreement.png
   :width: 183
   :alt: Open the Annotator Agreement API
   :target: https://einarolafsson.github.io/spacr/api/spacr/agreement/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 183
   :alt: Open the Image UMAP API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_activation| image:: ../../../spacr/resources/icons/workflow/apps/activation.png
   :width: 183
   :alt: Open the Activation API
   :target: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 183
   :alt: Open the Training Runs API
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_classifier_evaluation| image:: ../../../spacr/resources/icons/workflow/apps/classifier_evaluation.png
   :width: 183
   :alt: Open the Classifier Evaluation API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classifier_evaluation/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 183
   :alt: Open the Run History API
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 183
   :alt: Open the Report API
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_barcode_qc| image:: ../../../spacr/resources/icons/workflow/apps/barcode_qc.png
   :width: 183
   :alt: Open the Barcode QC API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html
   :align: middle
.. |App_hit_list| image:: ../../../spacr/resources/icons/workflow/apps/hit_list.png
   :width: 183
   :alt: Open the Hit List API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/hit_list/index.html
   :align: middle
.. |App_methods_export| image:: ../../../spacr/resources/icons/workflow/apps/methods_export.png
   :width: 183
   :alt: Open the Methods & Results API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/methods_export/index.html
   :align: middle
.. |App_volcano_explorer| image:: ../../../spacr/resources/icons/workflow/apps/volcano_explorer.png
   :width: 183
   :alt: Open the Volcano Explorer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/volcano_style/index.html
   :align: middle
.. |App_parameter_sweep| image:: ../../../spacr/resources/icons/workflow/apps/parameter_sweep.png
   :width: 183
   :alt: Open the Parameter Sweep API
   :target: https://einarolafsson.github.io/spacr/api/spacr/parameter_sweep/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 183
   :alt: Open the Run Compare API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_explain_cv| image:: ../../../spacr/resources/icons/workflow/apps/explain_cv.png
   :width: 183
   :alt: Open the Explain CV Model API
   :target: https://einarolafsson.github.io/spacr/api/spacr/surrogate/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 183
   :alt: Open the Investigate Hit API
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 183
   :alt: Open the Pipeline Graph API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 183
   :alt: Open the Prediction Profiler API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 183
   :alt: Open the QC Dashboard API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_image_scatter| image:: ../../../spacr/resources/icons/workflow/apps/image_scatter.png
   :width: 183
   :alt: Open the Image Scatter API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/image_scatter/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 183
   :alt: Open the Lineage API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 183
   :alt: Open the Layer Viewer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 183
   :alt: Open the Graph Builder API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_anndata_export| image:: ../../../spacr/resources/icons/workflow/apps/anndata_export.png
   :width: 183
   :alt: Open the AnnData Export API
   :target: https://einarolafsson.github.io/spacr/api/spacr/anndata_export/index.html
   :align: middle
.. |App_pca| image:: ../../../spacr/resources/icons/workflow/apps/pca.png
   :width: 183
   :alt: Open the PCA API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pca/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 183
   :alt: Open the Tabulate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 183
   :alt: Open the Plaque Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 183
   :alt: Open the Recruitment API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 183
   :alt: Open the Invasion Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 183
   :alt: Open the Replication Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 183
   :alt: Open the Experiment Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 183
   :alt: Open the Power / Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle

.. spacr-workflow-end

The main path is Mask → Measure → Annotate → Classify → Map Barcodes → Regression. The grid below it contains every other application in the same categories and order used on the spaCR home screen.


Instalar spaCR
--------------

Aplicación de escritorio
~~~~~~~~~~~~~~~~~~~~~~~~

Los instaladores de escritorio incluyen un entorno privado Python, por lo que conda y una instalación existente Python no son necesarios.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| =|InstallerWindows| +|InstallerLegacy|

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

En Linux, haga que el archivo descargado sea ejecutable y ejecútelo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

En macOS, abra el archivo ``.pkg``. La beta actual no está notarizada; si Gatekeeper la bloquea, seleccione **Ajustes del Sistema → Privacidad y seguridad → Abrir igualmente**.

Consulte las instrucciones `Guía del instalador <../../source/installer_guide.rst>`_ para actualizar, desinstalar, offline y solucionar problemas.

Instalación con Python
~~~~~~~~~~~~~~~~~~~~~~

Python 3.12 tiene la más amplia selección de envases científicos opcionales:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for CUDA workflows; macOS and Windows are also supported.

Para un servidor, clúster o corredor CI, omita Qt:

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

Establecer ``SPACR_LOG_LEVEL=DEBUG`` al solucionar problemas. Los registros de rotación se escriben en ``~/.spacr/logs/spacr.log``. La interfaz clásica Tk permanece disponible como ``spacr-legacy`` pero ya no está desarrollada.


Qué puede hacer
---------------

La mayoría de las cribados siguen seis módulos:

- **Mask** segmenta células, núcleos, patógenos y orgánelos con Cellpose.
- **Measure** escribe morfología, intensidad, textura, características espaciales y de colocalización, junto con recortes de objetos, a SQLite.
- **Annotate** recorte en una cuadrícula dirigida por teclado y soporta colas de aprendizaje activo.
- **Classify** forma modelos basados en imágenes o mediciones y registros de rendimiento retenido con cada puesto de control.
- **Map Barcodes** maps FASTQ lee a los pozos y gRNAs, con abundancia, colisión y cobertura QC.
- **Regression** guían, el gen, la condición y los efectos de control con familias modelo adecuadas para respuestas continuas, fraccionarias y de recuento.

El mismo proyecto también puede diseñar placas, estimar la potencia, corregir los efectos por lotes, inspeccionar la calidad de segmentación, explorar parcelas y recortes vinculados, exportar AnnData, reanudar el trabajo interrumpido y registrar los ajustes detrás de cada resultado.

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


Contribuciones y soporte
------------------------

Bug reports y focused feature requests are welcome through `GitHub Cuestiones <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings y the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Licencia
~~~~~~~~~

The current development branch is source-available under the `Licencia PolyForm no comercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Tutoriales
~~~~~~~~~~

El `biblioteca interactiva de tutoriales spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contiene recorridos narrados, con subtítulos de instalación y de cada flujo de trabajo de aplicaciones, en 73 lecciones con 50 voces en ocho idiomas.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343317>`_

Agradecimientos
~~~~~~~~~~~~~~~

spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la `atribución de los modelos de traducción <../TRANSLATION_MODELS.md>`_ para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
