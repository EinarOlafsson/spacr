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

`Información sobre los modelos de traducción <../TRANSLATION_MODELS.md>`_

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.

Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.

Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.

Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.


Flujo de trabajo de un vistazo
------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: Flujo de trabajo y organización de resultados de spaCR
   :align: center

Las imágenes de microscopía (TIFF, OME-TIFF, LIF, CZI, ND2) y las lecturas de secuenciación (FASTQ) pasan por flujos complementarios de análisis de imágenes y asignación de códigos de barras. Después se analizan conjuntamente las tablas de objetos, los recortes, las anotaciones, las predicciones, las identidades de guía, los resultados de QC y los resúmenes por pocillo.


Inicio rápido
-------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR es compatible con Python **3.9 a 3.14** (excepto Python 3.14.1, que torchvision excluye). Python 3.12 ofrece la mayor variedad de paquetes científicos opcionales. Se recomienda Linux para los flujos de trabajo con CUDA; macOS y Windows también son compatibles.


Detalles de instalación
-----------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Instaladores ligeros de escritorio:**

.. spacr-installer-links-begin

|InstallerWindows| |InstallerMacOS| |InstallerLinux|

.. |InstallerWindows| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: descargar SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel y silicio Apple): descargar SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-bit Linux: descargar SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run

.. spacr-installer-links-end


Los iconos de arriba apuntan siempre a la versión más reciente. Todas
las anteriores siguen descargándose desde el `archivo de instaladores
<https://einarolafsson.github.io/spacr/installers.html>`_ — una tabla, una fila por versión, y cada
instalador fija la versión para la que se construyó.

Instaladores ligeros — no requieren conda ni una instalación de Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Durante la instalación se descargan un entorno privado de Python 3.12, Qt, PyTorch, spaCR y las dependencias científicas, por lo que no hace falta tener conda ni Python instalados. La versión portátil para CPU es la opción predeterminada y evita descargar sin aviso varios gigabytes de bibliotecas CUDA. Windows ofrece la aceleración NVIDIA como componente opcional, Linux acepta ``--torch-backend auto`` y el wheel estándar de PyTorch para macOS conserva la aceleración Apple MPS.

La ayuda, el progreso y los errores del instalador siguen el idioma del sistema operativo en los diez idiomas de spaCR: inglés, sueco, alemán, español, chino simplificado, portugués, hindi, coreano, islandés y francés. Las configuraciones regionales no compatibles utilizan el inglés.

En Linux, haga que el instalador descargado sea ejecutable antes de abrirlo:

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

En macOS, abra el archivo ``.pkg`` descargado. Si Gatekeeper bloquea el instalador beta porque no está notarizado, abra **Ajustes del Sistema → Privacidad y seguridad**, seleccione **Abrir igualmente** para spaCR y vuelva a ejecutar el paquete.

El instalador valida spaCR, Qt, PyTorch y la coherencia de las dependencias antes de reemplazar una instalación anterior, por lo que una actualización interrumpida conserva el entorno de trabajo anterior. Se guarda un registro de diagnóstico como ``install.log`` dentro del directorio privado de instalación de spaCR.

Aplicación de escritorio desde PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Instalación sin interfaz gráfica o en servidor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Rama de desarrollo más reciente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Entornos conda
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Funciones opcionales
~~~~~~~~~~~~~~~~~~~~~

Instale sólo los extras que necesita su flujo de trabajo:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Los extras que pueden instalarse dependen de la versión de Python. En Python 3.13, ultrack limita ``spacr[all]`` y la restricción de NumPy de TorchCAM limita el extra ``attribution``; el paquete principal y la aplicación Qt no se ven afectados. En Python 3.14, btrack está disponible mediante su extra. El convertidor CZI pylibCZIrw es opcional y no se ha probado; la lectura de archivos CZI mediante czifile sigue disponible.

La interfaz Tk heredada todavía se instala como ``spacr-legacy``, pero ya no se desarrolla.


Comandos de línea de comandos
-----------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Para diagnosticar problemas, establezca ``SPACR_LOG_LEVEL=DEBUG``. Los registros rotatorios se escriben en ``~/.spacr/logs/spacr.log``.


Funciones
---------

Los seis módulos más usados en los cribados
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmenta células, núcleos, patógenos y orgánulos con Cellpose, tanto en imágenes 2D como en datos volumétricos o series temporales. La lista de modelos se obtiene de la instalación de Cellpose en lugar de estar codificada de forma fija, y antes de comenzar se estima el diámetro de los objetos a partir de las imágenes. Las máscaras pueden corregirse manualmente en el visor de capas o enviarse a napari para editarlas y recuperarlas después.

**Measure** guarda en la base de datos del proyecto la morfología, intensidad, textura y colocalización de cada objeto, junto con sus recortes. Como novedad de 1.5.0.0, la corrección de iluminación estima el campo plano a partir de la propia placa y corrige las imágenes antes de calcular cualquier característica de intensidad. Esto elimina el sesgo por posición de pocillo que aparece como efecto de borde en los mapas de calor de la placa. Antes de ejecutar Measure, un aviso de QC de segmentación describe las máscaras en lenguaje sencillo; informa, pero no bloquea la ejecución. Un polígono dibujado limita la medición a una región de interés.

**Annotate** muestra los recortes en una cuadrícula controlada con el teclado y guarda las etiquetas directamente en SQLite. El ciclo de aprendizaje activo está integrado en la pantalla: permite reentrenar un modelo con los datos ya etiquetados, reordenar la cola según la incertidumbre, observar la curva de aprendizaje y recibir una recomendación de parada cuando nuevas etiquetas dejan de modificar el modelo. La cobertura se informa por clase, pocillo y placa, y cada ronda queda registrada.

**Classify** entrena CNN y Transformer de PyTorch con recortes anotados, y modelos clásicos o de boosting con tablas de mediciones. Ahora conserva la exactitud por clase en cada epoch, y cada checkpoint recibe una ficha que registra el conjunto de datos, el equilibrio de clases, la regla de partición y las métricas del conjunto de reserva. En la pantalla de evaluación, cada celda de la matriz de confusión funciona como una consulta: al pulsarla se abren los recortes correspondientes y se separan las predicciones erróneas de alta confianza de los casos inciertos.

**Map Barcodes** decodifica los códigos de barras de fila, columna y gRNA de las lecturas FASTQ, asigna identidades de guía a los pocillos y las vincula con las células fotografiadas. Barcode QC informa del número de lecturas por pocillo, la tasa de colisión y la fracción sin asignar, y evalúa un intervalo alrededor del número esperado de gRNA por pocillo en lugar de aplicar un umbral fijo.

**Regression** estima los efectos de guía, gen, condición y control mediante 17 familias de modelos, incluidos modelos mixtos, Logistic, Probit, Quantile, Beta, GLM con varianza cuasibinomial, Lasso, Ridge, Elastic Net, Hinge y Horseshoe. El resultado es una lista de candidatos ordenada y anotada, no un simple volcado de coeficientes.

Novedades de 1.5.0.0
~~~~~~~~~~~~~~~~~~~~

Antes de que exista un cribado, el módulo Power / Design calcula cuántas células y cuántos pocillos necesita, teniendo en cuenta el error de secuenciación y la pérdida de pocillos con muy pocas células fotografiadas. Un diseñador de experimentos organiza la placa, sus controles y sus réplicas, y exporta el diseño para el flujo de trabajo. Después, un panel de QC reúne las comprobaciones de segmentación, placa, concordancia entre anotadores y fuga de datos en un solo veredicto; ComBat está disponible junto a ``center`` y ``zscore`` para la corrección por lotes.

Los resultados se exploran directamente, sin exportarlos y volverlos a importar. Graph Builder crea una gráfica arrastrando columnas a x, y, color, tamaño y faceta. Las regiones dibujadas en un histograma o diagrama de dispersión se convierten en filtros. Un explorador de características las ordena según su capacidad para separar las clases. Los paneles múltiples, los ajustes de dosis-respuesta, los gráficos de control y la detección robusta de valores atípicos comparten el mismo motor de ejes. Al seleccionar objetos en una vista quedan seleccionados en todas; al abrir la selección aparecen sus recortes de origen. Un visor de capas superpone imágenes, etiquetas, puntos y formas, con vistas ortogonales, una cuadrícula de comparación sincronizada y un árbol de linaje desde la célula hasta el núcleo y el patógeno.

Ahora cada ejecución es identificable. Lleva un identificador, una semilla y una política ``on_error``; Mask, Measure, Classify y la exportación a AnnData registran lo que escriben en un registro de artefactos, de modo que cada archivo de salida puede rastrearse hasta los ajustes que lo produjeron. Los módulos se abren con la salida real del paso anterior, el grafo del flujo marca las salidas obsoletas, la comparación de ejecuciones muestra las diferencias de ajustes, recuentos de objetos y listas de resultados, y cada ejecución desde la interfaz genera el script de Python equivalente. Las mediciones se exportan a ``.h5ad`` para scanpy; OME-Zarr y OMERO están disponibles mediante la API de Python. El exportador de métodos y resultados redacta esas dos secciones del manuscrito a partir de un resumen estructurado: el modelo escribe la prosa, pero todas las cifras proceden del resumen, y se rechaza cualquier borrador que contenga una cifra ausente de él. Si hay un problema con la instalación, ``spacr-doctor`` indica qué spaCR se está ejecutando, si la GPU funciona, si Cellpose coincide con la API utilizada por spaCR y si la base de datos y los ajustes del proyecto son válidos; además, ofrece una solución copiable para cada comprobación fallida.

Interfaz de escritorio multilingüe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Preferencias → Idioma** cambia la aplicación en ejecución a inglés, sueco, alemán, español, chino simplificado, portugués, hindi, coreano, islandés o francés sin reiniciarla. La selección se conserva y también se aplica a las pantallas que se abran después.

La navegación, las preferencias, los controles de AI y LIVE, las descripciones de los módulos y los avisos de consola escritos por spaCR siguen el idioma seleccionado. La salida de los procesos, los registros, los rastreos de error, las rutas, los valores de la base de datos, las anotaciones, las respuestas de AI, las mediciones y los resultados guardados no se traducen, de modo que la salida científica permanece en el inglés canónico. Las ayudas de ajustes que aún no se hayan revisado en un idioma permanecen en inglés para evitar explicaciones con idiomas mezclados. La `guía de localización <https://einarolafsson.github.io/spacr/localization.html>`_ documenta este comportamiento, la variable de entorno que permite cambiarlo y la `ayuda contextual <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ que se traduce con la interfaz.

Guía animada de ajustes
~~~~~~~~~~~~~~~~~~~~~~~~

94 animaciones breves muestran cómo afectan a una imagen 143 ajustes visuales. Pase el puntero sobre un ajuste y pulse **Animación** en su ayuda para reproducir la vista previa cuadrada situada junto al texto; vuelva a pulsar para plegarla. Las animaciones solo se reproducen cuando se solicitan y pueden desactivarse en Preferencias. La `galería <https://einarolafsson.github.io/spacr/setting_animations.html>`_ las reúne todas, y el `registro de animaciones de ajustes <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ indica a qué ajuste corresponde cada una.

Referencia de módulos
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Módulo
     - Función
     - Estado
     - Descripción
   * - **Experiencia de escritorio**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Estable
     - Traduce al instante las pantallas abiertas o creadas bajo demanda entre diez idiomas incluidos.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Estable
     - Localiza los resúmenes de módulos y la interfaz de ayuda de los ajustes sin alterar las URL de la API.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Estable
     - Localiza los controles de AI y LIVE sin modificar el contenido del usuario ni del modelo.
   * - |api-animations|_
     - |doc-animations|_
     - Estable
     - Reproduce desde la ayuda de cada ajuste 94 animaciones incluidas para 143 ajustes visuales.
   * - |api-selection|_
     - |api-linked-views|_
     - Alfa
     - Comparte una selección de objetos entre las vistas de tabla, placa, embedding, dispersión y gráfico.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alfa
     - Diagnostica la GPU, la API de Cellpose, la base de datos y los ajustes, y propone una solución para cada comprobación fallida.
   * - **Análisis de imágenes**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Estable
     - Segmenta células, núcleos, patógenos y orgánulos en imágenes 2D.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segmenta imágenes volumétricas y series temporales 4D.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alfa
     - Estima el campo plano a partir de la placa y lo corrige antes de medir la intensidad.
   * - |api-measure|_
     - |api-measure-2d|_
     - Estable
     - Mide morfología, intensidad, textura y colocalización, y guarda los recortes.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alfa
     - Describe la calidad de la segmentación antes de ejecutar Measure, sin bloquear la ejecución.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Sigue objetos con IoU, Trackpy, btrack, Trackastra o ultrack y cuantifica su motilidad.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alfa
     - Superpone capas de imágenes, etiquetas, puntos y formas, con vistas ortogonales y una cuadrícula comparativa.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alfa
     - Envía una máscara a napari para corregirla, la recupera y registra cada modificación.
   * - **AI y análisis de fenotipos**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Estable
     - Revisa recortes en una cuadrícula controlada con el teclado y guarda las anotaciones en SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alfa
     - Reentrena desde Annotate, reordena por incertidumbre e indica cuándo puede detenerse el etiquetado.
   * - |api-classify|_
     - |api-classification|_
     - Estable
     - Entrena y aplica modelos CNN y transformer de PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alfa
     - Registra junto a cada punto de control el conjunto de datos, el equilibrio de clases, la regla de partición y las métricas de reserva.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alfa
     - Abre los recortes asociados a una celda de la matriz de confusión y separa los errores seguros de los casos inciertos.
   * - |api-ml|_
     - |api-ml-models|_
     - Estable
     - Entrena modelos clásicos y de boosting interpretables a partir de tablas de mediciones.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explica las predicciones con Captum, SmoothGrad y TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explora de forma interactiva los embeddings de imágenes y propaga etiquetas de clúster.
   * - **Secuenciación y análisis de cribados**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Estable
     - Asigna los códigos de barras de fila, columna y gRNA de las lecturas FASTQ y vincula las guías con las células fotografiadas.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alfa
     - Informa de las lecturas por pocillo, la tasa de colisión y la fracción sin asignar según los gRNA esperados por pocillo.
   * - |api-regression|_
     - |api-regression-models|_
     - Estable
     - Estima los efectos de guía, gen, condición y control con 17 familias de modelos.
   * - |api-power|_
     - |api-power-design|_
     - Alfa
     - Calcula cuántas células y pocillos necesita un cribado teniendo en cuenta el error de secuenciación y la pérdida de pocillos.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alfa
     - Crea un gráfico arrastrando columnas a x, y, color, tamaño y faceta.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alfa
     - Registra el identificador, la semilla y los ajustes que produjeron las salidas de Mask, Measure, Classify y exportación.

.. |api-qt-app| replace:: **Aplicación Qt**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Localización en diez idiomas**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Ayuda contextual localizada**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **Consola asistida por AI**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Registro de animaciones de ajustes**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Animaciones de ajustes visuales**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selección**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Selección vinculada**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Diagnóstico de la instalación**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **Generación de máscaras 2D**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **Generación de máscaras 3D y 4D**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Iluminación**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Corrección de campo plano**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Mediciones de objetos**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **QC de segmentación**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Veredicto previo a la ejecución**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Seguimiento de objetos**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Capas**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Visor de capas**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **Puente con napari**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Curación de máscaras**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Anotación manual**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Aprendizaje activo**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Reentrenamiento y reordenación**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Clasificación de imágenes**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Fichas de modelos**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Mapas de activación**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Exploración de la matriz de confusión**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Aprendizaje automático**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Clasificación de mediciones**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Embedding interactivo**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Secuenciación**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Asignación de códigos de barras**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **QC de códigos de barras**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Informe de pocillos y colisiones**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Estimación de efectos del cribado**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Potencia y diseño**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artefactos**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Procedencia de la ejecución**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Datos
-----

Conjuntos de datos de referencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Conjunto completo de datos de microscopía: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Conjunto de datos de prueba: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Datos de secuenciación: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Análisis de potencia: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contribuciones y soporte
------------------------

Los informes de errores y las solicitudes concretas de funciones son bienvenidos en `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Al informar de un fallo, incluya la versión de spaCR, el sistema operativo, la versión de Python, los ajustes del módulo y el fragmento de registro pertinente. ``spacr-doctor`` recopila automáticamente la mayor parte de esta información.

Licencia
~~~~~~~~~

La rama de desarrollo actual ofrece su código fuente bajo la `licencia PolyForm Noncommercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. El uso comercial requiere una licencia aparte del titular de los derechos de autor. Las versiones publicadas hasta spaCR 1.4.9.9 siguen disponibles bajo la licencia MIT que acompañaba a esas versiones.

Tutoriales
~~~~~~~~~~

La `biblioteca interactiva de tutoriales de spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contiene recorridos narrados y subtitulados de la instalación y de cada flujo de trabajo de la aplicación, en ocho idiomas.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Un cribado CRISPR agrupado basado en imágenes identifica EAF1 como modulador de la subversión de ESCRT por *T. gondii*.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343317>`_
