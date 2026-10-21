# spaCR

<strong>Análisis espacial del fenotipo en cribados CRISPR.</strong>

spaCR segmenta y mide células individuales en imágenes de microscopía, integra los fenotipos por objeto con la abundancia de guías derivada de la secuenciación y estima qué genes están asociados con cambios fenotípicos. A partir de imágenes de placas y lecturas FASTQ, produce mediciones por objeto, clasificadores entrenados, estimaciones del efecto por guía y por gen y una lista ordenada de resultados.

Los módulos de segmentación, medición, anotación y clasificación también se pueden usar sin un componente de secuenciación.

Make Masks corrige máscaras de segmentación y anota rectángulos independientes con etiquetas de clase mediante la herramienta <strong>Box</strong> para exportarlos a YOLO. Los recuadros conservan sus propias etiquetas e historial sin modificar las imágenes ni las máscaras de origen.

Consulte la [guía de funciones](<../../source/features.rst>) para conocer cada herramienta.

Las imágenes, máscaras y recortes son archivos en disco. Las mediciones, anotaciones y predicciones del modelo son tablas de la base de datos SQLite ` measurements.db ` del proyecto.

Se ejecuta como una aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster.
