### Flujo de trabajo principal

El flujo de trabajo principal comprende seis módulos:

- <strong>Mask</strong> segmenta células, núcleos, patógenos y orgánulos con Cellpose.
- <strong>Measure</strong> guarda en SQLite características morfológicas, de intensidad, textura, espaciales y de colocalización, junto con recortes de objetos.
- <strong>Annotate</strong> etiqueta recortes en una cuadrícula controlada con el teclado y admite colas de aprendizaje activo.
- <strong>Classify</strong> entrena modelos basados en imágenes o mediciones y registra con cada punto de control el rendimiento en los datos reservados.
- <strong>Map Barcodes</strong> asigna las lecturas FASTQ a los pocillos y los gRNA, con controles de calidad de abundancia, colisiones y cobertura.
- <strong>Regression</strong> estima los efectos de guías, genes, condiciones y controles con familias de modelos adecuadas para respuestas continuas, fraccionarias y de recuento.
