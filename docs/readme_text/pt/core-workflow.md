### Fluxo de trabalho principal

O fluxo de trabalho principal compreende seis módulos:

- <strong>Mask</strong> segmenta células, núcleos, patógenos e organelas com Cellpose.
- <strong>Measure</strong> grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos.
- <strong>Annotate</strong> rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo.
- <strong>Classify</strong> treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados.
- <strong>Map Barcodes</strong> associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura.
- <strong>Regression</strong> estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.
