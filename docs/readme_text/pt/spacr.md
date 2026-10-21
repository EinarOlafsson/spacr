# spaCR

<strong>Análise espacial de fenótipos em triagens CRISPR.</strong>

O spaCR segmenta e mede células individuais em imagens de microscopia, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada.

Os módulos de segmentação, medição, anotação e classificação também podem ser usados sem um componente de sequenciamento.

Make Masks corrige máscaras de segmentação e anota retângulos independentes com rótulos de classe usando a ferramenta <strong>Box</strong> para exportação YOLO. Os retângulos mantêm seus próprios rótulos e histórico sem alterar as imagens ou máscaras de origem.

Consulte o [guia de funcionalidades](<../../source/features.rst>) para conhecer cada ferramenta.

Imagens, máscaras e recortes são arquivos em disco. Medições, anotações e previsões do modelo são tabelas do banco de dados SQLite ` measurements.db ` do projeto.

É executado como um aplicativo de desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster.
