## Biblioteca de modelos

O spaCR inclui um catálogo de modelos treinados e os baixa quando necessário. Abra <strong>Model Zoo</strong> na tela inicial para explorar e instalar modelos, ou indique uma chave no arquivo de configurações, como ` pathogen_model: toxoplasma_pv_v1 `. O modelo é baixado e verificado pela soma de verificação na primeira vez em que é necessário. Cada entrada publicada contém uma soma SHA-256. Entradas sem essa soma são recusadas, pois, sem ela, não seria possível distinguir um arquivo de modelo incompleto ou substituído do original.
