## Biblioteca de modelos

spaCR incluye un catálogo de modelos entrenados y los descarga cuando se necesitan. Abra <strong>Model Zoo</strong> desde la pantalla de inicio para explorar e instalar modelos, o indique una clave en un archivo de configuración, como ` pathogen_model: toxoplasma_pv_v1 `. El modelo se descarga y se verifica mediante su suma de comprobación la primera vez que se necesita. Cada entrada publicada incluye una suma SHA-256. Las entradas sin ella se rechazan, ya que, de lo contrario, no se podría distinguir un archivo de modelo incompleto o sustituido del original.
