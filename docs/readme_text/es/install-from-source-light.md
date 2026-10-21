### Instalación desde el código fuente (ligera)

Los colaboradores necesitan el historial de versiones; para ejecutar spaCR, elija una de las opciones siguientes. Mediciones de ` nightly ` en ` 05302fd5c ` el 2026-10-07, realizadas con ` packaging/measure_clone_forms.sh `.

Clon superficial: 2048 MB descargados en 140 s. Solo se conserva el último commit, por lo que ` git log `, ` git blame ` y ` git bisect ` no tienen acceso al historial anterior. ` git pull ` funciona, pero el clon sigue siendo superficial; ` git fetch --unshallow ` obtiene el historial completo.
