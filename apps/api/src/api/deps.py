"""Dependencias externas de la API."""

import keras
from prue_model import artifact_path, list_specs, load_model, models_dir


def load_all_models() -> dict[str, keras.Model]:
    """Carga los artefactos de todos los modelos registrados.

    Es todo o nada: si falta alguno, la API no arranca. Un servicio que solo
    responde 503 en una ruta es de los que se despliegan sin que nadie lo note.
    El barrido previo reporta TODOS los faltantes de una vez, y lo hace antes
    de pagar el costo de cargar los que si estan.
    """
    base = models_dir(required=True)
    specs = list_specs()

    missing = [spec.name for spec in specs if not artifact_path(spec, base).is_file()]
    if missing:
        raise RuntimeError(
            f"Faltan artefactos en {base}: {', '.join(missing)}. "
            "Corre 'python -m trainer --all' primero."
        )

    return {spec.name: load_model(artifact_path(spec, base)) for spec in specs}
