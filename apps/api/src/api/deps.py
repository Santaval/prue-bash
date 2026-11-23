"""Dependencias externas de la API."""

import keras
from prue_model import (artifact_path, get_spec, list_specs, load_model,
                        models_dir)


def load_models(names: tuple[str, ...] | None = None) -> dict[str, keras.Model]:
    """Carga todos los modelos registrados o solo los nombres solicitados.

    Es todo o nada: si falta alguno, la API no arranca. Un servicio que solo
    responde 503 en una ruta es de los que se despliegan sin que nadie lo note.
    El barrido previo reporta TODOS los faltantes de una vez, y lo hace antes
    de pagar el costo de cargar los que si estan.
    """
    base = models_dir(required=True)
    specs = tuple(get_spec(name) for name in names) if names else list_specs()

    missing = [spec.name for spec in specs if not artifact_path(spec, base).is_file()]
    if missing:
        raise RuntimeError(
            f"Faltan artefactos en {base}: {', '.join(missing)}. "
            "Entrena primero los modelos seleccionados."
        )

    return {spec.name: load_model(artifact_path(spec, base)) for spec in specs}


def load_all_models() -> dict[str, keras.Model]:
    """Compatibilidad: carga todos los modelos registrados."""
    return load_models()
