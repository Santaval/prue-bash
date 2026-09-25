"""Entrena modelos del registro y escribe los artefactos en MODELS_DIR."""

from pathlib import Path

import keras
from prue_model import ModelSpec, artifact_path, list_specs


def train_spec(spec: ModelSpec, base_dir: Path) -> Path:
    """Entrena un modelo y guarda su artefacto. Devuelve la ruta escrita."""
    # La semilla se fija ANTES de construir: la inicializacion de los pesos
    # tambien consume el RNG, asi que el orden importa. Reponerla por modelo
    # hace que --all produzca los mismos pesos que entrenar uno por uno.
    #
    # Ojo: los bytes del .keras igual varian entre corridas (es un zip con
    # metadata), asi que no compares el artefacto por hash, compara los pesos.
    keras.utils.set_random_seed(spec.seed)

    model = spec.build()
    features, targets = spec.dataset()
    model.fit(features, targets, epochs=spec.epochs, verbose=0)

    destination = artifact_path(spec, base_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    return destination


def train_all(base_dir: Path) -> list[Path]:
    """Entrena todos los modelos registrados, en orden estable por nombre."""
    return [train_spec(spec, base_dir) for spec in list_specs()]
