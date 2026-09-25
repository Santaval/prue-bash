"""Registro de modelos: declaracion, alta y consulta.

Cada modelo se declara una sola vez en un archivo bajo `models/` y queda
disponible para el entrenador y la API sin tocar ninguno de los dos.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import keras
import numpy as np

from .errors import DuplicateModelError, ModelNotFoundError

Dataset = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Declaracion completa de un modelo: arquitectura, datos y entrenamiento.

    El contrato de forma de entrada vive aqui (`input_features`), en un unico
    lugar por modelo, y lo consumen tanto el entrenador como la API.
    """

    name: str
    input_features: int
    feature_names: tuple[str, ...]
    build: Callable[[], keras.Model]
    dataset: Callable[[], Dataset]
    epochs: int
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        """Valida el spec al construirlo, es decir al importar el modulo."""
        if self.input_features < 1:
            raise ValueError(f"input_features debe ser >= 1 en '{self.name}'.")
        if len(self.feature_names) != self.input_features:
            raise ValueError(
                f"'{self.name}' declara {self.input_features} entradas "
                f"pero nombra {len(self.feature_names)}."
            )
        if self.epochs < 1:
            raise ValueError(f"epochs debe ser >= 1 en '{self.name}'.")


_REGISTRY: dict[str, ModelSpec] = {}

BuildFn = Callable[[], keras.Model]


def register(
    *,
    name: str,
    input_features: int,
    feature_names: Sequence[str],
    dataset: Callable[[], Dataset],
    epochs: int,
    seed: int = 42,
    description: str = "",
) -> Callable[[BuildFn], BuildFn]:
    """Decorador que registra la funcion decorada como constructor del modelo.

    Devuelve la funcion intacta, asi que `build` sigue siendo importable y
    testeable por separado.
    """

    def decorator(build: BuildFn) -> BuildFn:
        if name in _REGISTRY:
            raise DuplicateModelError(f"El modelo '{name}' ya esta registrado.")
        _REGISTRY[name] = ModelSpec(
            name=name,
            input_features=input_features,
            feature_names=tuple(feature_names),
            build=build,
            dataset=dataset,
            epochs=epochs,
            seed=seed,
            description=description,
        )
        return build

    return decorator


def get_spec(name: str) -> ModelSpec:
    """Devuelve el spec registrado con ese nombre."""
    spec = _REGISTRY.get(name)
    if spec is None:
        available = ", ".join(sorted(_REGISTRY)) or "(ninguno)"
        raise ModelNotFoundError(
            f"El modelo '{name}' no existe. Disponibles: {available}."
        )
    return spec


def list_specs() -> tuple[ModelSpec, ...]:
    """Specs registrados, ordenados por nombre para que la salida sea estable."""
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))


def list_names() -> tuple[str, ...]:
    """Nombres registrados, ordenados."""
    return tuple(sorted(_REGISTRY))


def snapshot() -> dict[str, ModelSpec]:
    """Copia del registro. Pensado para que las pruebas puedan restaurarlo."""
    return dict(_REGISTRY)


def restore(saved: dict[str, ModelSpec]) -> None:
    """Restaura el registro a partir de un `snapshot`. Solo para pruebas."""
    _REGISTRY.clear()
    _REGISTRY.update(saved)
