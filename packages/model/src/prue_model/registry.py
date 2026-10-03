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
class ModelSpec:  # pylint: disable=too-many-instance-attributes
    """Declaracion completa de un modelo: arquitectura, datos y entrenamiento.

    Son muchos atributos a proposito: es un contenedor de declaraciones, no una
    clase con comportamiento. Partirlo dispersaria en dos archivos lo que un
    modelo declara de si mismo.

    Los dos contratos del modelo viven aqui, en un unico lugar por modelo, y
    los consumen tanto el entrenador como la API: el de entrada
    (`input_features`) y el de salida (`class_names`).

    `class_names` vacio significa regresion. Con dos nombres el modelo es un
    clasificador binario y con tres o mas, multiclase; de ahi se derivan `task`
    y `output_units` en vez de declararlos por separado, para que no puedan
    contradecirse entre si.
    """

    name: str
    input_features: int
    feature_names: tuple[str, ...]
    build: Callable[[], keras.Model]
    dataset: Callable[[], Dataset]
    epochs: int
    seed: int = 42
    description: str = ""
    class_names: tuple[str, ...] = ()

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
        if len(self.class_names) == 1:
            raise ValueError(
                f"'{self.name}' declara una sola clase. Usa cero nombres para "
                "una regresion o dos o mas para una clasificacion."
            )
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(
                f"'{self.name}' repite nombres de clase: cada uno es una llave "
                "del diccionario de probabilidades y una colision perderia una."
            )

    @property
    def task(self) -> str:
        """Tarea del modelo, derivada de cuantas clases declara."""
        if not self.class_names:
            return "regression"
        return "binary" if len(self.class_names) == 2 else "multiclass"

    @property
    def output_units(self) -> int:
        """Unidades que debe tener la capa de salida de `build`."""
        return 1 if self.task in ("regression", "binary") else len(self.class_names)


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
    class_names: Sequence[str] = (),
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
            class_names=tuple(class_names),
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
