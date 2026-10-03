"""Pruebas del registro de modelos."""

import dataclasses

import numpy as np
import pytest
from prue_model import (DuplicateModelError, ModelNotFoundError, ModelSpec,
                        get_spec, list_names, list_specs, register)


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset trivial para specs de prueba."""
    return np.zeros((2, 1)), np.zeros(2)


def _spec(name: str = "dummy", **overrides) -> ModelSpec:
    """Arma un spec valido, con campos sustituibles."""
    kwargs = {
        "name": name,
        "input_features": 1,
        "feature_names": ("a",),
        "build": lambda: None,
        "dataset": _dataset,
        "epochs": 1,
    }
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


def test_register_stores_and_returns_build_unchanged(registry_snapshot):
    """El decorador registra el spec y devuelve la funcion intacta."""

    def build():
        """Constructor de prueba."""
        return "modelo"

    decorated = register(
        name="tmp", input_features=1, feature_names=("a",),
        dataset=_dataset, epochs=1,
    )(build)

    assert decorated is build, "register debe devolver la funcion sin envolver"
    assert registry_snapshot.get_spec("tmp").build() == "modelo"


def test_register_rejects_duplicate_names(registry_snapshot):
    """Registrar dos veces el mismo nombre es un error explicito."""
    decorator = register(
        name="tmp", input_features=1, feature_names=("a",),
        dataset=_dataset, epochs=1,
    )
    decorator(lambda: None)
    with pytest.raises(DuplicateModelError, match="ya esta registrado"):
        decorator(lambda: None)
    assert registry_snapshot is not None


def test_get_spec_unknown_lists_available_names():
    """El mensaje del error orienta al llamador."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        get_spec("no-existe")
    message = str(excinfo.value)
    assert "no-existe" in message
    assert "math" in message and "climate" in message


def test_get_spec_message_has_no_stray_quotes():
    """Regresion: heredar de KeyError metia comillas en el detalle del 404."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        get_spec("nope")
    assert not str(excinfo.value).startswith("'")


def test_list_specs_is_sorted_and_immutable():
    """El orden es estable por nombre y el resultado no permite mutar nada."""
    names = [spec.name for spec in list_specs()]
    assert names == sorted(names)
    assert isinstance(list_specs(), tuple)
    assert list_names() == tuple(sorted(list_names()))


def test_autoload_registers_every_model_file():
    """Un `import prue_model` pelado basta para que el registro este lleno.

    Esta es la prueba de que "agregar un modelo = agregar un archivo".
    """
    assert "math" in list_names()
    assert "climate" in list_names()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_features", 0),
        ("input_features", -1),
        ("epochs", 0),
    ],
)
def test_spec_rejects_invalid_numbers(field, value):
    """Un spec mal formado falla al importarse, no en la primera peticion."""
    with pytest.raises(ValueError):
        _spec(**{field: value})


def test_spec_rejects_feature_name_mismatch():
    """input_features y feature_names no pueden divergir entre si."""
    with pytest.raises(ValueError, match="pero nombra"):
        _spec(input_features=2, feature_names=("a",))


def test_spec_rejects_a_single_class():
    """Una sola clase no es ni regresion ni clasificacion."""
    with pytest.raises(ValueError, match="una sola clase"):
        _spec(class_names=("spam",))


def test_spec_rejects_duplicate_class_names():
    """Los nombres son llaves del diccionario de probabilidades: sin colisiones."""
    with pytest.raises(ValueError, match="repite nombres de clase"):
        _spec(class_names=("a", "b", "a"))


@pytest.mark.parametrize(
    ("class_names", "task", "units"),
    [
        ((), "regression", 1),
        (("no", "si"), "binary", 1),
        (("baja", "media", "alta"), "multiclass", 3),
    ],
)
def test_task_and_output_units_derive_from_class_names(class_names, task, units):
    """El contrato de salida se deriva, no se declara dos veces."""
    spec = _spec(class_names=class_names)
    assert spec.task == task
    assert spec.output_units == units


def test_spec_is_frozen():
    """Un spec es una declaracion: no se muta."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _spec().name = "otro"


@pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
def test_every_spec_builds_its_declared_shape(spec):
    """Contrato que cubre gratis a todo modelo futuro."""
    model = spec.build()
    assert model.input_shape == (None, spec.input_features)
    assert model.output_shape == (None, spec.output_units)


@pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
def test_every_dataset_matches_its_declared_shape(spec):
    """Los datos del spec tienen tantas columnas como entradas declaradas."""
    features, targets = spec.dataset()
    columns = np.asarray(features).reshape(len(targets), -1).shape[1]
    assert columns == spec.input_features
