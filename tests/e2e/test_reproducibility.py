"""Reproducibilidad del entrenamiento.

La semilla fija hace que los pesos entrenados sean identicos entre corridas.
Los bytes del .keras NO lo son: es un zip con metadata variable, asi que la
reproducibilidad se afirma cargando el artefacto y comparando get_weights(),
nunca comparando hashes del archivo.
"""

import numpy as np
import pytest
from prue_model import get_spec, list_specs, load_model
from trainer.train import train_all, train_spec

pytestmark = pytest.mark.slow


def test_training_twice_yields_identical_weights(tmp_path):
    """Misma semilla, mismos pesos."""
    spec = get_spec("math")
    first = load_model(train_spec(spec, tmp_path / "a")).get_weights()
    second = load_model(train_spec(spec, tmp_path / "b")).get_weights()

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("spec", list_specs(), ids=lambda spec: spec.name)
def test_all_matches_training_each_model_alone(tmp_path, spec):
    """Atrapa a quien saque set_random_seed del bucle por modelo.

    Sin reponer la semilla dentro de train_spec, el segundo modelo de --all
    arrancaria con el RNG que dejo el fit del primero y sus pesos no
    coincidirian con los de entrenarlo por separado.
    """
    batch_dir = tmp_path / "all"
    train_all(batch_dir)
    solo = train_spec(spec, tmp_path / "solo")

    batched = load_model(batch_dir / f"{spec.name}.keras").get_weights()
    alone = load_model(solo).get_weights()
    for left, right in zip(batched, alone):
        np.testing.assert_array_equal(left, right)
