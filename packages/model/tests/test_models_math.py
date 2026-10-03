"""Pruebas del modelo 'math'. Entrenan de verdad: marcadas como lentas."""

import numpy as np
import pytest
from prue_model import get_spec, load_model, predict
from trainer.train import train_spec

pytestmark = pytest.mark.slow

SPEC = get_spec("math")


@pytest.fixture(scope="module", name="trained")
def fixture_trained(tmp_path_factory):
    """Entrena 'math' una vez para todo el modulo."""
    base = tmp_path_factory.mktemp("math")
    return load_model(train_spec(SPEC, base))


def test_dataset_matches_the_documented_relation():
    """Los datos siguen siendo y = 2x - 1."""
    features, targets = SPEC.dataset()
    np.testing.assert_allclose(targets, 2 * features - 1)


def test_dataset_returns_copies():
    """Mutar lo devuelto no puede corromper los datos del modulo."""
    first, _ = SPEC.dataset()
    first[0] = 999.0
    second, _ = SPEC.dataset()
    assert second[0] != 999.0


def test_trained_weights_match_the_documented_values(trained):
    """Paridad con el comportamiento previo al refactor: 2.0 y -0.99."""
    kernel, bias = trained.get_weights()
    np.testing.assert_allclose(kernel.ravel(), [2.0], atol=0.01)
    np.testing.assert_allclose(bias, [-0.99], atol=0.01)


def test_prediction_follows_the_learned_relation(trained):
    """Para x = 10 la relacion aprendida da ~19."""
    assert predict(trained, [10.0], SPEC).value == pytest.approx(19.0, abs=0.1)
