"""El entrenador realmente aprende: los pesos recuperan los coeficientes.

Corre sobre los modelos de prueba de conftest.py, cuyos objetivos no tienen
ruido, asi que se pueden afirmar los coeficientes y no solo "algo bajo".
"""

import numpy as np
import pytest
from conftest import LINEAR_B, LINEAR_W, PLANE_B, PLANE_W
from prue_model import get_spec, load_model, predict
from trainer.train import train_spec

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("name", "weights", "bias"),
    [("linear", [LINEAR_W], LINEAR_B), ("plane", list(PLANE_W), PLANE_B)],
)
def test_trained_weights_recover_the_true_coefficients(tmp_path, name, weights, bias):
    """Una entrada y varias: ambos caminos convergen a la relacion real."""
    spec = get_spec(name)
    model = load_model(train_spec(spec, tmp_path))

    kernel, learned_bias = model.get_weights()
    np.testing.assert_allclose(kernel.ravel(), weights, atol=0.15)
    np.testing.assert_allclose(learned_bias, [bias], atol=0.15)

    features = [0.5] * spec.input_features
    expected = float(np.dot(features, weights) + bias)
    assert predict(model, features, spec).value == pytest.approx(expected, abs=0.15)
