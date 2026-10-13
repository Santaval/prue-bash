"""Fixtures compartidas por todas las suites."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prue_model import build_dense_regressor, registry
from trainer.train import train_all

# --- Modelos de prueba ------------------------------------------------------
#
# El paquete `prue_model.models` no trae modelos de produccion, pero la API, el
# entrenador y las pruebas e2e necesitan un registro con algo adentro. Estos
# dos se registran al cargar este conftest, es decir ANTES de recolectar las
# pruebas, asi que tambien los ven los `parametrize` sobre `list_specs()`.
#
# Cubren los dos caminos de entrada: una sola variable y varias. Sus objetivos
# no tienen ruido, asi que las pruebas pueden afirmar los coeficientes.

LINEAR_W, LINEAR_B = 2.0, -1.0
PLANE_W, PLANE_B = (2.0, -1.5, 0.5), 0.25


def _linear_dataset() -> tuple[np.ndarray, np.ndarray]:
    """y = 2x - 1 sobre seis puntos fijos."""
    features = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    return features, LINEAR_W * features + LINEAR_B


def _plane_dataset() -> tuple[np.ndarray, np.ndarray]:
    """y = 2a - 1.5b + 0.5c + 0.25 con entradas normalizadas en [0, 1)."""
    rng = np.random.default_rng(7)
    features = rng.uniform(0.0, 1.0, size=(256, 3))
    return features, features @ np.array(PLANE_W) + PLANE_B


registry.register(
    name="linear",
    input_features=1,
    feature_names=("x",),
    dataset=_linear_dataset,
    epochs=500,
    description="Modelo de prueba: y = 2x - 1",
)(lambda: build_dense_regressor(1))

registry.register(
    name="plane",
    input_features=3,
    feature_names=("a", "b", "c"),
    dataset=_plane_dataset,
    epochs=300,
    description="Modelo de prueba: y = 2a - 1.5b + 0.5c + 0.25",
)(lambda: build_dense_regressor(3))


@pytest.fixture(name="registry_snapshot")
def fixture_registry_snapshot():
    """Restaura el registro para que un spec de prueba no contamine el resto."""
    saved = registry.snapshot()
    yield registry
    registry.restore(saved)


@pytest.fixture(scope="session", name="trained_models_dir")
def fixture_trained_models_dir(tmp_path_factory):
    """Entrena todos los modelos UNA vez por sesion y devuelve el directorio."""
    base = tmp_path_factory.mktemp("models")
    train_all(base)
    return base


@pytest.fixture(name="client")
def fixture_client(trained_models_dir, monkeypatch):
    """TestClient con lifespan ejecutado y MODELS_DIR apuntando al tmp.

    Dos mecanicas que no hay que deshacer:
    1. monkeypatch.setenv va ANTES de entrar al contexto, porque lifespan lee
       el entorno al arrancar.
    2. TestClient(app) usado FUERA de un `with` no ejecuta lifespan y entonces
       todas las peticiones responden 503. Es la causa numero uno de fallos
       confusos aqui. Ver test_routes.py, que lo hace a proposito una vez.
    """
    monkeypatch.setenv("MODELS_DIR", str(trained_models_dir))
    from api.main import app  # import tardio: despues de fijar el entorno

    with TestClient(app) as test_client:
        yield test_client


class _ConstantModel:
    """Doble de prueba que siempre devuelve la misma fila de salida."""

    def __init__(self, row):
        self._row = np.array([row], dtype=float)

    def predict(self, batch, verbose=0):  # pylint: disable=unused-argument
        """Imita keras.Model.predict devolviendo un array (1, n)."""
        del batch
        return self._row


@pytest.fixture(name="fake_model")
def fixture_fake_model():
    """Doble de prueba con .predict, para validar sin cargar Keras."""

    class _FakeModel:
        """Devuelve la suma del batch, suficiente para probar el envoltorio."""

        def predict(self, batch, verbose=0):  # pylint: disable=unused-argument
            """Imita keras.Model.predict devolviendo un array (1, 1)."""
            return np.array([[float(np.asarray(batch).sum())]])

    return _FakeModel()


@pytest.fixture(name="fake_binary_model")
def fixture_fake_binary_model():
    """Clasificador binario falso: una sigmoide que ya decidio 0.8."""
    return _ConstantModel([0.8])


@pytest.fixture(name="fake_multiclass_model")
def fixture_fake_multiclass_model():
    """Clasificador multiclase falso: softmax con la clase 1 ganando."""
    return _ConstantModel([0.1, 0.7, 0.2])
