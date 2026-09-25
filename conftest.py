"""Fixtures compartidas por todas las suites."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prue_model import registry
from trainer.train import train_all


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


@pytest.fixture(name="fake_model")
def fixture_fake_model():
    """Doble de prueba con .predict, para validar sin cargar Keras."""

    class _FakeModel:
        """Devuelve la suma del batch, suficiente para probar el envoltorio."""

        def predict(self, batch, verbose=0):  # pylint: disable=unused-argument
            """Imita keras.Model.predict devolviendo un array (1, 1)."""
            return np.array([[float(np.asarray(batch).sum())]])

    return _FakeModel()
