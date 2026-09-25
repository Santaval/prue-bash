"""Flujo de negocio completo: entrenar, servir y predecir en un solo proceso.

Es la unica prueba que ejercita entrenador + registro + artefactos + API
juntos, o sea justo lo que el monorepo existe para mantener sincronizado.
"""

import pytest
from fastapi.testclient import TestClient
from prue_model import list_names
from trainer.train import train_all

pytestmark = pytest.mark.slow


def test_train_then_serve(tmp_path, monkeypatch):
    """Entrenar todo y servirlo deja cada modelo utilizable por HTTP."""
    written = train_all(tmp_path)
    assert sorted(path.name for path in written) == sorted(
        f"{name}.keras" for name in list_names()
    )

    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    from api.main import app

    with TestClient(app) as client:
        catalogue = client.get("/models").json()
        assert all(entry["loaded"] for entry in catalogue)

        for entry in catalogue:
            features = [0.5] * entry["input_features"]
            response = client.post(
                f"/predict/{entry['name']}", json={"features": features}
            )
            assert response.status_code == 200
            prediction = response.json()["prediction"]
            assert isinstance(prediction, float)
            assert prediction == prediction  # descarta NaN
