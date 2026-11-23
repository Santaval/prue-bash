"""Flujo de negocio completo: entrenar, servir y predecir en un solo proceso.

Es la unica prueba que ejercita entrenador + registro + artefactos + API
juntos, o sea justo lo que el monorepo existe para mantener sincronizado.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from prue_model import list_names
from trainer.train import train_all

pytestmark = pytest.mark.slow

# `diabetes` no usa {"features": [...]}: su contrato son campos nombrados.
_DIABETES_PAYLOAD = {
    "gender": "Female", "age": 54, "hypertension": 0, "heart_disease": 0,
    "smoking_history": "never", "bmi": 27.3, "HbA1c_level": 6.6,
    "blood_glucose_level": 140,
}


def _payload_for(entry):
    """Cuerpo valido para el contrato HTTP de cada modelo."""
    if entry["name"] == "diabetes":
        return _DIABETES_PAYLOAD
    return {"features": [0.5] * entry["input_features"]}


def test_train_then_serve(tmp_path, monkeypatch):
    """Entrenar todo y servirlo deja cada modelo utilizable por HTTP."""
    written = train_all(tmp_path)
    assert sorted(path.name for path in written) == sorted(
        f"{name}.keras" for name in list_names()
    )

    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    from api import main
    from api.main import app

    # La ruta de diabetes guarda cada ejecucion; aqui no hay base de datos.
    monkeypatch.setattr(main, "save_prediction", lambda row: {"id": str(uuid.uuid4())})

    with TestClient(app) as client:
        catalogue = client.get("/models").json()
        assert all(entry["loaded"] for entry in catalogue)

        for entry in catalogue:
            response = client.post(
                f"/predict/{entry['name']}", json=_payload_for(entry)
            )
            assert response.status_code == 200
            prediction = response.json()["prediction"]
            assert isinstance(prediction, float)
            assert prediction == prediction  # descarta NaN
