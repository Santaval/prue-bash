"""Contrato HTTP de M07, sin entrenar modelos ni usar PostgreSQL."""

from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pytest
from pydantic import ValidationError
from api.schemas import DiabetesPredictRequest


@pytest.fixture(name="fake_diabetes_api")
def fixture_fake_diabetes_api(monkeypatch):
    """Inyecta un clasificador constante y una persistencia falsa."""
    from api import main

    class FixedModel:
        def predict(self, batch, verbose=0):  # pylint: disable=unused-argument
            assert batch.shape == (1, 8)
            return np.array([[0.82]])

    saved_records = []

    def save_record(record):
        saved_records.append(record)
        return {
            **record,
            "id": UUID("9e5b1296-674e-47bd-b3fd-7cb78a8a85e8"),
            "created_at": datetime(2026, 11, 22, tzinfo=UTC),
        }

    monkeypatch.setitem(main._state, "diabetes", FixedModel())
    monkeypatch.setattr(main, "save_prediction", save_record)
    return main, saved_records


def _valid_request() -> dict:
    return {
        "gender": "Female",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 27.5,
        "HbA1c_level": 6.1,
        "blood_glucose_level": 140,
    }


def test_diabetes_handler_maps_fields_predicts_and_persists(fake_diabetes_api):
    main, saved_records = fake_diabetes_api
    request = main.DiabetesPredictRequest(**_valid_request())

    body = main.post_predict_diabetes(request).model_dump(mode="json")
    assert body["model"] == "diabetes"
    assert body["input"] == [0.0, 45.0, 0.0, 0.0, 4.0, 27.5, 6.1, 140.0]
    assert body["prediction"] == pytest.approx(0.82)
    assert body["label"] == "diabetes"
    assert body["probabilities"] == {
        "no_diabetes": pytest.approx(0.18),
        "diabetes": pytest.approx(0.82),
    }
    assert body["execution_id"] == "9e5b1296-674e-47bd-b3fd-7cb78a8a85e8"
    assert body["created_at"] == "2026-11-22T00:00:00Z"
    assert saved_records[0]["input_data"] == _valid_request()


@pytest.mark.parametrize(
    "changes",
    [
        {"gender": "Unknown"},
        {"hypertension": 2},
        {"heart_disease": True},
        {"smoking_history": "unknown"},
        {"age": -1},
        {"HbA1c_level": float("nan")},
        {"extra": "field"},
    ],
)
def test_diabetes_schema_rejects_invalid_contract(fake_diabetes_api, changes):
    _, saved_records = fake_diabetes_api

    with pytest.raises(ValidationError):
        DiabetesPredictRequest(**{**_valid_request(), **changes})
    assert saved_records == []


def test_diabetes_schema_rejects_missing_field(fake_diabetes_api):
    _, saved_records = fake_diabetes_api
    request = _valid_request()
    del request["gender"]

    with pytest.raises(ValidationError):
        DiabetesPredictRequest(**request)
    assert saved_records == []
