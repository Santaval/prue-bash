"""Esquemas de entrada y salida de la API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Cuerpo esperado por POST /predict/{name}."""

    features: list[float] = Field(..., min_length=1)


class DiabetesPredictRequest(BaseModel):
    """Entrada nombrada del modelo M07, con categorías del conjunto de datos."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male", "Other"]
    age: float = Field(ge=0, allow_inf_nan=False)
    hypertension: int = Field(strict=True, ge=0, le=1)
    heart_disease: int = Field(strict=True, ge=0, le=1)
    smoking_history: Literal[
        "No Info", "current", "ever", "former", "never", "not current"
    ]
    bmi: float = Field(ge=0, allow_inf_nan=False)
    HbA1c_level: float = Field(ge=0, allow_inf_nan=False)
    blood_glucose_level: float = Field(ge=0, allow_inf_nan=False)

    def to_features(self) -> list[float]:
        """Codifica categorías con el orden aprendido por el entrenamiento."""
        gender_codes = {"Female": 0, "Male": 1, "Other": 2}
        smoking_codes = {
            "No Info": 0,
            "current": 1,
            "ever": 2,
            "former": 3,
            "never": 4,
            "not current": 5,
        }
        return [
            float(gender_codes[self.gender]),
            self.age,
            float(self.hypertension),
            float(self.heart_disease),
            float(smoking_codes[self.smoking_history]),
            self.bmi,
            self.HbA1c_level,
            self.blood_glucose_level,
        ]


class PredictResponse(BaseModel):
    """Respuesta de POST /predict/{name}.

    `prediction` siempre viene: es el valor estimado en una regresion, la
    probabilidad de la clase positiva en un binario y el indice de la clase
    ganadora en un multiclase. Los dos campos restantes solo los llenan los
    clasificadores; en una regresion quedan en null.
    """

    model: str
    input: list[float]
    prediction: float
    label: str | None = None
    probabilities: dict[str, float] | None = None
    execution_id: UUID | None = None
    created_at: datetime | None = None


class ModelInfo(BaseModel):
    """Entrada de GET /models.

    Incluye `task` y `class_names` para que un cliente sepa como leer una
    prediccion antes de pedirla.
    """

    name: str
    input_features: int
    feature_names: list[str]
    description: str
    loaded: bool
    task: str
    class_names: list[str]
