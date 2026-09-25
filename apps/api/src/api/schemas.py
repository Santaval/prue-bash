"""Esquemas de entrada y salida de la API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Cuerpo esperado por POST /predict/{name}."""

    features: list[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Respuesta de POST /predict/{name}."""

    model: str
    input: list[float]
    prediction: float


class ModelInfo(BaseModel):
    """Entrada de GET /models."""

    name: str
    input_features: int
    feature_names: list[str]
    description: str
    loaded: bool
