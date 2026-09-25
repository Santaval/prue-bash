"""Esquemas de entrada y salida de la API."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Cuerpo esperado por POST /predict."""

    x: float


class PredictResponse(BaseModel):
    """Respuesta de POST /predict."""

    input: float
    prediction: float
