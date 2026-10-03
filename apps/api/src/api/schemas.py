"""Esquemas de entrada y salida de la API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Cuerpo esperado por POST /predict/{name}."""

    features: list[float] = Field(..., min_length=1)


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
