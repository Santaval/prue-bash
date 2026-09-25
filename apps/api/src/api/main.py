"""Aplicacion FastAPI para servir predicciones."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prue_model import predict

from .deps import load_configured_model
from .schemas import PredictRequest, PredictResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carga el modelo una sola vez al arrancar el servidor."""
    _state["model"] = load_configured_model()
    yield
    _state.clear()


app = FastAPI(
    title="API de Inferencia",
    description="Endpoint HTTP para realizar predicciones con el modelo entrenado",
    lifespan=lifespan,
)


@app.get("/")
def index() -> dict:
    """Health check."""
    return {"message": "API activa. Usa POST /predict para obtener predicciones."}


@app.post("/predict", response_model=PredictResponse)
def post_predict(data: PredictRequest) -> PredictResponse:
    """Devuelve la prediccion del modelo para un valor de entrada."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no esta disponible.")

    try:
        value = predict(model, data.x)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(input=data.x, prediction=value)
