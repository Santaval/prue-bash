"""Aplicacion FastAPI para servir predicciones de varios modelos."""

from contextlib import asynccontextmanager

import keras
from fastapi import FastAPI, HTTPException
from prue_model import (InvalidFeaturesError, ModelNotFoundError, get_spec,
                        list_specs, predict)

from .deps import load_all_models
from .schemas import ModelInfo, PredictRequest, PredictResponse

_state: dict[str, keras.Model] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carga todos los modelos una sola vez al arrancar el servidor."""
    _state.update(load_all_models())
    yield
    _state.clear()


app = FastAPI(
    title="API de Inferencia",
    description="Endpoints HTTP para predecir con los modelos registrados",
    lifespan=lifespan,
)


@app.get("/")
def index() -> dict:
    """Health check."""
    return {"message": "API activa. Usa GET /models y POST /predict/{name}."}


@app.get("/models", response_model=list[ModelInfo])
def get_models() -> list[ModelInfo]:
    """Lista los modelos registrados y si su artefacto esta cargado."""
    return [
        ModelInfo(
            name=spec.name,
            input_features=spec.input_features,
            feature_names=list(spec.feature_names),
            description=spec.description,
            loaded=spec.name in _state,
        )
        for spec in list_specs()
    ]


@app.post("/predict/{name}", response_model=PredictResponse)
def post_predict(name: str, data: PredictRequest) -> PredictResponse:
    """Devuelve la prediccion del modelo indicado para un vector de entradas."""
    # El 404 va antes que el 503: un nombre mal escrito es culpa del cliente y
    # no deberia reportar el estado del servidor.
    try:
        spec = get_spec(name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    model = _state.get(name)
    if model is None:
        raise HTTPException(
            status_code=503, detail=f"El modelo '{name}' no esta cargado."
        )

    try:
        value = predict(model, data.features, spec)
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(model=name, input=data.features, prediction=value)
