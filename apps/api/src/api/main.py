"""Aplicacion FastAPI para servir predicciones de varios modelos."""

import os
from contextlib import asynccontextmanager

import keras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prue_model import (InvalidFeaturesError, ModelNotFoundError, get_spec,
                        list_specs, predict)

from .deps import load_models
from .persistence import save_prediction
from .schemas import (DiabetesPredictRequest, ModelInfo, PredictRequest,
                      PredictResponse)

_state: dict[str, keras.Model] = {}

_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _cors_origins() -> list[str]:
    """Origenes permitidos por CORS, leidos de `API_CORS_ORIGINS`."""
    raw = os.getenv("API_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carga modelos al iniciar, salvo que se active el modo sin modelos."""
    skip_model_loading = os.getenv("API_SKIP_MODEL_LOADING", "false").lower()
    if skip_model_loading not in {"1", "true", "yes", "on"}:
        selected = os.getenv("API_MODELS", "").strip()
        model_names = tuple(
            name.strip() for name in selected.split(",") if name.strip()
        )
        _state.update(load_models(model_names or None))
    yield
    _state.clear()


app = FastAPI(
    title="API de Inferencia",
    description="Endpoints HTTP para predecir con los modelos registrados",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
            task=spec.task,
            class_names=list(spec.class_names),
        )
        for spec in list_specs()
    ]


@app.post("/predict/diabetes", response_model=PredictResponse)
def post_predict_diabetes(data: DiabetesPredictRequest) -> PredictResponse:
    """Predice M07 con un contrato de campos nombrados y categorías legibles."""
    model = _state.get("diabetes")
    if model is None:
        raise HTTPException(
            status_code=503, detail="El modelo 'diabetes' no esta cargado."
        )

    spec = get_spec("diabetes")
    features = data.to_features()
    try:
        result = predict(model, features, spec)
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        saved = save_prediction({
            "model": "diabetes",
            "input_data": data.model_dump(),
            "prediction": result.value,
            "label": result.label,
            "probabilities": result.probabilities,
        })
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        model="diabetes",
        input=features,
        prediction=result.value,
        label=result.label,
        probabilities=result.probabilities,
        execution_id=saved.get("id"),
        created_at=saved.get("created_at"),
    )


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
        result = predict(model, data.features, spec)
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        model=name,
        input=data.features,
        prediction=result.value,
        label=result.label,
        probabilities=result.probabilities,
    )
