# syntax=docker/dockerfile:1
# Imagen de la API. Build context: raíz del repositorio.
#
# Etapa 1 (trainer): limpia los datos originales y entrena los modelos, de modo
# que la imagen final ya trae sus artefactos .keras y la API arranca sin pasos
# manuales. Etapa 2 (runtime): solo el código de la API, el paquete del modelo
# y los artefactos.

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/packages/model/src:/app/apps/api/src:/app/apps/trainer/src
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM base AS trainer
# Modelos a entrenar: "--all" o "--model <nombre>".
ARG TRAIN_ARGS="--all"
COPY packages ./packages
COPY apps/trainer ./apps/trainer
COPY scripts ./scripts
COPY data/original ./data/original
RUN python scripts/clean_diabetes.py \
 && python scripts/clean_academic.py \
 && python scripts/clean_loan.py \
 && python -m trainer ${TRAIN_ARGS} --models-dir /models

FROM base AS runtime
RUN useradd --system --uid 10001 --no-create-home app
COPY packages ./packages
COPY apps/api/src ./apps/api/src
COPY --from=trainer /models /models
ENV MODELS_DIR=/models
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "apps/api/src", "--proxy-headers", "--forwarded-allow-ips", "*"]
