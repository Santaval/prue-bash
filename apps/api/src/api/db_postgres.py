"""Persistencia directa en PostgreSQL, separada del cliente Supabase."""

import os
from datetime import UTC, datetime
from uuid import uuid4


def save_prediction(record: dict) -> dict:
    """Inserta una ejecución M07 y devuelve el ID/fecha usados en la fila."""
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL debe estar definida para PostgreSQL.")

    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "El backend PostgreSQL requiere psycopg. "
            "Instala las dependencias de requirements.txt."
        ) from exc

    saved = {
        **record,
        "id": str(uuid4()),
        "created_at": datetime.now(UTC),
    }
    statement = """
        INSERT INTO prediction_runs
            (id, created_at, model, input_data, prediction, label, probabilities)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at
    """
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as connection:
            inserted = connection.execute(
                statement,
                (
                    saved["id"],
                    saved["created_at"],
                    saved["model"],
                    Jsonb(saved["input_data"]),
                    saved["prediction"],
                    saved["label"],
                    Jsonb(saved["probabilities"]),
                ),
            ).fetchone()
    except psycopg.Error as exc:
        raise RuntimeError("No se pudo persistir la predicción en PostgreSQL.") from exc

    if inserted is None:
        raise RuntimeError("PostgreSQL no devolvió el registro insertado.")
    return {**saved, **inserted}
