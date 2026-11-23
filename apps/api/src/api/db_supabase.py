"""Persistencia de ejecuciones usando el cliente REST de Supabase existente."""

from datetime import UTC, datetime
from uuid import uuid4

from .db import get_client


def save_prediction(record: dict) -> dict:
    """Inserta una predicción en Supabase y devuelve su ID y fecha."""
    saved = {
        **record,
        "id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    try:
        get_client().table("prediction_runs").insert(saved).execute()
    except Exception as exc:  # El cliente puede lanzar errores HTTP o de red.
        raise RuntimeError("No se pudo persistir la predicción en Supabase.") from exc
    return saved
