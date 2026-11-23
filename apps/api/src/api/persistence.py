"""Selecciona la persistencia configurada sin mezclar los clientes de DB."""

import os

from dotenv import load_dotenv

load_dotenv()


def save_prediction(record: dict) -> dict:
    """Guarda en Supabase por defecto o en PostgreSQL directo bajo demanda."""
    backend = os.getenv("PERSISTENCE_BACKEND", "supabase").strip().lower()
    if backend == "supabase":
        from .db_supabase import save_prediction as save_supabase

        return save_supabase(record)
    if backend == "postgres":
        from .db_postgres import save_prediction as save_postgres

        return save_postgres(record)
    raise RuntimeError(
        "PERSISTENCE_BACKEND debe ser 'supabase' o 'postgres'."
    )
