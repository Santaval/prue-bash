"""Cliente de Supabase."""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def get_client() -> Client:
    """Crea el cliente de Supabase a partir de las variables de entorno."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL y SUPABASE_KEY deben estar definidas.")
    return create_client(url, key)


def check_connection() -> list:
    """Verifica la conexion consultando la tabla de prueba."""
    response = get_client().table("test").select("*").limit(1).execute()
    return response.data


if __name__ == "__main__":
    print("Respuesta de la tabla test:", check_connection())
