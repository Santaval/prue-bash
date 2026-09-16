"""Main file to run the program."""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables de entorno desde .env
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

def test_connection():
    """function to test supabase conection."""

    # 1. Init client
    supabase: Client = create_client(url, key)
    print("Supabase client initialized correctly")

    response = supabase.table("test").select("*").limit(1).execute()

    print("Connection ok")
    print("Test table respose:", response.data)

if __name__ == "__main__":
    test_connection()
