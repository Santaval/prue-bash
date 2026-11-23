# PostgreSQL local con Podman Compose

## Propósito

Este entorno ejecuta PostgreSQL local para desarrollo e integración. El backend
se selecciona con `PERSISTENCE_BACKEND=postgres`; el adaptador de Supabase se
mantiene disponible con `PERSISTENCE_BACKEND=supabase`.

## Requisitos

- Podman con un proveedor de Compose disponible (`podman-compose`).
- Driver `psycopg` instalado en el entorno Python de la API.
- Puerto `54329` disponible en `localhost`.

## Inicio

Desde la raíz del repositorio:

```bash
podman compose up -d postgres
```

`compose.yaml` define el servicio `postgres`, publica el puerto local `54329` y
usa el volumen nombrado `postgres_data`. La primera ejecución descarga la imagen
`docker.io/library/postgres:17`.

El archivo `docs/sql/prediction_runs.sql` se monta en el directorio de
inicialización de PostgreSQL. Se ejecuta automáticamente cuando se crea un
volumen de datos vacío.

## Estado y logs

```bash
podman compose ps
podman compose logs postgres
```

Si el volumen ya existía antes de incorporar el script SQL, aplicar el esquema
manualmente:

```bash
podman compose exec -T postgres psql -U prue -d prue < docs/sql/prediction_runs.sql
```

## Configuración de conexión

Podman Compose carga `.env` desde la raíz del proyecto. Las variables siguientes
deben compartir los mismos valores entre el contenedor y la API:

```dotenv
PERSISTENCE_BACKEND=postgres
DATABASE_URL=postgresql://prue:local_only_password@localhost:54329/prue
POSTGRES_USER=prue
POSTGRES_PASSWORD=local_only_password
POSTGRES_DB=prue
```

La plantilla `.example.env` incluye valores locales de referencia.

El entorno Python requiere las dependencias declaradas en `requirements.txt`.

Para servir únicamente M07, la API puede iniciar con
`API_MODELS=diabetes`; así no requiere artefactos de modelos de otros temas.
El artefacto `apps/trainer/dist/diabetes.keras` debe existir para ejecutar
predicciones. `API_SKIP_MODEL_LOADING=true` sirve solo para probar rutas y
validaciones, pues deja las solicitudes válidas sin modelo y responden 503.

## Operación

Detener el servicio sin eliminar los datos persistidos:

```bash
podman compose stop
```

Reanudar el servicio:

```bash
podman compose start
```

Eliminar el contenedor manteniendo el volumen:

```bash
podman compose down
```

El volumen `postgres_data` conserva las filas entre reinicios. Eliminar el
volumen también elimina los datos locales.
