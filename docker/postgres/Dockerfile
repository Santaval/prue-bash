# PostgreSQL con el esquema de prediction_runs ya incluido: se aplica solo la
# primera vez que se inicializa el volumen de datos.
FROM postgres:17
COPY docs/sql/prediction_runs.sql /docker-entrypoint-initdb.d/01-prediction_runs.sql
