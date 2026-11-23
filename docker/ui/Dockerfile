# Imagen de la UI: archivos estáticos servidos por nginx, que además reenvía
# /api/* al contenedor de la API (mismo origen, sin CORS).
# Build context: raíz del repositorio.
FROM nginx:1.27-alpine
COPY docker/ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY apps/ui /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD wget -q --spider http://127.0.0.1/healthz || exit 1
