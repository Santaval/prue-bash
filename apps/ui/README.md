# UI

Interfaz web en HTML, CSS y JavaScript puros (módulos ES, sin build ni npm).
Lista los modelos de la API y permite ejecutar predicciones.

## Ejecutar

Desde la raíz del repositorio:

```bash
# 1. API (sin artefactos de modelo basta para ver la UI)
API_SKIP_MODEL_LOADING=true PERSISTENCE_BACKEND=postgres \
  PYTHONPATH="packages/model/src:apps/api/src" \
  uvicorn api.main:app --port 8000 --app-dir apps/api/src

# 2. UI
python -m http.server 5173 -d apps/ui
```

Abre <http://localhost:5173>. La URL de la API (por defecto
`http://127.0.0.1:8000`) se cambia desde la cabecera y se guarda en
`localStorage`.

## Pruebas

Sin dependencias, con el test runner de Node (22 o superior):

```bash
node --test apps/ui/tests/*.test.mjs
```

Cubren el cliente HTTP (`api.js`), la lógica pura de formularios y el
renderizado con un DOM simulado. No hay pruebas de navegador real.

## CORS

La API solo acepta los orígenes de `API_CORS_ORIGINS` (por defecto
`http://localhost:5173,http://127.0.0.1:5173`). Si sirves la UI en otro
puerto u origen, agrégalo a esa variable.

## Estructura

- `index.html`: página única.
- `css/tokens.css`: paleta "Paper" (clara y oscura), tipografía, espaciado y radios.
- `css/base.css`, `css/components.css`: base, layout y componentes `v-*`.
- `js/api.js`: cliente HTTP y normalización de errores.
- `js/forms.js`: formularios (campos nombrados para `diabetes`, genéricos para el resto).
- `js/render.js`: resultado, errores, historial y tarjetas de modelo.
- `js/app.js`: estado, tema, historial.

El tema sigue `prefers-color-scheme` y se puede forzar con el botón de la
cabecera. Fuentes: Bricolage Grotesque y DM Sans desde Google Fonts.
