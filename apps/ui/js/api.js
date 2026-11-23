// Cliente HTTP de la API de inferencia.
const KEY = "prue.apiBase";
// En desarrollo (UI en :5173) la API corre aparte en :8000. Desplegada con
// Docker, nginx sirve la UI y reenvía /api al contenedor de la API.
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);
export const DEFAULT_BASE = LOCAL_HOSTS.has(location.hostname) && location.port === "5173"
  ? "http://127.0.0.1:8000"
  : `${location.origin}/api`;

export class ApiError extends Error {
  constructor(message, { status = 0, network = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.network = network;
  }
}

export function getBase() {
  try { return localStorage.getItem(KEY) || DEFAULT_BASE; } catch { return DEFAULT_BASE; }
}

export function setBase(url) {
  const clean = (url || "").trim().replace(/\/+$/, "") || DEFAULT_BASE;
  try { localStorage.setItem(KEY, clean); } catch { /* sin almacenamiento */ }
  return clean;
}

// Convierte `detail` de FastAPI (texto o lista 422) en un mensaje legible.
export function errorMessage(body, fallback) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return loc ? `${loc}: ${d.msg}` : String(d.msg);
      })
      .join("; ");
  }
  return fallback;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(getBase() + path, options);
  } catch {
    throw new ApiError("No se pudo conectar con la API. Verifica la URL y que el servidor esté activo.", { network: true });
  }
  let body = null;
  try { body = await response.json(); } catch { /* cuerpo vacio */ }
  if (!response.ok) {
    throw new ApiError(errorMessage(body, `Error ${response.status} de la API.`), { status: response.status });
  }
  return body;
}

export const getHealth = () => request("/");
export const getModels = () => request("/models");
export function predict(name, payload) {
  return request(`/predict/${encodeURIComponent(name)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
