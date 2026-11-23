// Pruebas del cliente HTTP. Sin dependencias: `node --test apps/ui/tests`.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";

// api.js lee `location` y `localStorage` del navegador; los simulamos.
globalThis.location = { hostname: "example.com", port: "", origin: "https://example.com" };
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
};

const { ApiError, DEFAULT_BASE, errorMessage, getBase, setBase, getModels, predict } =
  await import("../js/api.js");

beforeEach(() => store.clear());

const reply = (status, body) => async () => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => {
    if (body === undefined) throw new Error("sin cuerpo");
    return body;
  },
});

test("errorMessage devuelve el detail de texto tal cual", () => {
  assert.equal(errorMessage({ detail: "Modelo no encontrado" }, "x"), "Modelo no encontrado");
});

test("errorMessage aplana la lista 422 de FastAPI sin el prefijo body", () => {
  const body = { detail: [
    { loc: ["body", "age"], msg: "Input should be a valid number" },
    { loc: ["body", "features", 2], msg: "field required" },
  ] };
  assert.equal(
    errorMessage(body, "x"),
    "age: Input should be a valid number; features.2: field required",
  );
});

test("errorMessage usa el mensaje de respaldo si no hay detail", () => {
  assert.equal(errorMessage(null, "respaldo"), "respaldo");
  assert.equal(errorMessage({}, "respaldo"), "respaldo");
});

test("DEFAULT_BASE apunta a /api del mismo origen fuera de localhost:5173", () => {
  assert.equal(DEFAULT_BASE, "https://example.com/api");
});

test("setBase quita espacios y barras finales, y getBase lo recupera", () => {
  assert.equal(setBase("  http://api.test:8000///  "), "http://api.test:8000");
  assert.equal(getBase(), "http://api.test:8000");
});

test("setBase con texto vacío vuelve al valor por defecto", () => {
  assert.equal(setBase("   "), DEFAULT_BASE);
});

test("getModels falla con ApiError de red si fetch lanza", async () => {
  globalThis.fetch = async () => { throw new TypeError("fetch failed"); };
  await assert.rejects(getModels(), (e) => {
    assert.ok(e instanceof ApiError);
    assert.equal(e.network, true);
    assert.equal(e.status, 0);
    return true;
  });
});

test("una respuesta 422 se convierte en ApiError con estado y mensaje", async () => {
  globalThis.fetch = reply(422, { detail: [{ loc: ["body", "age"], msg: "bad" }] });
  await assert.rejects(predict("diabetes", {}), (e) => {
    assert.equal(e.status, 422);
    assert.equal(e.message, "age: bad");
    assert.equal(e.network, false);
    return true;
  });
});

test("un error sin cuerpo JSON usa el mensaje genérico con el estado", async () => {
  globalThis.fetch = reply(503, undefined);
  await assert.rejects(getModels(), (e) => {
    assert.equal(e.status, 503);
    assert.match(e.message, /503/);
    return true;
  });
});

test("predict codifica el nombre del modelo y envía el JSON por POST", async () => {
  let call;
  globalThis.fetch = async (url, options) => {
    call = { url, options };
    return reply(200, { prediction: 1 })();
  };
  setBase("http://api.test");
  const out = await predict("a b/c", { x: 1 });
  assert.deepEqual(out, { prediction: 1 });
  assert.equal(call.url, "http://api.test/predict/a%20b%2Fc");
  assert.equal(call.options.method, "POST");
  assert.equal(call.options.body, '{"x":1}');
});
