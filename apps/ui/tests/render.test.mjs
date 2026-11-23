// Pruebas de renderizado con un DOM mínimo simulado.
import { test } from "node:test";
import assert from "node:assert/strict";

class Node_ {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.className = ""; this.textContent = ""; this.style = {}; this.listeners = {}; }
  append(...c) { this.children.push(...c); }
  replaceChildren(...c) { this.children = c; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(t, fn) { this.listeners[t] = fn; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.all().map((n) => n.textContent).join(" "); }
}
globalThis.document = {
  createElement: (t) => new Node_(t),
  createTextNode: (t) => Object.assign(new Node_("#text"), { textContent: t }),
};

const { formatDate, renderResult, renderError, renderHistory, modelCard } =
  await import("../js/render.js");

test("formatDate devuelve el texto original si la fecha no es válida", () => {
  assert.equal(formatDate("no-es-fecha"), "no-es-fecha");
});

test("la regresión muestra un solo valor y ninguna lista de probabilidades", () => {
  const c = new Node_("div");
  renderResult(c, { task: "regression" }, { prediction: 1234.56789, model: "loan_amount" });
  const nodes = c.all();
  assert.ok(nodes.some((n) => n.className === "result-value"));
  assert.ok(!nodes.some((n) => n.className === "prob-list"));
  assert.match(c.text(), /Modelo: loan_amount/);
});

test("la clasificación ordena las probabilidades de mayor a menor y marca la primera", () => {
  const c = new Node_("div");
  renderResult(c, { task: "multiclass" }, {
    prediction: 2, label: "alto", model: "academic_risk",
    probabilities: { bajo: 0.1, medio: 0.2, alto: 0.7 },
  });
  const items = c.all().filter((n) => n.tag === "li");
  assert.equal(items.length, 3);
  assert.equal(items[0].className, "prob prob--top");
  assert.match(items[0].text(), /alto/);
  assert.match(items[0].text(), /70\.0%/);
  const bars = c.all().filter((n) => n.className === "prob__fill");
  assert.equal(bars[0].style.width, "70%");
});

test("una probabilidad fuera de [0,1] no desborda la barra", () => {
  const c = new Node_("div");
  renderResult(c, { task: "binary" }, { prediction: 1, model: "m", probabilities: { a: 1.4, b: -0.2 } });
  const widths = c.all().filter((n) => n.className === "prob__fill").map((n) => n.style.width);
  assert.deepEqual(widths, ["100%", "0%"]);
});

test("renderError elige el título según el tipo de fallo", () => {
  const title = (err) => {
    const c = new Node_("div");
    renderError(c, err);
    return c.all().find((n) => n.className === "v-alert__title").textContent;
  };
  assert.equal(title({ message: "x", network: true }), "Sin conexión con la API");
  assert.equal(title({ message: "x", status: 503 }), "Modelo no disponible");
  assert.equal(title({ message: "x", status: 404 }), "Modelo no encontrado");
  assert.equal(title({ message: "x", status: 422 }), "Datos no válidos");
  assert.equal(title({ message: "x", status: 500 }), "No se pudo completar la predicción");
});

test("renderHistory vacío muestra el mensaje de lista vacía", () => {
  const c = new Node_("div");
  renderHistory(c, [], () => {});
  assert.match(c.text(), /Aún no hay predicciones/);
});

test("el botón Ver del historial llama a onSelect con su elemento", () => {
  const c = new Node_("div");
  const item = { model: "m", summary: "s", at: "2026-01-01T00:00:00Z" };
  let picked;
  renderHistory(c, [item], (i) => { picked = i; });
  c.all().find((n) => n.tag === "button").listeners.click();
  assert.equal(picked, item);
});

test("modelCard refleja el estado cargado y el tipo de tarea", () => {
  const card = modelCard(
    { name: "diabetes", loaded: false, task: "binary", input_features: 8, description: "" },
    true, () => {},
  );
  assert.equal(card.attrs["aria-pressed"], "true");
  assert.match(card.text(), /No cargado/);
  assert.match(card.text(), /Binario/);
  assert.match(card.text(), /8 entradas/);
  assert.match(card.text(), /Sin descripción/);
});
