// Pruebas de la lógica pura de formularios (sin DOM).
import { test } from "node:test";
import assert from "node:assert/strict";
import { fieldsFor, sampleFor } from "../js/forms.js";

const diabetes = { name: "diabetes", feature_names: [] };
const generic = { name: "linear", feature_names: ["height_cm", "weight"] };

test("diabetes usa campos nombrados con selects para variables categóricas", () => {
  const fields = fieldsFor(diabetes);
  assert.equal(fields.length, 8);
  const gender = fields.find((f) => f.name === "gender");
  assert.equal(gender.type, "select");
  assert.deepEqual(gender.options.map(([v]) => v), ["Female", "Male", "Other"]);
});

test("hipertensión y enfermedad cardíaca se envían como enteros", () => {
  const fields = fieldsFor(diabetes);
  for (const name of ["hypertension", "heart_disease"]) {
    assert.equal(fields.find((f) => f.name === name).int, true);
  }
});

test("un modelo desconocido genera un campo numérico por feature, en orden", () => {
  const fields = fieldsFor(generic);
  assert.deepEqual(fields.map((f) => f.name), ["height_cm", "weight"]);
  assert.deepEqual(fields.map((f) => f.label), ["Height cm", "Weight"]);
  assert.ok(fields.every((f) => f.type === "number"));
});

test("el ejemplo de diabetes cubre exactamente los campos del formulario", () => {
  const names = fieldsFor(diabetes).map((f) => f.name).sort();
  assert.deepEqual(Object.keys(sampleFor(diabetes)).sort(), names);
});

test("los ejemplos de loan_amount y academic_risk cubren todas sus entradas", () => {
  assert.equal(Object.keys(sampleFor({ name: "loan_amount", feature_names: [] })).length, 6);
  assert.equal(Object.keys(sampleFor({ name: "academic_risk", feature_names: [] })).length, 7);
});

test("un modelo sin ejemplo propio recibe '1' en cada feature", () => {
  assert.deepEqual(sampleFor(generic), { height_cm: "1", weight: "1" });
});
