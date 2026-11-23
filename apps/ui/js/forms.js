// Constructor de formularios: campos nombrados para diabetes, genéricos para el resto.

const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
};

const DIABETES_FIELDS = [
  { name: "gender", label: "Género", type: "select", options: [["Female", "Femenino"], ["Male", "Masculino"], ["Other", "Otro"]] },
  { name: "age", label: "Edad", type: "number", unit: "años", min: 0, step: "any", desc: "En años, 0 o mayor." },
  { name: "hypertension", label: "Hipertensión", type: "select", int: true, options: [["0", "No"], ["1", "Sí"]] },
  { name: "heart_disease", label: "Enfermedad cardíaca", type: "select", int: true, options: [["0", "No"], ["1", "Sí"]] },
  {
    name: "smoking_history", label: "Historial de tabaquismo", type: "select",
    options: [["No Info", "Sin información"], ["current", "Fuma actualmente"], ["ever", "Ha fumado alguna vez"],
      ["former", "Exfumador"], ["never", "Nunca ha fumado"], ["not current", "No fuma actualmente"]],
  },
  { name: "bmi", label: "IMC", type: "number", min: 0, step: "any", desc: "Índice de masa corporal." },
  { name: "HbA1c_level", label: "Nivel de HbA1c", type: "number", min: 0, step: "any", desc: "Hemoglobina glicosilada (%)." },
  { name: "blood_glucose_level", label: "Glucosa en sangre", type: "number", min: 0, step: "any", desc: "mg/dL." },
];

const DIABETES_SAMPLE = {
  gender: "Female", age: "54", hypertension: "0", heart_disease: "0",
  smoking_history: "never", bmi: "27.3", HbA1c_level: "6.6", blood_glucose_level: "140",
};

const SAMPLES = {
  diabetes: DIABETES_SAMPLE,
  academic_risk: {
    attendance_pct: "85", previous_gpa: "3.2", study_hours_weekly: "12", failed_courses_history: "0",
    assignments_on_time_pct: "90", forum_participation_count: "5", is_working: "0",
  },
  loan_amount: {
    applicant_income: "4500", coapplicant_income: "1500", credit_score: "720",
    years_employed: "5", owns_property: "1", loan_term_months: "360",
  },
};

function humanize(name) {
  const text = name.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function genericFields(model) {
  return model.feature_names.map((name) => ({
    name, label: humanize(name), type: "number", step: "any", desc: name,
  }));
}

export function fieldsFor(model) {
  return model.name === "diabetes" ? DIABETES_FIELDS : genericFields(model);
}

export function sampleFor(model) {
  const known = SAMPLES[model.name];
  if (known) return known;
  return Object.fromEntries(model.feature_names.map((n) => [n, "1"]));
}

// Devuelve { element, collect(), fill(values), reset(), controls }.
export function buildForm(model) {
  const fields = fieldsFor(model);
  const grid = el("div", { class: "form-grid" });
  const controls = new Map();
  const errors = new Map();

  fields.forEach((f) => {
    const id = `f-${model.name}-${f.name}`;
    const errId = `${id}-err`;
    const label = el("label", { class: "v-field__label", for: id, text: f.label });
    let control;
    if (f.type === "select") {
      control = el("select", { class: "v-select", id, name: f.name, required: "" });
      f.options.forEach(([value, text]) => control.append(el("option", { value, text })));
    } else {
      control = el("input", {
        class: "v-input", id, name: f.name, type: "number", inputmode: "decimal",
        step: f.step || "any", required: "", placeholder: "0",
      });
      if (f.min !== undefined) control.setAttribute("min", String(f.min));
    }
    const describedBy = [errId];
    const wrap = el("div", { class: "v-field" }, label, control);
    if (f.desc) {
      const dId = `${id}-desc`;
      wrap.append(el("p", { class: "v-field__desc", id: dId, text: f.desc }));
      describedBy.unshift(dId);
    }
    const err = el("p", { class: "v-field__error", id: errId, role: "alert" });
    control.setAttribute("aria-describedby", describedBy.join(" "));
    wrap.append(err);
    grid.append(wrap);
    controls.set(f.name, control);
    errors.set(f.name, err);
    control.addEventListener("input", () => clearError(f.name));
  });

  function clearError(name) {
    errors.get(name).textContent = "";
    controls.get(name).removeAttribute("aria-invalid");
  }
  function setError(name, message) {
    errors.get(name).textContent = message;
    controls.get(name).setAttribute("aria-invalid", "true");
  }

  function fill(values) {
    for (const [name, value] of Object.entries(values)) {
      const c = controls.get(name);
      if (c) { c.value = value; clearError(name); }
    }
  }

  function reset() {
    controls.forEach((c, name) => {
      c.value = c.tagName === "SELECT" ? c.options[0].value : "";
      clearError(name);
    });
  }

  // Valida en el cliente y arma el cuerpo JSON. Devuelve { ok, payload }.
  function collect() {
    const values = {};
    let firstBad = null;
    fields.forEach((f) => {
      const c = controls.get(f.name);
      clearError(f.name);
      const raw = c.value.trim();
      if (raw === "") {
        setError(f.name, "Este campo es obligatorio.");
        firstBad = firstBad || c;
        return;
      }
      if (f.type === "select") {
        values[f.name] = f.int ? Number(raw) : raw;
        return;
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        setError(f.name, "Ingresa un número válido.");
        firstBad = firstBad || c;
      } else if (model.name === "diabetes" && n < 0) {
        setError(f.name, "El valor debe ser 0 o mayor.");
        firstBad = firstBad || c;
      } else {
        values[f.name] = n;
      }
    });
    if (firstBad) { firstBad.focus(); return { ok: false }; }
    const payload = model.name === "diabetes"
      ? values
      : { features: model.feature_names.map((n) => values[n]) };
    return { ok: true, payload, values };
  }

  return { element: grid, collect, fill, reset, controls };
}
