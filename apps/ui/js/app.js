import * as api from "./api.js";
import { buildForm, sampleFor } from "./forms.js";
import { modelCard, renderError, renderHistory, renderLoading, renderResult } from "./render.js";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "prue.history";
const THEME_KEY = "prue.theme";

const state = { models: [], selected: null, form: null, history: [] };

function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* sin almacenamiento */ }
}
function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function toast(message) {
  const node = document.createElement("div");
  node.className = "v-toast";
  node.textContent = message;
  $("toasts").append(node);
  setTimeout(() => node.remove(), 2800);
}

/* Tema */
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
  else document.documentElement.removeAttribute("data-theme");
  const dark = currentTheme() === "dark";
  const btn = $("theme-toggle");
  btn.setAttribute("aria-pressed", String(dark));
  btn.textContent = dark ? "Modo claro" : "Modo oscuro";
}
function currentTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set) return set;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  store(THEME_KEY, next);
  applyTheme(next);
}

/* Estado de la API */
function setStatus(kind, text) {
  const badge = $("api-status");
  badge.className = `v-badge v-badge--${kind}`;
  badge.replaceChildren();
  const dot = document.createElement("span");
  dot.className = "v-badge__dot";
  badge.append(dot, document.createTextNode(text));
}

async function refresh() {
  setStatus("", "Conectando...");
  const list = $("model-list");
  list.replaceChildren(...[1, 2, 3].map(() => Object.assign(document.createElement("div"), { className: "v-skeleton" })));
  try {
    await api.getHealth();
    state.models = await api.getModels();
    setStatus("ok", "API en linea");
  } catch (error) {
    state.models = [];
    setStatus("danger", "API sin conexión");
    list.replaceChildren();
    const box = document.createElement("div");
    box.className = "v-alert v-alert--danger";
    box.setAttribute("role", "alert");
    const t = document.createElement("p");
    t.className = "v-alert__title";
    t.textContent = "No se pudieron cargar los modelos";
    const b = document.createElement("p");
    b.className = "v-alert__body";
    b.textContent = error.message;
    box.append(t, b);
    list.append(box);
    showWorkspace(false);
    return;
  }
  if (!state.models.some((m) => m.name === state.selected)) state.selected = state.models[0]?.name ?? null;
  renderModels();
  selectModel(state.selected, { keepResult: true });
}

function renderModels() {
  const list = $("model-list");
  list.replaceChildren(...state.models.map((m) => modelCard(m, m.name === state.selected, selectModel)));
}

function showWorkspace(show) {
  $("workspace").hidden = !show;
  $("workspace-empty").hidden = show;
}

function selectModel(name, { keepResult = false } = {}) {
  const model = state.models.find((m) => m.name === name);
  if (!model) { showWorkspace(false); return; }
  const changed = state.selected !== name || !state.form;
  state.selected = name;
  renderModels();
  showWorkspace(true);
  $("ws-title").textContent = model.name;
  $("ws-desc").textContent = model.description || "Sin descripción.";
  const taskLabel = { regression: "Regresión", binary: "Clasificación binaria", multiclass: "Clasificación multiclase" }[model.task] || model.task;
  $("ws-task").textContent = taskLabel;
  const classes = model.class_names.length ? ` Clases: ${model.class_names.join(", ")}.` : "";
  $("ws-hint").textContent = `${model.input_features} entradas.${classes}`;
  if (changed || !keepResult) {
    state.form = buildForm(model);
    $("form-fields").replaceChildren(state.form.element);
    $("result").replaceChildren();
  }
  $("load-warning").hidden = model.loaded;
}

function summarize(data) {
  if (data.label) return data.label;
  return String(Math.round(data.prediction * 10000) / 10000);
}

async function onSubmit(event) {
  event.preventDefault();
  const model = state.models.find((m) => m.name === state.selected);
  if (!model) return;
  const { ok, payload } = state.form.collect();
  if (!ok) return;
  const submit = $("submit");
  submit.setAttribute("aria-busy", "true");
  submit.disabled = true;
  const result = $("result");
  renderLoading(result);
  try {
    const data = await api.predict(model.name, payload);
    renderResult(result, model, data);
    pushHistory({ model: model.name, summary: summarize(data), at: data.created_at || new Date().toISOString(), data });
  } catch (error) {
    renderError(result, error);
  } finally {
    submit.removeAttribute("aria-busy");
    submit.disabled = false;
  }
}

/* Historial */
function pushHistory(item) {
  state.history = [item, ...state.history].slice(0, 20);
  store(HISTORY_KEY, JSON.stringify(state.history));
  drawHistory();
}
function drawHistory() {
  renderHistory($("history"), state.history, (item) => {
    const model = state.models.find((m) => m.name === item.model);
    if (!model) return;
    if (state.selected !== item.model) selectModel(item.model);
    renderResult($("result"), model, item.data);
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}
function loadHistory() {
  try {
    const parsed = JSON.parse(load(HISTORY_KEY) || "[]");
    state.history = Array.isArray(parsed) ? parsed.filter((i) => i && i.model && i.data).slice(0, 20) : [];
  } catch { state.history = []; }
}

function init() {
  applyTheme(load(THEME_KEY));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(load(THEME_KEY)));
  $("theme-toggle").addEventListener("click", toggleTheme);

  $("api-base").placeholder = api.DEFAULT_BASE;
  $("api-base").value = api.getBase();
  $("api-form").addEventListener("submit", (e) => {
    e.preventDefault();
    $("api-base").value = api.setBase($("api-base").value);
    refresh();
  });
  $("refresh").addEventListener("click", refresh);

  $("predict-form").addEventListener("submit", onSubmit);
  $("sample").addEventListener("click", () => {
    const model = state.models.find((m) => m.name === state.selected);
    if (model && state.form) { state.form.fill(sampleFor(model)); toast("Valores de ejemplo cargados."); }
  });
  $("clear").addEventListener("click", () => {
    if (state.form) state.form.reset();
    $("result").replaceChildren();
  });
  $("clear-history").addEventListener("click", () => {
    state.history = [];
    store(HISTORY_KEY, "[]");
    drawHistory();
  });

  loadHistory();
  drawHistory();
  refresh();
}

init();
