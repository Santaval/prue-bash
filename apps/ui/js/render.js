// Renderizado del resultado, errores e historial. Todo con textContent.

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const fmt = (n) => new Intl.NumberFormat("es", { maximumFractionDigits: 4 }).format(n);
const pct = (n) => `${(n * 100).toFixed(1)}%`;

export function formatDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString("es");
}

export function renderResult(container, model, data) {
  container.replaceChildren();
  const card = el("div", "v-card v-card--pink");
  card.append(el("span", "v-card__eyebrow", "Resultado"));

  if (model.task === "regression" || !data.probabilities) {
    card.append(el("div", "result-value", fmt(data.prediction)));
    card.append(el("p", "t-small", model.task === "regression" ? "Valor estimado por el modelo." : "Predicción del modelo."));
  } else {
    const row = el("div", "row");
    const label = data.label ?? String(data.prediction);
    row.append(el("span", "v-badge v-badge--lg v-badge--info", label));
    row.append(el("span", "t-small", `Predicción: ${fmt(data.prediction)}`));
    card.append(row);

    const list = el("ul", "prob-list");
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    entries.forEach(([name, p], i) => {
      const li = el("li", i === 0 ? "prob prob--top" : "prob");
      const head = el("div", "prob__head");
      head.append(el("span", "", name), el("span", "t-mono", pct(p)));
      const track = el("div", "prob__track");
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", name);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(Math.round(p * 100)));
      const fill = el("div", "prob__fill");
      fill.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
      track.append(fill);
      li.append(head, track);
      list.append(li);
    });
    card.append(list);
  }

  const meta = el("div", "result-meta");
  meta.append(el("span", "", `Modelo: ${data.model}`));
  if (data.execution_id) {
    const line = el("span", "", "ID de ejecución: ");
    line.append(el("code", "", String(data.execution_id)));
    meta.append(line);
  }
  if (data.created_at) meta.append(el("span", "", `Fecha: ${formatDate(data.created_at)}`));
  card.append(meta);
  container.append(card);
}

export function renderError(container, error) {
  container.replaceChildren();
  let title = "No se pudo completar la predicción";
  let body = error.message;
  if (error.network) title = "Sin conexión con la API";
  else if (error.status === 503) {
    title = "Modelo no disponible";
    body = `${error.message} El servidor no tiene el modelo cargado o no pudo guardar la ejecución.`;
  } else if (error.status === 404) title = "Modelo no encontrado";
  else if (error.status === 422) title = "Datos no válidos";
  const alert = el("div", "v-alert v-alert--danger");
  alert.setAttribute("role", "alert");
  alert.append(el("p", "v-alert__title", title), el("p", "v-alert__body", body));
  container.append(alert);
}

export function renderLoading(container) {
  container.replaceChildren(el("div", "v-skeleton"));
}

export function renderHistory(container, items, onSelect) {
  container.replaceChildren();
  if (!items.length) {
    container.append(el("div", "empty", "Aún no hay predicciones en esta sesión."));
    return;
  }
  const list = el("ul", "history");
  items.forEach((item) => {
    const li = el("li", "history__item");
    const main = el("div", "history__main");
    main.append(el("span", "history__title", `${item.model}: ${item.summary}`), el("span", "t-small", formatDate(item.at)));
    const btn = el("button", "v-btn v-btn--ghost v-btn--sm", "Ver");
    btn.type = "button";
    btn.addEventListener("click", () => onSelect(item));
    li.append(main, btn);
    list.append(li);
  });
  container.append(list);
}

export function modelCard(model, selected, onSelect) {
  const btn = el("button", "model-card");
  btn.type = "button";
  btn.setAttribute("aria-pressed", String(selected));
  const top = el("div", "row row--between");
  top.append(el("span", "model-card__name", model.name));
  const badge = el("span", model.loaded ? "v-badge v-badge--ok" : "v-badge v-badge--warn");
  badge.append(el("span", "v-badge__dot"), document.createTextNode(model.loaded ? "Cargado" : "No cargado"));
  top.append(badge);
  const tags = el("div", "row");
  const taskLabel = { regression: "Regresión", binary: "Binario", multiclass: "Multiclase" }[model.task] || model.task;
  tags.append(el("span", "v-badge", taskLabel), el("span", "v-badge", `${model.input_features} entradas`));
  btn.append(top, el("p", "model-card__desc", model.description || "Sin descripción."), tags);
  btn.addEventListener("click", () => onSelect(model.name));
  return btn;
}
