/* Dashboard behaviour: controls, theme, sortable tables, and drawing the plotly
   figures that dashboard/build.py wrote into the page as JSON.

   Charts are built in Python. This file only picks the light or dark version,
   hides traces whose `meta` tags don't match the current controls, and falls
   back to the static PNGs if plotly.js didn't load. */
(function () {
  "use strict";

  const FIGS = JSON.parse(document.getElementById("figs").textContent);
  const hasPlotly = typeof window.Plotly !== "undefined";
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const state = {};

  /* ---------- theme ---------- */
  const root = document.documentElement;
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const theme = () => root.getAttribute("data-theme") || (darkQuery.matches ? "dark" : "light");
  const toggle = document.getElementById("theme");
  function paintToggle() {
    const dark = theme() === "dark";
    toggle.textContent = dark ? "Light mode" : "Dark mode";
    toggle.setAttribute("aria-pressed", String(dark));
  }
  toggle.addEventListener("click", () => {
    root.setAttribute("data-theme", theme() === "dark" ? "light" : "dark");
    paintToggle();
    drawAll();
  });
  darkQuery.addEventListener("change", () => { paintToggle(); drawAll(); });

  /* ---------- figures ---------- */
  // No modebar: it sat on top of the legends. Drag to zoom, double-click to reset.
  const config = () => ({ displaylogo: false, responsive: true, scrollZoom: false, displayModeBar: false });

  // Which control keys each figure listens to.
  const deps = {};
  Object.entries(FIGS).forEach(([name, spec]) => {
    const keys = new Set();
    if (spec.variant_by) keys.add(spec.variant_by);
    Object.values(spec.variants).forEach((v) =>
      v.light.data.forEach((t) => Object.keys(t.meta || {}).forEach((k) => keys.add(k))));
    deps[name] = keys;
  });

  const matches = (meta) => !meta || Object.entries(meta).every(([k, v]) => !(k in state) || state[k] === v);

  function draw(node) {
    const name = node.dataset.fig;
    const spec = FIGS[name];
    const variant = spec.variant_by ? state[spec.variant_by] : "_";
    const fig = (spec.variants[variant] || Object.values(spec.variants)[0])[theme()];
    const data = fig.data.map((t) => Object.assign({}, t, { visible: matches(t.meta) }));
    const layout = Object.assign({}, fig.layout);
    if (coarse) layout.dragmode = false; // let a finger scroll the page
    window.Plotly.react(node, data, layout, config());
  }

  const plots = Array.from(document.querySelectorAll(".plot[data-fig]"));
  function drawAll() { if (hasPlotly) plots.forEach(draw); }

  /* ---------- controls ---------- */
  function applyFilters() {
    document.querySelectorAll("[data-filter]").forEach((el) => {
      const [k, v] = el.dataset.filter.split(":");
      el.hidden = k in state && state[k] !== v;
    });
  }

  function setState(key, value, redraw = true) {
    state[key] = value;
    document.querySelectorAll(`[data-key="${key}"]`).forEach((c) => {
      if (c.tagName === "SELECT") c.value = value;
      else c.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === value)));
    });
    applyFilters();
    if (redraw && hasPlotly) plots.filter((p) => deps[p.dataset.fig].has(key)).forEach(draw);
  }

  document.querySelectorAll("[data-key]").forEach((c) => {
    const key = c.dataset.key;
    if (c.tagName === "SELECT") {
      if (!(key in state)) state[key] = c.value;
      c.addEventListener("change", () => setState(key, c.value));
    } else {
      const buttons = Array.from(c.querySelectorAll("button"));
      if (!(key in state)) state[key] = buttons[0].dataset.value;
      buttons.forEach((b, i) => {
        b.addEventListener("click", () => setState(key, b.dataset.value));
        b.addEventListener("keydown", (e) => {
          const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
          if (!step) return;
          e.preventDefault();
          const next = buttons[(i + step + buttons.length) % buttons.length];
          next.focus();
          setState(key, next.dataset.value);
        });
      });
    }
  });
  Object.keys(state).forEach((k) => setState(k, state[k], false));

  /* ---------- sortable tables ---------- */
  document.querySelectorAll("table.sortable").forEach((table) => {
    const heads = Array.from(table.querySelectorAll("th"));
    heads.forEach((th, col) => {
      if (!th.dataset.sort) return;
      th.tabIndex = 0;
      const sort = () => {
        const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", dir);
        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        const val = (r) => {
          const c = r.cells[col];
          const v = c.dataset.v;
          return v !== undefined ? parseFloat(v) : c.textContent;
        };
        rows.sort((a, b) => {
          const x = val(a), y = val(b);
          const d = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
          return dir === "ascending" ? d : -d;
        });
        rows.forEach((r) => body.appendChild(r));
      };
      th.addEventListener("click", sort);
      th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
    });
  });

  /* ---------- start ---------- */
  paintToggle();
  if (hasPlotly) {
    drawAll();
  } else {
    plots.forEach((p) => { p.hidden = true; });
    document.querySelectorAll("img.fallback").forEach((img) => { img.hidden = false; });
  }
})();
