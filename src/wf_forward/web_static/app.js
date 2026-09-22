"use strict";

const $ = (id) => document.getElementById(id);
const colors = {teal: "#127d78", coral: "#c67d62", indigo: "#7c82b4", gold: "#b69a50"};
const state = {token: null, running: false, job: null, resultJob: null, summary: null,
  frame: null, index: 0, mode: "form", version: 0, playing: null, frameAbort: null, frameRequest: 0, dirty: false, densityYMax: null, sampling: false, trajectories: null, sampleVersion: 0, sfsRunning: false, sfsJob: null, sfsResult: null, sfsVersion: 0, sfsIndex: 0};
const presetNames = {neutral: "Neutral", selection: "Selection", mutation: "Mutation", bottleneck: "Bottleneck"};

function format(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return n === Infinity ? "∞" : "—";
  if (n !== 0 && Math.abs(n) < 0.0001) return n.toExponential(2);
  return n.toLocaleString("en-US", {maximumFractionDigits: digits});
}

function metric(value) { return Number(value).toFixed(4); }
function showError(message) { $("error-banner").textContent = message; $("error-banner").hidden = false; }
function clearError() { $("error-banner").hidden = true; }
function status(message, kind = "") {
  $("status-banner").className = "status-banner" + (kind ? " " + kind : "");
  $("status-text").textContent = message;
  $("status-icon").textContent = kind === "running" ? "◌" : kind === "stale" ? "△" : "○";
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {
    ...(options.body ? {"Content-Type": "application/json", "X-WF-Token": state.token} : {}),
    ...options.headers
  }});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function controls() {
  const schedule = $("schedule-enabled").checked;
  const mutation = $("boundary").value === "mutation";
  const beta = $("initial-type").value === "beta";
  $("schedule-fields").hidden = !schedule;
  $("mutation-fields").hidden = !mutation;
  $("beta-fields").hidden = !beta;
  $("delta-fields").hidden = beta;
  for (const input of $("form-panel").querySelectorAll("input,select,textarea")) input.disabled = state.mode !== "form";
  if (state.mode === "form") {
    $("ne").disabled = schedule;
    $("schedule").disabled = !schedule;
    $("u").disabled = $("v").disabled = !mutation;
    $("alpha").disabled = $("beta").disabled = !beta;
    $("x0").disabled = beta;
    $("dt").disabled = $("method").value === "expm";
  }
  $("config-json").disabled = state.mode !== "json";
}

function configFromForm() {
  const n = (id) => Number($(id).value);
  const count = n("outputs"), end = n("end-time");
  if (!Number.isInteger(count) || count < 2 || count > 201) throw new Error("Output count must be an integer from 2 to 201.");
  let Ne = n("ne");
  if ($("schedule-enabled").checked) {
    const entries = $("schedule").value.trim().split(/\n/).filter((line) => line.trim()).map((line) => {
      const parts = line.split(/[:：]/);
      if (parts.length !== 2 || parts.some((p) => !p.trim() || !Number.isFinite(Number(p)))) {
        throw new Error("Enter one population change per line as start generation: population size.");
      }
      return parts.map(Number);
    });
    Ne = {type: "piecewise_constant", breakpoints: entries.map((e) => e[0]), values: entries.map((e) => e[1])};
  }
  const mutation = $("boundary").value === "mutation";
  const config = {schema_version: 1,
    model: {Ne, selection: n("selection"), dominance: n("dominance"),
      mutation_forward: mutation ? n("u") : 0, mutation_backward: mutation ? n("v") : 0,
      boundary: $("boundary").value},
    grid: {type: "uniform", points: n("points")},
    initial_condition: $("initial-type").value === "beta" ? {type: "beta", alpha: n("alpha"), beta: n("beta")} : {type: "delta", x0: n("x0")},
    solver: {method: $("method").value, output_times: Array.from({length: count}, (_, i) => end*i/(count-1))}
  };
  if ($("dt").value && $("method").value !== "expm") config.solver.dt = n("dt");
  return config;
}

function setMode(mode, generate = true) {
  if (mode === "json" && generate && state.mode !== "json") {
    try { $("config-json").value = JSON.stringify(configFromForm(), null, 2); }
    catch (error) { showError(error.message); return; }
  }
  state.mode = mode;
  for (const name of ["form", "json"]) {
    $(name + "-tab").setAttribute("aria-selected", String(name === mode));
    $(name + "-tab").classList.toggle("active", name === mode);
    $(name + "-panel").hidden = name !== mode;
  }
  controls();
}

function markDirty() {
  clearError();
  state.dirty = true;
  if (state.summary && !state.running) {
    status("Settings have changed. The plots show the previous result; run again to update.", "stale");
    $("result-badge").textContent = "PREVIOUS RESULT";
  }
}

function preset(name) {
  const values = {ne: 1000, selection: 0, dominance: 0.5, u: 0.0005, v: 0.0005,
    boundary: "absorbing", "initial-type": "delta", x0: 0.3, alpha: 2, beta: 5,
    "end-time": 1000, points: 501, outputs: 51, method: "auto", dt: ""};
  if (name === "selection") values.selection = 0.004;
  if (name === "mutation") { values.boundary = "mutation"; values.x0 = 0; }
  if (name === "bottleneck") { values.ne = 10000; values.method = "implicit_euler"; values.dt = 2; }
  for (const [id, value] of Object.entries(values)) $(id).value = value;
  $("schedule-enabled").checked = name === "bottleneck";
  $("schedule").value = "0: 10000\n500: 1000\n800: 10000";
  for (const button of document.querySelectorAll(".preset")) {
    const selected = button.dataset.preset === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  setMode("form");
  markDirty();
}

function restoreConfig(config) {
  // JSON mode preserves schedules and arbitrary output times without simplifying them.
  $("config-json").value = JSON.stringify(config, null, 2);
  setMode("json", false);
  for (const button of document.querySelectorAll(".preset")) {
    button.classList.remove("active"); button.setAttribute("aria-pressed", "false");
  }
  state.dirty = false;
}

function setRunning(running) {
  state.running = running;
  $("settings").disabled = running || state.sampling || state.sfsRunning;
  $("run").disabled = running || state.sampling || state.sfsRunning || !state.token;
  $("sample-paths").disabled = running || state.sampling || state.sfsRunning || !state.resultJob || !state.token;
  $("run-label").textContent = running ? "Calculating…" : "Run simulation";
  $("cancel").hidden = !running;
  $("simulation-form").setAttribute("aria-busy", String(running));
  const busy = running || state.sampling || state.sfsRunning;
  $("sfs-settings").disabled = busy;
  $("sfs-run").disabled = busy || !state.token;
  $("sfs-cancel").hidden = !state.sfsRunning;
  if (running) stopPlayback();
}

async function poll(jobId, version, restore = false) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (version !== state.version) return;
    if (job.status === "running") {
      status(`Calculating · ${job.elapsed.toFixed(1)} s${state.summary ? " · showing the previous result" : ""}`, "running");
      setTimeout(() => poll(jobId, version, restore), 400);
    } else if (job.status === "complete") {
      const summary = await api(`/api/jobs/${jobId}/summary.json`);
      if (version !== state.version) return;
      state.summary = summary; state.resultJob = jobId; state.dirty = false; state.frame = null;
      if (restore) restoreConfig(summary.metadata.config);
      resetSamples();
      displayResult(summary);
      await selectFrame(summary.times.length-1);
      if (version !== state.version) return;
      setRunning(false);
      status(`Complete · ${summary.metadata.config.grid.points.toLocaleString()} grid points · ${summary.times.length} output times`);
      restoreSamples(jobId);
    } else {
      setRunning(false);
      if (job.status === "cancelled") status("Calculation cancelled. Adjust the settings and run again.");
      else { status("The calculation did not complete."); showError(job.error || "Calculation failed."); }
    }
  } catch (error) {
    if (version !== state.version) return;
    setRunning(false); showError(`Could not retrieve results. Reload the page to reconnect.\n${error.message}`);
    status("Check the connection to the local server.");
  }
}

async function run(event) {
  event.preventDefault();
  if (state.running || state.sampling || state.sfsRunning || !state.token) return;
  clearError();
  let config;
  try { config = state.mode === "json" ? JSON.parse($("config-json").value) : configFromForm(); }
  catch (error) { showError(`Check the configuration.\n${error.message}`); return; }
  const version = ++state.version;
  state.job = null; $("cancel").disabled = true;
  setRunning(true); status("Preparing the calculation…", "running");
  try {
    const job = await api("/api/jobs", {method: "POST", body: JSON.stringify(config)});
    state.job = job.id;
    $("cancel").disabled = false;
    poll(job.id, version);
  } catch (error) {
    setRunning(false); showError(error.message); status("Check the input and run again.");
  }
}

function warningText(message, diagnostics) {
  if (message.includes("Peclet")) return "The Péclet number exceeds 2. Refine the grid and check spatial convergence.";
  if (message.includes("delta near")) return "The initial frequency is under-resolved. Increase the grid size to reduce initial endpoint allocation.";
  if (message.includes("continuous initial density")) return `The initial density allocates mass to endpoints (total ${format(diagnostics.initial_endpoint_mass.reduce((a,b) => a+b, 0), 6)}). Refine the grid to check its impact.`;
  if (message.includes("zero mutation")) return "Both mutation rates are zero, so the endpoints are absorbing.";
  return message;
}

function displayResult(summary) {
  const {metadata, diagnostics} = summary;
  const mutation = metadata.endpoint_semantics === "occupancy";
  $("loss-label").textContent = mutation ? "Endpoint occupancy P₀" : "Loss probability P₀";
  $("fix-label").textContent = mutation ? "Endpoint occupancy P₁" : "Fixation probability P₁";
  $("loss-hint").textContent = mutation ? "Probability of occupying x = 0" : "Allele frequency x = 0";
  $("fix-hint").textContent = mutation ? "Probability of occupying x = 1" : "Allele frequency x = 1";
  $("boundary-title").textContent = mutation ? "Endpoint occupancy & interior mass" : "Loss, fixation & interior mass";
  $("boundary-subtitle").textContent = mutation ? "Includes re-entry from endpoints" : "Probability through time";
  $("legend-zero").textContent = mutation ? "x = 0" : "Loss";
  $("legend-one").textContent = mutation ? "x = 1" : "Fixation";
  $("result-badge").textContent = mutation ? "Mutation boundary" : "Absorbing boundary";
  $("downloads").hidden = false;
  for (const [id, file] of [["config", "config.json"], ["csv", "summary.csv"], ["npz", "result.npz"]]) {
    $("download-"+id).href = `/api/jobs/${state.resultJob}/${file}`;
    $("download-"+id).download = "wf-forward-"+file;
  }
  $("mass-error").textContent = Number(diagnostics.max_probability_error).toExponential(2);
  $("peclet").textContent = format(diagnostics.max_peclet, 3);
  $("actual-method").textContent = metadata.method === "expm" ? "Matrix exponential" : "Implicit Euler";
  $("runtime").textContent = format(metadata.elapsed_seconds, 3) + " s";
  $("diagnostic-status").textContent = diagnostics.correction_count ? `${diagnostics.correction_count} roundoff corrections` : "Mass and nonnegativity checks passed";
  $("warnings").replaceChildren();
  for (const message of summary.warnings || []) {
    const li = document.createElement("li"); li.textContent = warningText(message, diagnostics); $("warnings").append(li);
  }
  $("warnings").hidden = !$("warnings").children.length;
  $("time-slider").max = summary.times.length-1;
  $("time-slider").disabled = summary.times.length < 2;
  $("play").disabled = summary.times.length < 2;
  $("time-start").textContent = format(summary.times[0]) + " generations";
  $("time-end").textContent = format(summary.times.at(-1)) + " generations";
  $("density-empty").hidden = true;
}

async function selectFrame(index) {
  if (!state.summary) return;
  const job = state.resultJob;
  index = Math.min(Math.max(index, 0), state.summary.times.length-1);
  state.frameAbort?.abort();
  const controller = new AbortController(); state.frameAbort = controller;
  const request = ++state.frameRequest;
  try {
    const frame = await api(`/api/jobs/${job}/frame?index=${index}`, {signal: controller.signal});
    if (request !== state.frameRequest || job !== state.resultJob) return;
    state.index = index; state.frame = frame;
    $("time-slider").value = index;
    $("selected-time").textContent = format(frame.time);
    $("frame-index").textContent = `${index+1} / ${state.summary.times.length}`;
    $("mean-value").textContent = metric(state.summary.mean[index]);
    $("loss-value").textContent = metric(state.summary.p_at_zero[index]);
    $("fix-value").textContent = metric(state.summary.p_at_one[index]);
    $("hetero-value").textContent = metric(state.summary.heterozygosity[index]);
    $("density-note").textContent = frame.aggregated ? "Mass-preserving bin aggregation · endpoints shown separately" : "Endpoint probability mass is shown separately below";
    $("density-chart").setAttribute("aria-label", `${format(frame.time)} generations: interior density. Mean frequency ${metric(state.summary.mean[index])}`);
    drawAll();
  } catch (error) { if (error.name !== "AbortError") { stopPlayback(); showError(error.message); } }
}

function stopPlayback() {
  if (state.playing) clearInterval(state.playing);
  state.playing = null; $("play").textContent = "▶"; $("play").setAttribute("aria-label", "Play time evolution");
}

function startPlayback() {
  if (state.playing) { stopPlayback(); return; }
  if (!state.summary) return;
  let index = state.index >= state.summary.times.length-1 ? 0 : state.index+1;
  selectFrame(index);
  $("play").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "Pause playback");
  state.playing = setInterval(() => {
    index += 1;
    if (index >= state.summary.times.length) { stopPlayback(); return; }
    selectFrame(index);
  }, 350);
}

function niceMax(maximum) {
  if (!(maximum > 0)) return 1;
  const magnitude = 10**Math.floor(Math.log10(maximum));
  return Math.ceil(maximum/magnitude*2)/2*magnitude;
}

function drawChart(canvas, {series = [], domainX = [0,1], ymax = 1, marker, xlabel = "", ylabel = "", fill = false, logY = false, ymin = 0} = {}) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width*ratio); canvas.height = Math.round(bounds.height*ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const W = bounds.width, H = bounds.height;
  const margin = {l: 44, r: 15, t: 19, b: 35};
  const width = W-margin.l-margin.r, height = H-margin.t-margin.b;
  const [xmin, xmax] = domainX;
  const X = (x) => margin.l+(x-xmin)/(xmax-xmin || 1)*width;
  const Y = (y) => margin.t+height-(logY ? (Math.log10(y)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)) : y/ymax)*height;
  ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif'; ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const y = logY ? 10**(Math.log10(ymin)+(Math.log10(ymax)-Math.log10(ymin))*i/4) : ymax*i/4;
    ctx.strokeStyle = "#eaf0e8"; ctx.setLineDash([3,4]); ctx.beginPath(); ctx.moveTo(margin.l,Y(y)); ctx.lineTo(W-margin.r,Y(y)); ctx.stroke();
    ctx.fillStyle = "#99a79b"; ctx.fillText(logY ? y.toExponential(1) : format(y, 3), margin.l-9,Y(y));
  }
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const x = xmin+(xmax-xmin)*i/4;
    ctx.strokeStyle = "#eef2ec"; ctx.beginPath(); ctx.moveTo(X(x),margin.t); ctx.lineTo(X(x),H-margin.b); ctx.stroke();
    ctx.fillStyle = "#99a79b"; ctx.fillText(format(x, 2),X(x),H-margin.b+14);
  }
  ctx.setLineDash([]); ctx.fillStyle = "#94a195";
  ctx.fillText(xlabel, margin.l+width/2, H-5); ctx.textAlign = "left"; ctx.fillText(ylabel, margin.l, 7);
  ctx.save(); ctx.beginPath(); ctx.rect(margin.l,margin.t,width,height); ctx.clip();
  for (let k = 0; k < series.length; k++) {
    const {x, y, color, opacity = 1, lineWidth = 2, dash = []} = series[k];
    ctx.globalAlpha = opacity; ctx.setLineDash(dash);
    if (!x.length) continue;
    if (fill && k === 0) {
      const gradient = ctx.createLinearGradient(0,margin.t,0,H-margin.b);
      gradient.addColorStop(0,"#127d7829"); gradient.addColorStop(1,"#127d7803");
      ctx.beginPath(); ctx.moveTo(X(x[0]),Y(0));
      for (let i = 0; i < x.length; i++) ctx.lineTo(X(x[i]),Y(y[i]));
      ctx.lineTo(X(x.at(-1)),Y(0)); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
    }
    ctx.beginPath();
    let connected = false;
    x.forEach((v, i) => {
      if (!Number.isFinite(y[i]) || (logY && y[i] <= 0)) { connected = false; return; }
      if (connected) ctx.lineTo(X(v),Y(y[i])); else ctx.moveTo(X(v),Y(y[i]));
      connected = true;
    });
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = "round"; ctx.stroke();
    if (x.length === 1 && (!logY || y[0] > 0)) { ctx.beginPath(); ctx.arc(X(x[0]),Y(y[0]),3,0,Math.PI*2); ctx.fillStyle = color; ctx.fill(); }
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
  if (marker !== undefined) {
    ctx.beginPath(); ctx.setLineDash([4,4]); ctx.strokeStyle = "#91a791"; ctx.lineWidth = 1;
    ctx.moveTo(X(marker),margin.t); ctx.lineTo(X(marker),H-margin.b); ctx.stroke();
  }
  ctx.restore();
}

function drawAll() {
  drawSfs();
  const summary = state.summary, frame = state.frame;
  const peak = frame ? Math.max(...frame.density) : 0;
  const densityMax = state.densityYMax ?? niceMax(peak*1.08);
  $("axis-hint").textContent = state.densityYMax === null ? "Auto-scale at each time point" :
    peak > state.densityYMax ? "Density above the fixed maximum is clipped in this view" : "The same scale is used at every time point";
  if ($("lock-density-axis").checked && $("density-ymax").getAttribute("aria-invalid") === "true") {
    $("axis-hint").textContent = "Enter a positive maximum (keeping the last valid value)";
  }
  $("density-chart").dataset.yMax = String(densityMax);
  drawChart($("density-chart"), {series: frame ? [{x:frame.x,y:frame.density,color:colors.teal}] : [],
    ymax: densityMax, fill:true, xlabel:"Allele frequency x", ylabel:"Density"});
  const times = summary?.times || [0,1000];
  const domainX = times.length > 1 ? [times[0],times.at(-1)] : [0,Math.max(1,times[0])];
  const marker = frame?.time;
  const paths = state.trajectories;
  const pathSeries = paths ? paths.frequencies.map((y) => ({x:paths.times,y,color:colors.indigo,opacity:0.45,lineWidth:1})) : [];
  if (paths && summary) pathSeries.push({x:times,y:summary.mean,color:colors.teal,lineWidth:2.5,dash:[6,4]});
  drawChart($("trajectory-chart"), {series:pathSeries,domainX,marker,xlabel:"generations",ylabel:"Allele frequency x"});
  drawChart($("probability-chart"), {series: summary ? [
    {x:times,y:summary.p_at_zero,color:colors.coral}, {x:times,y:summary.p_at_one,color:colors.indigo},
    {x:times,y:summary.p_interior,color:colors.teal}] : [], domainX, marker, xlabel:"generations", ylabel:"Probability"});
  drawChart($("moments-chart"), {series: summary ? [
    {x:times,y:summary.mean,color:colors.teal}, {x:times,y:summary.heterozygosity,color:colors.gold}] : [],
    domainX, marker, ymax: summary ? niceMax(Math.max(...summary.mean,...summary.heterozygosity)*1.1) : 1,
    xlabel:"generations", ylabel:"Mean / heterozygosity"});
}

function samplingStatus(message, error = false) {
  $("sampling-status").textContent = message;
  $("sampling-status").classList.toggle("error", error);
}
function setSampling(value) {
  state.sampling = value;
  $("path-count").disabled = $("path-seed").disabled = value;
  $("cancel-sampling").hidden = !value;
  $("trajectory-form").setAttribute("aria-busy", String(value));
  setRunning(state.running);
}
function resetSamples() {
  ++state.sampleVersion; state.trajectories = null;
  $("trajectory-downloads").hidden = true;
  setSampling(false);
  samplingStatus("Choose a path count and seed, then sample trajectories.");
}
async function loadSamples(parent, version) {
  const paths = await api(`/api/jobs/${parent}/trajectories.json`);
  if (parent !== state.resultJob || version !== state.sampleVersion) return;
  state.trajectories = paths;
  for (const ext of ["csv", "npz"]) {
    $("download-path-"+ext).href = `/api/jobs/${parent}/trajectories.${ext}`;
    $("download-path-"+ext).download = `wf-trajectories.${ext}`;
  }
  $("trajectory-downloads").hidden = false;
  const m = paths.metadata;
  samplingStatus(`${m.paths} paths · seed ${m.seed} · ${m.events.toLocaleString()} jumps · ${format(m.elapsed_seconds, 2)} s`);
  $("trajectory-chart").setAttribute("aria-label", `${m.paths} sampled allele-frequency paths, seed ${m.seed}, from ${paths.times[0]} to ${paths.times.at(-1)} generations`);
  drawAll();
}
async function pollSamples(parent, version, restore = false) {
  try {
    const report = await api(`/api/jobs/${parent}/sampling`);
    if (parent !== state.resultJob || version !== state.sampleVersion) return;
    if (restore && report.available) await loadSamples(parent, version);
    if (parent !== state.resultJob || version !== state.sampleVersion) return;
    const job = report.current;
    if (job?.status === "running") {
      setSampling(true);
      samplingStatus(`Sampling trajectories · ${format(job.elapsed, 1)} s${state.trajectories ? " · showing previous samples" : ""}`);
      setTimeout(() => pollSamples(parent, version), 400);
    } else {
      setSampling(false);
      if (job?.status === "complete" && !restore) await loadSamples(parent, version);
      else if (job && job.status !== "complete") samplingStatus(
        (job.status === "cancelled" ? "Sampling cancelled." : job.error || "Sampling failed.") +
        (state.trajectories ? " Previous samples are retained." : ""), job.status === "failed");
    }
  } catch (error) {
    if (parent !== state.resultJob || version !== state.sampleVersion) return;
    setSampling(false); samplingStatus(`${error.message} Reload to reconnect.`, true);
  }
}
function restoreSamples(parent) { pollSamples(parent, ++state.sampleVersion, true); }
$("trajectory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.resultJob || state.running || state.sampling || state.sfsRunning) return;
  const options = {paths:Number($("path-count").value),seed:Number($("path-seed").value)};
  const parent = state.resultJob, version = ++state.sampleVersion;
  setSampling(true); $("cancel-sampling").disabled = true; samplingStatus("Preparing trajectory sampling…");
  try {
    await api(`/api/jobs/${parent}/sample`, {method:"POST",body:JSON.stringify(options)});
    $("cancel-sampling").disabled = false;
    pollSamples(parent, version);
  } catch (error) { setSampling(false); samplingStatus(error.message, true); }
});
$("cancel-sampling").addEventListener("click", async () => {
  $("cancel-sampling").disabled = true;
  try { await api(`/api/jobs/${state.resultJob}/cancel-sampling`, {method:"POST",body:"{}"}); }
  catch (error) { samplingStatus(error.message, true); }
  finally { $("cancel-sampling").disabled = false; }
});

function sfsStatus(message, error = false) {
  $("sfs-status").textContent = message;
  $("sfs-status").classList.toggle("error", error);
}
function setSfsRunning(value) {
  state.sfsRunning = value;
  $("sfs-form").setAttribute("aria-busy", String(value));
  setRunning(state.running);
}
function sfsConfig() {
  const rows = $("sfs-history").value.trim().split(/\n/).filter((line) => line.trim()).map((line) => {
    const parts = line.split(":");
    if (parts.length !== 2 || parts.some((v) => !v.trim() || !Number.isFinite(Number(v))))
      throw new Error("Enter each SFS history row as generation: effective population size.");
    return parts.map(Number);
  });
  const end = Number($("sfs-end").value);
  return {n:Number($("sfs-n").value),theta:Number($("sfs-theta").value),reference_Ne:Number($("sfs-ne").value),
    history:{times:rows.map((r) => r[0]),sizes:rows.map((r) => r[1]),interpolation:$("sfs-interpolation").value},
    output_times:Array.from({length:51}, (_,i) => end*i/50)};
}
async function pollSfs(jobId, version, restore = false) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (version !== state.sfsVersion) return;
    if (job.status === "running") {
      sfsStatus(`Calculating SFS · ${format(job.elapsed,1)} s${state.sfsResult ? " · showing previous result" : ""}`);
      setTimeout(() => pollSfs(jobId, version, restore),400);
      return;
    }
    if (job.status !== "complete") {
      setSfsRunning(false);
      sfsStatus(job.status === "cancelled" ? "SFS calculation cancelled; any previous result is retained." : job.error || "SFS calculation failed.", job.status !== "cancelled");
      return;
    }
    const result = await api(`/api/jobs/${jobId}/sfs.json`);
    if (version !== state.sfsVersion) return;
    state.sfsResult = result; state.sfsIndex = result.times.length-1;
    const c = result.metadata.config;
    if (restore) {
      $("sfs-n").value = c.n; $("sfs-theta").value = c.theta; $("sfs-ne").value = c.reference_Ne;
      $("sfs-end").value = c.output_times.at(-1);
      $("sfs-preset").value = "custom";
      $("sfs-interpolation").value = c.history.interpolation;
      $("sfs-history").value = c.history.times.map((t,i) => `${t}: ${c.history.sizes[i]}`).join("\n");
    }
    $("sfs-time").max = result.times.length-1; $("sfs-time").value = state.sfsIndex;
    $("sfs-time").disabled = result.times.length < 2;
    for (const ext of ["csv","npz","config"]) {
      $("sfs-"+ext).href = `/api/jobs/${jobId}/${ext === "config" ? "config.json" : "sfs."+ext}`;
      $("sfs-"+ext).download = ext === "config" ? "sfs-config.json" : "sfs."+ext;
    }
    $("sfs-downloads").hidden = false;
    setSfsRunning(false);
    sfsStatus(`Complete · n = ${c.n} · θ = ${format(c.theta)} · ${format(result.metadata.elapsed_seconds,3)} s · neutral demographic model`);
    drawSfs();
  } catch (error) {
    if (version !== state.sfsVersion) return;
    setSfsRunning(false); sfsStatus(`${error.message} Reload to reconnect.`,true);
  }
}
function sfsValues(row, n) {
  let values = row.slice();
  if ($("sfs-fold").value === "folded") values = Array.from({length:Math.floor(n/2)}, (_,k) => {
    const i = k+1; return i*2 === n ? row[k] : row[k]+row[n-i-1];
  });
  if ($("sfs-normalization").value === "relative") {
    const total = values.reduce((a,b) => a+b,0);
    values = values.map((v) => total > 0 ? v/total : 0);
  }
  return values;
}
function drawSfs() {
  const result = state.sfsResult, mode = $("sfs-mode").value;
  const relative = $("sfs-normalization").value === "relative";
  const folded = $("sfs-fold").value === "folded";
  const series = [];
  let n = 100, maximum = 1, minimum = 1;
  if (result) {
    const c = result.metadata.config; n = c.n;
    const index = state.sfsIndex;
    const values = sfsValues(result[mode][index],n);
    const x = values.map((_,i) => i+1);
    series.push({x,y:values,color:colors.teal});
    if ($("sfs-compare").checked) {
      const other = mode === "symmetric" ? "derived" : "symmetric";
      series.push({x,y:sfsValues(result[other][index],n),color:colors.gold,dash:[3,3]});
    }
    if ($("sfs-reference").checked) {
      const amplitude = c.theta*c.history.sizes[0]/c.reference_Ne;
      const reference = Array.from({length:n-1}, (_,k) => amplitude*(1/(k+1)+(mode === "symmetric" ? 1/(n-k-1) : 0)));
      series.push({x,y:sfsValues(reference,n),color:colors.indigo,dash:[6,4],lineWidth:1.5});
    }
    const positive = series.flatMap((s) => s.y).filter((v) => v>0);
    maximum = positive.length ? Math.max(...positive) : 1;
    minimum = positive.length ? Math.min(...positive) : 1;
    const total = result[mode][index].reduce((a,b) => a+b,0);
    $("sfs-time-label").textContent = `${format(result.times[index])} generations`;
    $("sfs-summary").textContent = `n = ${n} · θ = ${format(c.theta)} · Expected polymorphic sites: ${format(total,5)} · ${folded ? "Minor-count" : "Count"} 1: ${format(values[0],6)}${relative ? " (proportion)" : " sites"}. `+
      (total === 0 ? "No expected sites; normalized proportions are undefined and shown as zero." : "Monomorphic classes are excluded. Downloads contain both unfolded expected spectra at all saved times.");
    $("sfs-chart").setAttribute("aria-label", `${mode} ${folded ? "folded" : "unfolded"} SFS at ${result.times[index]} generations, n ${n}, theta ${c.theta}, expected polymorphic sites ${total}`);
  }
  const logY = $("sfs-log").checked;
  let ymin = 10**Math.floor(Math.log10(minimum)), ymax = 10**Math.ceil(Math.log10(maximum));
  if (ymin === ymax) { ymin /= 10; ymax *= 10; }
  $("sfs-other-legend").hidden = !$("sfs-compare").checked;
  $("sfs-reference-legend").hidden = !$("sfs-reference").checked;
  drawChart($("sfs-chart"), {series, domainX:[1,Math.max(2,folded ? Math.floor(n/2) : n-1)],
    ymax:logY ? ymax : niceMax(maximum*1.08),ymin,logY,
    xlabel:folded ? "Minor allele count i" : "Allele count i",ylabel:relative ? "Polymorphic proportion" : "Expected sites"});
}
$("sfs-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.running || state.sampling || state.sfsRunning || !state.token) return;
  let config;
  try { config = sfsConfig(); } catch (error) { sfsStatus(error.message,true); return; }
  const version = ++state.sfsVersion;
  state.sfsJob = null; setSfsRunning(true); $("sfs-cancel").disabled = true;
  sfsStatus("Preparing demographic SFS…");
  try {
    const job = await api("/api/sfs",{method:"POST",body:JSON.stringify(config)});
    state.sfsJob = job.id; $("sfs-cancel").disabled = false;
    pollSfs(job.id,version);
  } catch (error) { setSfsRunning(false); sfsStatus(error.message,true); }
});
$("sfs-cancel").addEventListener("click", async () => {
  if (!state.sfsJob) return;
  $("sfs-cancel").disabled = true;
  try { await api(`/api/jobs/${state.sfsJob}/cancel`,{method:"POST",body:"{}"}); }
  catch (error) { sfsStatus(error.message,true); }
  finally { $("sfs-cancel").disabled = false; }
});
$("sfs-settings").addEventListener("input", (event) => {
  if (event.target.id === "sfs-history") $("sfs-preset").value = "custom";
  if (state.sfsResult) sfsStatus("SFS settings changed. Plots and downloads show the previous result until recalculated.");
});
$("sfs-preset").addEventListener("change", () => {
  const size = Number($("sfs-ne").value) || 10000, end = Number($("sfs-end").value) || 10000;
  const preset = $("sfs-preset").value;
  if (preset === "custom") return;
  $("sfs-history").value = preset === "bottleneck" ? `0: ${size}\n${end*.4}: ${Math.max(1,size*.1)}\n${end*.6}: ${size}` :
    preset === "growth" ? `0: ${size}\n${end*.8}: ${size}\n${end}: ${size*10}` : `0: ${size}`;
  if (state.sfsResult) sfsStatus("SFS history changed. Calculate again to update the result.");
});
for (const id of ["sfs-mode","sfs-fold","sfs-normalization","sfs-log","sfs-compare","sfs-reference"])
  $(id).addEventListener("change",drawSfs);
$("sfs-time").addEventListener("input", () => { state.sfsIndex = Number($("sfs-time").value); drawSfs(); });

$("simulation-form").addEventListener("submit", run);
$("settings").addEventListener("input", () => { controls(); markDirty(); });
$("settings").addEventListener("change", () => { controls(); markDirty(); });
for (const button of document.querySelectorAll(".preset")) button.addEventListener("click", () => preset(button.dataset.preset));
$("form-tab").addEventListener("click", () => { setMode("form"); markDirty(); });
$("json-tab").addEventListener("click", () => { setMode("json"); markDirty(); });
$("time-slider").addEventListener("input", () => { stopPlayback(); selectFrame(Number($("time-slider").value)); });
$("play").addEventListener("click", startPlayback);
$("lock-density-axis").addEventListener("change", () => {
  const locked = $("lock-density-axis").checked;
  $("density-ymax").disabled = !locked;
  $("density-ymax").setCustomValidity("");
  $("density-ymax").setAttribute("aria-invalid", "false");
  if (locked) {
    state.densityYMax = Number($("density-chart").dataset.yMax) || 1;
    $("density-ymax").value = state.densityYMax;
  } else state.densityYMax = null;
  drawAll();
});
$("density-ymax").addEventListener("input", () => {
  const value = Number($("density-ymax").value);
  const valid = Number.isFinite(value) && value > 0;
  $("density-ymax").setCustomValidity(valid ? "" : "The maximum must be a finite number greater than zero.");
  $("density-ymax").setAttribute("aria-invalid", String(!valid));
  if (valid && $("lock-density-axis").checked) { state.densityYMax = value; drawAll(); }
  else $("axis-hint").textContent = "Enter a positive maximum (keeping the last valid value)";
});
$("cancel").addEventListener("click", async () => {
  if (!state.job) return;
  $("cancel").disabled = true;
  try { await api(`/api/jobs/${state.job}/cancel`, {method:"POST",body:"{}"}); }
  catch (error) { showError(error.message); }
  finally { $("cancel").disabled = false; }
});
const resize = new ResizeObserver(drawAll);
document.querySelectorAll(".chart-wrap").forEach((el) => resize.observe(el));

async function initialize() {
  controls(); setRunning(false); drawAll();
  if (window.location.protocol === "file:") {
    showError("Do not open this HTML file directly. Run wf-forward serve in a terminal, then open http://127.0.0.1:8765/.");
    status("Open the UI through the local server.");
    return;
  }
  try {
    const session = await api("/api/session"); state.token = session.token; setRunning(false);
    if (session.latest_sfs && ["running","complete"].includes(session.latest_sfs.status)) {
      state.sfsJob = session.latest_sfs.id; setSfsRunning(true);
      pollSfs(state.sfsJob,++state.sfsVersion,true);
    }
    if (session.latest && ["running","complete"].includes(session.latest.status)) {
      state.job = session.latest.id; setRunning(true);
      status("Loading the previous result…", "running");
      poll(session.latest.id, ++state.version, true);
    }
  } catch (error) { showError(`Cannot connect to the server. Check that wf-forward serve is running.\n${error.message}`); }
}
initialize();
