/* Reflex Guard frontend
 * - Real camera capture (getUserMedia) stays entirely client-side.
 * - Device telemetry (AI / safety / relay / tool) comes from the server
 *   over Socket.IO, shared live across every connected browser.
 */

const socket = io();

// ----------------------------------------------------------------- tabs
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// -------------------------------------------------------- data source
document.querySelectorAll(".ds-btn").forEach(btn => {
  btn.addEventListener("click", () => setDataSource(btn.dataset.mode));
});

function setDataSource(mode) {
  socket.emit("command", { action: "set_mode", value: mode });
  document.querySelectorAll(".ds-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  localStorage.setItem("rg_data_mode", mode);
  togglePcCameraPanel(mode);
}

function togglePcCameraPanel(mode) {
  const pcPanel = document.getElementById("pcCameraPanel");
  const browserPanel = document.getElementById("browserCameraPanel");
  const img = document.getElementById("pcCameraImg");
  if (!pcPanel || !browserPanel) return;
  if (mode === "pc") {
    pcPanel.style.display = "";
    browserPanel.style.display = "none";
    if (img && !img.src) img.src = "/video_feed_pc?" + Date.now();
  } else {
    pcPanel.style.display = "none";
    browserPanel.style.display = "";
  }
}
{
  const saved = localStorage.getItem("rg_data_mode");
  if (saved === "live" || saved === "pc") setDataSource(saved);
}

// ---------------------------------------------------------- socket wiring
socket.on("connect", () => document.getElementById("socketDot").classList.add("on"));
socket.on("disconnect", () => document.getElementById("socketDot").classList.remove("on"));

socket.on("state_update", renderState);
socket.on("event_log", entry => { addEventRow(entry); pushChartsFromEvent(entry); });
socket.on("event_log_bulk", entries => entries.forEach(addEventRow));
socket.on("event_log_cleared", () => { document.getElementById("eventsBody").innerHTML = ""; });
socket.on("system_log", entry => appendTerminal(`[${entry.time}] ${entry.message}`));

// command buttons (data-cmd="{...json...}")
document.querySelectorAll("[data-cmd]").forEach(btn => {
  btn.addEventListener("click", () => {
    const cmd = JSON.parse(btn.dataset.cmd);
    socket.emit("command", cmd);
  });
});

// -------------------------------------------------------------- state render
function renderState(state) {
  checkDangerFlash(state);
  togglePcCameraPanel(state.mode);
  document.querySelectorAll(".ds-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === state.mode);
  });
  setCard("system", state.system.state, state.system.state);
  setCard("ai", state.ai.connected ? state.ai.state : "DISCONNECTED", state.ai.connected ? state.ai.state : "DISCONNECTED");
  setCard("safety", state.safety.state, state.safety.state);
  setCard("tool", state.tool.power, state.tool.power);
  setCard("relay", state.relay.state, state.relay.state);

  setCard("unoq", state.mode === "simulation" ? "SIMULATION" : (state.hardwareConnected ? "CONNECTED" : "OFFLINE"));
  setCard("qnn", state.mode === "simulation" ? "SIMULATION" : (state.hardwareConnected ? "LOADED" : "NOT CONNECTED"));
  setCard("stm32hw", state.mode === "simulation" ? "SIMULATION" : (state.hardwareConnected ? "ARMED" : "OFFLINE"));
  setCard("relayhw", state.relay.state);
  setCard("estop", state.system.state === "FAULT" ? "ACTIVATED" : "READY");

  document.getElementById("dsStatus").textContent = state.mode === "simulation"
    ? "No physical hardware connected"
    : (state.hardwareConnected ? "Hardware connected" : "HARDWARE NOT CONNECTED");

  document.getElementById("uptimeVal").textContent = formatUptime(state.uptimeSeconds);

  highlightStateMachine(state);

  if (state.safety.heartbeat) {
    document.getElementById("heartbeatVal").textContent = state.safety.heartbeat;
    const dot = document.getElementById("heartbeatDot");
    dot.classList.toggle("on", state.system.running);
    dot.classList.toggle("off", !state.system.running);
  }

  if (state.system.running && state.mode === "simulation") {
    pushChartPoint(chartLatency, state.ai.latencyMs);
    pushChartPoint(chartConfidence, state.ai.confidence);
  }

  if (state.aiModel) renderAiModel(state.aiModel, state.mode);
  if (state.twin) renderTwin(state.twin, state);
}

// --------------------------------------------------------------- AI model
function renderAiModel(ai, mode) {
  document.getElementById("aiModelName").textContent = ai.name || "—";
  document.getElementById("aiModelSource").textContent = ai.source || "";
  document.getElementById("aiModelQuant").textContent = ai.quantization || "—";
  document.getElementById("aiModelBackend").textContent = ai.backend || "—";

  const realTag = document.getElementById("aiModelReal");
  realTag.textContent = ai.real ? "REAL ON-DEVICE INFERENCE" : "SIMULATED TELEMETRY";
  realTag.className = "tag " + (ai.real ? "real" : "sim");

  const pct = ai.npuUtilPct;
  const gauge = document.getElementById("npuGaugeFill");
  const circumference = 2 * Math.PI * 33;
  const frac = pct != null ? Math.max(0, Math.min(100, pct)) / 100 : 0;
  gauge.setAttribute("stroke-dasharray", `${(frac * circumference).toFixed(1)} ${circumference.toFixed(1)}`);
  document.getElementById("npuGaugePct").textContent = pct != null ? `${pct.toFixed(0)}%` : "—";
  document.getElementById("npuLatencyVal").textContent =
    (state_latencyMs(ai) != null) ? `${state_latencyMs(ai).toFixed(2)} ms` : "— ms";
}
function state_latencyMs(ai) { return ai.latencyMs != null ? ai.latencyMs : null; }

// -------------------------------------------------------------- digital twin
function renderTwin(t, fullState) {
  const pill = document.getElementById("twinHealthPill");
  pill.className = "health-pill " + (t.health || "NO_DATA");
  document.getElementById("twinHealthText").textContent = (t.health || "NO DATA").replace("_", " ");

  document.getElementById("twinTtd").textContent = t.timeToDangerMs != null ? `${t.timeToDangerMs.toFixed(0)} ms` : "no closing motion";
  document.getElementById("twinDist").textContent = t.handDistanceMm != null ? `${t.handDistanceMm.toFixed(0)} (proxy units)` : "—";
  document.getElementById("twinPredicted").textContent = `${t.predictedResponseMs.toFixed(2)} ms`;
  document.getElementById("twinPredictedWorst").textContent = `${t.predictedResponseWorstMs.toFixed(2)} ms`;
  document.getElementById("twinActual").textContent = t.lastActualResponseMs != null ? `${t.lastActualResponseMs.toFixed(2)} ms` : "no event yet";
  document.getElementById("twinDrift").textContent = t.driftMs != null
    ? `${t.driftMs >= 0 ? "+" : ""}${t.driftMs.toFixed(2)} ms (${(t.driftPct * 100).toFixed(0)}%)` : "—";

  if (t.stageBreakdown) {
    const list = document.getElementById("twinBudgetList");
    const maxMs = Math.max(...Object.values(t.stageBreakdown).map(s => s.worst_ms), 0.1);
    list.innerHTML = Object.entries(t.stageBreakdown).map(([stage, s]) => `
      <div class="twin-budget-row">
        <span class="stage">${stage.replace(/_/g, " ")}</span>
        <span style="font-family:var(--font-mono);color:var(--muted)">${s.nominal_ms.toFixed(1)} ms</span>
        <div class="twin-budget-bar"><span style="width:${(s.nominal_ms / maxMs * 100).toFixed(0)}%"></span></div>
      </div>`).join("");
  }

  animateTwinStage(t, fullState);
}

function animateTwinStage(t, fullState) {
  const hand = document.getElementById("twinHand");
  const zoneNear = 340, zoneFar = 480;
  let x = zoneFar;
  if (t.handDistanceMm != null) {
    const clamped = Math.max(0, Math.min(200, t.handDistanceMm));
    x = zoneNear + (clamped / 200) * (zoneFar - zoneNear);
  }
  hand.setAttribute("transform", `translate(${x.toFixed(1)},178)`);
  hand.classList.toggle("close", t.handDistanceMm != null && t.handDistanceMm < 60);

  const aiState = fullState && fullState.ai ? fullState.ai.state : "IDLE";
  const npuDot = document.getElementById("twinNpuDot");
  const npuActive = aiState === "DANGER" || aiState === "HAND_DETECTED" || aiState === "MONITORING";
  npuDot.setAttribute("fill", aiState === "DANGER" ? "#ff4433" : aiState === "IDLE" ? "#5f696d" : "#3dd6c6");
  npuDot.classList.toggle("active", npuActive);

  const zone = document.getElementById("twinDangerZone");
  zone.classList.toggle("armed", fullState && fullState.system && fullState.system.running);

  const stage = document.querySelector(".twin-stage");
  if (stage) stage.classList.toggle("danger-active", aiState === "DANGER");

  const relayOpen = fullState && fullState.relay && fullState.relay.state === "OPEN";
  const arm = document.getElementById("twinRelayArm");
  arm.setAttribute("x2", relayOpen ? "18" : "34");
  arm.setAttribute("y2", relayOpen ? "-14" : "0");
  arm.setAttribute("stroke", relayOpen ? "#ff4433" : "#2fbf6b");

  // Fire the traveling signal pulse + relay spark exactly once on the
  // CLOSED -> OPEN transition (a real state edge, not a looping animation).
  if (relayOpen && !_twinPrevRelayOpen) {
    const pulse = document.getElementById("twinSignalPulse");
    pulse.classList.remove("firing"); void pulse.offsetWidth; pulse.classList.add("firing");
    const spark = document.getElementById("twinSpark");
    spark.classList.remove("burst"); void spark.offsetWidth; spark.classList.add("burst");
  }
  _twinPrevRelayOpen = relayOpen;

  const spin = document.getElementById("twinMotorSpin");
  const motor = document.getElementById("twinMotor");
  const toolOn = fullState && fullState.tool && fullState.tool.power === "ON";
  spin.style.animation = toolOn ? "twinspin 0.4s linear infinite" : "none";
  motor.classList.toggle("running", toolOn);
  motor.classList.toggle("tripped", relayOpen && !toolOn);

  const aiCard = document.querySelector(".ai-model-card");
  if (aiCard) aiCard.classList.toggle("active-glow", npuActive);
  const gaugeFill = document.getElementById("npuGaugeFill");
  if (gaugeFill) gaugeFill.classList.toggle("pulsing", npuActive);
}
let _twinPrevRelayOpen = false;
if (!document.getElementById("twinSpinKeyframes")) {
  const style = document.createElement("style");
  style.id = "twinSpinKeyframes";
  style.textContent = "@keyframes twinspin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }";
  document.head.appendChild(style);
}

const DANGER_CLASSES = new Set(["DANGER", "FAULT", "TRIPPED", "OPEN", "OFF", "DISCONNECTED", "OFFLINE", "ANOMALY"]);
const SAFE_CLASSES = new Set(["SAFE", "ARMED", "ON", "CLOSED", "CONNECTED", "READY", "NOMINAL"]);
const _prevCardState = {};

function setCard(key, value, stateClass) {
  document.querySelectorAll(`[data-card="${key}"]`).forEach(card => {
    card.querySelector(".value").textContent = value;
    card.className = "stat-card" + (stateClass ? " state-" + stateClass : "");
    card.dataset.card = key;
  });

  const prev = _prevCardState[key];
  if (stateClass && prev !== undefined && prev !== stateClass) {
    const cls = DANGER_CLASSES.has(stateClass) ? "flash-danger" : SAFE_CLASSES.has(stateClass) ? "flash-safe" : null;
    if (cls) {
      document.querySelectorAll(`[data-card="${key}"]`).forEach(card => {
        card.classList.remove("flash-danger", "flash-safe");
        void card.offsetWidth; // restart animation
        card.classList.add(cls);
      });
    }
  }
  _prevCardState[key] = stateClass;
}

let _prevSystemState = null;
function checkDangerFlash(state) {
  const cur = state.system.state;
  const wasSafeish = _prevSystemState !== "DANGER" && _prevSystemState !== "FAULT";
  if (wasSafeish && (cur === "DANGER" || cur === "FAULT")) {
    const ms = (state.twin && state.twin.lastActualResponseMs != null)
      ? `${state.twin.lastActualResponseMs.toFixed(2)} ms` : "—";
    document.getElementById("dangerFlashMs").textContent = ms;
    const overlay = document.getElementById("dangerFlash");
    overlay.classList.add("show");
    clearTimeout(overlay._hideTimer);
    overlay._hideTimer = setTimeout(() => overlay.classList.remove("show"), 2200);
  }
  _prevSystemState = cur;
}

function formatUptime(s) {
  s = Math.floor(s || 0);
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = String(s % 60).padStart(2, "0");
  return `${h}:${m}:${sec}`;
}

const SM_ORDER = { SAFE: 0, HAND_DETECTED: 1, DANGER: 2, SAFETY_EVENT: 3, TRIPPED: 4, RELAY_OPEN: 5, FAULT: 6 };
function highlightStateMachine(state) {
  let current = "SAFE";
  if (state.system.state === "FAULT") current = state.relay.state === "OPEN" ? "RELAY_OPEN" : "FAULT";
  else if (state.safety.state === "TRIPPED") current = "TRIPPED";
  else if (state.system.state === "DANGER") current = "DANGER";
  else if (state.ai.state === "HAND_DETECTED") current = "HAND_DETECTED";

  document.querySelectorAll(".sm-step").forEach(el => {
    el.classList.toggle("active", el.dataset.state === current);
  });
  const rt = document.getElementById("demoResponseTime");
  if (rt && current === "FAULT") { /* left as last logged response time from event table */ }
}

// -------------------------------------------------------------- event log
function addEventRow(e) {
  const tbody = document.getElementById("eventsBody");
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${e.time}</td><td>${e.event}</td><td>${e.source || "—"}</td>` +
    `<td>${e.confidence != null ? e.confidence + "%" : "—"}</td><td>${e.action || "—"}</td>` +
    `<td>${(e.action && e.action.endsWith("ms")) ? e.action : "—"}</td><td>${e.status || "—"}</td>`;
  tbody.prepend(tr);
  while (tbody.children.length > 300) tbody.removeChild(tbody.lastChild);

  if (e.event === "TOOL POWER OFF" && e.action) {
    const respMatch = e.action.match(/([\d.]+)\s*ms/);
    if (respMatch) {
      pushChartPoint(chartResponse, parseFloat(respMatch[1]));
      const el = document.getElementById("demoResponseTime");
      if (el) el.textContent = e.action;
    }
  }
}

document.getElementById("clearLogBtn").addEventListener("click", () => socket.emit("clear_events"));
document.getElementById("exportLogBtn").addEventListener("click", exportCsv);

function exportCsv() {
  const rows = [["Time", "Event", "Source", "Confidence", "Action", "Status"]];
  document.querySelectorAll("#eventsBody tr").forEach(tr => {
    rows.push(Array.from(tr.children).slice(0, 5).map(td => td.textContent).concat(tr.children[6].textContent));
  });
  const csv = rows.map(r => r.map(v => `"${v}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `reflex-guard-events-${Date.now()}.csv`;
  a.click();
}

// ----------------------------------------------------------------- terminal
function appendTerminal(line) {
  const term = document.getElementById("terminalLog");
  const div = document.createElement("div");
  div.textContent = line;
  term.appendChild(div);
  while (term.children.length > 400) term.removeChild(term.firstChild);
  term.scrollTop = term.scrollHeight;
}

// ------------------------------------------------------------------ charts
function makeChart(canvas, color, max) {
  const ctx = canvas.getContext("2d");
  return { canvas, ctx, color, data: [], max: max || 100 };
}
function resizeCanvas(c) {
  const rect = c.canvas.getBoundingClientRect();
  c.canvas.width = rect.width * devicePixelRatio;
  c.canvas.height = rect.height * devicePixelRatio;
}
function pushChartPoint(chart, value) {
  if (value == null || isNaN(value)) return;
  chart.data.push(value);
  if (chart.data.length > 90) chart.data.shift();
  drawChart(chart);
}
function drawChart(chart) {
  resizeCanvas(chart);
  const { ctx, canvas, data, color } = chart;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const maxVal = Math.max(...data, 1) * 1.15;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - (v / maxVal) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "#e7ebf0";
  ctx.font = `${11 * devicePixelRatio}px sans-serif`;
  ctx.fillText(data[data.length - 1].toFixed(1), w - 40 * devicePixelRatio, 14 * devicePixelRatio);
}
const chartLatency = makeChart(document.getElementById("chartLatency"), "#3b82f6");
const chartFps = makeChart(document.getElementById("chartFps"), "#22c55e");
const chartConfidence = makeChart(document.getElementById("chartConfidence"), "#f59e0b");
const chartResponse = makeChart(document.getElementById("chartResponse"), "#ef4444");

function pushChartsFromEvent() { /* response chart handled in addEventRow */ }

// ------------------------------------------------------------------ camera
const camVideo = document.getElementById("camVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const octx = overlayCanvas.getContext("2d");
let camStream = null;
let camFpsHistory = [];
let lastFrameTime = performance.now();
let dangerZone = JSON.parse(localStorage.getItem("rg_danger_zone") || "null") || { x1: 0.35, y1: 0.3, x2: 0.65, y2: 0.7 };
let editingZone = false;
let dragStart = null;

async function populateCameraList() {
  const select = document.getElementById("cameraSelect");
  select.innerHTML = "";
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    devices.filter(d => d.kind === "videoinput").forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${i + 1}`;
      select.appendChild(opt);
    });
    const saved = localStorage.getItem("rg_camera_id");
    if (saved) select.value = saved;
  } catch (e) { /* enumeration requires permission on some browsers; fine */ }
}

document.getElementById("enableCamBtn").addEventListener("click", startCamera);
document.getElementById("stopCamBtn").addEventListener("click", stopCamera);
document.getElementById("restartCamBtn").addEventListener("click", async () => { stopCamera(); await startCamera(); });
document.getElementById("fullscreenBtn").addEventListener("click", () => {
  document.getElementById("cameraStage").requestFullscreen?.();
});
document.getElementById("editZoneBtn").addEventListener("click", () => {
  editingZone = !editingZone;
  document.getElementById("editZoneBtn").textContent = editingZone ? "Done Editing" : "Edit Danger Zone";
});

async function startCamera() {
  hideCamError();
  try {
    const deviceId = document.getElementById("cameraSelect").value;
    const constraints = {
      video: {
        width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
      audio: false,
    };
    camStream = await navigator.mediaDevices.getUserMedia(constraints);
    camVideo.srcObject = camStream;
    await populateCameraList();
    const id = camStream.getVideoTracks()[0].getSettings().deviceId;
    if (id) localStorage.setItem("rg_camera_id", id);

    document.getElementById("stopCamBtn").disabled = false;
    document.getElementById("restartCamBtn").disabled = false;
    document.getElementById("fullscreenBtn").disabled = false;
    document.getElementById("editZoneBtn").disabled = false;
    document.getElementById("cameraSelect").disabled = false;
    document.getElementById("camConnDot").classList.replace("off", "on");
    document.getElementById("camConnLabel").textContent = "CONNECTED";
    setCard("camera", "CONNECTED", "CONNECTED");

    camFpsHistory = [];
    requestAnimationFrame(frameLoop);
  } catch (err) {
    showCamError(cameraErrorMessage(err));
  }
}

function stopCamera() {
  if (camStream) camStream.getTracks().forEach(t => t.stop());
  camStream = null;
  document.getElementById("stopCamBtn").disabled = true;
  document.getElementById("restartCamBtn").disabled = true;
  document.getElementById("camConnDot").classList.replace("on", "off");
  document.getElementById("camConnLabel").textContent = "OFFLINE";
  setCard("camera", "OFFLINE", "OFFLINE");
}

function cameraErrorMessage(err) {
  if (err.name === "NotAllowedError") return "Camera permission was denied. Allow camera access and try again.";
  if (err.name === "NotFoundError") return "No camera device was found on this system.";
  return "Could not start the camera: " + err.message;
}
function showCamError(msg) { const el = document.getElementById("cameraError"); el.textContent = msg; el.style.display = "flex"; }
function hideCamError() { document.getElementById("cameraError").style.display = "none"; }

function frameLoop(t) {
  if (!camStream) return;
  const dt = t - lastFrameTime;
  lastFrameTime = t;
  const fps = 1000 / dt;
  camFpsHistory.push(fps);
  if (camFpsHistory.length > 30) camFpsHistory.shift();
  const avgFps = camFpsHistory.reduce((a, b) => a + b, 0) / camFpsHistory.length;

  if (camVideo.videoWidth) {
    document.getElementById("camRes").textContent = `${camVideo.videoWidth} × ${camVideo.videoHeight}`;
  }
  document.getElementById("camFps").textContent = `${avgFps.toFixed(1)} FPS`;
  pushChartPoint(chartFps, avgFps);

  drawOverlay();
  requestAnimationFrame(frameLoop);
}

function drawOverlay() {
  const rect = camVideo.getBoundingClientRect();
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;
  octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const x1 = dangerZone.x1 * overlayCanvas.width, y1 = dangerZone.y1 * overlayCanvas.height;
  const x2 = dangerZone.x2 * overlayCanvas.width, y2 = dangerZone.y2 * overlayCanvas.height;
  octx.strokeStyle = "#ef4444";
  octx.setLineDash([6, 4]);
  octx.lineWidth = 2;
  octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  octx.setLineDash([]);
  octx.fillStyle = "#ef4444";
  octx.font = "12px sans-serif";
  octx.fillText("DANGER ZONE", x1 + 4, y1 - 6 < 10 ? y1 + 14 : y1 - 6);
}

overlayCanvas.addEventListener("mousedown", e => {
  if (!editingZone) return;
  const r = overlayCanvas.getBoundingClientRect();
  dragStart = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
});
overlayCanvas.addEventListener("mousemove", e => {
  if (!editingZone || !dragStart) return;
  const r = overlayCanvas.getBoundingClientRect();
  const cur = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
  dangerZone = {
    x1: Math.min(dragStart.x, cur.x), y1: Math.min(dragStart.y, cur.y),
    x2: Math.max(dragStart.x, cur.x), y2: Math.max(dragStart.y, cur.y),
  };
});
window.addEventListener("mouseup", () => {
  if (editingZone && dragStart) {
    localStorage.setItem("rg_danger_zone", JSON.stringify(dangerZone));
  }
  dragStart = null;
});

navigator.mediaDevices?.addEventListener?.("devicechange", populateCameraList);
populateCameraList();

// initial terminal line
appendTerminal("Dashboard loaded. Waiting for server state...");
