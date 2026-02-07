// app.js
const $ = (id) => document.getElementById(id);

const state = {
  gateway: "",
  robot: "",
  aiserver: "",
  frame: "base",
  mode: "xyz", // xyz | rxyz
  speedPct: 50,
  // jogging repeat
  holdTimer: null,
  holdDir: null,
  polling: null,
};

// ---------- UI helpers ----------
function setDot(dotEl, status) {
  dotEl.classList.remove("dot-green", "dot-blue", "dot-red", "dot-gray");
  if (status === "connected") dotEl.classList.add("dot-green");
  else if (status === "connecting") dotEl.classList.add("dot-blue");
  else if (status === "error") dotEl.classList.add("dot-red");
  else dotEl.classList.add("dot-gray");
}

function normalizeGateway(input) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (/^https?:\/\//i.test(raw)) return raw;
  return `http://${raw}`;
}

function setSpeedUI(pct) {
  const p = Math.max(1, Math.min(100, Number(pct) || 1));
  state.speedPct = p;
  $("speed-pct").textContent = String(p);
  $("speed-fill").style.width = `${p}%`;
}

// ---------- HTTP ----------
async function apiGet(path) {
  if (!state.gateway) throw new Error("Gateway not set");
  const url = `${state.gateway}${path}`;
  const r = await fetch(url, { method: "GET" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${r.status}`);
  return data;
}

async function apiPost(path, body = {}) {
  if (!state.gateway) throw new Error("Gateway not set");
  const url = `${state.gateway}${path}`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${r.status}`);
  return data;
}

// ---------- Status polling ----------
async function refreshStatus() {
  try {
    const data = await apiGet("/api/status");
    const st = data.status || {};

    // Gateway dot
    setDot($("dot-gateway"), "connected");
    $("meta-gateway").textContent = "Connected";

    // Robot: heuristics from status.connected or ip
    const robotConnected = !!st.connected;
    setDot($("dot-robot"), robotConnected ? "connected" : "idle");
    $("meta-robot").textContent = robotConnected ? `OK (${st.ip || ""})` : "Idle";

    // Enabled state
    $("pill-enabled").textContent = `Enabled: ${st.is_enabled ? "YES" : "NO"}`;
    $("pill-state").textContent = `State: ${st.state ?? "?"}`;
    $("pill-error").textContent = `Err: ${st.error_code ?? "?"}`;

    // Optional: gripper percentage (if you add endpoint later)
    // Here we try to compute from status if available:
    if (st.gripper_available && typeof st.gripper_pos === "number") {
      const min = Number(st.gripper_min ?? 0);
      const max = Number(st.gripper_max ?? 1);
      const pos = Number(st.gripper_pos);
      const pct = max > min ? Math.round(((pos - min) / (max - min)) * 100) : 0;
      $("grip-pct").textContent = String(Math.max(0, Math.min(100, pct)));
      $("grip-fill").style.width = `${Math.max(0, Math.min(100, pct))}%`;
    }
  } catch (e) {
    // Gateway may be down
    setDot($("dot-gateway"), "error");
    $("meta-gateway").textContent = "Error";
  }
}

function startPolling() {
  stopPolling();
  refreshStatus();
  state.polling = setInterval(refreshStatus, 1000);
}
function stopPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = null;
}

// ---------- Camera ----------
function startCamera() {
  const img = $("img-camera");
  const overlay = $("camera-overlay");
  if (!state.gateway) return;

  // Fixed endpoint requested:
  img.src = `${state.gateway}/api/camera/realsense`;

  overlay.style.display = "none";

  img.onerror = () => {
    overlay.style.display = "flex";
    overlay.querySelector(".camera-overlay-title").textContent = "Camera error";
    overlay.querySelector(".camera-overlay-sub").textContent =
      "Could not load /api/camera/realsense. Check Gateway + endpoint.";
  };
}

// ---------- Connections ----------
async function connectGateway() {
  const raw = $("in-gateway").value;
  const gw = normalizeGateway(raw);
  if (!gw) return;

  state.gateway = gw;

  setDot($("dot-gateway"), "connecting");
  $("meta-gateway").textContent = "Connecting...";

  try {
    await apiGet("/api/status"); // ping
    setDot($("dot-gateway"), "connected");
    $("meta-gateway").textContent = "Connected";
    startCamera();
    startPolling();
  } catch (e) {
    setDot($("dot-gateway"), "error");
    $("meta-gateway").textContent = "Error";
    alert(String(e.message || e));
  }
}

async function setRobot() {
  const ip = ($("in-robot").value || "").trim();
  if (!ip) return;

  setDot($("dot-robot"), "connecting");
  $("meta-robot").textContent = "Connecting...";

  try {
    // Existing gateway endpoint from your backend:
    await apiPost("/api/connect", { ip });
    setDot($("dot-robot"), "connected");
    $("meta-robot").textContent = `OK (${ip})`;
  } catch (e) {
    setDot($("dot-robot"), "error");
    $("meta-robot").textContent = "Error";
    alert(String(e.message || e));
  }
}

async function setAiServer() {
  const url = ($("in-aiserver").value || "").trim();
  if (!url) return;

  setDot($("dot-aiserver"), "connecting");
  $("meta-aiserver").textContent = "Setting...";

  try {
    // If you implement: POST /api/ai_server { url }
    await apiPost("/api/ai_server", { url });
    setDot($("dot-aiserver"), "connected");
    $("meta-aiserver").textContent = "OK";
  } catch (e) {
    setDot($("dot-aiserver"), "error");
    $("meta-aiserver").textContent = "Error";
    alert(String(e.message || e));
  }
}

// ---------- Robot buttons ----------
async function enableRobot() {
  try { await apiPost("/api/enable", {}); } catch (e) { alert(String(e.message || e)); }
}
async function disableRobot() {
  try { await apiPost("/api/disable", {}); } catch (e) { alert(String(e.message || e)); }
}
async function stopRobot() {
  try { await apiPost("/api/stop", {}); } catch (e) { alert(String(e.message || e)); }
}
async function homeRobot() {
  try { await apiPost("/api/home", {}); } catch (e) { alert(String(e.message || e)); }
}

async function setFrame(frame) {
  state.frame = frame;
  $("pad-center-sub").textContent = frame === "tool" ? "Tool" : "Base";
  $("cross-center").textContent = frame === "tool" ? "Tool" : "Base";
  try { await apiPost("/api/frame", { frame }); } catch (e) { alert(String(e.message || e)); }
}

function setMode(mode) {
  state.mode = mode;
  $("pad-center-title").textContent = mode === "rxyz" ? "RXYZ" : "XYZ";
}

// ---------- Speed ----------
async function setSpeed(pct) {
  setSpeedUI(pct);
  try { await apiPost("/api/speed", { speed_pct: state.speedPct }); } catch (e) { alert(String(e.message || e)); }
}

// ---------- Jog mapping ----------
function buildJogPayload(dir) {
  // Step sizes (tune as needed)
  const s = 5;       // mm step
  const r = 2;       // deg step

  const mode = state.mode; // xyz or rxyz
  if (mode === "xyz") {
    if (dir === "up") return { dx: +s, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "down") return { dx: -s, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "left") return { dx: 0, dy: +s, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "right") return { dx: 0, dy: -s, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zplus") return { dx: 0, dy: 0, dz: +s, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zminus") return { dx: 0, dy: 0, dz: -s, droll: 0, dpitch: 0, dyaw: 0 };
  } else {
    // RXYZ mode: arrows control R/P/Y (example mapping)
    if (dir === "up") return { dx: 0, dy: 0, dz: 0, droll: +r, dpitch: 0, dyaw: 0 };
    if (dir === "down") return { dx: 0, dy: 0, dz: 0, droll: -r, dpitch: 0, dyaw: 0 };
    if (dir === "left") return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: +r, dyaw: 0 };
    if (dir === "right") return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: -r, dyaw: 0 };
    // Z still Z in your UI cross
    if (dir === "zplus") return { dx: 0, dy: 0, dz: +s, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zminus") return { dx: 0, dy: 0, dz: -s, droll: 0, dpitch: 0, dyaw: 0 };
  }
  return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
}

async function jogOnce(dir) {
  try {
    const payload = buildJogPayload(dir);
    await apiPost("/api/jog", payload);
  } catch (e) {
    // keep quiet or show once
  }
}

function startHold(dir) {
  stopHold();
  state.holdDir = dir;
  jogOnce(dir);
  state.holdTimer = setInterval(() => jogOnce(dir), 120);
}
function stopHold() {
  if (state.holdTimer) clearInterval(state.holdTimer);
  state.holdTimer = null;
  state.holdDir = null;
}

// ---------- Gripper ----------
async function gripper(action) {
  try { await apiPost("/api/gripper", { action }); } catch (e) { alert(String(e.message || e)); }
}

// ---------- Wire up ----------
window.addEventListener("DOMContentLoaded", () => {
  // defaults
  setSpeedUI(50);

  $("btn-gateway-connect").addEventListener("click", connectGateway);
  $("btn-robot-connect").addEventListener("click", setRobot);
  $("btn-aiserver-set").addEventListener("click", setAiServer);

  $("btn-enable").addEventListener("click", enableRobot);
  $("btn-disable").addEventListener("click", disableRobot);
  $("btn-stop").addEventListener("click", stopRobot);
  $("btn-home").addEventListener("click", homeRobot);
  $("btn-status").addEventListener("click", refreshStatus);

  $("sel-frame").addEventListener("change", (e) => setFrame(e.target.value));
  $("sel-mode").addEventListener("change", (e) => setMode(e.target.value));

  $("btn-speed-down").addEventListener("click", () => setSpeed(state.speedPct - 5));
  $("btn-speed-up").addEventListener("click", () => setSpeed(state.speedPct + 5));

  $("btn-grip-open").addEventListener("click", () => gripper("open"));
  $("btn-grip-close").addEventListener("click", () => gripper("close"));

  // Hold buttons (mouse + touch)
  document.querySelectorAll(".btn.hold").forEach((el) => {
    const dir = el.getAttribute("data-jog");

    const onDown = (ev) => { ev.preventDefault?.(); startHold(dir); };
    const onUp = (ev) => { ev.preventDefault?.(); stopHold(); };

    el.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);

    el.addEventListener("touchstart", onDown, { passive: false });
    window.addEventListener("touchend", onUp, { passive: false });
    window.addEventListener("touchcancel", onUp, { passive: false });
  });
});
