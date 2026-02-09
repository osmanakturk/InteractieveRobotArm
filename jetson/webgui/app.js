// webgui/app.js
const $ = (id) => document.getElementById(id);

const state = {
  gateway: "",
  frame: "base",
  mode: "xyz",
  speedPct: 50,

  robotConnected: false,
  robotEnabled: false,
  safetyHit: false,

  holdTimer: null,
  holdDir: null,

  cameraStarted: false,

  // WS
  ws: null,
  wsConnected: false,
  wsReconnectTimer: null,
  wsReconnectAttempts: 0,
  wsManualClose: false,          // user disconnect vs network drop
  wsLastMsgTs: 0,                // stale detection
  wsStaleTimer: null,
};

const HOLD_INTERVAL_MS = 120;

// -------------------------
// UI helpers
// -------------------------
function toast(msg, ms = 2500) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => t.classList.remove("show"), ms);
}

function setDot(dotEl, status) {
  dotEl.classList.remove("dot-green", "dot-blue", "dot-red", "dot-gray");
  if (status === "connected") dotEl.classList.add("dot-green");
  else if (status === "connecting") dotEl.classList.add("dot-blue");
  else if (status === "error") dotEl.classList.add("dot-red");
  else dotEl.classList.add("dot-gray");
}

// FINAL: one normalizer for all http inputs (Gateway + AI)
// Accepts: "192.168.1.20:8000" OR "http://..." OR "https://..."
function normalizeAnyHttp(input) {
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

function setCameraOverlay(show, title, sub) {
  const overlay = $("camera-overlay");
  $("camera-overlay-title").textContent = title || "";
  $("camera-overlay-sub").textContent = sub || "";
  overlay.style.display = show ? "flex" : "none";
}

// -------------------------
// Guards
// -------------------------
function requireGatewaySync() {
  if (!state.gateway) {
    toast("Gateway disconnected. Please connect Gateway first.");
    return false;
  }
  return true;
}
function requireRobotConnectedSync() {
  if (!state.robotConnected) {
    toast("Robot not connected. Connect robot IP first.");
    return false;
  }
  return true;
}
async function ensureRobotEnabledFresh() {
  if (!requireGatewaySync()) return false;
  if (!state.robotConnected) {
    toast("Robot not connected. Connect robot IP first.");
    return false;
  }
  if (!state.robotEnabled) {
    toast("Robot is not enabled. Press Enable to continue.");
    return false;
  }
  return true;
}

// -------------------------
// HTTP (fallback / commands)
// -------------------------
async function apiGet(path) {
  if (!requireGatewaySync()) throw new Error("Gateway not set");
  const url = `${state.gateway}${path}`;
  const r = await fetch(url, { method: "GET" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${r.status}`);
  return data;
}

async function apiPost(path, body = {}) {
  if (!requireGatewaySync()) throw new Error("Gateway not set");
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

// if WS drops, you can pull one snapshot
async function refreshOnce() {
  if (!state.gateway) return;
  const data = await apiGet("/api/status");
  applyStatus(data);
}

// -------------------------
// Modal (Safety)
// -------------------------
function showSafetyModal(message) {
  $("modal-text").textContent =
    message ||
    "Safety limit reached. Robot was disabled for safety. Press Enable again to continue.";
  $("modal").classList.add("show");
}
function hideSafetyModal() {
  $("modal").classList.remove("show");
}

// -------------------------
// Camera
// -------------------------
function attachCameraStream() {
  const img = $("img-camera");
  if (!state.gateway) return;

  img.src = `${state.gateway}/api/camera/realsense?ts=${Date.now()}`;

  img.onerror = () => {
    state.cameraStarted = false;
    setCameraOverlay(true, "Camera error", "Could not load /api/camera/realsense. Check Start + Gateway.");
    toast("Camera stream error.");
  };
}

async function cameraStart() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/camera/realsense/start", {});
    state.cameraStarted = true;
    attachCameraStream();
    setCameraOverlay(false);
    toast("Camera started.");
  } catch (e) {
    setCameraOverlay(true, "Camera not available", String(e.message || e));
    toast(String(e.message || e));
  }
}

async function cameraStop() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/camera/realsense/stop", {});
    state.cameraStarted = false;
    $("img-camera").src = "";
    setCameraOverlay(true, "Camera stopped", "Press Start Camera to run RealSense stream.");
    toast("Camera stopped.");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function cameraStatus() {
  if (!requireGatewaySync()) return;
  try {
    const data = await apiGet("/api/camera/realsense/status");
    const cam = data.camera || {};
    const started = !!cam.started;
    const err = cam.last_error || "";
    state.cameraStarted = started;

    $("pill-camera").textContent = `Camera: ${started ? "ON" : "OFF"}`;
    if (!started) setCameraOverlay(true, "Camera not started", err || "Press Start Camera.");
    else {
      setCameraOverlay(false);
      attachCameraStream();
    }
    toast(`Camera: ${started ? "ON" : "OFF"}`);
  } catch (e) {
    toast(String(e.message || e));
  }
}

// -------------------------
// Telemetry decode (user-friendly)
// -------------------------
// If you later add st.state_text / st.error_text in backend, JS will prefer them.
function decodeEnabled(v) {
  return v ? "YES (Robot enabled)" : "NO (Robot disabled)";
}

// Generic mapping (safe). Unknown -> "Unknown (X)"
function decodeRobotState(st) {
  if (typeof st?.state_text === "string" && st.state_text.trim()) return st.state_text;

  const code = st?.state;
  if (code === null || code === undefined) return "Unknown (?)";

  // NOTE: generic interpretation. If your SDK has an official table, we can align it.
  const map = {
    0: "READY / IDLE (0)",
    1: "STOPPED (1)",
    2: "RUNNING / MOVING (2)",
    3: "PAUSED (3)",
    4: "ERROR / FAULT (4)",
    5: "EMERGENCY STOP (5)",
  };

  return map.hasOwnProperty(code) ? map[code] : `Unknown (${code})`;
}

function decodeErrorCode(st) {
  if (typeof st?.error_text === "string" && st.error_text.trim()) return st.error_text;

  const code = st?.error_code;
  if (code === null || code === undefined) return "Unknown (?)";
  if (Number(code) === 0) return "OK (0) — No error";

  // If you have an official error table, we can map codes here.
  return `ERROR (${code}) — See logs / safety panel`;
}

// -------------------------
// STATUS UI (WS payload -> UI)
// -------------------------
function applyStatus(data) {
  // data: { ok: true, status: {...}, safety: {...}, camera: {...}, ai_server: {...} }
  const st = data.status || {};
  const safety = data.safety || {};
  const cam = data.camera || {};
  const ai = data.ai_server || {};

  // Gateway indicator: if WS connected -> green
  if (!state.gateway) {
    setDot($("dot-gateway"), "gray");
    $("meta-gateway").textContent = "Disconnected";
  } else {
    setDot($("dot-gateway"), state.wsConnected ? "connected" : "connecting");
    $("meta-gateway").textContent = state.wsConnected ? "Connected" : "Connecting...";
  }

  // Robot
  state.robotConnected = !!st.connected;
  state.robotEnabled = !!st.is_enabled;

  setDot($("dot-robot"), state.robotConnected ? "connected" : "gray");
  $("meta-robot").textContent = state.robotConnected ? `OK (${st.ip || ""})` : "Disconnected";

  $("pill-enabled").textContent = `Enabled: ${decodeEnabled(!!st.is_enabled)}`;
  $("pill-state").textContent = `State: ${decodeRobotState(st)}`;
  $("pill-error").textContent = `Error: ${decodeErrorCode(st)}`;

  // Safety
  const limitHit = !!(safety.limit_hit ?? false);
  state.safetyHit = limitHit;
  $("pill-safety").textContent = `Safety: ${limitHit ? "HIT (limit reached)" : "OK"}`;

  if (limitHit) {
    if (!applyStatus._safetyShown) {
      applyStatus._safetyShown = true;
      showSafetyModal(safety.message || "");
    }
  } else {
    applyStatus._safetyShown = false;
  }

  // Camera
  const camStarted = !!(cam.started ?? false);
  $("pill-camera").textContent = `Camera: ${camStarted ? "ON" : "OFF"}`;

  // AI server
  const aiConfigured = !!ai.configured;
  const aiConnected = !!ai.connected;

  setDot($("dot-aiserver"), aiConnected ? "connected" : (aiConfigured ? "error" : "gray"));
  if (aiConnected) $("meta-aiserver").textContent = `OK (${ai.latency_ms ?? "?"}ms)`;
  else $("meta-aiserver").textContent = aiConfigured ? "FAIL (health check)" : "Disconnected";

  // Gripper %
  if (typeof st.gripper_pct === "number") {
    const pct = Math.max(0, Math.min(100, st.gripper_pct));
    $("grip-pct").textContent = String(pct);
    $("grip-fill").style.width = `${pct}%`;
  }
}

// -------------------------
// WEBSOCKET
// -------------------------
function gatewayToWsUrl(gatewayHttpUrl) {
  const u = new URL(gatewayHttpUrl);
  const wsProto = u.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${u.host}/ws/status`;
}

function wsClearTimers() {
  if (state.wsReconnectTimer) {
    clearTimeout(state.wsReconnectTimer);
    state.wsReconnectTimer = null;
  }
  if (state.wsStaleTimer) {
    clearInterval(state.wsStaleTimer);
    state.wsStaleTimer = null;
  }
}

function wsClose(manual = false) {
  state.wsManualClose = manual;
  wsClearTimers();

  state.wsConnected = false;

  try {
    if (state.ws && state.ws.readyState <= 1) state.ws.close();
  } catch {}
  state.ws = null;
}

function wsScheduleReconnect() {
  if (state.wsManualClose) return;     // user requested close
  if (!state.gateway) return;          // gateway cleared

  const attempt = Math.min(15, state.wsReconnectAttempts + 1);
  state.wsReconnectAttempts = attempt;

  // backoff: 0.8s, 1.1s, 1.4s ... capped
  const backoffMs = Math.min(10000, 500 + attempt * 300);

  setDot($("dot-gateway"), "connecting");
  $("meta-gateway").textContent = "Reconnecting...";

  state.wsReconnectTimer = setTimeout(() => {
    wsConnect();
  }, backoffMs);
}

function wsStartStaleWatchdog() {
  // If no message received for N seconds, force reconnect.
  wsClearTimers();
  state.wsStaleTimer = setInterval(() => {
    if (!state.gateway) return;
    if (!state.ws) return;
    if (state.ws.readyState !== WebSocket.OPEN) return;

    const ageMs = Date.now() - (state.wsLastMsgTs || 0);
    // allow temporary stalls (e.g., model load) but avoid infinite hang
    if (ageMs > 5000) {
      // 5s no status -> reconnect
      try { state.ws.close(); } catch {}
    }
  }, 1000);
}

function wsConnect() {
  if (!state.gateway) return;

  // prevent duplicates: if existing OPEN/CONNECTING, don't create another
  if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  state.wsManualClose = false; // network-managed
  wsClearTimers();

  const wsUrl = gatewayToWsUrl(state.gateway);
  setDot($("dot-gateway"), "connecting");
  $("meta-gateway").textContent = "Connecting...";

  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.onopen = () => {
    state.wsConnected = true;
    state.wsReconnectAttempts = 0;
    state.wsLastMsgTs = Date.now();

    setDot($("dot-gateway"), "connected");
    $("meta-gateway").textContent = "Connected";
    toast("Gateway connected (WebSocket).");

    // camera overlay reset on connect
    setCameraOverlay(true, "Camera not started", "Press Start Camera.");
    $("img-camera").src = "";

    wsStartStaleWatchdog();
  };

  ws.onmessage = (ev) => {
    state.wsLastMsgTs = Date.now();
    try {
      const data = JSON.parse(ev.data);
      if (data?.ok) applyStatus(data);
    } catch {
      // ignore parse errors
    }
  };

  ws.onerror = () => {
    // close will fire afterwards in most browsers
  };

  ws.onclose = () => {
    state.wsConnected = false;
    wsClearTimers();

    if (!state.gateway) {
      setDot($("dot-gateway"), "gray");
      $("meta-gateway").textContent = "Disconnected";
      return;
    }

    wsScheduleReconnect();
  };
}

// -------------------------
// Gateway connect/disconnect
// -------------------------
async function connectGateway() {
  const gw = normalizeAnyHttp($("in-gateway").value);
  if (!gw) return;

  state.gateway = gw;
  wsConnect();

  // quick verify (optional)
  try {
    await apiGet("/api/status");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function disconnectGateway() {
  const oldGw = state.gateway;

  // stop reconnect loops first
  state.gateway = "";
  wsClose(true);

  // backend reset (robot/camera/ai cleared)
  if (oldGw) {
    try {
      await fetch(`${oldGw}/api/gateway/disconnect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    } catch {}
  }

  // local reset
  state.robotConnected = false;
  state.robotEnabled = false;

  stopHold();

  setDot($("dot-gateway"), "gray");
  $("meta-gateway").textContent = "Disconnected";
  setDot($("dot-robot"), "gray");
  $("meta-robot").textContent = "Disconnected";
  setDot($("dot-aiserver"), "gray");
  $("meta-aiserver").textContent = "Disconnected";

  $("pill-enabled").textContent = "Enabled: ?";
  $("pill-state").textContent = "State: ?";
  $("pill-error").textContent = "Error: ?";
  $("pill-safety").textContent = "Safety: ?";
  $("pill-camera").textContent = "Camera: ?";

  $("img-camera").src = "";
  setCameraOverlay(true, "Gateway disconnected", "Connect Gateway to use camera and controls.");
  toast("Gateway disconnected.");
}

// -------------------------
// Robot connect/disconnect
// -------------------------
async function robotConnect() {
  if (!requireGatewaySync()) return;
  const ip = ($("in-robot").value || "").trim();
  if (!ip) return;

  setDot($("dot-robot"), "connecting");
  $("meta-robot").textContent = "Connecting...";

  try {
    await apiPost("/api/connect", { ip });
    toast("Robot connected.");
    await refreshOnce();
  } catch (e) {
    setDot($("dot-robot"), "error");
    $("meta-robot").textContent = "Error";
    toast(String(e.message || e));
  }
}

async function robotDisconnect() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/disconnect", {});
    toast("Robot disconnected.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

// -------------------------
// AI connect/disconnect
// -------------------------
async function aiConnect() {
  if (!requireGatewaySync()) return;

  // user can type: "192.168.1.50:9000" OR "http://..." OR "https://..."
  const url = normalizeAnyHttp($("in-aiserver").value);
  if (!url) return;

  setDot($("dot-aiserver"), "connecting");
  $("meta-aiserver").textContent = "Connecting...";

  try {
    await apiPost("/api/ai_server/connect", { url });
    toast("AI server connected.");
    await refreshOnce();
  } catch (e) {
    setDot($("dot-aiserver"), "error");
    $("meta-aiserver").textContent = "Error";
    toast(String(e.message || e));
    await refreshOnce().catch(() => {});
  }
}

async function aiDisconnect() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/ai_server/disconnect", {});
    toast("AI server disconnected.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

// -------------------------
// Robot buttons
// -------------------------
async function enableRobot() {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;

  try {
    await apiPost("/api/enable", {});
    toast("Robot enabled.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function disableRobot() {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;

  try {
    await apiPost("/api/disable", {});
    toast("Robot disabled.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function stopRobot() {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;

  try {
    await apiPost("/api/stop", {});
    toast("STOP pressed. Robot stopped.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function homeRobot() {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;
  if (!(await ensureRobotEnabledFresh())) return;

  try {
    await apiPost("/api/home", {});
    toast("Home command sent.");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function clearSafety() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/safety/clear", {});
    toast("Safety cleared.");
    await refreshOnce();
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function setFrame(frame) {
  if (!requireGatewaySync()) return;
  state.frame = frame;
  $("pad-center-sub").textContent = frame === "tool" ? "Tool" : "Base";
  $("cross-center").textContent = frame === "tool" ? "Tool" : "Base";

  if (!requireRobotConnectedSync()) return;
  if (!(await ensureRobotEnabledFresh())) return;

  try {
    await apiPost("/api/frame", { frame });
  } catch (e) {
    toast(String(e.message || e));
  }
}

function setMode(mode) {
  state.mode = mode;
  $("pad-center-title").textContent = mode === "rxyz" ? "RXYZ" : "XYZ";
}

// -------------------------
// Speed
// -------------------------
async function setSpeed(pct) {
  setSpeedUI(pct);
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;
  if (!(await ensureRobotEnabledFresh())) return;

  try {
    await apiPost("/api/speed", { speed_pct: state.speedPct });
  } catch (e) {
    toast(String(e.message || e));
  }
}

// -------------------------
// Jog mapping
// -------------------------
function buildJogPayload(dir) {
  const s = 5; // mm step
  const r = 2; // deg step

  if (state.mode === "xyz") {
    if (dir === "up") return { dx: +s, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "down") return { dx: -s, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "left") return { dx: 0, dy: +s, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "right") return { dx: 0, dy: -s, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zplus") return { dx: 0, dy: 0, dz: +s, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zminus") return { dx: 0, dy: 0, dz: -s, droll: 0, dpitch: 0, dyaw: 0 };
  } else {
    if (dir === "up") return { dx: 0, dy: 0, dz: 0, droll: +r, dpitch: 0, dyaw: 0 };
    if (dir === "down") return { dx: 0, dy: 0, dz: 0, droll: -r, dpitch: 0, dyaw: 0 };
    if (dir === "left") return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: +r, dyaw: 0 };
    if (dir === "right") return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: -r, dyaw: 0 };
    if (dir === "zplus") return { dx: 0, dy: 0, dz: +s, droll: 0, dpitch: 0, dyaw: 0 };
    if (dir === "zminus") return { dx: 0, dy: 0, dz: -s, droll: 0, dpitch: 0, dyaw: 0 };
  }

  return { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };
}

async function jogOnce(dir) {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;
  if (!(await ensureRobotEnabledFresh())) return;

  try {
    await apiPost("/api/jog", buildJogPayload(dir));
  } catch (e) {
    const msg = String(e.message || e);
    // 429 throttle -> ignore silently
    if (!msg.includes("429")) {
      // toast(msg);
    }
  }
}

function startHold(dir) {
  stopHold();
  state.holdDir = dir;
  jogOnce(dir);
  state.holdTimer = setInterval(() => jogOnce(dir), HOLD_INTERVAL_MS);
}

function stopHold() {
  if (state.holdTimer) clearInterval(state.holdTimer);
  state.holdTimer = null;
  state.holdDir = null;
}

// -------------------------
// Gripper
// -------------------------
async function gripper(action) {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;
  if (!(await ensureRobotEnabledFresh())) return;

  try {
    await apiPost("/api/gripper", { action });
  } catch (e) {
    toast(String(e.message || e));
  }
}

// Stop hold if tab loses focus (prevents "stuck hold")
window.addEventListener("blur", () => stopHold());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopHold();
});

// -------------------------
// Wire up
// -------------------------
window.addEventListener("DOMContentLoaded", () => {
  setSpeedUI(50);
  setCameraOverlay(true, "Camera not started", "Connect Gateway → Start Camera.");
  $("img-camera").src = "";

  // gateway
  $("btn-gateway-connect").addEventListener("click", connectGateway);
  $("btn-gateway-disconnect").addEventListener("click", disconnectGateway);

  // robot
  $("btn-robot-connect").addEventListener("click", robotConnect);
  $("btn-robot-disconnect").addEventListener("click", robotDisconnect);

  // ai
  $("btn-aiserver-connect").addEventListener("click", aiConnect);
  $("btn-aiserver-disconnect").addEventListener("click", aiDisconnect);

  // robot controls
  $("btn-enable").addEventListener("click", enableRobot);
  $("btn-disable").addEventListener("click", disableRobot);
  $("btn-stop").addEventListener("click", stopRobot);
  $("btn-home").addEventListener("click", homeRobot);
  $("btn-safety-clear").addEventListener("click", clearSafety);

  // frame/mode
  $("sel-frame").addEventListener("change", (e) => setFrame(e.target.value));
  $("sel-mode").addEventListener("change", (e) => setMode(e.target.value));

  // speed
  $("btn-speed-down").addEventListener("click", () => setSpeed(state.speedPct - 5));
  $("btn-speed-up").addEventListener("click", () => setSpeed(state.speedPct + 5));

  // gripper
  $("btn-grip-open").addEventListener("click", () => gripper("open"));
  $("btn-grip-close").addEventListener("click", () => gripper("close"));

  // camera
  $("btn-camera-start").addEventListener("click", cameraStart);
  $("btn-camera-stop").addEventListener("click", cameraStop);
  $("btn-camera-status").addEventListener("click", cameraStatus);

  // hold buttons
  document.querySelectorAll(".btn.hold").forEach((el) => {
    const dir = el.getAttribute("data-jog");

    const onDown = (ev) => {
      ev.preventDefault?.();
      startHold(dir);
    };
    const onUp = (ev) => {
      ev.preventDefault?.();
      stopHold();
    };

    el.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);

    el.addEventListener("touchstart", onDown, { passive: false });
    window.addEventListener("touchend", onUp, { passive: false });
    window.addEventListener("touchcancel", onUp, { passive: false });
  });

  // modal actions
  $("btn-modal-close").addEventListener("click", hideSafetyModal);
  $("btn-modal-enable").addEventListener("click", async () => {
    hideSafetyModal();
    await enableRobot();
  });
  $("btn-modal-clear").addEventListener("click", async () => {
    await clearSafety();
    hideSafetyModal();
  });
});
