// webgui/app.js
const $ = (id) => document.getElementById(id);

const state = {
  gateway: "",
  frame: "base",
  mode: "xyz", // xyz | rxyz
  speedPct: 50,

  robotConnected: false,
  robotEnabled: false,
  safetyHit: false,

  polling: null,

  holdTimer: null,
  holdDir: null,

  cameraStarted: false,
};

const HOLD_INTERVAL_MS = 120; // client-side
// server-side: CONFIG.jog_min_interval_ms (örn 100ms)

function toast(msg, ms = 2500) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => t.classList.remove("show"), ms);
}

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

function setCameraOverlay(show, title, sub) {
  const overlay = $("camera-overlay");
  $("camera-overlay-title").textContent = title || "";
  $("camera-overlay-sub").textContent = sub || "";
  overlay.style.display = show ? "flex" : "none";
}

// ---------- Guards ----------
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

// IMPORTANT: enabled guard now can refresh once (race-condition fix)
async function ensureRobotEnabledFresh() {
  if (!requireGatewaySync()) return false;

  if (state.robotEnabled) return true;

  try {
    await refreshStatus();
  } catch {}

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

// ---------- HTTP ----------
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

// ---------- Modal (Safety) ----------
function showSafetyModal(message) {
  $("modal-text").textContent =
    message ||
    "Safety limit reached. Robot was disabled for safety. Press Enable again to continue.";
  $("modal").classList.add("show");
}
function hideSafetyModal() {
  $("modal").classList.remove("show");
}

// ---------- Camera ----------
function attachCameraStream() {
  const img = $("img-camera");
  if (!state.gateway) return;

  img.src = `${state.gateway}/api/camera/realsense`;

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
    if (!started) {
      setCameraOverlay(true, "Camera not started", err || "Press Start Camera.");
    } else {
      setCameraOverlay(false);
      attachCameraStream();
    }
    toast(`Camera: ${started ? "ON" : "OFF"}`);
  } catch (e) {
    toast(String(e.message || e));
  }
}

// ---------- Status polling ----------
async function refreshStatus() {
  if (!state.gateway) return;

  const data = await apiGet("/api/status");
  const st = data.status || {};
  const safety = data.safety || {};
  const cam = data.camera || {};
  const aiUrl = data.ai_server || null;

  // Gateway OK
  setDot($("dot-gateway"), "connected");
  $("meta-gateway").textContent = "Connected";

  // Robot
  state.robotConnected = !!st.connected;
  state.robotEnabled = !!st.is_enabled;

  setDot($("dot-robot"), state.robotConnected ? "connected" : "gray");
  $("meta-robot").textContent = state.robotConnected ? `OK (${st.ip || ""})` : "Disconnected";

  $("pill-enabled").textContent = `Enabled: ${st.is_enabled ? "YES" : "NO"}`;
  $("pill-state").textContent = `State: ${st.state ?? "?"}`;
  $("pill-error").textContent = `Err: ${st.error_code ?? "?"}`;

  // Safety
  const limitHit = !!(safety.limit_hit ?? false);
  state.safetyHit = limitHit;
  $("pill-safety").textContent = `Safety: ${limitHit ? "HIT" : "OK"}`;

  if (limitHit) {
    if (!refreshStatus._safetyShown) {
      refreshStatus._safetyShown = true;
      showSafetyModal(safety.message || "");
    }
  } else {
    refreshStatus._safetyShown = false;
  }

  // Camera
  const camStarted = !!(cam.started ?? false);
  $("pill-camera").textContent = `Camera: ${camStarted ? "ON" : "OFF"}`;

  // AI server (URL)
  const aiConnected = !!aiUrl;
  setDot($("dot-aiserver"), aiConnected ? "connected" : "gray");
  $("meta-aiserver").textContent = aiConnected ? "Set" : "Disconnected";

  // Gripper %
  if (typeof st.gripper_pct === "number") {
    const pct = Math.max(0, Math.min(100, st.gripper_pct));
    $("grip-pct").textContent = String(pct);
    $("grip-fill").style.width = `${pct}%`;
  }
}

function startPolling() {
  stopPolling();
  refreshStatus().catch(() => {});
  state.polling = setInterval(() => refreshStatus().catch(() => {}), 1000);
}
function stopPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = null;
}

// ---------- Gateway connect/disconnect ----------
async function connectGateway() {
  const gw = normalizeGateway($("in-gateway").value);
  if (!gw) return;

  state.gateway = gw;
  setDot($("dot-gateway"), "connecting");
  $("meta-gateway").textContent = "Connecting...";

  try {
    await apiGet("/api/status");
    setDot($("dot-gateway"), "connected");
    $("meta-gateway").textContent = "Connected";
    toast("Gateway connected.");
    startPolling();

    setCameraOverlay(true, "Camera not started", "Press Start Camera.");
    $("img-camera").src = "";
  } catch (e) {
    setDot($("dot-gateway"), "error");
    $("meta-gateway").textContent = "Error";
    toast(String(e.message || e));
  }
}

// IMPORTANT: backend'i de resetle!
async function disconnectGateway() {
  // önce polling durdur (interval daha fazla GET atmasın)
  stopPolling();
  stopHold();

  // gateway set ise server-side reset dene
  const gw = state.gateway;
  if (gw) {
    try {
      // manual fetch: apiPost kullanma çünkü birazdan state.gateway boşalacak
      const r = await fetch(`${gw}/api/gateway/disconnect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      // hata olsa bile UI temizlenmeli
      await r.json().catch(() => ({}));
    } catch {
      // gateway zaten düşmüş olabilir -> ignore
    }
  }

  // şimdi local state sıfırla
  state.gateway = "";
  state.robotConnected = false;
  state.robotEnabled = false;

  setDot($("dot-gateway"), "gray");
  $("meta-gateway").textContent = "Disconnected";

  setDot($("dot-robot"), "gray");
  $("meta-robot").textContent = "Disconnected";
  setDot($("dot-aiserver"), "gray");
  $("meta-aiserver").textContent = "Disconnected";

  $("pill-enabled").textContent = "Enabled: ?";
  $("pill-state").textContent = "State: ?";
  $("pill-error").textContent = "Err: ?";
  $("pill-safety").textContent = "Safety: ?";
  $("pill-camera").textContent = "Camera: ?";

  $("img-camera").src = "";
  setCameraOverlay(true, "Gateway disconnected", "Connect Gateway to use camera and controls.");
  toast("Gateway disconnected (server reset).");
}

// ---------- Robot connect/disconnect ----------
async function robotConnect() {
  if (!requireGatewaySync()) return;
  const ip = ($("in-robot").value || "").trim();
  if (!ip) return;

  setDot($("dot-robot"), "connecting");
  $("meta-robot").textContent = "Connecting...";

  try {
    await apiPost("/api/connect", { ip });
    toast("Robot connected.");
    await refreshStatus();
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
    await refreshStatus();
  } catch (e) {
    toast(String(e.message || e));
  }
}

// ---------- AI server set/clear ----------
async function aiSet() {
  if (!requireGatewaySync()) return;
  const url = ($("in-aiserver").value || "").trim();
  if (!url) return;

  setDot($("dot-aiserver"), "connecting");
  $("meta-aiserver").textContent = "Setting...";

  try {
    await apiPost("/api/ai_server", { url });
    toast("AI server set.");
    await refreshStatus();
  } catch (e) {
    setDot($("dot-aiserver"), "error");
    $("meta-aiserver").textContent = "Error";
    toast(String(e.message || e));
  }
}

async function aiClear() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/ai_server/clear", {});
    toast("AI server cleared.");
    await refreshStatus();
  } catch (e) {
    toast(String(e.message || e));
  }
}

// ---------- Robot buttons ----------
async function enableRobot() {
  if (!requireGatewaySync()) return;
  if (!requireRobotConnectedSync()) return;

  try {
    await apiPost("/api/enable", {});
    toast("Robot enabled.");
    await refreshStatus();
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
    await refreshStatus();
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
    await refreshStatus();
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

async function refreshBtn() {
  if (!requireGatewaySync()) return;
  try {
    await refreshStatus();
    toast("Status refreshed.");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function clearSafety() {
  if (!requireGatewaySync()) return;
  try {
    await apiPost("/api/safety/clear", {});
    toast("Safety cleared.");
    await refreshStatus();
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

// ---------- Speed ----------
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

// ---------- Jog mapping ----------
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

// ---------- Gripper ----------
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

// ---------- Wire up ----------
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
  $("btn-aiserver-connect").addEventListener("click", aiSet);
  $("btn-aiserver-disconnect").addEventListener("click", aiClear);

  // robot controls
  $("btn-enable").addEventListener("click", enableRobot);
  $("btn-disable").addEventListener("click", disableRobot);
  $("btn-stop").addEventListener("click", stopRobot);
  $("btn-home").addEventListener("click", homeRobot);
  $("btn-status").addEventListener("click", refreshBtn);
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
    const onDown = (ev) => { ev.preventDefault?.(); startHold(dir); };
    const onUp = (ev) => { ev.preventDefault?.(); stopHold(); };

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
