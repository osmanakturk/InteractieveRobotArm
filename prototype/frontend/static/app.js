// frontend/static/app.js
(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    connDot: $("connDot"),
    connLabel: $("connLabel"),
    stateLabel: $("stateLabel"),
    modeLabel: $("modeLabel"),
    errLabel: $("errLabel"),

    xLabel: $("xLabel"),
    yLabel: $("yLabel"),
    zLabel: $("zLabel"),
    rollLabel: $("rollLabel"),
    pitchLabel: $("pitchLabel"),
    yawLabel: $("yawLabel"),
    jointsShort: $("jointsShort"),

    ipInput: $("ipInput"),
    connectBtn: $("connectBtn"),
    disconnectBtn: $("disconnectBtn"),
    enableBtn: $("enableBtn"),
    disableBtn: $("disableBtn"),
    stopBtn: $("stopBtn"),
    homeBtn: $("homeBtn"),

    frameSelect: $("frameSelect"),
    speedRange: $("speedRange"),
    speedVal: $("speedVal"),
    rateSelect: $("rateSelect"),
    xyzStep: $("xyzStep"),
    rpyStep: $("rpyStep"),

    gripOpen: $("gripOpen"),
    gripClose: $("gripClose"),
    gripPos: $("gripPos"),
  };

  const state = {
    holdTimer: null,
    holdingEl: null,
    pollTimer: null,
    status: null,
    gripperMin: 0,
    gripperMax: 850,
    gripperPos: null,
    gripperAvail: false,
  };

  async function api(path, method = "GET", body = null) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      const msg = data.message || `Request failed: ${res.status}`;
      throw new Error(msg);
    }
    return data;
  }

  function setConnected(on, label) {
    els.connDot.classList.toggle("on", !!on);
    els.connLabel.textContent = label || (on ? "Connected" : "Disconnected");
  }

  function fmt(n) {
    if (n === null || n === undefined) return "-";
    if (typeof n === "number") return n.toFixed(1);
    return String(n);
  }

  function updateUIFromStatus(st) {
    state.status = st;

    setConnected(st.connected, st.connected ? `Connected (${st.ip || ""})` : "Disconnected");
    els.stateLabel.textContent = st.state ?? "-";
    els.modeLabel.textContent = st.mode ?? "-";
    els.errLabel.textContent = st.error_code ?? "-";

    if (Array.isArray(st.pose) && st.pose.length >= 6) {
      els.xLabel.textContent = fmt(st.pose[0]);
      els.yLabel.textContent = fmt(st.pose[1]);
      els.zLabel.textContent = fmt(st.pose[2]);
      els.rollLabel.textContent = fmt(st.pose[3]);
      els.pitchLabel.textContent = fmt(st.pose[4]);
      els.yawLabel.textContent = fmt(st.pose[5]);
    } else {
      els.xLabel.textContent = "-";
      els.yLabel.textContent = "-";
      els.zLabel.textContent = "-";
      els.rollLabel.textContent = "-";
      els.pitchLabel.textContent = "-";
      els.yawLabel.textContent = "-";
    }

    if (Array.isArray(st.joints) && st.joints.length) {
      // short format: j1..j6
      els.jointsShort.textContent = st.joints.map(v => (typeof v === "number" ? v.toFixed(1) : v)).join(", ");
    } else {
      els.jointsShort.textContent = "-";
    }

    // Gripper
    state.gripperAvail = !!st.gripper_available;
    state.gripperMin = st.gripper_min ?? state.gripperMin;
    state.gripperMax = st.gripper_max ?? state.gripperMax;
    state.gripperPos = (typeof st.gripper_pos === "number") ? st.gripper_pos : null;

    if (!state.gripperAvail) {
      els.gripPos.textContent = "N/A (no gripper)";
      els.gripOpen.disabled = true;
      els.gripClose.disabled = true;
      return;
    }

    if (state.gripperPos === null) {
      els.gripPos.textContent = "Unknown";
      els.gripOpen.disabled = false;
      els.gripClose.disabled = false;
      return;
    }

    els.gripPos.textContent = `Pos: ${state.gripperPos}`;
    // Disable if at limits
    els.gripOpen.disabled = state.gripperPos >= state.gripperMax;
    els.gripClose.disabled = state.gripperPos <= state.gripperMin;
  }

  async function pollStatus() {
    try {
      const data = await api("/api/status");
      updateUIFromStatus(data.status);
    } catch (e) {
      setConnected(false, "Disconnected");
    }
  }

  function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollStatus, 500);
    pollStatus();
  }

  // -------------------------
  // Top controls
  // -------------------------
  els.connectBtn.addEventListener("click", async () => {
    try {
      const ip = els.ipInput.value.trim();
      const data = await api("/api/connect", "POST", { ip });
      updateUIFromStatus(data.status);
    } catch (e) {
      alert(e.message);
    }
  });

  els.disconnectBtn.addEventListener("click", async () => {
    try {
      const data = await api("/api/disconnect", "POST");
      updateUIFromStatus(data.status);
    } catch (e) {
      alert(e.message);
    }
  });

  els.enableBtn.addEventListener("click", async () => {
    try {
      const data = await api("/api/enable", "POST");
      updateUIFromStatus(data.status);
    } catch (e) {
      alert(e.message);
    }
  });

  els.disableBtn.addEventListener("click", async () => {
    try {
      const data = await api("/api/disable", "POST");
      updateUIFromStatus(data.status);
    } catch (e) {
      alert(e.message);
    }
  });

  els.stopBtn.addEventListener("click", async () => {
    try {
      await api("/api/stop", "POST");
      await pollStatus();
    } catch (e) {
      alert(e.message);
    }
  });

  els.homeBtn.addEventListener("click", async () => {
    try {
      await api("/api/home", "POST");
      await pollStatus();
    } catch (e) {
      alert(e.message);
    }
  });

  els.frameSelect.addEventListener("change", async () => {
    try {
      await api("/api/frame", "POST", { frame: els.frameSelect.value });
    } catch (e) {
      alert(e.message);
    }
  });

  els.speedRange.addEventListener("input", () => {
    els.speedVal.textContent = `${els.speedRange.value}%`;
  });

  els.speedRange.addEventListener("change", async () => {
    try {
      await api("/api/speed", "POST", { speed_pct: parseInt(els.speedRange.value, 10) });
    } catch (e) {
      alert(e.message);
    }
  });

  // -------------------------
  // Hold-to-jog helper
  // -------------------------
  function getHoldRateMs() {
    return parseInt(els.rateSelect.value, 10) || 120;
  }
  function getXyzStep() {
    const v = parseFloat(els.xyzStep.value);
    return Number.isFinite(v) ? v : 5;
  }
  function getRpyStep() {
    const v = parseFloat(els.rpyStep.value);
    return Number.isFinite(v) ? v : 2;
  }

  async function jogOnce(kind, axis, dir) {
    const sign = dir === "+" ? 1 : -1;
    const payload = { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0 };

    if (kind === "xyz") {
      const step = getXyzStep() * sign;
      if (axis === "x") payload.dx = step;
      if (axis === "y") payload.dy = step;
      if (axis === "z") payload.dz = step;
    } else {
      const step = getRpyStep() * sign;
      if (axis === "rx") payload.droll = step;
      if (axis === "ry") payload.dpitch = step;
      if (axis === "rz") payload.dyaw = step;
    }

    await api("/api/jog", "POST", payload);
  }

  async function gripperOnce(action) {
    await api("/api/gripper", "POST", { action });
    // status returns can update limits quickly
    await pollStatus();
  }

  function stopHold() {
    if (state.holdTimer) clearInterval(state.holdTimer);
    state.holdTimer = null;
    if (state.holdingEl) state.holdingEl.classList.remove("active");
    state.holdingEl = null;
  }

  function startHold(el) {
    stopHold();
    state.holdingEl = el;
    el.classList.add("active");

    const kind = el.dataset.kind;
    const axis = el.dataset.axis;
    const dir = el.dataset.dir;
    const grip = el.dataset.grip;

    const tick = async () => {
      try {
        if (grip) {
          // Disable at limits
          if (!state.gripperAvail) return;
          if (state.gripperPos !== null) {
            if (grip === "open" && state.gripperPos >= state.gripperMax) return;
            if (grip === "close" && state.gripperPos <= state.gripperMin) return;
          }
          await gripperOnce(grip);
        } else {
          await jogOnce(kind, axis, dir);
        }
      } catch (e) {
        // stop on error to avoid spamming
        stopHold();
      }
    };

    tick();
    state.holdTimer = setInterval(tick, getHoldRateMs());
  }

  function bindHold(el) {
    // Mouse / touch / pen
    el.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      if (el.disabled) return;
      el.setPointerCapture?.(ev.pointerId);
      startHold(el);
    });

    el.addEventListener("pointerup", (ev) => {
      ev.preventDefault();
      stopHold();
    });

    el.addEventListener("pointercancel", () => stopHold());
    el.addEventListener("pointerleave", () => {
      // If user drags out while holding, stop
      if (state.holdingEl === el) stopHold();
    });

    // Safety: stop hold on window blur
    window.addEventListener("blur", stopHold);
  }

  document.querySelectorAll(".hold").forEach(bindHold);

  // Prevent long-press context menu on mobile
  window.addEventListener("contextmenu", (e) => {
    if (e.target && e.target.classList && e.target.classList.contains("hold")) {
      e.preventDefault();
    }
  });

  // Init
  startPolling();
})();
