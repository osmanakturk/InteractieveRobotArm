// src/screens/PickPlaceScreen.tsx
import React, { useCallback, useMemo, useRef, useState, useEffect } from "react";
import {
  ActivityIndicator,
  Alert,
  ImageBackground,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
  LayoutChangeEvent,
  GestureResponderEvent,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";
import { useFocusEffect } from "@react-navigation/native";
import MaterialCommunityIcons from "@expo/vector-icons/MaterialCommunityIcons";
import { useConnection } from "../connection/ConnectionContext";

// --- Types ---
type ControlMode = "XYZ" | "RXYZ";
type DotStatus = "idle" | "connecting" | "connected" | "error";

type Selection = {
  u: number; // 0..1
  v: number; // 0..1
  // UI marker position in the preview container (px)
  px: number;
  py: number;
  selectionId?: string;
};

// UPDATED: add manual confirmation phases
type WorkflowPhase = "idle" | "pick_armed" | "place_armed" | "busy" | "pick_hold" | "place_hold";

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  const withProto = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  return withProto.replace(/\/+$/, "");
}

async function safeJson(resp: Response) {
  try {
    return await resp.json();
  } catch {
    return {};
  }
}

async function sleep(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

function pctToWidth(pct: number) {
  const v = clamp(pct, 0, 100);
  return `${v}%`;
}

function dotStyle(status: DotStatus) {
  if (status === "connected") return styles.dotGreen;
  if (status === "connecting") return styles.dotBlue;
  if (status === "error") return styles.dotRed;
  return styles.dotGray;
}

function IconBtn({
  icon,
  label,
  onPress,
  tone = "ghost",
  disabled,
  size = "md",
}: {
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
  label?: string;
  onPress: () => void;
  tone?: "ghost" | "primary" | "danger" | "success";
  disabled?: boolean;
  size?: "sm" | "md";
}) {
  return (
    <Pressable
      disabled={!!disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.iconBtn,
        size === "sm" ? styles.iconBtnSm : null,
        tone === "primary"
          ? styles.iconBtnPrimary
          : tone === "danger"
          ? styles.iconBtnDanger
          : tone === "success"
          ? styles.iconBtnSuccess
          : null,
        disabled ? { opacity: 0.45 } : null,
        pressed && !disabled ? { opacity: 0.9 } : null,
      ]}
    >
      <MaterialCommunityIcons name={icon} size={size === "sm" ? 16 : 18} color="white" />
      {!!label && <Text style={[styles.iconBtnText, size === "sm" ? styles.iconBtnTextSm : null]}>{label}</Text>}
    </Pressable>
  );
}

function HoldBtn({
  label,
  onPress,
  onHoldStart,
  onHoldEnd,
  disabled,
}: {
  label: string;
  onPress?: () => void;
  onHoldStart?: () => void;
  onHoldEnd?: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      disabled={!!disabled}
      onPress={onPress}
      onPressIn={onHoldStart}
      onPressOut={onHoldEnd}
      style={({ pressed }) => [
        styles.holdBtn,
        disabled ? { opacity: 0.4 } : null,
        pressed && !disabled ? { opacity: 0.92, transform: [{ scale: 0.99 }] } : null,
      ]}
    >
      <Text style={styles.holdBtnText}>{label}</Text>
    </Pressable>
  );
}

function BarCard({
  title,
  rightText,
  valuePct,
  inactive,
}: {
  title: string;
  rightText: string;
  valuePct: number;
  inactive?: boolean;
}) {
  return (
    <View style={styles.barCard}>
      <View style={styles.barCardTop}>
        <Text style={styles.barCardTitle}>{title}</Text>
        <Text style={styles.barCardSub}>{rightText}</Text>
      </View>
      <View style={[styles.barTrack, inactive ? styles.barTrackInactive : null]}>
        <View style={[styles.barFill, inactive ? styles.barFillInactive : null, { width: pctToWidth(valuePct) }]} />
      </View>
    </View>
  );
}

function TelemetryRow({ k, v }: { k: string; v: string }) {
  return (
    <View style={styles.telemetryRow}>
      <Text style={styles.telemetryKey} numberOfLines={1}>
        {k}
      </Text>
      <Text style={styles.telemetryVal} numberOfLines={1}>
        {v}
      </Text>
    </View>
  );
}

// --- helpers for "contain" mapping (img is drawn with object-fit: contain)
function mapTapToNormalizedContain(
  tapX: number,
  tapY: number,
  viewW: number,
  viewH: number,
  imgAspect: number // width/height (e.g. 4/3)
): { u: number; v: number } | null {
  if (viewW <= 0 || viewH <= 0) return null;

  const viewAspect = viewW / viewH;

  let dispW = viewW;
  let dispH = viewH;
  if (viewAspect > imgAspect) {
    dispH = viewH;
    dispW = dispH * imgAspect;
  } else {
    dispW = viewW;
    dispH = dispW / imgAspect;
  }

  const offsetX = (viewW - dispW) / 2;
  const offsetY = (viewH - dispH) / 2;

  if (tapX < offsetX || tapX > offsetX + dispW || tapY < offsetY || tapY > offsetY + dispH) {
    return null;
  }

  const u = (tapX - offsetX) / dispW;
  const v = (tapY - offsetY) / dispH;

  return { u: clamp(u, 0, 1), v: clamp(v, 0, 1) };
}

export default function PickPlaceScreen({ navigation, route }: any) {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const { gateway, robot, aiserver, live, wsConnected } = useConnection();

  const gatewayFromParams: string = route?.params?.gateway || route?.params?.baseUrl || "";
  const gatewayBase = useMemo(
    () => normalizeBaseUrl(gateway.value || gatewayFromParams),
    [gateway.value, gatewayFromParams]
  );

  const aiBase = useMemo(() => normalizeBaseUrl(aiserver.value || ""), [aiserver.value]);

  const [streamNonce, setStreamNonce] = useState(0);

  const realsenseMjpegUrl = useMemo(() => {
    if (!gatewayBase) return "";
    return `${gatewayBase}/api/cameras/realsense/stream/mjpeg?t=${streamNonce}`;
  }, [gatewayBase, streamNonce]);

  // UI state
  const [mode] = useState<ControlMode>("XYZ");
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  const [speedPct, setSpeedPct] = useState<number>(35);

  const [gripperPct, setGripperPct] = useState<number>(0);
  const [gripperLoading, setGripperLoading] = useState(false);
  const [gripperAvailable, setGripperAvailable] = useState<boolean>(true);

  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraStarted, setCameraStarted] = useState<boolean>(false);
  const [cameraLastError, setCameraLastError] = useState<string>("");

  const [aiLoading, setAiLoading] = useState(false);
  const [aiModeActive, setAiModeActive] = useState(false);
  const [aiModeExitCode, setAiModeExitCode] = useState<number | null>(null);
  const [aiModeErr, setAiModeErr] = useState<string>("");

  // Pick/Place workflow state
  const [phase, setPhase] = useState<WorkflowPhase>("idle");
  const [hasPicked, setHasPicked] = useState(false);

  // Object selection state
  const [selection, setSelection] = useState<Selection | null>(null);
  const [selecting, setSelecting] = useState(false);
  const cameraLayout = useRef<{ w: number; h: number }>({ w: 0, h: 0 });

  // hold-to-jog loop
  const jogTimer = useRef<any>(null);
  const jogAbort = useRef<boolean>(false);

  const robotConnected = !!live?.status?.connected;
  const enabled = !!live?.status?.is_enabled;
  const safetyMsg = (live as any)?.safety?.message || "";
  const safetyLimit = !!(live as any)?.safety?.limit_hit;

  // Dots
  const gatewayUp = !!gatewayBase && wsConnected;
  const gwDot: DotStatus = !gatewayBase ? "idle" : wsConnected ? "connected" : "error";
  const rbDot: DotStatus = !gatewayBase ? "idle" : !gatewayUp ? "error" : robotConnected ? "connected" : "error";
  const camDot: DotStatus = !gatewayBase ? "idle" : !gatewayUp ? "error" : cameraStarted ? "connected" : "idle";

  const aiConnected = !!(live as any)?.ai_server?.connected;
  const aiConfigured = !!(live as any)?.ai_server?.configured;
  const aiDot: DotStatus = !gatewayBase
    ? "idle"
    : !gatewayUp
    ? "error"
    : aiConnected
    ? "connected"
    : aiConfigured
    ? "error"
    : "idle";

  // API helpers (Gateway)
  const apiPost = useCallback(
    async (path: string, body?: any) => {
      if (!gatewayBase) throw new Error("Gateway not set.");
      const resp = await fetch(`${gatewayBase}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body ?? {}),
      });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${resp.status}`);
      return data;
    },
    [gatewayBase]
  );

  const apiGet = useCallback(
    async (path: string) => {
      if (!gatewayBase) throw new Error("Gateway not set.");
      const resp = await fetch(`${gatewayBase}${path}`, { method: "GET" });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${resp.status}`);
      return data;
    },
    [gatewayBase]
  );

  // API helpers (AI)
  const aiPost = useCallback(
    async (path: string, body?: any) => {
      if (!aiBase) throw new Error("AI Server not set.");
      const resp = await fetch(`${aiBase}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body ?? {}),
      });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${resp.status}`);
      return data;
    },
    [aiBase]
  );

  const aiGet = useCallback(
    async (path: string) => {
      if (!aiBase) throw new Error("AI Server not set.");
      const resp = await fetch(`${aiBase}${path}`, { method: "GET" });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) throw new Error(data?.message || `HTTP ${resp.status}`);
      return data;
    },
    [aiBase]
  );

const waitAiWorkerIdle = useCallback(
  async (timeoutMs: number = 20000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      const st = await aiGet("/api/ai_server/pick_place/status");
      const busy = !!st?.busy;
      if (!busy) {
        const err = String(st?.error || "");
        const last = st?.last_action || {};
        if (err) throw new Error(err);
        if (last && last.ok === false) throw new Error("AI action failed.");
        return st;
      }
      await sleep(150);
    }
    throw new Error("Timeout waiting AI worker to finish.");
  },
  [aiGet]
);

  const refreshCameraStatus = useCallback(async () => {
    if (!gatewayBase) return;
    setCameraLoading(true);
    try {
      const data = await apiGet("/api/cameras/realsense/status");
      const cam = data?.camera || {};
      setCameraStarted(!!cam.started);
      setCameraLastError(cam.last_error || "");
    } catch (e: any) {
      setCameraStarted(false);
      setCameraLastError(e?.message || "Camera status failed.");
    } finally {
      setCameraLoading(false);
    }
  }, [apiGet, gatewayBase]);

  // Live camera sync
  const wsCamStarted = (live as any)?.camera?.started;
  const wsCamErr = (live as any)?.camera?.last_error;
  useEffect(() => {
    if (typeof wsCamStarted === "boolean") setCameraStarted(wsCamStarted);
    if (typeof wsCamErr === "string") setCameraLastError(wsCamErr);
  }, [wsCamStarted, wsCamErr]);

  const refreshGripperStatus = useCallback(async () => {
    if (!gatewayBase) return;
    setGripperLoading(true);
    try {
      const data = await apiGet("/api/robot/gripper/status");
      const g = data?.gripper || data || {};

      const available =
        typeof g.available === "boolean"
          ? g.available
          : typeof g.connected === "boolean"
          ? g.connected
          : typeof g.gripper_pct === "number" || typeof g.pct === "number"
          ? true
          : false;

      setGripperAvailable(available);

      if (!available) {
        setGripperPct(0);
        return;
      }

      const pct =
        typeof g.gripper_pct === "number"
          ? g.gripper_pct
          : typeof g.pct === "number"
          ? g.pct
          : typeof g.percent === "number"
          ? g.percent
          : 0;

      setGripperPct(clamp(Math.round(pct), 0, 100));
    } catch {
      setGripperAvailable(false);
      setGripperPct(0);
    } finally {
      setGripperLoading(false);
    }
  }, [apiGet, gatewayBase]);

  // AI status
  const refreshAiStatus = useCallback(async () => {
    if (!aiBase) return;
    try {
      const data = await aiGet("/api/ai_server/modes/status");
      const active = data?.active_mode || {};
      setAiModeActive(!!active.active);
      setAiModeExitCode(
        typeof active.exit_code === "number" ? active.exit_code : active.exit_code === null ? null : null
      );
      setAiModeErr("");
    } catch (e: any) {
      setAiModeActive(false);
      setAiModeExitCode(null);
      setAiModeErr(e?.message || "AI status failed.");
    }
  }, [aiBase, aiGet]);

  // Focus refresh
  useFocusEffect(
    useCallback(() => {
      refreshCameraStatus();
      refreshGripperStatus();
      refreshAiStatus();

      return () => {
        jogAbort.current = true;
        if (jogTimer.current) clearTimeout(jogTimer.current);
        jogTimer.current = null;
      };
    }, [refreshCameraStatus, refreshGripperStatus, refreshAiStatus])
  );

  // If AI mode stops, reset workflow
  useEffect(() => {
    if (!aiModeActive) {
      setSelection(null);
      setPhase("idle");
      setHasPicked(false);
    }
  }, [aiModeActive]);

  // If robot disconnects or disables, reset workflow
  useEffect(() => {
    if (!robotConnected || !enabled) {
      setSelection(null);
      setPhase("idle");
      setHasPicked(false);
    }
  }, [robotConnected, enabled]);

  // WS gripper_pct update only when it changes
  const wsGripperPct = (live as any)?.status?.gripper_pct;
  useEffect(() => {
    if (typeof wsGripperPct !== "number") return;
    setGripperAvailable(true);
    setGripperPct(clamp(Math.round(wsGripperPct), 0, 100));
  }, [wsGripperPct]);

  // --- Robot actions
  const sendJogDelta = useCallback(
    async (payload: any) => {
      try {
        await apiPost("/api/robot/jog", payload);
      } catch {
        // silent
      }
    },
    [apiPost]
  );

  const startHoldJog = useCallback(
    (axis: string, dir: 1 | -1) => {
      if (!gatewayBase || !robotConnected) return;

      jogAbort.current = false;
      const stepPos = 5; // mm
      const stepRot = 3; // deg

      const tick = async () => {
        if (jogAbort.current) return;

        if (mode === "XYZ") {
          const dx = axis === "x" ? stepPos * dir : 0;
          const dy = axis === "y" ? stepPos * dir : 0;
          const dz = axis === "z" ? stepPos * dir : 0;
          await sendJogDelta({ dx, dy, dz, droll: 0, dpitch: 0, dyaw: 0 });
        } else {
          const droll = axis === "rx" ? stepRot * dir : 0;
          const dpitch = axis === "ry" ? stepRot * dir : 0;
          const dyaw = axis === "rz" ? stepRot * dir : 0;
          await sendJogDelta({ dx: 0, dy: 0, dz: 0, droll, dpitch, dyaw });
        }

        jogTimer.current = setTimeout(tick, 120);
      };

      tick();
    },
    [gatewayBase, robotConnected, mode, sendJogDelta]
  );

  const endHoldJog = useCallback(() => {
    jogAbort.current = true;
    if (jogTimer.current) clearTimeout(jogTimer.current);
    jogTimer.current = null;
  }, []);

  const hardResetWorkflow = useCallback(async () => {
    setSelection(null);
    setSelecting(false);
    setAiLoading(false);
    setPhase("idle");
    setHasPicked(false);

    // best-effort: stop robot motion (important)
    try {
      await apiPost("/api/robot/stop");
    } catch {}

    if (!aiBase) return;

    // best-effort: cancel/reset AI worker
    try {
      await aiPost("/api/ai_server/pick_place/cancel");
      return;
    } catch {}
    try {
      await aiPost("/api/ai_server/pick_place/reset");
      return;
    } catch {}
  }, [aiBase, aiPost, apiPost]);

  const onStop = useCallback(async () => {
    endHoldJog();
    await hardResetWorkflow();
  }, [endHoldJog, hardResetWorkflow]);

  // UPDATED: gripper action = manual confirmation step (pick_hold/place_hold)
  const sendGripper = useCallback(
    async (action: "open" | "close") => {
      try {
        await apiPost("/api/robot/gripper", { action });
        await refreshGripperStatus();

        // --- manual confirmations ---
        if (phase === "pick_hold") {
          // any gripper action confirms pick (your requirement)
          setHasPicked(true);
          setPhase("idle");
          // best-effort: notify AI if endpoint exists
          try {
            await aiPost("/api/ai_server/pick_place/confirm_pick", { via: "gripper", action });
          } catch {
            // ignore if endpoint doesn't exist
          }
        } else if (phase === "place_hold") {
          // any gripper action confirms place (typical: open)
          setHasPicked(false);
          setPhase("idle");
          // best-effort: notify AI if endpoint exists
          try {
            await aiPost("/api/ai_server/pick_place/confirm_place", { via: "gripper", action });
          } catch {
            // ignore
          }
        }
      } catch (e: any) {
        Alert.alert("Gripper failed", e?.message || "Gripper action failed.");
      }
    },
    [apiPost, refreshGripperStatus, phase, aiPost]
  );

  const onEnableDisable = useCallback(async () => {
    try {
      if (!gatewayBase) return Alert.alert("Gateway missing", "Please set Gateway in Connection Hub.");
      if (!robotConnected) return Alert.alert("Robot not connected", "Connect the robot from Connection Hub first.");
      if (enabled) await apiPost("/api/robot/disable");
      else await apiPost("/api/robot/enable");
    } catch (e: any) {
      Alert.alert("Action failed", e?.message || "Enable/Disable failed.");
    }
  }, [apiPost, enabled, gatewayBase, robotConnected]);

  const onClearSafety = useCallback(async () => {
    try {
      await apiPost("/api/robot/safety/clear");
    } catch (e: any) {
      Alert.alert("Clear Safety failed", e?.message || "Clear Safety failed.");
    }
  }, [apiPost]);

  const onRobotHome = useCallback(async () => {
    try {
      await apiPost("/api/robot/home");
    } catch (e: any) {
      Alert.alert("Home failed", e?.message || "Robot home action failed.");
    }
  }, [apiPost]);

  const onCameraStart = useCallback(async () => {
    if (!gatewayBase) return;
    setCameraLoading(true);
    try {
      await apiPost("/api/cameras/realsense/start");
      setStreamNonce((n) => n + 1);
      await refreshCameraStatus();
    } catch (e: any) {
      Alert.alert("Camera start failed", e?.message || "Camera start failed.");
    } finally {
      setCameraLoading(false);
    }
  }, [apiPost, gatewayBase, refreshCameraStatus]);

  const onCameraStop = useCallback(async () => {
    if (!gatewayBase) return;
    setCameraLoading(true);
    try {
      await apiPost("/api/cameras/realsense/stop");
      setCameraStarted(false);
      setStreamNonce((n) => n + 1);
      await refreshCameraStatus();
    } catch (e: any) {
      Alert.alert("Camera stop failed", e?.message || "Camera stop failed.");
    } finally {
      setCameraLoading(false);
    }
  }, [apiPost, gatewayBase, refreshCameraStatus]);

  const pushSpeed = useCallback(
    async (nextPct: number) => {
      const v = clamp(Math.round(nextPct), 1, 100);
      setSpeedPct(v);
      try {
        await apiPost("/api/robot/speed", { speed_pct: v });
      } catch {
        // silent
      }
    },
    [apiPost]
  );

  const goConnectionHub = useCallback(() => navigation.navigate("ConnectionHub"), [navigation]);
  const goModeSelect = useCallback(() => navigation.navigate("ModeSelect"), [navigation]);

  // AI mode controls
  const onAiStartPickPlace = useCallback(async () => {
    if (!aiBase) return Alert.alert("AI Server missing", "Set AI Server base URL in Connection Hub.");
    if (!gatewayBase) return Alert.alert("Gateway missing", "Set Gateway base URL first.");
    setAiLoading(true);
    try {
      await aiPost("/api/ai_server/modes/start", { mode: "pick_and_place", gateway_url: gatewayBase });
      await refreshAiStatus();

      // starting AI should reset workflow state
      setSelection(null);
      setPhase("idle");
      setHasPicked(false);
    } catch (e: any) {
      Alert.alert("AI start failed", e?.message || "Failed to start pick_and_place mode.");
    } finally {
      setAiLoading(false);
    }
  }, [aiBase, gatewayBase, aiPost, refreshAiStatus]);

  const onAiStop = useCallback(async () => {
    if (!aiBase) return;
    setAiLoading(true);
    try {
      await aiPost("/api/ai_server/modes/stop");
      await refreshAiStatus();
      await hardResetWorkflow();
    } catch (e: any) {
      Alert.alert("AI stop failed", e?.message || "Failed to stop mode.");
    } finally {
      setAiLoading(false);
    }
  }, [aiBase, aiPost, refreshAiStatus, hardResetWorkflow]);

  const onAiToggle = useCallback(() => {
    if (aiModeActive) onAiStop();
    else onAiStartPickPlace();
  }, [aiModeActive, onAiStop, onAiStartPickPlace]);

  // Vision pose (MANUAL always allowed)
  const onVisionPose = useCallback(async () => {
    if (!gatewayBase) return Alert.alert("Gateway missing", "Set Gateway URL first.");
    if (!robotConnected) return Alert.alert("Robot not connected", "Connect robot first.");
    try {
      await apiPost("/api/robot/vision_pose");
    } catch {
      Alert.alert("Vision pose missing", "Add endpoint: POST /api/robot/vision_pose in gateway.");
    }
  }, [apiPost, gatewayBase, robotConnected]);

  // --- Object selection: camera layout + tap handler
  const onCameraAspectLayout = useCallback((e: LayoutChangeEvent) => {
    const { width: w, height: h } = e.nativeEvent.layout;
    cameraLayout.current = { w, h };
  }, []);

  const postSelectionToAi = useCallback(
    async (u: number, v: number) => {
      if (!aiBase) throw new Error("AI Server not set.");
      const data = await aiPost("/api/ai_server/pick_place/select", {
        u,
        v,
        source: "realsense_mjpeg",
        ts: Date.now(),
      });
      const sid = data?.selection?.id;
      return typeof sid === "string" ? sid : undefined;
    },
    [aiBase, aiPost]
  );

  // UPDATED: only enforce WAIT_PICK when starting a new PICK
  const ensureWorkerReadyForPick = useCallback(async () => {
    if (!aiBase) throw new Error("AI Server not set.");
    const st = await aiGet("/api/ai_server/pick_place/status");
    if (!!st?.busy) throw new Error("AI worker is busy. Please wait.");

    const p = String(st?.phase || "");
    if (p && p !== "WAIT_PICK") {
      // reset to WAIT_PICK
      await aiPost("/api/ai_server/pick_place/reset");
      const st2 = await aiGet("/api/ai_server/pick_place/status");
      const p2 = String(st2?.phase || "");
      if (p2 !== "WAIT_PICK") throw new Error(`AI worker phase mismatch after reset: got ${p2}`);
    }
    return st;
  }, [aiBase, aiGet, aiPost]);

  const executePickPlace = useCallback(
    async (action: "pick" | "place", u: number, v: number, px: number, py: number) => {
      if (!aiModeActive) throw new Error("AI mode is not running.");
      if (!aiBase) throw new Error("AI Server not set.");
      if (!gatewayBase) throw new Error("Gateway not set.");
      if (!robotConnected) throw new Error("Robot not connected.");
      if (!enabled) throw new Error("Robot is disabled.");

      // place requires user-confirmed pick
      if (action === "place" && !hasPicked) throw new Error("Place not ready. Do Pick first.");

      // pick should start from clean worker state
      if (action === "pick") {
        await ensureWorkerReadyForPick();
      } else {
        // For place: at least ensure worker not busy (do NOT block on WAIT_PICK here)
        const st = await aiGet("/api/ai_server/pick_place/status");
        if (!!st?.busy) throw new Error("AI worker is busy. Please wait.");
      }

      endHoldJog();
      setPhase("busy");
      setSelecting(true);
      setAiLoading(true);
      setSelection({ u, v, px, py });

      try {
        // 1) select
        const sid = await postSelectionToAi(u, v);
        if (!sid) throw new Error("Selection ID missing from AI response.");
        setSelection((prev) => (prev ? { ...prev, selectionId: sid } : prev));

        // 2) execute (async worker starts)
        await aiPost("/api/ai_server/pick_place/execute", {
          action,
          selection_id: sid,
        });

        // 3) wait worker done
        await waitAiWorkerIdle(25000);

        // 4) IMPORTANT: manual confirmation step
        setSelection(null);

        if (action === "pick") {
          // robot is at pick pose, user must close/open gripper to confirm
          setPhase("pick_hold");
        } else {
          // robot is at place pose, user must open/close gripper to confirm
          setPhase("place_hold");
        }
      } finally {
        setSelecting(false);
        setAiLoading(false);
      }
    },
    [
      aiBase,
      aiModeActive,
      aiPost,
      enabled,
      endHoldJog,
      ensureWorkerReadyForPick,
      gatewayBase,
      hasPicked,
      postSelectionToAi,
      robotConnected,
      waitAiWorkerIdle,
      aiGet,
    ]
  );

  const onTapCamera = useCallback(
    async (evt: GestureResponderEvent) => {
      if (!cameraStarted) return;

      if (!aiModeActive) return;
      if (!aiBase) return;
      if (!gatewayBase) return;
      if (!robotConnected) return;
      if (!enabled) return;

      // tap only meaningful when armed
      if (phase !== "pick_armed" && phase !== "place_armed") return;
      if (selecting || aiLoading) return;

      const { locationX, locationY } = evt.nativeEvent;
      const { w, h } = cameraLayout.current;

      const mapped = mapTapToNormalizedContain(locationX, locationY, w, h, 4 / 3);
      if (!mapped) return;

      const action: "pick" | "place" = phase === "pick_armed" ? "pick" : "place";

      if (action === "place" && !hasPicked) {
        Alert.alert("Place not ready", "You must confirm a Pick (with Gripper) before placing.");
        return;
      }

      try {
        await executePickPlace(action, mapped.u, mapped.v, locationX, locationY);
      } catch (e: any) {
        const msg = String(e?.message || "");

        // If AI complains about order, auto-reset
        if (
          msg.toLowerCase().includes("invalid order") ||
          msg.toLowerCase().includes("expected pick") ||
          msg.toLowerCase().includes("wait pick") ||
          msg.toLowerCase().includes("do first pick")
        ) {
          await hardResetWorkflow();
          Alert.alert("AI state reset", "AI worker was out of sync. Workflow reset. Please Pick again.");
          return;
        }

        Alert.alert(`${action === "pick" ? "Pick" : "Place"} failed`, msg || "Execution failed.");
        setPhase(action === "pick" ? "pick_armed" : "place_armed");
      }
    },
    [
      aiBase,
      aiLoading,
      aiModeActive,
      cameraStarted,
      enabled,
      executePickPlace,
      gatewayBase,
      hardResetWorkflow,
      hasPicked,
      phase,
      robotConnected,
      selecting,
    ]
  );

  const onClearSelection = useCallback(() => setSelection(null), []);

  const onArmPick = useCallback(async () => {
    if (!aiModeActive) return Alert.alert("AI not running", "Start AI first.");
    if (!cameraStarted) return Alert.alert("Camera off", "Start camera first.");
    if (!robotConnected) return Alert.alert("Robot not connected", "Connect robot first.");
    if (!enabled) return Alert.alert("Robot disabled", "Enable robot first.");
    if (!aiBase) return;

    try {
      await ensureWorkerReadyForPick();
    } catch (e: any) {
      Alert.alert("Pick not ready", String(e?.message || "AI worker not ready."));
      return;
    }

    setSelection(null);
    setPhase("pick_armed");
  }, [aiModeActive, aiBase, cameraStarted, enabled, ensureWorkerReadyForPick, robotConnected]);

  const onArmPlace = useCallback(async () => {
    if (!hasPicked) return Alert.alert("Pick required", "Confirm Pick with Gripper first.");
    if (!aiModeActive) return Alert.alert("AI not running", "Start AI first.");
    if (!cameraStarted) return Alert.alert("Camera off", "Start camera first.");
    if (!robotConnected) return Alert.alert("Robot not connected", "Connect robot first.");
    if (!enabled) return Alert.alert("Robot disabled", "Enable robot first.");

    setSelection(null);
    setPhase("place_armed");
  }, [aiModeActive, cameraStarted, enabled, hasPicked, robotConnected]);

  // Cancel MUST stop everything
  const onCancelWorkflow = useCallback(async () => {
    await hardResetWorkflow();
  }, [hardResetWorkflow]);

  // Layout
  const leftW = useMemo(() => clamp(Math.round(width * 0.2), 240, 310), [width]);
  const rightW = useMemo(() => clamp(Math.round(width * 0.2), 240, 310), [width]);

  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>Manual control is designed for landscape.</Text>
      </View>
    );
  }

 // Jog lock: keep jog locked during workflow, BUT gripper should stay usable
const workflowLocked = phase !== "idle";
const jogLocked = workflowLocked || selecting || aiLoading;

// Jog only when idle and not busy/selecting
const canJog = !!gatewayBase && robotConnected && !jogLocked;

// Gripper: allow even during workflow (pick_hold/place_hold), only block during "busy" or while sending
const canGrip = !!gatewayBase && robotConnected && !aiLoading && !selecting;

  const gripperRightText = gripperLoading ? "Loading…" : !gripperAvailable ? "N/A" : `${gripperPct}/100`;

  const canCameraToggle = !!gatewayBase && !cameraLoading && phase === "idle";
  const canRobotToggle = !!gatewayBase && robotConnected; // allow even during workflow
  const canAiToggle = !!aiBase && !!gatewayBase && !aiLoading && phase === "idle";

  const aiBtnLabel = aiModeActive ? "AI-Stop" : "AI-Start";
  const aiBtnIcon = aiModeActive ? "stop-circle-outline" : "play-circle-outline";
  const aiBtnTone = aiModeActive ? "danger" : "primary";

  const robotBtnLabel = enabled ? "Robot-Off" : "Robot-On";
  const robotBtnIcon = enabled ? "robot-industrial" : "robot-industrial-outline";
  const robotBtnTone = enabled ? "danger" : "success";

  const camBtnLabel = cameraStarted ? "Cam-Off" : "Cam-On";
  const camBtnIcon = cameraStarted ? "cctv-off" : "cctv";
  const camBtnTone = cameraStarted ? "danger" : "success";

  const onCameraToggle = () => {
    if (cameraStarted) onCameraStop();
    else onCameraStart();
  };

  const instructionText = useMemo(() => {
    if (!gatewayBase) return "Gateway is not set. Open Connection Hub.";
    if (!robotConnected) return "Robot is not connected.";
    if (!aiModeActive) return "Start AI to use pick&place.";
    if (!cameraStarted) return "Start camera first.";
    if (!enabled) return "Enable robot first.";

    if (phase === "pick_armed") return "Pick armed → tap camera → robot goes to pick pose.";
    if (phase === "pick_hold") return "Robot at pick pose → use Gripper (Grip-/Grip+) to CONFIRM pick.";
    if (phase === "place_armed") return "Place armed → tap camera → robot goes to place pose.";
    if (phase === "place_hold") return "Robot at place pose → use Gripper (Grip-/Grip+) to CONFIRM place.";
    if (phase === "busy") return "Working… Please wait.";

    if (hasPicked) return "Pick confirmed ✓ Now press Place, then tap camera.";
    return "Press Pick, then tap camera.";
  }, [aiModeActive, cameraStarted, enabled, gatewayBase, hasPicked, phase, robotConnected]);

  const cameraHint = useMemo(() => {
    if (!cameraStarted) return "";
    if (!aiModeActive) return "Start AI to use pick&place";
    if (phase === "pick_armed") return "Tap to GO PICK";
    if (phase === "place_armed") return "Tap to GO PLACE";
    if (phase === "pick_hold") return "Use GRIP to confirm";
    if (phase === "place_hold") return "Use GRIP to confirm";
    if (phase === "busy") return "Working…";
    if (hasPicked) return "Press Place to continue";
    return "Press Pick to start";
  }, [aiModeActive, cameraStarted, hasPicked, phase]);

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />
      <SafeAreaView style={styles.safe}>
        {/* Top bar */}
        <View style={styles.topBar}>
          <View style={styles.topLeft}>
            <IconBtn icon="menu" label="Status" onPress={() => setLeftOpen(true)} size="md" />
          </View>

          <View style={styles.topMid}>
            <View style={styles.topGroup}>
              <IconBtn
                icon={camBtnIcon as any}
                label={camBtnLabel}
                tone={camBtnTone}
                onPress={onCameraToggle}
                disabled={!canCameraToggle}
                size="sm"
              />

              {/* Vision is MANUAL always */}
              <IconBtn
                icon="eye-outline"
                label="Vision"
                tone="primary"
                onPress={onVisionPose}
                disabled={!canRobotToggle}
                size="sm"
              />
            </View>

            <View style={styles.topDots}>
              <View style={styles.dotLine}>
                <View style={[styles.dot, dotStyle(gwDot)]} />
                <Text style={styles.dotText}>GW</Text>
              </View>
              <View style={styles.dotLine}>
                <View style={[styles.dot, dotStyle(rbDot)]} />
                <Text style={styles.dotText}>RB</Text>
              </View>
              <View style={styles.dotLine}>
                <View style={[styles.dot, dotStyle(camDot)]} />
                <Text style={styles.dotText}>CAM</Text>
              </View>
              <View style={styles.dotLine}>
                <View style={[styles.dot, dotStyle(aiDot)]} />
                <Text style={styles.dotText}>AI</Text>
              </View>
            </View>

            <View style={styles.topGroup}>
              <IconBtn
                icon={aiBtnIcon as any}
                label={aiLoading ? "AI..." : aiBtnLabel}
                tone={aiBtnTone as any}
                onPress={onAiToggle}
                disabled={!canAiToggle}
                size="sm"
              />

              <IconBtn
                icon={robotBtnIcon as any}
                label={robotBtnLabel}
                tone={robotBtnTone}
                onPress={onEnableDisable}
                disabled={!gatewayBase || !robotConnected || phase !== "idle"} // safer: enable/disable only idle
                size="sm"
              />
            </View>
          </View>

          <View style={styles.topRight}>
            <IconBtn icon="cog-outline" label="Settings" onPress={() => setRightOpen(true)} size="md" />
          </View>
        </View>

        {/* Body */}
        <View style={styles.body}>
          {/* LEFT */}
          <View style={[styles.panel, { width: leftW }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 14 }}>
              <BarCard title="Speed" rightText={`${speedPct}%`} valuePct={speedPct} />
              <View style={styles.speedBtnsRow}>
                <IconBtn icon="minus" label="Speed" onPress={() => pushSpeed(speedPct - 5)} disabled={!gatewayBase || workflowLocked} />
                <IconBtn icon="plus" label="Speed" onPress={() => pushSpeed(speedPct + 5)} disabled={!gatewayBase || workflowLocked} />
              </View>

              <View style={{ marginTop: 12 }}>
                <Pressable onPress={onStop} style={({ pressed }) => [styles.stopBtn, pressed ? { opacity: 0.92 } : null]}>
                  <Text style={styles.stopText}>STOP</Text>
                </Pressable>
              </View>

              <Text style={styles.note}>{instructionText}</Text>

              {selection ? (
                <Text style={styles.note}>
                  Selected: u={selection.u.toFixed(3)}, v={selection.v.toFixed(3)}{" "}
                  {selection.selectionId ? "✓" : selecting ? "(sending…)" : ""}
                </Text>
              ) : null}

              <Text style={styles.note}>Gripper: {gripperRightText}</Text>
            </ScrollView>
          </View>

          {/* CENTER */}
          <View style={styles.center}>
            <View style={styles.cameraCard}>
              {!gatewayBase ? (
                <View style={styles.cameraPlaceholder}>
                  <Text style={styles.cameraPlaceholderText}>No Gateway URL</Text>
                </View>
              ) : !cameraStarted ? (
                <View style={styles.cameraPlaceholder}>
                  <Text style={styles.cameraPlaceholderText}>Camera is OFF</Text>
                  <Text style={[styles.cameraPlaceholderText, { marginTop: 8, fontSize: 12, opacity: 0.75 }]}>
                    Press Cam-On to start
                  </Text>
                  {!!cameraLastError && (
                    <Text style={[styles.cameraPlaceholderText, { marginTop: 10, fontSize: 11, opacity: 0.7 }]}>
                      {cameraLastError}
                    </Text>
                  )}
                </View>
              ) : (
                <View style={styles.cameraBox}>
                  <View style={styles.cameraAspect} onLayout={onCameraAspectLayout}>
                    <WebView
                      key={`cam-${streamNonce}`}
                      originWhitelist={["*"]}
                      javaScriptEnabled={false}
                      domStorageEnabled={false}
                      startInLoadingState
                      renderLoading={() => (
                        <View style={styles.webLoading}>
                          <ActivityIndicator />
                          <Text style={styles.webLoadingText}>Loading camera…</Text>
                        </View>
                      )}
                      source={{
                        html: `
                          <html>
                            <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
                            <body style="margin:0;background:#000;">
                              <img src="${realsenseMjpegUrl}" style="width:100%;height:100%;object-fit:contain;" />
                            </body>
                          </html>
                        `,
                      }}
                      style={styles.web}
                    />

                    {/* TAP overlay */}
                    <Pressable
                      style={styles.tapOverlay}
                      onPress={onTapCamera}
                      disabled={!cameraStarted || phase === "busy" || selecting || aiLoading}
                    >
                      {selection && (
                        <View pointerEvents="none" style={[styles.marker, { left: selection.px - 10, top: selection.py - 10 }]} />
                      )}
                    </Pressable>

                    {cameraStarted && (
                      <View pointerEvents="none" style={styles.cameraHintChip}>
                        <Text style={styles.cameraHintText}>{cameraHint}</Text>
                      </View>
                    )}
                  </View>
                </View>
              )}
            </View>
          </View>

          {/* RIGHT */}
          <View style={[styles.panel, { width: rightW }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 14 }}>
              {jogLocked && (
                <Text style={styles.note}>
                  {phase !== "idle" ? "Pick/Place active → Jog locked." : "Jog locked."}
                </Text>
              )}

              {/* XY */}
              <View style={styles.jogTopRow}>
                <HoldBtn label="Z+" disabled={!canJog} onHoldStart={() => startHoldJog("z", +1)} onHoldEnd={endHoldJog} />
                <HoldBtn label="X+" disabled={!canJog} onHoldStart={() => startHoldJog("x", +1)} onHoldEnd={endHoldJog} />
                <HoldBtn label="Z-" disabled={!canJog} onHoldStart={() => startHoldJog("z", -1)} onHoldEnd={endHoldJog} />
              </View>

              <View style={styles.jogMidRow}>
                <HoldBtn label="Y+" disabled={!canJog} onHoldStart={() => startHoldJog("y", +1)} onHoldEnd={endHoldJog} />
                <View style={styles.jogCenter}>
                  <IconBtn icon="eye-outline" label="" tone="primary" onPress={onVisionPose} disabled={!gatewayBase || !robotConnected} size="sm" />
                </View>
                <HoldBtn label="Y-" disabled={!canJog} onHoldStart={() => startHoldJog("y", -1)} onHoldEnd={endHoldJog} />
              </View>

              <View style={styles.jogBottomRow}>
                <HoldBtn label="Grip-" disabled={!canGrip || !gripperAvailable} onPress={() => sendGripper("close")} />
                <HoldBtn label="X-" disabled={!canJog} onHoldStart={() => startHoldJog("x", -1)} onHoldEnd={endHoldJog} />
                <HoldBtn label="Grip+" disabled={!canGrip || !gripperAvailable} onPress={() => sendGripper("open")} />
              </View>

              {/* Pick/Place */}
              <View style={{ marginTop: 14, gap: 10 }}>
                <View style={{ flexDirection: "row", justifyContent: "space-evenly", marginTop: 14 }}>
                  <IconBtn
                    icon="cube-scan"
                    label={phase === "pick_armed" ? "Pick ✓" : "Pick"}
                    onPress={onArmPick}
                    disabled={!aiModeActive || !cameraStarted || !robotConnected || !enabled || aiLoading || selecting || phase === "busy" || phase === "pick_hold" || phase === "place_hold"}
                    tone={phase === "pick_armed" ? "success" : "primary"}
                    size="md"
                  />
                  <IconBtn
                    icon="cube-send"
                    label={phase === "place_armed" ? "Place ✓" : "Place"}
                    onPress={onArmPlace}
                    disabled={!hasPicked || !aiModeActive || !cameraStarted || !robotConnected || !enabled || aiLoading || selecting || phase === "busy" || phase === "pick_hold" || phase === "place_hold"}
                    tone={phase === "place_armed" ? "success" : "primary"}
                    size="md"
                  />
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-evenly" }}>
                  <IconBtn icon="crosshairs-gps" label="Clear Sel" onPress={onClearSelection} disabled={!selection || phase === "busy"} size="md" tone="danger" />
                  <IconBtn icon="close-circle-outline" label="Cancel" onPress={onCancelWorkflow} disabled={phase === "busy"} size="md" tone="ghost" />
                </View>

                {phase === "pick_hold" && <Text style={styles.note}>Robot is at pick pose. Use Gripper to confirm Pick.</Text>}
                {phase === "place_hold" && <Text style={styles.note}>Robot is at place pose. Use Gripper to confirm Place.</Text>}
                {hasPicked && phase === "idle" && <Text style={styles.note}>Pick confirmed. Press Place, then tap camera.</Text>}
              </View>
            </ScrollView>
          </View>
        </View>

        {/* LEFT DRAWER: Status */}
        {leftOpen && (
          <View style={styles.drawerOverlay}>
            <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setLeftOpen(false)} />
            <View style={[styles.drawer, { left: 12, width: clamp(Math.round(width * 0.4), 320, 560) }]}>
              <View style={styles.drawerHeader}>
                <Text style={styles.drawerTitle}>Status</Text>
                <IconBtn icon="close" onPress={() => setLeftOpen(false)} />
              </View>

              <ScrollView contentContainerStyle={{ paddingBottom: 12 }} showsVerticalScrollIndicator={false}>
                <View style={styles.drawerCard}>
                  <View style={styles.drawerRow}>
                    <View style={[styles.dot, dotStyle(gwDot)]} />
                    <Text style={styles.drawerRowTitle}>Gateway</Text>
                  </View>
                  <Text style={styles.drawerRowSub} numberOfLines={2}>
                    {gateway.value || "-"}
                  </Text>
                  <Text style={styles.drawerHint}>WS: {wsConnected ? "Connected" : "Disconnected"}</Text>
                </View>

                <View style={styles.drawerCard}>
                  <View style={styles.drawerRow}>
                    <View style={[styles.dot, dotStyle(rbDot)]} />
                    <Text style={styles.drawerRowTitle}>Robot</Text>
                  </View>
                  <Text style={styles.drawerRowSub} numberOfLines={2}>
                    {robot.value || "-"}
                  </Text>
                  <Text style={styles.drawerHint}>{robotConnected ? "Connected (via gateway)." : "Not connected."}</Text>
                </View>

                <View style={styles.drawerCard}>
                  <View style={styles.drawerRow}>
                    <View style={[styles.dot, dotStyle(camDot)]} />
                    <Text style={styles.drawerRowTitle}>Camera</Text>
                  </View>
                  <Text style={styles.drawerRowSub}>{cameraStarted ? "Started" : "Stopped"}</Text>
                  {!!cameraLastError && <Text style={styles.drawerHint}>{cameraLastError}</Text>}
                </View>

                <View style={styles.drawerCard}>
                  <View style={styles.drawerRow}>
                    <View style={[styles.dot, dotStyle(aiDot)]} />
                    <Text style={styles.drawerRowTitle}>AI Server</Text>
                  </View>
                  <Text style={styles.drawerRowSub} numberOfLines={2}>
                    {aiserver.value || "-"}
                  </Text>
                  {!!aiserver.message && <Text style={styles.drawerHint}>{aiserver.message}</Text>}
                </View>

                <Pressable
                  onPress={() => {
                    setLeftOpen(false);
                    goModeSelect();
                  }}
                  style={({ pressed }) => [styles.hubLink, pressed ? { opacity: 0.9 } : null]}
                >
                  <MaterialCommunityIcons name="link-variant" size={18} color="white" />
                  <Text style={styles.hubLinkText}>Go to Mode Selection</Text>
                </Pressable>

                <Pressable
                  onPress={() => {
                    setLeftOpen(false);
                    goConnectionHub();
                  }}
                  style={({ pressed }) => [styles.hubLink, pressed ? { opacity: 0.9 } : null]}
                >
                  <MaterialCommunityIcons name="link-variant" size={18} color="white" />
                  <Text style={styles.hubLinkText}>Go to Connection Hub</Text>
                </Pressable>

                <Text style={styles.drawerTiny}>Note: Use Connection Hub to change addresses / IPs.</Text>
              </ScrollView>
            </View>
          </View>
        )}

        {/* RIGHT DRAWER: Settings */}
        {rightOpen && (
          <View style={styles.drawerOverlay}>
            <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setRightOpen(false)} />
            <View style={[styles.drawer, { right: 12, width: clamp(Math.round(width * 0.4), 320, 560) }]}>
              <View style={styles.drawerHeader}>
                <Text style={styles.drawerTitle}>Settings</Text>
                <IconBtn icon="close" onPress={() => setRightOpen(false)} />
              </View>

              <ScrollView contentContainerStyle={{ paddingBottom: 12 }} showsVerticalScrollIndicator={false}>
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Telemetry</Text>
                  <View style={styles.telemetryGrid}>
                    <TelemetryRow k="Robot connected" v={robotConnected ? "Yes" : "No"} />
                    <TelemetryRow k="Enabled" v={enabled ? "Yes" : "No"} />
                    <TelemetryRow k="Workflow" v={`${phase}${hasPicked ? " (picked)" : ""}`} />
                    <TelemetryRow k="Selection" v={selection ? `u=${selection.u.toFixed(2)} v=${selection.v.toFixed(2)}` : "-"} />
                  </View>

                  {!!(live as any)?.safety?.message && <Text style={styles.drawerHint}>Safety message: {String((live as any).safety.message)}</Text>}
                  {!!(live as any)?.camera?.last_error && <Text style={styles.drawerHint}>Camera error: {String((live as any).camera.last_error)}</Text>}
                </View>

                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>AI</Text>
                  <Text style={styles.drawerHint}>
                    Mode: {aiModeActive ? "running" : "idle"} • Exit: {aiModeExitCode == null ? "-" : String(aiModeExitCode)}
                  </Text>
                  {!!aiModeErr && <Text style={styles.warnText}>AI error: {aiModeErr}</Text>}
                  <View style={styles.drawerBtnRow}>
                    <IconBtn icon="refresh" label="Refresh" onPress={refreshAiStatus} disabled={!aiBase} />
                    <IconBtn icon={aiBtnIcon as any} label={aiBtnLabel} tone={aiBtnTone as any} onPress={onAiToggle} disabled={!canAiToggle} />
                    <IconBtn icon="close-circle-outline" label="Cancel" tone="danger" onPress={onCancelWorkflow} />
                  </View>
                </View>

                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Robot</Text>
                  <View style={styles.drawerBtnRow}>
                    <IconBtn
                      icon="power"
                      label={enabled ? "Disable" : "Enable"}
                      tone={enabled ? "ghost" : "primary"}
                      onPress={onEnableDisable}
                      disabled={!gatewayBase || !robotConnected || phase !== "idle"}
                    />
                    <IconBtn icon="shield-alert-outline" label="Clear Safety" onPress={onClearSafety} disabled={!gatewayBase || phase !== "idle"} />
                    <IconBtn icon="home" label="Home" onPress={onRobotHome} disabled={!gatewayBase || !robotConnected || phase !== "idle"} />
                    <IconBtn icon="eye-outline" label="Vision" onPress={onVisionPose} disabled={!gatewayBase || !robotConnected} />
                  </View>

                  {(safetyLimit || safetyMsg) && (
                    <Text style={styles.warnText}>
                      {safetyLimit ? "Safety: limit hit." : "Safety warning."} {safetyMsg ? `(${safetyMsg})` : ""}
                    </Text>
                  )}
                </View>

                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Camera</Text>
                  <Text style={styles.drawerHint}>
                    Status: {cameraLoading ? "Loading…" : cameraStarted ? "Started" : "Stopped"}
                    {cameraLastError ? ` • ${cameraLastError}` : ""}
                  </Text>

                  <View style={styles.drawerBtnRow}>
                    <IconBtn icon="play" label="Start" tone="primary" onPress={onCameraStart} disabled={!gatewayBase || cameraLoading || phase !== "idle"} />
                    <IconBtn icon="stop" label="Stop" tone="danger" onPress={onCameraStop} disabled={!gatewayBase || cameraLoading || phase !== "idle"} />
                    <IconBtn icon="refresh" label="Status" onPress={refreshCameraStatus} disabled={!gatewayBase || cameraLoading} />
                  </View>
                </View>
              </ScrollView>
            </View>
          </View>
        )}
      </SafeAreaView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(8, 12, 22, 0.74)" },
  safe: { flex: 1 },

  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingTop: 0,
    gap: 10,
  },

  topLeft: { alignItems: "flex-start" },
  topRight: { alignItems: "flex-end" },

  topMid: {
    flex: 1,
    height: 46,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },

  topGroup: {
    flexDirection: "row",
    gap: 8,
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    minWidth: 0,
  },

  topDots: {
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
    paddingHorizontal: 8,
    flexShrink: 0,
  },

  iconBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    justifyContent: "center",
    paddingHorizontal: 12,
    borderRadius: 999,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
  },

  iconBtnText: {
    color: "white",
    fontWeight: "900",
    fontSize: 12,
  },

  iconBtnSm: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    gap: 6,
  },

  iconBtnTextSm: {
    fontSize: 11,
  },

  dotLine: { flexDirection: "row", alignItems: "center", gap: 6 },
  dotText: { color: "rgba(255,255,255,0.75)", fontWeight: "900", fontSize: 11 },

  iconBtnPrimary: {
    backgroundColor: "rgba(37, 99, 235, 0.92)",
    borderColor: "rgba(37, 99, 235, 0.65)",
  },
  iconBtnDanger: {
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderColor: "rgba(239, 68, 68, 0.65)",
  },

  iconBtnSuccess: {
    backgroundColor: "rgba(40, 183, 99, 0.92)",
    borderColor: "rgba(40, 183, 99, 0.65)",
  },

  body: {
    flex: 1,
    flexDirection: "row",
    paddingHorizontal: 12,
    paddingTop: 10,
    paddingBottom: 12,
    gap: 10,
  },

  panel: {
    borderRadius: 22,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.62)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },

  center: { flex: 1, gap: 10 },

  cameraCard: {
    flex: 1,
    borderRadius: 22,
    backgroundColor: "rgba(18, 27, 47, 0.62)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
  },
  cameraPlaceholder: { flex: 1, alignItems: "center", justifyContent: "center" },
  cameraPlaceholderText: { color: "rgba(255,255,255,0.55)", fontWeight: "900" },

  cameraBox: { flex: 1, padding: 6, justifyContent: "center", alignItems: "center" },
  cameraAspect: {
    aspectRatio: 4 / 3,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.22)",
    alignSelf: "center",
    width: "100%",
  },

  web: { flex: 1, backgroundColor: "transparent" },
  webLoading: { flex: 1, alignItems: "center", justifyContent: "center", gap: 10 },
  webLoadingText: { color: "rgba(255,255,255,0.55)", fontWeight: "800", fontSize: 12 },

  tapOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "transparent",
  },
  marker: {
    position: "absolute",
    width: 20,
    height: 20,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: "rgba(37, 99, 235, 0.95)",
    backgroundColor: "rgba(37, 99, 235, 0.15)",
  },
  cameraHintChip: {
    position: "absolute",
    left: 10,
    top: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "rgba(0,0,0,0.35)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  cameraHintText: {
    color: "rgba(255,255,255,0.85)",
    fontWeight: "900",
    fontSize: 11,
  },

  stopBtn: {
    height: 80,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.60)",
  },
  stopText: { color: "white", fontWeight: "900", fontSize: 18, letterSpacing: 1 },

  holdBtn: {
    minWidth: 10,
    maxWidth: 90,
    borderRadius: 16,
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  holdBtnText: { color: "white", fontWeight: "900", fontSize: 13 },

  jogTopRow: { flexDirection: "row", gap: 10, justifyContent: "center", alignItems: "center" },
  jogMidRow: { flexDirection: "row", gap: 10, alignItems: "center", justifyContent: "center", marginVertical: 10 },
  jogBottomRow: { flexDirection: "row", gap: 10, alignItems: "center", justifyContent: "center" },

  jogCenter: {
    width: 52,
    height: 42,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },

  note: { marginTop: 14, color: "rgba(255,255,255,0.50)", fontWeight: "800", fontSize: 11, textAlign: "center" },

  dot: { width: 10, height: 10, borderRadius: 99 },
  dotGreen: { backgroundColor: "rgba(34, 197, 94, 1)" },
  dotBlue: { backgroundColor: "rgba(59, 130, 246, 1)" },
  dotRed: { backgroundColor: "rgba(239, 68, 68, 1)" },
  dotGray: { backgroundColor: "rgba(148, 163, 184, 1)" },

  barCard: {
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  barCardTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "baseline" },
  barCardTitle: { color: "rgba(255,255,255,0.90)", fontWeight: "900", fontSize: 12 },
  barCardSub: { color: "rgba(255,255,255,0.55)", fontWeight: "900", fontSize: 11 },

  barTrack: {
    marginTop: 10,
    height: 10,
    borderRadius: 99,
    backgroundColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  barTrackInactive: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderColor: "rgba(255,255,255,0.06)",
  },
  barFill: { height: "100%", backgroundColor: "rgba(37, 99, 235, 0.95)" },
  barFillInactive: { backgroundColor: "rgba(148, 163, 184, 0.35)" },

  speedBtnsRow: { flexDirection: "row", gap: 10, justifyContent: "space-evenly", margin: 10 },

  drawerOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.55)", zIndex: 2000 },
  drawer: {
    position: "absolute",
    top: 60,
    bottom: 16,
    borderRadius: 22,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  drawerHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingBottom: 10 },
  drawerTitle: { color: "white", fontWeight: "900", fontSize: 16 },

  drawerCard: {
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    marginBottom: 10,
  },
  drawerRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  drawerRowTitle: { color: "rgba(255,255,255,0.90)", fontWeight: "900", fontSize: 13 },
  drawerRowSub: { color: "rgba(255,255,255,0.70)", fontWeight: "800", marginTop: 8, fontSize: 12 },
  drawerHint: { color: "rgba(255,255,255,0.55)", fontWeight: "800", marginTop: 6, fontSize: 11 },

  drawerSectionTitle: { color: "rgba(255,255,255,0.92)", fontWeight: "900", fontSize: 13, marginBottom: 6 },
  drawerBtnRow: { flexDirection: "row", gap: 10, marginTop: 10, flexWrap: "wrap" },

  drawerTiny: { color: "rgba(255,255,255,0.55)", fontWeight: "800", marginTop: 8, fontSize: 11 },

  warnText: { marginTop: 10, color: "rgba(239,68,68,0.92)", fontWeight: "900", fontSize: 11 },

  hubLink: {
    marginTop: 6,
    borderRadius: 16,
    paddingVertical: 12,
    paddingHorizontal: 12,
    backgroundColor: "rgba(37, 99, 235, 0.22)",
    borderWidth: 1,
    borderColor: "rgba(37, 99, 235, 0.65)",
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  hubLinkText: { color: "white", fontWeight: "900" },

  telemetryGrid: { marginTop: 6, gap: 6 },
  telemetryRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  telemetryKey: { color: "rgba(255,255,255,0.70)", fontWeight: "900", fontSize: 11, flex: 1 },
  telemetryVal: { color: "rgba(255,255,255,0.92)", fontWeight: "900", fontSize: 11, maxWidth: "55%" },

  rotateWrap: { flex: 1, backgroundColor: "#0B1220", alignItems: "center", justifyContent: "center", padding: 22 },
  rotateTitle: { color: "white", fontSize: 20, fontWeight: "900" },
  rotateSub: { color: "rgba(255,255,255,0.65)", marginTop: 10, textAlign: "center", maxWidth: 420 },
});