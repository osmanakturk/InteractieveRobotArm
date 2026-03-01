// src/screens/ManualControlScreen.tsx
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
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";
import { useFocusEffect } from "@react-navigation/native";
import MaterialCommunityIcons from "@expo/vector-icons/MaterialCommunityIcons";
import { useConnection } from "../connection/ConnectionContext";

// --- Types ---
type JogFrame = "Base" | "Tool";
type ControlMode = "XYZ" | "RXYZ";
type DotStatus = "idle" | "connecting" | "connected" | "error";

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return `http://${raw}`;
  return raw;
}

async function safeJson(resp: Response) {
  try {
    return await resp.json();
  } catch {
    return {};
  }
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
        tone === "primary" ? styles.iconBtnPrimary : tone === "danger" ? styles.iconBtnDanger : tone === "success" ? styles.iconBtnSuccess : null,
        disabled ? { opacity: 0.45 } : null,
        pressed && !disabled ? { opacity: 0.9 } : null,
      ]}
    >
      <MaterialCommunityIcons name={icon} size={size === "sm" ? 16 : 18} color="white" />
      {!!label && <Text style={[styles.iconBtnText, size === "sm" ? styles.iconBtnTextSm : null]}>{label}</Text>}
    </Pressable>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <View style={styles.row}>{children}</View>;
}

function PillToggle({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.pillToggle,
        active ? styles.pillToggleActive : null,
        pressed ? { opacity: 0.92 } : null,
      ]}
    >
      <Text style={styles.pillToggleText}>{label}</Text>
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

export default function ManualControlScreen({ navigation, route }: any) {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const { gateway, robot, aiserver, live, wsConnected } = useConnection();

  const gatewayFromParams: string = route?.params?.gateway || route?.params?.baseUrl || "";
  const gatewayBase = useMemo(
    () => normalizeBaseUrl(gateway.value || gatewayFromParams),
    [gateway.value, gatewayFromParams]
  );
  const [streamNonce, setStreamNonce] = useState(0);
  // MJPEG stream
  //const realsenseMjpegUrl = useMemo(() => (gatewayBase ? `${gatewayBase}/api/cameras/realsense/stream/mjpeg` : ""), [gatewayBase]);
  const realsenseMjpegUrl = useMemo(() => {
    if (!gatewayBase) return "";
    return `${gatewayBase}/api/cameras/realsense/stream/mjpeg?t=${streamNonce}`;
  }, [gatewayBase, streamNonce]);
  // UI state
  const [frame, setFrame] = useState<JogFrame>("Tool");
  const [mode, setMode] = useState<ControlMode>("XYZ");

  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  const [speedPct, setSpeedPct] = useState<number>(35);

  const [gripperPct, setGripperPct] = useState<number>(0);
  const [gripperLoading, setGripperLoading] = useState(false);
  const [gripperAvailable, setGripperAvailable] = useState<boolean>(true);

  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraStarted, setCameraStarted] = useState<boolean>(false);
  const [cameraLastError, setCameraLastError] = useState<string>("");

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

  const rbDot: DotStatus =
    !gatewayBase
      ? "idle"
      : !gatewayUp
        ? "error"
        : robotConnected
          ? "connected"
          : "error";

  const camDot: DotStatus =
    !gatewayBase
      ? "idle"
      : !gatewayUp
        ? "error"
        : cameraStarted
          ? "connected"
          : "idle";

  const aiConnected = !!(live as any)?.ai_server?.connected;
  const aiConfigured = !!(live as any)?.ai_server?.configured;

  const aiDot: DotStatus =
    !gatewayBase
      ? "idle"
      : !gatewayUp
        ? "error"
        : aiConnected
          ? "connected"
          : aiConfigured
            ? "error"
            : "idle";

  // API helpers
  const apiPost = useCallback(
    async (path: string, body?: any) => {
      if (!gatewayBase) throw new Error("Gateway not set.");
      const resp = await fetch(`${gatewayBase}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : JSON.stringify({}),
      });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) {
        throw new Error(data?.message || `HTTP ${resp.status}`);
      }
      return data;
    },
    [gatewayBase]
  );

  const apiGet = useCallback(
    async (path: string) => {
      if (!gatewayBase) throw new Error("Gateway not set.");
      const resp = await fetch(`${gatewayBase}${path}`, { method: "GET" });
      const data = await safeJson(resp);
      if (!resp.ok || data?.ok === false) {
        throw new Error(data?.message || `HTTP ${resp.status}`);
      }
      return data;
    },
    [gatewayBase]
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

  // WS camera status sync (live payload)
  const wsCamStarted = (live as any)?.camera?.started;
  const wsCamErr = (live as any)?.camera?.last_error;

  useEffect(() => {
    if (typeof wsCamStarted === "boolean") setCameraStarted(wsCamStarted);
    if (typeof wsCamErr === "string") setCameraLastError(wsCamErr);
  }, [wsCamStarted, wsCamErr]);



  // IMPORTANT: no `live` dependency here (prevents refresh loop)
  const refreshGripperStatus = useCallback(async () => {
    if (!gatewayBase) return;
    setGripperLoading(true);
    try {
      const data = await apiGet("/api/robot/gripper/status");
      const g = data?.gripper || data || {};

      // Detect availability (adapt this to your backend payload if you have a better flag)
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
      // if the endpoint fails, treat it as unavailable (better UX than “0/100”)
      setGripperAvailable(false);
      setGripperPct(0);
    } finally {
      setGripperLoading(false);
    }
  }, [apiGet, gatewayBase]);

  // Focus refresh (no polling)
  useFocusEffect(
    useCallback(() => {
      refreshCameraStatus();
      refreshGripperStatus();
    }, [refreshCameraStatus, refreshGripperStatus])
  );

  // WS gripper_pct: update only when it changes
  const wsGripperPct = (live as any)?.status?.gripper_pct;
  useEffect(() => {
    if (typeof wsGripperPct !== "number") return;
    setGripperAvailable(true);
    setGripperPct(clamp(Math.round(wsGripperPct), 0, 100));
  }, [wsGripperPct]);

  // Actions
  const pushFrame = useCallback(
    async (f: JogFrame) => {
      setFrame(f);
      try {
        await apiPost("/api/robot/frame", { frame: f.toLowerCase() });
      } catch {
        // silent
      }
    },
    [apiPost]
  );

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

  const sendGripper = useCallback(
    async (action: "open" | "close") => {
      try {
        await apiPost("/api/robot/gripper", { action });
        await refreshGripperStatus(); // refresh only after an action
      } catch (e: any) {
        Alert.alert("Gripper failed", e?.message || "Gripper action failed.");
      }
    },
    [apiPost, refreshGripperStatus]
  );

  const onStop = useCallback(async () => {
    try {
      await apiPost("/api/robot/stop");
    } catch (e: any) {
      Alert.alert("STOP failed", e?.message || "Stop failed.");
    }
  }, [apiPost]);

  // Settings actions
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
      // Change this endpoint if your backend uses a different one.
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
      setStreamNonce((n) => n + 1); // force reconnect
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
      setCameraStarted(false); // UX: immediately show placeholder
      setStreamNonce((n) => n + 1); // drop old MJPEG connection
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

  const goConnectionHub = useCallback(() => {
    navigation.navigate("ConnectionHub");
  }, [navigation]);

    const goModeSelect = useCallback(() => {
    navigation.navigate("ModeSelect");
  }, [navigation]);


  // Layout
  const leftW = useMemo(() => clamp(Math.round(width * 0.28), 240, 310), [width]);
  const rightW = useMemo(() => clamp(Math.round(width * 0.28), 260, 360), [width]);

  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>Manual control is designed for landscape.</Text>
      </View>
    );
  }

  const canControl = !!gatewayBase && robotConnected;

  // Telemetry fields
  const st = (live as any)?.status || {};
  const ai = (live as any)?.ai_server || {};
  const cam = (live as any)?.camera || {};
  const safety = (live as any)?.safety || {};

  const gripperRightText = gripperLoading
    ? "Loading…"
    : !gripperAvailable
      ? "N/A"
      : `${gripperPct}/100`;

  // --- TopBar derived states (just above return is fine)
  const canRobotToggle = !!gatewayBase && robotConnected;
  const canCameraToggle = !!gatewayBase && !cameraLoading;

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


  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />
      <SafeAreaView style={styles.safe}>
        {/* Top bar (clean single row) */}


        <View style={styles.topBar}>
          <View style={styles.topLeft}>
            <IconBtn icon="menu" label="Status" onPress={() => setLeftOpen(true)} size="md" />
          </View>

          <View style={styles.topMid}>
            {/* Left group: Frame + Camera toggle */}
            <View style={styles.topGroup}>
              <PillToggle label="Base" active={frame === "Base"} onPress={() => pushFrame("Base")} />
              <PillToggle label="Tool" active={frame === "Tool"} onPress={() => pushFrame("Tool")} />

              <IconBtn
                icon={camBtnIcon as any}
                label={camBtnLabel}
                tone={camBtnTone}
                onPress={onCameraToggle}
                disabled={!canCameraToggle}
                size="sm"
              />
            </View>

            {/* Dots */}
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

            {/* Right group: Robot toggle + Mode */}
            <View style={styles.topGroup}>
              <IconBtn
                icon={robotBtnIcon as any}
                label={robotBtnLabel}
                tone={robotBtnTone}
                onPress={onEnableDisable}
                disabled={!canRobotToggle}
                size="sm"
              />

              <PillToggle label="XYZ" active={mode === "XYZ"} onPress={() => setMode("XYZ")} />
              <PillToggle label="RXYZ" active={mode === "RXYZ"} onPress={() => setMode("RXYZ")} />
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
              <BarCard title="Gripper" rightText={gripperRightText} valuePct={gripperAvailable ? gripperPct : 0} inactive={!gripperAvailable} />

              <View style={{ marginTop: 12, alignItems: "center" }}>
                <HoldBtn
                  label={mode === "XYZ" ? "Z+" : "RZ+"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "z" : "rz", +1)}
                  onHoldEnd={endHoldJog}
                />

                <View style={styles.jogMidRow}>
                  <HoldBtn label="Grip-" disabled={!canControl || !gripperAvailable} onPress={() => sendGripper("close")} />
                  <View style={styles.crossCenter}>
                    <Text style={styles.crossCenterText}>{frame}</Text>
                  </View>
                  <HoldBtn label="Grip+" disabled={!canControl || !gripperAvailable} onPress={() => sendGripper("open")} />
                </View>

                <HoldBtn
                  label={mode === "XYZ" ? "Z-" : "RZ-"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "z" : "rz", -1)}
                  onHoldEnd={endHoldJog}
                />
              </View>

              {!gatewayBase ? (
                <Text style={styles.note}>Gateway is not set. Open Connection Hub.</Text>
              ) : !robotConnected ? (
                <Text style={styles.note}>Robot is not connected.</Text>
              ) : (
                <Text style={styles.note}>Ready.</Text>
              )}
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
                  <View style={styles.cameraAspect}>
                    <WebView
                      key={`cam-${streamNonce}`}  // extra hard remount
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
                  </View>
                </View>
              )}
            </View>

            <Pressable onPress={onStop} style={({ pressed }) => [styles.stopBtn, pressed ? { opacity: 0.92 } : null]}>
              <Text style={styles.stopText}>STOP</Text>
            </Pressable>
          </View>

          {/* RIGHT */}
          <View style={[styles.panel, { width: rightW }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 14 }}>
              <BarCard title="Speed" rightText={`${speedPct}%`} valuePct={speedPct} />

              <View style={styles.speedBtnsRow}>
                <IconBtn icon="minus" label="Speed" onPress={() => pushSpeed(speedPct - 5)} disabled={!gatewayBase} />
                <IconBtn icon="plus" label="Speed" onPress={() => pushSpeed(speedPct + 5)} disabled={!gatewayBase} />
              </View>

              <View style={styles.jogTopRow}>
                <HoldBtn
                  label={mode === "XYZ" ? "X+" : "RX+"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", +1)}
                  onHoldEnd={endHoldJog}
                />
              </View>

              <View style={styles.jogMidRow}>
                <HoldBtn
                  label={mode === "XYZ" ? "Y+" : "RY+"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", +1)}
                  onHoldEnd={endHoldJog}
                />

                <View style={styles.jogCenter}>
                  <Text style={styles.jogCenterText}>{mode}</Text>
                </View>

                <HoldBtn
                  label={mode === "XYZ" ? "Y-" : "RY-"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", -1)}
                  onHoldEnd={endHoldJog}
                />
              </View>

              <View style={styles.jogBottomRow}>
                <HoldBtn
                  label={mode === "XYZ" ? "X-" : "RX-"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", -1)}
                  onHoldEnd={endHoldJog}
                />
              </View>
            </ScrollView>
          </View>
        </View>

        {/* LEFT DRAWER: Status */}
        {leftOpen && (
          <View style={styles.drawerOverlay}>
            <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setLeftOpen(false)} />
            <View style={[styles.drawer, { left: 12, width: clamp(Math.round(width * 0.40), 320, 560) }]}>
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

                <Text style={styles.drawerTiny}>
                  Note: Use Connection Hub to change addresses / IPs.
                </Text>
              </ScrollView>
            </View>
          </View>
        )}

        {/* RIGHT DRAWER: Settings */}
        {rightOpen && (
          <View style={styles.drawerOverlay}>
            <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setRightOpen(false)} />
            <View style={[styles.drawer, { right: 12, width: clamp(Math.round(width * 0.40), 320, 560) }]}>
              <View style={styles.drawerHeader}>
                <Text style={styles.drawerTitle}>Settings</Text>
                <IconBtn icon="close" onPress={() => setRightOpen(false)} />
              </View>

              <ScrollView contentContainerStyle={{ paddingBottom: 12 }} showsVerticalScrollIndicator={false}>
                {/* Telemetry */}
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Telemetry</Text>
                  <View style={styles.telemetryGrid}>
                    <TelemetryRow k="Robot connected" v={robotConnected ? "Yes" : "No"} />
                    <TelemetryRow k="Enabled" v={enabled ? "Yes" : "No"} />
                    <TelemetryRow k="State" v={typeof st.state === "number" ? String(st.state) : "-"} />
                    <TelemetryRow k="Error code" v={typeof st.error_code === "number" ? String(st.error_code) : "-"} />
                    <TelemetryRow k="Robot IP" v={st.ip ? String(st.ip) : "-"} />
                    <TelemetryRow k="AI connected" v={ai.connected ? "Yes" : "No"} />
                    <TelemetryRow k="AI latency" v={typeof ai.latency_ms === "number" ? `${ai.latency_ms} ms` : "-"} />
                    <TelemetryRow k="Camera started" v={cameraStarted ? "Yes" : "No"} />
                    <TelemetryRow k="Safety" v={safetyLimit ? "Limit hit" : safety?.message ? "Warning" : "OK"} />
                  </View>

                  {!!safety?.message && <Text style={styles.drawerHint}>Safety message: {String(safety.message)}</Text>}
                  {!!cam?.last_error && <Text style={styles.drawerHint}>Camera error: {String(cam.last_error)}</Text>}
                </View>

                {/* Robot */}
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Robot</Text>
                  <View style={styles.drawerBtnRow}>
                    <IconBtn
                      icon="power"
                      label={enabled ? "Disable" : "Enable"}
                      tone={enabled ? "ghost" : "primary"}
                      onPress={onEnableDisable}
                      disabled={!gatewayBase || !robotConnected}
                    />
                    <IconBtn
                      icon="shield-alert-outline"
                      label="Clear Safety"
                      onPress={onClearSafety}
                      disabled={!gatewayBase}
                    />
                    <IconBtn
                      icon="home"
                      label="Home"
                      onPress={onRobotHome}
                      disabled={!gatewayBase || !robotConnected}
                    />
                  </View>

                  {(safetyLimit || safetyMsg) && (
                    <Text style={styles.warnText}>
                      {safetyLimit ? "Safety: limit hit." : "Safety warning."} {safetyMsg ? `(${safetyMsg})` : ""}
                    </Text>
                  )}
                </View>

                {/* Camera */}
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Camera</Text>
                  <Text style={styles.drawerHint}>
                    Status: {cameraLoading ? "Loading…" : cameraStarted ? "Started" : "Stopped"}
                    {cameraLastError ? ` • ${cameraLastError}` : ""}
                  </Text>

                  <View style={styles.drawerBtnRow}>
                    <IconBtn
                      icon="play"
                      label="Start"
                      tone="primary"
                      onPress={onCameraStart}
                      disabled={!gatewayBase || cameraLoading}
                    />
                    <IconBtn
                      icon="stop"
                      label="Stop"
                      tone="danger"
                      onPress={onCameraStop}
                      disabled={!gatewayBase || cameraLoading}
                    />
                    <IconBtn
                      icon="refresh"
                      label="Status"
                      onPress={refreshCameraStatus}
                      disabled={!gatewayBase || cameraLoading}
                    />
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

  // Top bar
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
    justifyContent: "space-between",
    gap: 8,
  },

  topGroup: {
    flexDirection: "row",
    gap: 8,
    alignItems: "center",
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
    //width: "90%",
    aspectRatio: 4 / 3,
    //borderRadius: 18,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.22)",
    alignSelf: "center",
  },
  web: { flex: 1, backgroundColor: "transparent" },
  webLoading: { flex: 1, alignItems: "center", justifyContent: "center", gap: 10 },
  webLoadingText: { color: "rgba(255,255,255,0.55)", fontWeight: "800", fontSize: 12 },

  stopBtn: {
    height: 46,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.60)",
  },
  stopText: { color: "white", fontWeight: "900", fontSize: 18, letterSpacing: 1 },

  row: { flexDirection: "row", gap: 10, alignItems: "center" },

  pillToggle: {
    minWidth: 64,
    borderRadius: 16,
    paddingVertical: 6,
    paddingHorizontal: 8,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.16)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  pillToggleActive: {
    backgroundColor: "rgba(37, 99, 235, 0.22)",
    borderColor: "rgba(37, 99, 235, 0.65)",
  },
  pillToggleText: { color: "white", fontWeight: "900", fontSize: 12 },

  holdBtn: {
    minWidth: 10,
    maxWidth: 90,
    borderRadius: 16,
    paddingVertical: 10,
    paddingHorizontal: 20,
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

  crossCenter: {
    width: 52,
    height: 42,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  crossCenterText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 12 },

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
  jogCenterText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 12 },

  note: { marginTop: 14, color: "rgba(255,255,255,0.50)", fontWeight: "800", fontSize: 11, textAlign: "center" },

  // dots
  dot: { width: 10, height: 10, borderRadius: 99 },
  dotGreen: { backgroundColor: "rgba(34, 197, 94, 1)" },
  dotBlue: { backgroundColor: "rgba(59, 130, 246, 1)" },
  dotRed: { backgroundColor: "rgba(239, 68, 68, 1)" },
  dotGray: { backgroundColor: "rgba(148, 163, 184, 1)" },

  // Bar cards
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

  speedBtnsRow: { flexDirection: "row", gap: 10, justifyContent: "center", margin: 10 },

  // Drawers
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

  // Telemetry
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
