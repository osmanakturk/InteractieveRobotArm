import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ImageBackground,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { SafeAreaView } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";

type JogFrame = "Base" | "Tool";
type ControlMode = "XYZ" | "RXYZ";

const STORAGE_KEYS = {
  gateway: "@conn/gateway",
  robot: "@conn/robot",
  aiserver: "@conn/aiserver",
} as const;

type DotStatus = "idle" | "connecting" | "connected" | "error";

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return `http://${raw}`;
  return raw;
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function pctToBarWidth(pct: number) {
  const v = clamp(pct, 0, 100);
  return `${v}%`;
}

function dotStyleFor(status: DotStatus, styles: any) {
  return status === "connected"
    ? styles.dotGreen
    : status === "connecting"
    ? styles.dotBlue
    : status === "error"
    ? styles.dotRed
    : styles.dotGray;
}

// --- Small, non-layout-shifting dropdown (modal) ---
function Dropdown({
  label,
  value,
  options,
  onChange,
}: {
  label: string; // e.g. "Jog Frame"
  value: string; // e.g. "Tool"
  options: { key: string; title: string; subtitle?: string }[];
  onChange: (key: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Pressable
        onPress={() => setOpen(true)}
        style={({ pressed }) => [styles.dd, pressed ? { opacity: 0.92 } : null]}
      >
        <Text style={styles.ddText} numberOfLines={1}>
          {label}: {value}
        </Text>
        <Text style={styles.ddChevron}>{open ? "▲" : "▼"}</Text>
      </Pressable>

      {open && (
        <View style={styles.modalOverlay}>
          <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setOpen(false)} />
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>{label}</Text>

            <View style={{ gap: 10 }}>
              {options.map((o) => (
                <Pressable
                  key={o.key}
                  onPress={() => {
                    onChange(o.key);
                    setOpen(false);
                  }}
                  style={({ pressed }) => [
                    styles.modalItem,
                    o.key === value ? styles.modalItemActive : null,
                    pressed ? { opacity: 0.92 } : null,
                  ]}
                >
                  <Text style={styles.modalItemTitle}>{o.title}</Text>
                  {!!o.subtitle && <Text style={styles.modalItemSub}>{o.subtitle}</Text>}
                </Pressable>
              ))}
            </View>

            <Pressable onPress={() => setOpen(false)} style={styles.modalClose}>
              <Text style={styles.modalCloseText}>Close</Text>
            </Pressable>
          </View>
        </View>
      )}
    </>
  );
}

// --- Hold button (press&hold jog) ---
function HoldBtn({
  label,
  onPress,
  onHoldStart,
  onHoldEnd,
  variant = "ghost",
}: {
  label: string;
  onPress?: () => void;
  onHoldStart?: () => void;
  onHoldEnd?: () => void;
  variant?: "ghost" | "danger" | "primary";
}) {
  const timerRef = useRef<any>(null);
  const holdingRef = useRef(false);

  const bg =
    variant === "danger"
      ? styles.btnDanger
      : variant === "primary"
      ? styles.btnPrimary
      : styles.btn;

  const handlePressIn = () => {
    if (!onHoldStart) return;
    holdingRef.current = true;
    onHoldStart();

    // optional repeat pulse (if you later want repeated calls)
    timerRef.current = setInterval(() => {
      if (!holdingRef.current) return;
      // You can send repeated jog pulses here if your API is pulse-based.
      // In this implementation we assume "start" once is enough (server runs until stop).
    }, 250);
  };

  const handlePressOut = () => {
    holdingRef.current = false;
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    onHoldEnd?.();
  };

  return (
    <Pressable
      onPress={onPress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={({ pressed }) => [bg, pressed ? { opacity: 0.92 } : null]}
    >
      <Text style={styles.btnText}>{label}</Text>
    </Pressable>
  );
}

export default function ManualControlScreen({ navigation, route }: any) {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const [gatewayUrl, setGatewayUrl] = useState<string>(route?.params?.gateway || "");
  const [robotIp, setRobotIp] = useState<string>(route?.params?.robot || "");

  const [gatewayStatus, setGatewayStatus] = useState<DotStatus>("idle");
  const [robotStatus, setRobotStatus] = useState<DotStatus>("idle");

  const [enabled, setEnabled] = useState(false);

  const [frame, setFrame] = useState<JogFrame>("Tool");
  const [mode, setMode] = useState<ControlMode>("XYZ");

  const [speedPct, setSpeedPct] = useState(35); // 0..100 step 5
  const [gripperPct, setGripperPct] = useState(0); // 0..100

  const abortRef = useRef<AbortController | null>(null);

  const base = useMemo(() => normalizeBaseUrl(gatewayUrl), [gatewayUrl]);
  const cameraUrl = useMemo(() => (base ? `${base}/api/camera/realsense` : ""), [base]);

  const leftW = useMemo(() => clamp(Math.round(width * 0.24), 240, 290), [width]);
  const rightW = useMemo(() => clamp(Math.round(width * 0.26), 260, 320), [width]);

  const disconnect = async () => {
    setEnabled(false);
    // Optionally tell gateway to stop + disable
    try {
      if (base) {
        await fetch(`${base}/api/robot/stop`, { method: "POST" }).catch(() => {});
        await fetch(`${base}/api/robot/disable`, { method: "POST" }).catch(() => {});
      }
    } catch {}
    navigation.navigate("ModeSelect");
  };

  const goHome = () => {
    navigation.navigate("ModeSelect");
  };

  const stopAll = async () => {
    setEnabled(false);
    try {
      if (!base) return;
      await fetch(`${base}/api/robot/stop`, { method: "POST" }).catch(() => {});
      await fetch(`${base}/api/robot/disable`, { method: "POST" }).catch(() => {});
    } catch {}
  };

  const toggleEnable = async () => {
    if (!base) return;

    try {
      const url = enabled ? `${base}/api/robot/disable` : `${base}/api/robot/enable`;
      const resp = await fetch(url, { method: "POST" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setEnabled((v) => !v);
    } catch {
      Alert.alert("Action failed", "Could not toggle robot state. Check Gateway/Robot connection.");
    }
  };

  const speedChange = async (delta: number) => {
    const next = clamp(Math.round((speedPct + delta) / 5) * 5, 0, 100);
    setSpeedPct(next);

    // send speed to gateway (optional)
    try {
      if (!base) return;
      await fetch(`${base}/api/robot/speed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed_pct: next }),
      }).catch(() => {});
    } catch {}
  };

  const sendGripper = async (dir: "open" | "close") => {
    try {
      if (!base) return;
      // example: 5% change
      const next = clamp(gripperPct + (dir === "open" ? 5 : -5), 0, 100);
      setGripperPct(next);

      await fetch(`${base}/api/robot/gripper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: dir, pct: next }),
      }).catch(() => {});
    } catch {}
  };

  const startHoldJog = async (axis: string, sign: 1 | -1) => {
    try {
      if (!base) return;
      // Cancel any inflight
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
      const c = new AbortController();
      abortRef.current = c;

      await fetch(`${base}/api/robot/jog/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: c.signal,
        body: JSON.stringify({
          frame, // "Base" | "Tool"
          mode,  // "XYZ" | "RXYZ"
          axis,  // "x","y","z" or "rx","ry","rz"
          sign,  // 1 or -1
          speed_pct: speedPct,
        }),
      }).catch(() => {});
    } catch {}
  };

  const endHoldJog = async () => {
    try {
      if (!base) return;
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
      await fetch(`${base}/api/robot/jog/stop`, { method: "POST" }).catch(() => {});
    } catch {}
  };

  // Boot: load gateway/robot from storage to avoid “empty” crashes
  useEffect(() => {
    (async () => {
      try {
        const [gw, rb] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEYS.gateway),
          AsyncStorage.getItem(STORAGE_KEYS.robot),
        ]);
        const gwv = (route?.params?.gateway || gw || "").trim();
        const rbv = (route?.params?.robot || rb || "").trim();
        setGatewayUrl(gwv);
        setRobotIp(rbv);
      } catch {}
    })();
  }, [route?.params?.gateway, route?.params?.robot]);

  // Optional: quick status pings for dots
  useEffect(() => {
    let alive = true;
    const run = async () => {
      if (!base) {
        if (!alive) return;
        setGatewayStatus("error");
        setRobotStatus("idle");
        return;
      }

      try {
        setGatewayStatus("connecting");
        const r = await fetch(`${base}/api/status`);
        if (!r.ok) throw new Error("gw");
        if (!alive) return;
        setGatewayStatus("connected");
      } catch {
        if (!alive) return;
        setGatewayStatus("error");
        return;
      }

      // Robot status via gateway (optional)
      try {
        setRobotStatus("connecting");
        const r = await fetch(`${base}/api/robot/status`);
        if (!r.ok) throw new Error("rb");
        const data = await r.json().catch(() => ({}));
        if (!alive) return;
        setRobotStatus(data?.connected ? "connected" : "idle");
        setEnabled(!!data?.enabled);
      } catch {
        if (!alive) return;
        setRobotStatus("idle");
      }
    };

    run();
    const t = setInterval(run, 3500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [base]);

  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>Manual control is designed for landscape.</Text>
      </View>
    );
  }

  const Dot = ({ status }: { status: DotStatus }) => (
    <View style={[styles.dot, dotStyleFor(status, styles)]} />
  );

  const centerLabel = mode === "XYZ" ? "XYZ" : "RXYZ";

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />

      <SafeAreaView style={styles.safe}>
        {/* Top bar */}
        <View style={styles.topBar}>
          <View style={styles.topLeft}>
            <View style={styles.pill}>
              <Dot status={robotStatus} />
              <Text style={styles.pillText}>{enabled ? "Enabled" : "Disabled"}</Text>
            </View>

            <Pressable onPress={disconnect} style={({ pressed }) => [styles.topBtn, pressed ? { opacity: 0.92 } : null]}>
              <Text style={styles.topBtnText}>Disconnect</Text>
            </Pressable>
          </View>

          <View style={styles.infoBar}>
            <Text style={styles.infoLabel}>Info</Text>
            <Text style={styles.infoText}>X-mm  Y-mm  Z-mm</Text>
          </View>

          <View style={styles.topRight}>
            <Pressable
              onPress={toggleEnable}
              style={({ pressed }) => [
                styles.topBtn,
                enabled ? styles.topBtnMuted : styles.topBtnPrimary,
                pressed ? { opacity: 0.92 } : null,
              ]}
            >
              <Text style={styles.topBtnText}>{enabled ? "Disable" : "Enable"}</Text>
            </Pressable>

            <Pressable onPress={goHome} style={({ pressed }) => [styles.topBtn, pressed ? { opacity: 0.92 } : null]}>
              <Text style={styles.topBtnText}>Home</Text>
            </Pressable>
          </View>
        </View>

        <View style={styles.body}>
          {/* Left panel */}
          <View style={[styles.panel, { width: leftW }]}>
            <Dropdown
              label="Jog Frame"
              value={frame}
              options={[
                { key: "Base", title: "Base", subtitle: "Robot base coordinates" },
                { key: "Tool", title: "Tool", subtitle: "End-effector frame" },
              ]}
              onChange={(k) => setFrame(k as JogFrame)}
            />

            <View style={styles.panelCard}>
              <Text style={styles.cardTitle}>Z / Gripper</Text>

              {/* Gripper bar */}
              <View style={styles.barWrap}>
                <Text style={styles.barText}>Gripper: {gripperPct}%</Text>
                <View style={styles.barTrack}>
                  <View style={[styles.barFill, { width: pctToBarWidth(gripperPct) }]} />
                </View>
              </View>

              {/* Cross: Z vertical, Gripper horizontal */}
              <View style={styles.cross}>
                <HoldBtn
                  label="Z+"
                  onHoldStart={() => startHoldJog("z", +1)}
                  onHoldEnd={endHoldJog}
                />

                <View style={styles.crossMidRow}>
                  <HoldBtn label="Grip-" onPress={() => sendGripper("close")} />
                  <View style={styles.crossCenter}>
                    <Text style={styles.crossCenterText}>{frame}</Text>
                  </View>
                  <HoldBtn label="Grip+" onPress={() => sendGripper("open")} />
                </View>

                <HoldBtn
                  label="Z-"
                  onHoldStart={() => startHoldJog("z", -1)}
                  onHoldEnd={endHoldJog}
                />
              </View>
            </View>
          </View>

          {/* Center: Camera + STOP */}
          <View style={styles.center}>
            <View style={styles.cameraCard}>
              {!cameraUrl ? (
                <View style={styles.cameraPlaceholder}>
                  <Text style={styles.cameraPlaceholderText}>No Gateway URL</Text>
                </View>
              ) : (
                <View style={styles.cameraBox}>
                  {/* 4/3 aspect */}
                  <View style={styles.cameraAspect}>
                    <WebView
                      source={{ uri: cameraUrl }}
                      style={styles.web}
                      javaScriptEnabled
                      domStorageEnabled
                      scalesPageToFit
                      allowsInlineMediaPlayback
                      mediaPlaybackRequiresUserAction={false}
                      startInLoadingState
                      renderLoading={() => (
                        <View style={styles.webLoading}>
                          <ActivityIndicator />
                          <Text style={styles.webLoadingText}>Loading camera…</Text>
                        </View>
                      )}
                    />
                  </View>
                </View>
              )}
            </View>

            <Pressable onPress={stopAll} style={({ pressed }) => [styles.stopBtn, pressed ? { opacity: 0.92 } : null]}>
              <Text style={styles.stopText}>STOP</Text>
            </Pressable>
          </View>

          {/* Right panel */}
          <View style={[styles.panel, { width: rightW }]}>
            <Dropdown
              label="Control Mode"
              value={mode}
              options={[
                { key: "XYZ", title: "XYZ (Position)", subtitle: "Move X/Y axes" },
                { key: "RXYZ", title: "RXYZ (Orientation)", subtitle: "Rotate RX/RY/RZ" },
              ]}
              onChange={(k) => setMode(k as ControlMode)}
            />

            {/* Speed (same bar style) */}
            <View style={styles.panelCard}>
              <Text style={styles.cardTitle}>Speed</Text>

              <View style={styles.speedRow}>
                <HoldBtn label="-" onPress={() => speedChange(-5)} />
                <View style={styles.speedMid}>
                  <Text style={styles.speedPct}>{speedPct}%</Text>
                  <View style={styles.barTrack}>
                    <View style={[styles.barFill, { width: pctToBarWidth(speedPct) }]} />
                  </View>
                  <Text style={styles.speedHint}>Adjust in steps of 5%</Text>
                </View>
                <HoldBtn label="+" onPress={() => speedChange(+5)} />
              </View>
            </View>

            {/* Jog pad */}
            <View style={styles.panelCard}>
              <Text style={styles.cardTitle}>Jog</Text>

              <View style={styles.jogPad}>
                {/* Top */}
                <HoldBtn
                  label={mode === "XYZ" ? "Y+" : "RY+"}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", +1)}
                  onHoldEnd={endHoldJog}
                />

                {/* Middle row */}
                <View style={styles.jogMidRow}>
                  <HoldBtn
                    label={mode === "XYZ" ? "X-" : "RX-"}
                    onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", -1)}
                    onHoldEnd={endHoldJog}
                  />

                  <View style={styles.jogCenter}>
                    <Text style={styles.jogCenterText}>{centerLabel}</Text>
                  </View>

                  <HoldBtn
                    label={mode === "XYZ" ? "X+" : "RX+"}
                    onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", +1)}
                    onHoldEnd={endHoldJog}
                  />
                </View>

                {/* Bottom */}
                <HoldBtn
                  label={mode === "XYZ" ? "Y-" : "RY-"}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", -1)}
                  onHoldEnd={endHoldJog}
                />

                {/* Extra row for RZ if orientation */}
                {mode === "RXYZ" && (
                  <View style={styles.rzRow}>
                    <HoldBtn label="RZ-" onHoldStart={() => startHoldJog("rz", -1)} onHoldEnd={endHoldJog} />
                    <HoldBtn label="RZ+" onHoldStart={() => startHoldJog("rz", +1)} onHoldEnd={endHoldJog} />
                  </View>
                )}
              </View>
            </View>

            {/* Compact status dots (no “connections” panel) */}
            <View style={styles.statusMini}>
              <View style={styles.statusLine}>
                <Dot status={gatewayStatus} />
                <Text style={styles.statusText} numberOfLines={1}>
                  Gateway
                </Text>
              </View>
              <View style={styles.statusLine}>
                <Dot status={robotStatus} />
                <Text style={styles.statusText} numberOfLines={1}>
                  Robot
                </Text>
              </View>
            </View>
          </View>
        </View>
      </SafeAreaView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(8, 12, 22, 0.72)" },
  safe: { flex: 1 },

  topBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 14,
    paddingTop: 8,
    gap: 12,
  },
  topLeft: { flexDirection: "row", gap: 10, alignItems: "center", width: 260 },
  topRight: { flexDirection: "row", gap: 10, alignItems: "center", width: 260, justifyContent: "flex-end" },

  pill: {
    flexDirection: "row",
    gap: 8,
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
  },
  pillText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 12 },

  topBtn: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
  },
  topBtnPrimary: {
    backgroundColor: "rgba(37, 99, 235, 0.92)",
    borderColor: "rgba(37, 99, 235, 0.65)",
  },
  topBtnMuted: {
    backgroundColor: "rgba(0,0,0,0.18)",
  },
  topBtnText: { color: "white", fontWeight: "900", fontSize: 12 },

  infoBar: {
    flex: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
    flexDirection: "row",
    justifyContent: "center",
    gap: 14,
  },
  infoLabel: { color: "rgba(255,255,255,0.65)", fontWeight: "900", fontSize: 12 },
  infoText: { color: "rgba(255,255,255,0.90)", fontWeight: "900", fontSize: 12 },

  body: {
    flex: 1,
    flexDirection: "row",
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 12,
    gap: 12,
  },

  panel: {
    borderRadius: 22,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.62)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    gap: 12,
  },

  panelCard: {
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  cardTitle: { color: "rgba(255,255,255,0.88)", fontWeight: "900", fontSize: 12, marginBottom: 10 },

  center: { flex: 1, gap: 12 },
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

  cameraBox: { flex: 1, padding: 12 },
  cameraAspect: {
    width: "100%",
    aspectRatio: 4 / 3,
    borderRadius: 18,
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
    height: 72,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.60)",
  },
  stopText: { color: "white", fontWeight: "900", fontSize: 18, letterSpacing: 1 },

  // Dropdown
  dd: {
    borderRadius: 18,
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  ddText: { color: "rgba(255,255,255,0.90)", fontWeight: "900", fontSize: 12 },
  ddChevron: { color: "rgba(255,255,255,0.55)", fontWeight: "900" },

  modalOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.55)",
    alignItems: "center",
    justifyContent: "center",
    padding: 18,
    zIndex: 1000,
  },
  modalCard: {
    width: 520,
    maxWidth: "100%",
    borderRadius: 22,
    padding: 16,
    backgroundColor: "rgba(18, 27, 47, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  modalTitle: { color: "white", fontWeight: "900", fontSize: 16, marginBottom: 12 },
  modalItem: {
    borderRadius: 18,
    padding: 14,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  modalItemActive: {
    borderColor: "rgba(37, 99, 235, 0.70)",
    backgroundColor: "rgba(37, 99, 235, 0.14)",
  },
  modalItemTitle: { color: "white", fontWeight: "900", fontSize: 14 },
  modalItemSub: { color: "rgba(255,255,255,0.65)", marginTop: 4, fontSize: 12 },
  modalClose: {
    marginTop: 12,
    borderRadius: 18,
    paddingVertical: 12,
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  modalCloseText: { color: "rgba(255,255,255,0.85)", fontWeight: "900" },

  // Buttons
  btn: {
    minWidth: 72,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  btnPrimary: {
    minWidth: 72,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(37, 99, 235, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(37, 99, 235, 0.65)",
  },
  btnDanger: {
    minWidth: 72,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.60)",
  },
  btnText: { color: "white", fontWeight: "900", fontSize: 13 },

  // Bars (Gripper/Speed)
  barWrap: { gap: 8 },
  barText: { color: "rgba(255,255,255,0.75)", fontWeight: "900", fontSize: 12, textAlign: "center" },
  barTrack: {
    height: 8,
    borderRadius: 99,
    backgroundColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  barFill: { height: "100%", backgroundColor: "rgba(37, 99, 235, 0.95)" },

  // Cross
  cross: { marginTop: 10, gap: 10, alignItems: "center" },
  crossMidRow: { flexDirection: "row", gap: 10, alignItems: "center", justifyContent: "center" },
  crossCenter: {
    width: 60,
    height: 50,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  crossCenterText: { color: "rgba(255,255,255,0.80)", fontWeight: "900", fontSize: 12 },

  // Speed
  speedRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  speedMid: { flex: 1, gap: 8, alignItems: "center" },
  speedPct: { color: "white", fontWeight: "900", fontSize: 16 },
  speedHint: { color: "rgba(255,255,255,0.45)", fontWeight: "800", fontSize: 11 },

  // Jog pad
  jogPad: { gap: 10, alignItems: "center" },
  jogMidRow: { flexDirection: "row", gap: 10, alignItems: "center" },
  jogCenter: {
    width: 78,
    height: 56,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  jogCenterText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 14 },
  rzRow: { flexDirection: "row", gap: 10, marginTop: 6 },

  // Dots
  dot: { width: 10, height: 10, borderRadius: 99 },
  dotGreen: { backgroundColor: "rgba(34, 197, 94, 1)" },
  dotBlue: { backgroundColor: "rgba(59, 130, 246, 1)" },
  dotRed: { backgroundColor: "rgba(239, 68, 68, 1)" },
  dotGray: { backgroundColor: "rgba(148, 163, 184, 1)" },

  statusMini: {
    marginTop: -2,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 2,
  },
  statusLine: { flexDirection: "row", gap: 8, alignItems: "center" },
  statusText: { color: "rgba(255,255,255,0.60)", fontWeight: "900", fontSize: 11 },

  rotateWrap: {
    flex: 1,
    backgroundColor: "#0B1220",
    alignItems: "center",
    justifyContent: "center",
    padding: 22,
  },
  rotateTitle: { color: "white", fontSize: 20, fontWeight: "900" },
  rotateSub: { color: "rgba(255,255,255,0.65)", marginTop: 10, textAlign: "center", maxWidth: 420 },
});
