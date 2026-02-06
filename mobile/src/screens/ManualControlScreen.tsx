import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { WebView } from "react-native-webview";

// --- Types ---
type JogFrame = "Base" | "Tool";
type ControlMode = "XYZ" | "RXYZ";

type RobotStatus = {
  connected?: boolean;
  is_enabled?: boolean;
  state?: number | null;
  error_code?: number | null;
  warn_code?: number | null;
  pose?: number[] | null;
  tcp_speed?: number | null;
  gripper_available?: boolean;
  gripper_pos?: number | null;
  gripper_min?: number;
  gripper_max?: number;
  message?: string;
};

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return `http://${raw}`;
  return raw;
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export default function ManualControlScreen({ navigation, route }: any) {
  // gateway baseUrl: ConnectionHub / ModeSelect'ten geliyor
  const gatewayFromParams: string =
    route?.params?.gateway || route?.params?.baseUrl || route?.params?.gatewayUrl || "";

  const gatewayBase = useMemo(() => normalizeBaseUrl(gatewayFromParams), [gatewayFromParams]);

  // Kamera endpoint'i ileride netleşecek dediğin için tek yerde değiştir:
  // "/camera" veya "/api/camera"
  const cameraPath = "/camera";
  const cameraUrl = useMemo(() => (gatewayBase ? `${gatewayBase}${cameraPath}` : ""), [gatewayBase]);

  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  // --- UI state ---
  const [frame, setFrame] = useState<JogFrame>("Base");
  const [mode, setMode] = useState<ControlMode>("XYZ");

  const [speedPct, setSpeedPct] = useState<number>(35);

  const [status, setStatus] = useState<RobotStatus>({});
  const [loadingStatus, setLoadingStatus] = useState<boolean>(false);

  // Dropdown minimal: küçük modal yerine inline list toggle
  const [frameOpen, setFrameOpen] = useState(false);
  const [modeOpen, setModeOpen] = useState(false);

  // Hold-to-jog
  const jogTimer = useRef<any>(null);
  const jogAbort = useRef<boolean>(false);

  // --- Helpers ---
  const gripperPct = useMemo(() => {
    const min = status.gripper_min ?? 0;
    const max = status.gripper_max ?? 850;
    const pos = status.gripper_pos;
    if (pos == null || max === min) return 0;
    return clamp(Math.round(((pos - min) / (max - min)) * 100), 0, 100);
  }, [status.gripper_min, status.gripper_max, status.gripper_pos]);

  const isEnabled = !!status.is_enabled;
  const canControl = !!gatewayBase; // istersen status.connected vb. ile sıkılaştır

  const fetchStatus = useCallback(async () => {
    if (!gatewayBase) return;
    setLoadingStatus(true);
    try {
      const resp = await fetch(`${gatewayBase}/api/status`, { method: "GET" });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data?.ok === false) throw new Error(data?.message || "Status failed.");
      setStatus(data?.status || {});
    } catch {
      // App açıkken “hata istemiyorum” dediğin için sessiz geçiyoruz.
      // İstersen sadece status.message gösterebilirsin.
    } finally {
      setLoadingStatus(false);
    }
  }, [gatewayBase]);

  useEffect(() => {
    // Poll: 1.2s
    fetchStatus();
    const t = setInterval(fetchStatus, 1200);
    return () => clearInterval(t);
  }, [fetchStatus]);

  // Speed API
  const pushSpeed = useCallback(
    async (pct: number) => {
      if (!gatewayBase) return;
      const safe = clamp(Math.round(pct), 1, 100);
      setSpeedPct(safe);
      try {
        await fetch(`${gatewayBase}/api/speed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speed_pct: safe }),
        });
      } catch {}
    },
    [gatewayBase]
  );

  // Frame API
  const pushFrame = useCallback(
    async (f: JogFrame) => {
      setFrame(f);
      setFrameOpen(false);
      if (!gatewayBase) return;
      try {
        await fetch(`${gatewayBase}/api/frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame: f }),
        });
      } catch {}
    },
    [gatewayBase]
  );

  // Enable/Disable toggle
  const toggleEnable = useCallback(async () => {
    if (!gatewayBase) return;
    try {
      if (!isEnabled) {
        await fetch(`${gatewayBase}/api/enable`, { method: "POST" });
      } else {
        await fetch(`${gatewayBase}/api/disable`, { method: "POST" });
      }
      fetchStatus();
    } catch {}
  }, [gatewayBase, isEnabled, fetchStatus]);

  // STOP -> /api/stop + UI state reset (Enable/Disable mantığı değişsin)
  const stopAll = useCallback(async () => {
    if (!gatewayBase) return;
    try {
      await fetch(`${gatewayBase}/api/stop`, { method: "POST" });
    } catch {}
    // stop sonrası: enable/disable “disabled” gibi görünmeli
    setStatus((s) => ({ ...s, is_enabled: false, message: "Stopped." }));
  }, [gatewayBase]);

  // HOME -> ModeSelection
  const goHome = useCallback(() => {
    navigation.navigate("ModeSelect");
  }, [navigation]);

  // --- Jog logic ---
  const sendJog = useCallback(
    async (payload: any) => {
      if (!gatewayBase) return;
      try {
        await fetch(`${gatewayBase}/api/jog`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch {}
    },
    [gatewayBase]
  );

  const startHoldJog = useCallback(
    (axis: string, dir: 1 | -1) => {
      if (!canControl) return;

      jogAbort.current = false;
      const stepPos = 5; // mm
      const stepRot = 3; // deg

      const tick = async () => {
        if (jogAbort.current) return;

        if (mode === "XYZ") {
          const dx = axis === "x" ? stepPos * dir : 0;
          const dy = axis === "y" ? stepPos * dir : 0;
          const dz = axis === "z" ? stepPos * dir : 0;
          await sendJog({ dx, dy, dz, droll: 0, dpitch: 0, dyaw: 0 });
        } else {
          const droll = axis === "rx" ? stepRot * dir : 0;
          const dpitch = axis === "ry" ? stepRot * dir : 0;
          const dyaw = axis === "rz" ? stepRot * dir : 0;
          await sendJog({ dx: 0, dy: 0, dz: 0, droll, dpitch, dyaw });
        }

        jogTimer.current = setTimeout(tick, 120);
      };

      tick();
    },
    [canControl, mode, sendJog]
  );

  const endHoldJog = useCallback(() => {
    jogAbort.current = true;
    if (jogTimer.current) clearTimeout(jogTimer.current);
    jogTimer.current = null;
  }, []);

  // Gripper +/-
  const sendGripper = useCallback(
    async (action: "open" | "close") => {
      if (!gatewayBase) return;
      try {
        await fetch(`${gatewayBase}/api/gripper`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        fetchStatus();
      } catch {}
    },
    [gatewayBase, fetchStatus]
  );

  // --- Layout safety ---
  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>Manual control is optimized for landscape.</Text>
      </View>
    );
  }

  const infoLine = useMemo(() => {
    const p = status.pose;
    if (!p || p.length < 3) return "X-mm  Y-mm  Z-mm";
    return `X ${p[0]}  Y ${p[1]}  Z ${p[2]}`;
  }, [status.pose]);

  const controlCenterLabel = mode === "XYZ" ? "XYZ" : "RXYZ";

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />

      <View style={styles.root}>
        {/* LEFT PANEL */}
        <View style={styles.sideLeft}>
          {/* STATUS + DISCONNECT (istersen disconnect endpoint ekleyebilirsin) */}
          <View style={styles.topRow}>
            <Pill
              text={loadingStatus ? "Status..." : isEnabled ? "Enabled" : "Disabled"}
              tone={isEnabled ? "green" : "gray"}
              onPress={fetchStatus}
            />
            <SmallBtn
              text="Disconnect"
              onPress={() => Alert.alert("Not implemented", "Add /api/disconnect if needed.")}
            />
          </View>

          {/* Jog Frame Dropdown */}
          <Dropdown
            label={`Jog Frame: ${frame === "Base" ? "Base" : "Tool"}`}
            open={frameOpen}
            onToggle={() => {
              setModeOpen(false);
              setFrameOpen((v) => !v);
            }}
            items={[
              { key: "base", text: "Base", onPress: () => pushFrame("Base") },
              { key: "tool", text: "Tool", onPress: () => pushFrame("Tool") },
            ]}
          />

          {/* Z + Gripper cross */}
          <View style={styles.padBox}>
   
            <View style={styles.gripBarWrap}>
                  <Text style={styles.gripBarText}>Gripper: {gripperPct}%</Text>
              <View style={styles.gripBarTrack}>
                <View style={[styles.gripBarFill, { width: `${gripperPct}%` }]} />
              </View>

            </View>
            <View style={styles.cross}>
              <HoldBtn label="Z+" onHoldStart={() => startHoldJog("z", +1)} onHoldEnd={endHoldJog} />
              <View style={styles.crossMidRow}>
                <HoldBtn label="Grip-" onPress={() => sendGripper("close")} />
                <View style={styles.crossCenter}>
                  <Text style={styles.crossCenterText}>{frame}</Text>
                </View>
                <HoldBtn label="Grip+" onPress={() => sendGripper("open")} />
              </View>
              <HoldBtn label="Z-" onHoldStart={() => startHoldJog("z", -1)} onHoldEnd={endHoldJog} />
            </View>


          </View>
        </View>

        {/* CENTER */}
        <View style={styles.center}>
          <View style={styles.infoBar}>
            <Text style={styles.infoLabel}>Info</Text>
            <Text style={styles.infoText} numberOfLines={1}>
              {infoLine}
            </Text>
          </View>

          <View style={styles.cameraWrap}>
            {!cameraUrl ? (
              <View style={styles.cameraFallback}>
                <Text style={styles.cameraFallbackText}>No Gateway URL</Text>
              </View>
            ) : (
              <WebView
                style={styles.web}
                originWhitelist={["*"]}
                javaScriptEnabled={false}
                domStorageEnabled={false}
                // MJPEG için en pratik render:
                source={{
                  html: `
                    <html>
                      <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
                      <body style="margin:0;background:#000;">
                        <img src="${cameraUrl}" style="width:100%;height:100%;object-fit:contain;" />
                      </body>
                    </html>
                  `,
                }}
              />
            )}
          </View>

          <Pressable onPress={stopAll} style={({ pressed }) => [styles.stopBtn, pressed && { opacity: 0.9 }]}>
            <Text style={styles.stopText}>STOP</Text>
          </Pressable>
        </View>

        {/* RIGHT PANEL */}
        <View style={styles.sideRight}>
          <View style={styles.topRowRight}>
            {/* Enable/Disable tek toggle */}
            <Pill text={isEnabled ? "Disable" : "Enable"} tone={isEnabled ? "gray" : "blue"} onPress={toggleEnable} />
            <SmallBtn text="Home" onPress={goHome} />
          </View>

          {/* Control Mode Dropdown */}
          <Dropdown
            label={`Control Mode: ${mode}`}
            open={modeOpen}
            onToggle={() => {
              setFrameOpen(false);
              setModeOpen((v) => !v);
            }}
            items={[
              { key: "XYZ", text: "XYZ (Position)", onPress: () => (setMode("XYZ"), setModeOpen(false)) },
              { key: "RXYZ", text: "RXYZ (Orientation)", onPress: () => (setMode("RXYZ"), setModeOpen(false)) },
            ]}
          />

          {/* Speed +/- with % in center */}
          <View style={styles.speedBox}>
            <Text style={styles.speedTitle}>Speed</Text>
            <View style={styles.speedRow}>
              <SmallBtn text="-" onPress={() => pushSpeed(speedPct - 5)} />
              <View style={styles.speedPctPill}>
                <Text style={styles.speedPctText}>{speedPct}%</Text>
              </View>
              <SmallBtn text="+" onPress={() => pushSpeed(speedPct + 5)} />
            </View>
            <Text style={styles.speedHint}>Adjust in steps of 5%</Text>
          </View>

          {/* XY / RPY pad */}
          <View style={styles.padBox}>
            <Text style={styles.padTitle}>Jog</Text>

            <View style={styles.cross}>
              <HoldBtn
                label={mode === "XYZ" ? "Y+" : "RY+"}
                onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", +1)}
                onHoldEnd={endHoldJog}
              />

              <View style={styles.crossMidRow}>
                <HoldBtn
                  label={mode === "XYZ" ? "X-" : "RX-"}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", -1)}
                  onHoldEnd={endHoldJog}
                />
                <View style={styles.crossCenter}>
                  <Text style={styles.crossCenterText}>{controlCenterLabel}</Text>
                </View>
                <HoldBtn
                  label={mode === "XYZ" ? "X+" : "RX+"}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "x" : "rx", +1)}
                  onHoldEnd={endHoldJog}
                />
              </View>

              <HoldBtn
                label={mode === "XYZ" ? "Y-" : "RY-"}
                onHoldStart={() => startHoldJog(mode === "XYZ" ? "y" : "ry", -1)}
                onHoldEnd={endHoldJog}
              />

              {/* RZ only in RXYZ (yer kazandırmak için alt satır) */}
              {mode === "RXYZ" && (
                <View style={{ marginTop: 10 }}>
                  <View style={styles.rzRow}>
                    <HoldBtn label="RZ-" onHoldStart={() => startHoldJog("rz", -1)} onHoldEnd={endHoldJog} />
                    <HoldBtn label="RZ+" onHoldStart={() => startHoldJog("rz", +1)} onHoldEnd={endHoldJog} />
                  </View>
                </View>
              )}
            </View>
          </View>
        </View>
      </View>
    </ImageBackground>
  );
}

// ---------------- UI Components ----------------

function Pill({
  text,
  tone,
  onPress,
}: {
  text: string;
  tone: "green" | "blue" | "gray";
  onPress?: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.pill,
        tone === "green" ? styles.pillGreen : tone === "blue" ? styles.pillBlue : styles.pillGray,
        pressed && { opacity: 0.9 },
      ]}
    >
      <Text style={styles.pillText}>{text}</Text>
    </Pressable>
  );
}

function SmallBtn({ text, onPress }: { text: string; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.smallBtn, pressed && { opacity: 0.9 }]}>
      <Text style={styles.smallBtnText}>{text}</Text>
    </Pressable>
  );
}

function HoldBtn({
  label,
  onPress,
  onHoldStart,
  onHoldEnd,
}: {
  label: string;
  onPress?: () => void;
  onHoldStart?: () => void;
  onHoldEnd?: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      onPressIn={onHoldStart}
      onPressOut={onHoldEnd}
      style={({ pressed }) => [styles.holdBtn, pressed && { opacity: 0.9, transform: [{ scale: 0.99 }] }]}
    >
      <Text style={styles.holdBtnText}>{label}</Text>
    </Pressable>
  );
}

function Dropdown({
  label,
  open,
  onToggle,
  items,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  items: Array<{ key: string; text: string; onPress: () => void }>;
}) {
  return (
    <View style={{ marginTop: 12 }}>
      <Pressable onPress={onToggle} style={({ pressed }) => [styles.dropdown, pressed && { opacity: 0.9 }]}>
        <Text style={styles.dropdownText} numberOfLines={1}>
          {label}
        </Text>
        <Text style={styles.dropdownChevron}>{open ? "▲" : "▼"}</Text>
      </Pressable>

      {open && (
        <View style={styles.dropdownMenu}>
          {items.map((it) => (
            <Pressable key={it.key} onPress={it.onPress} style={({ pressed }) => [styles.dropdownItem, pressed && { opacity: 0.9 }]}>
              <Text style={styles.dropdownItemText}>{it.text}</Text>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
}

// ---------------- Styles ----------------

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(8, 12, 22, 0.70)" },

  root: {
    flex: 1,
    flexDirection: "row",
    padding: 14,
    gap: 12,
  },

  sideLeft: {
    width: 280,
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.74)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },

  sideRight: {
    width: 300,
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.74)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },

  center: {
    flex: 1,
    borderRadius: 18,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.60)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
  },

  topRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 10 },
  topRowRight: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 10 },

  pill: {
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderWidth: 1,
  },
  pillGreen: { backgroundColor: "rgba(34,197,94,0.18)", borderColor: "rgba(34,197,94,0.55)" },
  pillBlue: { backgroundColor: "rgba(37,99,235,0.22)", borderColor: "rgba(37,99,235,0.65)" },
  pillGray: { backgroundColor: "rgba(255,255,255,0.10)", borderColor: "rgba(255,255,255,0.12)" },
  pillText: { color: "white", fontWeight: "900", fontSize: 12 },

  smallBtn: {
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  smallBtnText: { color: "rgba(255,255,255,0.86)", fontWeight: "900" },

  dropdown: {
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  dropdownText: { color: "white", fontWeight: "900", fontSize: 12, flex: 1 },
  dropdownChevron: { color: "rgba(255,255,255,0.65)", fontWeight: "900" },

  dropdownMenu: {
    marginTop: 8,
    borderRadius: 14,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.22)",
  },
  dropdownItem: { paddingVertical: 12, paddingHorizontal: 12, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.06)" },
  dropdownItemText: { color: "rgba(255,255,255,0.88)", fontWeight: "800", fontSize: 12 },

  infoBar: {
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  infoLabel: { color: "rgba(255,255,255,0.75)", fontWeight: "900", fontSize: 12, width: 46 },
  infoText: { color: "white", fontWeight: "800", fontSize: 12, flex: 1 },

  cameraWrap: {
    marginTop: 12,
    flex: 1,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "rgba(0,0,0,0.35)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    // 4:3 görünümü koru (container taşmasın diye)
    // WebView içeride contain yapıyor; bu sadece kutunun hissi
  },
  web: { flex: 1, backgroundColor: "black" },
  cameraFallback: { flex: 1, alignItems: "center", justifyContent: "center" },
  cameraFallbackText: { color: "rgba(255,255,255,0.60)", fontWeight: "900" },

  stopBtn: {
    marginTop: 12,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239,68,68,0.95)",
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.75)",
  },
  stopText: { color: "white", fontWeight: "900", letterSpacing: 0.6 },

  padBox: {
    marginTop: 12,
    borderRadius: 16,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.16)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  padTitle: { color: "rgba(255,255,255,0.80)", fontWeight: "900", marginBottom: 10, fontSize: 12 },

  cross: { alignItems: "center" },
  crossMidRow: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10, marginVertical: 10 },
  crossCenter: {
    width: 62,
    height: 44,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
  },
  crossCenterText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 12 },

  holdBtn: {
    minWidth: 86,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  holdBtnText: { color: "white", fontWeight: "900", fontSize: 12 },

  rzRow: { flexDirection: "row", gap: 10 },

  speedBox: {
    marginTop: 12,
    borderRadius: 16,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.16)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  speedTitle: { color: "rgba(255,255,255,0.80)", fontWeight: "900", fontSize: 12 },
  speedRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginTop: 10 },
  speedPctPill: {
    flex: 1,
    marginHorizontal: 10,
    borderRadius: 14,
    paddingVertical: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  speedPctText: { color: "white", fontWeight: "900", fontSize: 14 },
  speedHint: { marginTop: 1, color: "rgba(255,255,255,0.45)", fontSize: 11, textAlign: "center" },

  gripBarWrap: { marginBottom: 10},
  gripBarTrack: {
    height: 6,
    marginVertical: 2,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  gripBarFill: { height: "100%", backgroundColor: "rgba(37,99,235,0.85)" },
  gripBarText: { marginTop: 8, color: "rgba(255,255,255,0.65)", fontSize: 11, fontWeight: "800", textAlign: "center" },

  rotateWrap: { flex: 1, backgroundColor: "#0B1220", alignItems: "center", justifyContent: "center", padding: 22 },
  rotateTitle: { color: "white", fontSize: 20, fontWeight: "900" },
  rotateSub: { color: "rgba(255,255,255,0.65)", marginTop: 10, textAlign: "center", maxWidth: 420 },
});
