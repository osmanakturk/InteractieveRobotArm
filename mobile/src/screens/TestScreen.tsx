// src/screens/TestScreen.tsx
import React, { useCallback, useMemo, useRef, useState } from "react";
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

/** Small icon button */
function IconBtn({
  icon,
  label,
  onPress,
  tone = "ghost",
  disabled,
}: {
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
  label?: string;
  onPress: () => void;
  tone?: "ghost" | "primary" | "danger";
  disabled?: boolean;
}) {
  return (
    <Pressable
      disabled={!!disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.iconBtn,
        tone === "primary" ? styles.iconBtnPrimary : tone === "danger" ? styles.iconBtnDanger : null,
        disabled ? { opacity: 0.45 } : null,
        pressed && !disabled ? { opacity: 0.9 } : null,
      ]}
    >
      <MaterialCommunityIcons name={icon} size={18} color="white" />
      {!!label && <Text style={styles.iconBtnText}>{label}</Text>}
    </Pressable>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <Text style={styles.sectionTitle}>{title}</Text>;
}

function Row({ children }: { children: React.ReactNode }) {
  return <View style={styles.row}>{children}</View>;
}

function Divider() {
  return <View style={styles.divider} />;
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

function BarCard({ title, rightText, valuePct }: { title: string; rightText: string; valuePct: number }) {
  return (
    <View style={styles.barCard}>
      <View style={styles.barCardTop}>
        <Text style={styles.barCardTitle}>{title}</Text>
        <Text style={styles.barCardSub}>{rightText}</Text>
      </View>
      <View style={styles.barTrack}>
        <View style={[styles.barFill, { width: pctToWidth(valuePct) }]} />
      </View>
    </View>
  );
}

/** Progress bar that supports tap + drag */
function ProgressSlider({
  value,
  onChange,
  disabled,
}: {
  value: number; // 0..100
  onChange: (pct: number) => void;
  disabled?: boolean;
}) {
  const [w, setW] = useState(240);

  const setByX = (x: number) => {
    const pct = clamp(Math.round((x / Math.max(1, w)) * 100), 1, 100);
    onChange(pct);
  };

  const onLayout = (e: LayoutChangeEvent) => {
    setW(Math.max(1, Math.round(e.nativeEvent.layout.width)));
  };

  const onPress = (e: GestureResponderEvent) => {
    if (disabled) return;
    setByX(e.nativeEvent.locationX);
  };

  return (
    <View
      style={[styles.sliderWrap, disabled ? { opacity: 0.5 } : null]}
      onLayout={onLayout}
      onStartShouldSetResponder={() => !disabled}
      onResponderGrant={(e) => setByX(e.nativeEvent.locationX)}
      onResponderMove={(e) => setByX(e.nativeEvent.locationX)}
    >
      <Pressable onPress={onPress} style={styles.sliderTrack}>
        <View style={[styles.sliderFill, { width: pctToWidth(value) }]} />
        <View style={[styles.sliderKnob, { left: `${clamp(value, 1, 100)}%` }]} />
      </Pressable>
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

  // ✅ Correct camera endpoint (MJPEG stream)
  const cameraUrl = useMemo(
    () => (gatewayBase ? `${gatewayBase}/api/camera/realsense` : ""),
    [gatewayBase]
  );

  // --- UI state ---
  const [frame, setFrame] = useState<JogFrame>("Tool");
  const [mode, setMode] = useState<ControlMode>("XYZ");

  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  const [speedPct, setSpeedPct] = useState<number>(35);

  const [gripperPct, setGripperPct] = useState<number>(0);
  const [gripperLoading, setGripperLoading] = useState(false);

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

  // Dots (no top big boxes)
  const gwDot: DotStatus = wsConnected ? "connected" : gatewayBase ? "error" : "idle";
  const rbDot: DotStatus = robotConnected ? "connected" : robot.value.trim() ? "idle" : "idle";
  const aiDot: DotStatus =
    aiserver.status === "connected" ? "connected" : aiserver.status === "error" ? "error" : "idle";
  const camDot: DotStatus = cameraStarted ? "connected" : "idle";

  // --- API helpers ---
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
      const data = await apiGet("/api/camera/realsense/status");
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

  const refreshGripperStatus = useCallback(async () => {
    if (!gatewayBase) return;
    setGripperLoading(true);
    try {
      const data = await apiGet("/api/gripper/status");
      const g = data?.gripper || {};
      const pct =
        typeof g.gripper_pct === "number"
          ? g.gripper_pct
          : typeof g.pct === "number"
            ? g.pct
            : typeof (live as any)?.status?.gripper_pct === "number"
              ? (live as any).status.gripper_pct
              : 0;
      setGripperPct(clamp(Math.round(pct), 0, 100));
    } catch {
      // silent
    } finally {
      setGripperLoading(false);
    }
  }, [apiGet, gatewayBase, live]);

  useFocusEffect(
    useCallback(() => {
      refreshCameraStatus();
      refreshGripperStatus();
    }, [refreshCameraStatus, refreshGripperStatus])
  );

  // --- Main actions ---
  const pushFrame = useCallback(
    async (f: JogFrame) => {
      setFrame(f);
      // Backend expects base|tool (string)
      try {
        await apiPost("/api/frame", { frame: f.toLowerCase() });
      } catch {
        // silent
      }
    },
    [apiPost]
  );

  const sendJogDelta = useCallback(
    async (payload: any) => {
      try {
        await apiPost("/api/jog", payload);
      } catch {
        // silent (429 throttling etc.)
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
        await apiPost("/api/gripper", { action });
        await refreshGripperStatus();
      } catch (e: any) {
        Alert.alert("Gripper failed", e?.message || "Gripper action failed.");
      }
    },
    [apiPost, refreshGripperStatus]
  );

  const onStop = useCallback(async () => {
    try {
      await apiPost("/api/stop");
    } catch (e: any) {
      Alert.alert("STOP failed", e?.message || "Stop failed.");
    }
  }, [apiPost]);

  // --- Settings actions ---
  const onEnableDisable = useCallback(async () => {
    try {
      if (!gatewayBase) return Alert.alert("Gateway missing", "Please set Gateway in Connection Hub.");
      if (!robotConnected) return Alert.alert("Robot not connected", "Connect robot from Connection Hub first.");
      if (enabled) await apiPost("/api/disable");
      else await apiPost("/api/enable");
    } catch (e: any) {
      Alert.alert("Action failed", e?.message || "Enable/Disable failed.");
    }
  }, [apiPost, enabled, gatewayBase, robotConnected]);

  const onClearSafety = useCallback(async () => {
    try {
      await apiPost("/api/safety/clear");
    } catch (e: any) {
      Alert.alert("Clear Safety failed", e?.message || "Clear Safety failed.");
    }
  }, [apiPost]);

  const onCameraStart = useCallback(async () => {
    if (!gatewayBase) return;
    setCameraLoading(true);
    try {
      await apiPost("/api/camera/realsense/start");
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
      await apiPost("/api/camera/realsense/stop");
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
        await apiPost("/api/speed", { speed_pct: v });
      } catch {
        // silent
      }
    },
    [apiPost]
  );

  // --- Navigation helpers ---
  const goConnectionHub = useCallback(() => {
    navigation.navigate("ConnectionHub");
  }, [navigation]);

  // --- Layout ---
  const leftW = useMemo(() => clamp(Math.round(width * 0.28), 240, 310), [width]);
  const rightW = useMemo(() => clamp(Math.round(width * 0.28), 260, 340), [width]);

  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>Manual control is designed for landscape.</Text>
      </View>
    );
  }

  const canControl = !!gatewayBase && robotConnected;

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />
      <SafeAreaView style={styles.safe}>
        {/* Top bar */}
        <View style={styles.topBar}>
          <View style={styles.topLeft}>
            <IconBtn icon="menu" label="Status" onPress={() => setLeftOpen(true)} />
          </View>
          <View style={styles.topLeft}>

            <Row>
              <PillToggle label="Base" active={frame === "Base"} onPress={() => pushFrame("Base")} />
              <PillToggle label="Tool" active={frame === "Tool"} onPress={() => pushFrame("Tool")} />
            </Row>

          </View>

          {/* Minimal status dots line (compact) */}
          <View style={styles.topCenterCompact}>
            <View style={styles.dotLine}>
              <View style={[styles.dot, dotStyle(gwDot)]} />
              <Text style={styles.dotText}>Gateway</Text>
            </View>
            <View style={styles.dotLine}>
              <View style={[styles.dot, dotStyle(rbDot)]} />
              <Text style={styles.dotText}>Robot</Text>
            </View>
            <View style={styles.dotLine}>
              <View style={[styles.dot, dotStyle(camDot)]} />
              <Text style={styles.dotText}>Camera</Text>
            </View>
            <View style={styles.dotLine}>
              <View style={[styles.dot, dotStyle(aiDot)]} />
              <Text style={styles.dotText}>AI Server</Text>
            </View>
          </View>

          <View style={styles.topRight}>
            <Row>
              <PillToggle label="XYZ" active={mode === "XYZ"} onPress={() => setMode("XYZ")} />
              <PillToggle label="RXYZ" active={mode === "RXYZ"} onPress={() => setMode("RXYZ")} />
            </Row>
          </View>

          <View style={styles.topRight}>
            <IconBtn icon="cog-outline" label="Ayarlar" onPress={() => setRightOpen(true)} />
          </View>
        </View>

        {/* Body (3 blocks) */}
        <View style={styles.body}>
          {/* LEFT BLOCK */}
          <View style={[styles.panel, { width: leftW }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 14 }}>


              <BarCard
                title="Gripper"
                rightText={gripperLoading ? "Loading…" : `${gripperPct}/100`}
                valuePct={gripperPct}
              />


              <View style={{ marginTop: 12,  alignItems: "center"}}>
                <HoldBtn
                  label={mode === "XYZ" ? "Z+" : "RZ+"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "z" : "rz", +1)}
                  onHoldEnd={endHoldJog}
                />

                <View style={styles.jogMidRow}>
                  <HoldBtn label="Grip-" disabled={!canControl} onPress={() => sendGripper("close")} />
                  <View style={styles.crossCenter}>
                    <Text style={styles.crossCenterText}>{frame}</Text>
                  </View>
                  <HoldBtn label="Grip+" disabled={!canControl} onPress={() => sendGripper("open")} />
                </View>

                <HoldBtn
                  label={mode === "XYZ" ? "Z-" : "RZ-"}
                  disabled={!canControl}
                  onHoldStart={() => startHoldJog(mode === "XYZ" ? "z": "rz", -1)}
                  onHoldEnd={endHoldJog}
                />
              </View>


            </ScrollView>
          </View>

          {/* CENTER BLOCK: Camera + STOP */}
          <View style={styles.center}>
            <View style={styles.cameraCard}>
              {!cameraUrl ? (
                <View style={styles.cameraPlaceholder}>
                  <Text style={styles.cameraPlaceholderText}>No Gateway URL</Text>
                </View>
              ) : (
                <View style={styles.cameraBox}>
                  {/* 4/3 aspect area */}
                  <View style={styles.cameraAspect}>
                    <WebView
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
                      // MJPEG: render via <img>
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

          {/* RIGHT BLOCK */}
          <View style={[styles.panel, { width: rightW }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 14 }}>



              <BarCard
                title="Speed"
                rightText={`${speedPct}%`}
                valuePct={speedPct}
              />

              <View style={{ flexDirection:"row", alignItems:"center", justifyContent:"space-around" }}>
                <IconBtn icon="minus" label="Speed" onPress={() => pushSpeed(speedPct - 5)} />

                <IconBtn icon="plus" label="Speed" onPress={() => pushSpeed(speedPct + 5)} />

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
            <View style={[styles.drawer, { left: 12, width: clamp(Math.round(width * 0.20), 320, 560) }]}>
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
                  <Text style={styles.drawerHint}>
                    {robotConnected ? "Connected (via gateway)." : "Not connected."}
                  </Text>
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
                    goConnectionHub();
                  }}
                  style={({ pressed }) => [styles.hubLink, pressed ? { opacity: 0.9 } : null]}
                >
                  <MaterialCommunityIcons name="link-variant" size={18} color="white" />
                  <Text style={styles.hubLinkText}>Connection Hub’a git</Text>
                </Pressable>

                <Text style={styles.drawerTiny}>
                  Not: Bağlantı adreslerini / IP’leri değiştirmek için Connection Hub ekranını kullan.
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
                <Text style={styles.drawerTitle}>Ayarlar</Text>
                <IconBtn icon="close" onPress={() => setRightOpen(false)} />
              </View>

              <ScrollView contentContainerStyle={{ paddingBottom: 12 }} showsVerticalScrollIndicator={false}>
                {/* Robot controls */}

                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Telemetry</Text>

                </View>
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Robot</Text>

                  <View style={styles.drawerBtnRow}>
                    <IconBtn
                      icon={enabled ? "power" : "power"}
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

                    <Text>Home</Text>
                  </View>

              

                  {(safetyLimit || safetyMsg) && (
                    <Text style={styles.warnText}>
                      {safetyLimit ? "Safety: limit hit." : "Safety warning."} {safetyMsg ? `(${safetyMsg})` : ""}
                    </Text>
                  )}
                </View>



                {/* Camera controls */}
                <View style={styles.drawerCard}>
                  <Text style={styles.drawerSectionTitle}>Camera</Text>
                  <Text style={styles.drawerHint}>
                    Status: {cameraLoading ? "Loading…" : cameraStarted ? "Started" : "Stopped"}
                    {cameraLastError ? ` • ${cameraLastError}` : ""}
                  </Text>

                  <View style={styles.drawerBtnRow}>
                    <IconBtn
                      icon="play-outline"
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
                    {/* ✅ FIX: no "camera-refresh" icon */}
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

  topBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingTop: 0,
    gap: 0,
  },
  topLeft: { flexDirection: "row", alignItems: "center", gap: 10, width: 180 },
  topRight: { width: 180, alignItems: "flex-end" },

  topCenterCompact: {
    flex: 1,
    height: 42,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
    padding: 10,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 14,
  },
  dotLine: { flexDirection: "row", alignItems: "center", gap: 6 },
  dotText: { color: "rgba(255,255,255,0.75)", fontWeight: "900", fontSize: 11 },

  iconBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    backgroundColor: "rgba(0,0,0,0.18)",
  },
  iconBtnPrimary: {
    backgroundColor: "rgba(37, 99, 235, 0.92)",
    borderColor: "rgba(37, 99, 235, 0.65)",
  },
  iconBtnDanger: {
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderColor: "rgba(239, 68, 68, 0.65)",
  },
  iconBtnText: { color: "white", fontWeight: "900", fontSize: 12 },

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

  cameraBox: { flex: 1, padding: 12 },
  cameraAspect: {
    width: "100%",
    height: "100%",
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
    height: 40,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(239, 68, 68, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.60)",
  },
  stopText: { color: "white", fontWeight: "900", fontSize: 18, letterSpacing: 1 },

  sectionTitle: { color: "rgba(255,255,255,0.88)", fontWeight: "900", fontSize: 12, marginBottom: 10 },
  row: { flexDirection: "row", gap: 10, alignItems: "center" },
  jogTopRow: { flexDirection: "row", gap: 10, justifyContent: "center", alignItems: "center" },

  pillToggle: {
    flex: 1,
    borderRadius: 16,
    paddingVertical: 12,
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

  divider: { marginVertical: 14, height: 1, backgroundColor: "rgba(255,255,255,0.08)" },

  holdBtn: {
    minWidth: 10,
    maxWidth: 80,
    borderRadius: 16,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  holdBtnText: { color: "white", fontWeight: "900", fontSize: 13 },

  jogMidRow: { flexDirection: "row", gap: 10, alignItems: "center", justifyContent: "center", marginVertical: 10 },
  crossCenter: {
    width: 62,
    height: 46,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.18)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  jogBottomRow: {
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
    justifyContent: "center"
  },
  crossCenterText: { color: "rgba(255,255,255,0.85)", fontWeight: "900", fontSize: 12 },

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

  rzRow: { flexDirection: "row", gap: 10, marginTop: 10 },

  note: { color: "rgba(255,255,255,0.50)", fontWeight: "800", fontSize: 11, textAlign: "center" },

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
  barFill: { height: "100%", backgroundColor: "rgba(37, 99, 235, 0.95)" },

  // Drawers
  drawerOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.55)",
    zIndex: 2000,
  },
  drawer: {
    position: "absolute",
    top: 26,
    bottom: 16,
    borderRadius: 22,
    padding: 12,
    backgroundColor: "rgba(18, 27, 47, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  drawerHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: 10,
  },
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
  drawerBtnRow: { flexDirection: "row", gap: 10, marginTop: 10, flexWrap: "wrap", justifyContent:"space-evenly" },

  drawerTiny: { color: "rgba(255,255,255,0.55)", fontWeight: "800", marginTop: 8, fontSize: 11 },
  drawerTinyCenter: { color: "rgba(255,255,255,0.55)", fontWeight: "900", marginTop: 10, fontSize: 11, textAlign: "center" },

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

  // Slider
  sliderWrap: { width: "100%" },
  sliderTrack: {
    width: "100%",
    height: 14,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
    justifyContent: "center",
  },
  sliderFill: { height: "100%", backgroundColor: "rgba(37, 99, 235, 0.95)" },
  sliderKnob: {
    position: "absolute",
    top: -5,
    width: 22,
    height: 22,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.92)",
    marginLeft: -11,
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.18)",
  },

  presetRow: { flexDirection: "row", gap: 8, marginTop: 12, flexWrap: "wrap" },
  preset: {
    minWidth: 54,
    height: 38,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255,255,255,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  presetText: { color: "white", fontWeight: "900" },

  rotateWrap: { flex: 1, backgroundColor: "#0B1220", alignItems: "center", justifyContent: "center", padding: 22 },
  rotateTitle: { color: "white", fontSize: 20, fontWeight: "900" },
  rotateSub: { color: "rgba(255,255,255,0.65)", marginTop: 10, textAlign: "center", maxWidth: 420 },
});
