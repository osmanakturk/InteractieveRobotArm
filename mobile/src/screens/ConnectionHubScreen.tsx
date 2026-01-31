import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ImageBackground,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
type ConnKey = "gateway" | "robot" | "aiserver";

export const STORAGE_KEYS = {
  gateway: "@conn/gateway",
  robot: "@conn/robot",
  aiserver: "@conn/aiserver",
} as const;

type ConnState = {
  value: string;
  status: "idle" | "connecting" | "connected" | "error";
  message?: string;
  lastOkAt?: number;
};

type RouteParams = {
  selected?: ConnKey;
  nextRoute?: "Manual" | "PickPlace" | "Voice";
  returnTo?: "ModeSelect" | "None"; 
  gateway?: string;
  robot?: string;
  aiserver?: string;
};

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return `http://${raw}`;
  return raw;
}

export default function ConnectionHubScreen({ navigation, route }: any) {
  const params: RouteParams = route?.params || {};
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const [selected, setSelected] = useState<ConnKey>(params.selected || "gateway");

  const [gateway, setGateway] = useState<ConnState>({ value: params.gateway || "", status: "idle" });
  const [robot, setRobot] = useState<ConnState>({ value: params.robot || "", status: "idle" });
  const [aiserver, setAiServer] = useState<ConnState>({ value: params.aiserver || "", status: "idle" });

  const abortRef = useRef<AbortController | null>(null);
  const saveTimer = useRef<any>(null);

  const scheduleSave = (key: ConnKey, value: string) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      try {
        await AsyncStorage.setItem(STORAGE_KEYS[key], value.trim());
      } catch {
        // sessiz
      }
    }, 250);
  };

  useEffect(() => {
    return () => {
   
      try {
        abortRef.current?.abort();
      } catch {}
      abortRef.current = null;

      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = null;
    };
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [gw, rb, ai] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEYS.gateway),
          AsyncStorage.getItem(STORAGE_KEYS.robot),
          AsyncStorage.getItem(STORAGE_KEYS.aiserver),
        ]);

        setGateway((s) => ({ ...s, value: s.value || (gw || "").trim() }));
        setRobot((s) => ({ ...s, value: s.value || (rb || "").trim() }));
        setAiServer((s) => ({ ...s, value: s.value || (ai || "").trim() }));
      } catch {
        // sessiz
      }
    })();
  }, []);

  const selectedState = useMemo(() => {
    if (selected === "gateway") return gateway;
    if (selected === "robot") return robot;
    return aiserver;
  }, [selected, gateway, robot, aiserver]);

  const setStatus = (key: ConnKey, patch: Partial<ConnState>) => {
    if (key === "gateway") setGateway((s) => ({ ...s, ...patch }));
    if (key === "robot") setRobot((s) => ({ ...s, ...patch }));
    if (key === "aiserver") setAiServer((s) => ({ ...s, ...patch }));
  };

  const setSelectedValue = (value: string) => {
    // connecting iken değişirse iptal
    if (selectedState.status === "connecting" && abortRef.current) {
      try {
        abortRef.current.abort();
      } catch {}
      abortRef.current = null;

      setStatus(selected, { status: "idle", message: "Cancelled." });
    }

    setStatus(selected, { value, message: undefined });

    scheduleSave(selected, value);
  };

  const canGoModeSelection = useMemo(() => {
    return !!gateway.value.trim() && !!robot.value.trim();
  }, [gateway.value, robot.value]);

  const ensureGatewayEntered = () => {
    if (!gateway.value.trim()) {
      Alert.alert("Gateway required", "Please enter the Gateway address first.");
      setSelected("gateway");
      return false;
    }
    return true;
  };

  const connectSelected = async () => {
    const key = selected;
    const value = selectedState.value.trim();

    if (!value) {
      Alert.alert("Missing address", "Please enter an IP/URL.");
      return;
    }
    if (key !== "gateway" && !ensureGatewayEntered()) return;

    // tek istek
    try {
      abortRef.current?.abort();
    } catch {}
    abortRef.current = null;

    const controller = new AbortController();
    abortRef.current = controller;

    setStatus(key, { status: "connecting", message: "Connecting..." });

    try {
      // 1) Gateway ping
      if (key === "gateway") {
        const base = normalizeBaseUrl(value);
        const resp = await fetch(`${base}/api/status`, { method: "GET", signal: controller.signal });

        if (!resp.ok) throw new Error(`Gateway not reachable (HTTP ${resp.status}).`);

        setStatus("gateway", { status: "connected", message: "Gateway connected.", lastOkAt: Date.now() });
        await AsyncStorage.setItem(STORAGE_KEYS.gateway, value);

        // ✅ Eğer buraya bir “eksik bağlantı” akışından geldiysek ModeSelection’a geri dön
        if (params.nextRoute && (params.returnTo === "ModeSelect" || !params.returnTo)) {
          navigation.replace("ModeSelect", { autoNavigateTo: params.nextRoute });
        }
        return;
      }

      // 2) Robot connect via gateway
      if (key === "robot") {
        const gwBase = normalizeBaseUrl(gateway.value.trim());
        const resp = await fetch(`${gwBase}/api/connect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({ ip: value }),
        });

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || data?.ok === false) {
          throw new Error(data?.message || `Robot connect failed (HTTP ${resp.status}).`);
        }

        setStatus("robot", { status: "connected", message: "Robot connected via Gateway.", lastOkAt: Date.now() });
        await AsyncStorage.setItem(STORAGE_KEYS.robot, value);

        if (params.nextRoute && (params.returnTo === "ModeSelect" || !params.returnTo)) {
          navigation.replace("ModeSelect", { autoNavigateTo: params.nextRoute });
        }
        return;
      }

      // 3) AI server set via gateway
      if (key === "aiserver") {
        const gwBase = normalizeBaseUrl(gateway.value.trim());
        const resp = await fetch(`${gwBase}/api/ai_server`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({ url: value }),
        });

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || data?.ok === false) {
          throw new Error(data?.message || `AI Server setup failed (HTTP ${resp.status}).`);
        }

        setStatus("aiserver", { status: "connected", message: "AI Server configured via Gateway.", lastOkAt: Date.now() });
        await AsyncStorage.setItem(STORAGE_KEYS.aiserver, value);

        if (params.nextRoute && (params.returnTo === "ModeSelect" || !params.returnTo)) {
          navigation.replace("ModeSelect", { autoNavigateTo: params.nextRoute });
        }
        return;
      }
    } catch (e: any) {
      if (e?.name === "AbortError") setStatus(key, { status: "idle", message: "Cancelled." });
      else setStatus(key, { status: "error", message: e?.message || "Connection failed." });
    } finally {
      abortRef.current = null;
    }
  };

  const goModeSelection = () => {
    if (!canGoModeSelection) return;
    navigation.navigate("ModeSelect");
  };

  const ConnRow = ({ k, title, subtitle }: { k: ConnKey; title: string; subtitle: string }) => {
    const st = k === "gateway" ? gateway : k === "robot" ? robot : aiserver;
    const active = selected === k;

    const dotStyle =
      st.status === "connected"
        ? styles.dotGreen
        : st.status === "connecting"
        ? styles.dotBlue
        : st.status === "error"
        ? styles.dotRed
        : styles.dotGray;

    return (
      <Pressable
        onPress={() => setSelected(k)}
        style={({ pressed }) => [
          styles.sideItem,
          active ? styles.sideItemActive : null,
          pressed ? { opacity: 0.9 } : null,
        ]}
      >
        <View style={[styles.dot, dotStyle]} />
        <View style={{ flex: 1 }}>
          <Text style={styles.sideTitle}>{title}</Text>
          <Text style={styles.sideSub} numberOfLines={1}>{subtitle}</Text>
          {!!st.value && <Text style={styles.sideValue} numberOfLines={1}>{st.value}</Text>}
        </View>
      </Pressable>
    );
  };

  const headerTitle =
    selected === "gateway" ? "Gateway Connection" : selected === "robot" ? "Robot Connection" : "AI Server Connection";

  const headerDesc =
    selected === "gateway"
      ? "Connect to the Gateway to control robot + camera."
      : selected === "robot"
      ? "Set the xArm robot IP. The Gateway will connect to the robot."
      : "Set the AI Server address. The Gateway will use it for AI features (Pick&Place / Voice).";

  const placeholder =
    selected === "gateway"
      ? "e.g. 192.168.1.20:8000"
      : selected === "robot"
      ? "e.g. 192.168.1.201"
      : "e.g. 192.168.1.50:9000";

  if (!isLandscape) {
    return (
      <View style={styles.rotateWrap}>
        <Text style={styles.rotateTitle}>Rotate your device</Text>
        <Text style={styles.rotateSub}>This app is designed for landscape mode for optimal robot control.</Text>
      </View>
    );
  }

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />

      <View style={styles.root}>
        {/* Sidebar */}
        <View style={styles.sidebar}>
          <ScrollView
            style={{ flex: 1 }}                       
            contentContainerStyle={styles.sidebarInner}
            showsVerticalScrollIndicator={false}
          >
            <View>
              <View style={styles.brand}>
                <View style={styles.brandLogo}>
                    <MaterialCommunityIcons name="robot-industrial-outline" size={24} color="black" />
                    </View>
                <View>
                  <Text style={styles.brandTitle}>xArm Controller</Text>
                  <Text style={styles.brandSub}>Connection Hub</Text>
                </View>
              </View>

              <View style={styles.sideSection}>
                <ConnRow k="gateway" title="Gateway" subtitle="Gateway address (Jetson)" />
                <ConnRow k="robot" title="Robot" subtitle="xArm robot IP (via Gateway)" />
                <ConnRow k="aiserver" title="AI Server" subtitle="AI service URL (via Gateway)" />
              </View>
            </View>

            <View style={styles.sideFooter}>
              <Text style={styles.sideFooterText}>
                Gateway: {gateway.status === "connected" ? "OK" : "Not connected"} • Robot:{" "}
                {robot.status === "connected" ? "OK" : "Not connected"} • AI:{" "}
                {aiserver.status === "connected" ? "OK" : "Optional"}
              </Text>
              <Text style={styles.sideFooterTiny}>UFACTORY xArm • Network Hub</Text>
            </View>
          </ScrollView>
        </View>

        {/* Main */}
        <View style={styles.main}>
          <View style={styles.mainHeader}>
            <View style={{ flex: 1 }}>
              <Text style={styles.hTitle}>{headerTitle}</Text>
              <Text style={styles.hDesc}>{headerDesc}</Text>
            </View>

            <View style={styles.headerPill}>
              <View
                style={[
                  styles.pillDot,
                  selectedState.status === "connected"
                    ? styles.dotGreen
                    : selectedState.status === "connecting"
                    ? styles.dotBlue
                    : selectedState.status === "error"
                    ? styles.dotRed
                    : styles.dotGray,
                ]}
              />
              <Text style={styles.pillText}>
                {selectedState.status === "connected"
                  ? "Connected"
                  : selectedState.status === "connecting"
                  ? "Connecting"
                  : selectedState.status === "error"
                  ? "Error"
                  : "Idle"}
              </Text>
            </View>
          </View>

          <ScrollView contentContainerStyle={styles.mainBody} showsVerticalScrollIndicator={false}>
            <Text style={styles.label}>
              {selected === "gateway" ? "Gateway IP / URL" : selected === "robot" ? "Robot IP" : "AI Server IP / URL"}
            </Text>

            <View style={styles.inputWrap}>
              <TextInput
                value={selectedState.value}
                onChangeText={setSelectedValue}
                placeholder={placeholder}
                placeholderTextColor="rgba(255,255,255,0.35)"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                style={styles.input}
              />

              {selectedState.status === "connecting" && (
                <View pointerEvents="none" style={styles.spinner}>
                  <ActivityIndicator />
                </View>
              )}
            </View>

            {!!selectedState.message && (
              <Text style={[styles.msg, selectedState.status === "error" ? styles.msgErr : styles.msgOk]}>
                {selectedState.message}
              </Text>
            )}

            <Pressable onPress={connectSelected} style={({ pressed }) => [styles.primaryBtn, pressed ? { opacity: 0.9 } : null]}>
              <Text style={styles.primaryText}>
                {selected === "gateway"
                  ? "Connect Gateway"
                  : selected === "robot"
                  ? "Connect Robot (via Gateway)"
                  : "Set AI Server (via Gateway)"}
              </Text>
            </Pressable>

            <Text style={styles.hint}>If you edit the address while connecting, the attempt is cancelled.</Text>

            <View style={styles.divider} />

            <Pressable
              disabled={!canGoModeSelection}
              onPress={goModeSelection}
              style={({ pressed }) => [
                styles.goBtn,
                !canGoModeSelection ? styles.goBtnDisabled : null,
                pressed && canGoModeSelection ? { opacity: 0.9 } : null,
              ]}
            >
              <Text style={styles.goBtnText}>Go to Mode Selection</Text>
            </Pressable>

            <Text style={styles.smallNote}>
              Mode Selection needs Gateway + Robot. AI Server is required only for Pick&Place/Voice.
            </Text>
          </ScrollView>

          <View style={styles.mainFooter}>
            <Text style={styles.footerText}>UFACTORY xArm • Connection Hub</Text>
          </View>
        </View>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(8, 12, 22, 0.68)" },

  root: {
    flex: 1,
    flexDirection: "row",
    padding: 14,
    gap: 14,
    alignItems: "stretch", 
  },

  sidebar: {
    width: 280,
    flexShrink: 0,        
    alignSelf: "stretch",  
    borderRadius: 22,
    backgroundColor: "rgba(18, 27, 47, 0.72)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
  },
  sidebarInner: {
    padding: 16,
    paddingBottom: 18,
    flexGrow: 1,      
    justifyContent: "space-between",
  },

  brand: { flexDirection: "row", alignItems: "center", gap: 12, marginBottom: 16 },
  brandLogo: { width: 44, height: 44, borderRadius: 12, backgroundColor: "rgba(37, 99, 235, 0.95)", alignItems: "center", justifyContent: "center" },
  brandLogoText: { color: "white", fontWeight: "900", fontSize: 18 },
  brandTitle: { color: "white", fontWeight: "900", fontSize: 16 },
  brandSub: { color: "rgba(255,255,255,0.55)", marginTop: 2, fontSize: 12 },

  sideSection: { gap: 10 },
  sideItem: { flexDirection: "row", gap: 10, padding: 12, borderRadius: 16, backgroundColor: "rgba(0,0,0,0.16)", borderWidth: 1, borderColor: "rgba(255,255,255,0.08)" },
  sideItemActive: { borderColor: "rgba(37, 99, 235, 0.65)", backgroundColor: "rgba(37, 99, 235, 0.12)" },
  sideTitle: { color: "white", fontWeight: "900", fontSize: 13 },
  sideSub: { color: "rgba(255,255,255,0.60)", fontSize: 11, marginTop: 2 },
  sideValue: { color: "rgba(255,255,255,0.75)", fontSize: 11, marginTop: 6, fontFamily: "monospace" },

  sideFooter: { marginTop: 18, paddingTop: 12, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.08)" },
  sideFooterText: { color: "rgba(255,255,255,0.70)", fontSize: 12, fontWeight: "700" },
  sideFooterTiny: { color: "rgba(255,255,255,0.35)", fontSize: 11, marginTop: 6 },

  dot: { width: 10, height: 10, borderRadius: 99, marginTop: 4 },
  dotGreen: { backgroundColor: "rgba(34, 197, 94, 1)" },
  dotBlue: { backgroundColor: "rgba(59, 130, 246, 1)" },
  dotRed: { backgroundColor: "rgba(239, 68, 68, 1)" },
  dotGray: { backgroundColor: "rgba(148, 163, 184, 1)" },

  main: { flex: 1, borderRadius: 22, backgroundColor: "rgba(18, 27, 47, 0.72)", borderWidth: 1, borderColor: "rgba(255,255,255,0.10)", overflow: "hidden" },
  mainHeader: { padding: 16, borderBottomWidth: 1, borderBottomColor: "rgba(255,255,255,0.08)", flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between", gap: 10 },
  hTitle: { color: "white", fontSize: 22, fontWeight: "900" },
  hDesc: { color: "rgba(255,255,255,0.65)", marginTop: 6, maxWidth: 680 },

  headerPill: { flexDirection: "row", alignItems: "center", gap: 8, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999, borderWidth: 1, borderColor: "rgba(255,255,255,0.10)", backgroundColor: "rgba(0,0,0,0.20)" },
  pillDot: { width: 10, height: 10, borderRadius: 99 },
  pillText: { color: "rgba(255,255,255,0.75)", fontSize: 12, fontWeight: "800" },

  mainBody: { padding: 16, paddingBottom: 18 },
  label: { color: "rgba(255,255,255,0.78)", fontWeight: "800", marginBottom: 8 },

  inputWrap: { position: "relative", borderRadius: 16, overflow: "hidden", borderWidth: 1, borderColor: "rgba(255,255,255,0.10)", backgroundColor: "rgba(0,0,0,0.18)" },
  input: { paddingHorizontal: 14, paddingVertical: 14, color: "white", fontSize: 16, fontFamily: "monospace" },
  spinner: { position: "absolute", right: 12, top: 0, bottom: 0, justifyContent: "center" },

  msg: { marginTop: 10, fontSize: 12, fontWeight: "800" },
  msgOk: { color: "rgba(34, 197, 94, 0.95)" },
  msgErr: { color: "rgba(239, 68, 68, 0.95)" },

  primaryBtn: { marginTop: 14, borderRadius: 16, paddingVertical: 14, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(37, 99, 235, 0.92)", borderWidth: 1, borderColor: "rgba(37, 99, 235, 0.65)" },
  primaryText: { color: "white", fontWeight: "900", fontSize: 15 },

  hint: { marginTop: 10, color: "rgba(255,255,255,0.40)", fontSize: 12, textAlign: "center", fontStyle: "italic" },

  divider: { marginTop: 16, marginBottom: 12, height: 1, backgroundColor: "rgba(255,255,255,0.08)" },

  goBtn: { borderRadius: 16, paddingVertical: 13, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(255,255,255,0.10)", borderWidth: 1, borderColor: "rgba(255,255,255,0.10)" },
  goBtnDisabled: { opacity: 0.45 },
  goBtnText: { color: "rgba(255,255,255,0.85)", fontWeight: "900" },

  smallNote: { marginTop: 10, color: "rgba(255,255,255,0.45)", fontSize: 11, textAlign: "center" },

  mainFooter: { padding: 10, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.08)", backgroundColor: "rgba(0,0,0,0.12)" },
  footerText: { color: "rgba(255,255,255,0.38)", fontSize: 11, textAlign: "center" },

  rotateWrap: { flex: 1, backgroundColor: "#0B1220", alignItems: "center", justifyContent: "center", padding: 22 },
  rotateTitle: { color: "white", fontSize: 20, fontWeight: "900" },
  rotateSub: { color: "rgba(255,255,255,0.65)", marginTop: 10, textAlign: "center", maxWidth: 420 },
});
