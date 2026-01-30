import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

type Props = { navigation: any; route: any };

function normalizeUrl(input: string) {
  let s = (input || "").trim();
  if (!s) return "";
  if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
  return s.replace(/\/+$/, "");
}

export default function ServerPCConnectionScreen({ navigation, route }: Props) {
  const jetsonBaseUrl: string = route?.params?.baseUrl || ""; // Jetson URL (mobile only connects to this)
  const nextRoute: string = route?.params?.nextRoute || "PickPlace"; // "PickPlace" or "Voice"

    useEffect(() => {
        const t = setTimeout(()=>{
          navigation.replace(nextRoute);
        }, 5000);
      
        return () => clearTimeout(t);
      }, [navigation])


  const [serverPcInput, setServerPcInput] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "err">("idle");
  const [statusText, setStatusText] = useState("");

  // Cancel in-flight request if user edits while connecting
  const abortRef = useRef<AbortController | null>(null);
  const reqIdRef = useRef(0);

  const serverPcUrl = useMemo(() => normalizeUrl(serverPcInput), [serverPcInput]);

  useEffect(() => {
    if (!connecting) return;
    abortRef.current?.abort();
    setConnecting(false);
    setStatusKind("idle");
    setStatusText("Connection attempt canceled (address changed).");
  }, [serverPcInput]); // eslint-disable-line react-hooks/exhaustive-deps

  const onConnect = async () => {
    if (!jetsonBaseUrl) {
      setStatusKind("err");
      setStatusText("Jetson connection is missing. Please go back and connect to Jetson first.");
      return;
    }
    if (!serverPcUrl) {
      setStatusKind("err");
      setStatusText("Please enter the Server PC address (IP:PORT).");
      return;
    }

    setConnecting(true);
    setStatusKind("idle");
    setStatusText("Jetson is connecting to Server PC...");

    const reqId = ++reqIdRef.current;
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const timeoutMs = 5000;
      const t = setTimeout(() => ac.abort(), timeoutMs);

      const res = await fetch(`${jetsonBaseUrl}/api/serverpc/connect`, {
        method: "POST",
        signal: ac.signal,
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ serverPcUrl }), // Jetson will store it dynamically
      });

      clearTimeout(t);
      if (reqId !== reqIdRef.current) return;

      const data = await res.json().catch(() => null);

      if (!res.ok || !data?.ok) {
        setConnecting(false);
        setStatusKind("err");
        setStatusText(data?.message || `Connection failed (HTTP ${res.status}).`);
        return;
      }

      setConnecting(false);
      setStatusKind("ok");
      setStatusText("Connected successfully.");

      // Go to intended mode page (PickPlace or Voice)
      navigation.replace(nextRoute, {
        baseUrl: jetsonBaseUrl,
        // optionally show connected server in UI:
        serverPcUrl: data?.serverPcUrl || serverPcUrl,
      });
    } catch (e: any) {
      if (reqId !== reqIdRef.current) return;
      setConnecting(false);
      setStatusKind("err");
      setStatusText(
        e?.name === "AbortError"
          ? "Connection timed out or was canceled."
          : "Request failed. Check Jetson connection and try again."
      );
    }
  };

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />

      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>Server PC Connection</Text>
          <Text style={styles.subtitle}>
            The app stays connected to Jetson. Jetson will connect to Server PC for {nextRoute === "Voice" ? "Voice" : "Pick & Place"}.
          </Text>

          <View style={styles.field}>
            <Text style={styles.label}>Server PC address</Text>
            <TextInput
              value={serverPcInput}
              onChangeText={setServerPcInput}
              placeholder="e.g. 192.168.0.50:8001"
              placeholderTextColor="rgba(255,255,255,0.35)"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              style={styles.input}
              returnKeyType="go"
              onSubmitEditing={onConnect}
            />
            <Text style={styles.hint}>You can change this anytime. Jetson will use this address for AI inference.</Text>
          </View>

          <Pressable
            onPress={onConnect}
            style={({ pressed }) => [styles.primaryBtn, pressed ? { opacity: 0.9 } : null]}
          >
            <Text style={styles.primaryText}>Connect via Jetson</Text>
            {connecting && (
              <View pointerEvents="none" style={styles.spinnerWrap}>
                <ActivityIndicator size="small" />
              </View>
            )}
          </Pressable>

          {!!statusText && (
            <View style={[styles.statusBox, statusKind === "ok" ? styles.statusOk : null, statusKind === "err" ? styles.statusErr : null]}>
              <Text style={styles.statusText}>{statusText}</Text>
            </View>
          )}

          <View style={styles.actions}>
            <Pressable
              onPress={() => navigation.replace("ModeSelect", { baseUrl: jetsonBaseUrl })}
              style={({ pressed }) => [styles.secondaryBtn, pressed ? { opacity: 0.85 } : null]}
            >
              <Text style={styles.secondaryText}>Back to Mode Selection</Text>
            </Pressable>
          </View>
        </View>

        <Text style={styles.footer}>UFACTORY xArm • Server PC Connection</Text>
      </KeyboardAvoidingView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(8, 12, 22, 0.68)" },
  container: { flex: 1, padding: 22, justifyContent: "center" },
  card: {
    borderRadius: 22,
    padding: 18,
    backgroundColor: "rgba(18, 27, 47, 0.78)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    maxWidth: 760,
    alignSelf: "center",
    width: "100%",
  },
  title: { color: "white", fontSize: 26, fontWeight: "800", letterSpacing: 0.2 },
  subtitle: { color: "rgba(255,255,255,0.70)", marginTop: 6, lineHeight: 18 },

  field: { marginTop: 16 },
  label: { color: "rgba(255,255,255,0.75)", fontWeight: "800", marginBottom: 8 },
  input: {
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: "rgba(0,0,0,0.22)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    color: "white",
    fontWeight: "700",
  },
  hint: { marginTop: 8, color: "rgba(255,255,255,0.45)", fontSize: 12, lineHeight: 16 },

  primaryBtn: {
    marginTop: 14,
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(43, 99, 255, 0.35)",
    borderWidth: 1,
    borderColor: "rgba(43, 99, 255, 0.55)",
    position: "relative",
  },
  primaryText: { color: "white", fontWeight: "900", letterSpacing: 0.2 },
  spinnerWrap: { position: "absolute", right: 14, top: 0, bottom: 0, justifyContent: "center" },

  statusBox: {
    marginTop: 12,
    borderRadius: 14,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "rgba(0,0,0,0.22)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  statusOk: { borderColor: "rgba(52, 211, 153, 0.45)" },
  statusErr: { borderColor: "rgba(248, 113, 113, 0.55)" },
  statusText: { color: "rgba(255,255,255,0.80)", fontWeight: "700", lineHeight: 18 },

  actions: { marginTop: 8, flexDirection: "row", justifyContent: "center" },
  secondaryBtn: {
    marginTop: 10,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.22)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  secondaryText: { color: "rgba(255,255,255,0.80)", fontWeight: "800" },

  footer: { marginTop: 14, textAlign: "center", color: "rgba(255,255,255,0.35)", fontSize: 12 },
});
