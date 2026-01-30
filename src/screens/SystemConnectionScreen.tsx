import React, { useEffect, useRef, useState } from "react";
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

type Status = "idle" | "connecting" | "success" | "error";

function normalizeBaseUrl(input: string) {
  const trimmed = input.trim();
  if (!trimmed) return "";
  const hasScheme = /^https?:\/\//i.test(trimmed);
  const url = hasScheme ? trimmed : `http://${trimmed}`;
  return url.replace(/\/+$/, "");
}

export default function SystemConnectionScreen({ navigation }: any) {
  const [host, setHost] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");

  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  const isConnecting = status === "connecting";

  function cancel(reason?: string) {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (isConnecting) {
      setStatus("idle");
      setMessage(reason ?? "");
    }
  }

  // Input değişirse: bağlanma iptal
  useEffect(() => {
    if (isConnecting) cancel("Address changed — cancelled.");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host]);

  useEffect(() => () => cancel(), []);

  async function connect() {
    const baseUrl = normalizeBaseUrl(host);
    if (!baseUrl) {
      setStatus("error");
      setMessage("Please enter the Jetson address.");
      return;
    }

    cancel();
    const controller = new AbortController();
    abortRef.current = controller;
    const reqId = ++requestIdRef.current;

    setStatus("connecting");
    setMessage("Connecting to system...");

    try {
      const res = await fetch(`${baseUrl}/health`, {
        method: "GET",
        signal: controller.signal,
      });

      if (reqId !== requestIdRef.current) return;

      if (!res.ok) {
        setStatus("error");
        setMessage("System not reachable. Check IP/port and try again.");
        return;
      }

      const json = await res.json().catch(() => ({}));
      // Basit doğrulama (istersen backend health'e service alanı ekle)
      if (json?.ok === false) {
        setStatus("error");
        setMessage("System responded with an error.");
        return;
      }

      setStatus("success");
      setMessage("System connected ✓");

      setTimeout(() => {
        if (reqId === requestIdRef.current) {
          navigation.navigate("RobotConnect", { baseUrl });
        }
      }, 600);
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setStatus("error");
      setMessage("Connection failed. Make sure Jetson API is running.");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  return (
    <ImageBackground
      source={require("../../assets/splash.jpg")}
      style={styles.bg}
      resizeMode="cover"
    >
      <View style={styles.dim} />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <View style={styles.container}>
          <View style={styles.card}>
            <Text style={styles.title}>System Connection</Text>
            <Text style={styles.subtitle}>
              Connect to the Jetson gateway (robot + camera controller).
            </Text>

            <Text style={styles.label}>Jetson IP / URL</Text>
            <View style={styles.inputWrap}>
              <TextInput
                value={host}
                onChangeText={setHost}
                placeholder="e.g. 192.168.1.20:8000"
                placeholderTextColor="rgba(255,255,255,0.45)"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                returnKeyType="go"
                onSubmitEditing={connect}
                style={styles.input}
              />
            </View>

            <Pressable
              onPress={connect}
              style={({ pressed }) => [
                styles.button,
                pressed ? styles.buttonPressed : null,
              ]}
            >
              <Text style={styles.buttonText}>
                {isConnecting ? "Connecting" : "Connect"}
              </Text>

              {isConnecting && (
                <View pointerEvents="none" style={styles.spinnerOverlay}>
                  <ActivityIndicator />
                </View>
              )}
            </Pressable>

            {!!message && (
              <Text
                style={[
                  styles.message,
                  status === "success"
                    ? styles.msgSuccess
                    : status === "error"
                    ? styles.msgError
                    : styles.msgNeutral,
                ]}
              >
                {message}
              </Text>
            )}

            <Text style={styles.hint}>
              If you edit the address while connecting, the attempt is cancelled.
            </Text>
          </View>

          <Text style={styles.footer}>UFACTORY xArm • Jetson Gateway</Text>
        </View>
      </KeyboardAvoidingView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(8, 12, 22, 0.68)",
  },
  container: { flex: 1, padding: 22, justifyContent: "center" },
  card: {
    borderRadius: 22,
    padding: 18,
    backgroundColor: "rgba(18, 27, 47, 0.78)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  title: { color: "white", fontSize: 26, fontWeight: "800", letterSpacing: 0.2 },
  subtitle: { color: "rgba(255,255,255,0.70)", marginTop: 6, marginBottom: 18, lineHeight: 18 },
  label: { color: "rgba(255,255,255,0.72)", marginBottom: 8, fontSize: 13, fontWeight: "600" },
  inputWrap: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
    backgroundColor: "rgba(0,0,0,0.25)",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  input: { color: "white", fontSize: 16 },
  button: {
    marginTop: 14,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(43, 99, 255, 0.92)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
  },
  buttonPressed: { opacity: 0.86, transform: [{ scale: 0.99 }] },
  buttonText: { color: "white", fontSize: 16, fontWeight: "800", letterSpacing: 0.2 },
  spinnerOverlay: {
    position: "absolute",
    left: 0, right: 0, top: 0, bottom: 0,
    alignItems: "center",
    justifyContent: "center",
    opacity: 0.95,
  },
  message: { marginTop: 12, fontSize: 13.5, textAlign: "center" },
  msgSuccess: { color: "rgba(124, 255, 178, 0.95)" },
  msgError: { color: "rgba(255, 124, 124, 0.95)" },
  msgNeutral: { color: "rgba(255,255,255,0.70)" },
  hint: { marginTop: 10, fontSize: 12, color: "rgba(255,255,255,0.45)", textAlign: "center" },
  footer: { marginTop: 14, textAlign: "center", color: "rgba(255,255,255,0.35)", fontSize: 12 },
});
