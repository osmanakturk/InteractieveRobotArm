import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

export type ConnKey = "gateway" | "robot" | "aiserver";

export const STORAGE_KEYS = {
  gateway: "@conn/gateway",
  robot: "@conn/robot",
  aiserver: "@conn/aiserver",
} as const;

export type ConnStatus = "idle" | "connecting" | "connected" | "error";

export type ConnState = {
  value: string;
  status: ConnStatus;
  message?: string;
  lastOkAt?: number;
};

export type LiveStatusPayload = {
  ok?: boolean;
  status?: {
    connected?: boolean;
    is_enabled?: boolean;
    state?: number;
    error_code?: number;
    ip?: string;
    gripper_pct?: number;
  };
  ai_server?: {
    configured?: boolean;
    connected?: boolean;
    latency_ms?: number;
    error?: string;
  };
  camera?: { started?: boolean; last_error?: string };
  safety?: { limit_hit?: boolean; message?: string };
};

type ConnectionContextValue = {
  gateway: ConnState;
  robot: ConnState;
  aiserver: ConnState;

  live: LiveStatusPayload | null;
  wsConnected: boolean;

  setValue: (key: ConnKey, value: string) => Promise<void>;
  connect: (key: ConnKey) => Promise<void>;
  disconnect: (key: ConnKey) => Promise<void>;
};

const ConnectionContext = createContext<ConnectionContextValue | null>(null);

function normalizeBaseUrl(input: string) {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return `http://${raw}`;
  return raw;
}

function gatewayToWsUrl(gatewayHttpUrl: string) {
  const u = new URL(gatewayHttpUrl);
  const wsProto = u.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${u.host}/ws/status`;
}

export function ConnectionProvider({ children }: { children: React.ReactNode }) {
  const [gateway, setGateway] = useState<ConnState>({ value: "", status: "idle" });
  const [robot, setRobot] = useState<ConnState>({ value: "", status: "idle" });
  const [aiserver, setAiServer] = useState<ConnState>({ value: "", status: "idle" });

  const [live, setLive] = useState<LiveStatusPayload | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<any>(null);
  const reconnectAttemptsRef = useRef(0);

  const intentionalCloseRef = useRef(false);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const stopWs = (reason: "manual" | "restart" | "clear") => {
    clearReconnectTimer();
    reconnectAttemptsRef.current = 0;
    setWsConnected(false);

    intentionalCloseRef.current = true;
    try {
      wsRef.current?.close();
    } catch {}
    wsRef.current = null;

    setTimeout(() => {
      if (reason !== "manual") intentionalCloseRef.current = false;
    }, 0);
  };

  const scheduleReconnect = (baseHttp: string) => {
    const attempt = Math.min(10, reconnectAttemptsRef.current + 1);
    reconnectAttemptsRef.current = attempt;

    const backoffMs = Math.min(5000, 300 + attempt * 300);

    clearReconnectTimer();
    reconnectTimerRef.current = setTimeout(() => {
      startWs(baseHttp);
    }, backoffMs);
  };

  const applyLiveToConnStates = (data: LiveStatusPayload) => {
    setGateway((s) => ({
      ...s,
      status: "connected",
      message: s.status === "connected" ? s.message : undefined,
      lastOkAt: Date.now(),
    }));

    // Robot
    const st = data?.status || {};
    const robotConnected = !!st.connected;
    setRobot((s) => ({
      ...s,
      status: robotConnected ? "connected" : "idle",
      message: robotConnected ? "Robot connected (Gateway)." : s.value.trim() ? "Robot not connected." : undefined,
      lastOkAt: robotConnected ? Date.now() : s.lastOkAt,
    }));

    // AI
    const ai = data?.ai_server || {};
    const aiConfigured = !!ai.configured;
    const aiConnected = !!ai.connected;

    setAiServer((s) => ({
      ...s,
      status: aiConnected ? "connected" : aiConfigured ? "error" : "idle",
      message: aiConnected ? "AI Server OK." : aiConfigured ? ai.error || "AI Server FAIL." : undefined,
      lastOkAt: aiConnected ? Date.now() : s.lastOkAt,
    }));
  };

  const startWs = (baseHttp: string) => {
    const wsUrl = gatewayToWsUrl(baseHttp);

    stopWs("restart");
    setWsConnected(false);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        intentionalCloseRef.current = false; 
        setWsConnected(true);
        reconnectAttemptsRef.current = 0;

        setGateway((s) => ({ ...s, status: "connected", message: undefined, lastOkAt: Date.now() }));
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (!data || data.ok === false) return;

          setLive(data);
          applyLiveToConnStates(data);
        } catch {}
      };

      ws.onerror = () => {
        setWsConnected(false);
      };

      ws.onclose = () => {
        setWsConnected(false);

        if (intentionalCloseRef.current) return;

        scheduleReconnect(baseHttp);
      };
    } catch {
      scheduleReconnect(baseHttp);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [gw, rb, ai] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEYS.gateway),
          AsyncStorage.getItem(STORAGE_KEYS.robot),
          AsyncStorage.getItem(STORAGE_KEYS.aiserver),
        ]);

        const gwV = (gw || "").trim();
        const rbV = (rb || "").trim();
        const aiV = (ai || "").trim();

        if (gwV) setGateway((s) => ({ ...s, value: gwV }));
        if (rbV) setRobot((s) => ({ ...s, value: rbV }));
        if (aiV) setAiServer((s) => ({ ...s, value: aiV }));

        if (gwV) {
          const base = normalizeBaseUrl(gwV);
          startWs(base);
        }
      } catch {}
    })();

    return () => stopWs("manual");
  }, []);

  useEffect(() => {
    const gwRaw = gateway.value.trim();

    if (!gwRaw) {
      stopWs("clear");
      setLive(null);

      setGateway((s) => ({ ...s, status: "idle", message: undefined }));
      setRobot((s) => ({ ...s, status: "idle", message: undefined }));
      setAiServer((s) => ({ ...s, status: "idle", message: undefined }));
      return;
    }

    const base = normalizeBaseUrl(gwRaw);
    startWs(base);

  }, [gateway.value]);

  const setValue = async (key: ConnKey, value: string) => {
    const v = (value || "").trim();

    if (key === "gateway") setGateway((s) => ({ ...s, value: v }));
    if (key === "robot") setRobot((s) => ({ ...s, value: v }));
    if (key === "aiserver") setAiServer((s) => ({ ...s, value: v }));

    try {
      await AsyncStorage.setItem((STORAGE_KEYS as any)[key], v);
    } catch {}
  };

  const connect = async (key: ConnKey) => {
    if (key === "gateway") {
      const gw = normalizeBaseUrl(gateway.value.trim());
      if (!gw) throw new Error("Gateway required.");

      setGateway((s) => ({ ...s, status: "connecting", message: "Connecting..." }));

      const resp = await fetch(`${gw}/api/status`, { method: "GET" });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data?.ok === false) {
        setGateway((s) => ({
          ...s,
          status: "error",
          message: data?.message || `Gateway not reachable (HTTP ${resp.status}).`,
        }));
        return;
      }

      setGateway((s) => ({ ...s, status: "connected", message: "Gateway connected.", lastOkAt: Date.now() }));
      return;
    }

    const gw = normalizeBaseUrl(gateway.value.trim());
    if (!gw) throw new Error("Gateway required.");

    if (key === "robot") {
      const ip = robot.value.trim();
      if (!ip) throw new Error("Robot IP required.");

      setRobot((s) => ({ ...s, status: "connecting", message: "Connecting..." }));

      const resp = await fetch(`${gw}/api/robot/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ip }),
      });

      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data?.ok === false) {
        setRobot((s) => ({ ...s, status: "error", message: data?.message || `Robot connect failed (HTTP ${resp.status}).` }));
        return;
      }

      setRobot((s) => ({ ...s, status: "connected", message: "Robot connected via Gateway.", lastOkAt: Date.now() }));
      return;
    }

    if (key === "aiserver") {
      const url = normalizeBaseUrl(aiserver.value.trim());
      if (!url) throw new Error("AI Server URL required.");

      setAiServer((s) => ({ ...s, status: "connecting", message: "Connecting..." }));

      const resp = await fetch(`${gw}/api/ai_server/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data?.ok === false) {
        setAiServer((s) => ({ ...s, status: "error", message: data?.message || `AI connect failed (HTTP ${resp.status}).` }));
        return;
      }

      setAiServer((s) => ({ ...s, status: "connected", message: "AI Server configured via Gateway.", lastOkAt: Date.now() }));
      return;
    }
  };

  const disconnect = async (key: ConnKey) => {
    const gw = normalizeBaseUrl(gateway.value.trim());

    if (key === "gateway") {
      if (gw) {
        await fetch(`${gw}/api/disconnect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }).catch(() => {});
      }

      stopWs("manual");
      setLive(null);

      setGateway((s) => ({ ...s, status: "idle", message: "Gateway disconnected." }));
      setRobot((s) => ({ ...s, status: "idle", message: undefined }));
      setAiServer((s) => ({ ...s, status: "idle", message: undefined }));
      return;
    }

    if (!gw) throw new Error("Gateway required.");

    if (key === "robot") {
      await fetch(`${gw}/api/robot/disconnect`, { method: "POST" }).catch(() => {});
      setRobot((s) => ({ ...s, status: "idle", message: "Robot disconnected." }));
      return;
    }

    if (key === "aiserver") {
      await fetch(`${gw}/api/ai_server/disconnect`, { method: "POST" }).catch(() => {});
      setAiServer((s) => ({ ...s, status: "idle", message: "AI Server disconnected." }));
      return;
    }
  };

  const value = useMemo<ConnectionContextValue>(
    () => ({ gateway, robot, aiserver, live, wsConnected, setValue, connect, disconnect }),
    [gateway, robot, aiserver, live, wsConnected]
  );

  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

export function useConnection() {
  const ctx = useContext(ConnectionContext);
  if (!ctx) throw new Error("useConnection must be used inside <ConnectionProvider />");
  return ctx;
}



