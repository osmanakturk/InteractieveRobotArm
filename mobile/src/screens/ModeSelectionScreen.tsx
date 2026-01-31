import React, { useMemo, useState } from "react";
import {
  Alert,
  Image,
  ImageBackground,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useFocusEffect } from "@react-navigation/native";
import { STORAGE_KEYS } from "../screens/SplashScreen"; 

type TileItem = {
  title: string;
  subtitle: string;
  routeName: "Manual" | "PickPlace" | "Voice";
  icon: any;
};

type ConnKey = keyof typeof STORAGE_KEYS;

export default function ModeSelectionScreen({ navigation, route }: any) {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const autoNavigateTo: TileItem["routeName"] | undefined = route?.params?.autoNavigateTo;

  const [conn, setConn] = useState({ gateway: "", robot: "", aiserver: "" });

  const loadConn = async () => {
    try {
      const [gw, rb, ai] = await Promise.all([
        AsyncStorage.getItem(STORAGE_KEYS.gateway),
        AsyncStorage.getItem(STORAGE_KEYS.robot),
        AsyncStorage.getItem(STORAGE_KEYS.aiserver),
      ]);
      setConn({
        gateway: (gw || "").trim(),
        robot: (rb || "").trim(),
        aiserver: (ai || "").trim(),
      });
    } catch {
      setConn({ gateway: "", robot: "", aiserver: "" });
    }
  };


  useFocusEffect(
    React.useCallback(() => {
      loadConn();
    }, [])
  );

  useFocusEffect(
    React.useCallback(() => {
      if (!autoNavigateTo) return;

      // paramı bir kere kullan: tekrar tekrar navigate etmesin
      navigation.setParams({ autoNavigateTo: undefined });

      // tekrar kontrol edip git
      setTimeout(() => {
        handleTilePress(autoNavigateTo);
      }, 0);
    }, [autoNavigateTo])
  );

  const { columns, tileW, tileH } = useMemo(() => {
    const cols = isLandscape ? 3 : 2;
    const screenPadding = 22 * 2;
    const cardPadding = 18 * 2;
    const usable = width - screenPadding - cardPadding;
    const gap = 10;
    const w = Math.floor((usable - gap * (cols - 1)) / cols);
    const tileWidth = Math.max(120, Math.min(w, 175));
    const tileHeight = Math.floor(tileWidth * 0.82);
    return { columns: cols, tileW: tileWidth, tileH: tileHeight };
  }, [width, isLandscape]);

  const tiles: TileItem[] = useMemo(
    () => [
      { title: "Manual Control", subtitle: "Direction buttons / jog", routeName: "Manual", icon: require("../../assets/manual.png") },
      { title: "Pick & Place Control", subtitle: "Touch to pick & place", routeName: "PickPlace", icon: require("../../assets/pickplace.png") },
      { title: "Voice Control", subtitle: "Speech commands", routeName: "Voice", icon: require("../../assets/voice.png") },
    ],
    []
  );

  const rows = useMemo(() => {
    const r: TileItem[][] = [];
    for (let i = 0; i < tiles.length; i += columns) r.push(tiles.slice(i, i + columns));
    return r;
  }, [tiles, columns]);

  const goToConnectionHub = (selected: ConnKey, nextRoute: TileItem["routeName"]) => {
    navigation.navigate("ConnectionHub", {
      selected,
      nextRoute,
      returnTo: "ModeSelect",
      gateway: conn.gateway,
      robot: conn.robot,
      aiserver: conn.aiserver,
    });
  };

  const showMissingAlert = (missing: ConnKey, nextRoute: TileItem["routeName"]) => {
    const titles: Record<ConnKey, string> = {
      gateway: "Gateway required",
      robot: "Robot required",
      aiserver: "AI Server required",
    };

    const messages: Record<ConnKey, string> = {
      gateway: "Please enter the Gateway address first. The app communicates only with the Gateway.",
      robot: "Please enter the Robot IP. The Gateway needs it to connect and control the xArm.",
      aiserver: "Pick & Place and Voice require an AI Server address (AI models).",
    };

    Alert.alert(titles[missing], messages[missing], [
      { text: "Cancel", style: "cancel" },
      { text: "OK", onPress: () => goToConnectionHub(missing, nextRoute) },
    ]);
  };

  /**
   * ✅ Mantıklı sıra:
   * 1) Gateway (her mod)
   * 2) Robot (her mod)
   * 3) AI Server (sadece PickPlace/Voice)
   */
  const handleTilePress = (routeName: TileItem["routeName"]) => {
    if (!conn.gateway) return showMissingAlert("gateway", routeName);
    if (!conn.robot) return showMissingAlert("robot", routeName);

    if ((routeName === "PickPlace" || routeName === "Voice") && !conn.aiserver) {
      return showMissingAlert("aiserver", routeName);
    }

    navigation.navigate(routeName, {
      gateway: conn.gateway,
      robot: conn.robot,
      aiserver: conn.aiserver,
    });
  };

  const Tile = ({ title, subtitle, routeName, icon }: TileItem) => (
    <Pressable
      onPress={() => handleTilePress(routeName)}
      style={({ pressed }) => [
        styles.tile,
        { width: tileW, height: tileH },
        pressed ? styles.pressed : null,
      ]}
    >
      <View style={styles.tileImageArea}>
        <Image source={icon} style={styles.tileImage} resizeMode="contain" />
      </View>

      <View style={styles.tileContent}>
        <Text style={styles.tileTitle} numberOfLines={1}>{title}</Text>
        <Text style={styles.tileSubtitle} numberOfLines={2}>{subtitle}</Text>
      </View>
    </Pressable>
  );

  return (
    <ImageBackground source={require("../../assets/splash.jpg")} style={styles.bg} resizeMode="cover">
      <View style={styles.dim} />
      <View style={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>Select Mode</Text>
          <Text style={styles.subtitle}>Choose how you want to control the xArm.</Text>

          {!!conn.gateway && (
            <View style={styles.badge}>
              <Text style={styles.badgeText} numberOfLines={1}>
                Gateway: {conn.gateway} • Robot: {conn.robot ? "OK" : "Missing"} • AI: {conn.aiserver ? "OK" : "Optional"}
              </Text>
            </View>
          )}

          <View style={styles.gridWrap}>
            {rows.map((row, idx) => (
              <View key={idx} style={styles.row}>
                {row.map((t) => (
                  <View key={t.routeName} style={styles.cell}>
                    <Tile {...t} />
                  </View>
                ))}
                {row.length < columns &&
                  Array.from({ length: columns - row.length }).map((_, i) => (
                    <View key={`empty-${idx}-${i}`} style={[styles.cell, { width: tileW }]} />
                  ))}
              </View>
            ))}
          </View>

          <View style={styles.actions}>
            <Pressable
              onPress={() => navigation.navigate("ConnectionHub", { selected: "gateway" })}
              style={({ pressed }) => [styles.secondaryBtn, pressed ? { opacity: 0.85 } : null]}
            >
              <Text style={styles.secondaryText}>Open Connection Hub</Text>
            </Pressable>
          </View>
        </View>

        <Text style={styles.footer}>UFACTORY xArm • Mode Selection</Text>
      </View>
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
  title: { color: "white", fontSize: 26, fontWeight: "800" },
  subtitle: { color: "rgba(255,255,255,0.70)", marginTop: 6 },

  badge: {
    marginTop: 12,
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    backgroundColor: "rgba(0,0,0,0.28)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    maxWidth: "100%",
  },
  badgeText: { color: "rgba(255,255,255,0.75)", fontSize: 12, fontWeight: "700" },

  gridWrap: { marginTop: 16 },
  row: { flexDirection: "row", justifyContent: "center", marginBottom: 10 },
  cell: { marginHorizontal: 5 },

  tile: {
    borderRadius: 16,
    backgroundColor: "rgba(0,0,0,0.22)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
    overflow: "hidden",
  },
  pressed: { opacity: 0.9, transform: [{ scale: 0.99 }] },

  tileImageArea: {
    flex: 1,
    padding: 0,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(5, 10, 22, 0.20)",
  },
  tileImage: { 
    width: "100%", 
    //height: "100%" 
  },

  tileContent: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(255,255,255,0.08)",
    backgroundColor: "rgba(5, 10, 22, 0.35)",
  },
  tileTitle: { color: "white", fontSize: 15, fontWeight: "900" },
  tileSubtitle: { marginTop: 4, color: "rgba(255,255,255,0.72)", fontSize: 12 },

  actions: { flexDirection: "row", marginTop: 8, justifyContent: "center" },
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
