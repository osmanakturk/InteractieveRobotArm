import React, { useMemo } from "react";
import {
  Image,
  ImageBackground,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";

type TileItem = {
  title: string;
  subtitle: string;
  routeName: string;
  icon: any;
};

export default function ModeSelectionScreen({ navigation, route }: any) {
  const baseUrl: string = route?.params?.baseUrl || "";
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const { columns, tileW, tileH } = useMemo(() => {
    const cols = isLandscape ? 3 : 2;

    const screenPadding = 22 * 2;
    const cardPadding = 18 * 2;
    const usable = width - screenPadding - cardPadding;

    const gap = 10;
    const w = Math.floor((usable - gap * (cols - 1)) / cols);

    const tileWidth = Math.max(120, Math.min(w, 175));
    const tileHeight = Math.floor(tileWidth * 0.78);

    return { columns: cols, tileW: tileWidth, tileH: tileHeight };
  }, [width, isLandscape]);

  const tiles: TileItem[] = useMemo(
    () => [
      {
        title: "Manual Control",
        subtitle: "Direction buttons / jog",
        routeName: "Manual",
        icon: require("../../assets/manual.png"),
      },
      {
        title: "Pick & Place Control",
        subtitle: "Touch to pick & place",
        routeName: "PickPlace",
        icon: require("../../assets/pickplace.png"),
      },
      {
        title: "Voice Control",
        subtitle: "Speech commands",
        routeName: "Voice",
        icon: require("../../assets/voice.png"),
      },
    ],
    []
  );

  const rows = useMemo(() => {
    const r: TileItem[][] = [];
    for (let i = 0; i < tiles.length; i += columns) {
      r.push(tiles.slice(i, i + columns));
    }
    return r;
  }, [tiles, columns]);

const Tile = ({ title, subtitle, routeName, icon }: TileItem) => (
  <Pressable
    onPress={() => navigation.navigate(routeName, { baseUrl })}
    style={({ pressed }) => [
      styles.tile,
      { width: tileW, height: tileH },
      pressed ? styles.pressed : null,
    ]}
  >
    {/* Image area (fixed height) */}
    <View style={styles.tileImageArea}>
      <Image source={icon} style={styles.tileImage} resizeMode="cover" />
    </View>

    {/* Text area */}
    <View style={styles.tileContent}>
      <Text style={styles.tileTitle} numberOfLines={1}>
        {title}
      </Text>
      <Text style={styles.tileSubtitle} numberOfLines={2}>
        {subtitle}
      </Text>
    </View>
  </Pressable>
);


  return (
    <ImageBackground
      source={require("../../assets/splash.jpg")}
      style={styles.bg}
      resizeMode="cover"
    >
      <View style={styles.dim} />

      <View style={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>Select Mode</Text>
          <Text style={styles.subtitle}>
            Choose how you want to control the xArm.
          </Text>

          {!!baseUrl && (
            <View style={styles.badge}>
              <Text style={styles.badgeText} numberOfLines={1}>
                Connected to: {baseUrl}
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
                    <View
                      key={`empty-${idx}-${i}`}
                      style={[styles.cell, { width: tileW }]}
                    />
                  ))}
              </View>
            ))}
          </View>

          <View style={styles.actions}>


            <Pressable
              onPress={() =>
                navigation.reset({
                  index: 0,
                  routes: [{ name: "SystemConnect" }],
                })
              }
              style={({ pressed }) => [
                styles.secondaryBtn,
                pressed ? { opacity: 0.85 } : null,
              ]}
            >
              <Text style={styles.secondaryText}>Back to System Connection</Text>
            </Pressable>

                        <Pressable
              onPress={() => navigation.navigate("RobotConnect", { baseUrl })}
              style={({ pressed }) => [
                styles.secondaryBtn,
                pressed ? { opacity: 0.85 } : null,
              ]}
            >
              <Text style={styles.secondaryText}>Back to Robot Connection</Text>
            </Pressable>
          </View>
        </View>

        <Text style={styles.footer}>UFACTORY xArm • Mode Selection</Text>
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  bg: { flex: 1 },
  dim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(8, 12, 22, 0.68)",
  },
  container: {
    flex: 1,
    padding: 22,
    justifyContent: "center",
  },
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

  title: {
    color: "white",
    fontSize: 26,
    fontWeight: "800",
    letterSpacing: 0.2,
  },
  subtitle: {
    color: "rgba(255,255,255,0.70)",
    marginTop: 6,
    lineHeight: 18,
  },

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
  badgeText: {
    color: "rgba(255,255,255,0.75)",
    fontSize: 12,
    fontWeight: "700",
  },

  gridWrap: {
    marginTop: 16,
  },
  row: {
    flexDirection: "row",
    justifyContent: "center",
    marginBottom: 10,
  },
  cell: {
    marginHorizontal: 5,
  },

tile: {
  borderRadius: 16,
  backgroundColor: "rgba(0,0,0,0.22)",
  borderWidth: 1,
  borderColor: "rgba(255,255,255,0.10)",
  overflow: "hidden",
},

  pressed: {
    opacity: 0.9,
    transform: [{ scale: 0.99 }],
  },
tileImageArea: {
  flex: 1,
  padding: 10,
  justifyContent: "center",
  alignItems: "center",
  backgroundColor: "rgba(5, 10, 22, 0.20)", // hafif kontrast
},

tileImage: {
  width: "100%",
  height: "100%",
},
tileContent: {
  paddingHorizontal: 12,
  paddingVertical: 10,
  borderTopWidth: 1,
  borderTopColor: "rgba(255,255,255,0.08)",
  backgroundColor: "rgba(5, 10, 22, 0.35)",
},


  tileTitle: {
    color: "white",
    fontSize: 15,
    fontWeight: "900",
    letterSpacing: 0.1,
  },
  tileSubtitle: {
    marginTop: 4,
    color: "rgba(255,255,255,0.72)",
    fontSize: 12,
    lineHeight: 15,
  },

  actions: {
    flexDirection: "row",
    marginTop: 8,
    justifyContent: "space-evenly",
    flexWrap: "wrap",
    gap: 10,
  },
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
  secondaryText: {
    color: "rgba(255,255,255,0.80)",
    fontWeight: "800",
  },

  footer: {
    marginTop: 14,
    textAlign: "center",
    color: "rgba(255,255,255,0.35)",
    fontSize: 12,
  },
});
