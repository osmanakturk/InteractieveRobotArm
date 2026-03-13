// src/screens/SplashScreen.tsx
import React, { useEffect, useRef } from "react";
import { Image, StyleSheet, View, Text } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

export const STORAGE_KEYS = {
  gateway: "@conn/gateway",
  robot: "@conn/robot",
  aiserver: "@conn/aiserver",
} as const;

export default function SplashScreen({ navigation }: any) {
  const navigatedRef = useRef(false);

  useEffect(() => {
    let alive = true;

    const boot = async () => {
      try {
        await AsyncStorage.multiRemove([
          STORAGE_KEYS.gateway,
          STORAGE_KEYS.robot,
          STORAGE_KEYS.aiserver,
        ]);
      } catch {
        // silent
      }

      await new Promise((r) => setTimeout(r, 1200));

      if (!alive) return;
      if (navigatedRef.current) return;
      navigatedRef.current = true;

      navigation.replace("ConnectionHub", { selected: "gateway" });
    };

    boot();

    return () => {
      alive = false;
    };
  }, [navigation]);

  return (
    <View style={styles.container}>
      <Image
        source={require("../../assets/splash.jpg")}
        style={styles.image}
        resizeMode="cover"
      />

      <Text style={styles.signature}>
        Developed by Osman Akturk
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "black",
  },

  image: {
    flex: 1,
    width: "100%",
  },

  signature: {
    position: "absolute",
    bottom: 20,
    alignSelf: "center",
    color: "black",
    fontSize: 16,
    opacity: 0.8,
    letterSpacing: 1
  },
});