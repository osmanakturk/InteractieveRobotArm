import React, { useCallback, useEffect, useRef } from "react";
import { AppState, AppStateStatus, Platform } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import * as NavigationBar from "expo-navigation-bar";

import SplashScreen from "./src/screens/SplashScreen";
import ConnectionHubScreen from "./src/screens/ConnectionHubScreen";
import ModeSelectionScreen from "./src/screens/ModeSelectionScreen";
import ManualControlScreen from "./src/screens/ManualControlScreen";
import PickPlaceScreen from "./src/screens/PickPlaceScreen";
import VoiceControlScreen from "./src/screens/VoiceControlScreen";
import { ConnectionProvider } from "./src/connection/ConnectionContext";

const Stack = createNativeStackNavigator();

export default function App() {
  const appState = useRef<AppStateStatus>(AppState.currentState);

  const hideNavBar = useCallback(async () => {
    if (Platform.OS !== "android") return;

    try {
      await NavigationBar.setVisibilityAsync("hidden");
    } catch (error) {
      console.warn("Failed to hide navigation bar:", error);
    }
  }, []);

  useEffect(() => {
    hideNavBar();

    const sub = AppState.addEventListener("change", async (nextState) => {
      const wasBackground =
        appState.current === "background" || appState.current === "inactive";

      appState.current = nextState;

      if (wasBackground && nextState === "active") {
        await hideNavBar();
      }
    });

    return () => {
      sub.remove();
    };
  }, [hideNavBar]);

  return (
    <ConnectionProvider>
      <NavigationContainer
        onReady={hideNavBar}
        onStateChange={hideNavBar}
      >
        <Stack.Navigator
          initialRouteName="Splash"
          screenOptions={{ headerShown: false }}
        >
          <Stack.Screen name="Splash" component={SplashScreen} />
          <Stack.Screen name="ConnectionHub" component={ConnectionHubScreen} />
          <Stack.Screen name="ModeSelect" component={ModeSelectionScreen} />
          <Stack.Screen name="Manual" component={ManualControlScreen} />
          <Stack.Screen name="PickPlace" component={PickPlaceScreen} />
          <Stack.Screen name="Voice" component={VoiceControlScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </ConnectionProvider>
  );
}