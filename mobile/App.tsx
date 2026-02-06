import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import SplashScreen from './src/screens/SplashScreen';
import ConnectionHubScreen from './src/screens/ConnectionHubScreen';
import RobotConnectionScreen from './src/screens/RobotConnectionScreen';
import SystemConnectionScreen from './src/screens/SystemConnectionScreen';
import ModeSelectionScreen from './src/screens/ModeSelectionScreen';
import ManualControlScreen from "./src/screens/ManualControlScreen";
import PickPlaceScreen from "./src/screens/PickPlaceScreen";
import VoiceControlScreen from "./src/screens/VoiceControlScreen";
import ServerPCConnectionScreen from "./src/screens/ServerPCConnectionScreen";
import TestScreen from "./src/screens/TestScreen";



const Stack = createNativeStackNavigator();



export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Test" screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Test" component={TestScreen} />
        <Stack.Screen name="Splash" component={SplashScreen} />
        <Stack.Screen name="ConnectionHub" component={ConnectionHubScreen} />
        <Stack.Screen name="SystemConnect" component={SystemConnectionScreen} />
        <Stack.Screen name="RobotConnect" component={RobotConnectionScreen} />
        <Stack.Screen name="ServerPCConnect" component={ServerPCConnectionScreen} />
        <Stack.Screen name="ModeSelect" component={ModeSelectionScreen} />
        <Stack.Screen name="Manual" component={ManualControlScreen} />
        <Stack.Screen name="PickPlace" component={PickPlaceScreen} />
        <Stack.Screen name="Voice" component={VoiceControlScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
