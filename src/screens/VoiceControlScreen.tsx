import { StyleSheet, Text, View } from 'react-native'
import React from 'react'

const VoiceControlScreen = () => {
  return (
    <View style={styles.container}>
      <Text>VoiceControlScreen</Text>
    </View>
  )
}

export default VoiceControlScreen

const styles = StyleSheet.create({
    container:{
    flex:1,
    justifyContent:'center',
    alignItems:'center'
  
  }
})