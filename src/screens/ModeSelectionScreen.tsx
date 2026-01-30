import { StyleSheet, Text, View } from 'react-native'
import React from 'react'

const ModeSelectionScreen = () => {
  return (
    <View style={styles.container}>
      <Text>ModeSelectionScreen</Text>
    </View>
  )
}

export default ModeSelectionScreen

const styles = StyleSheet.create({
    container:{
      flex:1,
      justifyContent:'center',
      alignItems:'center',
  
  }
})