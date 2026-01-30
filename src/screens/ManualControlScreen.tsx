import { StyleSheet, Text, View } from 'react-native'
import React from 'react'

const ManualControlScreen = () => {
  return (
    <View style={styles.container}>
      <Text>ManualControlScreen</Text>
    </View>
  )
}

export default ManualControlScreen

const styles = StyleSheet.create({
  container:{
    flex:1,
    justifyContent:'center',
    alignItems:'center'
  
  }
})