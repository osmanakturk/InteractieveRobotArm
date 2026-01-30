import { StyleSheet, Text, View } from 'react-native'
import React from 'react'

const PickPlaceScreen = () => {
  return (
    <View style={styles.container}>
      <Text>PickPlaceScreen</Text>
    </View>
  )
}

export default PickPlaceScreen

const styles = StyleSheet.create({
    container:{
    flex:1,
    justifyContent:'center',
    alignItems:'center'
  
  }
})