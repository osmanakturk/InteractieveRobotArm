import { StyleSheet, Text, View, Image } from 'react-native'
import React, { use, useEffect } from 'react'



const SplashScreen = ({navigation}:any) => {
    useEffect(() => {
      const t = setTimeout(()=>{
        navigation.replace('SystemConnect');
      }, 1500);
    
      return () => clearTimeout(t);
    }, [navigation])
    
  return (
    <View style={styles.container}>
      <Image style={styles.image}
        source={require("../../assets/splash.jpg")}
        />
    </View>
  )
}

export default SplashScreen

const styles = StyleSheet.create({
    container:{
        flex:1,
        justifyContent:'center',
        alignItems:'center',
    
    },
    image:{
        flex:1,
        justifyContent:'center',
        width: '100%'
    
    }
})