from xarm.wrapper import XArmAPI
import time

ROBOT_IP = "172.28.96.1"

def main():
    arm = XArmAPI(ROBOT_IP)
    arm.connect()

   
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)   
    arm.set_state(0) 
    time.sleep(0.5)

    
    code, joints = arm.get_servo_angle(is_radian=False)
    if code != 0:
        print("Failed to read joints, code:", code)
        return
    print("Current joints (deg):", joints)

    
    target = joints.copy()
    target[0] += 10  
    target[1] -= 5   

   
    code = arm.set_servo_angle(angle=target, speed=20, mvacc=200, is_radian=False, wait=True)
    print("Move result code:", code)

    arm.disconnect()

if __name__ == "__main__":
    main()
