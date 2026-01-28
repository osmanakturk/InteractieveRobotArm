from xarm.wrapper import XArmAPI
import time

ROBOT_IP =  "172.28.96.1"

STEP_MM = 10
STEP_DEG = 2

def main():
    arm = XArmAPI(ROBOT_IP)
    arm.connect()

    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print("Controls:")
    print("  w/s: +Y/-Y")
    print("  a/d: -X/+X")
    print("  r/f: +Z/-Z")
    print("  q/e: yaw -/+")
    print("  x: exit")

    while True:
        cmd = input("cmd> ").strip().lower()
        if cmd == "x":
            break

        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            print("Failed to read pose, code:", code)
            continue

        x, y, z, roll, pitch, yaw = pose

        if cmd == "w":
            y += STEP_MM
        elif cmd == "s":
            y -= STEP_MM
        elif cmd == "a":
            x -= STEP_MM
        elif cmd == "d":
            x += STEP_MM
        elif cmd == "r":
            z += STEP_MM
        elif cmd == "f":
            z -= STEP_MM
        elif cmd == "q":
            yaw -= STEP_DEG
        elif cmd == "e":
            yaw += STEP_DEG
        else:
            print("Unknown command")
            continue
        
        

        code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                speed=80, mvacc=500, wait=True, is_radian=False)
        print("Move code:", code)

    arm.disconnect()

if __name__ == "__main__":
    main()
