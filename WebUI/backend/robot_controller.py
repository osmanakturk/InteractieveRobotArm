# backend/robot_controller.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

from config import CONFIG

try:
    # pip install xarm-python-sdk
    from xarm.wrapper import XArmAPI  # type: ignore
except Exception:
    XArmAPI = None  # type: ignore


@dataclass
class RobotStatus:
    connected: bool
    ip: str = ""
    is_enabled: bool = False
    state: Optional[int] = None
    mode: Optional[int] = None
    error_code: Optional[int] = None
    warn_code: Optional[int] = None
    pose: Optional[list] = None          # [x,y,z,roll,pitch,yaw]
    joints: Optional[list] = None        # [j1..j6] in degrees
    tcp_speed: Optional[float] = None
    gripper_available: bool = False
    gripper_pos: Optional[int] = None
    gripper_min: int = CONFIG.gripper_min
    gripper_max: int = CONFIG.gripper_max
    message: str = ""


class RobotController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arm = None
        self._ip = ""
        self._frame = "base"  # "base" or "tool"
        self._speed_pct = CONFIG.default_speed_pct

        # gripper state cache
        self._gripper_available = False
        self._gripper_pos = None

    # -------------------------
    # Connection & state
    # -------------------------
    def connect(self, ip: str) -> Tuple[bool, str]:
        with self._lock:
            if XArmAPI is None:
                return False, "xArm SDK not installed. Install with: pip install xarm-python-sdk"

            if self._arm is not None:
                try:
                    self._arm.disconnect()
                except Exception:
                    pass
                self._arm = None

            self._ip = ip.strip()
            if not self._ip:
                return False, "IP address is empty."

            try:
                self._arm = XArmAPI(self._ip, is_radian=False)
                # some SDK versions require connect() explicitly
                try:
                    self._arm.connect()
                except Exception:
                    pass

                # Basic init sequence
                # NOTE: On some controllers, calling motion_enable may fail if not in correct state.
                # We'll expose Enable button in UI. Here we only set safe defaults.
                try:
                    self._arm.set_mode(0)   # 0: position mode (common)
                except Exception:
                    pass
                try:
                    self._arm.set_state(0)  # 0: ready
                except Exception:
                    pass

                # detect gripper support (won't error until used on some versions)
                self._gripper_available = True
                try:
                    # If supported, enables gripper; if not, it may raise or return nonzero.
                    if hasattr(self._arm, "set_gripper_enable"):
                        self._arm.set_gripper_enable(True)
                except Exception:
                    self._gripper_available = False

                # initial cache
                self._refresh_gripper_cache()
                return True, "Connected."
            except Exception as e:
                self._arm = None
                return False, f"Connect failed: {e}"

    def disconnect(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                self._ip = ""
                return True, "Already disconnected."
            try:
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None
            self._ip = ""
            self._gripper_available = False
            self._gripper_pos = None
            return True, "Disconnected."

    def enable(self) -> Tuple[bool, str]:
        """Enable motion (equivalent to 'Enable' in xArm Studio)."""
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                # Typical sequence:
                # 1) motion_enable(True)
                # 2) set_mode(0)
                # 3) set_state(0)
                code = self._arm.motion_enable(True)
                if code != 0:
                    return False, f"SDK call failed: motion_enable code={code}"
                self._arm.set_mode(0)
                self._arm.set_state(0)
                return True, "Enabled."
            except Exception as e:
                return False, f"Enable failed: {e}"

    def disable(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                # Some SDKs accept motion_enable(False). If not, fall back to set_state(4).
                if hasattr(self._arm, "motion_enable"):
                    try:
                        self._arm.motion_enable(False)
                    except Exception:
                        pass
                try:
                    self._arm.set_state(4)  # 4: stop
                except Exception:
                    pass
                return True, "Disabled/Stopped."
            except Exception as e:
                return False, f"Disable failed: {e}"

    def stop(self) -> Tuple[bool, str]:
        """Soft stop: stop motion and set state to STOP."""
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                # Different SDK versions offer different stop calls
                if hasattr(self._arm, "set_state"):
                    self._arm.set_state(4)  # STOP
                if hasattr(self._arm, "stop_lite"):
                    try:
                        self._arm.stop_lite()
                    except Exception:
                        pass
                if hasattr(self._arm, "emergency_stop"):
                    # DON'T call real emergency stop by default; it may require manual reset.
                    pass
                return True, "Stopped."
            except Exception as e:
                return False, f"Stop failed: {e}"

    def go_home(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                # Many xArm configs expose move_gohome()
                if hasattr(self._arm, "move_gohome"):
                    code = self._arm.move_gohome(wait=False)
                    if code != 0:
                        return False, f"move_gohome failed code={code}"
                    return True, "Going home."
                # fallback: set_servo_angle to zeros (may be unsafe depending on mounting)
                return False, "Home not supported on this SDK/robot config."
            except Exception as e:
                return False, f"Home failed: {e}"

    # -------------------------
    # Jog frame & speed
    # -------------------------
    def set_frame(self, frame: str) -> Tuple[bool, str]:
        frame = (frame or "").strip().lower()
        if frame not in ("base", "tool"):
            return False, "Frame must be 'base' or 'tool'."
        with self._lock:
            self._frame = frame
        return True, "OK"

    def set_speed_pct(self, pct: int) -> Tuple[bool, str]:
        pct = int(pct)
        if pct < 1 or pct > 100:
            return False, "Speed must be 1..100."
        with self._lock:
            self._speed_pct = pct
        return True, "OK"

    # -------------------------
    # Jogging (hold-to-move)
    # -------------------------
    def jog_delta(self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0) -> Tuple[bool, str]:
        """Relative incremental move. Called repeatedly while button is held."""
        with self._lock:
            if self._arm is None:
                return False, "Not connected."

            # Speed mapping: keep conservative by default
            speed = max(1, int(200 * (self._speed_pct / 100.0)))  # 1..200
            mvacc = max(1, int(2000 * (self._speed_pct / 100.0))) # 1..2000

            kwargs = dict(
                x=dx, y=dy, z=dz, roll=droll, pitch=dpitch, yaw=dyaw,
                relative=True, wait=False, speed=speed, mvacc=mvacc, is_radian=False,
            )

            # Coordinate frame mapping:
            # Many SDK versions support coordinate_mode: 0 (base), 1 (tool).
            # If not supported, we fall back to base.
            if self._frame == "tool":
                kwargs["coordinate_mode"] = 1
            else:
                kwargs["coordinate_mode"] = 0

            try:
                # Try with coordinate_mode first
                try:
                    code = self._arm.set_position(**kwargs)
                except TypeError:
                    # Older SDK without coordinate_mode
                    kwargs.pop("coordinate_mode", None)
                    code = self._arm.set_position(**kwargs)

                if code != 0:
                    return False, f"set_position failed code={code}"
                return True, "OK"
            except Exception as e:
                return False, f"Jog failed: {e}"

    # -------------------------
    # Gripper (hold-to-open/close)
    # -------------------------
    def _refresh_gripper_cache(self) -> None:
        if self._arm is None or not self._gripper_available:
            self._gripper_pos = None
            return
        try:
            if hasattr(self._arm, "get_gripper_position"):
                code, pos = self._arm.get_gripper_position()
                if code == 0:
                    self._gripper_pos = int(pos)
        except Exception:
            # Some versions: get_gripper_position returns pos only
            try:
                pos = self._arm.get_gripper_position()
                self._gripper_pos = int(pos)
            except Exception:
                self._gripper_pos = None

    def gripper_jog(self, direction: str) -> Tuple[bool, str]:
        """direction: 'open' or 'close'"""
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            if not self._gripper_available:
                return False, "Gripper not available (simulation or hardware missing)."

            self._refresh_gripper_cache()
            cur = self._gripper_pos
            if cur is None:
                # start from mid as fallback
                cur = (CONFIG.gripper_min + CONFIG.gripper_max) // 2

            step = CONFIG.gripper_step
            if direction == "open":
                target = min(CONFIG.gripper_max, cur + step)
            elif direction == "close":
                target = max(CONFIG.gripper_min, cur - step)
            else:
                return False, "direction must be open|close"

            try:
                # set_gripper_position(pos, wait=False, speed=??)
                if hasattr(self._arm, "set_gripper_position"):
                    code = self._arm.set_gripper_position(target, wait=False)
                    if isinstance(code, int) and code != 0:
                        return False, f"set_gripper_position failed code={code}"
                else:
                    return False, "SDK has no set_gripper_position."
                self._gripper_pos = target
                return True, "OK"
            except Exception as e:
                return False, f"Gripper failed: {e}"

    # -------------------------
    # Telemetry
    # -------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._arm is None:
                return asdict(RobotStatus(connected=False, ip=self._ip, message="Disconnected."))

            st = RobotStatus(connected=True, ip=self._ip, message="OK")
            try:
                st.state = getattr(self._arm, "state", None)
                st.mode = getattr(self._arm, "mode", None)
                st.error_code = getattr(self._arm, "error_code", None)
                st.warn_code = getattr(self._arm, "warn_code", None)
            except Exception:
                pass

            # Determine "enabled" heuristically
            st.is_enabled = (st.state == 0 and (st.error_code in (0, None)))

            try:
                code, pose = self._arm.get_position(is_radian=False)
                if code == 0:
                    st.pose = [round(float(v), 2) for v in pose]
            except Exception:
                # older versions
                try:
                    pose = self._arm.get_position(is_radian=False)
                    st.pose = [round(float(v), 2) for v in pose]
                except Exception:
                    pass

            try:
                code, angles = self._arm.get_servo_angle(is_radian=False)
                if code == 0:
                    st.joints = [round(float(v), 2) for v in angles]
            except Exception:
                try:
                    angles = self._arm.get_servo_angle(is_radian=False)
                    st.joints = [round(float(v), 2) for v in angles]
                except Exception:
                    pass

            try:
                if hasattr(self._arm, "get_tcp_speed"):
                    code, sp = self._arm.get_tcp_speed()
                    if code == 0:
                        st.tcp_speed = float(sp)
            except Exception:
                pass

            st.gripper_available = self._gripper_available
            self._refresh_gripper_cache()
            st.gripper_pos = self._gripper_pos
            st.gripper_min = CONFIG.gripper_min
            st.gripper_max = CONFIG.gripper_max

            return asdict(st)
