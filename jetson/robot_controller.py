from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from config import CONFIG

try:
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

    gripper_available: bool = False
    gripper_pos: Optional[int] = None
    gripper_pct: Optional[int] = None
    gripper_min: int = CONFIG.gripper_min
    gripper_max: int = CONFIG.gripper_max

    message: str = ""


class RobotController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arm = None
        self._ip = ""
        self._frame = "base"  # base|tool
        self._speed_pct = CONFIG.default_speed_pct

        self._gripper_available = False
        self._gripper_pos: Optional[int] = None

    def connect(self, ip: str) -> Tuple[bool, str]:
        with self._lock:
            if XArmAPI is None:
                return False, "xArm SDK not installed (pip install xarm-python-sdk)."

            if self._arm is not None:
                try:
                    self._arm.disconnect()
                except Exception:
                    pass
                self._arm = None

            self._ip = ip.strip()
            if not self._ip:
                return False, "IP is empty."

            try:
                self._arm = XArmAPI(self._ip, is_radian=False)
                try:
                    self._arm.connect()
                except Exception:
                    pass

                # Safe defaults
                try:
                    self._arm.set_mode(0)
                except Exception:
                    pass
                try:
                    self._arm.set_state(0)
                except Exception:
                    pass

                # Gripper detect/enable
                self._gripper_available = True
                try:
                    if hasattr(self._arm, "set_gripper_enable"):
                        self._arm.set_gripper_enable(True)
                except Exception:
                    self._gripper_available = False

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
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                code = self._arm.motion_enable(True)
                if isinstance(code, int) and code != 0:
                    return False, f"motion_enable failed code={code}"
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
                if hasattr(self._arm, "motion_enable"):
                    try:
                        self._arm.motion_enable(False)
                    except Exception:
                        pass
                try:
                    self._arm.set_state(4)
                except Exception:
                    pass
                return True, "Disabled/Stopped."
            except Exception as e:
                return False, f"Disable failed: {e}"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                if hasattr(self._arm, "set_state"):
                    self._arm.set_state(4)
                return True, "Stopped."
            except Exception as e:
                return False, f"Stop failed: {e}"

    def go_home(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                if hasattr(self._arm, "move_gohome"):
                    code = self._arm.move_gohome(wait=False)
                    if isinstance(code, int) and code != 0:
                        return False, f"move_gohome failed code={code}"
                    return True, "Going home."
                return False, "Home not supported."
            except Exception as e:
                return False, f"Home failed: {e}"

    def set_frame(self, frame: str) -> Tuple[bool, str]:
        f = (frame or "").strip().lower()
        if f not in ("base", "tool"):
            return False, "frame must be base|tool"
        with self._lock:
            self._frame = f
        return True, "OK"

    def set_speed_pct(self, pct: int) -> Tuple[bool, str]:
        try:
            p = int(pct)
        except Exception:
            return False, "speed_pct must be int 1..100"
        if p < 1 or p > 100:
            return False, "speed_pct must be 1..100"
        with self._lock:
            self._speed_pct = p
        return True, "OK"

    def jog_delta(self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."

            speed = max(1, int(200 * (self._speed_pct / 100.0)))
            mvacc = max(1, int(2000 * (self._speed_pct / 100.0)))

            kwargs = dict(
                x=float(dx), y=float(dy), z=float(dz),
                roll=float(droll), pitch=float(dpitch), yaw=float(dyaw),
                relative=True, wait=False, speed=speed, mvacc=mvacc, is_radian=False,
            )

            kwargs["coordinate_mode"] = 1 if self._frame == "tool" else 0

            try:
                try:
                    code = self._arm.set_position(**kwargs)
                except TypeError:
                    kwargs.pop("coordinate_mode", None)
                    code = self._arm.set_position(**kwargs)

                if isinstance(code, int) and code != 0:
                    return False, f"set_position failed code={code}"
                return True, "OK"
            except Exception as e:
                return False, f"Jog failed: {e}"

    def _refresh_gripper_cache(self) -> None:
        if self._arm is None or not self._gripper_available:
            self._gripper_pos = None
            return
        try:
            if hasattr(self._arm, "get_gripper_position"):
                out = self._arm.get_gripper_position()
                if isinstance(out, tuple) and len(out) == 2:
                    code, pos = out
                    if code == 0:
                        self._gripper_pos = int(pos)
                else:
                    self._gripper_pos = int(out)
        except Exception:
            self._gripper_pos = None

    def gripper_jog(self, direction: str) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            if not self._gripper_available:
                return False, "Gripper not available."

            self._refresh_gripper_cache()
            cur = self._gripper_pos
            if cur is None:
                cur = (CONFIG.gripper_min + CONFIG.gripper_max) // 2

            step = CONFIG.gripper_step
            if direction == "open":
                target = min(CONFIG.gripper_max, cur + step)
            elif direction == "close":
                target = max(CONFIG.gripper_min, cur - step)
            else:
                return False, "action must be open|close"

            try:
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

    def gripper_status(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_gripper_cache()
            pos = self._gripper_pos
            pct = None
            if pos is not None:
                denom = max(1, CONFIG.gripper_max - CONFIG.gripper_min)
                pct = int(round((pos - CONFIG.gripper_min) * 100.0 / denom))
                pct = max(0, min(100, pct))
            return {
                "available": self._gripper_available,
                "pos": pos,
                "pct": pct,
                "min": CONFIG.gripper_min,
                "max": CONFIG.gripper_max,
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._arm is None:
                st = RobotStatus(connected=False, ip=self._ip, message="Disconnected.")
                return asdict(st)

            st = RobotStatus(connected=True, ip=self._ip, message="OK")
            try:
                st.state = getattr(self._arm, "state", None)
                st.mode = getattr(self._arm, "mode", None)
                st.error_code = getattr(self._arm, "error_code", None)
                st.warn_code = getattr(self._arm, "warn_code", None)
            except Exception:
                pass

            st.is_enabled = (st.state == 0 and (st.error_code in (0, None)))

            gs = self.gripper_status()
            st.gripper_available = bool(gs["available"])
            st.gripper_pos = gs["pos"]
            st.gripper_pct = gs["pct"]
            st.gripper_min = gs["min"]
            st.gripper_max = gs["max"]

            return asdict(st)
