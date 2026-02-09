# jetson/robot_controller.py
from __future__ import annotations

import threading
import time
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

    safety_limit_hit: bool = False
    safety_message: str = ""

    message: str = ""


class RobotController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arm: Optional["XArmAPI"] = None
        self._ip = ""
        self._frame = "base"  # base|tool
        self._speed_pct = CONFIG.default_speed_pct

        self._gripper_available = False
        self._gripper_pos: Optional[int] = None

        self._safety_limit_hit: bool = False
        self._safety_message: str = ""

        # server-side jog throttle
        self._last_jog_ts: float = 0.0

        # authoritative enabled flag (we control it)
        self._enabled_flag: bool = False

    # -------------------------
    # Low-level hardware snapshot (NO recursion)
    # -------------------------
    def _read_hw(self) -> Dict[str, Optional[int]]:
        if self._arm is None:
            return {"state": None, "mode": None, "error_code": None, "warn_code": None}
        try:
            return {
                "state": getattr(self._arm, "state", None),
                "mode": getattr(self._arm, "mode", None),
                "error_code": getattr(self._arm, "error_code", None),
                "warn_code": getattr(self._arm, "warn_code", None),
            }
        except Exception:
            return {"state": None, "mode": None, "error_code": None, "warn_code": None}

    # -------------------------
    # Safety helpers
    # -------------------------
    def clear_safety(self) -> None:
        with self._lock:
            self._safety_limit_hit = False
            self._safety_message = ""

    def safety_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "limit_hit": self._safety_limit_hit,
                "message": self._safety_message,
                "limits": {
                    "x_min": CONFIG.safety_x_min_mm,
                    "x_max": CONFIG.safety_x_max_mm,
                    "y_min": CONFIG.safety_y_min_mm,
                    "y_max": CONFIG.safety_y_max_mm,
                    "z_min": CONFIG.safety_z_min_mm,
                    "z_max": CONFIG.safety_z_max_mm,
                    "roll_min": CONFIG.safety_roll_min_deg,
                    "roll_max": CONFIG.safety_roll_max_deg,
                    "pitch_min": CONFIG.safety_pitch_min_deg,
                    "pitch_max": CONFIG.safety_pitch_max_deg,
                    "yaw_min": CONFIG.safety_yaw_min_deg,
                    "yaw_max": CONFIG.safety_yaw_max_deg,
                },
            }

    def _set_safety_hit(self, msg: str) -> None:
        self._safety_limit_hit = True
        self._safety_message = msg

    @staticmethod
    def _in_range(val: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    def _get_pose(self) -> Optional[list]:
        if self._arm is None:
            return None
        try:
            out = self._arm.get_position(is_radian=False)
            if isinstance(out, tuple) and len(out) == 2:
                code, pose = out
                if code == 0:
                    return [float(v) for v in pose]
            else:
                return [float(v) for v in out]
        except Exception:
            return None

    def _check_limits(self, target_pose: list) -> Tuple[bool, str]:
        x, y, z, roll, pitch, yaw = target_pose[:6]

        # only limited if that min/max is set
        if not self._in_range(x, CONFIG.safety_x_min_mm, CONFIG.safety_x_max_mm):
            return False, "Safety limit reached on X axis."
        if not self._in_range(y, CONFIG.safety_y_min_mm, CONFIG.safety_y_max_mm):
            return False, "Safety limit reached on Y axis."
        if not self._in_range(z, CONFIG.safety_z_min_mm, CONFIG.safety_z_max_mm):
            return False, "Safety limit reached on Z axis."

        if not self._in_range(roll, CONFIG.safety_roll_min_deg, CONFIG.safety_roll_max_deg):
            return False, "Safety limit reached on Roll."
        if not self._in_range(pitch, CONFIG.safety_pitch_min_deg, CONFIG.safety_pitch_max_deg):
            return False, "Safety limit reached on Pitch."
        if not self._in_range(yaw, CONFIG.safety_yaw_min_deg, CONFIG.safety_yaw_max_deg):
            return False, "Safety limit reached on Yaw."

        return True, "OK"

    def _force_disable_for_safety(self, reason: str) -> None:
        if self._arm is None:
            return

        try:
            if hasattr(self._arm, "motion_enable"):
                try:
                    self._arm.motion_enable(False)
                except Exception:
                    pass
            try:
                self._arm.set_state(4)  # STOP
            except Exception:
                pass
        except Exception:
            pass

        self._enabled_flag = False
        self._set_safety_hit(reason + " Robot was disabled for safety. Press Enable again to continue.")

    # -------------------------
    # Motion gate (NO recursion!)
    # -------------------------
    def _is_motion_allowed(self) -> Tuple[bool, str]:
        if self._arm is None:
            return False, "Not connected."

        hw = self._read_hw()
        err = hw.get("error_code")
        if err not in (0, None):
            return False, f"Robot error_code={err}. Clear error on robot/controller first."

        if self._safety_limit_hit:
            return False, self._safety_message or "Safety limit hit. Press Enable to continue."

        if not self._enabled_flag:
            return False, "Robot is disabled. Press Enable to continue."

        return True, "OK"

    # -------------------------
    # Connection
    # -------------------------
    def connect(self, ip: str) -> Tuple[bool, str]:
        with self._lock:
            if XArmAPI is None:
                return False, "xArm SDK not installed (pip install xarm-python-sdk)."

            # reset existing
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

                # safe defaults (DO NOT auto-enable)
                try:
                    self._arm.set_mode(0)
                except Exception:
                    pass
                try:
                    self._arm.set_state(0)
                except Exception:
                    pass

                # gripper enable attempt
                self._gripper_available = True
                try:
                    if hasattr(self._arm, "set_gripper_enable"):
                        self._arm.set_gripper_enable(True)
                except Exception:
                    self._gripper_available = False

                self._refresh_gripper_cache()
                self.clear_safety()

                self._enabled_flag = False  # must press Enable explicitly
                return True, "Connected."
            except Exception as e:
                self._arm = None
                self._enabled_flag = False
                return False, f"Connect failed: {e}"

    def disconnect(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                self._ip = ""
                self._enabled_flag = False
                return True, "Already disconnected."
            try:
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None
            self._ip = ""
            self._gripper_available = False
            self._gripper_pos = None
            self.clear_safety()
            self._enabled_flag = False
            return True, "Disconnected."

    # -------------------------
    # State actions
    # -------------------------
    def enable(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                code = self._arm.motion_enable(True)
                if isinstance(code, int) and code != 0:
                    self._enabled_flag = False
                    return False, f"motion_enable failed code={code}"

                try:
                    self._arm.set_mode(0)
                except Exception:
                    pass
                try:
                    self._arm.set_state(0)
                except Exception:
                    pass

                time.sleep(0.05)

                self.clear_safety()
                self._enabled_flag = True
                return True, "Enabled."
            except Exception as e:
                self._enabled_flag = False
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
                self._enabled_flag = False
                return True, "Disabled/Stopped."
            except Exception as e:
                return False, f"Disable failed: {e}"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if self._arm is None:
                return False, "Not connected."
            try:
                try:
                    self._arm.set_state(4)
                except Exception:
                    pass
                self._enabled_flag = False
                return True, "Stopped."
            except Exception as e:
                return False, f"Stop failed: {e}"

    def go_home(self) -> Tuple[bool, str]:
        with self._lock:
            ok, msg = self._is_motion_allowed()
            if not ok:
                return False, msg
            try:
                if hasattr(self._arm, "move_gohome"):
                    code = self._arm.move_gohome(wait=False)
                    if isinstance(code, int) and code != 0:
                        return False, f"move_gohome failed code={code}"
                    return True, "Going home."
                return False, "Home not supported."
            except Exception as e:
                return False, f"Home failed: {e}"

    # -------------------------
    # Jog config
    # -------------------------
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

    # -------------------------
    # Jog (server-side throttle + safety)
    # -------------------------
    def jog_delta(
        self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0
    ) -> Tuple[bool, str, bool]:
        """
        Returns: (success, message, throttled)
        throttled=True => server-side rate-limit triggered (429)
        """
        with self._lock:
            ok, msg = self._is_motion_allowed()
            if not ok:
                return False, msg, False

            # throttle
            now = time.monotonic()
            min_dt = max(0.0, float(CONFIG.jog_min_interval_ms) / 1000.0)
            if min_dt > 0 and (now - self._last_jog_ts) < min_dt:
                return False, "Jog throttled (too fast).", True
            self._last_jog_ts = now

            # safety check
            pose = self._get_pose()
            if pose is not None and len(pose) >= 6:
                target = [
                    pose[0] + float(dx),
                    pose[1] + float(dy),
                    pose[2] + float(dz),
                    pose[3] + float(droll),
                    pose[4] + float(dpitch),
                    pose[5] + float(dyaw),
                ]
                ok2, reason = self._check_limits(target)
                if not ok2:
                    self._force_disable_for_safety(reason)
                    return False, self._safety_message, False

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
                    code = self._arm.set_position(**kwargs)  # type: ignore[union-attr]
                except TypeError:
                    kwargs.pop("coordinate_mode", None)
                    code = self._arm.set_position(**kwargs)  # type: ignore[union-attr]

                if isinstance(code, int) and code != 0:
                    return False, f"set_position failed code={code}", False
                return True, "OK", False
            except Exception as e:
                return False, f"Jog failed: {e}", False

    # -------------------------
    # Gripper
    # -------------------------
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
            ok, msg = self._is_motion_allowed()
            if not ok:
                return False, msg

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
                    code = self._arm.set_gripper_position(target, wait=False)  # type: ignore[union-attr]
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
            return {"available": self._gripper_available, "pos": pos, "pct": pct, "min": CONFIG.gripper_min, "max": CONFIG.gripper_max}

    # -------------------------
    # Status (NO motion checks here!)
    # -------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._arm is None:
                st = RobotStatus(
                    connected=False,
                    ip=self._ip,
                    message="Disconnected.",
                    safety_limit_hit=self._safety_limit_hit,
                    safety_message=self._safety_message,
                    is_enabled=False,
                )
                return asdict(st)

            hw = self._read_hw()
            st = RobotStatus(
                connected=True,
                ip=self._ip,
                message="OK",
                safety_limit_hit=self._safety_limit_hit,
                safety_message=self._safety_message,
                state=hw.get("state"),
                mode=hw.get("mode"),
                error_code=hw.get("error_code"),
                warn_code=hw.get("warn_code"),
            )

            st.is_enabled = bool(self._enabled_flag) and (st.error_code in (0, None)) and (not self._safety_limit_hit)

            gs = self.gripper_status()
            st.gripper_available = bool(gs["available"])
            st.gripper_pos = gs["pos"]
            st.gripper_pct = gs["pct"]
            st.gripper_min = gs["min"]
            st.gripper_max = gs["max"]

            return asdict(st)
