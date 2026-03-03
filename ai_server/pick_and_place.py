from __future__ import annotations

import base64
import math
import os
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import cv2
import httpx
import numpy as np
import scipy.ndimage as ndimage
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from skimage.feature import peak_local_max

from config import CONFIG

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")

PPC = CONFIG.pick_and_place

PICKPLACE_HOST = (os.environ.get("PICKPLACE_API_HOST") or "127.0.0.1").strip()
PICKPLACE_PORT = int((os.environ.get("PICKPLACE_API_PORT") or "9011").strip())


# -------------------------
# Helpers
# -------------------------
def normalize_any_http(input_value: str) -> str:
    raw = (input_value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw.rstrip("/")
    return f"http://{raw}".rstrip("/")


GATEWAY_URL = normalize_any_http(os.environ.get("GATEWAY_URL", ""))
if not GATEWAY_URL:
    raise RuntimeError("GATEWAY_URL is empty. Start via ai_server/main.py with gateway_url")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import models.ggcnn2  # noqa
from models.ggcnn2 import GGCNN2  # noqa


# -------------------------
# Gateway API
# -------------------------
class GatewayApi:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _url(self, path: str) -> str:
        return self.base + path

    def status(self) -> Dict[str, Any]:
        r = self.client.get(self._url("/api/status"))
        r.raise_for_status()
        return r.json()

    def ensure_camera_started(self) -> None:
        try:
            st = self.status()
            started = bool(st.get("cameras", {}).get("realsense", {}).get("started"))
            if not started:
                self.client.post(self._url("/api/cameras/realsense/start"))
        except Exception:
            pass

    def get_grasp_pack(self, x: int, y: int, crop_size: int) -> Dict[str, Any]:
        payload = {"x": int(x), "y": int(y), "crop_size": int(crop_size)}
        r = self.client.post(self._url(PPC.grasp_pack_path), json=payload)
        if r.status_code != 200:
            return {"valid": False, "message": f"gateway grasp_pack http={r.status_code}"}
        return r.json()

    def robot_enable(self) -> None:
        try:
            self.client.post(self._url(PPC.robot_enable_path))
        except Exception:
            pass

    def robot_stop(self) -> None:
        try:
            self.client.post(self._url(PPC.robot_stop_path))
        except Exception:
            pass

    def robot_move_pose(
        self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float, speed: int, wait: bool = True
    ) -> Tuple[bool, str]:
        body = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "speed": int(speed),
            "wait": bool(wait),
        }
        path = PPC.robot_move_pose_path
        try:
            r = self.client.post(self._url(path), json=body)
            if r.status_code == 200:
                return True, "OK"
            try:
                j = r.json()
                return False, f"{path} http={r.status_code} {j}"
            except Exception:
                return False, f"{path} http={r.status_code}"
        except Exception as e:
            return False, f"{path} error: {e}"


# -------------------------
# Math/utils
# -------------------------
def safe_z(z: float) -> float:
    min_z = float(PPC.table_z_mm + PPC.clearance_mm)
    return max(min_z, float(z))


def normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def load_homography() -> Optional[np.ndarray]:
    f = PPC.homography_file
    if os.path.exists(f):
        try:
            H = np.load(f)
            if H.shape == (3, 3):
                return H
        except Exception:
            return None
    return None


def project_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def recover_robot(gw: GatewayApi) -> None:
    try:
        gw.robot_stop()
    except Exception:
        pass
    try:
        gw.robot_enable()
    except Exception:
        pass


# -------------------------
# Cancellation
# -------------------------
_cancel_event = threading.Event()


def _cancel_requested() -> bool:
    return _cancel_event.is_set()


def _set_cancel(v: bool) -> None:
    if v:
        _cancel_event.set()
    else:
        _cancel_event.clear()


# -------------------------
# GGCNN
# -------------------------
_prev_mp = np.array([150, 150], dtype=np.int64)


def process_depth_image(depth_m: np.ndarray, out_size: int = 300):
    depth = depth_m.copy()
    invalid = (depth <= 0.0).astype(np.uint8)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    invalid = cv2.copyMakeBorder(invalid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

    kernel = np.ones((3, 3), np.uint8)
    invalid = cv2.dilate(invalid, kernel, iterations=1)
    depth[invalid == 1] = 0.0

    scale = float(np.max(depth)) if float(np.max(depth)) > 0 else 1.0
    depth_norm = (depth / scale).astype(np.float32)

    depth_inp = cv2.inpaint(depth_norm, invalid, 1, cv2.INPAINT_NS) * scale
    depth_inp = depth_inp[1:-1, 1:-1]
    invalid = invalid[1:-1, 1:-1]

    depth_inp = cv2.resize(depth_inp, (out_size, out_size), cv2.INTER_AREA)
    invalid = cv2.resize(invalid, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_inp, invalid


def load_ggcnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(SCRIPT_DIR, PPC.ggcnn_weight_path)
    torch.serialization.add_safe_globals([GGCNN2])
    model = torch.load(weight_path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model, device


def ggcnn_infer(depth_crop_m: np.ndarray, model, device, out_size: int):
    global _prev_mp

    depth_proc, invalid_mask = process_depth_image(depth_crop_m, out_size)
    depth_norm = np.clip((depth_proc - depth_proc.mean()), -1, 1)
    depthT = torch.from_numpy(depth_norm.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)

    with torch.no_grad():
        pred = model(depthT)

    points = pred[0].cpu().numpy().squeeze()
    cos_o = pred[1].cpu().numpy().squeeze()
    sin_o = pred[2].cpu().numpy().squeeze()
    ang = np.arctan2(sin_o, cos_o) / 2.0

    points[invalid_mask == 1] = 0
    points = ndimage.gaussian_filter(points, 5.0)
    ang = ndimage.gaussian_filter(ang, 2.0)
    points = np.clip(points, 0.0, 1.0 - 1e-3)

    maxes = peak_local_max(points, min_distance=10, threshold_abs=0.1, num_peaks=3)
    if maxes.shape[0] == 0:
        return None

    mp = maxes[np.argmin(np.linalg.norm(maxes - _prev_mp, axis=1))]
    if np.linalg.norm(mp - _prev_mp) > 30:
        mp = np.array(np.unravel_index(np.argmax(points), points.shape))
        _prev_mp = mp.astype(np.int64)
    else:
        _prev_mp = (mp * 0.25 + _prev_mp * 0.75).astype(np.int64)

    angle_rad = float(ang[_prev_mp[0], _prev_mp[1]])
    angle_deg = math.degrees(angle_rad)

    return {
        "best_u": int(_prev_mp[1]),
        "best_v": int(_prev_mp[0]),
        "angle_deg": float(angle_deg),
    }


# -------------------------
# Worker state
# -------------------------
Phase = Literal["WAIT_PICK", "WAIT_PLACE"]


@dataclass
class SelectionData:
    id: str
    u: float
    v: float
    pixel_x: int
    pixel_y: int
    tx: float
    ty: float
    yaw_deg: float
    pick_z: float
    place_z: float
    created_at: float


_lock = threading.Lock()
_busy = False
_last_error: str = ""
_last_selection: Optional[SelectionData] = None
_phase: Phase = "WAIT_PICK"
_picked_done: bool = False  # <--- critical: manual flow latch


def _set_busy(v: bool) -> None:
    global _busy
    with _lock:
        _busy = bool(v)


def _try_set_busy_true() -> bool:
    global _busy
    with _lock:
        if _busy:
            return False
        _busy = True
        return True


def _get_busy() -> bool:
    with _lock:
        return bool(_busy)


def _set_err(msg: str) -> None:
    global _last_error
    with _lock:
        _last_error = (msg or "")[:300]


def _get_err() -> str:
    with _lock:
        return _last_error


def _set_selection(sel: Optional[SelectionData]) -> None:
    global _last_selection
    with _lock:
        _last_selection = sel


def _get_selection() -> Optional[SelectionData]:
    with _lock:
        return _last_selection


def _get_phase() -> Phase:
    with _lock:
        return _phase


def _set_phase(p: Phase) -> None:
    global _phase
    with _lock:
        _phase = p


def _get_picked_done() -> bool:
    with _lock:
        return bool(_picked_done)


def _set_picked_done(v: bool) -> None:
    global _picked_done
    with _lock:
        _picked_done = bool(v)


# -------------------------
# MANUAL FLOW ACTION
# -------------------------
def perform_move_and_hold(gw: GatewayApi, x: float, y: float, z: float, yaw_deg: float) -> bool:
    """
    Manual flow:
      - recover (stop+enable)
      - move to XY at vision Z
      - descend to target Z
      - HOLD (no gripper, no ascend, no auto-return)
    """
    try:
        recover_robot(gw)

        if _cancel_requested():
            _set_err("cancelled")
            return False

        z = safe_z(z)

        ok, msg = gw.robot_move_pose(
            x=x,
            y=y,
            z=float(PPC.vision_z),
            roll=float(PPC.vision_roll),
            pitch=float(PPC.vision_pitch),
            yaw=float(yaw_deg),
            speed=int(PPC.speed_move),
            wait=True,
        )
        if not ok:
            _set_err(f"move XY failed: {msg}")
            return False

        if _cancel_requested():
            _set_err("cancelled")
            return False

        if bool(PPC.dry_run_xy_only):
            return True

        drop = float(PPC.vision_z) - float(z)
        if drop > float(PPC.max_drop_mm):
            _set_err(f"drop too large: {drop:.1f}mm > max_drop_mm={float(PPC.max_drop_mm):.1f}")
            return False

        ok, msg = gw.robot_move_pose(
            x=x,
            y=y,
            z=float(z),
            roll=float(PPC.vision_roll),
            pitch=float(PPC.vision_pitch),
            yaw=float(yaw_deg),
            speed=int(PPC.speed_descend),
            wait=True,
        )
        if not ok:
            _set_err(f"descend failed: {msg}")
            return False

        return True

    except Exception as e:
        _set_err(f"action failed: {e}")
        return False


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="pick_and_place_worker", version="2.1.0-manual-synced")


class SelectBody(BaseModel):
    u: float = Field(..., ge=0.0, le=1.0)
    v: float = Field(..., ge=0.0, le=1.0)
    source: str = "realsense_mjpeg"
    ts: Optional[int] = None


class ExecuteBody(BaseModel):
    action: str = Field(..., description="pick|place")
    selection_id: str


class SyncBody(BaseModel):
    phase: Phase
    picked_done: Optional[bool] = None


@app.get("/health")
def health():
    return JSONResponse(
        {
            "ok": True,
            "service": "pick_and_place_worker",
            "busy": _get_busy(),
            "phase": _get_phase(),
            "picked_done": _get_picked_done(),
            "error": _get_err(),
        }
    )


@app.get("/status")
def status():
    sel = _get_selection()
    return JSONResponse(
        {
            "ok": True,
            "busy": _get_busy(),
            "phase": _get_phase(),
            "picked_done": _get_picked_done(),
            "error": _get_err(),
            "selection": (sel.__dict__ if sel else None),
        }
    )


@app.post("/sync")
def sync(body: SyncBody):
    if _get_busy():
        return JSONResponse(status_code=409, content={"ok": False, "message": "robot is busy"})
    _set_phase(body.phase)
    if body.picked_done is not None:
        _set_picked_done(bool(body.picked_done))
    return JSONResponse({"ok": True, "phase": _get_phase(), "picked_done": _get_picked_done()})


@app.post("/cancel")
def cancel():
    _set_cancel(True)
    try:
        gw = GatewayApi(GATEWAY_URL, timeout_s=float(PPC.http_timeout_s))
        try:
            gw.robot_stop()
        finally:
            gw.close()
    except Exception:
        pass
    return JSONResponse({"ok": True, "message": "cancel requested"})


@app.post("/reset")
def reset():
    if _get_busy():
        return JSONResponse(status_code=409, content={"ok": False, "message": "robot is busy"})

    _set_cancel(False)
    _set_err("")
    _set_selection(None)
    _set_phase("WAIT_PICK")
    _set_picked_done(False)
    return JSONResponse({"ok": True, "message": "reset done", "phase": _get_phase(), "picked_done": _get_picked_done()})


@app.post("/select")
def select(body: SelectBody):
    if _get_busy():
        return JSONResponse(status_code=409, content={"ok": False, "message": "robot is busy"})

    _set_cancel(False)

    gw: Optional[GatewayApi] = None
    try:
        gw = GatewayApi(GATEWAY_URL, timeout_s=float(PPC.http_timeout_s))
        gw.ensure_camera_started()

        px = int(round(float(body.u) * 640))
        py = int(round(float(body.v) * 480))
        px = max(0, min(639, px))
        py = max(0, min(479, py))

        H = load_homography()
        if H is None:
            return JSONResponse(status_code=503, content={"ok": False, "message": "homography not loaded"})

        pack = gw.get_grasp_pack(px, py, int(PPC.crop_size))
        if not pack.get("valid"):
            return JSONResponse(status_code=503, content={"ok": False, "message": pack.get("message", "grasp_pack invalid")})

        depth_scale = float(pack["depth_scale"])
        origin = pack["crop_origin"]

        depth_png = base64.b64decode(pack["depth_crop_b64"])
        depth_u16 = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_ANYDEPTH)
        if depth_u16 is None:
            return JSONResponse(status_code=503, content={"ok": False, "message": "depth decode failed"})

        depth_crop_m = depth_u16.astype(np.float32) * depth_scale

        if not hasattr(select, "_model"):
            m, d = load_ggcnn_model()
            setattr(select, "_model", m)
            setattr(select, "_device", d)

        model = getattr(select, "_model")
        device = getattr(select, "_device")

        g = ggcnn_infer(depth_crop_m, model, device, out_size=int(PPC.crop_size))
        if g is None:
            return JSONResponse(status_code=503, content={"ok": False, "message": "no grasp found"})

        crop_h, crop_w = depth_u16.shape[:2]
        u_in_crop = int(g["best_u"] / float(PPC.crop_size) * crop_w)
        v_in_crop = int(g["best_v"] / float(PPC.crop_size) * crop_h)

        u_full = int(origin["x"] + u_in_crop)
        v_full = int(origin["y"] + v_in_crop)

        tx, ty = project_xy(H, u_full, v_full)

        yaw_target = float(PPC.vision_yaw)
        if bool(PPC.enable_auto_yaw):
            yaw_target = normalize_deg(
                float(PPC.vision_yaw) + float(PPC.yaw_sign) * float(g["angle_deg"]) + float(PPC.yaw_offset_deg)
            )

        pick_z = float(PPC.table_z_mm + PPC.pick_clearance_mm)
        place_z = float(PPC.place_z_mm) if PPC.place_z_mm is not None else float(PPC.table_z_mm + PPC.place_clearance_mm)

        sid = str(uuid.uuid4())

        sel = SelectionData(
            id=sid,
            u=float(body.u),
            v=float(body.v),
            pixel_x=px,
            pixel_y=py,
            tx=float(tx),
            ty=float(ty),
            yaw_deg=float(yaw_target),
            pick_z=float(pick_z),
            place_z=float(place_z),
            created_at=time.time(),
        )
        _set_selection(sel)
        _set_err("")

        return JSONResponse(
            {"ok": True, "selection": {"id": sid, "u": sel.u, "v": sel.v, "tx": sel.tx, "ty": sel.ty, "yaw_deg": sel.yaw_deg}}
        )
    except Exception as e:
        _set_err(str(e))
        return JSONResponse(status_code=500, content={"ok": False, "message": f"select failed: {e}"})
    finally:
        if gw is not None:
            gw.close()


@app.post("/execute")
def execute(body: ExecuteBody):
    act = (body.action or "").strip().lower()
    if act not in ("pick", "place"):
        return JSONResponse(status_code=400, content={"ok": False, "message": "action must be pick|place"})

    if not _try_set_busy_true():
        return JSONResponse(status_code=409, content={"ok": False, "message": "robot is busy"})

    _set_cancel(False)

    # ---- order check based on picked_done (robust against phase drift)
    picked_done = _get_picked_done()
    if act == "pick" and picked_done:
        _set_busy(False)
        return JSONResponse(status_code=409, content={"ok": False, "message": "invalid order: already picked (expected place)"})
    if act == "place" and (not picked_done):
        _set_busy(False)
        return JSONResponse(status_code=409, content={"ok": False, "message": "invalid order: expected pick (no pick latched)"})

    sel = _get_selection()
    if not sel or sel.id != body.selection_id:
        _set_busy(False)
        return JSONResponse(status_code=404, content={"ok": False, "message": "selection not found"})

    # reflect phase (mostly UI hint)
    _set_phase("WAIT_PLACE" if _get_picked_done() else "WAIT_PICK")

    def _run():
        gw: Optional[GatewayApi] = None
        ok_action = False
        try:
            gw = GatewayApi(GATEWAY_URL, timeout_s=float(PPC.http_timeout_s))
            time.sleep(float(PPC.stabilize_sleep_s))

            z = sel.pick_z if act == "pick" else sel.place_z
            ok_action = perform_move_and_hold(gw, sel.tx, sel.ty, z, sel.yaw_deg)

        except Exception as e:
            _set_err(f"execute failed: {e}")
        finally:
            if gw is not None:
                try:
                    gw.close()
                except Exception:
                    pass

            _set_busy(False)

            # latch update if succeeded
            if ok_action and (not _cancel_requested()):
                if act == "pick":
                    _set_picked_done(True)
                    _set_phase("WAIT_PLACE")
                else:
                    _set_picked_done(False)
                    _set_phase("WAIT_PICK")

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    return JSONResponse(
        {
            "ok": True,
            "message": f"{act} started (manual gripper)",
            "selection_id": sel.id,
            "phase": _get_phase(),
            "picked_done": _get_picked_done(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=PICKPLACE_HOST, port=PICKPLACE_PORT, access_log=True)