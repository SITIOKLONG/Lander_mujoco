# 20250220 Wakkk
# Quadrotor SE3 Control Demo
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import cv2
import mujoco 
import mujoco.viewer as viewer 
import numpy as np
import torch
from nmpc_controller import NMPC_Controller

# 新建NMPC控制器
controller = NMPC_Controller()

DEG2RAD = math.pi / 180.0
APRILTAG_ID = 0
APRILTAG_SIZE_M = 0.4
BOTTOM_CAM_WINDOW = "Bottom Camera"
BOTTOM_CAM_WIDTH = 320
BOTTOM_CAM_HEIGHT = 240
APRILTAG_TEXTURE_PATH = Path(__file__).resolve().parent / "crazyfile" / "assets" / "apriltag36h11_id0.png"


def _bottom_camera_display_worker(frame_queue, window_name):
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(f"[BottomCam] failed to create OpenCV window: {exc}")
        return

    last_frame = None
    try:
        while True:
            try:
                item = frame_queue.get(timeout=0.03)
                if item is None:
                    break
                last_frame = item
            except queue.Empty:
                pass

            if last_frame is not None:
                cv2.imshow(window_name, last_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        cv2.destroyAllWindows()


class BottomCameraDisplayProcess:
    """Separate process for OpenCV HighGUI to avoid callback-thread GUI calls."""

    def __init__(self, window_name=BOTTOM_CAM_WINDOW):
        self.window_name = window_name
        self.ctx = mp.get_context("spawn")
        self.frame_queue = None
        self.process = None

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.frame_queue = self.ctx.Queue(maxsize=2)
        self.process = self.ctx.Process(
            target=_bottom_camera_display_worker,
            args=(self.frame_queue, self.window_name),
            daemon=True,
        )
        self.process.start()

    def submit(self, frame_bgr):
        if self.frame_queue is None or self.process is None or not self.process.is_alive():
            return
        try:
            self.frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self.frame_queue.put_nowait(frame_bgr)
            except Exception:
                pass
        except Exception:
            pass

    def close(self):
        if self.frame_queue is not None:
            try:
                self.frame_queue.put_nowait(None)
            except Exception:
                pass

        if self.process is not None:
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
            self.process = None

        if self.frame_queue is not None:
            try:
                self.frame_queue.close()
            except Exception:
                pass
            self.frame_queue = None


def _require_aruco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is required. Please install opencv-contrib-python.")
    if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError("OpenCV build does not expose DICT_APRILTAG_36h11.")
    return cv2.aruco


def ensure_apriltag_texture(texture_path=APRILTAG_TEXTURE_PATH, tag_id=APRILTAG_ID, pixels=512):
    """Generate a real tag36h11 texture file used by the MuJoCo scene."""
    aruco = _require_aruco()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    marker_pixels = int(pixels * 0.75)
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, int(tag_id), int(marker_pixels))
    else:
        marker = np.zeros((int(marker_pixels), int(marker_pixels)), dtype=np.uint8)
        aruco.drawMarker(dictionary, int(tag_id), int(marker_pixels), marker, 1)

    # Add a white margin so OpenCV can find the marker boundary robustly.
    tag_img = np.full((int(pixels), int(pixels)), 255, dtype=np.uint8)
    margin = max((int(pixels) - int(marker_pixels)) // 2, 1)
    tag_img[margin:margin + int(marker_pixels), margin:margin + int(marker_pixels)] = marker

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(texture_path), tag_img)
    if not ok:
        raise RuntimeError(f"Failed to write AprilTag texture: {texture_path}")


def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """Convert XYZ Euler angles to MuJoCo quaternion order: wxyz."""
    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw

    cr = math.cos(hr)
    sr = math.sin(hr)
    cp = math.cos(hp)
    sp = math.sin(hp)
    cy = math.cos(hy)
    sy = math.sin(hy)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / q_norm


def quat_mul_wxyz(q1, q2):
    """Quaternion multiply q = q1 * q2 for quaternions in wxyz order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q = np.array([w, x, y, z], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / q_norm


def quat_conj_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_rotmat_wxyz(q):
    q0, q1, q2, q3 = q
    return rotation_matrix(q0, q1, q2, q3)


def rotmat_to_quat_wxyz(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / q_norm


def quat_to_ang_vel_wxyz(q_prev, q_curr, dt):
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float64)
    dq = quat_mul_wxyz(quat_conj_wxyz(q_prev), q_curr)
    dq = dq / np.linalg.norm(dq)
    w = np.clip(dq[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = dq[1:] / s
    return axis * angle / dt


class WavePadMotion:
    """Spectral + OU + smoothing wave motion for a MuJoCo mocap body."""

    def __init__(
        self,
        body_name="wave_pad",
        amp_heave=0.1,
        amp_roll_deg=30.0,
        amp_pitch_deg=30.0,
        freq_min=0.05,
        freq_max=0.25,
        n_sines=6,
        ou_rho=0.99,
        ou_sigma=0.05,
        smoother_alpha=0.02,
        seed=7,
    ):
        self.body_name = body_name
        self.amp_heave = float(amp_heave)
        self.amp_roll = float(amp_roll_deg) * DEG2RAD
        self.amp_pitch = float(amp_pitch_deg) * DEG2RAD
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.n_sines = int(n_sines)
        self.ou_rho = float(ou_rho)
        self.ou_sigma = float(ou_sigma)
        self.alpha = float(np.clip(smoother_alpha, 0.0, 1.0))

        self.rng = np.random.default_rng(seed)

        self.mocap_id = None
        self.dt = 0.01
        self.t = 0.0
        self.base_pos = None
        self.base_quat = None

        self.omegas = None
        self.phases = None
        self.amps = None

        self.ou_state = np.zeros(3, dtype=np.float64)
        self.smooth_state = np.zeros(3, dtype=np.float64)
        self.prev_pos = None
        self.prev_quat = None
        self.current_lin_vel = np.zeros(3, dtype=np.float64)
        self.current_ang_vel = np.zeros(3, dtype=np.float64)

    def bind_model(self, m, d):
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id < 0:
            raise ValueError(f"Body '{self.body_name}' was not found in MuJoCo model")

        mocap_id = int(m.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body '{self.body_name}' is not configured as mocap='true'")

        self.mocap_id = mocap_id
        self.dt = float(m.opt.timestep)
        self.base_pos = d.mocap_pos[self.mocap_id].copy()
        self.base_quat = d.mocap_quat[self.mocap_id].copy()
        self.prev_pos = self.base_pos.copy()
        self.prev_quat = self.base_quat.copy()
        self.current_lin_vel[:] = 0.0
        self.current_ang_vel[:] = 0.0

        self.reset(d, resample=True)

    def _resample_spectrum(self):
        freqs_hz = self.rng.uniform(self.freq_min, self.freq_max, size=(self.n_sines,))
        self.omegas = 2.0 * math.pi * freqs_hz
        self.phases = self.rng.uniform(0.0, 2.0 * math.pi, size=(self.n_sines,))

        inv_f = 1.0 / np.clip(freqs_hz, 0.05, None)
        weights = inv_f / np.sum(inv_f)

        self.amps = np.vstack(
            [
                self.amp_heave * weights,
                self.amp_roll * weights,
                self.amp_pitch * weights,
            ]
        )

    def reset(self, d, resample=True):
        self.t = 0.0
        self.ou_state[:] = 0.0
        self.smooth_state[:] = 0.0
        if resample or self.omegas is None:
            self._resample_spectrum()

        if self.mocap_id is not None:
            d.mocap_pos[self.mocap_id] = self.base_pos
            d.mocap_quat[self.mocap_id] = self.base_quat
            self.prev_pos = self.base_pos.copy()
            self.prev_quat = self.base_quat.copy()
            self.current_lin_vel[:] = 0.0
            self.current_ang_vel[:] = 0.0

    def step(self, d):
        if self.mocap_id is None:
            return

        self.t += self.dt

        angles = self.omegas * self.t + self.phases
        sines = np.sin(angles)
        disturbance = np.sum(self.amps * sines[None, :], axis=1)

        noise_scale = np.array([self.amp_heave, self.amp_roll, self.amp_pitch], dtype=np.float64)
        self.ou_state = self.ou_rho * self.ou_state + self.ou_sigma * noise_scale * self.rng.standard_normal(3)
        disturbance += self.ou_state

        self.smooth_state = (1.0 - self.alpha) * self.smooth_state + self.alpha * disturbance
        z_offset, roll, pitch = self.smooth_state

        prev_pos = d.mocap_pos[self.mocap_id].copy()
        prev_quat = d.mocap_quat[self.mocap_id].copy()

        d.mocap_pos[self.mocap_id] = self.base_pos
        d.mocap_pos[self.mocap_id, 2] = self.base_pos[2] + z_offset

        dq = euler_xyz_to_quat_wxyz(roll, pitch, 0.0)
        d.mocap_quat[self.mocap_id] = quat_mul_wxyz(self.base_quat, dq)

        curr_pos = d.mocap_pos[self.mocap_id].copy()
        curr_quat = d.mocap_quat[self.mocap_id].copy()
        self.current_lin_vel = (curr_pos - prev_pos) / self.dt
        self.current_ang_vel = quat_to_ang_vel_wxyz(prev_quat, curr_quat, self.dt)
        self.prev_pos = curr_pos
        self.prev_quat = curr_quat

    def get_pose(self, d):
        return d.mocap_pos[self.mocap_id].copy(), d.mocap_quat[self.mocap_id].copy()

    def get_phase_features(self):
        """Return [cos(phase_i), sin(phase_i)] for each active wave component."""
        k = max(int(self.n_sines), 1)
        if self.omegas is None or self.phases is None:
            return np.zeros((2 * k,), dtype=np.float64)
        angles = self.omegas * self.t + self.phases
        return np.concatenate([np.cos(angles), np.sin(angles)]).astype(np.float64)


class WindDisturbance:
    """Apply smooth wind disturbance as external force/torque on the drone body."""

    def __init__(
        self,
        body_name="cf2",
        base_wind=(0.01, 0.01, 0.01),
        gust_amp=(0.01, 0.01, 0.01),
        gust_freq_hz=(0.03, 0.05, 0.04),
        drag_coeff=0.015,
        ou_rho=0.998,
        ou_sigma=0.03,
        torque_sigma=1.2e-4,
        max_force=0.04,
        max_torque=0.0003,
        seed=17,
    ):
        self.body_name = body_name
        self.base_wind = np.array(base_wind, dtype=np.float64)
        self.gust_amp = np.array(gust_amp, dtype=np.float64)
        self.gust_freq_hz = np.array(gust_freq_hz, dtype=np.float64)
        self.drag_coeff = float(drag_coeff)
        self.ou_rho = float(ou_rho)
        self.ou_sigma = float(ou_sigma)
        self.torque_sigma = float(torque_sigma)
        self.max_force = float(max_force)
        self.max_torque = float(max_torque)

        self.rng = np.random.default_rng(seed)
        self.gust_phase = self.rng.uniform(0.0, 2.0 * math.pi, size=(3,))

        self.body_id = None
        self.dt = 0.01
        self.t = 0.0

        self.wind_ou = np.zeros(3, dtype=np.float64)
        self.torque_ou = np.zeros(3, dtype=np.float64)

        self.current_wind = np.zeros(3, dtype=np.float64)
        self.current_force = np.zeros(3, dtype=np.float64)
        self.current_torque = np.zeros(3, dtype=np.float64)

    def bind_model(self, m, d):
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id < 0:
            raise ValueError(f"Body '{self.body_name}' was not found in MuJoCo model")
        self.body_id = body_id
        self.dt = float(m.opt.timestep)
        self.reset(d)

    def reset(self, d):
        self.t = 0.0
        self.wind_ou[:] = 0.0
        self.torque_ou[:] = 0.0
        self.current_wind[:] = 0.0
        self.current_force[:] = 0.0
        self.current_torque[:] = 0.0
        if self.body_id is not None:
            d.xfrc_applied[self.body_id, :] = 0.0

    def _clip_vec_norm(self, v, max_norm):
        n = float(np.linalg.norm(v))
        if n <= max_norm or n < 1e-8:
            return v
        return v * (max_norm / n)

    def step_and_apply(self, d):
        if self.body_id is None:
            return

        self.t += self.dt

        gust = self.gust_amp * np.sin(2.0 * math.pi * self.gust_freq_hz * self.t + self.gust_phase)
        self.wind_ou = self.ou_rho * self.wind_ou + self.ou_sigma * self.rng.standard_normal(3)
        self.current_wind = self.base_wind + gust + self.wind_ou

        # For freejoint, first 3 components are world-frame linear velocity.
        body_lin_vel_world = np.array(d.qvel[0:3], dtype=np.float64)
        rel_air_vel = self.current_wind - body_lin_vel_world
        rel_speed = float(np.linalg.norm(rel_air_vel))

        # Quadratic drag-like wind force in world frame.
        force = self.drag_coeff * rel_air_vel * rel_speed
        force = self._clip_vec_norm(force, self.max_force)

        # Small random disturbance torque in world frame.
        self.torque_ou = self.ou_rho * self.torque_ou + self.torque_sigma * self.rng.standard_normal(3)
        torque = np.clip(self.torque_ou, -self.max_torque, self.max_torque)

        d.xfrc_applied[self.body_id, 0:3] = force
        d.xfrc_applied[self.body_id, 3:6] = torque

        self.current_force = force
        self.current_torque = torque


class HeightPolicyJIT:
    """Run TorchScript policy with IsaacLab-compatible observation layout."""

    def __init__(
        self,
        model_path,
        min_height=0.15,
        max_height=1.20,
        output_is_normalized=False,
        action_scale=1.0,
        action_bias=0.0,
        max_roll_deg=25.0,
        max_pitch_deg=25.0,
        max_yaw_deg=45.0,
    ):
        self.model_path = Path(model_path)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.output_is_normalized = bool(output_is_normalized)
        self.action_scale = float(action_scale)
        self.action_bias = float(action_bias)
        self.max_roll_rad = float(max_roll_deg) * DEG2RAD
        self.max_pitch_rad = float(max_pitch_deg) * DEG2RAD
        self.max_yaw_rad = float(max_yaw_deg) * DEG2RAD

        self.model = None
        self.prev_slider_pos = None
        self.prev_tag_quat = None
        self.device = torch.device("cpu")
        self.expected_input_dim = None
        self._warned_obs_pad = False
        self._warned_obs_trunc = False
        self._warned_action_dim = False

        # IsaacLab policy layout (54 dims):
        # pos_hist(4), vel_hist(4), quat_hist(16), ang_vel_hist(12), wave_phase(12), contact_hist(6)
        self._slider_pos_hist = np.zeros((4,), dtype=np.float64)
        self._slider_vel_hist = np.zeros((4,), dtype=np.float64)
        self._pad_quat_hist = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (4, 1))
        self._pad_ang_vel_hist = np.zeros((4, 3), dtype=np.float64)
        self._contact_hist = np.zeros((2, 3), dtype=np.float64)

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.model_path}")
        # Default to MPS on Apple Silicon unless user explicitly disables it.
        allow_mps = os.getenv("POLICY_USE_MPS", "1") == "1"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif allow_mps:
            mps_avail = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_avail is not None and getattr(torch.backends.mps, "is_available", lambda: False)():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        try:
            # load to the selected device if supported by map_location
            self.model = torch.jit.load(str(self.model_path), map_location=str(self.device))
        except Exception:
            # fallback: load to CPU then move
            self.model = torch.jit.load(str(self.model_path), map_location="cpu")
            try:
                self.model.to(self.device)
            except Exception:
                # some ScriptModules may not support .to; keep on CPU
                self.device = torch.device("cpu")

        self.model.eval()
        # Try to infer expected observation dimension from model.normalizer._mean
        try:
            normalizer = getattr(self.model, "normalizer", None)
            if normalizer is not None:
                mean = getattr(normalizer, "_mean", None)
                if mean is not None:
                    self.expected_input_dim = int(mean.numel())
        except Exception:
            self.expected_input_dim = None
        if self.expected_input_dim is not None:
            print(f"[HeightPolicyJIT] model expected input dim = {self.expected_input_dim}")
        print(f"[HeightPolicyJIT] using device = {self.device}")

    def reset(self):
        self.prev_slider_pos = None
        self.prev_tag_quat = None
        self._slider_pos_hist[:] = 0.0
        self._slider_vel_hist[:] = 0.0
        self._pad_quat_hist[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._pad_ang_vel_hist[:] = 0.0
        self._contact_hist[:] = 0.0

    def _push_scalar_hist(self, hist, value):
        hist[:-1] = hist[1:]
        hist[-1] = float(value)

    def _push_vector_hist(self, hist, vec):
        hist[:-1, :] = hist[1:, :]
        hist[-1, :] = np.asarray(vec, dtype=np.float64)

    def _fit_vec(self, vec, target_dim):
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if vec.size == target_dim:
            return vec
        out = np.zeros((target_dim,), dtype=np.float64)
        n = min(target_dim, vec.size)
        out[:n] = vec[:n]
        return out

    def build_obs_from_tag(self, apriltag_pose, dt, wave_phase_features=None, contact_force=None):
        if apriltag_pose is None:
            tag_pos = np.zeros(3, dtype=np.float64)
            tag_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            tag_pos = np.asarray(apriltag_pose.get("position", np.zeros(3)), dtype=np.float64)
            tag_quat = np.asarray(apriltag_pose.get("quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float64)
            q_norm = float(np.linalg.norm(tag_quat))
            if q_norm < 1e-8:
                tag_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                tag_quat = tag_quat / q_norm

        # In OpenCV camera frame, +Z points forward from the camera.
        # Bottom camera points down, so -z_cam is the signed vertical slider-like position.
        slider_pos = -float(tag_pos[2])
        if self.prev_slider_pos is None or dt <= 0.0:
            slider_vel = 0.0
        else:
            slider_vel = (slider_pos - self.prev_slider_pos) / dt
        self.prev_slider_pos = slider_pos

        if self.prev_tag_quat is None or dt <= 0.0:
            tag_ang_vel = np.zeros(3, dtype=np.float64)
        else:
            tag_ang_vel = quat_to_ang_vel_wxyz(self.prev_tag_quat, tag_quat, dt)
        self.prev_tag_quat = tag_quat.copy()

        wave_phase = self._fit_vec(
            np.zeros((12,), dtype=np.float64) if wave_phase_features is None else wave_phase_features,
            12,
        )
        contact_force = self._fit_vec(
            np.zeros((3,), dtype=np.float64) if contact_force is None else contact_force,
            3,
        )

        self._push_scalar_hist(self._slider_pos_hist, slider_pos)
        self._push_scalar_hist(self._slider_vel_hist, slider_vel)
        self._push_vector_hist(self._pad_quat_hist, tag_quat)
        self._push_vector_hist(self._pad_ang_vel_hist, tag_ang_vel)
        self._push_vector_hist(self._contact_hist, contact_force)

        obs = np.concatenate(
            [
                self._slider_pos_hist,
                self._slider_vel_hist,
                self._pad_quat_hist.reshape(-1),
                self._pad_ang_vel_hist.reshape(-1),
                wave_phase,
                self._contact_hist.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
        return self._pad_obs(obs)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        """Pad or pass-through observation to match model expected input dim.

        If the loaded model exposed a `normalizer._mean` we pad to that length (zeros).
        Otherwise we return the original obs.
        """
        if self.expected_input_dim is None:
            return obs
        cur = int(obs.size)
        if cur == self.expected_input_dim:
            return obs
        if cur > self.expected_input_dim:
            # truncate with warning (print once)
            if not self._warned_obs_trunc:
                print(f"[HeightPolicyJIT] WARNING: obs len {cur} > expected {self.expected_input_dim}, truncating")
                self._warned_obs_trunc = True
            return obs[: self.expected_input_dim]
        # pad with zeros
        padded = np.zeros((self.expected_input_dim,), dtype=np.float32)
        padded[:cur] = obs
        if not self._warned_obs_pad:
            print(f"[HeightPolicyJIT] WARNING: obs len {cur} < expected {self.expected_input_dim}, padding with zeros")
            self._warned_obs_pad = True
        return padded

    def _infer_raw_action(self, obs):
        if self.model is None:
            raise RuntimeError("Policy model is not loaded")

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(obs)):
            obs = obs.copy()
            obs[~np.isfinite(obs)] = 0.0

        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0)
            try:
                y = self.model(x.to(self.device))
            except Exception as exc:
                if self.device.type != "cpu":
                    print(f"[HeightPolicyJIT] device inference failed on {self.device}: {exc}")
                    print("[HeightPolicyJIT] falling back to CPU")
                    self.device = torch.device("cpu")
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
                    y = self.model(x.to(self.device))
                else:
                    raise

        if isinstance(y, (tuple, list)):
            y = y[0]
        y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if y_np.size == 0:
            raise RuntimeError("Policy output is empty")
        return y_np

    def infer_height_attitude(self, obs):
        """Return policy-derived (height_cmd_m, attitude_cmd_rpy_rad, raw_action_vec)."""
        y_np = self._infer_raw_action(obs)
        raw = float(y_np[0])
        if not np.isfinite(raw):
            raw = 0.0

        if self.output_is_normalized:
            mapped = 0.5 * (raw + 1.0) * (self.max_height - self.min_height) + self.min_height
        else:
            mapped = self.action_scale * raw + self.action_bias
        if not np.isfinite(mapped):
            mapped = self.min_height

        height_cmd = float(np.clip(mapped, self.min_height, self.max_height))

        # Policy action layout: [height, roll, pitch, yaw].
        # If attitude channels are missing, keep missing commands at zero.
        att_raw = np.zeros((3,), dtype=np.float64)
        n_att = min(max(int(y_np.size) - 1, 0), 3)
        if n_att > 0:
            att_raw[:n_att] = np.clip(y_np[1 : 1 + n_att], -1.0, 1.0)
        if n_att < 3 and not self._warned_action_dim:
            print(
                f"[HeightPolicyJIT] WARNING: policy action dim={y_np.size}; "
                "expected >=4 for [height, roll, pitch, yaw]. Missing attitude channels set to 0."
            )
            self._warned_action_dim = True

        attitude_cmd_rpy = np.array(
            [
                att_raw[0] * self.max_roll_rad,
                att_raw[1] * self.max_pitch_rad,
                att_raw[2] * self.max_yaw_rad,
            ],
            dtype=np.float64,
        )
        attitude_cmd_rpy[~np.isfinite(attitude_cmd_rpy)] = 0.0
        return height_cmd, attitude_cmd_rpy, y_np

    def infer_height(self, obs):
        """Backward-compatible API: return (height_cmd_m, raw_height_action)."""
        height_cmd, _, raw_action = self.infer_height_attitude(obs)
        return height_cmd, float(raw_action[0])


class AprilTagPoseTracker:
    """Render bottom camera and estimate AprilTag pose (tag36h11) with OpenCV."""

    def __init__(
        self,
        camera_name="bottom_cam",
        tag_id=APRILTAG_ID,
        tag_size_m=APRILTAG_SIZE_M,
        render_width=BOTTOM_CAM_WIDTH,
        render_height=BOTTOM_CAM_HEIGHT,
        window_name=BOTTOM_CAM_WINDOW,
    ):
        self.camera_name = camera_name
        self.tag_id = int(tag_id)
        self.tag_size_m = float(tag_size_m)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.window_name = window_name

        self.cam_id = None
        self.renderer = None
        self.aruco = None
        self.dictionary = None
        self.detector = None

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        half = 0.5 * self.tag_size_m
        self.tag_object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        self.last_pose = {
            "detected": False,
            "position": np.zeros(3, dtype=np.float64),
            "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        self.last_overlay = None
        self.display_process = None
        self.local_window_enabled = True
        self.local_window_created = False

    def set_display_process(self, display_process):
        self.display_process = display_process

    def bind_model(self, m):
        self.aruco = _require_aruco()
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_APRILTAG_36h11)
        if hasattr(self.aruco, "ArucoDetector"):
            if hasattr(self.aruco, "DetectorParameters"):
                params = self.aruco.DetectorParameters()
            else:
                params = self.aruco.DetectorParameters_create()
            self.detector = self.aruco.ArucoDetector(self.dictionary, params)

        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' was not found in MuJoCo model")

        fovy_deg = float(m.cam_fovy[self.cam_id])
        h = float(self.render_height)
        w = float(self.render_width)
        fy = 0.5 * h / math.tan(0.5 * math.radians(fovy_deg))
        fx = fy
        cx = 0.5 * (w - 1.0)
        cy = 0.5 * (h - 1.0)
        self.camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.renderer = mujoco.Renderer(m, height=self.render_height, width=self.render_width)

    def _detect_markers(self, gray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return self.aruco.detectMarkers(gray, self.dictionary)

    def detect_pose(self, m, d):
        if self.cam_id is None or self.renderer is None:
            return self.last_pose

        self.renderer.update_scene(d, camera=self.camera_name)
        frame_rgb = self.renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self._detect_markers(gray)

        detected = False
        pos_cam = self.last_pose["position"].copy()
        quat_cam = self.last_pose["quat_wxyz"].copy()
        overlay = frame_bgr.copy()

        if ids is not None and len(ids) > 0:
            self.aruco.drawDetectedMarkers(overlay, corners, ids)
            ids_flat = ids.reshape(-1)
            match = np.where(ids_flat == self.tag_id)[0]
            if match.size > 0:
                idx = int(match[0])
                image_points = corners[idx].reshape(4, 2).astype(np.float64)
                solve_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
                ok, rvec, tvec = cv2.solvePnP(
                    self.tag_object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=solve_flag,
                )
                if ok:
                    detected = True
                    pos_cam = tvec.reshape(3).astype(np.float64)
                    rot_cam_tag, _ = cv2.Rodrigues(rvec)
                    quat_cam = rotmat_to_quat_wxyz(rot_cam_tag).astype(np.float64)
                    cv2.drawFrameAxes(
                        overlay,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        0.04,
                        2,
                    )

        if detected:
            status = f"tag36h11 id={self.tag_id}, z={pos_cam[2]:.3f}m"
        else:
            status = f"tag36h11 id={self.tag_id} not detected"
        cv2.putText(overlay, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self.last_overlay = overlay
        if self.display_process is not None:
            self.display_process.submit(overlay)

        self.last_pose = {
            "detected": bool(detected),
            "position": pos_cam,
            "quat_wxyz": quat_cam,
        }
        return self.last_pose

    def estimate_tag_world_position(self, d, pose=None):
        """Estimate tag center in world frame from camera pose + OpenCV tvec.

        OpenCV camera coordinates are x-right, y-down, z-forward.
        MuJoCo camera local frame is x-right, y-up, z-backward.
        """
        if self.cam_id is None:
            return None

        if pose is None:
            pose = self.last_pose
        if pose is None or not bool(pose.get("detected", False)):
            return None

        pos_cam_cv = np.asarray(pose.get("position", np.zeros(3)), dtype=np.float64)
        if pos_cam_cv.size != 3:
            return None

        # Convert OpenCV camera-frame vector to MuJoCo camera-frame vector.
        cam_to_tag_mj = np.array([pos_cam_cv[0], -pos_cam_cv[1], -pos_cam_cv[2]], dtype=np.float64)
        cam_world_pos = np.asarray(d.cam_xpos[self.cam_id], dtype=np.float64).copy()
        cam_world_rot = np.asarray(d.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3)
        return cam_world_pos + cam_world_rot @ cam_to_tag_mj

    def show_last_frame(self):
        if self.last_overlay is None or not self.local_window_enabled:
            return
        try:
            if not self.local_window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.local_window_created = True
            cv2.imshow(self.window_name, self.last_overlay)
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"[AprilTagPoseTracker] disabling local OpenCV window: {exc}")
            self.local_window_enabled = False

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        if self.local_window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.local_window_created = False


wave_pad_motion = WavePadMotion()
wind_disturbance = WindDisturbance()
height_policy = HeightPolicyJIT(model_path=Path(__file__).resolve().parent / "models" / "model_2.pt")
apriltag_tracker = AprilTagPoseTracker()
bottom_cam_display = BottomCameraDisplayProcess(window_name=BOTTOM_CAM_WINDOW)

gravity = 9.8066        # 重力加速度 单位m/s^2
mass = 0.033            # 飞行器质量 单位kg
Ct = 3.25e-4            # 电机推力系数 (N/krpm^2)
Cd = 7.9379e-6          # 电机反扭系数 (Nm/krpm^2)

arm_length = 0.065/2.0  # 电机力臂长度 单位m
max_thrust = 0.1573     # 单个电机最大推力 单位N (电机最大转速22krpm)
max_torque = 3.842e-03  # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)

# 仿真周期 100Hz 10ms 0.01s
dt = 0.01

# 根据电机转速计算电机推力
def calc_motor_force(krpm):
    global Ct
    return Ct * krpm**2

# 根据电机转速计算电机归一化输入
def calc_motor_input(krpm):
    if not np.isfinite(krpm):
        krpm = float(np.sqrt((mass * gravity) / (4.0 * Ct)))
    if krpm > 22:
        krpm = 22
    elif krpm < 0:
        krpm = 0
    _force = calc_motor_force(krpm)
    _input = _force / max_thrust
    if _input > 1:
        _input = 1
    elif _input < 0:
        _input = 0
    return _input


_CONTACT_CACHE = {
    "drone_body_id": -2,
    "pad_geom_id": -2,
}


def estimate_contact_force_between_drone_and_pad(m, d):
    """Approximate contact force (world xyz) between `cf2` body and `wave_pad_geom`."""
    if _CONTACT_CACHE["drone_body_id"] < 0:
        _CONTACT_CACHE["drone_body_id"] = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cf2"))
    if _CONTACT_CACHE["pad_geom_id"] < 0:
        _CONTACT_CACHE["pad_geom_id"] = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wave_pad_geom"))

    drone_body_id = _CONTACT_CACHE["drone_body_id"]
    pad_geom_id = _CONTACT_CACHE["pad_geom_id"]
    if drone_body_id < 0 or pad_geom_id < 0:
        return np.zeros((3,), dtype=np.float64)

    total_world_force = np.zeros((3,), dtype=np.float64)
    force_contact = np.zeros((6,), dtype=np.float64)

    for i in range(int(d.ncon)):
        contact = d.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 == pad_geom_id:
            other_geom = g2
        elif g2 == pad_geom_id:
            other_geom = g1
        else:
            continue

        if int(m.geom_bodyid[other_geom]) != drone_body_id:
            continue

        mujoco.mj_contactForce(m, d, i, force_contact)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        total_world_force += frame @ force_contact[:3]

    return total_world_force

# 加载模型回调函数
def setup_simulation(use_external_display=False):
    ensure_apriltag_texture()
    m = mujoco.MjModel.from_xml_path('./crazyfile/scene.xml')
    d = mujoco.MjData(m)
    wave_pad_motion.bind_model(m, d)
    wind_disturbance.bind_model(m, d)
    height_policy.load()
    height_policy.reset()
    apriltag_tracker.set_display_process(bottom_cam_display if use_external_display else None)
    apriltag_tracker.local_window_enabled = not use_external_display
    apriltag_tracker.local_window_created = False
    apriltag_tracker.bind_model(m)
    return m, d


def load_callback_for_viewer_launch(m=None, d=None):
    # Compatibility path for platforms/environments where launch_passive is unavailable.
    mujoco.set_mjcb_control(None)
    m, d = setup_simulation(use_external_display=True)
    mujoco.set_mjcb_control(lambda mm, dd: step_control(mm, dd))
    return m, d


def run_headless_bottom_camera_loop():
    print("[main] viewer unavailable; running simulation loop with bottom camera window")
    print("[main] press q or ESC in the bottom camera window to quit")

    m, d = setup_simulation(use_external_display=False)
    try:
        while True:
            step_start = time.time()

            step_control(m, d)
            mujoco.mj_step(m, d)
            apriltag_tracker.show_last_frame()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            remaining = float(m.opt.timestep) - (time.time() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("[main] headless loop interrupted by user")

# 根据四元数计算旋转矩阵
def rotation_matrix(q0, q1, q2, q3):
    _row0 = np.array([1-2*(q2**2)-2*(q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    _row1 = np.array([2*(q1*q2+q0*q3), 1-2*(q1**2)-2*(q3**2), 2*(q2*q3-q0*q1)])
    _row2 = np.array([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2)-2*(q2**2)])
    return np.vstack((_row0, _row1, _row2))

log_count = 0
def step_control(m, d):
    global log_count, gravity, mass, controller, wave_pad_motion, wind_disturbance, height_policy, apriltag_tracker
    wave_pad_motion.step(d)
    wind_disturbance.step_and_apply(d)

    apriltag_pose = apriltag_tracker.detect_pose(m, d)
    wave_phase_features = wave_pad_motion.get_phase_features()
    contact_force = estimate_contact_force_between_drone_and_pad(m, d)
    obs = height_policy.build_obs_from_tag(
        apriltag_pose,
        float(m.opt.timestep),
        wave_phase_features=wave_phase_features,
        contact_force=contact_force,
    )
    desired_height_raw, desired_att_rpy, raw_action_vec = height_policy.infer_height_attitude(obs)
    desired_height = float(desired_height_raw)
    if not np.isfinite(desired_height):
        desired_height = float(height_policy.min_height)
    desired_height = float(np.clip(desired_height, height_policy.min_height, height_policy.max_height))

    # Build visual target from AprilTag center in world frame.
    tag_world_pos = apriltag_tracker.estimate_tag_world_position(d, apriltag_pose)
    pad_pos_fallback, _ = wave_pad_motion.get_pose(d)

    _pos = d.qpos
    _vel = d.qvel
    _sensor_data = d.sensordata
    gyro_x = _sensor_data[0]
    gyro_y = _sensor_data[1]
    gyro_z = _sensor_data[2]
    acc_x = _sensor_data[3]
    acc_y = _sensor_data[4]
    acc_z = _sensor_data[5]
    quat_w = _sensor_data[6]
    quat_x = _sensor_data[7]
    quat_y = _sensor_data[8]
    quat_z = _sensor_data[9]
    quat = np.array([quat_x, quat_y, quat_z, quat_w])  # x y z w
    omega = np.array([gyro_x, gyro_y, gyro_z])         # 角速度

    if tag_world_pos is not None and np.all(np.isfinite(tag_world_pos)):
        goal_pos = np.array(
            [
                tag_world_pos[0],
                tag_world_pos[1],
                tag_world_pos[2] + desired_height,
            ],
            dtype=np.float64,
        )
    else:
        goal_pos = np.array(
            [
                pad_pos_fallback[0],
                pad_pos_fallback[1],
                pad_pos_fallback[2] + desired_height,
            ],
            dtype=np.float64,
        )

    # Final safety clamp against any non-finite target values.
    fallback_goal = np.array([_pos[0], _pos[1], _pos[2] + desired_height], dtype=np.float64)
    if not np.all(np.isfinite(goal_pos)):
        goal_pos = fallback_goal

    goal_att_rpy = np.asarray(desired_att_rpy, dtype=np.float64)
    if goal_att_rpy.size != 3 or not np.all(np.isfinite(goal_att_rpy)):
        goal_att_rpy = np.zeros((3,), dtype=np.float64)
    # 构建当前状态
    current_state = np.array([_pos[0], _pos[1], _pos[2], quat[3], quat[0], quat[1], quat[2], _vel[0], _vel[1], _vel[2], omega[0], omega[1], omega[2]])
    # MPC tracks AprilTag-center hover target with direct policy attitude.
    goal_quat = euler_xyz_to_quat_wxyz(goal_att_rpy[0], goal_att_rpy[1], goal_att_rpy[2])
    goal_state = np.array(
        [
            goal_pos[0],
            goal_pos[1],
            goal_pos[2],
            goal_quat[0],
            goal_quat[1],
            goal_quat[2],
            goal_quat[3],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    # NMPC Update
    _dt, _control = controller.nmpc_state_control(current_state, goal_state)
    d.actuator('motor1').ctrl[0] = calc_motor_input(_control[0])
    d.actuator('motor2').ctrl[0] = calc_motor_input(_control[1])
    d.actuator('motor3').ctrl[0] = calc_motor_input(_control[2])
    d.actuator('motor4').ctrl[0] = calc_motor_input(_control[3])

    log_count += 1
    if log_count >= 50:
        log_count = 0
        if apriltag_pose is not None:
            raw_str = np.array2string(raw_action_vec, precision=3, suppress_small=True)
            att_deg = np.degrees(goal_att_rpy)
            print(
                f"policy_raw={raw_str}, h_cmd={desired_height:.3f}, "
                f"att_cmd_deg={att_deg}, goal_pos={goal_pos}, "
                f"tag_detected={apriltag_pose['detected']}, tag_cam_pos={apriltag_pose['position']}, "
                f"wind={wind_disturbance.current_wind}, wind_force={wind_disturbance.current_force}"
            )


def _find_mjpython_executable():
    local = Path(sys.executable).with_name("mjpython")
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which("mjpython")


def _ensure_dyld_python_lib_path_for_mjpython():
    if sys.platform != "darwin":
        return

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.base_prefix) / "lib",
        Path(sys.prefix) / "lib",
    ]

    lib_dirs = []
    for lib_dir in candidates:
        if (lib_dir / f"libpython{ver}.dylib").exists():
            lib_dirs.append(str(lib_dir))

    if not lib_dirs:
        return

    old = os.getenv("DYLD_FALLBACK_LIBRARY_PATH", "")
    merged = list(lib_dirs)
    if old:
        merged.extend([p for p in old.split(":") if p])

    dedup = []
    seen = set()
    for p in merged:
        if p not in seen:
            seen.add(p)
            dedup.append(p)

    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dedup)


def _reexec_under_mjpython_if_needed():
    if sys.platform != "darwin":
        return
    if os.getenv("MUJOCO_REEXEC_DONE") == "1":
        return

    # Default to normal python on macOS: OpenCV HighGUI windows are often unstable under mjpython.
    if os.getenv("MUJOCO_USE_MJPYTHON", "0") != "1":
        return

    mjpython = _find_mjpython_executable()
    if mjpython is None:
        print("[main] WARNING: launch_passive on macOS requires mjpython, but none was found in this environment.")
        print("[main] Please run: uv run mjpython main.py")
        return

    _ensure_dyld_python_lib_path_for_mjpython()
    os.environ["MUJOCO_REEXEC_DONE"] = "1"
    script_path = str(Path(__file__).resolve())
    print(f"[main] Relaunching with mjpython: {mjpython}")
    os.execv(mjpython, [mjpython, script_path, *sys.argv[1:]])

if __name__ == '__main__':
    m = None
    d = None
    try:
        _reexec_under_mjpython_if_needed()

        # On macOS normal python, launch_passive is unavailable and cv2 windows are unstable under mjpython.
        # Use viewer.launch directly so MuJoCo + OpenCV windows can coexist.
        if sys.platform == "darwin" and os.getenv("MUJOCO_USE_MJPYTHON", "0") != "1":
            print("[main] macOS python runtime: using viewer.launch (OpenCV window compatible)")
            bottom_cam_display.start()
            try:
                viewer.launch(loader=load_callback_for_viewer_launch)
            except Exception as exc:
                print(f"[main] viewer.launch failed: {exc}")
                bottom_cam_display.close()
                run_headless_bottom_camera_loop()
            finally:
                bottom_cam_display.close()
            sys.exit(0)

        m, d = setup_simulation(use_external_display=False)
        try:
            with viewer.launch_passive(m, d) as v:
                while v.is_running():
                    step_start = time.time()

                    step_control(m, d)
                    mujoco.mj_step(m, d)
                    apriltag_tracker.show_last_frame()

                    v.sync()

                    remaining = float(m.opt.timestep) - (time.time() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except RuntimeError as exc:
            msg = str(exc)
            if "launch_passive" in msg and "mjpython" in msg:
                print("[main] launch_passive unavailable in this runtime; falling back to viewer.launch")
                apriltag_tracker.close()
                bottom_cam_display.start()
                viewer.launch(loader=load_callback_for_viewer_launch)
            else:
                raise
    finally:
        bottom_cam_display.close()
        apriltag_tracker.close()
        cv2.destroyAllWindows()
