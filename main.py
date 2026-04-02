# 20250220 Wakkk
# Quadrotor SE3 Control Demo
import math
from pathlib import Path
import mujoco 
import mujoco.viewer as viewer 
import numpy as np
import torch
from nmpc_controller import NMPC_Controller

# 新建NMPC控制器
controller = NMPC_Controller()

DEG2RAD = math.pi / 180.0


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


class WindDisturbance:
    """Apply smooth wind disturbance as external force/torque on the drone body."""

    def __init__(
        self,
        body_name="cf2",
        base_wind=(1.6, 0.3, 0.05),
        gust_amp=(0.5, 0.35, 0.1),
        gust_freq_hz=(0.03, 0.05, 0.04),
        drag_coeff=0.015,
        ou_rho=0.998,
        ou_sigma=0.03,
        torque_sigma=1.2e-4,
        max_force=0.08,
        max_torque=0.0012,
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
    ):
        self.model_path = Path(model_path)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.output_is_normalized = bool(output_is_normalized)
        self.action_scale = float(action_scale)
        self.action_bias = float(action_bias)

        self.model = None
        self.prev_slider_pos = None
        self.device = torch.device("cpu")
        self.expected_input_dim = None
        self._warned_obs_pad = False
        self._warned_obs_trunc = False

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.model_path}")
        # choose device: prefer CUDA, then MPS (Apple), then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            mps_avail = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_avail is not None and getattr(torch.backends.mps, "is_available", lambda: False)():
                self.device = torch.device("mps")
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

    def reset(self):
        self.prev_slider_pos = None

    def build_obs(self, d, wave_pad, dt):
        pad_pos, pad_quat = wave_pad.get_pose(d)
        pad_rot = quat_to_rotmat_wxyz(pad_quat)
        pad_normal = pad_rot[:, 2]

        drone_pos = d.qpos[0:3].astype(np.float64)
        rel = drone_pos - pad_pos
        pad_height = float(np.dot(rel, pad_normal))

        slider_pos = -pad_height
        if self.prev_slider_pos is None or dt <= 0.0:
            slider_vel = 0.0
        else:
            slider_vel = (slider_pos - self.prev_slider_pos) / dt
        self.prev_slider_pos = slider_pos

        pad_ang_vel = wave_pad.current_ang_vel.astype(np.float64)

        obs = np.array(
            [
                slider_pos,
                slider_vel,
                pad_quat[0],
                pad_quat[1],
                pad_quat[2],
                pad_quat[3],
                pad_ang_vel[0],
                pad_ang_vel[1],
                pad_ang_vel[2],
            ],
            dtype=np.float32,
        )
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

    def infer_height(self, obs):
        if self.model is None:
            raise RuntimeError("Policy model is not loaded")

        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            y = self.model(x)

        if isinstance(y, (tuple, list)):
            y = y[0]
        y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        raw = float(y_np[0])

        if self.output_is_normalized:
            mapped = 0.5 * (raw + 1.0) * (self.max_height - self.min_height) + self.min_height
        else:
            mapped = self.action_scale * raw + self.action_bias

        height_cmd = float(np.clip(mapped, self.min_height, self.max_height))
        return height_cmd, raw


class AprilTagPoseTracker:
    """Compute tag pose in bottom camera frame as detection proxy."""

    def __init__(self, camera_name="bottom_cam", tag_site_name="apriltag_site"):
        self.camera_name = camera_name
        self.tag_site_name = tag_site_name
        self.cam_id = None
        self.site_id = None
        self.last_pose = None

    def bind_model(self, m):
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.tag_site_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' was not found in MuJoCo model")
        if self.site_id < 0:
            raise ValueError(f"Site '{self.tag_site_name}' was not found in MuJoCo model")

    def detect_pose(self, d):
        if self.cam_id is None or self.site_id is None:
            return None

        cam_pos = d.cam_xpos[self.cam_id].copy()
        cam_rot = d.cam_xmat[self.cam_id].reshape(3, 3).copy()

        tag_pos = d.site_xpos[self.site_id].copy()
        tag_rot = d.site_xmat[self.site_id].reshape(3, 3).copy()

        rot_c_w = cam_rot.T
        tag_pos_c = rot_c_w @ (tag_pos - cam_pos)
        tag_rot_c = rot_c_w @ tag_rot
        tag_quat_c = rotmat_to_quat_wxyz(tag_rot_c)

        detected = bool(tag_pos_c[2] < 0.0)
        self.last_pose = {
            "detected": detected,
            "position": tag_pos_c,
            "quat_wxyz": tag_quat_c,
        }
        return self.last_pose


wave_pad_motion = WavePadMotion()
wind_disturbance = WindDisturbance()
height_policy = HeightPolicyJIT(model_path=Path(__file__).resolve().parent / "models" / "model_9999.pt")
apriltag_tracker = AprilTagPoseTracker()

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

# 加载模型回调函数
def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('./crazyfile/scene.xml')
    d = mujoco.MjData(m)
    if m is not None:
        wave_pad_motion.bind_model(m, d)
        wind_disturbance.bind_model(m, d)
        height_policy.load()
        height_policy.reset()
        apriltag_tracker.bind_model(m)
        mujoco.set_mjcb_control(lambda m, d: control_callback(m, d))  # 设置控制回调函数
    return m, d

# 根据四元数计算旋转矩阵
def rotation_matrix(q0, q1, q2, q3):
    _row0 = np.array([1-2*(q2**2)-2*(q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    _row1 = np.array([2*(q1*q2+q0*q3), 1-2*(q1**2)-2*(q3**2), 2*(q2*q3-q0*q1)])
    _row2 = np.array([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2)-2*(q2**2)])
    return np.vstack((_row0, _row1, _row2))

log_count = 0
def control_callback(m, d):
    global log_count, gravity, mass, controller, wave_pad_motion, wind_disturbance, height_policy, apriltag_tracker
    wave_pad_motion.step(d)
    wind_disturbance.step_and_apply(d)

    obs = height_policy.build_obs(d, wave_pad_motion, float(m.opt.timestep))
    desired_height, raw_action = height_policy.infer_height(obs)

    pad_pos, _ = wave_pad_motion.get_pose(d)
    apriltag_pose = apriltag_tracker.detect_pose(d)

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
    # 构建当前状态
    current_state = np.array([_pos[0], _pos[1], _pos[2], quat[3], quat[0], quat[1], quat[2], _vel[0], _vel[1], _vel[2], omega[0], omega[1], omega[2]])
    # 位置控制模式 目标位点
    goal_position = np.array([pad_pos[0], pad_pos[1], pad_pos[2] + desired_height])

    # NMPC Update
    _dt, _control = controller.nmpc_position_control(current_state, goal_position)
    d.actuator('motor1').ctrl[0] = calc_motor_input(_control[0])
    d.actuator('motor2').ctrl[0] = calc_motor_input(_control[1])
    d.actuator('motor3').ctrl[0] = calc_motor_input(_control[2])
    d.actuator('motor4').ctrl[0] = calc_motor_input(_control[3])

    log_count += 1
    if log_count >= 50:
        log_count = 0
        if apriltag_pose is not None:
            print(
                f"policy_raw={raw_action:.4f}, h_cmd={desired_height:.3f}, "
                f"tag_detected={apriltag_pose['detected']}, tag_cam_pos={apriltag_pose['position']}, "
                f"wind={wind_disturbance.current_wind}, wind_force={wind_disturbance.current_force}"
            )

if __name__ == '__main__':
    viewer.launch(loader=load_callback)
