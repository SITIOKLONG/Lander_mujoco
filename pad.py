import numpy as np
import math
import mujoco

from utils_math import *


DEG2RAD = math.pi / 180.0

class WavePadMotion:
    """Spectral + OU + smoothing wave motion for a MuJoCo mocap body."""

    def __init__(
        self,
        body_name="wave_pad",
        amp_heave=0.1,
        amp_roll_deg=30.0,
        amp_pitch_deg=30.0,
        freq_min=0.05,
        freq_max=0.1,
        n_sines=6,
        ou_rho=0.99,
        ou_sigma=0.05,
        smoother_alpha=0.01,
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
        d.mocap_quat[self.mocap_id] = quat_multiply(self.base_quat, dq)

        curr_pos = d.mocap_pos[self.mocap_id].copy()
        curr_quat = d.mocap_quat[self.mocap_id].copy()
        self.current_lin_vel = (curr_pos - prev_pos) / self.dt
        self.current_ang_vel = quat_to_ang_vel_wxyz(prev_quat, curr_quat, self.dt)
        self.prev_pos = curr_pos
        self.prev_quat = curr_quat

    # def get_pose(self, d):
    #     return d.mocap_pos[self.mocap_id].copy(), d.mocap_quat[self.mocap_id].copy()

    # def get_phase_features(self):
    #     """Return [cos(phase_i), sin(phase_i)] for each active wave component."""
    #     k = max(int(self.n_sines), 1)
    #     if self.omegas is None or self.phases is None:
    #         return np.zeros((2 * k,), dtype=np.float64)
    #     angles = self.omegas * self.t + self.phases
    #     return np.concatenate([np.cos(angles), np.sin(angles)]).astype(np.float64)
