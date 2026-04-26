import torch
import math
import mujoco
from utils_math import *          # 需保证这些函数已支持 torch.Tensor

DEG2RAD = math.pi / 180.0

class WavePadMotion:
    """Spectral + OU + smoothing wave motion for a MuJoCo mocap body.
       Internal state & calculations are all torch tensors.
    """

    def __init__(
        self,
        body_name="wave_pad",
        amp_heave=0.0,
        amp_roll_deg=30.0,
        amp_pitch_deg=30.0,
        freq_min=0.05,
        freq_max=0.15,
        n_sines=6,
        ou_rho=0.99,
        ou_sigma=0.05,
        smoother_alpha=0.02,
        seed=7,
        trans_vel=(0.0, 0.0)
    ):
        self.body_name = body_name
        self.amp_heave = float(amp_heave)
        self.amp_roll  = float(amp_roll_deg) * DEG2RAD
        self.amp_pitch = float(amp_pitch_deg) * DEG2RAD
        self.freq_min  = float(freq_min)
        self.freq_max  = float(freq_max)
        self.n_sines   = int(n_sines)
        self.ou_rho    = float(ou_rho)
        self.ou_sigma  = float(ou_sigma)
        self.alpha     = float(max(0.0, min(1.0, smoother_alpha)))

        # Replace numpy RNG with torch generator (CPU)
        self.rng = torch.Generator().manual_seed(seed)

        self.mocap_id = None
        self.dt = 0.01
        self.t = 0.0

        # Base pose is initially None; set in bind_model and kept as tensors
        self.base_pos  = None   # torch.Tensor(3,)
        self.base_quat = None   # torch.Tensor(4,)  wxyz

        # Spectrum parameters (tensors)
        self.omegas = None      # torch.Tensor(n_sines,)
        self.phases = None      # torch.Tensor(n_sines,)
        self.amps   = None      # torch.Tensor(3, n_sines)   rows: heave, roll, pitch

        # Filter states
        self.ou_state     = torch.zeros(3, dtype=torch.float32)
        self.smooth_state = torch.zeros(3, dtype=torch.float32)

        # For velocity computation
        self.prev_pos  = None
        self.prev_quat = None
        self.current_lin_vel  = torch.zeros(3, dtype=torch.float32)
        self.current_ang_vel  = torch.zeros(3, dtype=torch.float32)

        # Horizontal translation
        self.trans_vel   = torch.tensor(trans_vel, dtype=torch.float32)
        self.translation = torch.zeros(2, dtype=torch.float32)

    def bind_model(self, m, d):
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id < 0:
            raise ValueError(f"Body '{self.body_name}' was not found in MuJoCo model")

        mocap_id = int(m.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body '{self.body_name}' is not configured as mocap='true'")

        self.mocap_id = mocap_id
        self.dt = float(m.opt.timestep)

        # Store base pose as tensors (copy from mujoco numpy arrays)
        self.base_pos  = torch.from_numpy(d.mocap_pos[self.mocap_id].copy()).float()
        self.base_quat = torch.from_numpy(d.mocap_quat[self.mocap_id].copy()).float()

        self.prev_pos  = self.base_pos.clone()
        self.prev_quat = self.base_quat.clone()
        self.current_lin_vel.zero_()
        self.current_ang_vel.zero_()

        self.reset(d, resample=True)

    def _resample_spectrum(self):
        """Randomly generate frequencies, phases, and spectral amplitudes."""
        freqs_hz = self.freq_min + (self.freq_max - self.freq_min) * torch.rand(
            (self.n_sines,), generator=self.rng, dtype=torch.float32
        )
        self.omegas = 2.0 * math.pi * freqs_hz
        self.phases = 2.0 * math.pi * torch.rand(
            (self.n_sines,), generator=self.rng, dtype=torch.float32
        )

        inv_f = 1.0 / torch.clamp(freqs_hz, min=0.05)
        weights = inv_f / inv_f.sum()   # (n_sines,)

        self.amps = torch.stack([
            self.amp_heave * weights,
            self.amp_roll  * weights,
            self.amp_pitch * weights,
        ], dim=0)   # (3, n_sines)

    def reset(self, d, resample=True):
        self.t = 0.0
        self.translation.zero_()
        self.ou_state.zero_()
        self.smooth_state.zero_()

        if resample or self.omegas is None:
            self._resample_spectrum()

        if self.mocap_id is not None:
            # Write base pose back to mujoco (tensor → numpy)
            d.mocap_pos[self.mocap_id]  = self.base_pos.numpy()
            d.mocap_quat[self.mocap_id] = self.base_quat.numpy()
            self.prev_pos  = self.base_pos.clone()
            self.prev_quat = self.base_quat.clone()
            self.current_lin_vel.zero_()
            self.current_ang_vel.zero_()

    def step(self, d):
        if self.mocap_id is None:
            return

        self.t += self.dt

        # ----- 1. Spectral disturbance -----
        angles = self.omegas * self.t + self.phases   # (n_sines,)
        sines  = torch.sin(angles)                    # (n_sines,)
        disturbance = (self.amps * sines.unsqueeze(0)).sum(dim=1)   # (3,)

        # ----- 2. Ornstein–Uhlenbeck noise -----
        noise_scale = torch.tensor([self.amp_heave, self.amp_roll, self.amp_pitch],
                                   dtype=torch.float32)
        # torch.randn generates standard normal, equivalent to numpy standard_normal
        ou_noise = self.ou_sigma * noise_scale * torch.randn(3, generator=self.rng)
        self.ou_state = self.ou_rho * self.ou_state + ou_noise
        disturbance = disturbance + self.ou_state

        # ----- 3. Exponential smoother -----
        self.smooth_state = (1.0 - self.alpha) * self.smooth_state + self.alpha * disturbance
        z_offset, roll, pitch = self.smooth_state[0], self.smooth_state[1], self.smooth_state[2]

        # ----- 4. Update mocap pose (write back to mujoco) -----
        # Retrieve previous pose from mujoco (used for velocity computation)
        prev_pos  = torch.from_numpy(d.mocap_pos[self.mocap_id].copy()).float()
        prev_quat = torch.from_numpy(d.mocap_quat[self.mocap_id].copy()).float()

        # Horizontal translation accumulation
        self.translation = self.translation + self.trans_vel * self.dt

        new_pos = self.base_pos.clone()
        new_pos[0] += self.translation[0]
        new_pos[1] += self.translation[1]
        new_pos[2] = self.base_pos[2] + z_offset

        # Orientation: base_quat * dq (roll, pitch, 0 yaw)
        dq = euler_xyz_to_quat_wxyz(roll, pitch, 0.0)   # returns tensor (4,)
        new_quat = quat_multiply(self.base_quat, dq)

        # Assign to MuJoCo (tensor → numpy)
        d.mocap_pos[self.mocap_id]  = new_pos.numpy()
        d.mocap_quat[self.mocap_id] = new_quat.numpy()

        # ----- 5. Compute velocity ---
        self.current_lin_vel = (new_pos - prev_pos) / self.dt
        self.current_ang_vel = quat_to_ang_vel_wxyz(prev_quat, new_quat, self.dt)

        # Update stored previous pose
        self.prev_pos  = new_pos.clone()
        self.prev_quat = new_quat.clone()   #     return np.concatenate([np.cos(angles), np.sin(angles)]).astype(np.float64)
