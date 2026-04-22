from __future__ import annotations

import numpy as np
import time
import mujoco
import mujoco.viewer

from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
from .model import QuadParams


class Controller:
    """Controller for quadrotor trajectory tracking."""

    def __init__(self, params: QuadParams) -> None:
        """Initialize controller with quadrotor parameters.

        Args:
            params: Quadrotor physical parameters (mass, inertia, etc.)
        """
        self.params = params

        # TODO: Tune these gains for better performance, if needed you can add more parameters
        self.Kp_pos = np.array([8.45, 8.45, 8.45])
        self.Kd_pos = np.array([4.85, 4.85, 4.85])
        self.Kp_angle = np.array([255.0, 255.0, 255.0])
        self.Kd_angle = np.array([15.0, 15.0, 15.0])

    def reset(self) -> None:
        """Reset controller state (if needed)."""
        pass

    def __call__(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute control outputs (thrust and moments) for the quadrotor.

        Args:
            t: Current time [s]
            s: Current state vector (13,)
               [position, velocity, quaternion, angular_velocity]
            s_des: Desired state vector (11,)
               [position, velocity, acceleration, yaw, yaw_rate]

        Returns:
            F: Total thrust force [N]
            M: Moment vector [Mx, My, Mz] in body frame [N⋅m]
        """
        # Extract quadrotor parameters
        m = self.params.mass
        g = self.params.grav
        I = self.params.I

        # ========================================================================
        # TODO: Implement your controller here
        # ========================================================================

        # position controller
        pos_err = s_des[:3] - s[:3]
        vel_err = s_des[3:6] - s[3:6]
        s_rpy = rot_to_rpy_zxy(quaternion_to_R(s[6:10]))

        acc_cmd = s_des[6:9] + self.Kp_pos * pos_err + self.Kd_pos * vel_err
        u1 = m * np.linalg.norm(np.array([0, 0, g]) + acc_cmd)

        # attitude control
        phi_cmd = (acc_cmd[0] * np.sin(s_rpy[2]) -
                   acc_cmd[1] * np.cos(s_rpy[2])) / g
        theta_cmd = (acc_cmd[0] * np.cos(s_rpy[2]) +
                     acc_cmd[1] * np.sin(s_rpy[2])) / g

        ang_err = wrap_to_pi(np.array([phi_cmd, theta_cmd, s_des[9]]) - s_rpy)
        ang_vel_cmd = np.array([0, 0, s_des[10]])
        ang_vel_err = ang_vel_cmd - s[10:13]
        u2 = I @ (self.Kp_angle * ang_err + self.Kd_angle *
                  ang_vel_err) + np.cross(s[10:13], I @ s[10:13])

        # result
        F: float = u1
        M: np.ndarray = u2

        # ========================================================================
        # End of your implementation
        # ========================================================================

        return F, M


if __name__ == "__main__":

    # params
    dt = 1/120
    control_decimation = 2

    # drone params
    g0 = 9.8066
    mq = 33e-3
    Ixx = 1.395e-5
    Iyy = 1.395e-5
    Izz = 2.173e-5
    Cd = 7.9379e-06     # drag coef
    Ct = 3.25e-4        # thrust coef
    dq = 65e-3          # distance between motors' center
    l = dq/2            # distance between motors' center and axis of rotation

    model = mujoco.MjModel.from_xml_path("./crazyfile/scene.xml")
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data)

    # target_state
    target_height = 0.5
    # TODO input target height

    step = 0
    last_u_mpc = None
    while viewer.is_running():
        step_start = time.time()

        pos = data.body("cf2").xpos.copy()
        vel = data.body("cf2").cvel[3:6].copy()

        quat = data.body("cf2").xquat.copy()
        r = Rot.from_quat(
            [quat[1], quat[2], quat[3], quat[0]])   # wxyz -> xyzw
        euler = r.as_euler('xyz')

        current_state = np.concatenate([pos, vel])

        # MPC
        if step % control_decimation == 0:
            u_mpc = solve_mpc(current_state, target_height, N=50)
        else:
            u_mpc = last_u_mpc
        last_u_mpc = u_mpc

        pitch_cmd, roll_cmd, accel_z_cmd = u_mpc

        kp = 2.0
        kd = 0.05

        angvel_body = data.body("cf2").cvel[:3].copy()

        tau_roll = kp * (roll_cmd - euler[0]) + kd * (0 - angvel_body[0])
        tau_pitch = kp * (pitch_cmd - euler[1]) + kd * (0 - angvel_body[1])
        tau_yaw = kd * (0 - angvel_body[2])

        thrust_total = mq * (g0 + accel_z_cmd)

        throttles = mixer(thrust_total, tau_roll, tau_pitch, tau_yaw)

        data.ctrl[0] = throttles[0]
        data.ctrl[1] = throttles[1]
        data.ctrl[2] = throttles[2]
        data.ctrl[3] = throttles[3]

        mujoco.mj_step(model, data)

        viewer.sync()
        step += 1

        elapsed = time.time() - step_start
        time.sleep(max(0, dt - elapsed))

    viewer.close()
