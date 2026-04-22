import cvxpy as cp
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as Rot

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

A = np.eye(6)   # A simple COM model
A[0, 3] = A[1, 4] = A[2, 5] = dt

B = np.zeros((6, 3))    # u = [angle_x, angle_y, acceleration_z]
B[3, 0] = g0 * dt
B[4, 1] = g0 * dt
B[5, 2] = dt
B[0, 0] = 0.5 * g0 * dt**2
B[1, 1] = 0.5 * g0 * dt**2
B[2, 2] = 0.5 * dt**2

# state: [px, py, pz, vx, vy, vz]
Q = np.diag([100.0, 100.0, 5.0, 1.0, 1.0, 1.0])
# control: [roll_cmd, pitch_cmd, accel_z_cmd]
R = np.diag([0.1, 0.1, 0.05])


def mixer(thrust_total, Mx, My, Mz):
    A = np.array([
        [1, 1, 1, 1],
        [l, -l, -l, l],
        [l, l, -l, -l],
        [Ct, -Ct, Ct, -Ct],
    ])
    b = np.array([thrust_total, Mx, My, Mz])
    forces = np.linalg.solve(A, b)

    throttles = np.clip(forces / Ct, 0, 1)
    return throttles


def solve_mpc(state, target_height, N=10):
    u = cp.Variable((3, N))     # control: pitch, roll, thrust
    x = cp.Variable((6, N + 1))     # state: [pos, vel]

    goal = np.array([0, 0, target_height, 0, 0, 0])

    cost = 0
    constraints = [x[:, 0] == state]

    for t in range(N):
        cost += cp.quad_form(x[:, t] - goal, Q)
        cost += cp.quad_form(u[:, t], R)

        # dynamics
        constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]

        # physical
        constraints += [u[0, t] <= 0.5, u[0, t] >= - 0.5]  # roll angle
        constraints += [u[1, t] <= 0.5, u[1, t] >= - 0.5]  # pitch angle
        constraints += [u[2, t] <= 2*g0, u[2, t] >= -2*g0]  # z acc

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return np.zeros(3)

    print("u: ", u[:, 0].value)
    return u[:, 0].value


if __name__ == "__main__":
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

        quat = data.body("cf2").xquat
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
