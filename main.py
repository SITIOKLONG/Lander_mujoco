from src.math_utils import quaternion_to_R, rot_to_rpy_zxy
import numpy as np
import cvxpy as cp
import cv2
from cv2 import solvePnP
import glfw
import mujoco
import mujoco.viewer
from pupil_apriltags import Detector

import os
os.environ["MUJOCO_GL"] = "egl"


# params
dt_sim = 1/120
control_decimation = 2
dt_control = control_decimation * dt_sim
control_step = 0

# drone params
g0 = 9.8066
mq = 33e-3
Ixx = 1.395e-5
Iyy = 1.395e-5
Izz = 2.173e-5
Cd = 7.9379e-06
Ct = 3.25e-4
dq = 65e-3
l = dq/2

A = np.eye(6)
A[0, 3] = A[1, 4] = A[2, 5] = dt_control

B = np.zeros((6, 3))    # u = [angle_x, angle_y, acceleration_z]
B[3, 0] = g0 * dt_control
B[4, 1] = g0 * dt_control
B[5, 2] = dt_control
B[0, 0] = 0.5 * g0 * dt_control**2
B[1, 1] = 0.5 * g0 * dt_control**2
B[2, 2] = 0.5 * dt_control**2

# state: [px, py, pz, vx, vy, vz]
Q = np.diag([10, 10, 200, 20.0, 20.0, 50.0])
# control: [roll_cmd, pitch_cmd, accel_z_cmd]
R = np.diag([0.5, 0.5, 0.05])

# mixer gain
Kp_roll = 0.005
Kd_roll = 0.00011
Kp_pitch = 0.005
Kd_pitch = 0.00011
Kp_yaw = 0.005
Kd_yaw = 0.00011


def build_B(psi, dt, g):
    c = np.cos(psi)
    s = np.sin(psi)
    B = np.zeros((6, 3))

    B[0, 0] = 0.5 * g * dt**2 * c
    B[0, 1] = 0.5 * g * dt**2 * s
    B[1, 0] = 0.5 * g * dt**2 * s
    B[1, 1] = -0.5 * g * dt**2 * c
    B[2, 2] = 0.5 * dt**2

    B[3, 0] = g * dt * c
    B[3, 1] = g * dt * s
    B[4, 0] = g * dt * s
    B[4, 1] = -g * dt * c
    B[5, 2] = dt
    return B


def solve_mpc(state, target_height, Q, R, A, B, dt, N=50):
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

        # print("A: ", A, "B: ", B)

        # physical
        constraints += [u[0, t] <= 0.3, u[0, t] >= - 0.3]  # roll angle
        constraints += [u[1, t] <= 0.3, u[1, t] >= - 0.3]  # pitch angle
        constraints += [u[2, t] <= 5.0, u[2, t] >= - 2.0]  # accele_z

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    # if prob.status not in ["optimal", "optimal_inaccurate"]:
    #     print("MPC not optimal", prob.status)
    #     return np.zeros(3)  # last

    # print("prob: ", prob.status)

    return u[:, 0].value


# openGL drone track camera
resolution = (1920, 1080)
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(
    resolution[0], resolution[1], "Offscreen", None, None)
glfw.make_context_current(window)

# MuJoCo setup
model = mujoco.MjModel.from_xml_path("./crazyfile/scene.xml")
data = mujoco.MjData(model)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# set camera
camera_name = "track"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
if camera_id != -1:
    print("camera_id", camera_id)
    camera.fixedcamid = camera_id

# frame buffer
frambuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# april tag
detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0,
                    refine_edges=1)
TAG_SIZE = 0.1
half_s = TAG_SIZE / 2
object_points = np.array([
    [-half_s, half_s, 0],
    [half_s, half_s, 0],
    [half_s, -half_s, 0],
    [-half_s, -half_s, 0],
], dtype=np.float32)
CAMERA_MATRIX = np.array([
    [554.26, 0, 960],
    [0, 554.26, 540],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# setup params
last_pos_rel = None
vel_rel = np.zeros(3)
last_u_cmd = np.array([0.0, 0.0, 0.0])
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        control_step += 1
        # print("control_step: ", control_step)
        viewer.sync()

        if control_step % control_decimation == 0:
            # track camera cv2
            viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(
            ), mujoco.MjvPerturb(), camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewport, context)
            bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)

            # detector apriltag
            gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray_image)
            if len(tags) > 0:
                for tag in tags:
                    # print(f"detected tag ID: {tag.tag_id}")
                    img_points = np.array(tag.corners, dtype=np.float32)
                    success, rvec, tvec = solvePnP(
                        object_points,
                        img_points,
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        cv2.drawFrameAxes(bgr, CAMERA_MATRIX,
                                          DIST_COEFFS, rvec, tvec, 0.1)

                # get quat
                quat = data.sensor("body_quat").data
                p, q, r = data.sensor("body_gyro").data
                R_body_to_world = quaternion_to_R(quat).T       # ! .T
                phi, theta, psi = rot_to_rpy_zxy(R_body_to_world)
                p_cam_in_body = np.array([0.0, 0.0, -0.01])
                R_cam_to_body = np.eye(3)
                p_tag_in_boat = np.array([0.0, 0.0, 0.05])

                # TODO boat attutite

                p_tag_in_body = R_cam_to_body @ tvec.flatten() + p_cam_in_body
                # p_drone_in_world = R_body_to_world @ p_tag_in_body + p_tag_in_boat
                p_drone_in_world = data.body("cf2").xpos.copy()  # TODO test

                # solve MPC
                # est velocity TODO
                pos_rel = p_drone_in_world
                if last_pos_rel is not None:
                    vel_rel = (pos_rel - last_pos_rel) / dt_control
                else:
                    vel_rel = np.zeros(3)

                state = np.concatenate([pos_rel, vel_rel])

                target_height = 1.0     # TODO height
                B = build_B(psi, dt_control, g0)
                u_cmd = solve_mpc(
                    state, target_height, Q, R, A, B, dt_control, N=30)
                if u_cmd is not None and not np.all(u_cmd == 0):
                    last_u_cmd = u_cmd.copy()
                else:
                    u_cmd = last_u_cmd
                    print("MPC failed")
                theta_des, phi_des, accel_z_cmd = u_cmd

                e_phi = phi_des - phi
                e_theta = theta_des - theta
                psi_des = 0
                e_psi = psi_des - psi

                # WTF TODO error
                tau_theta = Kp_roll * e_phi + Kd_roll * (0 - p)
                tau_phi = Kp_pitch * e_theta + Kd_pitch * (0 - q)
                tau_psi = Kp_yaw * e_psi + Kd_yaw * (0 - r)

                T_total = mq * (g0 + accel_z_cmd)

                # kappa = 1e-9
                F1 = T_total/4 + tau_theta / \
                    (2*l) - tau_phi/(2*l) - tau_psi/(4*l)
                F2 = T_total/4 - tau_theta / \
                    (2*l) - tau_phi/(2*l) + tau_psi/(4*l)
                F3 = T_total/4 - tau_theta / \
                    (2*l) + tau_phi/(2*l) - tau_psi/(4*l)
                F4 = T_total/4 + tau_theta / \
                    (2*l) + tau_phi/(2*l) + tau_psi/(4*l)

                des_forces = np.array([F1, F2, F3, F4])
                data.ctrl[:] = np.clip(des_forces, 0, 0.1573) / 0.1573

                # debug
                print("tvec: ", tvec)
                print("p_tag_in_body: ", p_tag_in_body,
                      "p_drone_in_world", p_drone_in_world)
                # print("mujoco_cf2_xpos:", data.body("cf2").xpos)
                # print("mujoco_cf2_quat:", data.body("cf2").xquat)

                print("throttles: ", data.ctrl[:])
                print("mujoco actuator force", data.actuator_force)
                print(f"u_cmd: roll={u_cmd[0]:.3f}, pitch={
                      u_cmd[1]:.3f}, accel_z={u_cmd[2]:.3f}")
                print(f"state: pos={state[:3]}, vel={state[3:]}")

                last_pos_rel = pos_rel

            # show track camera
            bgr = cv2.resize(bgr, (640, 480))
            cv2.imshow('MuJoCo Camera Output', bgr)
        # else:
            #     # lost tag
            #     # data.ctrl[:] = mq * g0 / 4
            # print("no tag")
            #     data.ctrl[:] = 1

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
glfw.terminate()
del context
del scene
