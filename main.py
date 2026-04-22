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


def solve_mpc(current_error, target_height):
    N = 20

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

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return u[:, 0].value


# params
dt = 1/120
control_decimation = 2

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
Q = np.diag([100.0, 100.0, 15.0, 1.0, 1.0, 1.0])
# control: [roll_cmd, pitch_cmd, accel_z_cmd]
R = np.diag([0.1, 0.1, 0.05])

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

last_tvec = None
vel = np.zeros(3)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

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

            # est velocity
            curr_tvec = tvec.flatten()      # [x, y, z]
            dx = curr_tvec[0]
            dy = curr_tvec[1]
            dz = curr_tvec[2]
            if last_tvec is not None:
                vel = (curr_tvec - last_tvec) / dt
            else:
                vel = np.zeros(3)

            state = np.concatenate([[-dx, -dy, dz], vel])

            control_cmd = solve_mpc(
                state, target_height=1.0)  # TODO height
            roll = control_cmd[0]
            pitch = control_cmd[1]
            thrust_z = control_cmd[2]

            hover_thrust = (mq * g0) / 4

            data.ctrl[0] = hover_thrust + roll - pitch
            data.ctrl[1] = hover_thrust - roll - pitch
            data.ctrl[2] = hover_thrust - roll + pitch
            data.ctrl[3] = hover_thrust + roll + pitch

            last_tvec = curr_tvec
        else:
            # lost tag
            data.ctrl[:] = hover_thrust

        # show track camera
        bgr = cv2.resize(bgr, (640, 480))
        cv2.imshow('MuJoCo Camera Output', bgr)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
glfw.terminate()
del context
del scene
