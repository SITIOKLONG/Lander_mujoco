import mujoco
from mujoco.glfw import glfw
import cv2
from cv2 import solvePnP
import numpy as np
from pupil_apriltags import Detector
import torch

import os
os.environ["MUJOCO_GL"] = "glfw"

from controller import control_callback
from pad import WavePadMotion
from utils_math import *

# 100Hz 10ms 0.01s
dt_sim = 0.01
# dt_sim = 1/120
decimation = 2
dt_control = dt_sim * decimation


policy = torch.jit.load("./policy/policy_3.pt")
policy.eval()
device = torch.device("mps")

# 设定阈值 (米)
xy_threshold = 0.3
# 基础期望（不启用策略时的默认值）
default_height = 1.0

def load_model(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path("./crazyfile/scene.xml")
    d = mujoco.MjData(m)
    mujoco.set_mjcb_control(None)
    m.opt.timestep = dt_sim

    wave_pad = WavePadMotion(body_name="wave_pad", trans_vel=(0.2, 0.2))
    wave_pad.bind_model(m, d)
    return m, d, wave_pad

def check_contact(data, body1_id, body2_id):
    """检查两个 body 是否接触（通过 body id）"""
    for i in range(data.ncon):
        con = data.contact[i]
        # 接触的 geom 属于哪个 body
        geom1_body = model.geom_bodyid[con.geom1]
        geom2_body = model.geom_bodyid[con.geom2]
        if (geom1_body == body1_id and geom2_body == body2_id) or \
           (geom1_body == body2_id and geom2_body == body1_id):
            return True
    return False

# cv apriltag
detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, refine_edges=1)
TAG_SIZE = 0.1
half_s = TAG_SIZE / 2
object_points = np.array([
    [-half_s, half_s, 0],
    [half_s, half_s, 0],
    [half_s, -half_s, 0],
    [-half_s, -half_s, 0],
], dtype=np.float32)
# 1920*1080 120fov
CAMERA_MATRIX = np.array([
    [554.26, 0, 960],
    [0, 554.26, 540],
    [0, 0, 1]
], dtype=np.float32)
# DIST_COEFFS = np.array((5, 1), dtype=np.float32)
DIST_COEFFS = np.zeros(4, dtype=np.float32)   # 假设使用 k1,k2,p1,p2 模型


P_cam_in_body = torch.tensor([0.0, 0.0, -0.05], dtype=torch.float32)
R_cam_to_body = torch.tensor([  # TODO
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=torch.float32)
control_action = torch.zeros(1,1)
P_body_from_tag_w = torch.zeros(1,3)

log_count = 0
# pad_quat_prev = torch.tensor([1,0,0,0])
# filtered_pad_ang_vel_w = torch.tensor([0,0,])
# first_apriltag_detected = False
reached_count = 0
higher_error_count = 0
def cv_apriltag(raw_image_, m, d):
    global log_count, control_action, tag_detected, policy, device, reached_count, P_body_from_tag_w
    rgb_drone_camera = cv2.cvtColor(cv2.flip(raw_image_, 0), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_drone_camera, cv2.COLOR_BGR2GRAY)
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
                tag_detected = True
                cv2.drawFrameAxes(rgb_drone_camera, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)

                # calculate pos error for controller
                P_tag_in_cam = torch.from_numpy(tvec.ravel()).float()
                P_tag_in_body = R_cam_to_body @ P_tag_in_cam + P_cam_in_body
                # print(f"P_tag_in_body: {P_tag_in_body}")


                _sensor = torch.from_numpy(d.sensordata).float()
                quat  = _sensor[6:10]         # [w, x, y, z]
                R_body_to_world = rotation_matrix(
                    quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
                ).float()
                P_tag_from_body_w = R_body_to_world @ P_tag_in_body
                # print(f"P_tag_from_body_w: {P_tag_from_body_w}")

                P_des_body_from_pad_w = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
                P_body_from_tag_w = -P_tag_from_body_w


                # get pad quat for nn model obs
                R_tag_to_cam, _ = cv2.Rodrigues(rvec)
                R_tag_to_cam = torch.from_numpy(R_tag_to_cam).float()   # 转 torch 张量方便后续运算
                R_tag_to_world = R_body_to_world @ R_cam_to_body @ R_tag_to_cam
                pitch = torch.asin(torch.clamp(-R_tag_to_world[2, 0], -1.0, 1.0))
                roll  = torch.atan2(R_tag_to_world[2, 1], R_tag_to_world[2, 2])
                # pad_quat = rotation_matrix_to_quat_wxyz(R_tag_to_world)
                # pad_ang_vel_w = quat_to_ang_vel_wxyz(pad_quat_prev, pad_quat, dt_sim)
                # if first_apriltag_detected is False:
                #     filtered_pad_ang_vel_w = pad_ang_vel_w
                # filtered_pad_ang_vel_w = 0.99 * filtered_pad_ang_vel_w + 0.01 * pad_ang_vel_w

                # print(f"pad_quat: {pad_quat} | pad_quat_last: {pad_quat_prev} | pad_ang_vel_w: {filtered_pad_ang_vel_w}")
                # pad_quat_prev = pad_quat
                # first_apriltag_detected = True

                # print(f"obs: {obs} | action: {action}")

                xy_error = torch.norm(P_body_from_tag_w[:2])   # P_body_from_tag_w 是 [x, y, z]
                if xy_error < xy_threshold:
                    reached_count += 1
                    if reached_count >= 100:   # 0.02 * 100 = 2sec
                        obs = torch.cat([P_body_from_tag_w[2].reshape(1), roll.reshape(1), pitch.reshape(1)]).unsqueeze(0)  # (1, 5)
                        # print(f"obs: {obs}")
                        control_action = 0.2 * policy(obs)  # action scale 0.2
                        # print(f"control_action: {control_action}")
                else:
                    reached_count = 0
                    control_action = torch.zeros(1,1)


            log_count += 1
            if log_count >= 50:
                log_count = 0
                print(f"rvec: {rvec}\n tvec: {tvec}\n")
                pass
    else:
        tag_detected = False



    # cv2.imshow("Drone Camera 1920x1080", rgb_drone_camera)    # bug
    # cv2.waitKey(1)



if __name__ == '__main__':
    glfw.init()
    window = glfw.create_window(1200, 900, "CF2 with Camera", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 加载模型
    model, data, wave_pad = load_model()

    # 创建可视化数据结构
    cam = mujoco.MjvCamera()          # 主视角相机
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True

    # 设置默认相机参数( tracking drone )
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    cam.lookat = np.array([0, 0, 0])          # 相对于 body 的注视点（0 就是 body 中心）
    cam.distance = 2.0                        # 相机距离注视点的距离
    cam.azimuth = 90                          # 在水平面内的角度（度）
    cam.elevation = -30                       # 俯仰角

    # drone camera setting
    camera_name = 'drone_camera'
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    drone_camera = mujoco.MjvCamera()
    drone_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    drone_camera.fixedcamid = camera_id
    small_width, small_height = 640, 480  # 可根据需要调整
    full_width, full_height = 1920, 1080
    full_viewport = mujoco.MjrRect(0, 0, full_width, full_height)
    full_rgb = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    full_depth = np.zeros((full_height, full_width), dtype=np.float32)

    cf2_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cf2')
    wave_pad_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wave_pad')

    # main loop
    while not glfw.window_should_close(window):
        # get main viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # high resoluation drone camera for cv2
        mujoco.mjr_render(full_viewport, scene, context)
        mujoco.mjr_readPixels(full_rgb, full_depth, full_viewport, context)

        global ctrl_pos_error, tag_detected
        cv_apriltag(full_rgb, model, data)

        # update and render main_viewport
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(main_viewport, scene, context)

        # update and render mujoco drone_camera_viewport ( right top of main_viewport)
        loc_x = viewport_width - small_width
        loc_y = viewport_height - small_height
        drone_camera_viewport = mujoco.MjrRect(loc_x, loc_y, small_width, small_height)
        mujoco.mjv_updateScene(model, data, opt, None, drone_camera,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(drone_camera_viewport, scene, context)

        # 交换缓冲区、处理事件
        glfw.swap_buffers(window)
        glfw.poll_events()

        # update sim
        for _ in range(decimation):
            wave_pad.step(data)
            mujoco.mj_step(model, data)

        # 控制周期
        if check_contact(data, cf2_body, wave_pad_body):
            # 所有电机 ctrl 置零
            data.actuator('motor1').ctrl[0] = 0
            data.actuator('motor2').ctrl[0] = 0
            data.actuator('motor3').ctrl[0] = 0
            data.actuator('motor4').ctrl[0] = 0
        else:
            control_callback(model, data, dt_control, P_body_from_tag_w, control_action)

    cv2.destroyAllWindows()
    glfw.terminate()
