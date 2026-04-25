import mujoco
from mujoco.glfw import glfw
import cv2
from cv2 import solvePnP
import numpy as np
from pupil_apriltags import Detector

import os
os.environ["MUJOCO_GL"] = "glfw"

from controller import control_callback
from pad import WavePadMotion

# 100Hz 10ms 0.01s
dt_sim = 0.01
decimation = 2
dt_control = dt_sim * decimation

def load_model(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path("./crazyfile/scene.xml")
    d = mujoco.MjData(m)
    mujoco.set_mjcb_control(None)
    m.opt.timestep = dt_sim

    wave_pad = WavePadMotion(body_name="wave_pad")
    wave_pad.bind_model(m, d)
    return m, d, wave_pad

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
CAMERA_MATRIX = np.array([
    [554.26, 0, 960],
    [0, 554.26, 540],
    [0, 0, 1]
], dtype=np.float32)
# DIST_COEFFS = np.array((5, 1), dtype=np.float32)
DIST_COEFFS = np.zeros(4, dtype=np.float32)   # 假设使用 k1,k2,p1,p2 模型


log_count = 0
def cv_apriltag(raw_image_):
    global log_count
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
                cv2.drawFrameAxes(rgb_drone_camera, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)

                log_count += 1
                if log_count >= 50:
                    log_count = 0
                    print(f"rvec: {rvec}\n tvec: {tvec}\n")
                    pass


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
    # opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    # 设置默认相机参数( tracking drone )
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
    cam.lookat = np.array([0, 0, 0])          # 相对于 body 的注视点（0 就是 body 中心）
    cam.distance = 2.0                        # 相机距离注视点的距离
    cam.azimuth = 90                          # 在水平面内的角度（度）
    cam.elevation = -60                       # 俯仰角

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

    # main loop
    while not glfw.window_should_close(window):
        # get main viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # high resoluation drone camera for cv2
        mujoco.mjr_render(full_viewport, scene, context)
        mujoco.mjr_readPixels(full_rgb, full_depth, full_viewport, context)

        cv_apriltag(full_rgb)

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

        # 控制周期到了，执行控制
        # 实际 data.time 是物理时间，控制依赖物理时间
        control_callback(model, data)

    cv2.destroyAllWindows()
    glfw.terminate()
