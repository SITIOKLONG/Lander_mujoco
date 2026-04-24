import mujoco
from mujoco.glfw import glfw
import torch
import cv2
import numpy as np

import os
os.environ["MUJOCO_GL"] = "glfw"

gravity = 9.0866        # m/s^2
mass = 0.033            # kg
Ct = 3.25e-4            # N/krpm^2
Cd = 7.9379e-6          # Nm/krpm^2
arm_length = 0.065/2.0      # m
max_thrust = 0.1573         # N
max_torque = 3.842e-03      # Nm

dt_sim = 0.01
decimation = 2
dt_control = dt_sim * decimation

control_counter = 0
log_count = 0

def load_model(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path("./crazyfile/scene.xml")
    d = mujoco.MjData(m)
    mujoco.set_mjcb_control(None)
    
    m.opt.timestep = dt_sim
    return m, d

def get_state(data):
    _pos = data.qpos
    _vel = data.qvel
    _sensor = data.sensordata
    gyro = _sensor[0:3]      # x, y, z
    acc  = _sensor[3:6]
    quat = _sensor[6:10]     # x, y, z, w
    return torch.tensor([
        _pos[0], _pos[1], _pos[2],
        quat[3], quat[0], quat[1], quat[2],   # w, x, y, z
        _vel[0], _vel[1], _vel[2],
        gyro[0], gyro[1], gyro[2]
    ])

def apply_control(m, d, current_state, goal_position):
    # 暂时用固定值，后期替换成你的控制器
    ctrl = 0.51
    d.actuator('motor1').ctrl[0] = ctrl
    d.actuator('motor2').ctrl[0] = ctrl
    d.actuator('motor3').ctrl[0] = ctrl
    d.actuator('motor4').ctrl[0] = ctrl

def control_callback(m, d):
    global log_count, gravity, mass, controller
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
    quat = torch.tensor([quat_x, quat_y, quat_z, quat_w])  # x y z w
    omega = torch.tensor([gyro_x, gyro_y, gyro_z])         # 角速度
    # 构建当前状态
    current_state = torch.tensor([_pos[0], _pos[1], _pos[2], quat[3], quat[0], quat[1], quat[2], _vel[0], _vel[1], _vel[2], omega[0], omega[1], omega[2]])
    # 位置控制模式 目标位点
    goal_position = torch.tensor([0.0, 0.0, 0.5])

    # # NMPC Update
    # _dt, _control = controller.nmpc_position_control(current_state, goal_position)
    # d.actuator('motor1').ctrl[0] = calc_motor_input(_control[0])
    # d.actuator('motor2').ctrl[0] = calc_motor_input(_control[1])
    # d.actuator('motor3').ctrl[0] = calc_motor_input(_control[2])
    # d.actuator('motor4').ctrl[0] = calc_motor_input(_control[3])
    
    # debug
    d.actuator('motor1').ctrl[0] = 0.51
    d.actuator('motor2').ctrl[0] = 0.51
    d.actuator('motor3').ctrl[0] = 0.51
    d.actuator('motor4').ctrl[0] = 0.51

    log_count += 1
    if log_count >= 50:
        log_count = 0
        # 这里输出log

if __name__ == '__main__':
    glfw.init()
    window = glfw.create_window(1200, 900, "CF2 with Camera", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 加载模型
    model, data = load_model()

    # 创建可视化数据结构
    cam = mujoco.MjvCamera()          # 主视角相机
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    # 设置默认相机参数（可选）
    # mujoco.mjv_defaultCamera(cam)
    # mujoco.mjv_defaultOption(opt)
    # cam.azimuth = 90
    # cam.elevation = -30
    # cam.distance = 2
    # cam.lookat = np.array([0.0, 0.0, 0.5])

    # 目标位置（悬停）
    goal_position = torch.tensor([0.0, 0.0, 0.5])

    # 小窗口摄像头设置
    camera_name = 'drone_camera'
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    offcam = mujoco.MjvCamera()
    offcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    offcam.fixedcamid = camera_id

    # 小窗口尺寸（放在主窗口右下角）
    small_width, small_height = 256, 192  # 可根据需要调整

    # 主循环
    control_time_accum = 0.0
    sim_time = 0.0
    while not glfw.window_should_close(window):
        # 计算视口大小
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # --- 物理与控制更新 ---
        # 按 dt_sim 步进，每 decimation 步执行一次控制
        for _ in range(decimation):
            # 注意：data.time 会自动累积，这里我们也可以用 while 循环保证时间差
            # 简单起见，直接 mj_step 一次（它按 model.opt.timestep 步进）
            # 确保 model.opt.timestep == dt_sim
            mujoco.mj_step(model, data)

        # 控制周期到了，执行控制
        control_time_accum += dt_control
        # 实际 data.time 是物理时间，控制依赖物理时间
        current_state = get_state(data)
        apply_control(model, data, current_state, goal_position)

        # 日志输出（每 50 次控制循环）
        log_count += 1
        if log_count >= 50:
            log_count = 0
            # print(f"Time: {data.time:.3f}, Pos: {data.qpos[0:3]}")
            pass

        # --- 渲染主窗口 ---
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(main_viewport, scene, context)

        # --- 小窗摄像头（无人机视角） ---
        # 小窗位置：右下角
        loc_x = viewport_width - small_width
        loc_y = viewport_height - small_height
        off_viewport = mujoco.MjrRect(loc_x, loc_y, small_width, small_height)

        # 更新并渲染 fixed camera 视角
        mujoco.mjv_updateScene(model, data, opt, None, offcam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(off_viewport, scene, context)

        # 用 OpenCV 显示这个小窗图像（可选）
        # 读取像素
        rgb = np.zeros((small_height, small_width, 3), dtype=np.uint8)
        depth = np.zeros((small_height, small_width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, off_viewport, context)
        # OpenCV 显示图像（需要垂直翻转）
        rgb_flipped = cv2.flip(rgb, 0)
        bgr = cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR)
        cv2.imshow("Drone Camera", bgr)
        cv2.waitKey(1)

        # 交换缓冲区、处理事件
        glfw.swap_buffers(window)
        glfw.poll_events()

    cv2.destroyAllWindows()
    glfw.terminate()
