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
arm_length = 0.065 / 2.0      # m
max_thrust = 0.1573         # N
max_torque = 3.842e-03      # Nm

# 100Hz 10ms 0.01s
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


def control_callback(m, d):
    """
    motor 1 CW     X       motor 2 CCW
            \\            /
              \\        /
                \\    /
    Y             \\/
                  /\\
                /    \\
              /        \\
            /            \\
    motor 4 CCW             motor 3  CW
    """
    global log_count
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

    # debug ctrl
    d.actuator('motor1').ctrl[0] = calc_motor_input(15)
    d.actuator('motor2').ctrl[0] = calc_motor_input(15)
    d.actuator('motor3').ctrl[0] = calc_motor_input(15)
    d.actuator('motor4').ctrl[0] = calc_motor_input(15)

    # 日志输出（每 50 次控制循环）
    log_count += 1
    if log_count >= 50:
        log_count = 0
        print(f"Time: {data.time:.3f}, Pos: {data.qpos[0:3]}, Ctrl: {data.ctrl[:]}")
        pass


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
    control_time_accum = 0.0
    sim_time = 0.0
    while not glfw.window_should_close(window):
        # get main viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # update sim
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # 控制周期到了，执行控制
        # 实际 data.time 是物理时间，控制依赖物理时间
        control_callback(model, data)

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

        # height resoluation drone camera for cv2
        mujoco.mjr_render(full_viewport, scene, context)
        mujoco.mjr_readPixels(full_rgb, full_depth, full_viewport, context)
        rgb_drone_camera = cv2.cvtColor(cv2.flip(full_rgb, 0), cv2.COLOR_RGB2BGR)
        # cv2.imshow("Drone Camera 1920x1080", rgb_drone_camera)
        # cv2.waitKey(1)

    cv2.destroyAllWindows()
    glfw.terminate()
