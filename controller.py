import torch
from utils_math import *

g0 = 9.0866        # m/s^2
mq = 0.033            # kg
Ct = 3.25e-4            # N/krpm^2
Cd = 7.9379e-6          # Nm/krpm^2
max_torque = 3.842e-03      # Nm
max_krpm = 22.0        # max krpm
arm_length = 0.065 / 2.0      # m
max_thrust = 0.1573         # N

log_count = 0

def attitude_error_quat(q_curr_: torch.Tensor, q_des_: torch.Tensor) -> torch.Tensor:
    """
    Returns attitude error vector (3,) = 2 * vector part of (q_des^{-1} ⊗ q_curr)
    """
    if torch.dot(q_curr_, q_des_) < 0:
        q_des_ = -q_des_
    q_des_inv = quat_inverse(q_des_)
    q_error = quat_multiply(q_des_inv, q_curr_)
    return 2.0 * q_error[1:]   # [x, y, z]

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
    global log_count, Ct, mq, g0, max_krpm
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
    quat = torch.tensor([quat_w, quat_x, quat_y, quat_z])  # w y x z
    R_body_to_world = rotation_matrix(quat_w, quat_x, quat_y, quat_z)
    omega = torch.tensor([gyro_x, gyro_y, gyro_z])         # 角速度

    # 构建当前状态
    # current_state = torch.tensor([_pos[0], _pos[1], _pos[2], quat[0], quat[1], quat[2], quat[3], _vel[0], _vel[1], _vel[2], omega[0], omega[1], omega[2]])
    # 位置控制模式 目标位点
    goal_position = torch.tensor([0.0, 0.0, 0.5])

    Kp_pos = 5.0   # 位置环比例增益，后面可调
    v_des = Kp_pos * (goal_position - _pos[:3])     # TODO pos
    # v_des = torch.clamp(v_des, -2, 2)

    Kv_p = 2.5     # 速度环比例增益
    a_des = Kv_p * (v_des - _vel[:3]) + torch.tensor([0.0, 0.0, g0])        # TODO vel
    # g0 是重力加速度（正的，因为我们要抵消重力）

    # print(f"v_des: {v_des}, a_des: {a_des}")


    z_curr = R_body_to_world[:, 2]   # 第三列
    z_des = a_des / torch.norm(a_des)
    axis = torch.linalg.cross(z_curr, z_des)
    axis_norm = torch.norm(axis)
    dot = torch.dot(z_curr, z_des)
    # 避免数值问题
    if axis_norm < 1e-6:
        q_rot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    else:
        angle = torch.atan2(axis_norm, dot)  # 更稳定
        axis = axis / axis_norm
        half_angle = angle * 0.5
        q_rot = torch.tensor([torch.cos(half_angle), 
                            axis[0]*torch.sin(half_angle),
                            axis[1]*torch.sin(half_angle),
                            axis[2]*torch.sin(half_angle)])

    q_des = quat_multiply(q_rot, quat)
    e_R = attitude_error_quat(q_curr_=quat, q_des_=q_des)
    e_R[2] = 0.0 # free yaw

    # att ctrl
    # q_des = torch.tensor([1,0,0,0.5])
    Kp_att = 10.0
    omega_des = -Kp_att * e_R
    # print(f"omega_des: {omega_des}, q_curr_: {quat}, q_des_: {q_des}")

    Kd_rate = 0.001
    M_des = Kd_rate * (omega_des - omega)
    # print(f"M_des: {M_des}")

    # M_des = torch.Tensor([0.0000,0.0000,0.0000])

    # distrubute motors krpm
    a_norm = torch.norm(a_des)
    if a_norm < 1e-6:
        F_des = mq * g0
    else:
        F_des = mq * a_norm
    Mx, My, Mz = M_des[0], M_des[1], M_des[2]
    # 系数
    a = Ct          # 推力系数，注意单位与下面转速单位一致
    b = Ct * arm_length      # 滚转/俯仰力矩系数
    c = Cd          # 偏航力矩系数
    # 解算四个电机转速平方（krpm²）
    w1_sq = (F_des/a + Mx/b - My/b - Mz/c) / 4.0
    w2_sq = (F_des/a - Mx/b - My/b + Mz/c) / 4.0
    w3_sq = (F_des/a - Mx/b + My/b - Mz/c) / 4.0
    w4_sq = (F_des/a + Mx/b + My/b + Mz/c) / 4.0
    w1 = torch.sqrt(torch.clamp(w1_sq, min=0.0))
    w2 = torch.sqrt(torch.clamp(w2_sq, min=0.0))
    w3 = torch.sqrt(torch.clamp(w3_sq, min=0.0))
    w4 = torch.sqrt(torch.clamp(w4_sq, min=0.0))
    # hover_w = torch.sqrt(torch.tensor((mq*g0)/(4*Ct)))
    # print("hover_w", hover_w)
    # hovor speed: 15.777730167256925 krpm
    d.actuator('motor1').ctrl[0] = calc_motor_input(torch.clamp(w1, 0, max_krpm))
    d.actuator('motor2').ctrl[0] = calc_motor_input(torch.clamp(w2, 0, max_krpm))
    d.actuator('motor3').ctrl[0] = calc_motor_input(torch.clamp(w3, 0, max_krpm))
    d.actuator('motor4').ctrl[0] = calc_motor_input(torch.clamp(w4, 0, max_krpm))

    # 日志输出（每 50 次控制循环）
    log_count += 1
    if log_count >= 50:
        log_count = 0
        # print(f"Time: {d.time:.3f}, Pos: {d.qpos[0:3]}, Ctrl: {d.ctrl[:]}")
        pass
