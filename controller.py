import torch
from utils_math import *

g0 = 9.0866          # m/s^2
mq = 0.033           # kg
Ct = 3.25e-4         # N/krpm^2
Cd = 7.9379e-6       # Nm/krpm^2
max_torque = 3.842e-03
max_krpm = 22.0
arm_length = 0.065 / 2.0
max_thrust = 0.1573

log_count = 0


def attitude_error_quat(q_curr_: torch.Tensor, q_des_: torch.Tensor) -> torch.Tensor:
    if torch.dot(q_curr_, q_des_) < 0:
        q_des_ = -q_des_
    q_des_inv = quat_inverse(q_des_)
    q_error = quat_multiply(q_des_inv, q_curr_)
    return 2.0 * q_error[1:]


def calc_motor_force(krpm):
    return Ct * krpm ** 2


def calc_motor_input(krpm):
    krpm = torch.clamp(krpm, 0, 22)
    force = calc_motor_force(krpm)
    inp = force / max_thrust
    inp = torch.clamp(inp, 0, 1)
    return inp


def control_callback(m, d, control_pos_error):
    global log_count

    # ---- 步骤 1：从仿真器读取数据，全部转为 torch.Tensor ----
    _pos   = torch.from_numpy(d.qpos).float()
    _vel   = torch.from_numpy(d.qvel).float()
    _sensor = torch.from_numpy(d.sensordata).float()

    gyro  = _sensor[0:3]          # [gyro_x, gyro_y, gyro_z]
    acc   = _sensor[3:6]          # [acc_x,  acc_y,  acc_z ]
    quat  = _sensor[6:10]         # [w, x, y, z]

    # 角速度 (body frame)
    omega = gyro

    # 旋转矩阵 (body -> world)，rotation_matrix 需要四个标量，这里取 .item()
    R_body_to_world = rotation_matrix(
        quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    ).float()

    # ---- 步骤 2：位置 + 速度 + 加速度期望 ----
    # goal_position = torch.tensor([0.0, 0.0, 2.0])     # debug test
    Kp_pos = torch.Tensor([2.0, 2.0, 4.0])  # 可分别调节 XY 和 Z
    v_des = Kp_pos * control_pos_error               # 期望速度

    Kv_p = torch.Tensor([2.5, 2.5, 4.0])     # 阻尼增益
    a_des = Kv_p * (v_des - _vel[:3]) + torch.tensor([0.0, 0.0, g0])

    # ---- 步骤 3：根据期望加速度计算期望姿态 ----
    z_curr = R_body_to_world[:, 2]          # 世界系下当前机体 Z 轴
    z_des  = a_des / torch.norm(a_des)      # 期望推力方向（单位向量）

    axis = torch.linalg.cross(z_curr, z_des)
    axis_norm = torch.norm(axis)
    dot = torch.dot(z_curr, z_des)

    if axis_norm < 1e-6:
        q_rot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    else:
        angle = torch.atan2(axis_norm, dot)
        axis = axis / axis_norm
        half_angle = angle * 0.5
        q_rot = torch.tensor([
            torch.cos(half_angle),
            axis[0] * torch.sin(half_angle),
            axis[1] * torch.sin(half_angle),
            axis[2] * torch.sin(half_angle)
        ])

    q_des = quat_multiply(q_rot, quat)

    # 姿态误差向量（自由偏航）
    e_R = attitude_error_quat(q_curr_=quat, q_des_=q_des)
    e_R[2] = 0.0

    # ---- 步骤 4：角速度 + 力矩控制 ----
    Kp_att = 10.0
    omega_des = -Kp_att * e_R

    Kd_rate = 0.001
    M_des = Kd_rate * (omega_des - omega)

    # ---- 步骤 5：推力 + 力矩 -> 四个电机转速 ----
    a_norm = torch.norm(a_des)
    F_des = mq * a_norm if a_norm > 1e-6 else mq * g0

    Mx, My, Mz = M_des[0], M_des[1], M_des[2]
    a = Ct
    b = Ct * arm_length
    c = Cd

    w1_sq = (F_des / a + Mx / b - My / b - Mz / c) / 4.0
    w2_sq = (F_des / a - Mx / b - My / b + Mz / c) / 4.0
    w3_sq = (F_des / a - Mx / b + My / b - Mz / c) / 4.0
    w4_sq = (F_des / a + Mx / b + My / b + Mz / c) / 4.0

    w1 = torch.sqrt(torch.clamp(w1_sq, min=0.0))
    w2 = torch.sqrt(torch.clamp(w2_sq, min=0.0))
    w3 = torch.sqrt(torch.clamp(w3_sq, min=0.0))
    w4 = torch.sqrt(torch.clamp(w4_sq, min=0.0))

    # ---- 步骤 6：下发控制指令（必须转回 Python float） ----
    d.actuator('motor1').ctrl[0] = calc_motor_input(w1).item()
    d.actuator('motor2').ctrl[0] = calc_motor_input(w2).item()
    d.actuator('motor3').ctrl[0] = calc_motor_input(w3).item()
    d.actuator('motor4').ctrl[0] = calc_motor_input(w4).item()

    # 日志
    log_count += 1
    if log_count >= 50:
        log_count = 0
