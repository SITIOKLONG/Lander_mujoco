import torch





# 根据四元数计算旋转矩阵
def rotation_matrix(q0, q1, q2, q3) -> torch.Tensor:
    _row0 = torch.Tensor([1-2*(q2**2)-2*(q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    _row1 = torch.Tensor([2*(q1*q2+q0*q3), 1-2*(q1**2)-2*(q3**2), 2*(q2*q3-q0*q1)])
    _row2 = torch.Tensor([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2)-2*(q2**2)])
    return torch.Tensor((_row0, _row1, _row2))


def quat_multiply(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Hamilton product: q ⊗ p"""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    pw, px, py, pz = p[0], p[1], p[2], p[3]
    return torch.tensor([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse (conjugate for unit quaternion)"""
    return torch.tensor([q[0], -q[1], -q[2], -q[3]])


def attitude_error_quat(q_curr: torch.Tensor, q_des: torch.Tensor) -> torch.Tensor:
    """
    Returns attitude error vector (3,) = 2 * vector part of (q_des^{-1} ⊗ q_curr)
    """
    q_des_inv = quat_inverse(q_des)
    q_error = quat_multiply(q_des_inv, q_curr)
    return 2.0 * q_error[1:]   # [x, y, z]


# test pd controller
def attitude_pd_control(q_curr, q_des, omega_curr, Kp_att, Kd_att):
    att_err = attitude_error_quat(q_curr, q_des)
    omega_err = omega_curr   # 因为期望角速度为 0
    M = -Kp_att * att_err - Kd_att * omega_err
    return M
