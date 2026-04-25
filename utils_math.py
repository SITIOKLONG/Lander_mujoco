import torch
import math
import numpy as np

# 根据四元数计算旋转矩阵
def rotation_matrix(q0, q1, q2, q3):
    q0 = torch.as_tensor(q0)
    q1 = torch.as_tensor(q1)
    q2 = torch.as_tensor(q2)
    q3 = torch.as_tensor(q3)

    _row0 = torch.stack([1-2*q2**2-2*q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    _row1 = torch.stack([2*(q1*q2+q0*q3), 1-2*q1**2-2*q3**2, 2*(q2*q3-q0*q1)])
    _row2 = torch.stack([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*q1**2-2*q2**2])

    return torch.stack([_row0, _row1, _row2])


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

def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """Convert XYZ Euler angles to MuJoCo quaternion order: wxyz."""
    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw

    cr = math.cos(hr)
    sr = math.sin(hr)
    cp = math.cos(hp)
    sp = math.sin(hp)
    cy = math.cos(hy)
    sy = math.sin(hy)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / q_norm


def quat_to_ang_vel_wxyz(q_prev, q_curr, dt):
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float64)
    dq = quat_multiply(quat_inverse(q_prev), q_curr)
    dq = dq / np.linalg.norm(dq)
    w = np.clip(dq[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = dq[1:] / s
    return axis * angle / dt
