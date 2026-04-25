import torch

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
    """
    Convert XYZ Euler angles (radians) to quaternion (w, x, y, z) – MuJoCo order.
    Supports scalar tensors or Python numbers.
    """
    # Ensure tensors (will be 0-dim if scalar)
    roll = torch.as_tensor(roll, dtype=torch.float32)
    pitch = torch.as_tensor(pitch, dtype=torch.float32)
    yaw = torch.as_tensor(yaw, dtype=torch.float32)

    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw

    cr = torch.cos(hr)
    sr = torch.sin(hr)
    cp = torch.cos(hp)
    sp = torch.sin(hp)
    cy = torch.cos(hy)
    sy = torch.sin(hy)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = torch.stack([qw, qx, qy, qz], dim=0)   # (4,)
    q_norm = torch.norm(q)
    # Avoid division by zero
    q = torch.where(q_norm < 1e-8, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), q / q_norm)
    return q


def quat_to_ang_vel_wxyz(q_prev, q_curr, dt):
    """
    Angular velocity (body frame?) from two consecutive quaternions (w,x,y,z) and time step.
    Returns tensor of shape (3,).
    """
    if dt <= 0.0:
        return torch.zeros(3, dtype=torch.float32)

    q_prev = torch.as_tensor(q_prev, dtype=torch.float32)
    q_curr = torch.as_tensor(q_curr, dtype=torch.float32)

    # Relative rotation: dq = q_prev^{-1} * q_curr
    dq = quat_multiply(quat_inverse(q_prev), q_curr)
    dq = dq / torch.norm(dq)   # normalize to avoid drift

    w = torch.clamp(dq[0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

    # If s or angle is negligible, axis is zero
    small_angle = (s < 1e-8) | (angle < 1e-8)
    axis = torch.where(small_angle, torch.zeros(3, dtype=torch.float32), dq[1:] / s)

    return axis * angle / dt
