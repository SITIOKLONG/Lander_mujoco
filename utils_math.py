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

def rotation_matrix_to_quat_wxyz(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z) – MuJoCo order.
    Input R: torch.Tensor of shape (3,3), or any array convertable to float32 tensor.
    Returns a 1‑D tensor of length 4: (qw, qx, qy, qz), normalized.
    """
    # Ensure it's a float32 tensor
    R = torch.as_tensor(R, dtype=torch.float32)

    # Validate shape (optional, but good for debugging)
    if R.shape != (3, 3):
        raise ValueError(f"Expected a (3,3) tensor, got {R.shape}")

    # Extract elements for readability
    r00, r01, r02 = R[0,0], R[0,1], R[0,2]
    r10, r11, r12 = R[1,0], R[1,1], R[1,2]
    r20, r21, r22 = R[2,0], R[2,1], R[2,2]

    # Trace of the matrix
    trace = r00 + r11 + r22

    # Helper to compute quaternion for a given largest diagonal element case
    def case0():
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r21 - r12) * s
        y = (r02 - r20) * s
        z = (r10 - r01) * s
        return torch.stack([w, x, y, z])

    def case1():
        s = 0.5 / torch.sqrt(1.0 + r00 - r11 - r22)
        w = (r21 - r12) * s
        x = 0.25 / s
        y = (r01 + r10) * s
        z = (r02 + r20) * s
        return torch.stack([w, x, y, z])

    def case2():
        s = 0.5 / torch.sqrt(1.0 + r11 - r00 - r22)
        w = (r02 - r20) * s
        x = (r01 + r10) * s
        y = 0.25 / s
        z = (r12 + r21) * s
        return torch.stack([w, x, y, z])

    def case3():
        s = 0.5 / torch.sqrt(1.0 + r22 - r00 - r11)
        w = (r10 - r01) * s
        x = (r02 + r20) * s
        y = (r12 + r21) * s
        z = 0.25 / s
        return torch.stack([w, x, y, z])

    # Choose the appropriate case based on trace and diagonal dominance
    if trace > 0:
        q = case0()
    elif r00 > r11 and r00 > r22:
        q = case1()
    elif r11 > r22:
        q = case2()
    else:
        q = case3()

    # Normalize and guard against near‑zero quaternion (return identity)
    q_norm = torch.norm(q)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=q.device)
    q = torch.where(q_norm < 1e-8, identity, q / q_norm)

    return q

def quat_to_ang_vel_wxyz(q_prev, q_curr, dt):
    """
    Angular velocity (world frame?) from two consecutive quaternions (w,x,y,z) and time step.
    Returns tensor of shape (3,).
    """
    if dt <= 0.0:
        return torch.zeros(3, dtype=torch.float32)

    q_prev = torch.as_tensor(q_prev, dtype=torch.float32)
    q_curr = torch.as_tensor(q_curr, dtype=torch.float32)
    if torch.dot(q_curr, q_prev) < 0.0:
        q_curr = -q_curr

    # Relative rotation: dq = q_prev^{-1} * q_curr
    # dq = quat_multiply(quat_inverse(q_prev), q_curr)
    dq = quat_multiply(q_curr, quat_inverse(q_prev))
    dq = dq / torch.norm(dq)   # normalize to avoid drift

    w = torch.clamp(dq[0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

    # If s or angle is negligible, axis is zero
    small_angle = (s < 1e-8) | (angle < 1e-8)
    axis = torch.where(small_angle, torch.zeros(3, dtype=torch.float32), dq[1:] / s)

    return axis * angle / dt
