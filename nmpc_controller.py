# 四旋翼NMPC控制器 20250304 Wakkk
# ACADOS NMPC
import os
import sys
from pathlib import Path


def _inject_acados_template_path():
    """Try to locate local acados_template package and prepend it to sys.path."""
    candidates = []

    direct = os.getenv("ACADOS_TEMPLATE_PATH")
    if direct:
        candidates.append(Path(direct).expanduser())

    for key in ("ACADOS_SOURCE_DIR", "ACADOS_ROOT"):
        root = os.getenv(key)
        if root:
            candidates.append(Path(root).expanduser() / "interfaces" / "acados_template")

    # Common local defaults.
    candidates.append(Path.home() / "acados" / "interfaces" / "acados_template")
    candidates.append(Path("/Users/jacksit/acados/interfaces/acados_template"))

    for candidate in candidates:
        pkg_init = candidate / "acados_template" / "__init__.py"
        if pkg_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    return None


try:
    from acados_template import AcadosOcp, AcadosOcpSolver
except ModuleNotFoundError as exc:
    injected = _inject_acados_template_path()
    if injected is None:
        raise ModuleNotFoundError(
            "Could not import acados_template. Set ACADOS_SOURCE_DIR/ACADOS_ROOT, "
            "or ACADOS_TEMPLATE_PATH to <acados>/interfaces/acados_template."
        ) from exc
    from acados_template import AcadosOcp, AcadosOcpSolver

from export_model import *
import numpy as np
import scipy.linalg
from os.path import dirname, join, abspath
import time

# np.set_printoptions(precision=3)  # 设置精度
np.set_printoptions(suppress=True)  # 禁用科学计数法输出

# ACADOS NMPC控制器
class NMPC_Controller:
    def __init__(self):
        self.ocp = AcadosOcp()       # OCP 优化问题
        self.model = export_model()  # 导出四旋翼物理模型

        self.Tf = 0.75                      # 预测时间长度(s)
        self.N = 50                         # 预测步数(节点数量)
        self.nx = self.model.x.size()[0]    # 状态维度 13维度
        self.nu = self.model.u.size()[0]    # 控制输入维度 4维度
        self.ny = self.nx + self.nu         # 评估维度
        self.ny_e = self.nx                 # 终端评估维度

        # set ocp_nlp_dimensions
        self.nlp_dims     = self.ocp.dims
        self.nlp_dims.N   = self.N

        # parameters
        self.g0  = 9.8066    # [m.s^2] accerelation of gravity
        self.mq  = 33e-3     # [kg] total mass (with one marker)
        self.Ct  = 3.25e-4   # [N/krpm^2] Thrust coef

        # bounds
        self.hov_w = np.sqrt((self.mq*self.g0)/(4*self.Ct))  # 悬停时单个电机推力
        print(f"hovor speed: {self.hov_w} krpm")
        # hovor speed: 15.777730167256925 krpm

        self.max_speed = 22.0  # 电机最高转速(krpm)
        self.safe_hover_control = np.full((4,), self.hov_w, dtype=np.float64)
        self.last_control = self.safe_hover_control.copy()
        self.last_valid_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.fail_count = 0

        # set weighting matrices 状态权重矩阵
        Q = np.eye(self.nx)
        Q[0,0] = 100.0      # x
        Q[1,1] = 100.0      # y
        Q[2,2] = 200.0      # z
        Q[3,3] = 0.0        # qw
        Q[4,4] = 0.0        # qx
        Q[5,5] = 0.0        # qy
        Q[6,6] = 0.0        # qz
        Q[7,7] = 1.0        # vbx
        Q[8,8] = 1.0        # vby
        Q[9,9] = 4.0        # vbz
        Q[10,10] = 1e-5     # wx
        Q[11,11] = 1e-5     # wy
        Q[12,12] = 10.0     # wz

        R = np.eye(self.nu)   # 控制输入权重矩阵
        R[0,0] = 0.06    # w1
        R[1,1] = 0.06    # w2
        R[2,2] = 0.06    # w3
        R[3,3] = 0.06    # w4

        # ensure W is positive definite by adding a tiny regularization on the diagonal
        eps_W = 1e-9
        self.ocp.cost.W = scipy.linalg.block_diag(Q, R) + eps_W * np.eye(self.ny)

        Vx = np.zeros((self.ny, self.nx))
        Vx[0,0] = 1.0
        Vx[1,1] = 1.0
        Vx[2,2] = 1.0
        Vx[3,3] = 1.0
        Vx[4,4] = 1.0
        Vx[5,5] = 1.0
        Vx[6,6] = 1.0
        Vx[7,7] = 1.0
        Vx[8,8] = 1.0
        Vx[9,9] = 1.0
        Vx[10,10] = 1.0
        Vx[11,11] = 1.0
        Vx[12,12] = 1.0
        self.ocp.cost.Vx = Vx

        Vu = np.zeros((self.ny, self.nu))
        Vu[13,0] = 1.0
        Vu[14,1] = 1.0
        Vu[15,2] = 1.0
        Vu[16,3] = 1.0
        self.ocp.cost.Vu = Vu

        # terminal weight: add small regularization so W_e is positive definite
        eps_We = 1e-9
        self.ocp.cost.W_e = 50.0 * Q + eps_We * np.eye(self.ny_e)

        Vx_e = np.zeros((self.ny_e, self.nx))
        Vx_e[0,0] = 1.0
        Vx_e[1,1] = 1.0
        Vx_e[2,2] = 1.0
        Vx_e[3,3] = 1.0
        Vx_e[4,4] = 1.0
        Vx_e[5,5] = 1.0
        Vx_e[6,6] = 1.0
        Vx_e[7,7] = 1.0
        Vx_e[8,8] = 1.0
        Vx_e[9,9] = 1.0
        Vx_e[10,10] = 1.0
        Vx_e[11,11] = 1.0
        Vx_e[12,12] = 1.0
        self.ocp.cost.Vx_e = Vx_e

        # 过程参考向量(状态+输入)
        self.ocp.cost.yref   = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.hov_w, self.hov_w, self.hov_w, self.hov_w])
        # 终端参考向量(状态)
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 构建约束
        self.ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0])  # 电机最低转速输入
        self.ocp.constraints.ubu = np.array([+self.max_speed,+self.max_speed,+self.max_speed,+self.max_speed])  # 电机最高转速输入
        self.ocp.constraints.x0  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])  # 所有电机转速参与评估

        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 显然更快 ~100Hz
        # self.ocp.solver_options.nlp_solver_type = 'SQP'  # ~10Hz

        self.ocp.model = self.model  # 传入模型
        # 构建编译OCP求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = 'acados_ocp.json')
        print("NMPC Controller Init Done")

    def _sanitize_state(self, state, fallback_state):
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        if x.size != self.nx:
            raise ValueError(f"Expected state dim {self.nx}, got {x.size}")

        fallback = np.asarray(fallback_state, dtype=np.float64).reshape(-1)
        if fallback.size != self.nx:
            fallback = self.last_valid_state

        out = x.copy()
        invalid = ~np.isfinite(out)
        if np.any(invalid):
            out[invalid] = fallback[invalid]

        q = out[3:7]
        q_norm = float(np.linalg.norm(q))
        if (not np.isfinite(q_norm)) or q_norm < 1e-8:
            out[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            out[3:7] = q / q_norm
        return out

    def _recover_solver_warmstart(self, safe_state):
        """Reset solver trajectory with safe values after an infeasible/NaN step."""
        try:
            for i in range(self.N):
                self.acados_solver.set(i, "x", safe_state)
                self.acados_solver.set(i, "u", self.safe_hover_control)
            self.acados_solver.set(self.N, "x", safe_state)
        except Exception:
            pass

    # 状态空间位点控制
    # current_state当前状态: [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz] 
    # goal_state目标状态:    [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz] 
    def nmpc_state_control(self, current_state, goal_state):
        _start = time.perf_counter()
        safe_current = self._sanitize_state(current_state, self.last_valid_state)
        safe_goal = self._sanitize_state(goal_state, safe_current)
        self.last_valid_state = safe_current.copy()

        # Set initial condition, equality constraint
        self.acados_solver.set(0, 'lbx', safe_current)
        self.acados_solver.set(0, 'ubx', safe_current)

        y_ref = np.concatenate((safe_goal, np.array([self.hov_w, self.hov_w, self.hov_w, self.hov_w])))
        # Set Goal State
        for i in range(self.N):
            self.acados_solver.set(i, 'yref', y_ref)   # 过程参考
        y_refN = safe_goal 
        self.acados_solver.set(self.N, 'yref', y_refN)   # 终端参考

        # Solve Problem
        status = int(self.acados_solver.solve())

        # Use only first control action and guard against NaN/infeasible solver outputs.
        try:
            u0 = np.asarray(self.acados_solver.get(0, "u"), dtype=np.float64).reshape(-1)
        except Exception:
            u0 = np.full((self.nu,), np.nan, dtype=np.float64)

        valid_u0 = (status == 0) and (u0.size == self.nu) and bool(np.all(np.isfinite(u0)))
        if valid_u0:
            u0 = np.clip(u0, 0.0, self.max_speed)
            self.last_control = u0.copy()
            self.fail_count = 0
        else:
            self.fail_count += 1
            if self.fail_count <= 5 or self.fail_count % 20 == 0:
                print(
                    f"[NMPC] WARNING: invalid solver output (status={status}). "
                    f"Using fallback hover control. fail_count={self.fail_count}"
                )
            u0 = self.last_control.copy()
            if not np.all(np.isfinite(u0)):
                u0 = self.safe_hover_control.copy()
                self.last_control = u0.copy()
            self._recover_solver_warmstart(safe_current)

        _end = time.perf_counter()
        _dt = _end - _start
        return _dt, u0  # 返回最近控制输入 4 Vector
        # control_input = self.acados_solver.get(0, "u")
        # state_estimate = self.acados_solver.get(self.N, "x")
        # return control_input, state_estimate  # 返回所有控制输入和状态

    # NMPC位置控制
    # goal_pos: 目标三维位置[x y z]
    def nmpc_position_control(self, current_state, goal_pos):
        goal_state = np.array([goal_pos[0], goal_pos[1], goal_pos[2], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _dt, control = self.nmpc_state_control(current_state, goal_state)
        return _dt, control

# TEST
if __name__ == '__main__':
    print("20250304 ACADOS NMPC TEST")
    nmpc_controller = NMPC_Controller()
    current_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_state = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _dt, w = nmpc_controller.nmpc_state_control(current_state, goal_state)

    # print("Control Input:")
    # print(w)
    # print("State Estimation:")
    # print(x)
