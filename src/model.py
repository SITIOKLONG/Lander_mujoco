from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class QuadParams:
    """physical params for quadrotor"""

    # drone params
    grav = 9.8066
    mass = 33e-3
    I: np.ndarray = field(default=lambda: np.diag(
        [1.395e-5, 1.395e-5, 2.173e-5]))
    maxangle: float = np.deg2rad(40.0)
    kforce: float = 3.25e-4
    kmoment: float = 1.587e-5
    armlength: float = 65e-3/2

    invI: np.ndarray = field(init=False)
    maxF: float = field(init=False)
    minF: float = field(init=False)
    FM_omega2: np.ndarray = field(init=False)
    omega2_FM: np.ndarray = field(init=False)
    maxomega: float = field(init=False)
    minomega: float = field(init=False)

    def __post_init__(self) -> None:
        self.invI = np.linalg.inv(self.I)
        self.maxF = 2.5 * self.mass * self.grav
        self.minF = 0.05 * self.mass * self.grav

        self.FM_omega2 = np.array([
            [self.kforce, self.kforce, self.kforce, self.kforce],
            [0.0, self.armlength * self.kforce, 0.0, -self.armlength * self.kforce],
            [-self.armlength * self.kforce, 0.0,
                self.armlength * self.kforce, 0.0],
            [self.kmoment, -self.kmoment, self.kmoment, -self.kmoment],
        ])
        self.omega2_FM = np.linalg.inv(self.FM_omega2)

        self.maxomega = np.sqrt(self.maxF / (4.0 * self.kforce))
        self.minomega = np.sqrt(self.minF / (4.0 * self.kforce))
