"""自由边界平衡求解器的接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EquilibriumResult:
    R_m: np.ndarray
    Z_m: np.ndarray
    psi_Wb: np.ndarray
    converged: bool
    residual: float
    method: str


class EquilibriumSolver(Protocol):
    def solve(self, Ip_A: float, pressure_profile, coil_currents) -> EquilibriumResult: ...


class NotImplementedGradShafranov:
    """提醒使用者：CFS 前不能把常规 GS 解当作既成事实。"""

    def solve(self, Ip_A: float, pressure_profile=None, coil_currents=None) -> EquilibriumResult:
        raise NotImplementedError(
            "请接入自由边界 Grad--Shafranov/Biot--Savart 求解器；"
            "v0.1 的 CFS 结果来自 circular_surrogate。"
        )
