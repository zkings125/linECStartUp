"""层间契约。

物理层只依赖这些小接口，不依赖 CLI、文件格式或绘图代码。后续可以在不改
工作流的情况下注入真实截面数据库、射线追踪器、1D-2V 求解器或 GS 求解器。
"""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np

from .state import GlobalState


class RateProvider(Protocol):
    def evaluate(self, Te_eV: float, ne_m3: float, nn_m3: float, EoverN_Td: float): ...

    def electron_thermal_speed_m_s(self, Te_eV: float) -> float: ...

    def ion_sound_speed_m_s(self, Te_eV: float, Ti_eV: float) -> float: ...


class RFProvider(Protocol):
    def evaluate(self, t_s: float, state: GlobalState): ...


class TopologyProvider(Protocol):
    def detect(self, Ip_A: float, t_s: float = 0.0): ...


class MagneticFieldProvider(Protocol):
    """磁场/位形层契约；可由 Biot--Savart 或 GS 后端实现。

    ``CompositeMagneticField`` also implements this contract, so a caller can
    superpose independently driven coil providers without changing the ODE.
    """

    configuration: str

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0): ...

    def flux_function(self, R_m, Z_m, Ip_A: float, t_s: float = 0.0): ...

    def vertical_field(self, t_s: float = 0.0) -> float: ...

    def configuration_state(self, Ip_A: float, t_s: float = 0.0): ...

    def diagnostics(self, Ip_A: float, t_s: float = 0.0): ...


class StartupRHS(Protocol):
    def __call__(self, t_s: float, vector: np.ndarray) -> np.ndarray: ...


class SpatialSolver(Protocol):
    """未来 1D/2D 求解器的最小契约。"""

    def advance(self, t_s: float, state: Mapping[str, np.ndarray], dt_s: float) -> Mapping[str, np.ndarray]: ...
