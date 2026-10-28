"""模型状态与状态向量转换。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .constants import E_CHARGE_C


class StartupStage(str, Enum):
    VACUUM = "vacuum"
    BREAKDOWN = "breakdown"
    PRE_IONIZATION = "preionization"
    CFS_FORMING = "cfs_forming"
    CFS_ESTABLISHED = "cfs_established"
    FAILED = "failed"


@dataclass
class GlobalState:
    """0D 状态。

    electron_energy_J_m3 和 hot_energy_J_m3 是能量密度；Te 由前者和 ne
    反算，因此不会同时积分两个相互矛盾的温度变量。
    """

    ne_m3: float
    ni_m3: float
    nn_m3: float
    electron_energy_J_m3: float
    Ip_A: float
    n_hot_m3: float = 0.0
    hot_energy_J_m3: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.ne_m3,
                self.ni_m3,
                self.nn_m3,
                self.electron_energy_J_m3,
                self.Ip_A,
                self.n_hot_m3,
                self.hot_energy_J_m3,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vector: Iterable[float]) -> "GlobalState":
        values = list(vector)
        if len(values) != 7:
            raise ValueError("GlobalState 向量必须有 7 个分量")
        return cls(*map(float, values))

    @property
    def Te_eV(self) -> float:
        if self.ne_m3 <= 0.0 or self.electron_energy_J_m3 <= 0.0:
            return 1.0e-3
        return max(1.0e-3, self.electron_energy_J_m3 / (1.5 * self.ne_m3 * E_CHARGE_C))

    @property
    def hot_mean_energy_eV(self) -> float:
        if self.n_hot_m3 <= 0.0 or self.hot_energy_J_m3 <= 0.0:
            return 0.0
        return self.hot_energy_J_m3 / (self.n_hot_m3 * E_CHARGE_C)

    def projected(self, floors: tuple[float, ...] | None = None) -> "GlobalState":
        """把数值积分的微小负值投影回物理可行域。"""

        f = floors or (0.0, 0.0, 0.0, 0.0, -float("inf"), 0.0, 0.0)
        v = self.to_vector()
        for i, floor in enumerate(f):
            if np.isfinite(floor):
                v[i] = max(v[i], floor)
        return self.from_vector(v)


@dataclass(frozen=True)
class Event:
    type: str
    time_s: float
    stage_from: str
    stage_to: str
    metrics: dict
