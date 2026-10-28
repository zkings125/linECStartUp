"""从版本化 CSV 表格读取碰撞率的示例 provider。

CSV 最小列：`Te_eV,EoverN_Td,K_ion_m3_s,K_exc_m3_s,K_att_m3_s,`
`K_momentum_m3_s,nu_ei_s`。实际生产数据应注明截面来源和插值适用域。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from ..config import GasConfig
from .rates import CollisionRateModel, RateCoefficients


class TabulatedRateModel(CollisionRateModel):
    """二维 `(Te,E/N)` 率表 provider，超出凸包时使用最近邻。"""

    required_columns = (
        "Te_eV",
        "EoverN_Td",
        "K_ion_m3_s",
        "K_exc_m3_s",
        "K_att_m3_s",
        "K_momentum_m3_s",
        "nu_ei_s",
    )

    def __init__(self, gas: GasConfig, csv_path: str | Path, database_label: str = "tabulated") -> None:
        super().__init__(gas)
        self.csv_path = Path(csv_path)
        self.database_label = database_label
        rows = []
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in self.required_columns if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"率表缺少列: {missing}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    values = [float(row[name]) for name in self.required_columns]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"率表第 {line_number} 行包含非数值字段") from exc
                rows.append(values)
        if len(rows) < 3:
            raise ValueError("率表至少需要 3 行")
        data = np.asarray(rows, dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("率表只能包含有限数，不能有 NaN/Inf")
        points = data[:, :2]
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("率表的 (Te_eV,EoverN_Td) 采样点不能重复")
        centered = points - np.mean(points, axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            raise ValueError("率表的 (Te_eV,EoverN_Td) 采样点不能共线")
        self._points = data[:, :2]
        self._values = data[:, 2:]
        self._point_scale = np.ptp(self._points, axis=0)
        try:
            self._interpolators = [LinearNDInterpolator(self._points, self._values[:, i]) for i in range(5)]
        except QhullError as exc:
            raise ValueError("率表的二维采样点无法构造非退化三角剖分") from exc

    def _lookup(self, Te_eV: float, EoverN_Td: float) -> np.ndarray:
        point = np.asarray([[max(float(Te_eV), 1.0e-6), max(float(EoverN_Td), 1.0e-6)]])
        values = np.asarray([float(interp(point)[0]) for interp in self._interpolators])
        if not np.isfinite(values).all():
            # Te and E/N have different units and ranges; normalize each
            # coordinate before selecting an out-of-convex-hull fallback.
            normalized_delta = (self._points - point[0]) / self._point_scale
            distance = np.sum(normalized_delta**2, axis=1)
            values = self._values[int(np.argmin(distance)), :5]
        return np.maximum(values, 0.0)

    def evaluate(self, Te_eV: float, ne_m3: float, nn_m3: float, EoverN_Td: float) -> RateCoefficients:
        values = self._lookup(Te_eV, EoverN_Td)
        # 率表中 nu_ei 可来自真实 EEDF；若列为 0，退回 Coulomb 近似。
        nu_ei = values[4]
        if nu_ei <= 0.0:
            nu_ei = super().evaluate(Te_eV, ne_m3, nn_m3, EoverN_Td).nu_ei_s
        return RateCoefficients(values[0], values[1], values[2], self.gas.recombination_coeff_m3_s, values[3], nu_ei, self.database_label)
