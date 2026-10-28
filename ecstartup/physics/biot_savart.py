"""轴对称圆形线圈的 Biot--Savart 磁场 provider。

本模块提供一个不依赖线圈网格的真实解析后端，作为从局部解析 surrogate
过渡到线圈/场线模型的最小步骤。每个 ``CircularLoop`` 是一根位于
``(R=radius_m, Z=z0_m)`` 的圆形细丝线圈；有限长度、多匝线圈通过多个
不同 ``z0_m`` 的 loop 离散表示。场由完整椭圆积分公式计算，而不是把
线圈场近似为常量。导线本身是理想细丝，因此在导线位置场发散，provider
会显式抛出 ``ValueError``。

该 provider 仍不求解自由边界 Grad--Shafranov 平衡，也不做场线追踪或
壁命中率计算。它实现 ``MagneticFieldProvider`` 契约，可通过
``GlobalStartupModel(..., magnetic_field=provider)`` 注入而不修改 ODE。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import ellipe, ellipk

from ..config import GeometryConfig
from ..constants import MU0_H_M, PI
from ..waveform import linear_value, validate_points
from .magnetic import (
    CircularMagneticField,
    FieldPoint,
    MagneticConfigurationState,
    MagneticFieldProvider,
)


_AXIS_EPS = 1.0e-10
_WIRE_EPS = 1.0e-12


@dataclass(frozen=True)
class CircularLoop:
    """理想圆形细丝线圈。

    ``current_A`` 是每匝电流；``turns`` 允许用重合匝表示匝数，有限厚度
    线圈应改用多个 ``CircularLoop``（给每匝实际的半径和 ``z0_m``）。
    ``current_waveform_A`` 为 ``(time_s, current_A)`` 点列，采用分段线性
    插值，区间外保持端点值。
    """

    radius_m: float
    z0_m: float = 0.0
    current_A: float = 0.0
    current_waveform_A: Sequence[Tuple[float, float]] | None = None
    turns: int = 1
    name: str = "loop"

    def __post_init__(self) -> None:
        radius = float(self.radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("CircularLoop.radius_m 必须为有限正数")
        z0 = float(self.z0_m)
        if not math.isfinite(z0):
            raise ValueError("CircularLoop.z0_m 必须为有限数")
        current = float(self.current_A)
        if not math.isfinite(current):
            raise ValueError("CircularLoop.current_A 必须为有限数")
        if int(self.turns) != self.turns or int(self.turns) < 1:
            raise ValueError("CircularLoop.turns 必须为正整数")
        name = str(self.name).strip()
        if not name:
            raise ValueError("CircularLoop.name 不能为空")
        waveform = None
        if self.current_waveform_A is not None:
            waveform = tuple(validate_points(self.current_waveform_A))
            if not waveform:
                waveform = None
            elif any(not (math.isfinite(time) and math.isfinite(value)) for time, value in waveform):
                raise ValueError("CircularLoop.current_waveform_A 必须只包含有限数")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "z0_m", z0)
        object.__setattr__(self, "current_A", current)
        object.__setattr__(self, "turns", int(self.turns))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "current_waveform_A", waveform)

    def current_at(self, t_s: float = 0.0) -> float:
        """Return the total ampere-turn current at ``t_s``."""

        return float(self.turns) * linear_value(self.current_waveform_A, t_s, self.current_A)


def _validate_observation_point(R_m: float, Z_m: float) -> tuple[float, float]:
    R = float(R_m)
    Z = float(Z_m)
    if not math.isfinite(R) or R < 0.0:
        raise ValueError("柱坐标 R_m 必须为有限非负数")
    if not math.isfinite(Z):
        raise ValueError("柱坐标 Z_m 必须为有限数")
    return R, Z


def circular_loop_field(
    R_m: float,
    Z_m: float,
    radius_m: float,
    z0_m: float = 0.0,
    current_A: float = 0.0,
) -> FieldPoint:
    """Evaluate one circular filament using complete elliptic integrals.

    The elliptic-integral parameter is ``m=k**2`` as expected by
    ``scipy.special.ellipk``/``ellipe``.  On the symmetry axis the analytic
    limit is used to avoid the removable ``0/0`` in the off-axis formula.
    """

    R, Z = _validate_observation_point(R_m, Z_m)
    a = float(radius_m)
    z0 = float(z0_m)
    I = float(current_A)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("radius_m 必须为有限正数")
    if not math.isfinite(z0) or not math.isfinite(I):
        raise ValueError("z0_m/current_A 必须为有限数")
    dz = Z - z0

    if R <= _AXIS_EPS * max(a, 1.0):
        denominator = (a * a + dz * dz) ** 1.5
        return FieldPoint(0.0, MU0_H_M * I * a * a / (2.0 * denominator), 0.0)

    alpha2 = (a - R) ** 2 + dz * dz
    # A filament has a logarithmically divergent field at alpha2=0.  Raising
    # here is safer than silently clipping a potentially enormous value.
    wire_distance = math.sqrt(alpha2)
    if wire_distance <= _WIRE_EPS * max(a, 1.0):
        raise ValueError("不能在理想细丝线圈本身的位置评估磁场（场发散）")
    beta2 = (a + R) ** 2 + dz * dz
    beta = math.sqrt(beta2)
    m = 4.0 * a * R / beta2
    # Round-off can put m infinitesimally outside [0,1].  scipy's functions
    # are evaluated below one; the explicit clip also keeps the near-wire
    # behaviour finite and deterministic.
    m = min(1.0 - np.finfo(float).eps, max(0.0, m))
    K = float(ellipk(m))
    E = float(ellipe(m))
    prefactor = MU0_H_M * I / (2.0 * PI * alpha2 * beta)
    B_R = prefactor * dz * ((a * a + R * R + dz * dz) * E - alpha2 * K) / R
    B_Z = prefactor * ((a * a - R * R - dz * dz) * E + alpha2 * K)
    return FieldPoint(B_R, B_Z, 0.0)


def circular_loop_flux(
    R_m,
    Z_m,
    radius_m: float,
    z0_m: float = 0.0,
    current_A: float = 0.0,
):
    """Return the poloidal flux ``psi=R*A_phi`` of one circular loop.

    ``R_m`` and ``Z_m`` may be scalars or NumPy arrays.  The axis limit is
    exactly zero (the gauge used by the analytic expression).
    """

    R = np.asarray(R_m, dtype=float)
    Z = np.asarray(Z_m, dtype=float)
    a = float(radius_m)
    z0 = float(z0_m)
    I = float(current_A)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("radius_m 必须为有限正数")
    if not math.isfinite(z0) or not math.isfinite(I):
        raise ValueError("z0_m/current_A 必须为有限数")
    if np.any(~np.isfinite(R)) or np.any(R < 0.0) or np.any(~np.isfinite(Z)):
        raise ValueError("R_m/Z_m 必须为有限数，且 R_m 非负")
    dz = Z - z0
    alpha2 = (a - R) ** 2 + dz * dz
    if np.any(np.sqrt(alpha2) <= _WIRE_EPS * max(a, 1.0)):
        raise ValueError("不能在理想细丝线圈本身的位置评估磁通（场发散）")
    beta2 = (a + R) ** 2 + dz * dz
    m = np.divide(4.0 * a * R, beta2, out=np.zeros_like(beta2), where=beta2 > 0.0)
    m = np.clip(m, 0.0, 1.0 - np.finfo(float).eps)
    k = np.sqrt(m)
    K = ellipk(m)
    E = ellipe(m)
    # The expression has a removable 0/0 at R=0; assigning zero is the
    # selected regular gauge and avoids floating-point noise on axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = MU0_H_M * I * np.sqrt(a * R) / (PI * k) * ((1.0 - 0.5 * m) * K - E)
    psi = np.where(R <= _AXIS_EPS * max(a, 1.0), 0.0, psi)
    if np.ndim(psi) == 0:
        return float(psi)
    return psi


class BiotSavartMagneticField:
    """基础 provider + 多个圆形线圈的轴对称 Biot--Savart 叠加。

    ``base_provider`` 默认为现有的 ``CircularMagneticField``，因此仍提供
    等效等离子体自场、环向场和基础垂直场；传入自定义 base 可将本类用作
    纯外部线圈包装器。``loops`` 的每个电流可以有自己的时间波形。
    """

    def __init__(
        self,
        geometry: GeometryConfig,
        loops: Iterable[CircularLoop],
        base_provider: MagneticFieldProvider | None = None,
    ) -> None:
        self.geometry = geometry
        self.base = base_provider if base_provider is not None else CircularMagneticField(geometry)
        def normalize_loop(item: Any) -> CircularLoop:
            if isinstance(item, CircularLoop):
                return item
            if hasattr(item, "model_dump"):
                item = item.model_dump(mode="python")
            if isinstance(item, Mapping):
                return CircularLoop(**item)
            raise TypeError("loops 的每个元素必须是 CircularLoop、映射或 Pydantic 配置对象")

        normalized = tuple(normalize_loop(item) for item in loops)
        if not normalized:
            raise ValueError("BiotSavartMagneticField 至少需要一个 CircularLoop")
        names = [item.name for item in normalized]
        if len(names) != len(set(names)):
            raise ValueError("线圈 name 必须唯一")
        self.loops = normalized
        self.configuration = f"biot_savart[{getattr(self.base, 'configuration', 'custom')}]"

    def _loop_point(self, loop: CircularLoop, R_m: float, Z_m: float, t_s: float) -> FieldPoint:
        return circular_loop_field(
            R_m,
            Z_m,
            loop.radius_m,
            loop.z0_m,
            loop.current_at(t_s),
        )

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        point = self.base.evaluate(R_m, Z_m, Ip_A, t_s)
        B_R = point.B_R_T
        B_Z = point.B_Z_T
        B_phi = point.B_phi_T
        for loop in self.loops:
            addition = self._loop_point(loop, R_m, Z_m, t_s)
            B_R += addition.B_R_T
            B_Z += addition.B_Z_T
        return FieldPoint(B_R, B_Z, B_phi)

    def flux_function(self, R_m, Z_m, Ip_A: float, t_s: float = 0.0):
        psi = self.base.flux_function(R_m, Z_m, Ip_A, t_s)
        for loop in self.loops:
            psi = psi + circular_loop_flux(
                R_m,
                Z_m,
                loop.radius_m,
                loop.z0_m,
                loop.current_at(t_s),
            )
        return psi

    def vertical_field(self, t_s: float = 0.0) -> float:
        axis_R = self.geometry.R0_m
        return float(
            self.base.vertical_field(t_s)
            + sum(self._loop_point(loop, axis_R, 0.0, t_s).B_Z_T for loop in self.loops)
        )

    def _loop_vertical_gradient(self, t_s: float = 0.0) -> float:
        R0 = self.geometry.R0_m
        h = max(1.0e-6 * R0, 1.0e-8)
        plus = sum(self._loop_point(loop, R0 + h, 0.0, t_s).B_Z_T for loop in self.loops)
        minus = sum(self._loop_point(loop, max(R0 - h, 1.0e-9), 0.0, t_s).B_Z_T for loop in self.loops)
        return (plus - minus) / (2.0 * h)

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        base_state = self.base.configuration_state(Ip_A, t_s)
        if not isinstance(base_state, MagneticConfigurationState):
            return base_state
        b_vertical = self.vertical_field(t_s)
        return replace(
            base_state,
            configuration=self.configuration,
            vertical_alignment=0.0 if b_vertical == 0.0 else math.copysign(1.0, b_vertical),
            vertical_field_T=b_vertical,
            vertical_gradient_T_per_m=base_state.vertical_gradient_T_per_m + self._loop_vertical_gradient(t_s),
        )

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        out = dict(self.base.diagnostics(Ip_A, t_s))
        R0 = self.geometry.R0_m
        edge_R = R0 + self.geometry.minor_radius_m
        axis_points = [self._loop_point(loop, R0, 0.0, t_s) for loop in self.loops]
        edge_points: list[FieldPoint] = []
        edge_singularities = 0
        for loop in self.loops:
            try:
                edge_points.append(self._loop_point(loop, edge_R, 0.0, t_s))
            except ValueError as exc:
                # The edge point is a representative diagnostic location, not
                # a required integration point.  A filament exactly on that
                # location has a genuine singular field; keep the simulation
                # usable and expose the condition explicitly instead of
                # silently clipping it.
                if "细丝线圈" not in str(exc):
                    raise
                edge_singularities += 1
        base_topology = float(out.get("B_vertical_topology_T", out.get("B_vertical_provider_T", 0.0)))
        out["field_configuration"] = self.configuration
        out["field_provider"] = "biot_savart_circular_loops"
        out["base_field_configuration"] = str(getattr(self.base, "configuration", "custom"))
        out["loop_count"] = float(len(self.loops))
        out["loop_current_total_A"] = float(sum(loop.current_at(t_s) for loop in self.loops))
        out["loop_B_R_axis_T"] = float(sum(point.B_R_T for point in axis_points))
        out["loop_B_Z_axis_T"] = float(sum(point.B_Z_T for point in axis_points))
        out["loop_B_phi_axis_T"] = 0.0
        out["loop_Bp_edge_T"] = (
            float("nan")
            if edge_singularities
            else float(
                math.hypot(
                    sum(point.B_R_T for point in edge_points),
                    sum(point.B_Z_T for point in edge_points),
                )
            )
        )
        out["loop_edge_singularity_count"] = float(edge_singularities)
        # Keep the surrogate topology's Bp definition tied to the base plasma
        # current.  The external coil poloidal field is exposed separately;
        # a GS/field-line backend must decide how it enters LCFS detection.
        out["Bp_edge_base_T"] = float(out.get("Bp_edge_provider_T", 0.0))
        out["B_vertical_provider_T"] = self.vertical_field(t_s)
        out["B_vertical_topology_T"] = base_topology + sum(point.B_Z_T for point in edge_points)
        out["vertical_field_state_T"] = self.vertical_field(t_s)
        out["vertical_gradient_T_per_m"] = float(
            out.get("vertical_gradient_T_per_m", 0.0)
        ) + self._loop_vertical_gradient(t_s)
        for index, (loop, point) in enumerate(zip(self.loops, axis_points)):
            prefix = f"loop_{index}"
            out[f"{prefix}_name"] = loop.name
            out[f"{prefix}_radius_m"] = loop.radius_m
            out[f"{prefix}_z0_m"] = loop.z0_m
            out[f"{prefix}_turns"] = float(loop.turns)
            out[f"{prefix}_current_A"] = loop.current_at(t_s)
            out[f"{prefix}_B_R_axis_T"] = point.B_R_T
            out[f"{prefix}_B_Z_axis_T"] = point.B_Z_T
        return out


def make_biot_savart_field(
    geometry: GeometryConfig,
    loops: Iterable[CircularLoop],
    base_provider: MagneticFieldProvider | None = None,
) -> BiotSavartMagneticField:
    """Convenience factory for explicit provider injection."""

    return BiotSavartMagneticField(geometry, loops, base_provider=base_provider)


__all__ = [
    "CircularLoop",
    "BiotSavartMagneticField",
    "circular_loop_field",
    "circular_loop_flux",
    "make_biot_savart_field",
]
