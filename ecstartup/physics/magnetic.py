"""磁场 provider 与启动前磁位形的低阶实现。

`CircularMagneticField` 仍然是 v0.1 的回归基线。`TPCMagneticField` 和
`VFCMagneticField` 只是在同一个可审计接口下提供两种不同的解析闭合：

* TPC (trapped-particle configuration) 暴露镜像比、捕获粒子比例和衰减
  指数，用来表示镜像约束及其对开场线损失的影响；
* VFC (vertical-field configuration) 保留外加垂直场控制，并暴露垂直场
  衰减指数和连接长度因子。

这些解析场不是线圈 Biot--Savart 或自由边界 Grad--Shafranov 解。真实
装置使用时，只需实现同一个 ``MagneticFieldProvider`` 契约即可替换。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Protocol

import numpy as np

from ..config import CoilComponentConfig, GeometryConfig
from ..constants import MU0_H_M, PI
from ..waveform import linear_value


@dataclass(frozen=True)
class FieldPoint:
    """单个 (R, Z) 点的轴对称磁场分量，单位 T。"""

    B_R_T: float
    B_Z_T: float
    B_phi_T: float

    @property
    def magnitude_T(self) -> float:
        return math.sqrt(self.B_R_T**2 + self.B_Z_T**2 + self.B_phi_T**2)


@dataclass(frozen=True)
class MagneticConfigurationState:
    """供损失、电流和拓扑模块使用的位形级指标。

    这些量是从场 provider 得到的降阶指标，而不是新的独立状态变量。
    ``loss_multiplier`` 和 ``connection_length_factor`` 明确标出经验闭合
    的位置，便于将来由场线追踪结果替换。
    """

    configuration: str
    mirror_ratio: float = 1.0
    trapped_fraction: float = 0.0
    decay_index: float = 0.0
    vertical_alignment: float = 0.0
    connection_length_factor: float = 1.0
    loss_multiplier: float = 1.0
    vertical_field_T: float = 0.0
    vertical_gradient_T_per_m: float = 0.0

    @property
    def mirror_confinement_factor(self) -> float:
        return max(1.0, 1.0 + self.trapped_fraction * max(0.0, self.mirror_ratio - 1.0))

    @property
    def tail_confinement_metric(self) -> float:
        """无量纲的热尾优先约束指标，零表示没有 TPC 镜像效应。"""

        return max(0.0, self.mirror_confinement_factor - 1.0)


class MagneticFieldProvider(Protocol):
    """Analytic, Biot--Savart and GS backends share this interface."""

    configuration: str

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint: ...

    def flux_function(self, R_m, Z_m, Ip_A, t_s: float = 0.0): ...

    def vertical_field(self, t_s: float = 0.0) -> float: ...

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState: ...

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]: ...


class CircularMagneticField:
    """圆截面、轴对称的自场近似（软件联调基线）。"""

    def __init__(self, geometry: GeometryConfig) -> None:
        self.geometry = geometry
        self.configuration = "circular"

    def _effective_ip(self, Ip_A: float) -> float:
        return float(Ip_A)

    def vertical_field(self, t_s: float = 0.0) -> float:
        return linear_value(self.geometry.vertical_field_waveform, t_s, self.geometry.B_vertical_T)

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        x = float(R_m) - self.geometry.R0_m
        z = float(Z_m)
        r = max(math.hypot(x, z), 0.02 * self.geometry.minor_radius_m)
        bp_edge = MU0_H_M * self._effective_ip(Ip_A) / (2.0 * PI * self.geometry.minor_radius_m)
        bp = bp_edge * self.geometry.minor_radius_m / r
        return FieldPoint(
            B_R_T=-bp * z / r,
            B_Z_T=bp * x / r + self.vertical_field(t_s),
            B_phi_T=self.geometry.B_t_T * self.geometry.R0_m / max(float(R_m), 1.0e-6),
        )

    def flux_function(self, R_m, Z_m, Ip_A, t_s: float = 0.0):
        """局域 psi 近似（Wb/rad），只用于画图/拓扑替代判据。"""

        R = np.asarray(R_m, dtype=float)
        Z = np.asarray(Z_m, dtype=float)
        r2 = (R - self.geometry.R0_m) ** 2 + Z**2
        bp_edge = MU0_H_M * self._effective_ip(Ip_A) / (2.0 * PI * self.geometry.minor_radius_m)
        return 0.5 * self.geometry.R0_m * bp_edge * r2 + 0.5 * self.vertical_field(t_s) * R**2

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        return MagneticConfigurationState(
            configuration=self.configuration,
            vertical_field_T=self.vertical_field(t_s),
        )

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        a = self.geometry.minor_radius_m
        bp = MU0_H_M * abs(self._effective_ip(Ip_A)) / (2.0 * PI * a)
        state = self.configuration_state(Ip_A, t_s)
        return {
            "field_configuration": self.configuration,
            "Bp_edge_provider_T": bp,
            "B_vertical_provider_T": self.vertical_field(t_s),
            "B_vertical_topology_T": self.vertical_field(t_s),
            "mirror_ratio": state.mirror_ratio,
            "trapped_fraction": state.trapped_fraction,
            "decay_index": state.decay_index,
            "vertical_alignment": state.vertical_alignment,
            "connection_length_factor": state.connection_length_factor,
            "field_loss_multiplier": state.loss_multiplier,
            "mirror_confinement_factor": state.mirror_confinement_factor,
            "tail_confinement_metric": state.tail_confinement_metric,
            "vertical_field_state_T": state.vertical_field_T,
            "vertical_gradient_T_per_m": state.vertical_gradient_T_per_m,
        }


class AnalyticCoilComponent:
    """单个线圈的局部解析场 surrogate。

    The component is linear in its prescribed current.  The poloidal part is
    generated from the local stream function

    ``psi = I * (B_Z_per_A * R**2 / 2 - B_R_per_A * R * Z)``

    so that ``div(B)=0`` for the axisymmetric approximation.  The toroidal
    part uses the usual ``R0/R`` scaling.  This is deliberately a small,
    auditable closure: the coefficients are not a coil geometry or a
    Biot--Savart solution, and should be replaced by a real field map when
    quantitative device predictions are required.
    """

    def __init__(self, config: CoilComponentConfig, geometry: GeometryConfig) -> None:
        self.config = config
        self.geometry = geometry

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role

    def current_A(self, t_s: float = 0.0) -> float:
        return linear_value(self.config.current_waveform_A, t_s, self.config.current_A)

    def evaluate(self, R_m: float, Z_m: float, t_s: float = 0.0) -> FieldPoint:
        current = self.current_A(t_s)
        br = current * self.config.B_R_T_per_A
        bz_axis = current * self.config.B_Z_T_per_A
        bphi_axis = current * self.config.B_phi_T_per_A
        radius = max(float(R_m), 1.0e-9)
        # Stream-function-consistent local field.  A non-zero B_R coefficient
        # therefore also gives the required compensating Z dependence in B_Z.
        bz = bz_axis - br * float(Z_m) / radius
        bphi = bphi_axis * self.geometry.R0_m / radius
        return FieldPoint(br, bz, bphi)

    def flux_function(self, R_m, Z_m, t_s: float = 0.0):
        current = self.current_A(t_s)
        R = np.asarray(R_m, dtype=float)
        Z = np.asarray(Z_m, dtype=float)
        return current * (
            0.5 * self.config.B_Z_T_per_A * R**2
            - self.config.B_R_T_per_A * R * Z
        )


class CompositeMagneticField:
    """叠加一个基础 provider 与多个时变解析线圈分量。

    ``base`` 仍负责自场、TPC/VFC 的降阶拓扑指标；组件只对 ``B_R``、
    ``B_Z``、``B_phi`` 和 ``psi`` 做线性叠加。这样可以先用
    ``circular + coil_components`` 表示独立 TPC/VFC 线圈电流，也可以在
    TPC/VFC surrogate 上添加校正线圈，而不把线圈细节耦合进 ODE。
    组件的 ``role`` 仅是诊断标签，不会自动改变镜像比、损失时间或电流
    机制；这些量必须由真实场线/GS provider 或显式闭合提供。
    """

    def __init__(
        self,
        base: MagneticFieldProvider,
        components: tuple[CoilComponentConfig, ...] | list[CoilComponentConfig],
        geometry: GeometryConfig,
    ) -> None:
        self.base = base
        self.geometry = geometry
        normalized_components = (
            item if isinstance(item, CoilComponentConfig) else CoilComponentConfig.model_validate(item)
            for item in components
        )
        self.components = tuple(AnalyticCoilComponent(item, geometry) for item in normalized_components)
        if not self.components:
            raise ValueError("CompositeMagneticField 至少需要一个线圈分量")
        self.configuration = f"composite[{getattr(base, 'configuration', 'custom')}]"

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        point = self.base.evaluate(R_m, Z_m, Ip_A, t_s)
        br = point.B_R_T
        bz = point.B_Z_T
        bphi = point.B_phi_T
        for component in self.components:
            addition = component.evaluate(R_m, Z_m, t_s)
            br += addition.B_R_T
            bz += addition.B_Z_T
            bphi += addition.B_phi_T
        return FieldPoint(br, bz, bphi)

    def flux_function(self, R_m, Z_m, Ip_A, t_s: float = 0.0):
        psi = self.base.flux_function(R_m, Z_m, Ip_A, t_s)
        for component in self.components:
            psi = psi + component.flux_function(R_m, Z_m, t_s)
        return psi

    def vertical_field(self, t_s: float = 0.0) -> float:
        axis_R = self.geometry.R0_m
        return float(
            self.base.vertical_field(t_s)
            + sum(component.evaluate(axis_R, 0.0, t_s).B_Z_T for component in self.components)
        )

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        base_state = self.base.configuration_state(Ip_A, t_s)
        if not isinstance(base_state, MagneticConfigurationState):
            # Third-party providers may return a structurally compatible object;
            # preserve it rather than making the composite wrapper unusable.
            return base_state
        b_vertical = self.vertical_field(t_s)
        return replace(
            base_state,
            configuration=self.configuration,
            vertical_alignment=0.0 if b_vertical == 0.0 else math.copysign(1.0, b_vertical),
            vertical_field_T=b_vertical,
        )

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        out = dict(self.base.diagnostics(Ip_A, t_s))
        axis_R = self.geometry.R0_m
        edge_R = self.geometry.R0_m + self.geometry.minor_radius_m
        axis_points = [component.evaluate(axis_R, 0.0, t_s) for component in self.components]
        edge_points = [component.evaluate(edge_R, 0.0, t_s) for component in self.components]
        total_axis = FieldPoint(
            sum(point.B_R_T for point in axis_points),
            sum(point.B_Z_T for point in axis_points),
            sum(point.B_phi_T for point in axis_points),
        )
        # The base provider may use an edge/flux-surface representative for
        # topology (not its axis value). Add the same representative component
        # vertical field once, keeping TPC/VFC topology closures auditable.
        # The CFS surrogate intentionally retains the base provider's
        # plasma-current Bp: an external coil contribution cannot be classified
        # as confining Bp without field-line tracing or a GS solve.
        base_topology = float(out.get("B_vertical_topology_T", out.get("B_vertical_provider_T", 0.0)))
        out["Bp_edge_base_T"] = float(out.get("Bp_edge_provider_T", 0.0))
        out["coil_Bp_edge_T"] = math.hypot(
            sum(point.B_R_T for point in edge_points),
            sum(point.B_Z_T for point in edge_points),
        )
        out["field_configuration"] = self.configuration
        out["field_provider"] = "composite_analytic_coils"
        out["base_field_configuration"] = str(getattr(self.base, "configuration", "custom"))
        out["coil_component_count"] = float(len(self.components))
        out["coil_current_total_A"] = float(sum(component.current_A(t_s) for component in self.components))
        out["coil_B_R_axis_T"] = total_axis.B_R_T
        out["coil_B_Z_axis_T"] = total_axis.B_Z_T
        out["coil_B_phi_axis_T"] = total_axis.B_phi_T
        out["B_vertical_provider_T"] = self.vertical_field(t_s)
        out["B_vertical_topology_T"] = base_topology + sum(point.B_Z_T for point in edge_points)
        out["vertical_field_state_T"] = self.vertical_field(t_s)
        out["vertical_alignment"] = 0.0 if abs(self.vertical_field(t_s)) <= 1.0e-15 else math.copysign(1.0, self.vertical_field(t_s))
        for index, (component, point) in enumerate(zip(self.components, axis_points)):
            prefix = f"coil_{index}"
            out[f"{prefix}_name"] = component.name
            out[f"{prefix}_role"] = component.role
            out[f"{prefix}_current_A"] = component.current_A(t_s)
            out[f"{prefix}_B_R_axis_T"] = point.B_R_T
            out[f"{prefix}_B_Z_axis_T"] = point.B_Z_T
            out[f"{prefix}_B_phi_axis_T"] = point.B_phi_T
        return out


class TPCMagneticField(CircularMagneticField):
    """Trapped-particle configuration 的解析 surrogate。

    TPC 的关键差别不是“把 Ip 放大”，而是镜像场导致的捕获粒子群和
    较长的有效约束时间。这里仍用一个圆截面自场作底图，并以
    ``tpc_mirror_ratio``/``tpc_trapped_fraction`` 显式携带该差别；局部
    toroidal 场的平滑修正只用于让场图和诊断可见，不能替代真实线圈场。
    """

    def __init__(self, geometry: GeometryConfig) -> None:
        super().__init__(geometry)
        self.configuration = "TPC"

    def _effective_ip(self, Ip_A: float) -> float:
        # 保留旧版 scale 的可重复行为；它代表当前解析 provider 的幅值
        # 校准，而不是 TPC 的物理定义。
        return float(Ip_A) * self.geometry.field_configuration_scale

    def _mirror_shape(self, R_m: float, Z_m: float) -> float:
        """Return the normalized analytic mirror profile at ``(R, Z)``.

        Keeping this profile in one helper makes the field evaluation and its
        diagnostic representative use exactly the same closure.  The profile
        is only a surrogate for a field-line ``B(s)`` calculation.
        """

        a = max(self.geometry.minor_radius_m, 1.0e-9)
        rho_norm = min(
            1.0,
            math.hypot(float(R_m) - self.geometry.R0_m, float(Z_m)) / a,
        )
        profile_power = max(0.25, 1.0 + self.geometry.tpc_decay_index)
        return 1.0 + max(0.0, self.geometry.tpc_mirror_ratio - 1.0) * rho_norm**profile_power

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        base = super().evaluate(R_m, Z_m, Ip_A, t_s)
        # A mirror is represented as a toroidal-field variation over the
        # poloidal cross-section. In an axisymmetric model this perturbation
        # does not add div(B) from the toroidal component. The profile is
        # normalized so that B_phi on axis is unchanged and the nominal edge
        # ratio approaches tpc_mirror_ratio. A real mirror ratio must instead
        # be computed from B(s) along traced field lines.
        return FieldPoint(base.B_R_T, base.B_Z_T, base.B_phi_T * self._mirror_shape(R_m, Z_m))

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        mirror = self.geometry.tpc_mirror_ratio
        trapped = self.geometry.tpc_trapped_fraction
        factor = 1.0 + trapped * max(0.0, mirror - 1.0)
        b_vertical = self.vertical_field(t_s)
        return MagneticConfigurationState(
            configuration=self.configuration,
            mirror_ratio=mirror,
            trapped_fraction=trapped,
            decay_index=self.geometry.tpc_decay_index,
            vertical_alignment=0.0 if b_vertical == 0.0 else math.copysign(1.0, b_vertical),
            connection_length_factor=factor,
            loss_multiplier=factor,
            vertical_field_T=b_vertical,
        )

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        out = super().diagnostics(Ip_A, t_s)
        out["tpc_self_field_scale"] = self.geometry.field_configuration_scale
        out["tpc_mirror_ratio"] = self.geometry.tpc_mirror_ratio
        out["tpc_trapped_fraction"] = self.geometry.tpc_trapped_fraction
        out["tpc_decay_index"] = self.geometry.tpc_decay_index
        # A mid-radius value makes the effect of ``tpc_decay_index`` visible
        # in diagnostics even when the RF launch point is on axis (where all
        # normalized profiles are one).  It is not a measured mirror ratio.
        sample_R = self.geometry.R0_m + 0.5 * self.geometry.minor_radius_m
        sample = self.evaluate(sample_R, 0.0, Ip_A, t_s)
        out["tpc_B_phi_midradius_T"] = sample.B_phi_T
        out["tpc_mirror_shape_midradius"] = self._mirror_shape(sample_R, 0.0)
        return out


class VFCMagneticField(CircularMagneticField):
    """Vertical-field configuration 的解析 surrogate。

    VFC 保留等效等离子体自场，显式缩放外加垂直场，并允许输入垂直场
    衰减指数和连接长度因子。真实 VFC 线圈几何应通过同一接口以
    Biot--Savart 或自由边界 GS provider 实现。
    """

    def __init__(self, geometry: GeometryConfig) -> None:
        super().__init__(geometry)
        self.configuration = "VFC"

    def vertical_field(self, t_s: float = 0.0) -> float:
        return super().vertical_field(t_s) * self.geometry.field_configuration_scale

    def vertical_field_at(self, R_m: float, Z_m: float = 0.0, t_s: float = 0.0) -> float:
        """Evaluate the analytic VFC radial profile at a point.

        ``vfc_decay_index`` uses the conventional local definition
        ``n=-R/Bz*dBz/dR`` at the magnetic axis.  The exponential profile is
        only a low-order replacement for a coil-field map.
        """

        radius = max(self.geometry.R0_m, 1.0e-9)
        exponent = -self.geometry.vfc_decay_index * (float(R_m) - self.geometry.R0_m) / radius
        exponent = min(50.0, max(-50.0, exponent))
        return self.vertical_field(t_s) * math.exp(exponent)

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        base = super().evaluate(R_m, Z_m, Ip_A, t_s)
        # Bz(R) independent of Z is divergence-free for an axisymmetric field
        # when the added Br is zero.  This keeps the analytic VFC profile
        # internally consistent while exposing radial decay to ray launch and
        # diagnostics.
        delta_bz = self.vertical_field_at(R_m, Z_m, t_s) - self.vertical_field(t_s)
        return FieldPoint(base.B_R_T, base.B_Z_T + delta_bz, base.B_phi_T)

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        b_vertical = self.vertical_field(t_s)
        factor = self.geometry.vfc_connection_length_factor
        return MagneticConfigurationState(
            configuration=self.configuration,
            mirror_ratio=1.0,
            trapped_fraction=0.0,
            decay_index=self.geometry.vfc_decay_index,
            vertical_alignment=0.0 if b_vertical == 0.0 else math.copysign(1.0, b_vertical),
            connection_length_factor=factor,
            loss_multiplier=factor,
            vertical_field_T=b_vertical,
            vertical_gradient_T_per_m=-self.geometry.vfc_decay_index * b_vertical / max(self.geometry.R0_m, 1.0e-9),
        )

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        out = super().diagnostics(Ip_A, t_s)
        out["vfc_vertical_field_scale"] = self.geometry.field_configuration_scale
        out["vfc_decay_index"] = self.geometry.vfc_decay_index
        out["vfc_connection_length_factor"] = self.geometry.vfc_connection_length_factor
        edge_r = self.geometry.R0_m + self.geometry.minor_radius_m
        out["vfc_vertical_field_edge_T"] = self.vertical_field_at(edge_r, 0.0, t_s)
        # Representative field used by the topology surrogate. A real
        # provider should supply a flux-surface or field-line average.
        out["B_vertical_topology_T"] = self.vertical_field_at(edge_r, 0.0, t_s)
        out["vertical_gradient_T_per_m"] = -self.geometry.vfc_decay_index * self.vertical_field(t_s) / max(self.geometry.R0_m, 1.0e-9)
        return out


def make_magnetic_field(geometry: GeometryConfig) -> MagneticFieldProvider:
    """Construct the configured provider while retaining the old default.

    When ``geometry.coil_components`` is non-empty, the selected historical
    provider becomes the base and is wrapped by :class:`CompositeMagneticField`.
    This keeps old ``circular``/``TPC``/``VFC`` cases byte-for-byte compatible
    while making additive, time-dependent coil currents available to new cases.
    """

    providers = {
        "circular": CircularMagneticField,
        "TPC": TPCMagneticField,
        "VFC": VFCMagneticField,
    }
    try:
        provider = providers[geometry.field_configuration]
    except KeyError as exc:  # defensive for custom objects bypassing Pydantic
        raise ValueError(f"未知磁场位形: {geometry.field_configuration!r}") from exc
    base = provider(geometry)
    # Keep the historical analytic factory as the default.  The Biot--Savart
    # backend is opt-in and constructed lazily to avoid importing scipy's
    # elliptic-integral routines for ordinary surrogate runs.  Wrapping the
    # selected base preserves TPC/VFC configuration-state semantics.
    if getattr(geometry, "magnetic_backend", "analytic") == "biot_savart":
        from .biot_savart import BiotSavartMagneticField, CircularLoop

        loops = tuple(
            item if isinstance(item, CircularLoop) else CircularLoop(**item.model_dump(mode="python"))
            for item in geometry.biot_savart_loops
        )
        return BiotSavartMagneticField(geometry, loops, base_provider=base)
    if geometry.coil_components:
        return CompositeMagneticField(base, geometry.coil_components, geometry)
    return base
