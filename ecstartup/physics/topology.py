"""闭合磁面(CFS)诊断接口与解析位形替代实现。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import GeometryConfig
from .magnetic import MagneticConfigurationState, MagneticFieldProvider, make_magnetic_field


@dataclass(frozen=True)
class TopologyResult:
    closed_fraction: float
    candidate_radius_m: float
    Bp_edge_T: float
    threshold_T: float
    has_closed_surface: bool
    connection_length_m: float
    method: str
    confidence: float
    reason: str
    # 位形元数据；追加在旧字段之后以保持 v0.1 的位置参数兼容性。
    field_configuration: str = "circular"
    mirror_ratio: float = 1.0
    trapped_fraction: float = 0.0
    decay_index: float = 0.0
    connection_length_factor: float = 1.0
    loss_multiplier: float = 1.0


class SurrogateTopologyDetector:
    """用外部场与等效 Ip 的竞争估计 CFS 候选区。

    真实版本应替换为自由边界 Grad--Shafranov + 场线追踪。这里仍保留
    TPC/VFC provider 的位形指标，使不同配置不会在比较时悄然共用同一
    个连接长度/损失假设。
    """

    def __init__(self, geometry: GeometryConfig, field_provider: MagneticFieldProvider | None = None) -> None:
        self.geometry = geometry
        self.field = field_provider or make_magnetic_field(geometry)

    def _configuration_state(self, Ip_A: float, t_s: float) -> MagneticConfigurationState:
        method = getattr(self.field, "configuration_state", None)
        if callable(method):
            return method(Ip_A, t_s)
        # 兼容尚未升级的第三方 provider。
        return MagneticConfigurationState(configuration=getattr(self.field, "configuration", "custom"))

    def detect(self, Ip_A: float, t_s: float = 0.0) -> TopologyResult:
        a = self.geometry.minor_radius_m
        field_diagnostics = self.field.diagnostics(Ip_A, t_s)
        field_state = self._configuration_state(Ip_A, t_s)
        bp_edge = float(field_diagnostics["Bp_edge_provider_T"])
        # Use the provider's representative topology field when available;
        # VFC supplies an edge value so its radial decay affects the CFS
        # threshold instead of remaining a diagnostics-only parameter.
        vertical_field = float(field_diagnostics.get("B_vertical_topology_T", field_diagnostics["B_vertical_provider_T"]))
        threshold = abs(vertical_field) + self.geometry.B_penetration_T
        margin = bp_edge - threshold
        method = f"{field_state.configuration}_surrogate"
        base_connection = self.geometry.connection_length_m * max(field_state.connection_length_factor, 1.0e-9)
        if margin <= 0.0 or bp_edge <= 0.0:
            return TopologyResult(
                0.0,
                0.0,
                bp_edge,
                threshold,
                False,
                base_connection,
                method,
                0.1,
                "等效极向场尚未超过外部穿透场阈值",
                field_state.configuration,
                field_state.mirror_ratio,
                field_state.trapped_fraction,
                field_state.decay_index,
                field_state.connection_length_factor,
                field_state.loss_multiplier,
            )
        fraction = min(1.0, max(0.0, (margin / bp_edge) ** 1.5))
        # 简化的第一壁裁剪：圆心到 R/Z 边界的可容纳比例。
        radial_room = min(self.geometry.R0_m - self.geometry.R_bounds_m[0], self.geometry.R_bounds_m[1] - self.geometry.R0_m)
        vertical_room = min(abs(self.geometry.Z_bounds_m[0]), abs(self.geometry.Z_bounds_m[1]))
        room = max(0.0, min(radial_room, vertical_room))
        radius = a * math.sqrt(fraction)
        if radius > room:
            radius = room
            fraction = min(fraction, (radius / a) ** 2)
        has = fraction > 0.0
        conn = base_connection * (1.0 + 20.0 * fraction)
        return TopologyResult(
            fraction,
            radius,
            bp_edge,
            threshold,
            has,
            conn,
            method,
            0.25,
            "解析通量函数存在内部闭合候选区；需用 GS/场线追踪复核",
            field_state.configuration,
            field_state.mirror_ratio,
            field_state.trapped_fraction,
            field_state.decay_index,
            field_state.connection_length_factor,
            field_state.loss_multiplier,
        )
