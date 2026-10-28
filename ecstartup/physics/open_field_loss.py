"""开磁力线损失和壁面边界的低阶闭合。"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GeometryConfig
from ..state import GlobalState
from .magnetic import MagneticConfigurationState
from .rates import CollisionRateModel


@dataclass(frozen=True)
class LossTimes:
    electron_s: float
    ion_s: float
    energy_s: float
    connection_length_m: float = 0.0
    configuration_multiplier: float = 1.0


class OpenFieldLossModel:
    """用连接长度、热速度和声速估计开场线损失时间。

    真实实现应由场线追踪、漂移轨道、鞘层和壁几何 provider 替换。
    """

    def __init__(self, geometry: GeometryConfig, rates_model: CollisionRateModel, Ti_eV: float) -> None:
        self.geometry = geometry
        self.rates_model = rates_model
        self.Ti_eV = Ti_eV

    def evaluate(
        self,
        state: GlobalState,
        closed_fraction: float = 0.0,
        connection_length_m: float | None = None,
        magnetic_state: MagneticConfigurationState | None = None,
    ) -> LossTimes:
        vte = self.rates_model.electron_thermal_speed_m_s(state.Te_eV)
        cs = self.rates_model.ion_sound_speed_m_s(state.Te_eV, self.Ti_eV)
        # ``TopologyResult.connection_length_m`` is an *effective* length: the
        # topology provider has already folded its configuration factor into
        # it.  In that path do not multiply the same TPC/VFC factor again.
        # Direct callers that omit a length retain the legacy magnetic-state
        # multiplier so the closure is applied exactly once in either API.
        field_factor = 1.0
        if magnetic_state is not None and connection_length_m is None:
            field_factor = max(1.0e-9, magnetic_state.loss_multiplier)
        base_length = (
            max(1.0e-9, float(connection_length_m))
            if connection_length_m is not None
            else self.geometry.connection_length_m
        )
        confinement_factor = field_factor * (
            1.0 + self.geometry.closed_confinement_gain * max(0.0, min(1.0, closed_fraction))
        )
        tau_e = base_length / max(vte, 1.0) * confinement_factor
        tau_i = base_length / max(cs, 1.0) * confinement_factor
        tau_energy = max(1.0e-9, 4.0 * tau_e)
        return LossTimes(
            max(1.0e-9, tau_e),
            max(1.0e-9, tau_i),
            tau_energy,
            base_length,
            confinement_factor,
        )
