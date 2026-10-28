"""外电路、感应场和电流闭合。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import CircuitConfig, GeometryConfig, RFConfig
from ..constants import E_CHARGE_C, M_E_KG
from ..state import GlobalState
from ..waveform import linear_value
from .current_drive import CurrentDriveModel
from .magnetic import MagneticConfigurationState
from .open_field_loss import LossTimes
from .rates import RateCoefficients
from .rf import RFSourceTerms


@dataclass(frozen=True)
class CircuitTerms:
    loop_voltage_V: float
    effective_loop_voltage_V: float
    E_loop_V_m: float
    nu_en_s: float
    nu_ei_s: float
    conductivity_S_m: float
    plasma_resistance_ohm: float
    I_ohmic_A: float
    I_rf_A: float
    I_orbit_A: float
    I_drive_A: float
    dIp_dt_A_s: float
    effective_charge: float = 1.0
    I_tpc_A: float = 0.0
    I_vfc_A: float = 0.0
    I_kinetic_A: float = 0.0
    current_drive_model: str = "empirical_rf_hot_tail"

    @property
    def I_tail_A(self) -> float:
        return self.I_orbit_A

    @property
    def I_vertical_charge_A(self) -> float:
        return self.I_vfc_A

    @property
    def I_vtf_A(self) -> float:
        return self.I_vfc_A


class CircuitModel:
    """降阶电路闭合。

    `dIp/dt` 采用带电导门控的 L-R 方程。电流分解中的 orbit 项是热尾
    优先约束/损失不对称的可调替代项，便于消融实验；它应由导引中心或
    2V 动理学模块替换。
    """

    def __init__(self, config: CircuitConfig, geometry: GeometryConfig, rf: RFConfig, current_drive=None) -> None:
        self.config = config
        self.geometry = geometry
        self.rf = rf
        self.current_drive = current_drive or CurrentDriveModel(geometry, rf)
        self.area_m2 = math.pi * geometry.minor_radius_m**2
        self.length_m = config.plasma_length_m or (2.0 * math.pi * geometry.R0_m)

    def loop_voltage(self, t_s: float) -> float:
        if self.config.cs_mode == "without_cs":
            return 0.0
        if self.config.loop_voltage_waveform:
            return linear_value(self.config.loop_voltage_waveform, t_s, self.config.loop_voltage_V)
        if self.config.loop_voltage_ramp_s <= 0.0:
            return self.config.loop_voltage_V
        return self.config.loop_voltage_V * min(1.0, max(0.0, t_s / self.config.loop_voltage_ramp_s))

    def evaluate(
        self,
        t_s: float,
        state: GlobalState,
        rates: RateCoefficients,
        rf_terms: RFSourceTerms,
        Te_eV: float,
        magnetic_state: MagneticConfigurationState | None = None,
        loss_times: LossTimes | None = None,
        vertical_field_T: float | None = None,
        effective_charge: float = 1.0,
    ) -> CircuitTerms:
        nn = max(0.0, state.nn_m3)
        # ν_en = n_n <σv>；低阶 K_momentum 来自 provider。
        nu_en = nn * rates.K_momentum_m3_s
        effective_charge = max(1.0, float(effective_charge))
        nu_ei = rates.nu_ei_s * effective_charge
        nu_total = max(1.0, nu_en + nu_ei)
        sigma = E_CHARGE_C**2 * max(0.0, state.ne_m3) / (M_E_KG * nu_total)
        sigma = max(self.config.conductivity_floor_S_m, sigma)
        resistance = self.length_m / (sigma * self.area_m2)
        resistance = min(self.config.resistance_cap_ohm, max(1.0e-12, resistance))
        V_loop = self.loop_voltage(t_s)
        E_loop = V_loop / self.length_m
        I_ohmic = sigma * E_loop * self.area_m2

        try:
            components = self.current_drive.evaluate(
                state,
                rf_terms,
                magnetic_state=magnetic_state,
                loss_times=loss_times,
                vertical_field_T=vertical_field_T,
                t_s=t_s,
            )
        except TypeError as exc:
            if "unexpected keyword argument 't_s'" not in str(exc):
                raise
            components = self.current_drive.evaluate(
                state,
                rf_terms,
                magnetic_state=magnetic_state,
                loss_times=loss_times,
                vertical_field_T=vertical_field_T,
            )
        I_drive = components.I_drive_A

        # 真空/弱电离时外加回路电压不应凭空产生等离子体电流。
        conductance_gate = sigma / (sigma + self.config.conductivity_floor_S_m)
        V_effective = conductance_gate * V_loop
        dIp_dt = (V_effective - resistance * state.Ip_A + resistance * I_drive) / self.config.total_inductance_H
        return CircuitTerms(
            loop_voltage_V=V_loop,
            effective_loop_voltage_V=V_effective,
            E_loop_V_m=E_loop,
            nu_en_s=nu_en,
            nu_ei_s=nu_ei,
            conductivity_S_m=sigma,
            plasma_resistance_ohm=resistance,
            I_ohmic_A=I_ohmic,
            I_rf_A=components.I_rf_A,
            I_orbit_A=components.I_orbit_A,
            I_drive_A=I_drive,
            dIp_dt_A_s=dIp_dt,
            effective_charge=effective_charge,
            I_tpc_A=components.I_tpc_A,
            I_vfc_A=components.I_vfc_A,
            I_kinetic_A=components.I_kinetic_A,
            current_drive_model=components.current_drive_model,
        )
