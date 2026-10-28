"""碰撞、电离和能量损失率。

这里的默认闭合是一个“可运行的占位模型”，用平滑的截面等效拟合产生
数量级正确的反应率。生产计算必须把 :class:`CollisionRateModel` 换成由
LXCat/NIST/AMJUEL 截面和 Boltzmann 或 Fokker--Planck EEDF 计算的 provider，
并在运行清单中记录数据库版本与适用范围。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import GasConfig
from ..constants import E_CHARGE_C, M_E_KG, PI


@dataclass(frozen=True)
class RateCoefficients:
    K_ion_m3_s: float
    K_exc_m3_s: float
    K_att_m3_s: float
    alpha_rec_m3_s: float
    K_momentum_m3_s: float
    nu_ei_s: float
    eedf_label: str = "analytic_surrogate"


class CollisionRateModel:
    """由局域平均电子能量和 E/N 返回反应率。

    参数
    ----
    gas:
        气体和低阶截面缩放参数。
    ln_lambda:
        电子--离子碰撞的 Coulomb 对数；真正的动理学模型应自行计算它。
    """

    def __init__(self, gas: GasConfig, ln_lambda: float = 10.0) -> None:
        self.gas = gas
        self.ln_lambda = max(2.0, float(ln_lambda))

    @staticmethod
    def _safe_te(Te_eV: float) -> float:
        return max(1.0e-3, float(Te_eV))

    def evaluate(self, Te_eV: float, ne_m3: float, nn_m3: float, EoverN_Td: float) -> RateCoefficients:
        """返回 SI 反应率系数；系数本身不含密度。"""

        Te = self._safe_te(Te_eV)
        ne = max(0.0, float(ne_m3))
        nn = max(0.0, float(nn_m3))
        # 低阶 E/N 修正，只用于让参数扫描可运行；不是数据库截面。
        en = max(1.0e-6, float(EoverN_Td))
        field_factor = 1.0 + 0.15 * math.tanh(math.log10(en / 100.0))
        threshold = math.exp(-self.gas.ionization_energy_eV / Te)
        K_ion = self.gas.ionization_rate_scale_m3_s * math.sqrt(Te / 15.4) * threshold * field_factor
        K_ion = max(0.0, K_ion)

        K_exc = self.gas.excitation_rate_scale_m3_s * math.sqrt(Te / 10.0) * math.exp(
            -self.gas.excitation_energy_eV / Te
        )
        K_exc = max(0.0, K_exc)

        # 对 D2/H2 默认设为近似零；用户可用 attachment_coeff_m3_s 注入截面结果。
        K_att = max(0.0, self.gas.attachment_coeff_m3_s)
        alpha_rec = max(0.0, self.gas.recombination_coeff_m3_s)
        K_mom = self.gas.momentum_rate_scale_m3_s * math.sqrt(Te / 10.0)

        # ν_ei = 2.91e-12 n_e lnΛ / Te^(3/2)，n_e 使用 m^-3。
        nu_ei = 2.91e-12 * ne * self.ln_lambda / (Te**1.5)
        # 避免未电离极限下的除零；它不是额外的物理碰撞源。
        _ = nn
        return RateCoefficients(K_ion, K_exc, K_att, alpha_rec, K_mom, max(0.0, nu_ei))

    def electron_thermal_speed_m_s(self, Te_eV: float) -> float:
        Te = self._safe_te(Te_eV)
        return math.sqrt(8.0 * Te * E_CHARGE_C / (PI * M_E_KG))

    def ion_sound_speed_m_s(self, Te_eV: float, Ti_eV: float) -> float:
        Te = max(1.0e-3, float(Te_eV))
        Ti = max(0.0, float(Ti_eV))
        return math.sqrt((Te + Ti) * E_CHARGE_C / self.gas.ion_mass_kg)

    def hot_ionization_rate(self, mean_energy_eV: float, EoverN_Td: float) -> float:
        """高能尾电子的等效电离系数。

        这是把 2V EEDF 尚未接入时的接口占位。能量低于电离阈值时返回零，
        高能时使用平滑的有效截面；生产实现应直接对 EEDF 积分。
        """

        energy = max(0.0, float(mean_energy_eV))
        if energy <= self.gas.ionization_energy_eV:
            return 0.0
        field_factor = 1.0 + 0.10 * math.tanh(math.log10(max(EoverN_Td, 1.0e-6) / 100.0))
        return max(
            0.0,
            0.8
            * self.gas.ionization_rate_scale_m3_s
            * math.sqrt(energy / self.gas.ionization_energy_eV)
            * math.exp(-self.gas.ionization_energy_eV / energy)
            * field_factor,
        )
