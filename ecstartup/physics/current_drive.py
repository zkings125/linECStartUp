"""非欧姆电流机制分解。

这里把磁位形和电流机制分成两层：TPC/VFC 是 ``MagneticFieldProvider``
的配置名，而 ``I_tail``/``I_vertical_charge`` 是可被实验诊断约束的
电流通道。``I_orbit``、``I_tpc``、``I_vfc`` 和 ``I_vtf`` 仅作为兼容或
文献映射别名，避免把不同论文的缩写混为一个物理量。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import GeometryConfig, RFConfig
from ..constants import E_CHARGE_C, M_E_KG, PI
from ..state import GlobalState
from .magnetic import MagneticConfigurationState
from .open_field_loss import LossTimes
from .rf import RFSourceTerms


@dataclass(frozen=True)
class CurrentComponents:
    """当前步的非欧姆电流分项（A）。

    ``I_orbit_A`` 保留 v0.1 字段名，语义上对应 ``I_tail_A``：开场线热
    尾的优先约束/损失不对称项。``I_vfc_A`` 对应垂直电荷分离闭合，并不
    表示 VFC 磁场本身必然产生电流；其系数必须由输入或实验标定给出。
    """

    I_rf_A: float
    I_orbit_A: float
    I_vfc_A: float = 0.0
    I_kinetic_A: float = 0.0
    current_drive_model: str = "empirical_rf_hot_tail"

    @property
    def I_tail_A(self) -> float:
        return self.I_orbit_A

    @property
    def I_tpc_A(self) -> float:
        """历史/文献别名；不是 TPC 位形的独立电流方程。"""

        return self.I_orbit_A

    @property
    def I_vertical_charge_A(self) -> float:
        return self.I_vfc_A

    @property
    def I_vtf_A(self) -> float:
        """EXL-50 文献常用的 vertical charge-separation 别名。"""

        return self.I_vfc_A

    @property
    def I_drive_A(self) -> float:
        return self.I_rf_A + self.I_orbit_A + self.I_vfc_A


class CurrentDriveModel:
    """独立的非欧姆电流闭合。

    ``I_rf`` 可由射线/准线性动理学替换；``I_tail`` 可由导引中心或
    2V 动理学替换；``I_vertical_charge`` 是一个基于净壁面电荷通量的
    校准闭合。所有 surrogate 都通过显式参数和诊断输出，不能当作普适
    的 A/W 常数。
    """

    def __init__(self, geometry: GeometryConfig, rf: RFConfig) -> None:
        self.geometry = geometry
        self.rf = rf

    def evaluate(
        self,
        state: GlobalState,
        rf_terms: RFSourceTerms,
        magnetic_state: MagneticConfigurationState | None = None,
        loss_times: LossTimes | None = None,
        vertical_field_T: float | None = None,
        t_s: float | None = None,
    ) -> CurrentComponents:
        if state.n_hot_m3 > 0.0 and state.hot_energy_J_m3 > 0.0:
            mean_energy = state.hot_energy_J_m3 / state.n_hot_m3
            v_hot = math.sqrt(max(0.0, 2.0 * mean_energy / M_E_KG))
        else:
            v_hot = 0.0

        # 热尾优先约束/损失不对称项。圆截面基线保持 v0.1 公式；TPC
        # 的镜像指标只作为一个可审计的增强因子进入，不会被误认为直接
        # ECCD 效率。
        tail = self.rf.current_drive_sign * E_CHARGE_C * state.n_hot_m3 * self.geometry.resolved_volume_m3
        tail *= v_hot / max(2.0 * PI * self.geometry.R0_m, 1.0e-9)
        tail *= self.rf.orbit_asymmetry
        tail_factor = 1.0
        if magnetic_state is not None:
            # The field provider already modifies loss times through
            # connection_length_factor. Do not multiply the current by the
            # same confinement factor a second time; retain this metric only
            # as a diagnostic for a future kinetic closure.
            tail_factor += 0.0
        tail *= tail_factor

        # 垂直电荷分离项使用净电子/离子壁面通量的差作为驱动尺度：
        # e V (Gamma_e - Gamma_i).  coefficient=0 时严格关闭，避免在没有
        # 限制器/壁电流标定时凭空加入一个“VFC 电流”。
        coefficient = self.geometry.effective_vertical_charge_fraction
        if loss_times is not None:
            gamma_e = max(0.0, state.ne_m3) / max(loss_times.electron_s, 1.0e-12)
            gamma_i = max(0.0, state.ni_m3) / max(loss_times.ion_s, 1.0e-12)
            charge_flux_rate = gamma_e - gamma_i
        else:
            # 向后兼容直接调用 evaluate 的代码；热尾损失是最小可用尺度。
            charge_flux_rate = max(0.0, state.n_hot_m3) / max(self.rf.hot_loss_time_s, 1.0e-12)
        b_v = vertical_field_T
        if b_v is None:
            b_v = self.geometry.B_vertical_T
        # Use the signed field ratio directly.  This gives an exactly zero
        # contribution at ``B_vertical=0`` and keeps the response continuous
        # for arbitrarily small non-zero fields (including polarity reversal).
        b_ratio = float(b_v) / max(abs(self.geometry.B_t_T), 1.0e-9)
        vertical = self.rf.current_drive_sign * E_CHARGE_C * self.geometry.resolved_volume_m3
        vertical *= coefficient * b_ratio * charge_flux_rate
        return CurrentComponents(rf_terms.current_drive_A, tail, vertical)
