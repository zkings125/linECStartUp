"""Pre-CFS EC resonance accessibility and startup-difficulty diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ..config import RFSourceConfig, SimulationConfig
from ..constants import E_CHARGE_C, M_E_KG, PI


@dataclass(frozen=True)
class HarmonicAccessibility:
    source_name: str
    frequency_GHz: float
    harmonic: int
    resonance_B_T: float
    resonance_R_m: float
    accessible_in_plasma: bool
    plasma_R_interval_m: tuple[float, float]
    model: str = "vacuum_toroidal_B=B0*R0/R"

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class StartupResonanceAssessment:
    device: str
    aspect_ratio: float
    pressure_mPa: float
    collision_penalty: float
    mirror_ratio: float
    geometric_trapping_proxy: float
    fundamental_accessible: bool
    second_harmonic_accessible: bool
    startup_score: float
    hypotheses: tuple[str, ...]
    layers: tuple[HarmonicAccessibility, ...]
    evidence_level: str = "order-of-magnitude diagnostic; not experimental validation"

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "aspect_ratio": self.aspect_ratio,
            "pressure_mPa": self.pressure_mPa,
            "collision_penalty": self.collision_penalty,
            "mirror_ratio": self.mirror_ratio,
            "geometric_trapping_proxy": self.geometric_trapping_proxy,
            "fundamental_accessible": self.fundamental_accessible,
            "second_harmonic_accessible": self.second_harmonic_accessible,
            "startup_score": self.startup_score,
            "hypotheses": list(self.hypotheses),
            "layers": [item.to_dict() for item in self.layers],
            "evidence_level": self.evidence_level,
            "equilibrium_model": "none; pre-CFS vacuum toroidal-field geometry",
        }


def resonance_field_T(frequency_Hz: float, harmonic: int) -> float:
    if frequency_Hz <= 0.0 or harmonic < 1:
        raise ValueError("frequency must be positive and harmonic >= 1")
    return 2.0 * PI * M_E_KG * frequency_Hz / (harmonic * E_CHARGE_C)


def toroidal_resonance_layer(
    *,
    source_name: str,
    frequency_Hz: float,
    harmonic: int,
    B0_T: float,
    R0_m: float,
    minor_radius_m: float,
) -> HarmonicAccessibility:
    B_res = resonance_field_T(frequency_Hz, harmonic)
    R_res = B0_T * R0_m / B_res
    interval = (R0_m - minor_radius_m, R0_m + minor_radius_m)
    return HarmonicAccessibility(
        source_name, frequency_Hz / 1.0e9, harmonic, B_res, R_res,
        interval[0] <= R_res <= interval[1], interval,
    )


def _enabled_sources(config: SimulationConfig) -> Iterable[RFSourceConfig]:
    enabled = [source for source in config.rf.sources if source.enabled]
    if enabled:
        return enabled
    return (RFSourceConfig(
        name="legacy_rf",
        frequency_Hz=config.rf.frequency_Hz,
        harmonic=config.rf.harmonic,
        power_W=config.rf.power_W,
        absorption_fraction=config.rf.absorption_fraction,
        pulse=config.rf.pulse,
        launch=config.rf.launch,
    ),)


def assess_startup_resonance(config: SimulationConfig, *, device: str | None = None) -> StartupResonanceAssessment:
    """Evaluate harmonic access and transparent pre-CFS startup proxies.

    The score is deliberately diagnostic, not a fitted probability.  Its
    components are returned separately so experiments can falsify each
    hypothesis without treating this function as an empirical law.
    """

    geometry = config.geometry
    layers: list[HarmonicAccessibility] = []
    for source in _enabled_sources(config):
        # Always inspect both harmonics.  The configured harmonic remains the
        # wave solver choice, while this diagnostic answers whether a missing
        # second layer could explain poor energetic-electron production.
        for harmonic in (1, 2):
            layers.append(toroidal_resonance_layer(
                source_name=source.name,
                frequency_Hz=source.frequency_Hz,
                harmonic=harmonic,
                B0_T=geometry.B_t_T,
                R0_m=geometry.R0_m,
                minor_radius_m=geometry.minor_radius_m,
            ))
    fundamental = any(item.harmonic == 1 and item.accessible_in_plasma for item in layers)
    second = any(item.harmonic == 2 and item.accessible_in_plasma for item in layers)
    aspect = geometry.R0_m / geometry.minor_radius_m
    mirror = geometry.tpc_mirror_ratio if geometry.field_configuration == "TPC" else 1.0
    # Large aspect ratio and M~1 imply weak mirror trapping in this pre-CFS
    # surrogate.  This is not the neoclassical banana trapped fraction.
    trapping = min(1.0, math.sqrt(max(2.0 / aspect, 0.0)) * max(0.0, 1.0 - 1.0 / mirror))
    pressure_mPa = config.gas.pressure_Pa * 1.0e3
    pressure_penalty = 1.0 / (1.0 + pressure_mPa / 5.0)
    resonance_factor = (0.55 if fundamental else 0.0) + (0.30 if second else 0.0)
    trapping_factor = 0.15 * min(1.0, trapping / 0.25)
    score = pressure_penalty * (resonance_factor + trapping_factor)
    hypotheses: list[str] = []
    if not second:
        hypotheses.append("二次谐波层不在代表性等离子体径向范围内，高能尾形成可能受限。")
    else:
        hypotheses.append("代表性磁场下二次谐波层在径向范围内，因此“缺少二次层”不能单独解释启动失败；仍需检查偏振、光路、吸收和快电子约束。")
    if trapping < 0.05:
        hypotheses.append("大纵横比/弱镜像的开场线几何给出较低捕获代理值，净电流不对称性可能不足。")
    if pressure_mPa > 5.0:
        hypotheses.append("较高压强增加电离机会但也增强碰撞热化与开场线损失，可能出现击穿而电流建立失败。")
    if fundamental:
        hypotheses.append("存在基频几何共振层只说明可达性，不保证有足够吸收、轨道不对称或无感电流。")
    if not hypotheses:
        hypotheses.append("几何筛选未发现单一阻断项，需继续计算吸收、Dql、壁损失和分布反馈。")
    return StartupResonanceAssessment(
        device or config.model, aspect, pressure_mPa, pressure_penalty, mirror,
        trapping, fundamental, second, score, tuple(hypotheses), tuple(layers),
    )
