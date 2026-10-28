"""跨装置比较工作流。

该模块的目的不是给出“球形/常规”标签的先验结论，而是把差异投影到可
测量、可扫描的无量纲量和机制分项上。每个装置使用同一套方程和相同的
数据 provider，才能讨论因果差异。
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import SimulationConfig
from .constants import E_CHARGE_C, M_E_KG, PI
from .physics.global_model import GlobalStartupModel
from .solver import RunResult, integrate


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_with_override(base: SimulationConfig, override: Mapping[str, Any]) -> SimulationConfig:
    """对一个基准配置做局部覆盖并重新严格校验。"""

    raw = _deep_merge(base.model_dump(mode="python"), override)
    # 基准配置经过校验后 plasma_length_m 已被补全；若比较案例改变 R0，
    # 且未显式给出长度，应让 schema 重新按新几何推导，而不是沿用旧值。
    if "geometry" in override and "R0_m" in override.get("geometry", {}):
        if "circuit" not in override or "plasma_length_m" not in override.get("circuit", {}):
            raw.setdefault("circuit", {})["plasma_length_m"] = None
    return SimulationConfig.model_validate(raw)


@dataclass(frozen=True)
class DeviceSummary:
    name: str
    aspect_ratio: float
    inverse_aspect_ratio: float
    omega_ce_over_omega_rf: float
    n_critical_m3: float
    ne_over_ncritical: float
    ne_over_ncritical_active: float
    rho_e_over_a: float
    omega_ce_over_nu: float
    lambda_D_over_a: float
    mean_free_path_over_a: float
    ionization_fraction: float
    Bp_edge_over_Bt: float
    Da_ion: float
    Da_ion_active: float
    EoverN_Td: float
    open_loss_time_s: float
    rf_accessibility: float
    closed_fraction: float
    cfs_margin_T: float
    final_stage: str
    breakdown_time_s: float | None
    preionization_time_s: float | None
    cfs_time_s: float | None
    peak_Ip_A: float
    peak_I_ohmic_A: float
    peak_I_rf_A: float
    peak_I_orbit_A: float
    # 位形与机制分项，便于跨 TPC/VFC/圆截面案例审计。
    field_configuration: str
    mirror_ratio: float
    trapped_fraction: float
    peak_I_tail_A: float
    peak_I_vertical_charge_A: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _event_time(result: RunResult, event_type: str) -> float | None:
    for event in result.events:
        if event.type == event_type:
            return event.time_s
    return None


def summarize_device(name: str, config: SimulationConfig, result: RunResult | None = None) -> DeviceSummary:
    model = GlobalStartupModel(config)
    if result is None:
        result = integrate(config)
    final = result.final_state
    evaluation = model.evaluate(float(result.times_s[-1]), final)
    g = config.geometry
    rf = config.rf
    a = g.minor_radius_m
    aspect = g.R0_m / a
    omega_ce = E_CHARGE_C * g.B_t_T / M_E_KG
    reference_frequency_Hz = rf.sources[0].frequency_Hz if rf.sources else rf.frequency_Hz
    omega_rf = 2.0 * PI * reference_frequency_Hz
    active_index = max(
        range(len(result.diagnostics)),
        key=lambda i: float(result.diagnostics[i].get("rf_power_absorbed_W", 0.0)),
    )
    active_row = result.diagnostics[active_index]
    active_state = result.states[active_index]
    rho_e = math.sqrt(2.0 * max(active_state.Te_eV, 1.0e-3) * E_CHARGE_C / M_E_KG) / max(omega_ce, 1.0)
    bp_edge = float(evaluation.topology.Bp_edge_T)
    da_values = [float(row.get("nu_ion_s", 0.0)) / max(float(row.get("nu_loss_s", 1.0)), 1.0e-30) for row in result.diagnostics]
    da_active = float(active_row.get("nu_ion_s", 0.0)) / max(float(active_row.get("nu_loss_s", 1.0)), 1.0e-30)
    return DeviceSummary(
        name=name,
        aspect_ratio=aspect,
        inverse_aspect_ratio=1.0 / aspect,
        omega_ce_over_omega_rf=omega_ce / max(omega_rf, 1.0e-30),
        n_critical_m3=model.rf_model.critical_density_m3,
        ne_over_ncritical=final.ne_m3 / max(model.rf_model.critical_density_m3, 1.0),
        ne_over_ncritical_active=active_state.ne_m3 / max(model.rf_model.critical_density_m3, 1.0),
        rho_e_over_a=rho_e / a,
        omega_ce_over_nu=float(active_row.get("omega_ce_over_nu", 0.0)),
        lambda_D_over_a=float(active_row.get("lambda_D_over_a", 0.0)),
        mean_free_path_over_a=float(active_row.get("mean_free_path_over_a", 0.0)),
        ionization_fraction=float(active_row.get("ionization_fraction", 0.0)),
        Bp_edge_over_Bt=bp_edge / max(g.B_t_T, 1.0e-30),
        Da_ion=max(da_values),
        Da_ion_active=da_active,
        EoverN_Td=float(active_row.get("EoverN_Td", 0.0)),
        open_loss_time_s=1.0 / max(float(active_row.get("nu_loss_s", 1.0)), 1.0e-30),
        rf_accessibility=float(active_row.get("rf_accessibility", 0.0)),
        closed_fraction=evaluation.values["closed_fraction"],
        cfs_margin_T=evaluation.topology.Bp_edge_T - evaluation.topology.threshold_T,
        final_stage=str(result.diagnostics[-1].get("stage", "unknown")),
        breakdown_time_s=_event_time(result, "breakdown"),
        preionization_time_s=_event_time(result, "preionization"),
        cfs_time_s=_event_time(result, "cfs"),
        peak_Ip_A=max(abs(float(row.get("Ip_A", 0.0))) for row in result.diagnostics),
        peak_I_ohmic_A=max(abs(float(row.get("I_ohmic_A", 0.0))) for row in result.diagnostics),
        peak_I_rf_A=max(abs(float(row.get("I_rf_A", 0.0))) for row in result.diagnostics),
        peak_I_orbit_A=max(abs(float(row.get("I_orbit_A", 0.0))) for row in result.diagnostics),
        field_configuration=str(active_row.get("field_configuration", g.field_configuration)),
        mirror_ratio=float(active_row.get("mirror_ratio", 1.0)),
        trapped_fraction=float(active_row.get("trapped_fraction", 0.0)),
        peak_I_tail_A=max(abs(float(row.get("I_tail_A", row.get("I_orbit_A", 0.0)))) for row in result.diagnostics),
        peak_I_vertical_charge_A=max(
            abs(float(row.get("I_vertical_charge_A", row.get("I_vfc_A", 0.0)))) for row in result.diagnostics
        ),
    )


def run_device_matrix(cases: Mapping[str, SimulationConfig]) -> list[DeviceSummary]:
    """对多个装置用相同模型运行并返回可比较摘要。"""

    summaries = []
    for name, config in cases.items():
        summaries.append(summarize_device(name, config))
    return summaries


def load_device_matrix(path: str | Path) -> dict[str, SimulationConfig]:
    """读取比较文件。

    格式：`{"base_config":"...", "devices":{"ST":{...},"conventional":{...}}}`。
    每个 device 的值是相对于基准配置的局部覆盖。
    """

    matrix_path = Path(path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    base_path = Path(payload["base_config"])
    if not base_path.is_absolute():
        base_path = matrix_path.parent / base_path
    # 延迟导入避免 comparison 与 config 形成循环。
    from .config import load_config

    base = load_config(base_path)
    devices = payload.get("devices", {})
    if not isinstance(devices, dict) or not devices:
        raise ValueError("比较文件的 devices 必须是非空对象")
    return {name: config_with_override(base, override) for name, override in devices.items()}
