"""机制消融实验：区分预电离、环向场、RF 动量和轨道不对称。"""

from __future__ import annotations

from typing import Any

from .comparison import config_with_override, summarize_device
from .config import SimulationConfig
from .solver import integrate


DEFAULT_ABLATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_rf": {"rf": {"enabled": False}},
    "no_loop_voltage": {"circuit": {"loop_voltage_V": 0.0}},
    "no_orbit_asymmetry": {"rf": {"orbit_asymmetry": 0.0}},
    "no_direct_rf_current": {"rf": {"current_drive_A_per_W": 0.0}},
    "no_hot_tail": {"rf": {"hot_fraction": 0.0}},
    "no_tpc_mirror_confinement": {
        "geometry": {"tpc_trapped_fraction": 0.0, "tpc_mirror_ratio": 1.0}
    },
    "no_vertical_charge_separation": {"geometry": {"vfc_current_fraction": 0.0}},
}


def run_ablation(
    base_config: SimulationConfig,
    cases: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """运行一组机制消融并返回可序列化摘要。"""

    selected = cases or DEFAULT_ABLATIONS
    output: list[dict[str, Any]] = []
    for name, override in selected.items():
        config = config_with_override(base_config, override)
        result = integrate(config)
        summary = summarize_device(name, config, result).as_dict()
        summary["events"] = [event.type for event in result.events]
        output.append(summary)
    return output
