"""Pressure/Zeff/EC-power threshold scans with a zero-power baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SimulationConfig
from .solver import RunResult, integrate


def _source_name(config: SimulationConfig) -> str:
    requested = config.threshold_scan.source_name
    if requested:
        return requested
    if config.rf.sources:
        enabled = next((source for source in config.rf.sources if source.enabled), None)
        return (enabled or config.rf.sources[0]).name
    return "legacy_single_source"


def _case_config(
    base: SimulationConfig,
    *,
    pressure_Pa: float,
    effective_charge: float,
    power_W: float,
    cs_mode: str,
    source_name: str,
) -> SimulationConfig:
    raw = base.model_dump(mode="python")
    raw["gas"]["pressure_Pa"] = float(pressure_Pa)
    raw["gas"]["pressure_waveform_Pa"] = None
    raw["circuit"]["cs_mode"] = cs_mode
    raw["circuit"]["effective_charge_override"] = float(effective_charge)
    raw["threshold_scan"]["enabled"] = False
    requested_kinetic_mode = base.threshold_scan.kinetic_coupling_mode
    if requested_kinetic_mode != "inherit":
        raw["kinetic"]["coupling_mode"] = requested_kinetic_mode
    raw["outputs"]["write_netcdf"] = False
    raw["outputs"]["write_csv"] = False
    raw["outputs"]["write_current_profile"] = False
    raw["outputs"]["write_pdf_report"] = False
    raw["rf"]["enabled"] = True
    if raw["rf"].get("sources"):
        found = False
        for source in raw["rf"]["sources"]:
            selected = source["name"] == source_name
            if selected:
                source["power_W"] = float(power_W)
                source["enabled"] = True
                found = True
            elif base.threshold_scan.disable_other_sources:
                source["enabled"] = False
        if not found:
            available = ", ".join(source["name"] for source in raw["rf"]["sources"])
            raise ValueError(f"threshold scan source {source_name!r} not found; available: {available}")
    elif source_name == "legacy_single_source":
        raw["rf"]["power_W"] = float(power_W)
    else:
        raise ValueError("source_name was provided but rf.sources is empty")
    return SimulationConfig.model_validate(raw)


def _preionization_reached(result: RunResult, config: SimulationConfig) -> bool:
    if any(event.type == "preionization" for event in result.events):
        return True
    return any(
        float(row.get("ne_m3", 0.0)) >= config.thresholds.preionization_density_m3
        and float(row.get("Te_eV", 0.0)) >= config.thresholds.preionization_Te_eV
        for row in result.diagnostics
    )


def _signed_peak(values: np.ndarray, sign: int) -> float:
    return float(np.max(sign * np.asarray(values, dtype=float)))


def run_threshold_scan(config: SimulationConfig) -> dict[str, Any]:
    scan = config.threshold_scan
    source_name = _source_name(config)
    cases: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []

    for cs_mode in scan.cs_modes:
        for pressure in scan.pressures_Pa:
            for effective_charge in scan.effective_charges:
                baseline_config = _case_config(
                    config,
                    pressure_Pa=float(pressure),
                    effective_charge=float(effective_charge),
                    power_W=0.0,
                    cs_mode=cs_mode,
                    source_name=source_name,
                )
                baseline = integrate(baseline_config)
                baseline_ip = np.asarray(
                    [float(row.get("Ip_A", 0.0)) for row in baseline.diagnostics]
                )
                evaluated: dict[float, dict[str, Any]] = {}

                def evaluate_power(power: float, sample_kind: str) -> dict[str, Any]:
                    key = float(power)
                    if key in evaluated:
                        return evaluated[key]
                    case_config = _case_config(
                        config,
                        pressure_Pa=float(pressure),
                        effective_charge=float(effective_charge),
                        power_W=key,
                        cs_mode=cs_mode,
                        source_name=source_name,
                    )
                    result = integrate(case_config)
                    ip = np.asarray([float(row.get("Ip_A", 0.0)) for row in result.diagnostics])
                    if ip.shape != baseline_ip.shape or not np.allclose(
                        result.times_s, baseline.times_s, rtol=0.0, atol=1.0e-15
                    ):
                        baseline_on_case_grid = np.interp(result.times_s, baseline.times_s, baseline_ip)
                    else:
                        baseline_on_case_grid = baseline_ip
                    peak_total = _signed_peak(ip, scan.current_direction_sign)
                    peak_increment = _signed_peak(
                        ip - baseline_on_case_grid, scan.current_direction_sign
                    )
                    preionization = _preionization_reached(result, case_config)
                    success = (
                        result.success
                        and peak_increment >= scan.success_increment_A
                        and peak_total >= scan.success_total_current_A
                        and (preionization or not scan.require_preionization)
                    )
                    row = {
                            "cs_mode": cs_mode,
                            "pressure_Pa": float(pressure),
                            "effective_charge": float(effective_charge),
                            "ec_power_W": key,
                            "peak_signed_total_current_A": peak_total,
                            "peak_signed_ec_increment_A": peak_increment,
                            "preionization_reached": preionization,
                            "solver_success": bool(result.success),
                            "threshold_success": bool(success),
                            "final_stage": str(result.diagnostics[-1].get("stage", "unknown")),
                            "sample_kind": sample_kind,
                        }
                    evaluated[key] = row
                    return row

                coarse_powers = sorted(set(float(item) for item in scan.ec_powers_W))
                coarse_rows = [evaluate_power(power, "coarse") for power in coarse_powers]
                first_success_index = next(
                    (index for index, row in enumerate(coarse_rows) if row["threshold_success"]),
                    None,
                )
                lower_bound = None
                upper_bound = None
                iterations = 0
                if first_success_index is not None:
                    upper_bound = coarse_powers[first_success_index]
                    lower_bound = (
                        coarse_powers[first_success_index - 1]
                        if first_success_index > 0
                        else 0.0
                    )
                    if lower_bound == 0.0:
                        # The baseline has zero EC increment by construction.
                        lower_success = False
                    else:
                        lower_success = bool(evaluated[lower_bound]["threshold_success"])
                    if scan.refine_enabled and not lower_success:
                        while (
                            upper_bound - lower_bound > scan.refine_power_tolerance_W
                            and iterations < scan.refine_max_iterations
                        ):
                            midpoint = 0.5 * (lower_bound + upper_bound)
                            row = evaluate_power(midpoint, "refined")
                            if row["threshold_success"]:
                                upper_bound = midpoint
                            else:
                                lower_bound = midpoint
                            iterations += 1
                cases.extend(sorted(evaluated.values(), key=lambda item: item["ec_power_W"]))
                thresholds.append(
                    {
                        "cs_mode": cs_mode,
                        "pressure_Pa": float(pressure),
                        "effective_charge": float(effective_charge),
                        "minimum_successful_ec_power_W": upper_bound,
                        "last_failed_ec_power_W": lower_bound,
                        "refinement_iterations": iterations,
                        "refinement_tolerance_W": scan.refine_power_tolerance_W,
                    }
                )
    return {
        "source_name": source_name,
        "scan_switch_enabled": scan.enabled,
        "criterion": {
            "current_direction_sign": scan.current_direction_sign,
            "minimum_peak_ec_increment_A": scan.success_increment_A,
            "minimum_peak_total_current_A": scan.success_total_current_A,
            "require_preionization": scan.require_preionization,
            "adaptive_refinement": scan.refine_enabled,
            "power_tolerance_W": scan.refine_power_tolerance_W,
            "baseline": "same pressure, effective charge, and CS mode with EC power set to zero",
            "kinetic_coupling_mode": scan.kinetic_coupling_mode,
        },
        "cases": cases,
        "thresholds": thresholds,
        "interpretation_warning": (
            "A refined threshold is the successful upper edge of a numerical failure/success "
            "bracket, not a proven physical bifurcation. It inherits all model/provider uncertainty."
        ),
    }
