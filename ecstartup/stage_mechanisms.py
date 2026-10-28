"""Diagnostic startup windows and causal current-drive mechanism summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SimulationConfig
from .current_profile import MechanismCurrentEvolution, reconstruct_mechanism_currents
from .solver import RunResult


MECHANISM_LABELS = {
    "cs_inductive": "CS inductive",
    "fp_ec_moment": "F-P EC moment",
    "hot_tail": "hot-tail preferential confinement",
    "vertical_charge": "vertical charge separation",
}


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def diagnostic_stage_windows(result: RunResult, config: SimulationConfig) -> list[dict[str, Any]]:
    """Return explicit windows even when state-machine thresholds are not reached."""

    times = np.asarray(result.times_s, dtype=float)
    ne = np.asarray([float(row.get("ne_m3", 0.0)) for row in result.diagnostics])
    te = np.asarray([float(row.get("Te_eV", 0.0)) for row in result.diagnostics])
    ip = np.asarray([float(row.get("Ip_A", 0.0)) for row in result.diagnostics])
    breakdown = _first_index(ne >= config.thresholds.breakdown_density_m3)
    preionization = _first_index(
        (ne >= config.thresholds.preionization_density_m3)
        & (te >= config.thresholds.preionization_Te_eV)
    )
    peak_value = float(np.max(np.abs(ip)))
    peak_candidates = np.flatnonzero(np.isclose(np.abs(ip), peak_value, rtol=1.0e-12, atol=1.0e-12))
    peak = int(peak_candidates[-1]) if peak_candidates.size else len(times) - 1
    end = len(times) - 1

    seed_end = breakdown if breakdown is not None else peak
    windows: list[dict[str, Any]] = [
        {
            "name": "seed_to_breakdown",
            "label": "seed / breakdown-density rise",
            "start_index": 0,
            "end_index": max(0, seed_end),
            "available": True,
            "terminal_criterion_reached": breakdown is not None,
            "terminal_criterion": "breakdown density",
        }
    ]
    if breakdown is not None:
        second_end = preionization if preionization is not None else max(peak, breakdown)
        windows.append(
            {
                "name": "breakdown_to_preionization",
                "label": "breakdown-density to current formation",
                "start_index": breakdown,
                "end_index": second_end,
                "available": True,
                "terminal_criterion_reached": preionization is not None,
                "terminal_criterion": "preionization density and temperature",
            }
        )
    else:
        windows.append(
            {
                "name": "breakdown_to_preionization",
                "label": "breakdown-density to current formation",
                "start_index": None,
                "end_index": None,
                "available": False,
                "terminal_criterion_reached": False,
                "terminal_criterion": "preionization density and temperature",
            }
        )
    if preionization is not None:
        ramp_end = peak if peak >= preionization else end
        windows.append(
            {
                "name": "post_preionization_ramp",
                "label": "post-preionization current ramp",
                "start_index": preionization,
                "end_index": ramp_end,
                "available": True,
                "terminal_criterion_reached": peak >= preionization,
                "terminal_criterion": "peak |Ip|",
            }
        )
    else:
        windows.append(
            {
                "name": "post_preionization_ramp",
                "label": "post-preionization current ramp",
                "start_index": None,
                "end_index": None,
                "available": False,
                "terminal_criterion_reached": False,
                "terminal_criterion": "peak |Ip|",
            }
        )
    windows.append(
        {
            "name": "post_peak_termination",
            "label": "post-peak / termination",
            "start_index": peak if peak < end else None,
            "end_index": end if peak < end else None,
            "available": peak < end,
            "terminal_criterion_reached": False,
            "terminal_criterion": "simulation end; no disruption event model",
        }
    )
    for window in windows:
        if window["available"]:
            start = int(window["start_index"])
            stop = int(window["end_index"])
            window["time_range_s"] = [float(times[start]), float(times[stop])]
        else:
            window["time_range_s"] = None
    return windows


def summarize_stage_mechanisms(
    result: RunResult,
    config: SimulationConfig,
    currents: MechanismCurrentEvolution | None = None,
) -> list[dict[str, Any]]:
    """Summarize physical current contributions; numerical residual is excluded."""

    currents = currents or reconstruct_mechanism_currents(result, config)
    components = {
        "cs_inductive": np.asarray(currents.I_cs_A),
        "fp_ec_moment": np.asarray(currents.I_rf_A),
        "hot_tail": np.asarray(currents.I_tail_A),
        "vertical_charge": np.asarray(currents.I_vertical_A),
    }
    summaries: list[dict[str, Any]] = []
    for window in diagnostic_stage_windows(result, config):
        row = dict(window)
        if not row["available"]:
            row.update(
                {
                    "dominant_mechanism": "not reached",
                    "dominant_label": "not reached",
                    "mechanism_mean_abs_A": {},
                    "mechanism_peak_abs_A": {},
                    "mechanism_fraction": {},
                    "mean_Ip_A": None,
                    "peak_abs_Ip_A": None,
                }
            )
            summaries.append(row)
            continue
        start = int(row["start_index"])
        stop = int(row["end_index"]) + 1
        means = {name: float(np.mean(np.abs(values[start:stop]))) for name, values in components.items()}
        peaks = {name: float(np.max(np.abs(values[start:stop]))) for name, values in components.items()}
        total = sum(means.values())
        fractions = {name: value / total if total > 0.0 else 0.0 for name, value in means.items()}
        dominant = max(means, key=means.get) if total > 0.0 else "none"
        ip = np.asarray(currents.Ip_A[start:stop], dtype=float)
        row.update(
            {
                "dominant_mechanism": dominant,
                "dominant_label": MECHANISM_LABELS.get(dominant, dominant),
                "mechanism_mean_abs_A": means,
                "mechanism_peak_abs_A": peaks,
                "mechanism_fraction": fractions,
                "mean_Ip_A": float(np.mean(ip)),
                "peak_abs_Ip_A": float(np.max(np.abs(ip))),
            }
        )
        summaries.append(row)
    return summaries
