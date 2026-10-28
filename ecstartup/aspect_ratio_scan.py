"""Major/minor-radius scan for EC-startup mechanism studies.

The scan varies ``R0_m`` and ``minor_radius_m`` independently.  It is a
geometry sensitivity study, not a substitute for validating each physical
device with its own field, wall, gas and RF data.
"""

from __future__ import annotations

import itertools
from typing import Iterable

from .comparison import config_with_override, summarize_device
from .config import SimulationConfig
from .constants import E_CHARGE_C, MU0_H_M
from .device_prediction_quality import assess_device_prediction_quality
from .physics.startup_resonance import assess_startup_resonance
from .solver import integrate


def _geometry_pairs(config: SimulationConfig) -> Iterable[tuple[float, float]]:
    scan = config.aspect_ratio_scan
    if scan.combination == "paired":
        return zip(scan.R0_values_m, scan.minor_radius_values_m)
    return itertools.product(scan.R0_values_m, scan.minor_radius_values_m)


def _scaled_sources(base: SimulationConfig, R0_m: float, minor_radius_m: float) -> list[dict]:
    sources = [item.model_dump(mode="python") for item in base.rf.sources]
    if not base.aspect_ratio_scan.scale_rf_launch_with_geometry:
        return sources
    radial_scale = minor_radius_m / float(base.geometry.minor_radius_m)
    for source in sources:
        launch = source["launch"]
        if launch.get("R_m") is not None:
            launch["R_m"] = R0_m + (float(launch["R_m"]) - float(base.geometry.R0_m)) * radial_scale
        launch["Z_m"] = float(launch.get("Z_m", 0.0)) * radial_scale
    return sources


def _peak_thermal_beta(config: SimulationConfig, result) -> float:
    magnetic_pressure_denominator = max(float(config.geometry.B_t_T) ** 2, 1.0e-30)
    values = []
    for row in result.diagnostics:
        electron_pressure = float(row.get("ne_m3", 0.0)) * float(row.get("Te_eV", 0.0)) * E_CHARGE_C
        ion_pressure = float(row.get("ni_m3", 0.0)) * float(row.get("Ti_eV", 0.0)) * E_CHARGE_C
        values.append(2.0 * MU0_H_M * (electron_pressure + ion_pressure) / magnetic_pressure_denominator)
    return max(values, default=0.0)


def run_aspect_ratio_scan(base: SimulationConfig) -> dict:
    """Run the configured R0/a grid and return auditable trend diagnostics."""

    points = []
    for R0_value, a_value in _geometry_pairs(base):
        R0_m = float(R0_value)
        minor_radius_m = float(a_value)
        margin = 1.25 * minor_radius_m
        override = {
            "geometry": {
                "R0_m": R0_m,
                "minor_radius_m": minor_radius_m,
                "volume_m3": None,
                "wall_area_m2": None,
                "R_bounds_m": [max(1.0e-4, R0_m - margin), R0_m + margin],
                "Z_bounds_m": [-margin, margin],
            },
            "rf": {"sources": _scaled_sources(base, R0_m, minor_radius_m)},
            "aspect_ratio_scan": {"enabled": False},
            "outputs": {
                "write_netcdf": False,
                "write_csv": False,
                "write_current_profile": False,
                "write_pdf_report": False,
            },
            "threshold_scan": {"enabled": False},
        }
        config = config_with_override(base, override)
        result = integrate(config)
        summary = summarize_device(
            f"R0={R0_m:g} m, a={minor_radius_m:g} m", config, result
        ).as_dict()
        resonance = assess_startup_resonance(config, device=summary["name"]).to_dict()
        accessible_layers = [
            layer for layer in resonance["layers"] if layer["accessible_in_plasma"]
        ]
        points.append(
            {
                "R0_m": R0_m,
                "minor_radius_m": minor_radius_m,
                "aspect_ratio": R0_m / minor_radius_m,
                "inverse_aspect_ratio": minor_radius_m / R0_m,
                "peak_thermal_beta": _peak_thermal_beta(config, result),
                "accessible_resonance_layer_count": len(accessible_layers),
                "accessible_resonance_layers": [
                    {
                        "source_name": layer["source_name"],
                        "frequency_GHz": layer["frequency_GHz"],
                        "harmonic": layer["harmonic"],
                        "resonance_R_m": layer["resonance_R_m"],
                    }
                    for layer in accessible_layers
                ],
                "startup_resonance_diagnostic": resonance,
                "startup_result": summary,
            }
        )

    base_quality = assess_device_prediction_quality(base).to_dict()
    # A generic geometry grid never inherits authorization from one base
    # device: every target machine must supply and pass its own validation set.
    cross_device_authorized = False
    return {
        "scan_type": "independent_major_minor_radius_scan",
        "combination": base.aspect_ratio_scan.combination,
        "scale_rf_launch_with_geometry": base.aspect_ratio_scan.scale_rf_launch_with_geometry,
        "point_count": len(points),
        "points": points,
        "base_device_quality_gate": base_quality,
        "cross_device_quantitative_authorized": cross_device_authorized,
        "evidence_level": "geometry/mechanism sensitivity; each real device requires its own validated input set",
        "interpretation": (
            "Aspect ratio, resonance-layer access, beta proxy and startup/current responses are reported "
            "separately. The scan must not be converted into a universal ST-versus-conventional success law."
        ),
    }
