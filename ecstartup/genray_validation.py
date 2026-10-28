"""Reproducible native-GENRAY versus internal reduced-model validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .genray_io import plot_genray_summary, read_genray_netcdf
from .kinetic_workflow import run_kinetic
from .physics.fokker_planck import VelocityGrid
from .physics.genray_like import build_internal_genray_like_reference


def _agreement_band(ratio: float) -> str:
    if 0.5 <= ratio <= 2.0:
        return "same_scale_within_factor_2"
    if 0.1 <= ratio <= 10.0:
        return "same_order_of_magnitude_only"
    return "failed_order_of_magnitude"


def validate_genray_case(
    config: SimulationConfig,
    netcdf_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run one public/native GENRAY comparison and persist all audit inputs."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    native = read_genray_netcdf(netcdf_path)
    native_payload = native.summary()
    (output / "genray_native_summary.json").write_text(
        json.dumps(native_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_genray_summary(native, output / "genray_native_validation.png")

    grid = VelocityGrid.uniform(
        config.kinetic.n_v_parallel,
        config.kinetic.n_v_perp,
        config.kinetic.v_parallel_max_m_s,
        config.kinetic.v_perp_max_m_s,
    )
    internal = build_internal_genray_like_reference(config, grid)
    np.savez_compressed(
        output / "internal_dql.npz",
        D_parallel_m2_s3=internal.diffusion.D_parallel_m2_s3,
        D_perp_m2_s3=internal.diffusion.D_perp_m2_s3,
        v_parallel_m_s=grid.v_parallel_m_s,
        v_perp_m_s=grid.v_perp_m_s,
        reference_absorbed_power_W=np.asarray(internal.reference_absorbed_power_W),
        generator=np.asarray("linEC_reduced_genray_like"),
    )
    internal_payload = {
        "reference_absorbed_power_W": internal.reference_absorbed_power_W,
        "diffusion_source": internal.diffusion.source,
        "diagnostics": [item.to_dict() for item in internal.diagnostics],
    }
    (output / "internal_dql_summary.json").write_text(
        json.dumps(internal_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    kinetic = run_kinetic(config)
    kinetic_payload = kinetic.summary()
    (output / "fp_summary.json").write_text(
        json.dumps(kinetic_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    kinetic.save_distribution(output / "fp_distribution.npz")

    reference_parallel = float(native.parallel_current_A)
    fp_current = float(kinetic.final_moments.current_parallel_A)
    adjoint_current = float(
        sum(item.adjoint_estimated_parallel_current_A for item in internal.diagnostics)
    )
    fp_ratio = abs(fp_current) / max(abs(reference_parallel), 1.0e-300)
    adjoint_ratio = abs(adjoint_current) / max(abs(reference_parallel), 1.0e-300)
    commit_path = Path(netcdf_path).with_name("upstream_commit.txt")
    payload: dict[str, Any] = {
        "reference_case": Path(netcdf_path).name,
        "reference_toroidal_current_A": native.toroidal_current_A,
        "reference_parallel_current_A": reference_parallel,
        "reference_absorbed_power_W": native.absorbed_power_W,
        "native_power_balance_relative": native.power_balance_relative,
        "linec_fp_parallel_current_A": fp_current,
        "linec_fp_to_reference_abs_ratio": fp_ratio,
        "linec_fp_validation_status": _agreement_band(fp_ratio),
        "linec_adjoint_scale_parallel_current_A": adjoint_current,
        "linec_adjoint_to_reference_abs_ratio": adjoint_ratio,
        "linec_adjoint_scale_status": _agreement_band(adjoint_ratio),
        "upstream_commit": (
            commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else None
        ),
        "interpretation": (
            "Native NetCDF reproduction checks the external GENRAY result exactly. "
            "The reduced 0D-2V F-P current is an independent transient prediction and "
            "this historical ITER regression deliberately retains its original velocity-grid "
            "settings. linEC now provides optional relativistic momentum coordinates, a "
            "linearized momentum-conserving field-particle correction and a pre-CFS mirror "
            "bounce surrogate, but these are not full Rosenbluth/orbit/adjoint physics and "
            "do not make ITER a startup validation. The normalized-efficiency current is an independent scale "
            "diagnostic and is not used to tune D_ql."
        ),
    }
    (output / "validation_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_current_comparison(payload, output / "current_validation.png")
    return payload


def _plot_current_comparison(payload: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["GENRAY native", "linEC adjoint scale", "linEC 0D-2V F-P"]
    values = np.abs(
        [
            payload["reference_parallel_current_A"],
            payload["linec_adjoint_scale_parallel_current_A"],
            payload["linec_fp_parallel_current_A"],
        ]
    ) / 1.0e3
    colors = ["#15616d", "#2a9d8f", "#bc4749"]
    figure, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("absolute parallel current [kA]")
    ax.set_title(
        "GENRAY ITER regression: current-scale validation\n"
        "Adjoint scale is diagnostic; 0D-2V F-P remains a failed quality gate",
        fontsize=12,
    )
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.08, f"{value:.2f}", ha="center")
    figure.savefig(path, dpi=180)
    plt.close(figure)
