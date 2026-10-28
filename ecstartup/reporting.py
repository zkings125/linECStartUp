"""Scientific figures and a unified multi-page startup PDF report."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .config import SimulationConfig
from .current_profile import (
    CurrentProfileEvolution,
    MechanismCurrentEvolution,
    reconstruct_current_profiles,
    reconstruct_mechanism_currents,
)
from .solver import RunResult


_STAGE_COLORS = {
    "vacuum": "#e7e5e4",
    "breakdown": "#fde68a",
    "preionization": "#bae6fd",
    "cfs_forming": "#bbf7d0",
    "cfs_established": "#86efac",
    "failed": "#fecaca",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _array(result: RunResult, key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray([float(row.get(key, default)) for row in result.diagnostics], dtype=float)


def _time_ms(result: RunResult) -> np.ndarray:
    return np.asarray(result.times_s, dtype=float) * 1.0e3


def _shade_stages(axis, result: RunResult) -> None:
    times = _time_ms(result)
    stages = [str(row.get("stage", "unknown")) for row in result.diagnostics]
    if not stages:
        return
    start = 0
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[start]:
            left = times[start]
            right = times[index] if index < len(times) else times[-1]
            axis.axvspan(
                left,
                right,
                color=_STAGE_COLORS.get(stages[start], "#f5f5f4"),
                alpha=0.20,
                linewidth=0.0,
                zorder=0,
            )
            start = index


def _finish_time_axis(axis, result: RunResult, ylabel: str) -> None:
    _shade_stages(axis, result)
    axis.set_xlabel("time [ms]")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d6d3d1", linewidth=0.6, alpha=0.55)
    axis.set_xlim(float(_time_ms(result)[0]), float(_time_ms(result)[-1]))


def _snapshot_indices(count: int, snapshots: int) -> np.ndarray:
    return np.unique(np.linspace(0, max(count - 1, 0), min(snapshots, count), dtype=int))


def _draw_profile_page(
    fig,
    axes,
    profiles: CurrentProfileEvolution,
    result: RunResult,
    snapshot_count: int,
) -> None:
    plt = _pyplot()
    indices = _snapshot_indices(len(profiles.times_s), snapshot_count)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(indices)))
    for color, index in zip(colors, indices):
        axes[0].plot(
            profiles.rho_geo,
            profiles.j_total_A_m2[index] / 1.0e3,
            color=color,
            linewidth=1.6,
            label=f"{profiles.times_s[index] * 1.0e3:.2f} ms",
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"geometric $\rho=r/a$")
    axes[0].set_ylabel(r"$j_{total}$ [kA m$^{-2}$]")
    axes[0].set_title("Reconstructed radial current profiles")
    axes[0].grid(True, color="#d6d3d1", linewidth=0.6, alpha=0.55)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    image = axes[1].contourf(
        profiles.times_s * 1.0e3,
        profiles.rho_geo,
        (profiles.j_total_A_m2 / 1.0e3).T,
        levels=31,
        cmap="RdBu_r",
        extend="both",
    )
    axes[1].contour(
        profiles.times_s * 1.0e3,
        profiles.rho_geo,
        (profiles.j_total_A_m2 / 1.0e3).T,
        levels=9,
        colors="black",
        linewidths=0.35,
        alpha=0.45,
    )
    axes[1].set_xlabel("time [ms]")
    axes[1].set_ylabel(r"geometric $\rho=r/a$")
    axes[1].set_title(
        "$j_{total}(\\rho_{geo},t)$ evolution\n"
        "normalized 0D reconstruction; pre-CFS $\\rho_{geo}$ is not magnetic flux",
        fontsize=11,
    )
    fig.colorbar(image, ax=axes[1], label=r"$j_{total}$ [kA m$^{-2}$]")


def write_current_profile_figure(
    profiles: CurrentProfileEvolution,
    result: RunResult,
    config: SimulationConfig,
    path: str | Path,
) -> Path:
    plt = _pyplot()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    _draw_profile_page(fig, axes, profiles, result, config.outputs.profile_snapshot_count)
    fig.suptitle("Startup current-profile evolution", fontsize=14)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def _report_overview_page(pdf, result: RunResult, config: SimulationConfig) -> None:
    plt = _pyplot()
    time_ms = _time_ms(result)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    axes[0, 0].plot(time_ms, _array(result, "Ip_A") / 1.0e3, color="#b42318", linewidth=2.0)
    _finish_time_axis(axes[0, 0], result, "$I_p$ [kA]")
    axes[0, 0].set_title("Plasma-current startup")

    axes[0, 1].semilogy(time_ms, np.maximum(_array(result, "ne_m3"), 1.0), label="$n_e$")
    axes[0, 1].semilogy(time_ms, np.maximum(_array(result, "nn_m3"), 1.0), label="$n_n$")
    _finish_time_axis(axes[0, 1], result, "density [m$^{-3}$]")
    axes[0, 1].set_title("Particle evolution")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(time_ms, _array(result, "Te_eV"), color="#0369a1", linewidth=1.8)
    _finish_time_axis(axes[1, 0], result, "$T_e$ [eV]")
    axes[1, 0].set_title("Electron temperature")

    axes[1, 1].plot(
        time_ms, _array(result, "rf_power_absorbed_W") / 1.0e3, label="absorbed EC", linewidth=1.8
    )
    axes[1, 1].plot(
        time_ms, _array(result, "effective_loop_voltage_V"), label="effective loop voltage", linewidth=1.4
    )
    _finish_time_axis(axes[1, 1], result, "kW or V")
    axes[1, 1].set_title("External drive")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        f"EC startup overview | CS: {config.circuit.cs_mode} | model: {config.model}",
        fontsize=14,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _report_mechanism_page(
    pdf,
    result: RunResult,
    currents: MechanismCurrentEvolution,
) -> None:
    plt = _pyplot()
    time_ms = _time_ms(result)
    drive_model = str(result.diagnostics[0].get("current_drive_model", "empirical_rf_hot_tail"))
    rf_label = (
        "F-P moment response"
        if drive_model == "fp_moment_response_table"
        else "direct RF response"
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    axes[0].plot(time_ms, currents.Ip_A / 1.0e3, color="black", linewidth=2.3, label="actual $I_p$")
    for values, label, color in (
        (currents.I_cs_A, "CS/inductive response", "#b45309"),
        (currents.I_rf_A, rf_label, "#2563eb"),
        (currents.I_tail_A, "hot-tail response", "#7c3aed"),
        (currents.I_vertical_A, "vertical-charge response", "#059669"),
        (currents.I_closure_residual_A, "sampled closure residual", "#78716c"),
    ):
        axes[0].plot(time_ms, values / 1.0e3, linewidth=1.5, label=label, color=color)
    _finish_time_axis(axes[0], result, "causal current contribution [kA]")
    axes[0].set_title("Mechanism-resolved L-R current response")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)

    for key, label, color in (
        ("I_ohmic_A", "instantaneous ohmic target", "#b45309"),
        ("I_rf_A", rf_label.replace("response", "target"), "#2563eb"),
        ("I_tail_A", "hot-tail target", "#7c3aed"),
        ("I_vertical_charge_A", "vertical-charge target", "#059669"),
    ):
        axes[1].plot(time_ms, _array(result, key) / 1.0e3, linewidth=1.5, label=label, color=color)
    _finish_time_axis(axes[1], result, "instantaneous target current [kA]")
    axes[1].set_title("Simultaneous source terms before circuit response")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Current-drive mechanism accounting", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _report_conditions_page(pdf, result: RunResult) -> None:
    plt = _pyplot()
    time_ms = _time_ms(result)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    axes[0, 0].plot(time_ms, _array(result, "neutral_pressure_Pa"), label="neutral pressure")
    axes[0, 0].plot(time_ms, _array(result, "target_pressure_Pa"), label="gas target", linestyle="--")
    _finish_time_axis(axes[0, 0], result, "pressure [Pa]")
    axes[0, 0].set_title("Gas condition")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(time_ms, _array(result, "Z_eff_used", 1.0), color="#7c3aed")
    _finish_time_axis(axes[0, 1], result, "$Z_{eff}$")
    axes[0, 1].set_title("Effective charge used by conductivity")

    for key, label in (
        ("P_ohm_W_m3", "ohmic"),
        ("P_rf_W_m3", "absorbed RF"),
        ("P_ion_W_m3", "ionization loss"),
        ("P_rad_W_m3", "radiation loss"),
        ("P_wall_W_m3", "wall loss"),
    ):
        axes[1, 0].plot(time_ms, _array(result, key) / 1.0e3, label=label)
    _finish_time_axis(axes[1, 0], result, "power density [kW m$^{-3}$]")
    axes[1, 0].set_title("Power channels")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1, 1].plot(time_ms, _array(result, "net_growth_s"), label="net electron growth")
    axes[1, 1].plot(time_ms, _array(result, "nu_ion_s"), label="ionization")
    axes[1, 1].plot(time_ms, _array(result, "nu_loss_s"), label="wall loss")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    _finish_time_axis(axes[1, 1], result, "rate [s$^{-1}$]")
    axes[1, 1].set_title("Breakdown competition")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Operating conditions and balances", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _report_threshold_scan_pages(pdf, payload: dict) -> None:
    plt = _pyplot()
    cases = payload["cases"]
    thresholds = payload["thresholds"]
    cs_modes = list(dict.fromkeys(item["cs_mode"] for item in cases))
    figure, axes = plt.subplots(
        len(cs_modes),
        1,
        figsize=(11.69, 8.27),
        squeeze=False,
        constrained_layout=True,
    )
    criterion = float(payload["criterion"]["minimum_peak_ec_increment_A"])
    for axis, cs_mode in zip(axes[:, 0], cs_modes):
        selected = [item for item in cases if item["cs_mode"] == cs_mode]
        combinations = sorted(
            {(item["pressure_Pa"], item["effective_charge"]) for item in selected}
        )
        for pressure, zeff in combinations:
            rows = sorted(
                (
                    item
                    for item in selected
                    if item["pressure_Pa"] == pressure
                    and item["effective_charge"] == zeff
                ),
                key=lambda item: item["ec_power_W"],
            )
            axis.plot(
                [row["ec_power_W"] / 1.0e3 for row in rows],
                [row["peak_signed_ec_increment_A"] / 1.0e3 for row in rows],
                marker="o",
                linewidth=1.4,
                label=f"p={pressure * 1.0e3:g} mPa, Zeff={zeff:g}",
            )
        axis.axhline(criterion / 1.0e3, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"{cs_mode}: EC incremental current relative to zero-power baseline")
        axis.set_xlabel("injected EC power [kW]")
        axis.set_ylabel("peak signed EC increment [kA]")
        axis.grid(True, color="#d6d3d1", linewidth=0.6, alpha=0.55)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(
        f"EC threshold scan | source: {payload['source_name']} | "
        f"kinetic: {payload['criterion'].get('kinetic_coupling_mode', 'inherit')}",
        fontsize=14,
    )
    pdf.savefig(figure)
    plt.close(figure)

    pressures = sorted({item["pressure_Pa"] for item in thresholds})
    zeffs = sorted({item["effective_charge"] for item in thresholds})
    figure, axes = plt.subplots(
        1,
        len(cs_modes),
        figsize=(11.69, 8.27),
        squeeze=False,
        constrained_layout=True,
    )
    matrices = []
    lower_matrices = []
    for cs_mode in cs_modes:
        matrix = np.full((len(pressures), len(zeffs)), np.nan)
        lower_matrix = np.full((len(pressures), len(zeffs)), np.nan)
        for item in thresholds:
            if item["cs_mode"] != cs_mode:
                continue
            row = pressures.index(item["pressure_Pa"])
            column = zeffs.index(item["effective_charge"])
            value = item["minimum_successful_ec_power_W"]
            if value is not None:
                matrix[row, column] = float(value) / 1.0e3
            lower = item.get("last_failed_ec_power_W")
            if lower is not None:
                lower_matrix[row, column] = float(lower) / 1.0e3
        matrices.append(matrix)
        lower_matrices.append(lower_matrix)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    color_min = float(np.min(finite)) if finite.size else 0.0
    color_max = float(np.max(finite)) if finite.size else 1.0
    if color_max <= color_min:
        color_max = color_min + 1.0
    max_scanned_power = max(item["ec_power_W"] for item in cases) / 1.0e3
    last_image = None
    for axis, cs_mode, matrix, lower_matrix in zip(
        axes[0], cs_modes, matrices, lower_matrices
    ):
        last_image = axis.pcolormesh(
            np.arange(len(zeffs) + 1) - 0.5,
            np.arange(len(pressures) + 1) - 0.5,
            np.ma.masked_invalid(matrix),
            shading="flat",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_xlim(-0.5, len(zeffs) - 0.5)
        axis.set_ylim(-0.5, len(pressures) - 0.5)
        axis.set_xticks(range(len(zeffs)))
        axis.set_xticklabels([f"{value:g}" for value in zeffs])
        axis.set_yticks(range(len(pressures)))
        axis.set_yticklabels([f"{value * 1.0e3:g}" for value in pressures])
        axis.set_xlabel("effective charge $Z_{eff}$")
        axis.set_ylabel("pressure [mPa]")
        axis.set_title(cs_mode)
        for row in range(len(pressures)):
            for column in range(len(zeffs)):
                value = matrix[row, column]
                lower = lower_matrix[row, column]
                if np.isfinite(value) and np.isfinite(lower) and lower < value:
                    label = f"({lower:.1f}, {value:.1f}]"
                elif np.isfinite(value):
                    label = f"<= {value:.1f}"
                else:
                    label = f">{max_scanned_power:.0f}"
                axis.text(column, row, label, ha="center", va="center", color="black", fontsize=9)
    if last_image is not None:
        figure.colorbar(
            last_image,
            ax=list(axes[0]),
            label="lowest successful sampled EC power [kW]",
        )
    figure.suptitle(
        "EC-current numerical failure/success brackets\n"
        "Successful upper bounds are model-dependent, not proven physical bifurcations\n"
        f"kinetic scan mode: {payload['criterion'].get('kinetic_coupling_mode', 'inherit')}",
        fontsize=14,
    )
    pdf.savefig(figure)
    plt.close(figure)


def write_startup_pdf_report(
    result: RunResult,
    config: SimulationConfig,
    path: str | Path,
    profiles: CurrentProfileEvolution | None = None,
    threshold_scan_payload: dict | None = None,
) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles = profiles or reconstruct_current_profiles(result, config)
    currents = reconstruct_mechanism_currents(result, config)
    with PdfPages(output) as pdf:
        _report_overview_page(pdf, result, config)
        _report_mechanism_page(pdf, result, currents)
        plt = _pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
        _draw_profile_page(fig, axes, profiles, result, config.outputs.profile_snapshot_count)
        fig.suptitle("Geometric-rho current-profile reconstruction", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
        _report_conditions_page(pdf, result)
        if threshold_scan_payload is not None:
            _report_threshold_scan_pages(pdf, threshold_scan_payload)
        metadata = pdf.infodict()
        metadata["Title"] = "EC startup simulation report"
        metadata["Subject"] = "Breakdown, current-drive mechanisms, and reconstructed profiles"
        metadata["Keywords"] = "EC startup, Fokker-Planck, current drive, pre-CFS"
    return output


def write_threshold_scan_pdf(payload: dict, path: str | Path) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _report_threshold_scan_pages(pdf, payload)
        metadata = pdf.infodict()
        metadata["Title"] = "EC startup threshold scan"
        metadata["Subject"] = "Pressure, effective-charge, EC-power, and CS-mode scan"
    return output


def write_report_artifacts(
    result: RunResult, config: SimulationConfig, output_directory: str | Path
) -> dict[str, str]:
    out = Path(output_directory)
    if not (config.outputs.write_current_profile or config.outputs.write_pdf_report):
        return {}
    profiles = reconstruct_current_profiles(result, config)
    artifacts: dict[str, str] = {}
    threshold_payload = None
    if config.threshold_scan.enabled:
        from .threshold_scan import run_threshold_scan

        threshold_payload = run_threshold_scan(config)
        threshold_path = out / "threshold_scan.json"
        threshold_path.write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts["threshold_scan_json"] = threshold_path.name
    if config.outputs.write_current_profile:
        npz_path = out / "current_profile_evolution.npz"
        png_path = out / "current_profile_evolution.png"
        profiles.save_npz(npz_path)
        write_current_profile_figure(profiles, result, config, png_path)
        artifacts["current_profile_npz"] = npz_path.name
        artifacts["current_profile_png"] = png_path.name
    if config.outputs.write_pdf_report:
        pdf_path = out / config.outputs.pdf_report_filename
        write_startup_pdf_report(
            result,
            config,
            pdf_path,
            profiles=profiles,
            threshold_scan_payload=threshold_payload,
        )
        artifacts["pdf_report"] = pdf_path.name
    return artifacts
