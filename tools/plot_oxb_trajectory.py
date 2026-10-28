"""Plot a pre-CFS O-X-B/EBW schematic from physical R-Z providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np

from ecstartup.config import load_config
from ecstartup.constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI
from ecstartup.physics.ebw import GaussianDensityCloud, analyze_oxb
from ecstartup.physics.ec_raytrace import HamiltonianECRayTracer
from ecstartup.physics.magnetic import make_magnetic_field
from ecstartup.physics.rf import ECRHSource


def _finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def make_figure(
    config_path: Path,
    output_path: Path,
    *,
    source_name: str = "EC28_O",
    t_s: float = -0.05,
    Ip_A: float = 0.0,
    peak_density_m3: float | None = None,
    poloidal_angle_deg: float | None = None,
    toroidal_angle_deg: float | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    sources = ECRHSource(config.rf, config.geometry).sources
    source = next((item for item in sources if item.name == source_name), None)
    if source is None:
        raise ValueError(f"unknown RF source {source_name!r}")
    launch_updates = {}
    if poloidal_angle_deg is not None:
        launch_updates["poloidal_angle_deg"] = poloidal_angle_deg
    if toroidal_angle_deg is not None:
        launch_updates["toroidal_angle_deg"] = toroidal_angle_deg
    if launch_updates:
        source = source.model_copy(update={"launch": source.launch.model_copy(update=launch_updates)})

    peak_density = peak_density_m3 or config.ebw.schematic_peak_density_m3
    center_R = config.ebw.center_R_m or config.geometry.R0_m
    density = GaussianDensityCloud(
        center_R,
        config.ebw.center_Z_m,
        config.ebw.sigma_R_m,
        config.ebw.sigma_Z_m,
        peak_density,
        config.ebw.density_floor_m3,
    )
    magnetic = make_magnetic_field(config.geometry)
    ray_config = config.ray_tracing.model_copy(update={"terminate_on_wkb_failure": False})
    ray = HamiltonianECRayTracer(
        magnetic,
        density,
        config.geometry,
        ray_config,
        ion_mass_kg=config.gas.ion_mass_kg,
    ).trace(source, Ip_A=Ip_A, t_s=t_s)
    result = analyze_oxb(
        source,
        ray,
        magnetic,
        density,
        config.geometry,
        config.ebw,
        Ip_A=Ip_A,
        t_s=t_s,
    )
    if result.O_cutoff is None or result.UHR is None:
        raise RuntimeError(f"no complete O-X-B candidate: {result.reason}")

    R_bounds = config.ebw.R_bounds_m or config.geometry.R_bounds_m
    Z_bounds = config.ebw.Z_bounds_m or config.geometry.Z_bounds_m
    R = np.linspace(R_bounds[0], R_bounds[1], 280)
    Z = np.linspace(Z_bounds[0], Z_bounds[1], 260)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    density_grid = np.empty_like(RR)
    B_grid = np.empty_like(RR)
    for row in range(RR.shape[0]):
        for col in range(RR.shape[1]):
            r = float(RR[row, col])
            z = float(ZZ[row, col])
            density_grid[row, col] = density.electron_density_m3(r, z, t_s)
            B_grid[row, col] = magnetic.evaluate(r, z, Ip_A, t_s).magnitude_T

    omega = 2.0 * PI * source.frequency_Hz
    cutoff_density = EPS0_F_M * M_E_KG * omega**2 / E_CHARGE_C**2
    resonance_B = omega * M_E_KG / (source.harmonic * E_CHARGE_C)
    wp2 = density_grid * E_CHARGE_C**2 / (EPS0_F_M * M_E_KG)
    wc2 = (E_CHARGE_C * B_grid / M_E_KG) ** 2
    uhr_function = (wp2 + wc2) / omega**2

    o_points = ray.points[: result.O_cutoff.path_index + 1]
    x_points = ray.points[result.UHR.path_index : result.O_cutoff.path_index + 1]
    ebw_points = result.ebw_path

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.7), constrained_layout=True)
    colors = {
        "O": "#146b8c",
        "X": "#d77c13",
        "B": "#a13b6b",
        "cutoff": "#2b7a3d",
        "uhr": "#8b5a2b",
        "res": "#b3261e",
    }
    density_levels = np.geomspace(max(config.ebw.density_floor_m3, 1.0e14), peak_density, 15)

    def draw_background(ax, *, local: bool) -> None:
        filled = ax.contourf(
            RR,
            ZZ,
            density_grid,
            levels=density_levels,
            norm=LogNorm(vmin=density_levels[0], vmax=density_levels[-1]),
            cmap="Greys",
            alpha=0.15,
        )
        ax.contour(RR, ZZ, density_grid, levels=[cutoff_density], colors=[colors["cutoff"]], linewidths=1.8)
        ax.contour(RR, ZZ, uhr_function, levels=[1.0], colors=[colors["uhr"]], linewidths=1.8, linestyles="--")
        ax.contour(RR, ZZ, B_grid, levels=[resonance_B], colors=[colors["res"]], linewidths=1.8, linestyles=":")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(True, alpha=0.16)
        if not local:
            ax.set_xlim(R_bounds)
            ax.set_ylim(Z_bounds)
        return filled

    ax = axes[0]
    filled = draw_background(ax, local=False)
    ax.plot([point.R_m for point in o_points], [point.Z_m for point in o_points], color=colors["O"], lw=2.5, label="O ray")
    ax.plot([point.R_m for point in x_points], [point.Z_m for point in x_points], color=colors["X"], lw=2.8, label="slow X candidate")
    if ebw_points:
        ax.plot([point.R_m for point in ebw_points], [point.Z_m for point in ebw_points], color=colors["B"], lw=2.8, label="EBW schematic")
    ax.scatter([source.launch.R_m], [source.launch.Z_m], marker=">", s=70, color=colors["O"], zorder=6, label="launcher")
    ax.scatter([result.O_cutoff.R_m], [result.O_cutoff.Z_m], marker="s", s=58, color=colors["cutoff"], zorder=7)
    ax.scatter([result.UHR.R_m], [result.UHR.Z_m], marker="D", s=58, color=colors["uhr"], zorder=7)
    if ebw_points:
        final = ebw_points[-1]
        ax.scatter([final.R_m], [final.Z_m], marker="*", s=130, color=colors["res"], zorder=8)
    ax.set_title(
        "28 GHz O-X-B candidate in the pre-CFS R-Z field\n"
        "Density shading only; no magnetic surfaces"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    density_ticks = [
        value
        for value in (1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 3.0e19)
        if density_levels[0] <= value <= density_levels[-1]
    ]
    cbar = fig.colorbar(
        filled,
        ax=ax,
        shrink=0.79,
        pad=0.02,
        ticks=density_ticks,
        format=LogFormatterSciNotation(),
    )
    cbar.set_label(r"Schematic $n_e$ (m$^{-3}$)")

    ax = axes[1]
    draw_background(ax, local=True)
    ax.plot([point.R_m for point in x_points], [point.Z_m for point in x_points], color=colors["X"], lw=3.0)
    ax.scatter([result.O_cutoff.R_m], [result.O_cutoff.Z_m], marker="s", s=70, color=colors["cutoff"], zorder=7)
    ax.scatter([result.UHR.R_m], [result.UHR.Z_m], marker="D", s=70, color=colors["uhr"], zorder=7)
    ax.annotate(
        "O cutoff candidate",
        (result.O_cutoff.R_m, result.O_cutoff.Z_m),
        xytext=(8, -24),
        textcoords="offset points",
    )
    ax.annotate(
        "UHR / X-B candidate",
        (result.UHR.R_m, result.UHR.Z_m),
        xytext=(8, 13),
        textcoords="offset points",
    )
    midpoint = (
        0.5 * (result.O_cutoff.R_m + result.UHR.R_m),
        0.5 * (result.O_cutoff.Z_m + result.UHR.Z_m),
    )
    ax.annotate(
        f"conversion interval ~ {1.0e2 * (result.conversion_separation_m or 0.0):.1f} cm",
        midpoint,
        xytext=(0, 26),
        textcoords="offset points",
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "0.35"},
    )
    margin = max(0.075, 1.8 * (result.conversion_separation_m or 0.04))
    ax.set_xlim(min(result.O_cutoff.R_m, result.UHR.R_m) - margin, max(result.O_cutoff.R_m, result.UHR.R_m) + margin)
    ax.set_ylim(min(result.O_cutoff.Z_m, result.UHR.Z_m) - margin, max(result.O_cutoff.Z_m, result.UHR.Z_m) + margin)
    ax.set_title(
        "Local O-X-B conversion region\n"
        + rf"$N_\parallel={result.n_parallel_at_O_cutoff:.3f}$, "
        + rf"$N_{{\parallel,opt}}={result.optimal_n_parallel:.3f}$, "
        + rf"$\eta_{{OX}}\approx{result.ox_efficiency_estimate:.2f}$"
    )
    legend_handles = [
        plt.Line2D([], [], color=colors["cutoff"], lw=1.8, label="O cutoff"),
        plt.Line2D([], [], color=colors["uhr"], lw=1.8, ls="--", label="UHR"),
        plt.Line2D([], [], color=colors["res"], lw=1.8, ls=":", label="EC resonance"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=9)

    status = (
        "geometric EBWCD candidate only"
        if result.ebwcd_candidate
        else "conversion candidate only"
    )
    fig.suptitle(
        f"{source.name}: {status}; no GS/EFIT surfaces used\n"
        "Cold O-X estimate + candidate X-B layer + schematic EBW path (not a full-wave efficiency)",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    json_path = output_path.with_suffix(".json")
    payload = result.to_dict(include_path=True)
    payload["incident_O_ray"] = ray.to_dict(include_path=True)
    payload["density_profile"] = {
        "model": "gaussian_cloud",
        "center_R_m": center_R,
        "center_Z_m": config.ebw.center_Z_m,
        "sigma_R_m": config.ebw.sigma_R_m,
        "sigma_Z_m": config.ebw.sigma_Z_m,
        "peak_density_m3": peak_density,
    }
    payload["equilibrium_model"] = "none; MagneticFieldProvider only"
    json_path.write_text(json.dumps(_finite_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/linec_parameters.ini")
    parser.add_argument("--source", default="EC28_O")
    parser.add_argument("--time-s", type=float, default=-0.05)
    parser.add_argument("--Ip-A", type=float, default=0.0)
    parser.add_argument("--peak-ne-m3", type=float, default=None)
    parser.add_argument("--poloidal-angle-deg", type=float, default=None)
    parser.add_argument("--toroidal-angle-deg", type=float, default=None)
    parser.add_argument("--output", default="runs/oxb_demo/ec28_O1_oxb_trajectory.png")
    args = parser.parse_args()
    image, data = make_figure(
        Path(args.config),
        Path(args.output),
        source_name=args.source,
        t_s=args.time_s,
        Ip_A=args.Ip_A,
        peak_density_m3=args.peak_ne_m3,
        poloidal_angle_deg=args.poloidal_angle_deg,
        toroidal_angle_deg=args.toroidal_angle_deg,
    )
    print(image)
    print(data)


if __name__ == "__main__":
    main()
