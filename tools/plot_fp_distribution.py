"""Plot initial/final 0D-2V distributions and current evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def _add_velocity_grid(axis: plt.Axes, maximum_speed: float) -> None:
    """Overlay constant-speed semicircles and pitch-angle rays."""

    theta = np.linspace(0.0, np.pi, 361)
    ring_step = 2.0 if maximum_speed >= 8.0 else maximum_speed / 4.0
    ring_speeds = np.arange(ring_step, maximum_speed + 0.5 * ring_step, ring_step)
    for speed in ring_speeds:
        axis.plot(
            speed * np.cos(theta),
            speed * np.sin(theta),
            color="black",
            linewidth=0.55,
            linestyle=(0, (2, 3)),
            alpha=0.38,
            zorder=4,
        )
    for angle_deg in (30.0, 60.0, 90.0, 120.0, 150.0):
        angle = np.deg2rad(angle_deg)
        axis.plot(
            [0.0, maximum_speed * np.cos(angle)],
            [0.0, maximum_speed * np.sin(angle)],
            color="black",
            linewidth=0.45,
            linestyle=(0, (2, 4)),
            alpha=0.25,
            zorder=4,
        )


def _plot_log_distribution(
    axis: plt.Axes,
    v_parallel: np.ndarray,
    v_perp: np.ndarray,
    values: np.ndarray,
    filled_levels: np.ndarray,
) -> plt.contourf:
    image = axis.contourf(
        v_parallel,
        v_perp,
        values.T,
        levels=filled_levels,
        cmap="viridis",
        extend="both",
        antialiased=True,
    )
    line_levels = filled_levels[::4]
    axis.contour(
        v_parallel,
        v_perp,
        values.T,
        levels=line_levels,
        colors="black",
        linewidths=0.70,
        alpha=0.90,
        zorder=3,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("distribution", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with np.load(args.distribution) as archive:
        initial = archive["f_initial"]
        final = archive["f_e"]
        v_parallel = archive["v_parallel_m_s"] / 1.0e6
        v_perp = archive["v_perp_m_s"] / 1.0e6

    floor = max(float(np.max(initial)) * 1.0e-14, np.finfo(float).tiny)
    initial_log = np.log10(np.maximum(initial, floor))
    final_log = np.log10(np.maximum(final, floor))
    difference = (final - initial) / max(float(np.max(initial)), np.finfo(float).tiny)
    limit = max(float(np.max(np.abs(difference))), 1.0e-12)
    difference_plot = difference.copy()
    difference_plot[np.abs(difference_plot) < 1.0e-4 * limit] = 0.0
    log_min = float(min(np.min(initial_log), np.min(final_log)))
    log_max = float(max(np.max(initial_log), np.max(final_log)))
    log_levels = np.linspace(log_min, log_max, 29)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    image0 = _plot_log_distribution(
        axes[0, 0], v_parallel, v_perp, initial_log, log_levels
    )
    axes[0, 0].set_title("Initial $\\log_{10} f_e$")
    fig.colorbar(image0, ax=axes[0, 0], label="log10 SI distribution")

    image1 = _plot_log_distribution(
        axes[0, 1], v_parallel, v_perp, final_log, log_levels
    )
    axes[0, 1].set_title("Final $\\log_{10} f_e$")
    fig.colorbar(image1, ax=axes[0, 1], label="log10 SI distribution")

    difference_levels = np.linspace(-limit, limit, 29)
    image2 = axes[1, 0].contourf(
        v_parallel,
        v_perp,
        difference_plot.T,
        levels=difference_levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
        extend="both",
        antialiased=True,
    )
    nonzero_levels = limit * np.asarray([-0.75, -0.50, -0.25, 0.25, 0.50, 0.75])
    axes[1, 0].contour(
        v_parallel,
        v_perp,
        difference_plot.T,
        levels=nonzero_levels,
        colors="black",
        linewidths=0.6,
        alpha=0.65,
        zorder=3,
    )
    axes[1, 0].set_title("$(f_e-f_{e0})/\\max(f_{e0})$")
    fig.colorbar(image2, ax=axes[1, 0], label="normalized change")

    history = summary["history"]
    time_s = np.asarray([item["time_s"] for item in history])
    if float(np.max(time_s, initial=0.0)) >= 1.0e-3:
        plot_time = time_s * 1.0e3
        time_label = "time [ms]"
    else:
        plot_time = time_s * 1.0e6
        time_label = "time [us]"
    current_A = np.asarray([item["current_parallel_A"] for item in history])
    axes[1, 1].plot(plot_time, current_A, color="#b42318", linewidth=2.0)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Distribution-moment parallel current")
    axes[1, 1].set_xlabel(time_label)
    axes[1, 1].set_ylabel("current [A]")
    axes[1, 1].grid(alpha=0.25)

    for axis in axes.flat[:3]:
        axis.set_xlabel("$v_\\parallel$ [$10^6$ m/s]")
        axis.set_ylabel("$v_\\perp$ [$10^6$ m/s]")
        axis.set_xlim(float(v_parallel[0]), float(v_parallel[-1]))
        axis.set_ylim(0.0, float(v_perp[-1]))
        axis.set_aspect("equal", adjustable="box")
        _add_velocity_grid(axis, min(abs(v_parallel[0]), v_parallel[-1], v_perp[-1]))

    source = str(summary.get("rf_diffusion_source", "unknown"))
    if source.startswith("genray_like_internal"):
        title = "Reduced 0D-2V Fokker-Planck with internal GENRAY-like D_ql"
        subtitle = "Power-normalized reduced pre-CFS model; not full GENRAY/CQL3D"
    elif source.startswith("genray_npz"):
        title = "Reduced 0D-2V Fokker-Planck with imported GENRAY-normalized D_ql"
        subtitle = "Custom D_ql exchange archive; not native genray.nc"
    elif source.startswith("synthetic"):
        title = "Reduced 0D-2V Fokker-Planck synthetic resonance"
        subtitle = "Sensitivity demonstration only; D_ql is not a validated EC/EBW prediction"
    else:
        title = "Reduced 0D-2V Fokker-Planck distribution evolution"
        subtitle = source
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
