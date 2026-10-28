"""Run a DYON-like case and plot its species, wall, and power evolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecstartup.config import load_config
from ecstartup.solver import integrate


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)


def _positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def make_figure(config_path: Path, output_path: Path) -> Path:
    config = load_config(config_path)
    if not config.species_wall.enabled:
        raise ValueError("plot requires species_wall.enabled=true")
    result = integrate(config)
    if not result.success:
        raise RuntimeError(result.message)

    rows = result.diagnostics
    time_ms = result.times_s * 1.0e3
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    fig.suptitle("DYON-like species and wall evolution\nAnalytic-rate surrogate; not an ADAS prediction", fontsize=16)

    ax = axes[0, 0]
    charge_keys = sorted(
        key
        for key in rows[-1]
        if key.startswith("n_C") and key.endswith("_m3") and key[3:-3].isdigit()
    )
    for key in charge_keys:
        ax.plot(time_ms, _positive(_series(rows, key)), label=key.removesuffix("_m3"))
    ax.set_yscale("log")
    ax.set_title("Carbon charge-state population")
    ax.set_ylabel(r"Density (m$^{-3}$)")
    ax.legend(ncol=2, frameon=False)

    ax = axes[0, 1]
    ax.plot(time_ms, _positive(_series(rows, "ne_m3")), label=r"integrated $n_e$", linewidth=2.0)
    ax.plot(
        time_ms,
        _positive(_series(rows, "electron_density_quasineutral_m3")),
        label=r"$\sum_s\sum_q qn_{s,q}$",
        linestyle="--",
    )
    ax.set_yscale("log")
    ax.set_title("Electron density consistency")
    ax.set_ylabel(r"Density (m$^{-3}$)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    inventory_keys = sorted(key for key in rows[-1] if key.startswith("wall_inventory_") and key.endswith("_m2"))
    for key in inventory_keys:
        ax.plot(time_ms, _positive(_series(rows, key)), label=key.removeprefix("wall_inventory_").removesuffix("_m2"))
    ax.set_yscale("log")
    ax.set_title("Retained wall inventory")
    ax.set_ylabel(r"Surface inventory (m$^{-2}$)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    source_specs = [
        ("S_wall_C_m3_s", "C ion wall loss"),
        ("S_recycle_C_m3_s", "C recycling/desorption"),
        ("S_sputter_C_m3_s", "C sputtering"),
        ("S_fuel_recycle_m3_s", "D return source"),
    ]
    for key, label in source_specs:
        ax.plot(time_ms, _positive(_series(rows, key)), label=label)
    ax.set_yscale("log")
    ax.set_title("Wall particle sources and sinks")
    ax.set_ylabel(r"Rate (m$^{-3}$ s$^{-1}$)")
    ax.legend(frameon=False)

    ax = axes[2, 0]
    power_specs = [
        ("P_rf_W_m3", "absorbed EC"),
        ("P_rad_base_W_m3", "base radiation"),
        ("P_rad_species_W_m3", "species radiation"),
        ("P_ion_species_W_m3", "impurity ionization"),
    ]
    for key, label in power_specs:
        ax.plot(time_ms, _positive(_series(rows, key)), label=label)
    ax.set_yscale("log")
    ax.set_title("Electron power channels")
    ax.set_ylabel(r"Power density (W m$^{-3}$)")
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.plot(time_ms, _series(rows, "ambipolar_potential_V"), label=r"ambipolar $\phi_a$", linewidth=2.0)
    ax.set_title("Ambipolar closure and charge balance")
    ax.set_ylabel("Ambipolar potential (V)")
    ax2 = ax.twinx()
    residual = np.abs(_series(rows, "charge_balance_relative"))
    ax2.plot(time_ms, np.maximum(residual, 1.0e-30), label="relative charge imbalance", linestyle="--", color="tab:red")
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative charge imbalance", color="tab:red")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    for row in axes:
        for ax in row:
            ax.set_xlabel("Time (ms)")
            ax.grid(True, which="both", alpha=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/dyon_species_wall.yaml")
    parser.add_argument("--output", default="runs/dyon_species_wall/dyon_species_wall_evolution.png")
    args = parser.parse_args()
    print(make_figure(Path(args.config), Path(args.output)))


if __name__ == "__main__":
    main()
