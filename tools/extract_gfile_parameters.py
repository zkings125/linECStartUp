"""Extract auditable linEC reference parameters from a standard G-EQDSK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecstartup.constants import E_CHARGE_C, M_E_KG, PI
from ecstartup.physics.eqdsk import read_gfile


def _launch_argument(value: str) -> tuple[str, float, float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("launch must be NAME,FREQUENCY_HZ,R_M,Z_M")
    try:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("launch frequency/R/Z must be numeric") from exc


def extract(path: Path, launches: list[tuple[str, float, float, float]]) -> dict[str, Any]:
    data = read_gfile(path)
    B_R, B_Z, B_phi = data.field_arrays()
    B_magnitude = np.sqrt(B_R**2 + B_Z**2 + B_phi**2)
    report: dict[str, Any] = {
        "source": {
            "path": str(path.resolve()),
            "header": data.header,
            "snapshot_kind": "closed-surface EFIT equilibrium",
            "startup_warning": "Do not use this snapshot as a pre-CFS magnetic history.",
        },
        "standard_gfile_values": {
            "rcentr_m": data.rcentr_m,
            "bcentr_T": data.bcentr_T,
            "plasma_current_A": data.current_A,
            "magnetic_axis_R_m": data.rmaxis_m,
            "magnetic_axis_Z_m": data.zmaxis_m,
            "psi_axis_Wb": data.simag_Wb,
            "psi_boundary_Wb": data.sibry_Wb,
            "R_grid_bounds_m": [float(data.R_grid_m[0]), float(data.R_grid_m[-1])],
            "Z_grid_bounds_m": [float(data.Z_grid_m[0]), float(data.Z_grid_m[-1])],
            "boundary_area_m2": data.boundary_area_m2,
            "equivalent_minor_radius_m": data.equivalent_minor_radius_m,
            "estimated_volume_m3": data.volume_m3,
            "plasma_pressure_axis_Pa": float(data.pressure_Pa[0]),
            "plasma_pressure_max_Pa": float(np.max(data.pressure_Pa)),
            "q_axis": float(data.qpsi[0]),
            "q_edge": float(data.qpsi[-1]),
            "B_magnitude_grid_min_T": float(np.min(B_magnitude)),
            "B_magnitude_grid_max_T": float(np.max(B_magnitude)),
        },
        "suggested_linec_reference": {
            "geometry.R0_m": data.rmaxis_m,
            "geometry.minor_radius_m": data.equivalent_minor_radius_m,
            "geometry.volume_m3": data.volume_m3,
            "geometry.B_t_T": abs(data.bcentr_T),
            "geometry.R_bounds_m": [float(data.R_grid_m[0]), float(data.R_grid_m[-1])],
            "geometry.Z_bounds_m": [float(data.Z_grid_m[0]), float(data.Z_grid_m[-1])],
            "initial.Ip_A": data.current_A,
        },
        "not_available_in_standard_gfile": [
            "neutral gas pressure or gas-valve waveform",
            "EC frequency, injected power, timing, launch polarization, or launcher steering",
            "individual PF/TF/CS coil currents and their waveforms",
            "pre-breakdown or pre-CFS magnetic-field evolution",
            "wall recycling, desorption, and sputtering parameters",
        ],
        "interpretation": {
            "pressure_note": "G-EQDSK pressure_Pa is plasma kinetic pressure p(psi), not neutral fill-gas pressure.",
            "current_note": "current_A is equilibrium plasma current, not CS-coil current.",
            "B_t_note": "bcentr_T is the reference toroidal field at rcentr_m for this snapshot.",
        },
    }

    field = data.to_field_provider(boundary="error")
    launch_reports = []
    for name, frequency, R_m, Z_m in launches:
        inside = data.R_grid_m[0] <= R_m <= data.R_grid_m[-1] and data.Z_grid_m[0] <= Z_m <= data.Z_grid_m[-1]
        item: dict[str, Any] = {
            "name": name,
            "frequency_Hz": frequency,
            "R_m": R_m,
            "Z_m": Z_m,
            "inside_gfile_grid": bool(inside),
            "note": "Launcher coordinates do not by themselves locate the EC resonance layer.",
        }
        if inside:
            point = field.evaluate(R_m, Z_m, data.current_A, 0.0)
            item["B_magnitude_T"] = point.magnitude_T
            item["omega_ce_over_omega_rf"] = E_CHARGE_C * point.magnitude_T / M_E_KG / (2.0 * PI * frequency)
        launch_reports.append(item)
    report["launch_points"] = launch_reports
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gfile", type=Path)
    parser.add_argument(
        "--launch",
        action="append",
        type=_launch_argument,
        default=[],
        help="repeatable NAME,FREQUENCY_HZ,R_M,Z_M",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = extract(args.gfile, args.launch)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
