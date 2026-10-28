"""命令行入口：配置校验、启动积分、场线诊断和批量比较。"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

from .config import load_config
from .workflow import run_simulation


def _json_finite(value):
    """Convert non-finite diagnostic floats to JSON null recursively."""

    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        # Convert NumPy scalar values to native Python numbers before handing
        # them to json.dumps.
        return numeric if not isinstance(value, (int, float)) else value
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, tuple):
        return [_json_finite(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecstartup", description="Tokamak startup (breakdown-preionization-CFS) 0D prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate input and print resolved SI configuration")
    validate.add_argument("config", type=Path)
    inspect_mat = sub.add_parser(
        "inspect-equilibrium-mat",
        help="extract J-TEXT/MAT equilibrium shape and device-reference parameters",
    )
    inspect_mat.add_argument("mat_file", type=Path)
    inspect_mat.add_argument("--output", type=Path, default=None)
    inspect_shots = sub.add_parser(
        "inspect-exl-shots",
        help="inspect EXL-50U per-channel CSV waveforms and derive provisional validation labels",
    )
    inspect_shots.add_argument("directory", type=Path)
    inspect_shots.add_argument("--t-start", type=float, default=-0.1)
    inspect_shots.add_argument("--t-end", type=float, default=0.1)
    inspect_shots.add_argument("--calibration", type=Path, default=None)
    inspect_shots.add_argument("--output", type=Path, default=None)
    validation_set = sub.add_parser(
        "build-exl-validation",
        help="build shot-resolved EXL-50U validation JSON and multi-page diagnostic PDF",
    )
    validation_set.add_argument("directory", type=Path)
    validation_set.add_argument("--t-start", type=float, default=-0.1)
    validation_set.add_argument("--t-end", type=float, default=0.1)
    validation_set.add_argument("--calibration", type=Path, default=None)
    validation_set.add_argument("--output", type=Path, required=True)
    audit_geometry = sub.add_parser(
        "audit-device-geometry",
        help="audit whether metric coil geometry, wall and shot channels are ready for field-line physics",
    )
    audit_geometry.add_argument("geometry", type=Path)
    audit_geometry.add_argument("--shot-csv", type=Path, default=None)
    audit_geometry.add_argument("--output", type=Path, default=None)
    readiness = sub.add_parser(
        "assess-device-readiness",
        help="audit whether startup classification and Ip(t) may be reported quantitatively",
    )
    readiness.add_argument("config", type=Path)
    readiness.add_argument("--output", type=Path, default=None)
    frozen_validation = sub.add_parser(
        "audit-validation-manifest",
        help="verify a frozen, non-overlapping success/failure fit and validation split",
    )
    frozen_validation.add_argument("manifest", type=Path)
    frozen_validation.add_argument("--no-file-check", action="store_true")
    frozen_validation.add_argument("--output", type=Path, default=None)
    p0 = sub.add_parser(
        "assess-p0",
        help="audit all six pre-quantitative physics gates using checksummed evidence",
    )
    p0.add_argument("config", type=Path)
    p0.add_argument("--evidence", type=Path, default=None)
    p0.add_argument("--output", type=Path, default=None)
    startup_resonance = sub.add_parser(
        "analyze-startup-resonance",
        help="diagnose pre-CFS EC harmonic access, trapping proxy, and pressure penalty",
    )
    startup_resonance.add_argument("config", type=Path)
    startup_resonance.add_argument("--device", default=None)
    startup_resonance.add_argument("--output", type=Path, default=None)
    inspect_genray = sub.add_parser(
        "inspect-genray",
        help="read native GENRAY NetCDF, convert cgs quantities to SI, and check power/current output",
    )
    inspect_genray.add_argument("netcdf", type=Path)
    inspect_genray.add_argument("--output", type=Path, default=None, help="optional summary JSON path")
    inspect_genray.add_argument("--plot", type=Path, default=None, help="optional validation PNG path")
    inspect_adas = sub.add_parser(
        "inspect-adas",
        help="validate an OpenADAS ADF11 manifest and evaluate its SI coefficients",
    )
    inspect_adas.add_argument("manifest", type=Path)
    inspect_adas.add_argument("--Te-eV", dest="Te_eV", type=float, default=20.0)
    inspect_adas.add_argument("--ne-m3", dest="ne_m3", type=float, default=1.0e19)
    inspect_adas.add_argument("--bounds", choices=("raise", "clip"), default="raise")
    inspect_adas.add_argument("--output", type=Path, default=None)
    inspect_fullwave = sub.add_parser(
        "inspect-fullwave",
        help="validate full-wave convergence, mesh verification and O-X-B port-power balance",
    )
    inspect_fullwave.add_argument("npz", type=Path)
    inspect_fullwave.add_argument("--power-tolerance", type=float, default=1.0e-3)
    inspect_fullwave.add_argument("--output", type=Path, default=None)
    project_fullwave = sub.add_parser(
        "project-fullwave-dql",
        help="project verified full-wave fields onto an open-field-line 1D-2V Dql grid",
    )
    project_fullwave.add_argument("fullwave_npz", type=Path)
    project_fullwave.add_argument("fieldline_npz", type=Path)
    project_fullwave.add_argument("config", type=Path, help="configuration supplying the kinetic grid")
    project_fullwave.add_argument("--output", type=Path, required=True, help="spatial Dql NPZ path")
    project_fullwave.add_argument("--summary", type=Path, default=None)
    project_fullwave.add_argument("--time-s", type=float, default=0.0)
    project_fullwave.add_argument(
        "--resonance-width-fraction",
        type=float,
        default=1.0e-3,
        help="Gaussian angular-frequency resonance width divided by omega",
    )
    project_fullwave.add_argument("--power-tolerance", type=float, default=1.0e-3)
    project_fullwave.add_argument("--harmonic", type=int, choices=(1, 2), default=1)
    project_fullwave.add_argument(
        "--k-perpendicular-m-inv",
        type=float,
        default=None,
        help="required for EBW/mixed fields or n=2 FLR projection",
    )
    generate_genray_like = sub.add_parser(
        "generate-genray-like",
        help="generate a reduced internal ray/resonance/power-normalized D_ql NPZ",
    )
    generate_genray_like.add_argument("config", type=Path)
    generate_genray_like.add_argument("--output", type=Path, required=True, help="D_ql NPZ path")
    generate_genray_like.add_argument("--summary", type=Path, default=None, help="optional diagnostics JSON path")
    assess_dql = sub.add_parser(
        "assess-dql",
        help="audit D_ql availability, pre-CFS coordinates, mode-conversion evidence and quantitative status",
    )
    assess_dql.add_argument("config", type=Path)
    assess_dql.add_argument("--dql-file", type=Path, default=None)
    assess_dql.add_argument("--output", type=Path, default=None)
    validate_genray = sub.add_parser(
        "validate-genray",
        help="compare native GENRAY current/power with internal reduced D_ql and F-P output",
    )
    validate_genray.add_argument("config", type=Path)
    validate_genray.add_argument("netcdf", type=Path)
    validate_genray.add_argument("--output", type=Path, required=True, help="validation directory")
    run = sub.add_parser("run", help="run a case and write outputs")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, default=None)
    post = sub.add_parser("postprocess", help="print summary.json")
    post.add_argument("run_directory", type=Path)
    trace = sub.add_parser(
        "trace",
        help="trace an axisymmetric magnetic field line and print topology diagnostics",
    )
    trace.add_argument("config", type=Path)
    trace.add_argument(
        "--R",
        "--r",
        "--R-m",
        dest="R_m",
        type=float,
        default=None,
        help="start major-radius coordinate [m]; defaults to RF launch R or R0",
    )
    trace.add_argument(
        "--Z",
        "--z",
        "--Z-m",
        dest="Z_m",
        type=float,
        default=None,
        help="start vertical coordinate [m]; defaults to RF launch Z",
    )
    trace.add_argument("--phi", "--phi-rad", dest="phi_rad", type=float, default=0.0)
    trace.add_argument("--Ip", "--ip", "--Ip-A", dest="Ip_A", type=float, default=0.0)
    trace.add_argument("--time", "--time-s", dest="t_s", type=float, default=0.0)
    trace.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    trace.add_argument("--step", "--step-m", dest="step_m", type=float, default=1.0e-3)
    trace.add_argument(
        "--max-length",
        "--max-length-m",
        dest="max_length_m",
        type=float,
        default=100.0,
    )
    trace.add_argument("--closure-tolerance", dest="closure_tolerance_m", type=float, default=None)
    trace.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    trace.add_argument(
        "--no-path",
        action="store_true",
        help="omit sampled path arrays from JSON output",
    )
    trace.add_argument("--output", type=Path, default=None)
    ec_ray = sub.add_parser(
        "raytrace-ec",
        help="trace one EC source with the cold-plasma Hamiltonian backend (no GS surfaces required)",
    )
    ec_ray.add_argument("config", type=Path)
    ec_ray.add_argument("--source", default=None, help="named rf_source; defaults to the first enabled source")
    ec_ray.add_argument("--time", "--time-s", dest="t_s", type=float, default=0.0)
    ec_ray.add_argument("--Ip", "--ip", "--Ip-A", dest="Ip_A", type=float, default=None)
    ec_ray.add_argument("--ne", "--ne-m3", dest="ne_m3", type=float, default=None)
    ec_ray.add_argument("--no-path", action="store_true")
    ec_ray.add_argument("--output", type=Path, default=None)
    oxb = sub.add_parser(
        "analyze-oxb",
        help="analyze an O-X-B/EBW candidate at one startup snapshot without GS surfaces",
    )
    oxb.add_argument("config", type=Path)
    oxb.add_argument("--source", default="EC28_O")
    oxb.add_argument("--time", "--time-s", dest="t_s", type=float, default=0.0)
    oxb.add_argument("--Ip", "--ip", "--Ip-A", dest="Ip_A", type=float, default=None)
    oxb.add_argument("--average-ne", dest="average_ne_m3", type=float, default=None)
    oxb.add_argument("--peak-ne", dest="peak_ne_m3", type=float, default=None)
    oxb.add_argument("--poloidal-angle-deg", type=float, default=None)
    oxb.add_argument("--toroidal-angle-deg", type=float, default=None)
    oxb.add_argument("--no-path", action="store_true")
    oxb.add_argument("--output", type=Path, default=None)
    xb = sub.add_parser(
        "analyze-xb",
        help="diagnose direct X-B R/L cutoffs, UHR, evanescent barrier and X branches",
    )
    xb.add_argument("config", type=Path)
    xb.add_argument("--source", default="EC50_X")
    xb.add_argument("--time", "--time-s", dest="t_s", type=float, default=0.0)
    xb.add_argument("--Ip", "--ip", "--Ip-A", dest="Ip_A", type=float, default=None)
    xb.add_argument("--average-ne", dest="average_ne_m3", type=float, default=None)
    xb.add_argument("--peak-ne", dest="peak_ne_m3", type=float, default=None)
    xb.add_argument("--no-path", action="store_true")
    xb.add_argument("--output", type=Path, default=None)
    fp = sub.add_parser(
        "solve-fp",
        help="solve the independent reduced 0D-2V electron Fokker-Planck model",
    )
    fp.add_argument("config", type=Path)
    fp.add_argument("--output", type=Path, default=None, help="summary JSON path")
    fp.add_argument(
        "--distribution-output",
        type=Path,
        default=None,
        help="optional NPZ path for f_e and velocity coordinates",
    )
    threshold_scan = sub.add_parser(
        "threshold-scan",
        help="scan pressure, effective charge, EC power, and CS mode against a zero-power baseline",
    )
    threshold_scan.add_argument("config", type=Path)
    threshold_scan.add_argument("--output", type=Path, default=None, help="scan JSON path")
    threshold_scan.add_argument("--pdf-output", type=Path, default=None, help="optional PDF override")
    aspect_scan = sub.add_parser(
        "aspect-ratio-scan",
        help="scan independently configured major/minor radius values",
    )
    aspect_scan.add_argument("config", type=Path)
    aspect_scan.add_argument("--output", type=Path, default=None, help="scan JSON path")
    scan = sub.add_parser(
        "scan",
        help="scan a poloidal cross-section with field-line tracing and print topology statistics",
    )
    scan.add_argument("config", type=Path)
    scan.add_argument("--Ip", "--ip", "--Ip-A", dest="Ip_A", type=float, default=None)
    scan.add_argument("--time", "--time-s", dest="t_s", type=float, default=0.0)
    scan.add_argument(
        "--radial-seeds",
        "--n-radial",
        dest="radial_seeds",
        type=int,
        default=5,
        help="number of radial rho=r/a seed rings",
    )
    scan.add_argument(
        "--poloidal-seeds",
        "--n-poloidal",
        dest="poloidal_seeds",
        type=int,
        default=16,
        help="number of uniformly spaced poloidal seeds",
    )
    scan.add_argument("--step", "--step-m", dest="step_m", type=float, default=1.0e-3)
    scan.add_argument(
        "--max-length",
        "--max-length-m",
        dest="max_length_m",
        type=float,
        default=100.0,
    )
    scan.add_argument("--closure-tolerance", dest="closure_tolerance_m", type=float, default=None)
    scan.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    scan.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    scan.add_argument(
        "--invalid-policy",
        choices=("record", "raise"),
        default="record",
        help="record invalid/singular seeds (default) or abort on the first one",
    )
    scan.add_argument(
        "--no-area-weighting",
        action="store_true",
        help="use equal weight for every seed instead of annulus/angle weights",
    )
    scan.add_argument(
        "--no-traces",
        action="store_true",
        help="do not retain native FieldLineTrace objects in the in-memory result",
    )
    scan.add_argument(
        "--no-path",
        action="store_true",
        help="omit sampled path arrays when traces are serialized",
    )
    scan.add_argument("--no-samples", action="store_true", help="omit per-seed records from JSON")
    scan.add_argument(
        "--topology",
        "--detector",
        dest="include_topology",
        action="store_true",
        help="also include a compatibility TopologyResult mapping",
    )
    scan.add_argument(
        "--closed-fraction-threshold",
        type=float,
        default=0.05,
        help="threshold used for topology.has_closed_surface",
    )
    scan.add_argument("--output", type=Path, default=None)
    compare = sub.add_parser("compare", help="compare multiple devices with one model")
    compare.add_argument("matrix", type=Path, help="JSON 比较矩阵")
    compare.add_argument("--output", type=Path, default=None)
    ensemble = sub.add_parser("ensemble", help="propagate parameter uncertainty and estimate event probability")
    ensemble.add_argument("config", type=Path)
    ensemble.add_argument("--parameters", type=Path, required=True, help="参数范围 JSON")
    ensemble.add_argument("--samples", type=int, default=8)
    ensemble.add_argument("--event", default="preionization", help="成功事件类型，默认 preionization")
    ensemble.add_argument("--output", type=Path, default=None)
    ablate = sub.add_parser("ablate", help="mechanism ablation: disable RF, loop field, orbit term, etc.")
    ablate.add_argument("config", type=Path)
    ablate.add_argument("--output", type=Path, default=None)
    exl_report = sub.add_parser(
        "exl-report",
        help="run the EXL-50U 0--100 ms 28/50 GHz and CS/no-CS report suite",
    )
    exl_report.add_argument("config", type=Path)
    exl_report.add_argument("--output", type=Path, default=Path("runs/exl50u_formal"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config)
        print(json.dumps(config.resolved_dict(), ensure_ascii=False, indent=2))
    elif args.command == "inspect-equilibrium-mat":
        from .physics.mat_equilibrium import read_mat_equilibrium

        payload = read_mat_equilibrium(args.mat_file).summary()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "audit-validation-manifest":
        from .validation_manifest import audit_frozen_validation_manifest

        payload = audit_frozen_validation_manifest(
            args.manifest,
            verify_files=not args.no_file_check,
        ).to_dict()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "assess-p0":
        from .p0_readiness import assess_p0_readiness

        config = load_config(args.config)
        payload = assess_p0_readiness(config, args.evidence).to_dict()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "inspect-exl-shots":
        from .experimental import load_signal_calibrations, summarize_exl50u_shot_directory

        payload = summarize_exl50u_shot_directory(
            args.directory,
            t_start_s=args.t_start,
            t_end_s=args.t_end,
            calibrations=(
                None if args.calibration is None else load_signal_calibrations(args.calibration)
            ),
        )
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "build-exl-validation":
        from .experimental import load_signal_calibrations
        from .experimental_validation import build_exl50u_validation_artifacts

        payload = build_exl50u_validation_artifacts(
            args.directory,
            args.output,
            t_start_s=args.t_start,
            t_end_s=args.t_end,
            calibrations=(
                None if args.calibration is None else load_signal_calibrations(args.calibration)
            ),
        )
        print(json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False))
    elif args.command == "audit-device-geometry":
        from .device_geometry import load_device_geometry
        from .experimental import read_exl50u_shot_csv

        geometry = load_device_geometry(args.geometry)
        shot = None
        if args.shot_csv is not None:
            requested = {coil.channel for coil in geometry.coils}
            requested.add(geometry.toroidal_field_channel)
            shot = read_exl50u_shot_csv(
                args.shot_csv,
                channels=requested,
                maximum_points_per_signal=4000,
            )
        payload = geometry.audit(shot)
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "assess-device-readiness":
        from .device_prediction_quality import assess_device_prediction_quality

        config = load_config(args.config)
        payload = assess_device_prediction_quality(config).to_dict()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "analyze-startup-resonance":
        from .physics.startup_resonance import assess_startup_resonance

        config = load_config(args.config)
        payload = assess_startup_resonance(config, device=args.device).to_dict()
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "inspect-genray":
        from .genray_io import plot_genray_summary, read_genray_netcdf

        result = read_genray_netcdf(args.netcdf)
        payload = result.summary()
        if args.plot is not None:
            plot_genray_summary(result, args.plot)
            payload["plot"] = str(args.plot)
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "inspect-adas":
        from .physics.adas import OpenADASAtomicRateProvider

        provider = OpenADASAtomicRateProvider(args.manifest, bounds=args.bounds)
        species = next(iter(provider.tables.values())).element
        payload = {
            "provider": provider.label,
            "element": species,
            "Te_eV": args.Te_eV,
            "ne_m3": args.ne_m3,
            "coefficients_si": {
                name: table.coefficient_si(args.Te_eV, args.ne_m3, bounds=args.bounds)
                for name, table in provider.tables.items()
            },
            "units": {name: table.si_unit for name, table in provider.tables.items()},
            "manifest": str(args.manifest.resolve()),
        }
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "inspect-fullwave":
        from .physics.fullwave_dql import load_fullwave_npz

        field = load_fullwave_npz(args.npz, power_tolerance=args.power_tolerance)
        payload = {
            "source": str(args.npz.resolve()),
            "solver": field.solver_name,
            "solver_version": field.solver_version,
            "frequency_Hz": field.frequency_Hz,
            "converged": field.converged,
            "mesh_verified": field.mesh_verified,
            "power_budget": field.power_budget.__dict__,
            "power_balance_relative": field.power_budget.balance_relative,
            "oxb_conversion_efficiency": field.power_budget.oxb_conversion_efficiency,
            "wave_branch": field.wave_branch,
            "mode_decomposition_verified": field.mode_decomposition_verified,
            "axisymmetric_absorbed_power_W": field.axisymmetric_absorbed_power_W(),
            "local_absorption_balance_relative": field.local_absorption_balance_relative,
            "quantitative_status": (
                "accepted_verified_fullwave_and_mode_decomposition"
                if field.mode_decomposition_verified
                else "accepted_field_solution_but_wave_branch_unresolved"
            ),
        }
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "project-fullwave-dql":
        import numpy as np

        from .physics.fokker_planck import RelativisticMomentumGrid, VelocityGrid
        from .physics.fullwave_dql import (
            load_fullwave_npz,
            load_open_fieldline_npz,
            project_fullwave_fundamental_dql,
            project_fullwave_harmonic_dql,
            save_fullwave_dql_npz,
        )

        config = load_config(args.config)
        field = load_fullwave_npz(args.fullwave_npz, power_tolerance=args.power_tolerance)
        spatial = load_open_fieldline_npz(args.fieldline_npz)
        kinetic = config.kinetic
        velocity = (
            RelativisticMomentumGrid.uniform(
                kinetic.n_v_parallel,
                kinetic.n_v_perp,
                kinetic.u_parallel_max,
                kinetic.u_perp_max,
            )
            if kinetic.coordinate_system == "relativistic_momentum"
            else VelocityGrid.uniform(
                kinetic.n_v_parallel,
                kinetic.n_v_perp,
                kinetic.v_parallel_max_m_s,
                kinetic.v_perp_max_m_s,
            )
        )
        if args.resonance_width_fraction <= 0.0:
            raise ValueError("resonance-width-fraction must be positive")
        resonance_width = 2.0 * np.pi * field.frequency_Hz * args.resonance_width_fraction
        use_flr_projection = field.wave_branch in {"ebw", "mixed"} or args.harmonic == 2
        if use_flr_projection:
            if args.k_perpendicular_m_inv is None or args.k_perpendicular_m_inv <= 0.0:
                raise ValueError(
                    "EBW/mixed or second-harmonic projection requires positive "
                    "--k-perpendicular-m-inv from the resolved full-wave spectrum"
                )
            result = project_fullwave_harmonic_dql(
                field,
                spatial,
                velocity,
                harmonic=args.harmonic,
                k_perpendicular_m_inv=args.k_perpendicular_m_inv,
                time_s=args.time_s,
                resonance_width_rad_s=resonance_width,
            )
        else:
            result = project_fullwave_fundamental_dql(
                field,
                spatial,
                velocity,
                time_s=args.time_s,
                resonance_width_rad_s=resonance_width,
            )
        output = save_fullwave_dql_npz(args.output, result, spatial, velocity, field)
        determinant = (
            result.diffusion.D_parallel_m2_s3 * result.diffusion.D_perp_m2_s3
            - result.diffusion.D_cross_m2_s3**2
        )
        determinant_scale = np.maximum(
            result.diffusion.D_parallel_m2_s3 * result.diffusion.D_perp_m2_s3,
            1.0,
        )
        payload = {
            "output": str(output),
            "source_fullwave": str(args.fullwave_npz.resolve()),
            "source_fieldline": str(args.fieldline_npz.resolve()),
            "shape": list(result.diffusion.D_parallel_m2_s3.shape),
            "coordinate_system": kinetic.coordinate_system,
            "target_absorbed_power_W": result.target_absorbed_power_W,
            "oxb_conversion_efficiency": result.oxb_conversion_efficiency,
            "fullwave_power_balance_relative": result.fullwave_power_balance_relative,
            "local_absorption_balance_relative": field.local_absorption_balance_relative,
            "minimum_diffusion_determinant": float(np.min(determinant)),
            "minimum_diffusion_determinant_relative": float(
                np.min(determinant / determinant_scale)
            ),
            "projection_model": result.projection_model,
            "wave_branch": field.wave_branch,
            "mode_decomposition_verified": field.mode_decomposition_verified,
            "harmonic": args.harmonic,
            "k_perpendicular_m_inv": args.k_perpendicular_m_inv,
            "pre_cfs_equilibrium_model": "none; physical open-field-line coordinates",
            "quantitative_boundary": (
                "full-wave conversion is accepted from the verified external solver; "
                "startup current remains subject to 1D-2V and device validation"
            ),
        }
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "generate-genray-like":
        import numpy as np

        from .physics.fokker_planck import RelativisticMomentumGrid, VelocityGrid
        from .physics.genray_like import build_internal_genray_like_reference

        config = load_config(args.config)
        grid = (
            RelativisticMomentumGrid.uniform(
                config.kinetic.n_v_parallel,
                config.kinetic.n_v_perp,
                config.kinetic.u_parallel_max,
                config.kinetic.u_perp_max,
            )
            if config.kinetic.coordinate_system == "relativistic_momentum"
            else VelocityGrid.uniform(
                config.kinetic.n_v_parallel,
                config.kinetic.n_v_perp,
                config.kinetic.v_parallel_max_m_s,
                config.kinetic.v_perp_max_m_s,
            )
        )
        result = build_internal_genray_like_reference(config, grid)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from .dql_provenance import internal_dql_metadata

        metadata = internal_dql_metadata(config)
        archive_payload = dict(
            D_parallel_m2_s3=result.diffusion.D_parallel_m2_s3,
            D_perp_m2_s3=result.diffusion.D_perp_m2_s3,
            D_cross_m2_s3=result.diffusion.D_cross_m2_s3,
            reference_absorbed_power_W=np.asarray(result.reference_absorbed_power_W),
            generator=np.asarray("linEC_reduced_genray_like"),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        if isinstance(grid, RelativisticMomentumGrid):
            archive_payload.update(u_parallel=grid.u_parallel, u_perp=grid.u_perp)
        else:
            archive_payload.update(v_parallel_m_s=grid.v_parallel_m_s, v_perp_m_s=grid.v_perp_m_s)
        np.savez_compressed(args.output, **archive_payload)
        payload = {
            "output": str(args.output),
            "reference_absorbed_power_W": result.reference_absorbed_power_W,
            "diffusion_source": result.diffusion.source,
            "diagnostics": [item.to_dict() for item in result.diagnostics],
            "scope": (
                "reduced cold-Stix ray + relativistic resonance kernel + initial-Maxwellian "
                "power normalization; not full GENRAY"
            ),
            "dql_assessment": metadata,
        }
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "assess-dql":
        from .dql_provenance import assess_dql_source

        config = load_config(args.config)
        payload = assess_dql_source(config, source_file=args.dql_file).to_dict()
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "validate-genray":
        from .genray_validation import validate_genray_case

        config = load_config(args.config)
        payload = validate_genray_case(config, args.netcdf, args.output)
        print(json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False))
    elif args.command == "run":
        config = load_config(args.config)
        result = run_simulation(config, args.output)
        print(f"success={result.success}; samples={len(result.times_s)}; events={len(result.events)}")
        print(result.message)
    elif args.command == "exl-report":
        from .exl50u_suite import run_exl50u_suite

        config = load_config(args.config)
        payload = run_exl50u_suite(config, args.output)
        print(json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False))
    elif args.command == "trace":
        from .physics.fieldline import FieldLineTracer
        from .physics.magnetic import make_magnetic_field

        config = load_config(args.config)
        field = make_magnetic_field(config.geometry)
        launch_R = config.rf.launch.R_m
        start_R = args.R_m if args.R_m is not None else (launch_R if launch_R is not None else config.geometry.R0_m)
        start_Z = args.Z_m if args.Z_m is not None else config.rf.launch.Z_m
        tracer = FieldLineTracer(
            field,
            step_m=args.step_m,
            max_length_m=args.max_length_m,
            closure_tolerance_m=args.closure_tolerance_m,
            max_steps=args.max_steps,
        )
        trace_result = tracer.trace(
            start_R_m=start_R,
            start_Z_m=start_Z,
            start_phi_rad=args.phi_rad,
            Ip_A=args.Ip_A,
            t_s=args.t_s,
            direction=args.direction,
        )
        payload = _json_finite(trace_result.to_dict(include_path=not args.no_path))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "raytrace-ec":
        from .physics.ec_raytrace import HamiltonianECRayTracer, UniformElectronDensity
        from .physics.magnetic import make_magnetic_field
        from .physics.rf import ECRHSource

        config = load_config(args.config)
        sources = ECRHSource(config.rf, config.geometry).sources
        if args.source is None:
            source = next((item for item in sources if item.enabled), sources[0])
        else:
            source = next((item for item in sources if item.name == args.source), None)
            if source is None:
                available = ", ".join(item.name for item in sources)
                raise SystemExit(f"未知 RF source {args.source!r}；可用源: {available}")
        ne_m3 = config.initial.ne_m3 if args.ne_m3 is None else args.ne_m3
        if ne_m3 < 0.0:
            raise SystemExit("--ne-m3 must be non-negative")
        Ip_A = config.initial.Ip_A if args.Ip_A is None else args.Ip_A
        tracer = HamiltonianECRayTracer(
            make_magnetic_field(config.geometry),
            UniformElectronDensity(ne_m3),
            config.geometry,
            config.ray_tracing,
            ion_mass_kg=config.gas.ion_mass_kg,
        )
        result = tracer.trace(source, Ip_A=Ip_A, t_s=args.t_s)
        payload = result.to_dict(include_path=not args.no_path)
        payload["source_configured_enabled"] = source.enabled
        payload["ray_tracing_switch_enabled"] = config.ray_tracing.enabled
        payload["density_model"] = "uniform_0d_snapshot"
        payload["equilibrium_model"] = "none; MagneticFieldProvider only"
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "analyze-oxb":
        from .physics.ebw import GaussianDensityCloud, analyze_oxb
        from .physics.ec_raytrace import HamiltonianECRayTracer
        from .physics.magnetic import make_magnetic_field
        from .physics.rf import ECRHSource

        config = load_config(args.config)
        sources = ECRHSource(config.rf, config.geometry).sources
        source = next((item for item in sources if item.name == args.source), None)
        if source is None:
            available = ", ".join(item.name for item in sources)
            raise SystemExit(f"未知 RF source {args.source!r}；可用源: {available}")
        launch_updates = {}
        if args.poloidal_angle_deg is not None:
            launch_updates["poloidal_angle_deg"] = args.poloidal_angle_deg
        if args.toroidal_angle_deg is not None:
            launch_updates["toroidal_angle_deg"] = args.toroidal_angle_deg
        if launch_updates:
            source = source.model_copy(
                update={"launch": source.launch.model_copy(update=launch_updates)}
            )
        Ip_A = config.initial.Ip_A if args.Ip_A is None else args.Ip_A
        if args.peak_ne_m3 is not None:
            peak_density = args.peak_ne_m3
            density_source = "command_line_peak"
        elif args.average_ne_m3 is not None:
            peak_density = args.average_ne_m3 * config.ebw.peak_to_average_factor
            density_source = "command_line_average_times_profile_factor"
        else:
            peak_density = config.ebw.schematic_peak_density_m3
            density_source = "schematic_config_peak"
        if peak_density < 0.0:
            raise SystemExit("density must be non-negative")
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
        # O-X conversion is expected precisely where geometrical optics loses
        # quality. Continue the diagnostic ray to the cutoff candidate while
        # preserving the low WKB ratio in the output; do not interpret this as
        # a valid full-wave solution through the layer.
        ray_config = config.ray_tracing.model_copy(update={"terminate_on_wkb_failure": False})
        tracer = HamiltonianECRayTracer(
            magnetic,
            density,
            config.geometry,
            ray_config,
            ion_mass_kg=config.gas.ion_mass_kg,
        )
        incident_ray = tracer.trace(source, Ip_A=Ip_A, t_s=args.t_s)
        result = analyze_oxb(
            source,
            incident_ray,
            magnetic,
            density,
            config.geometry,
            config.ebw,
            Ip_A=Ip_A,
            t_s=args.t_s,
        )
        payload = result.to_dict(include_path=not args.no_path)
        payload["incident_O_ray"] = incident_ray.to_dict(include_path=not args.no_path)
        payload["density_source"] = density_source
        payload["density_profile"] = {
            "model": "gaussian_cloud",
            "center_R_m": center_R,
            "center_Z_m": config.ebw.center_Z_m,
            "sigma_R_m": config.ebw.sigma_R_m,
            "sigma_Z_m": config.ebw.sigma_Z_m,
            "peak_density_m3": peak_density,
            "floor_density_m3": config.ebw.density_floor_m3,
        }
        payload["equilibrium_model"] = "none; MagneticFieldProvider only"
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "analyze-xb":
        from .physics.ebw import GaussianDensityCloud, analyze_direct_xb
        from .physics.ec_raytrace import HamiltonianECRayTracer
        from .physics.magnetic import make_magnetic_field
        from .physics.rf import ECRHSource

        config = load_config(args.config)
        sources = ECRHSource(config.rf, config.geometry).sources
        source = next((item for item in sources if item.name == args.source), None)
        if source is None:
            available = ", ".join(item.name for item in sources)
            raise SystemExit(f"未知 RF source {args.source!r}；可用源: {available}")
        Ip_A = config.initial.Ip_A if args.Ip_A is None else args.Ip_A
        if args.peak_ne_m3 is not None:
            peak_density = args.peak_ne_m3
            density_source = "command_line_peak"
        elif args.average_ne_m3 is not None:
            peak_density = args.average_ne_m3 * config.ebw.peak_to_average_factor
            density_source = "command_line_average_times_profile_factor"
        else:
            peak_density = config.ebw.schematic_peak_density_m3
            density_source = "schematic_config_peak"
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
        incident_ray = HamiltonianECRayTracer(
            magnetic,
            density,
            config.geometry,
            ray_config,
            ion_mass_kg=config.gas.ion_mass_kg,
        ).trace(source, Ip_A=Ip_A, t_s=args.t_s)
        result = analyze_direct_xb(
            source,
            incident_ray,
            magnetic,
            density,
            Ip_A=Ip_A,
            t_s=args.t_s,
        )
        payload = result.to_dict(include_path=not args.no_path)
        payload["incident_X_ray"] = incident_ray.to_dict(include_path=not args.no_path)
        payload["density_source"] = density_source
        payload["equilibrium_model"] = "none; MagneticFieldProvider only"
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "solve-fp":
        from .kinetic_workflow import run_kinetic

        config = load_config(args.config)
        if config.kinetic.backend == "open_field_line_1d2v":
            from .spatial_kinetic_workflow import run_spatial_kinetic

            result = run_spatial_kinetic(config)
        else:
            result = run_kinetic(config)
        if args.distribution_output is not None:
            result.save_distribution(args.distribution_output)
        payload = result.summary()
        payload["kinetic_switch_enabled"] = config.kinetic.enabled
        payload["explicit_command_runs_when_disabled"] = True
        payload["distribution_output"] = (
            str(args.distribution_output) if args.distribution_output is not None else None
        )
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "threshold-scan":
        from .reporting import write_threshold_scan_pdf
        from .threshold_scan import run_threshold_scan

        config = load_config(args.config)
        payload = run_threshold_scan(config)
        output_path = args.output or (Path(config.outputs.directory) / "threshold_scan.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        pdf_path = args.pdf_output
        if pdf_path is None and config.threshold_scan.write_pdf_report:
            pdf_path = Path(config.outputs.directory) / config.threshold_scan.pdf_report_filename
        if pdf_path is not None:
            write_threshold_scan_pdf(payload, pdf_path)
        print(rendered)
    elif args.command == "aspect-ratio-scan":
        from .aspect_ratio_scan import run_aspect_ratio_scan

        config = load_config(args.config)
        if not config.aspect_ratio_scan.enabled:
            raise ValueError(
                "[aspect_ratio_scan] enabled 必须为 'enabled' 才运行；请先编辑输入文件"
            )
        payload = run_aspect_ratio_scan(config)
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        output_path = args.output or (Path(config.outputs.directory) / "aspect_ratio_scan.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "scan":
        from .physics.fieldline import FieldLineTracer
        from .physics.fieldline_topology import FieldLineScanConfig, FieldLineTopologyDetector, FieldLineTopologyScanner
        from .physics.magnetic import make_magnetic_field

        config = load_config(args.config)
        field = make_magnetic_field(config.geometry)
        tracer = FieldLineTracer(
            field,
            geometry=config.geometry,
            step_m=args.step_m,
            max_length_m=args.max_length_m,
            closure_tolerance_m=args.closure_tolerance_m,
            max_steps=args.max_steps,
        )
        scan_config = FieldLineScanConfig(
            radial_seeds=args.radial_seeds,
            poloidal_seeds=args.poloidal_seeds,
            area_weighted=not args.no_area_weighting,
            direction=args.direction,
            invalid_policy=args.invalid_policy,
            retain_traces=not args.no_traces,
        )
        scanner = FieldLineTopologyScanner(field, tracer, config=scan_config)
        Ip_A = config.initial.Ip_A if args.Ip_A is None else args.Ip_A
        scan_result = scanner.scan(Ip_A=Ip_A, t_s=args.t_s)
        payload = scan_result.to_dict(
            include_samples=not args.no_samples,
            include_traces=not args.no_traces,
            include_paths=not args.no_path,
        )
        if args.include_topology:
            detector = FieldLineTopologyDetector(
                scanner,
                closed_fraction_threshold=args.closed_fraction_threshold,
            )
            topology = detector.topology_from_scan(scan_result)
            payload["topology"] = {
                "closed_fraction": topology.closed_fraction,
                "candidate_radius_m": topology.candidate_radius_m,
                "Bp_edge_T": topology.Bp_edge_T,
                "threshold_T": topology.threshold_T,
                "has_closed_surface": topology.has_closed_surface,
                "connection_length_m": topology.connection_length_m,
                "method": topology.method,
                "confidence": topology.confidence,
                "reason": topology.reason,
                "field_configuration": topology.field_configuration,
                "mirror_ratio": topology.mirror_ratio,
                "trapped_fraction": topology.trapped_fraction,
                "decay_index": topology.decay_index,
                "connection_length_factor": topology.connection_length_factor,
                "loss_multiplier": topology.loss_multiplier,
            }
        rendered = json.dumps(_json_finite(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "postprocess":
        summary_path = args.run_directory / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"找不到 {summary_path}")
        print(summary_path.read_text(encoding="utf-8"))
    elif args.command == "compare":
        from .comparison import load_device_matrix, run_device_matrix

        cases = load_device_matrix(args.matrix)
        summaries = run_device_matrix(cases)
        payload = [summary.as_dict() for summary in summaries]
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "ensemble":
        from .ensemble import run_ensemble

        config = load_config(args.config)
        ranges = json.loads(args.parameters.read_text(encoding="utf-8"))
        result = run_ensemble(config, ranges, samples=args.samples, success_event=args.event)
        payload = result.as_dict()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "ablate":
        from .ablation import run_ablation

        config = load_config(args.config)
        payload = run_ablation(config)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
