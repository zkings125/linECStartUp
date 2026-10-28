"""运行结果写出：NetCDF、CSV、JSON 和可复现实验清单。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import __version__
from .config import SimulationConfig, dump_resolved_config
from .diagnostics import event_to_dict
from .solver import RunResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def write_run(result: RunResult, config: SimulationConfig, directory: str | Path | None = None) -> Path:
    out = Path(directory or config.outputs.directory)
    out.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, out / "config.resolved.json")
    config_bytes = (out / "config.resolved.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    rows = result.diagnostics
    keys = sorted({key for row in rows for key in row})
    data_vars = {}
    for key in keys:
        values = [row.get(key, np.nan) for row in rows]
        if key in {"stage", "field_configuration"} or any(isinstance(value, str) for value in values):
            # NetCDF 对字符串使用 object/U 类型；自定义 provider 也可以
            # 返回额外的文本标签。
            data_vars[key] = ("time", np.asarray([str(value) for value in values], dtype=str))
        else:
            data_vars[key] = ("time", np.asarray(values, dtype=float))
    ds = xr.Dataset(data_vars, coords={"time": result.times_s})
    ds["time"].attrs.update({"units": "s", "long_name": "simulation time"})
    units = {
        "electron_energy_J_m3": "J m^-3",
        "hot_energy_J_m3": "J m^-3",
        "ne_m3": "m^-3",
        "ni_m3": "m^-3",
        "nn_m3": "m^-3",
        "Te_eV": "eV",
        "Ti_eV": "eV",
        "Ip_A": "A",
        "n_hot_m3": "m^-3",
        "hot_mean_energy_eV": "eV",
        "E_loop_V_m": "V m^-1",
        "loop_voltage_V": "V",
        "effective_loop_voltage_V": "V",
        "EoverN_Td": "Td",
        "neutral_pressure_Pa": "Pa",
        "target_pressure_Pa": "Pa",
        "pL_Pa_m": "Pa m",
        "B_vertical_T": "T",
        "B_R_T": "T",
        "B_Z_T": "T",
        "B_phi_T": "T",
        "B_magnitude_T": "T",
        "K_ion_m3_s": "m^3 s^-1",
        "K_exc_m3_s": "m^3 s^-1",
        "K_ion_hot_m3_s": "m^3 s^-1",
        "nu_ion_s": "s^-1",
        "nu_loss_s": "s^-1",
        "net_growth_s": "s^-1",
        "townsend_alpha_eff_m": "m^-1",
        "nu_en_s": "s^-1",
        "nu_ei_s": "s^-1",
        "Dreicer_ratio": "1",
        "omega_ce_over_nu": "1",
        "lambda_D_over_a": "1",
        "mean_free_path_over_a": "1",
        "ionization_fraction": "1",
        "rf_accessibility": "1",
        "omega_ce_over_omega_rf": "1",
        "P_ohm_W_m3": "W m^-3",
        "P_rf_W_m3": "W m^-3",
        "P_rf_bulk_W_m3": "W m^-3",
        "P_ion_W_m3": "W m^-3",
        "P_exc_W_m3": "W m^-3",
        "P_rad_W_m3": "W m^-3",
        "P_rad_base_W_m3": "W m^-3",
        "P_rad_species_W_m3": "W m^-3",
        "P_ion_species_W_m3": "W m^-3",
        "P_wall_W_m3": "W m^-3",
        "P_particle_W_m3": "W m^-3",
        "P_loss_W_m3": "W m^-3",
        "P_balance_W_m3": "W m^-3",
        "sigma_S_m": "S m^-1",
        "R_plasma_ohm": "ohm",
        "I_ohmic_A": "A",
        "I_bulk_A": "A",
        "I_rf_A": "A",
        "I_kinetic_A": "A",
        "I_orbit_A": "A",
        "I_tail_A": "A",
        "I_tpc_A": "A",
        "I_vfc_A": "A",
        "I_vtf_A": "A",
        "I_vertical_charge_A": "A",
        "I_drive_A": "A",
        "I_component_sum_A": "A",
        "dIp_dt_A_s": "A s^-1",
        "rf_power_absorbed_W": "W",
        "closed_fraction": "1",
        "topology_confidence": "1",
        "candidate_radius_m": "m",
        "Bp_edge_T": "T",
        "connection_length_effective_m": "m",
        "connection_length_factor": "1",
        "field_loss_multiplier": "1",
        "loss_configuration_multiplier": "1",
        "mirror_ratio": "1",
        "trapped_fraction": "1",
        "decay_index": "1",
        "vertical_alignment": "1",
        "mirror_confinement_factor": "1",
        "tail_confinement_metric": "1",
        "tpc_self_field_scale": "1",
        "tpc_mirror_ratio": "1",
        "tpc_trapped_fraction": "1",
        "tpc_decay_index": "1",
        "tpc_B_phi_midradius_T": "T",
        "tpc_mirror_shape_midradius": "1",
        "vfc_vertical_field_scale": "1",
        "vfc_decay_index": "1",
        "vfc_connection_length_factor": "1",
        "vfc_vertical_field_edge_T": "T",
        "vertical_gradient_T_per_m": "T m^-1",
        "Bp_edge_base_T": "T",
        "coil_Bp_edge_T": "T",
        "coil_component_count": "1",
        "coil_current_total_A": "A",
        "coil_B_R_axis_T": "T",
        "coil_B_Z_axis_T": "T",
        "coil_B_phi_axis_T": "T",
        "loop_count": "1",
        "loop_current_total_A": "A",
        "loop_B_R_axis_T": "T",
        "loop_B_Z_axis_T": "T",
        "loop_B_phi_axis_T": "T",
        "loop_Bp_edge_T": "T",
        "loop_edge_singularity_count": "1",
        "S_ion_m3_s": "m^-3 s^-1",
        "S_ion_bulk_m3_s": "m^-3 s^-1",
        "S_ion_hot_m3_s": "m^-3 s^-1",
        "S_rec_m3_s": "m^-3 s^-1",
        "S_att_m3_s": "m^-3 s^-1",
        "S_e_wall_m3_s": "m^-3 s^-1",
        "S_e_wall_free_m3_s": "m^-3 s^-1",
        "S_e_wall_target_m3_s": "m^-3 s^-1",
        "S_positive_charge_wall_m3_s": "m^-3 s^-1",
        "S_impurity_charge_wall_m3_s": "m^-3 s^-1",
        "S_i_wall_m3_s": "m^-3 s^-1",
        "S_secondary_m3_s": "m^-3 s^-1",
        "S_seed_m3_s": "m^-3 s^-1",
        "S_e_species_m3_s": "m^-3 s^-1",
        "S_fuel_recycle_m3_s": "m^-3 s^-1",
        "electron_density_quasineutral_m3": "m^-3",
        "charge_balance_residual_m3": "m^-3",
        "charge_balance_relative": "1",
        "charge_continuity_residual_m3_s": "m^-3 s^-1",
        "charge_continuity_relative": "1",
        "ambipolar_potential_V": "V",
        "ambipolar_potential_over_Te": "1",
        "ambipolar_flux_limited": "1",
        "species_particle_balance_residual_m3_s": "m^-3 s^-1",
        "species_particle_balance_relative": "1",
        "Z_eff": "1",
        "Z_eff_used": "1",
        "wall_incident_energy_eV": "eV",
    }
    for key, unit in units.items():
        if key in ds:
            ds[key].attrs["units"] = unit
    # Per-component numeric diagnostics use a dynamic index in their name.
    # Keep their units explicit without requiring a fixed number of coils in
    # the schema.
    for key in ds.data_vars:
        if key.startswith("rf_"):
            if key.endswith("_frequency_Hz"):
                ds[key].attrs["units"] = "Hz"
            elif key.endswith("_power_absorbed_W"):
                ds[key].attrs["units"] = "W"
            elif key.endswith("_power_density_W_m3") or key.endswith("_hot_power_density_W_m3"):
                ds[key].attrs["units"] = "W m^-3"
            elif key.endswith("_current_drive_A"):
                ds[key].attrs["units"] = "A"
            elif key.endswith("_critical_density_m3"):
                ds[key].attrs["units"] = "m^-3"
            elif key.endswith("_launch_R_m") or key.endswith("_launch_Z_m"):
                ds[key].attrs["units"] = "m"
            elif key.endswith("_B_launch_T"):
                ds[key].attrs["units"] = "T"
            elif key.endswith("_accessibility") or key.endswith("_pulse_factor") or key.endswith("_omega_ce_over_omega_rf") or key.endswith("_omega_ce_over_omega_rf_local"):
                ds[key].attrs["units"] = "1"
        if key.startswith("n_") and key.endswith("_m3"):
            ds[key].attrs["units"] = "m^-3"
        elif key.startswith("wall_inventory_") and key.endswith("_m2"):
            ds[key].attrs["units"] = "m^-2"
        elif (key.startswith("S_ion_") or key.startswith("S_rec_") or key.startswith("S_wall_") or key.startswith("S_pump_") or key.startswith("S_recycle_") or key.startswith("S_sputter_")) and key.endswith("_m3_s"):
            ds[key].attrs["units"] = "m^-3 s^-1"
        elif (key.startswith("P_rad_") or key.startswith("P_ion_")) and key.endswith("_W_m3"):
            ds[key].attrs["units"] = "W m^-3"
        elif key.startswith("sputtering_yield_"):
            ds[key].attrs["units"] = "1"
        elif key.startswith("wall_incident_energy_") and key.endswith("_eV"):
            ds[key].attrs["units"] = "eV"
        if key.startswith("coil_") and any(
            key.endswith(suffix)
            for suffix in ("_current_A", "_B_R_axis_T", "_B_Z_axis_T", "_B_phi_axis_T")
        ):
            ds[key].attrs["units"] = "A" if key.endswith("_current_A") else "T"
        if key.startswith("loop_"):
            if key.endswith("_current_A"):
                ds[key].attrs["units"] = "A"
            elif key.endswith("_radius_m") or key.endswith("_z0_m"):
                ds[key].attrs["units"] = "m"
            elif key.endswith("_turns"):
                ds[key].attrs["units"] = "1"
            elif key.endswith("_B_R_axis_T") or key.endswith("_B_Z_axis_T"):
                ds[key].attrs["units"] = "T"
    ds.attrs.update(
        {
            "ecstartup_version": __version__,
            "schema_version": config.schema_version,
            "model": config.model,
            "equations": "docs/theory.md",
            "topology_note": "closed_fraction is a provider-specific analytic surrogate; validate with GS/field-line tracing",
            "species_wall_note": (
                "DYON-like independent implementation; analytic atomic rates are surrogates, not ADAS"
                if config.species_wall.enabled
                else "disabled"
            ),
            # A composite provider exposes its resolved label in the time
            # diagnostics; retain that label in global metadata as well.
            "field_configuration": str(
                rows[-1].get("field_configuration", config.geometry.field_configuration)
                if rows
                else config.geometry.field_configuration
            ),
            "config_sha256": config_sha256,
            "seed": config.seed,
        }
    )
    if config.outputs.write_netcdf:
        # scipy 后端写 NetCDF3，避免 Windows 下 netCDF4 对含中文路径的兼容性问题；
        # 变量均为一维时间序列，NetCDF3 足够。二维/大文件版本可切换到 h5netcdf。
        ds.to_netcdf(out / "state.nc", engine="scipy")

    if config.outputs.write_csv:
        with (out / "diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time_s"] + keys)
            writer.writeheader()
            for t_s, row in zip(result.times_s, rows):
                payload = {"time_s": float(t_s), **{k: _json_safe(v) for k, v in row.items()}}
                writer.writerow(payload)

    (out / "events.json").write_text(
        json.dumps([event_to_dict(event) for event in result.events], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    kinetic_artifact = None
    if result.kinetic_history is not None:
        history = result.kinetic_history
        kinetic_artifact = "kinetic_time_history.npz"
        np.savez_compressed(
            out / kinetic_artifact,
            times_s=history.times_s,
            incremental_current_A=history.incremental_current_A,
            powered_current_A=history.powered_current_A,
            zero_dql_current_A=history.zero_dql_current_A,
            kinetic_temperature_eV=history.kinetic_temperature_eV,
            pressure_parallel_Pa=history.pressure_parallel_Pa,
            pressure_perp_Pa=history.pressure_perp_Pa,
            snapshot_times_s=history.snapshot_times_s,
            snapshot_f_e=history.snapshot_f_e,
            snapshot_f_zero_dql=history.snapshot_f_zero_dql,
            v_parallel_m_s=history.v_parallel_m_s,
            v_perp_m_s=history.v_perp_m_s,
            diffusion_source=np.asarray(history.diffusion_source),
            fixed_point_iterations=np.asarray(history.fixed_point_iterations),
            fixed_point_residual_A=np.asarray(history.fixed_point_residual_A),
        )
    from .reporting import write_report_artifacts

    report_artifacts = write_report_artifacts(result, config, out)
    summary = {
        "success": result.success,
        "message": result.message,
        "n_samples": len(result.times_s),
        "events": len(result.events),
        "final_stage": rows[-1].get("stage") if rows else None,
        "final": {k: _json_safe(v) for k, v in rows[-1].items()} if rows else {},
        "config_sha256": config_sha256,
        "seed": config.seed,
        "report_artifacts": report_artifacts,
        "kinetic_time_history": kinetic_artifact,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "ecstartup_version": __version__,
        "config_sha256": config_sha256,
        "seed": config.seed,
    }
    (out / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines = [
        f"success={result.success}",
        f"message={result.message}",
        f"samples={len(result.times_s)}",
        f"events={len(result.events)}",
        f"config_sha256={config_sha256}",
    ]
    (out / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return out
