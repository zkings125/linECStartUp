"""Bidirectional global/open-field-line 1D-2V fixed-point coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import SimulationConfig
from ..solver import RunResult
from .fokker_planck import KineticState2V, RelativisticMomentumGrid, VelocityGrid
from .fullwave_dql import load_open_fieldline_npz, load_spatial_fullwave_dql_npz
from .spatial_fokker_planck import OpenFieldLineKinetic1D2V, SpatialKineticState
from .time_dependent_kinetic import TimeSeriesKineticCurrentDrive
from .wall_sheath import QuasineutralAmbipolarSheath1D


@dataclass(frozen=True)
class SpatialKineticClosureTerms:
    ionization_rate_m3_s: float
    electron_wall_loss_rate_m3_s: float
    absorbed_rf_power_W: float
    electron_density_m3: float
    ion_density_m3: float
    neutral_density_m3: float


class TimeSeriesSpatialKineticClosure:
    def __init__(
        self,
        times_s: np.ndarray,
        ionization_rate_m3_s: np.ndarray,
        electron_wall_loss_rate_m3_s: np.ndarray,
        absorbed_rf_power_W: np.ndarray,
        electron_density_m3: np.ndarray,
        ion_density_m3: np.ndarray,
        neutral_density_m3: np.ndarray,
    ) -> None:
        self.times_s = np.asarray(times_s, dtype=float)
        arrays = (
            ionization_rate_m3_s,
            electron_wall_loss_rate_m3_s,
            absorbed_rf_power_W,
            electron_density_m3,
            ion_density_m3,
            neutral_density_m3,
        )
        self.arrays = tuple(np.asarray(item, dtype=float) for item in arrays)
        if any(item.shape != self.times_s.shape for item in self.arrays):
            raise ValueError("all spatial kinetic closure trajectories must share one time grid")

    def evaluate(self, t_s: float) -> SpatialKineticClosureTerms:
        values = [
            float(np.interp(t_s, self.times_s, item, left=item[0], right=item[-1]))
            for item in self.arrays
        ]
        return SpatialKineticClosureTerms(*values)


@dataclass(frozen=True)
class SpatialTimeDependentKineticHistory:
    times_s: np.ndarray
    incremental_current_A: np.ndarray
    powered_current_A: np.ndarray
    zero_dql_current_A: np.ndarray
    kinetic_temperature_eV: np.ndarray
    pressure_parallel_Pa: np.ndarray
    pressure_perp_Pa: np.ndarray
    snapshot_times_s: np.ndarray
    snapshot_f_e: np.ndarray
    snapshot_f_zero_dql: np.ndarray
    v_parallel_m_s: np.ndarray
    v_perp_m_s: np.ndarray
    diffusion_source: str
    fixed_point_iterations: int
    fixed_point_residual_A: tuple[float, ...]
    coordinate_system: str
    ionization_rate_m3_s: np.ndarray
    electron_wall_loss_rate_m3_s: np.ndarray
    absorbed_rf_power_W: np.ndarray
    electron_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    neutral_density_m3: np.ndarray
    coupling_definition: str = (
        "global trajectory -> E_parallel, collisions and requested RF power; "
        "1D-2V -> replacement ionization, absorption, wall loss and current moment"
    )

    def closure(self) -> TimeSeriesSpatialKineticClosure:
        return TimeSeriesSpatialKineticClosure(
            self.times_s,
            self.ionization_rate_m3_s,
            self.electron_wall_loss_rate_m3_s,
            self.absorbed_rf_power_W,
            self.electron_density_m3,
            self.ion_density_m3,
            self.neutral_density_m3,
        )


def _volume_average(values: np.ndarray, cell_volume_m3: np.ndarray) -> float:
    return float(np.sum(values * cell_volume_m3) / np.sum(cell_volume_m3))


def _density_profile(solver: OpenFieldLineKinetic1D2V, state: SpatialKineticState) -> np.ndarray:
    return np.sum(
        state.f_e * solver.velocity.phase_volume_weights_m3_s3[None, :, :],
        axis=(1, 2),
    )


def advance_spatial_kinetic_trajectory(
    config: SimulationConfig,
    result: RunResult,
    *,
    fixed_point_iterations: int = 0,
    fixed_point_residual_A: tuple[float, ...] = (),
) -> SpatialTimeDependentKineticHistory:
    kinetic = config.kinetic
    assert kinetic.spatial_fieldline_file is not None
    spatial = load_open_fieldline_npz(kinetic.spatial_fieldline_file)
    grid = (
        RelativisticMomentumGrid.uniform(
            kinetic.n_v_parallel, kinetic.n_v_perp,
            kinetic.u_parallel_max, kinetic.u_perp_max,
        )
        if kinetic.coordinate_system == "relativistic_momentum"
        else VelocityGrid.uniform(
            kinetic.n_v_parallel, kinetic.n_v_perp,
            kinetic.v_parallel_max_m_s, kinetic.v_perp_max_m_s,
        )
    )
    solver = OpenFieldLineKinetic1D2V(spatial, grid, streaming_scheme="implicit_upwind")
    diffusion = (
        None
        if kinetic.spatial_dql_file is None
        else load_spatial_fullwave_dql_npz(kinetic.spatial_dql_file, spatial, grid)
    )
    first = result.diagnostics[0]
    powered = solver.maxwellian_state(
        max(float(first.get("ne_m3", config.initial.ne_m3)), 1.0),
        max(float(first.get("Te_eV", config.initial.Te_eV)), 1.0e-3),
        neutral_density_m3=max(float(first.get("nn_m3", config.initial.nn_m3 or 0.0)), 0.0),
        ion_density_m3=max(float(first.get("ni_m3", config.initial.ni_m3)), 0.0),
        drift_v_parallel_m_s=kinetic.initial_drift_v_parallel_m_s,
        time_s=float(result.times_s[0]),
    )
    baseline = SpatialKineticState(
        powered.f_e.copy(), powered.neutral_density_m3.copy(), powered.ion_density_m3.copy(), powered.time_s
    )
    sheath = (
        QuasineutralAmbipolarSheath1D(
            ion_mass_kg=config.gas.ion_mass_kg,
            secondary_electron_yield=config.gas.secondary_emission,
        )
        if kinetic.spatial_sheath_model == "quasineutral_ambipolar"
        else None
    )
    count = len(result.times_s)
    arrays = {name: np.zeros(count) for name in (
        "powered_current", "baseline_current", "temperature", "pressure_parallel",
        "pressure_perp", "ionization", "wall_loss", "absorbed", "ne", "ni", "nn",
    )}
    snapshot_indices = set(np.unique(np.linspace(0, count - 1, kinetic.time_dependent_snapshot_count, dtype=int)))
    powered_snapshots: list[np.ndarray] = []
    baseline_snapshots: list[np.ndarray] = []
    snapshot_times: list[float] = []
    diffusion_source = "zero"

    def record(index: int, state: SpatialKineticState, current: float, baseline_current: float) -> None:
        density = _density_profile(solver, state)
        arrays["powered_current"][index] = current
        arrays["baseline_current"][index] = baseline_current
        arrays["ne"][index] = _volume_average(density, spatial.cell_volume_m3)
        arrays["ni"][index] = _volume_average(state.ion_density_m3, spatial.cell_volume_m3)
        arrays["nn"][index] = _volume_average(state.neutral_density_m3, spatial.cell_volume_m3)
        moments = [
            local.moments(KineticState2V(state.f_e[cell], grid, state.time_s))
            for cell, local in enumerate(solver._local_solvers)
        ]
        arrays["temperature"][index] = float(np.mean([item.temperature_eV for item in moments]))
        arrays["pressure_parallel"][index] = float(np.mean([item.pressure_parallel_Pa for item in moments]))
        arrays["pressure_perp"][index] = float(np.mean([item.pressure_perp_Pa for item in moments]))

    record(0, powered, 0.0, 0.0)
    if 0 in snapshot_indices:
        powered_snapshots.append(powered.f_e.copy())
        baseline_snapshots.append(baseline.f_e.copy())
        snapshot_times.append(float(result.times_s[0]))
    volume = float(np.sum(spatial.cell_volume_m3))
    for index in range(count - 1):
        dt_s = float(result.times_s[index + 1] - result.times_s[index])
        left, right = result.diagnostics[index], result.diagnostics[index + 1]

        def midpoint(key: str, default: float = 0.0) -> float:
            return 0.5 * (float(left.get(key, default)) + float(right.get(key, default)))

        requested_power = max(midpoint("rf_power_absorbed_W"), 0.0)
        if requested_power > 0.0 and diffusion is None:
            raise ValueError(
                "open_field_line_1d2v time feedback has non-zero RF power but no spatial_dql_file"
            )
        electric_field = midpoint("E_loop_V_m")
        collision_frequency = max(midpoint("nu_en_s") + midpoint("nu_ei_s"), 0.0)
        collision_temperature = max(midpoint("Te_eV", config.initial.Te_eV), 1.0e-3)
        ionization_rate = (
            max(midpoint("K_ion_m3_s"), 0.0)
            if kinetic.spatial_ionization_from_global_rates else 0.0
        )
        powered, powered_diag = solver.advance(
            powered,
            dt_s,
            E_parallel_V_m=electric_field,
            collision_frequency_s=collision_frequency,
            collision_temperature_eV=collision_temperature,
            ionization_rate_coefficient_m3_s=ionization_rate,
            rf_diffusion=diffusion,
            target_absorbed_power_W=(requested_power if kinetic.spatial_use_global_rf_power else None),
            sheath_closure=sheath,
            ion_temperature_eV=config.initial.Ti_eV,
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
        )
        baseline, baseline_diag = solver.advance(
            baseline,
            dt_s,
            E_parallel_V_m=electric_field,
            collision_frequency_s=collision_frequency,
            collision_temperature_eV=collision_temperature,
            ionization_rate_coefficient_m3_s=ionization_rate,
            rf_diffusion=None,
            sheath_closure=sheath,
            ion_temperature_eV=config.initial.Ti_eV,
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
        )
        diffusion_source = "zero" if diffusion is None else diffusion.source
        arrays["ionization"][index + 1] = powered_diag.ionization_particles / (dt_s * volume)
        arrays["wall_loss"][index + 1] = powered_diag.wall_particle_loss / (dt_s * volume)
        arrays["absorbed"][index + 1] = powered_diag.absorbed_rf_power_W
        record(
            index + 1,
            powered,
            powered_diag.current_parallel_A,
            baseline_diag.current_parallel_A,
        )
        if index + 1 in snapshot_indices:
            powered_snapshots.append(powered.f_e.copy())
            baseline_snapshots.append(baseline.f_e.copy())
            snapshot_times.append(float(result.times_s[index + 1]))
    for name in ("ionization", "wall_loss", "absorbed"):
        if count > 1:
            arrays[name][0] = arrays[name][1]
    return SpatialTimeDependentKineticHistory(
        np.asarray(result.times_s),
        arrays["powered_current"] - arrays["baseline_current"],
        arrays["powered_current"], arrays["baseline_current"], arrays["temperature"],
        arrays["pressure_parallel"], arrays["pressure_perp"], np.asarray(snapshot_times),
        np.asarray(powered_snapshots), np.asarray(baseline_snapshots), grid.v_parallel_m_s.copy(),
        grid.v_perp_m_s.copy(), diffusion_source, fixed_point_iterations,
        fixed_point_residual_A, kinetic.coordinate_system, arrays["ionization"],
        arrays["wall_loss"], arrays["absorbed"], arrays["ne"], arrays["ni"], arrays["nn"],
    )


def integrate_bidirectional_spatial_kinetic(
    config: SimulationConfig,
    integrate_global: Callable[[SimulationConfig, object], RunResult],
) -> RunResult:
    from .global_model import GlobalStartupModel

    cadence = config.outputs.cadence_s or config.numerics.dt_s
    count = max(1, int(np.ceil((config.numerics.t_end_s - config.numerics.t_start_s) / cadence)))
    times = np.linspace(config.numerics.t_start_s, config.numerics.t_end_s, count + 1)
    current = np.zeros_like(times)
    closure = None
    residuals: list[float] = []
    iterations = 0
    for iteration in range(config.kinetic.time_dependent_iterations):
        drive = TimeSeriesKineticCurrentDrive(config, times, current)
        result = integrate_global(
            config,
            GlobalStartupModel(config, current_drive=drive, kinetic_closure=closure),
        )
        history = advance_spatial_kinetic_trajectory(config, result)
        proposed = history.incremental_current_A
        updated = (
            (1.0 - config.kinetic.time_dependent_relaxation) * current
            + config.kinetic.time_dependent_relaxation * proposed
        )
        residual = float(np.max(np.abs(updated - current)))
        residuals.append(residual)
        current = updated
        closure = history.closure()
        iterations = iteration + 1
        if residual <= config.kinetic.time_dependent_tolerance_A:
            break
    final_drive = TimeSeriesKineticCurrentDrive(config, times, current)
    final_result = integrate_global(
        config,
        GlobalStartupModel(config, current_drive=final_drive, kinetic_closure=closure),
    )
    final_history = advance_spatial_kinetic_trajectory(
        config,
        final_result,
        fixed_point_iterations=iterations,
        fixed_point_residual_A=tuple(residuals),
    )
    final_result.kinetic_history = final_history
    return final_result


__all__ = [
    "SpatialKineticClosureTerms",
    "SpatialTimeDependentKineticHistory",
    "TimeSeriesSpatialKineticClosure",
    "advance_spatial_kinetic_trajectory",
    "integrate_bidirectional_spatial_kinetic",
]
