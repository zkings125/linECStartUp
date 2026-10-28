"""Time-resolved, bidirectional coupling between the global model and 0D-2V F-P.

The implicit global ODE solver may evaluate its RHS repeatedly and out of time
order, so the kinetic distribution is never mutated inside an RHS call.  An
outer fixed-point loop instead alternates a complete global trajectory and a
causal F-P advance along that trajectory.  The resulting distribution-moment
current is interpolated back into the next global L-R solve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..config import SimulationConfig
from ..constants import PI
from ..kinetic_workflow import make_power_normalized_diffusion
from ..solver import RunResult
from ..state import GlobalState
from .current_drive import CurrentComponents, CurrentDriveModel
from .fokker_planck import (
    KineticState2V,
    QuasilinearDiffusion2V,
    ReducedFokkerPlanck2V,
    RelativisticFokkerPlanck2V,
    RelativisticMomentumGrid,
    VelocityGrid,
)
from .magnetic import MagneticConfigurationState
from .open_field_loss import LossTimes
from .rf import RFSourceTerms


@dataclass(frozen=True)
class TimeDependentKineticHistory:
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
    coupling_definition: str = (
        "global trajectory -> ne, Te, E_parallel, collisions, losses and absorbed RF power; "
        "F-P distribution moment -> global L-R current target"
    )


class TimeSeriesKineticCurrentDrive:
    """Current-drive provider backed by a prescribed kinetic-current trajectory."""

    def __init__(
        self,
        config: SimulationConfig,
        times_s: np.ndarray,
        incremental_current_A: np.ndarray,
    ) -> None:
        self.config = config
        self.times_s = np.asarray(times_s, dtype=float)
        self.incremental_current_A = np.asarray(incremental_current_A, dtype=float)
        if self.times_s.shape != self.incremental_current_A.shape:
            raise ValueError("kinetic time/current trajectories must have the same shape")
        self.baseline = CurrentDriveModel(config.geometry, config.rf)

    def evaluate(
        self,
        state: GlobalState,
        rf_terms: RFSourceTerms,
        magnetic_state: MagneticConfigurationState | None = None,
        loss_times: LossTimes | None = None,
        vertical_field_T: float | None = None,
        t_s: float | None = None,
    ) -> CurrentComponents:
        empirical = self.baseline.evaluate(
            state,
            rf_terms,
            magnetic_state=magnetic_state,
            loss_times=loss_times,
            vertical_field_T=vertical_field_T,
            t_s=t_s,
        )
        sample_time = self.times_s[0] if t_s is None else float(t_s)
        kinetic = float(
            np.interp(
                sample_time,
                self.times_s,
                self.incremental_current_A,
                left=self.incremental_current_A[0],
                right=self.incremental_current_A[-1],
            )
        )
        direct = kinetic if self.config.kinetic.feedback_replace_direct_rf else empirical.I_rf_A
        tail = 0.0 if self.config.kinetic.feedback_replace_hot_tail else empirical.I_orbit_A
        return CurrentComponents(
            direct,
            tail,
            empirical.I_vfc_A,
            I_kinetic_A=kinetic,
            current_drive_model="fp_time_dependent_fixed_point",
        )


def _rescale_density(state: KineticState2V, density_m3: float) -> KineticState2V:
    weights = state.grid.phase_volume_weights_m3_s3
    current = float(np.sum(np.maximum(state.f_e, 0.0) * weights))
    if current <= 0.0:
        raise FloatingPointError("cannot rescale an empty kinetic distribution")
    return KineticState2V(
        np.maximum(state.f_e, 0.0) * max(float(density_m3), 0.0) / current,
        state.grid,
        state.time_s,
    )


def advance_kinetic_trajectory(
    config: SimulationConfig,
    result: RunResult,
    *,
    fixed_point_iterations: int = 0,
    fixed_point_residual_A: tuple[float, ...] = (),
) -> TimeDependentKineticHistory:
    """Advance powered and zero-D_ql distributions along one global trajectory."""

    kinetic = config.kinetic
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
    area_m2 = config.geometry.resolved_volume_m3 / (2.0 * PI * config.geometry.R0_m)
    solver = (
        RelativisticFokkerPlanck2V(grid, cross_section_area_m2=area_m2)
        if kinetic.coordinate_system == "relativistic_momentum"
        else ReducedFokkerPlanck2V(grid, cross_section_area_m2=area_m2)
    )
    first = result.diagnostics[0]
    powered = solver.maxwellian_state(
        max(float(first.get("ne_m3", config.initial.ne_m3)), 1.0),
        max(float(first.get("Te_eV", config.initial.Te_eV)), 1.0e-3),
        drift_v_parallel_m_s=kinetic.initial_drift_v_parallel_m_s,
        time_s=float(result.times_s[0]),
    )
    baseline = KineticState2V(powered.f_e.copy(), grid, powered.time_s)
    count = len(result.times_s)
    powered_current = np.zeros(count)
    baseline_current = np.zeros(count)
    temperatures = np.zeros(count)
    pressure_parallel = np.zeros(count)
    pressure_perp = np.zeros(count)
    initial_moments = solver.moments(powered)
    powered_current[0] = initial_moments.current_parallel_A
    baseline_current[0] = initial_moments.current_parallel_A
    temperatures[0] = initial_moments.temperature_eV
    pressure_parallel[0] = initial_moments.pressure_parallel_Pa
    pressure_perp[0] = initial_moments.pressure_perp_Pa
    snapshot_indices = np.unique(
        np.linspace(0, count - 1, kinetic.time_dependent_snapshot_count, dtype=int)
    )
    powered_snapshots: list[np.ndarray] = []
    baseline_snapshots: list[np.ndarray] = []
    snapshot_times: list[float] = []
    snapshot_set = set(int(item) for item in snapshot_indices)
    diffusion_source = "disabled"
    if 0 in snapshot_set:
        powered_snapshots.append(powered.f_e.copy())
        baseline_snapshots.append(baseline.f_e.copy())
        snapshot_times.append(float(result.times_s[0]))

    for index in range(count - 1):
        dt_s = float(result.times_s[index + 1] - result.times_s[index])
        left = result.diagnostics[index]
        right = result.diagnostics[index + 1]

        def midpoint(key: str, default: float = 0.0) -> float:
            return 0.5 * (
                float(left.get(key, default)) + float(right.get(key, default))
            )

        target_density = max(midpoint("ne_m3", 1.0), 1.0)
        powered = _rescale_density(powered, target_density)
        baseline = _rescale_density(baseline, target_density)
        absorbed_power = max(midpoint("rf_power_absorbed_W"), 0.0)
        diffusion, _, _ = make_power_normalized_diffusion(
            config, grid, absorbed_power_W=absorbed_power
        )
        if kinetic.orbit_model == "trapped_passing_bounce_average":
            from .orbit_average import apply_bounce_weight_to_diffusion, classify_pitch

            vp, vt = grid.mesh()
            classification = classify_pitch(
                vp, vt,
                B_min_T=kinetic.orbit_B_min_T,
                B_max_T=kinetic.orbit_B_max_T,
                weights=grid.phase_volume_weights_m3_s3 * powered.f_e,
            )
            diffusion = apply_bounce_weight_to_diffusion(diffusion, classification).validated(grid)
        diffusion_source = diffusion.source
        collision_frequency = max(midpoint("nu_en_s") + midpoint("nu_ei_s"), 0.0)
        collision_temperature = max(midpoint("Te_eV", config.initial.Te_eV), 1.0e-3)
        electric_field = midpoint("E_loop_V_m")
        loss_frequency = max(midpoint("nu_loss_s"), 0.0)
        powered, _ = solver.advance(
            powered,
            dt_s,
            E_parallel_V_m=electric_field,
            collision_frequency_s=collision_frequency,
            collision_temperature_eV=collision_temperature,
            rf_diffusion=diffusion,
            loss_frequency_s=loss_frequency,
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
        )
        baseline, _ = solver.advance(
            baseline,
            dt_s,
            E_parallel_V_m=electric_field,
            collision_frequency_s=collision_frequency,
            collision_temperature_eV=collision_temperature,
            rf_diffusion=QuasilinearDiffusion2V.zeros(grid),
            loss_frequency_s=loss_frequency,
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
        )
        next_density = max(float(right.get("ne_m3", target_density)), 1.0)
        powered = _rescale_density(powered, next_density)
        baseline = _rescale_density(baseline, next_density)
        powered = replace(powered, time_s=float(result.times_s[index + 1]))
        baseline = replace(baseline, time_s=float(result.times_s[index + 1]))
        powered_moments = solver.moments(powered)
        baseline_moments = solver.moments(baseline)
        powered_current[index + 1] = powered_moments.current_parallel_A
        baseline_current[index + 1] = baseline_moments.current_parallel_A
        temperatures[index + 1] = powered_moments.temperature_eV
        pressure_parallel[index + 1] = powered_moments.pressure_parallel_Pa
        pressure_perp[index + 1] = powered_moments.pressure_perp_Pa
        if index + 1 in snapshot_set:
            powered_snapshots.append(powered.f_e.copy())
            baseline_snapshots.append(baseline.f_e.copy())
            snapshot_times.append(float(result.times_s[index + 1]))

    return TimeDependentKineticHistory(
        times_s=np.asarray(result.times_s, dtype=float),
        incremental_current_A=powered_current - baseline_current,
        powered_current_A=powered_current,
        zero_dql_current_A=baseline_current,
        kinetic_temperature_eV=temperatures,
        pressure_parallel_Pa=pressure_parallel,
        pressure_perp_Pa=pressure_perp,
        snapshot_times_s=np.asarray(snapshot_times),
        snapshot_f_e=np.asarray(powered_snapshots),
        snapshot_f_zero_dql=np.asarray(baseline_snapshots),
        v_parallel_m_s=grid.v_parallel_m_s.copy(),
        v_perp_m_s=grid.v_perp_m_s.copy(),
        diffusion_source=diffusion_source,
        fixed_point_iterations=fixed_point_iterations,
        fixed_point_residual_A=fixed_point_residual_A,
        coordinate_system=kinetic.coordinate_system,
    )


def integrate_bidirectional_kinetic(
    config: SimulationConfig,
    integrate_global: Callable[[SimulationConfig, object], RunResult],
) -> RunResult:
    """Solve the staggered global/F-P fixed point and return the global result."""

    from .global_model import GlobalStartupModel

    cadence = config.outputs.cadence_s or config.numerics.dt_s
    start = float(config.numerics.t_start_s)
    end = float(config.numerics.t_end_s)
    count = max(1, int(np.ceil((end - start) / cadence)))
    times = np.linspace(start, end, count + 1)
    current = np.zeros_like(times)
    residuals: list[float] = []
    iterations = 0
    for iteration in range(config.kinetic.time_dependent_iterations):
        drive = TimeSeriesKineticCurrentDrive(config, times, current)
        result = integrate_global(config, GlobalStartupModel(config, current_drive=drive))
        history = advance_kinetic_trajectory(config, result)
        proposed = history.incremental_current_A
        updated = (
            (1.0 - config.kinetic.time_dependent_relaxation) * current
            + config.kinetic.time_dependent_relaxation * proposed
        )
        residual = float(np.max(np.abs(updated - current)))
        residuals.append(residual)
        current = updated
        iterations = iteration + 1
        if residual <= config.kinetic.time_dependent_tolerance_A:
            break

    final_drive = TimeSeriesKineticCurrentDrive(config, times, current)
    final_result = integrate_global(config, GlobalStartupModel(config, current_drive=final_drive))
    final_history = advance_kinetic_trajectory(
        config,
        final_result,
        fixed_point_iterations=iterations,
        fixed_point_residual_A=tuple(residuals),
    )
    final_result.kinetic_history = final_history
    return final_result
