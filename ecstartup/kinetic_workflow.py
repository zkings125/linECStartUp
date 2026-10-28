"""Application-layer workflow for the reduced electron Fokker--Planck backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .constants import PI
from .physics.fokker_planck import (
    FokkerPlanckStepDiagnostics,
    KineticMoments,
    KineticState2V,
    QuasilinearDiffusion2V,
    ReducedFokkerPlanck2V,
    RelativisticMomentumGrid,
    RelativisticFokkerPlanck2V,
    VelocityGrid,
)


@dataclass(frozen=True)
class KineticRunResult:
    """In-memory result with serializable summary and optional array export."""

    state: KineticState2V
    initial_state: KineticState2V
    initial_moments: KineticMoments
    final_moments: KineticMoments
    step_diagnostics: tuple[FokkerPlanckStepDiagnostics, ...]
    diffusion_source: str
    cross_section_area_m2: float
    diffusion_reference_power_W: float | None
    applied_absorbed_power_W: float | None
    diffusion_diagnostics: tuple[dict[str, Any], ...] = ()
    dql_assessment: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        balances = [abs(item.particle_balance_relative) for item in self.step_diagnostics]
        residuals = [item.linear_residual_relative for item in self.step_diagnostics]
        history_stride = max(1, int(np.ceil(len(self.step_diagnostics) / 500)))
        history_indices = list(range(0, len(self.step_diagnostics), history_stride))
        if history_indices and history_indices[-1] != len(self.step_diagnostics) - 1:
            history_indices.append(len(self.step_diagnostics) - 1)
        dt_s = (
            (self.state.time_s - self.initial_state.time_s) / len(self.step_diagnostics)
            if self.step_diagnostics
            else 0.0
        )
        relativistic = isinstance(self.state.grid, RelativisticMomentumGrid)
        grid_summary = {
            "coordinate_system": "relativistic_momentum" if relativistic else "velocity",
            "n_parallel": self.state.grid.shape[0],
            "n_perp": self.state.grid.shape[1],
            "v_parallel_min_m_s": float(self.state.grid.v_parallel_m_s[0]),
            "v_parallel_max_m_s": float(self.state.grid.v_parallel_m_s[-1]),
            "v_perp_min_m_s": float(self.state.grid.v_perp_m_s[0]),
            "v_perp_max_m_s": float(self.state.grid.v_perp_m_s[-1]),
        }
        if relativistic:
            grid_summary.update({
                "u_parallel_min": float(self.state.grid.u_parallel[0]),
                "u_parallel_max": float(self.state.grid.u_parallel[-1]),
                "u_perp_min": float(self.state.grid.u_perp[0]),
                "u_perp_max": float(self.state.grid.u_perp[-1]),
            })
        return {
            "backend": "reduced_0d2v",
            "time_s": self.state.time_s,
            "steps": len(self.step_diagnostics),
            "velocity_grid": grid_summary,
            "cross_section_area_m2": self.cross_section_area_m2,
            "initial_moments": self.initial_moments.to_dict(),
            "final_moments": self.final_moments.to_dict(),
            "max_abs_particle_balance_relative": max(balances, default=0.0),
            "max_linear_residual_relative": max(residuals, default=0.0),
            "minimum_f": min((item.minimum_f for item in self.step_diagnostics), default=0.0),
            "rf_diffusion_source": self.diffusion_source,
            "rf_diffusion_reference_power_W": self.diffusion_reference_power_W,
            "applied_absorbed_power_W": self.applied_absorbed_power_W,
            "rf_diffusion_diagnostics": list(self.diffusion_diagnostics),
            "dql_assessment": self.dql_assessment,
            "history_stride_steps": history_stride,
            "history": [
                {
                    "step": index + 1,
                    "time_s": self.initial_state.time_s + (index + 1) * dt_s,
                    "current_parallel_A": self.step_diagnostics[index].current_parallel_A,
                    "temperature_eV": self.step_diagnostics[index].temperature_eV,
                    "particle_balance_relative": self.step_diagnostics[index].particle_balance_relative,
                }
                for index in history_indices
            ],
            "current_definition": "J_parallel = q_e * integral(v_parallel*f_e*d3v)",
            "equilibrium_model": "none; no GS/EFIT or closed-flux-surface coordinate",
            "scope": (
                "reduced 0D-2V model with optional relativistic momentum coordinates, "
                "PSD off-diagonal D_ql, linearized momentum-conserving field-particle "
                "closure and pre-CFS mirror bounce surrogate; no radial guiding-centre "
                "orbit or nonlinear Rosenbluth operator"
            ),
            "quantitative_current_warning": (
                "A physical EC/EBW-driven current requires a power-normalized D_ql from a "
                "validated ray/full-wave solver and spatial magnetic projection. Synthetic "
                "Gaussian diffusion is a sensitivity test only."
            ),
        }

    def save_distribution(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "f_e": self.state.f_e,
            "f_initial": self.initial_state.f_e,
            "time_s": np.asarray(self.state.time_s),
        }
        if isinstance(self.state.grid, RelativisticMomentumGrid):
            payload.update({"u_parallel": self.state.grid.u_parallel, "u_perp": self.state.grid.u_perp})
        else:
            payload.update({"v_parallel_m_s": self.state.grid.v_parallel_m_s, "v_perp_m_s": self.state.grid.v_perp_m_s})
        np.savez_compressed(output, **payload)


def _load_external_diffusion(
    path: str | Path,
    grid: VelocityGrid,
) -> tuple[QuasilinearDiffusion2V, float | None]:
    """Load a provider-neutral diagonal diffusion map from a documented NPZ contract."""

    input_path = Path(path)
    with np.load(input_path) as archive:
        required = {"D_parallel_m2_s3", "D_perp_m2_s3"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"external diffusion NPZ missing arrays: {', '.join(missing)}")
        diffusion = QuasilinearDiffusion2V(
            archive["D_parallel_m2_s3"],
            archive["D_perp_m2_s3"],
            f"external_npz:{input_path}",
            archive["D_cross_m2_s3"] if "D_cross_m2_s3" in archive else None,
        ).validated(grid)
        reference_power = None
        for key in ("reference_absorbed_power_W", "absorbed_power_W"):
            if key in archive:
                reference_power = float(np.asarray(archive[key]).reshape(()))
                if not np.isfinite(reference_power) or reference_power <= 0.0:
                    raise ValueError(f"{key} must be a finite positive scalar")
                break
        if "v_parallel_m_s" in archive and not np.allclose(
            archive["v_parallel_m_s"], grid.v_parallel_m_s, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("external diffusion v_parallel_m_s does not match configured grid")
        if "v_perp_m_s" in archive and not np.allclose(
            archive["v_perp_m_s"], grid.v_perp_m_s, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("external diffusion v_perp_m_s does not match configured grid")
        if isinstance(grid, RelativisticMomentumGrid):
            if "u_parallel" in archive and not np.allclose(archive["u_parallel"], grid.u_parallel, rtol=1.0e-12, atol=0.0):
                raise ValueError("external diffusion u_parallel does not match configured grid")
            if "u_perp" in archive and not np.allclose(archive["u_perp"], grid.u_perp, rtol=1.0e-12, atol=0.0):
                raise ValueError("external diffusion u_perp does not match configured grid")
    return diffusion, reference_power


def make_power_normalized_diffusion(
    config: SimulationConfig,
    grid: VelocityGrid,
    *,
    diffusion_scale: float = 1.0,
    absorbed_power_W: float | None = None,
) -> tuple[QuasilinearDiffusion2V, float | None, float | None]:
    """Build D_ql and apply a traceable absorbed-power normalization.

    ``genray_npz`` requires a reference absorbed power either in the archive
    or in the configuration. Its D_ql arrays are scaled linearly by
    ``absorbed_power_W/reference_power_W``. This is a provider interface, not
    a claim that the reduced solver reproduces GENRAY's ray geometry.
    """
    kinetic = config.kinetic
    if diffusion_scale < 0.0 or not np.isfinite(diffusion_scale):
        raise ValueError("diffusion_scale must be finite and non-negative")
    if kinetic.rf_diffusion_model == "zero":
        return QuasilinearDiffusion2V.zeros(grid), None, absorbed_power_W
    reference_power = None
    if kinetic.rf_diffusion_model in {"external_npz", "genray_npz"}:
        assert kinetic.rf_diffusion_file is not None
        diffusion, archive_reference = _load_external_diffusion(
            kinetic.rf_diffusion_file, grid
        )
        if kinetic.rf_diffusion_model == "genray_npz":
            configured = kinetic.rf_diffusion_reference_power_W
            if configured is not None and archive_reference is not None and not np.isclose(
                float(configured), archive_reference, rtol=1.0e-6, atol=0.0
            ):
                raise ValueError(
                    "GENRAY D_ql reference power disagrees between config and NPZ"
                )
            reference_power = float(configured or archive_reference or 0.0)
            if reference_power <= 0.0:
                raise ValueError(
                    "genray_npz requires reference_absorbed_power_W in NPZ "
                    "or rf_diffusion_reference_power_W in config"
                )
            diffusion = QuasilinearDiffusion2V(
                diffusion.D_parallel_m2_s3,
                diffusion.D_perp_m2_s3,
                diffusion.source.replace("external_npz:", "genray_npz:"),
                diffusion.D_cross_m2_s3,
            )
    elif kinetic.rf_diffusion_model == "genray_like_internal":
        from .physics.genray_like import build_internal_genray_like_reference

        internal = build_internal_genray_like_reference(config, grid)
        diffusion = internal.diffusion
        reference_power = internal.reference_absorbed_power_W
    else:
        diffusion = QuasilinearDiffusion2V.gaussian_resonance(
        grid,
        peak_D_parallel_m2_s3=kinetic.rf_diffusion_parallel_peak_m2_s3,
        peak_D_perp_m2_s3=kinetic.rf_diffusion_perp_peak_m2_s3,
        resonant_v_parallel_m_s=kinetic.rf_resonant_v_parallel_m_s,
        resonant_v_perp_m_s=kinetic.rf_resonant_v_perp_m_s,
        sigma_v_parallel_m_s=kinetic.rf_sigma_v_parallel_m_s,
        sigma_v_perp_m_s=kinetic.rf_sigma_v_perp_m_s,
    )
    applied_power = absorbed_power_W
    scale = float(diffusion_scale)
    if absorbed_power_W is not None:
        normalization_power = (
            reference_power
            if reference_power is not None
            else float(kinetic.feedback_reference_absorbed_power_W)
        )
        scale *= max(0.0, float(absorbed_power_W)) / normalization_power
    return (
        QuasilinearDiffusion2V(
            diffusion.D_parallel_m2_s3 * scale,
            diffusion.D_perp_m2_s3 * scale,
            f"{diffusion.source};power_scale={scale:g}",
            None if diffusion.D_cross_m2_s3 is None else diffusion.D_cross_m2_s3 * scale,
        ).validated(grid),
        reference_power,
        applied_power,
    )


def run_kinetic(
    config: SimulationConfig,
    *,
    diffusion_scale: float = 1.0,
    absorbed_power_W: float | None = None,
) -> KineticRunResult:
    """Run the reduced backend at a prescribed D_ql scale.

    The explicit ``solve-fp`` command uses scale 1.  The global feedback
    provider calls this function at several scales to build a cached response
    table; it never advances the sparse FP system inside an ODE RHS call.
    """

    kinetic = config.kinetic
    if config.initial.Te_eV <= 0.0:
        raise ValueError("solve-fp requires initial.Te_eV > 0")
    if kinetic.coordinate_system == "relativistic_momentum":
        grid = RelativisticMomentumGrid.uniform(
            kinetic.n_v_parallel,
            kinetic.n_v_perp,
            kinetic.u_parallel_max,
            kinetic.u_perp_max,
        )
    else:
        grid = VelocityGrid.uniform(
            kinetic.n_v_parallel,
            kinetic.n_v_perp,
            kinetic.v_parallel_max_m_s,
            kinetic.v_perp_max_m_s,
        )
    area_m2 = config.geometry.resolved_volume_m3 / (2.0 * PI * config.geometry.R0_m)
    solver = (
        RelativisticFokkerPlanck2V(grid, cross_section_area_m2=area_m2)
        if kinetic.coordinate_system == "relativistic_momentum"
        else ReducedFokkerPlanck2V(grid, cross_section_area_m2=area_m2)
    )
    reference_density = kinetic.feedback_reference_density_m3 or config.initial.ne_m3
    state = solver.maxwellian_state(
        reference_density,
        config.initial.Te_eV,
        drift_v_parallel_m_s=kinetic.initial_drift_v_parallel_m_s,
    )
    initial_state = KineticState2V(state.f_e.copy(), state.grid, state.time_s)
    initial = solver.moments(state)
    diffusion, reference_power, applied_power = make_power_normalized_diffusion(
        config,
        grid,
        diffusion_scale=diffusion_scale,
        absorbed_power_W=absorbed_power_W,
    )
    if kinetic.orbit_model == "trapped_passing_bounce_average":
        from .physics.orbit_average import apply_bounce_weight_to_diffusion, classify_pitch

        v_parallel, v_perp = grid.mesh()
        classification = classify_pitch(
            v_parallel,
            v_perp,
            B_min_T=kinetic.orbit_B_min_T,
            B_max_T=kinetic.orbit_B_max_T,
            weights=grid.phase_volume_weights_m3_s3 * initial_state.f_e,
        )
        diffusion = apply_bounce_weight_to_diffusion(diffusion, classification).validated(grid)
    collision_temperature_eV = (
        kinetic.collision_temperature_eV
        if kinetic.collision_temperature_eV is not None
        else config.initial.Te_eV
    )
    diagnostics: list[FokkerPlanckStepDiagnostics] = []
    for _ in range(kinetic.steps):
        state, step = solver.advance(
            state,
            kinetic.time_step_s,
            E_parallel_V_m=kinetic.electric_field_V_m,
            collision_frequency_s=kinetic.collision_frequency_s,
            collision_temperature_eV=collision_temperature_eV,
            rf_diffusion=diffusion,
            loss_frequency_s=kinetic.loss_frequency_s,
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
        )
        diagnostics.append(step)
    diffusion_diagnostics: tuple[dict[str, Any], ...] = ()
    if kinetic.rf_diffusion_model == "genray_like_internal":
        from .physics.genray_like import build_internal_genray_like_reference

        diffusion_diagnostics = tuple(
            item.to_dict()
            for item in build_internal_genray_like_reference(config, grid).diagnostics
        )
    from .dql_provenance import assess_dql_source

    return KineticRunResult(
        state=state,
        initial_state=initial_state,
        initial_moments=initial,
        final_moments=solver.moments(state),
        step_diagnostics=tuple(diagnostics),
        diffusion_source=diffusion.source,
        cross_section_area_m2=area_m2,
        diffusion_reference_power_W=reference_power,
        applied_absorbed_power_W=applied_power,
        diffusion_diagnostics=diffusion_diagnostics,
        dql_assessment=assess_dql_source(config).to_dict(),
    )
