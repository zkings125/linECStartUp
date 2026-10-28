"""Application workflow for the open-field-line 1D-2V kinetic backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .physics.fokker_planck import RelativisticMomentumGrid, VelocityGrid
from .physics.fullwave_dql import load_open_fieldline_npz, load_spatial_fullwave_dql_npz
from .physics.rates import CollisionRateModel
from .physics.spatial_fokker_planck import (
    OpenFieldLineKinetic1D2V,
    SpatialKineticDiagnostics,
    SpatialKineticState,
)
from .physics.wall_sheath import QuasineutralAmbipolarSheath1D


@dataclass(frozen=True)
class SpatialKineticRunResult:
    state: SpatialKineticState
    initial_state: SpatialKineticState
    diagnostics: tuple[SpatialKineticDiagnostics, ...]
    spatial_grid_source: str
    dql_source: str
    coordinate_system: str

    def summary(self) -> dict[str, Any]:
        last = self.diagnostics[-1] if self.diagnostics else None
        return {
            "backend": "open_field_line_1d2v",
            "time_s": self.state.time_s,
            "steps": len(self.diagnostics),
            "coordinate_system": self.coordinate_system,
            "shape": list(self.state.f_e.shape),
            "spatial_grid_source": self.spatial_grid_source,
            "dql_source": self.dql_source,
            "final": None if last is None else last.to_dict(),
            "maximum_rf_power_balance_relative": max(
                (
                    abs(item.rf_power_balance_relative)
                    for item in self.diagnostics
                    if item.rf_power_balance_relative is not None
                ),
                default=None,
            ),
            "maximum_rf_particle_balance_relative": max(
                (
                    abs(item.rf_particle_balance_relative)
                    for item in self.diagnostics
                    if item.rf_particle_balance_relative is not None
                ),
                default=None,
            ),
            "coupled_terms": [
                "E_parallel",
                "Coulomb/e-neutral collisions",
                "ionization birth",
                "spatial Dql absorption",
                "open-end wall loss",
                "current moment",
                "ambipolar sheath" if last and last.sheath_drop_left_V is not None else "no sheath",
            ],
            "equilibrium_model": "none; laboratory/open-field-line coordinates",
            "quantitative_boundary": (
                "Quantitative use requires a shot-resolved traced field line, real wall, "
                "versioned molecular rates and a converged power/momentum-normalized Dql."
            ),
        }

    def save_distribution(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            f_e=self.state.f_e,
            f_initial=self.initial_state.f_e,
            neutral_density_m3=self.state.neutral_density_m3,
            ion_density_m3=self.state.ion_density_m3,
            time_s=np.asarray(self.state.time_s),
        )


def run_spatial_kinetic(config: SimulationConfig) -> SpatialKineticRunResult:
    """Run the configured 1D-2V backend without inventing a field line or Dql."""

    kinetic = config.kinetic
    if kinetic.backend != "open_field_line_1d2v":
        raise ValueError("run_spatial_kinetic requires backend='open_field_line_1d2v'")
    assert kinetic.spatial_fieldline_file is not None
    spatial = load_open_fieldline_npz(kinetic.spatial_fieldline_file)
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
    solver = OpenFieldLineKinetic1D2V(
        spatial,
        velocity,
        streaming_scheme="implicit_upwind",
    )
    state = solver.maxwellian_state(
        config.initial.ne_m3,
        config.initial.Te_eV,
        neutral_density_m3=(
            config.initial.nn_m3
            if config.initial.nn_m3 is not None
            else config.gas.pressure_Pa / (1.380649e-23 * config.gas.Tn_K)
        ),
        ion_density_m3=config.initial.ni_m3,
        drift_v_parallel_m_s=kinetic.initial_drift_v_parallel_m_s,
        time_s=config.numerics.t_start_s,
    )
    initial = SpatialKineticState(
        state.f_e.copy(),
        state.neutral_density_m3.copy(),
        state.ion_density_m3.copy(),
        state.time_s,
    )
    diffusion = None
    if kinetic.spatial_dql_file is not None:
        diffusion = load_spatial_fullwave_dql_npz(
            kinetic.spatial_dql_file,
            spatial,
            velocity,
        )
    rates = CollisionRateModel(config.gas).evaluate(
        config.initial.Te_eV,
        config.initial.ne_m3,
        float(np.mean(state.neutral_density_m3)),
        0.0,
    )
    sheath = (
        QuasineutralAmbipolarSheath1D(
            ion_mass_kg=config.gas.ion_mass_kg,
            secondary_electron_yield=config.gas.secondary_emission,
        )
        if kinetic.spatial_sheath_model == "quasineutral_ambipolar"
        else None
    )
    diagnostics: list[SpatialKineticDiagnostics] = []
    for _ in range(kinetic.steps):
        state, step = solver.advance(
            state,
            kinetic.time_step_s,
            E_parallel_V_m=kinetic.electric_field_V_m,
            collision_frequency_s=kinetic.collision_frequency_s,
            collision_temperature_eV=(
                kinetic.collision_temperature_eV or config.initial.Te_eV
            ),
            loss_frequency_s=kinetic.loss_frequency_s,
            ionization_rate_coefficient_m3_s=(
                rates.K_ion_m3_s if kinetic.spatial_ionization_from_global_rates else 0.0
            ),
            rf_diffusion=diffusion,
            target_absorbed_power_W=(
                None if diffusion is None else diffusion.target_absorbed_power_W
            ),
            momentum_conserving_field_particle=kinetic.momentum_conserving_field_particle,
            sheath_closure=sheath,
            ion_temperature_eV=config.initial.Ti_eV,
        )
        diagnostics.append(step)
    return SpatialKineticRunResult(
        state,
        initial,
        tuple(diagnostics),
        str(Path(kinetic.spatial_fieldline_file).resolve()),
        "zero" if diffusion is None else diffusion.source,
        kinetic.coordinate_system,
    )


__all__ = ["SpatialKineticRunResult", "run_spatial_kinetic"]
