"""Cached Fokker--Planck moment response for the global startup circuit.

The sparse 0D-2V system is solved before time integration at configured D_ql
power fractions.  ODE evaluations only interpolate that table.  This makes
the coupling deterministic and avoids placing a stateful kinetic advance
inside the implicit BDF residual evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from ..config import SimulationConfig
from ..kinetic_workflow import run_kinetic
from ..state import GlobalState
from .current_drive import CurrentComponents, CurrentDriveModel
from .magnetic import MagneticConfigurationState
from .open_field_loss import LossTimes
from .rf import RFSourceTerms


@dataclass(frozen=True)
class KineticCurrentResponseTable:
    power_fractions: np.ndarray
    incremental_current_A: np.ndarray
    reference_absorbed_power_W: float
    reference_density_m3: float
    diffusion_model: str

    def current_at(self, absorbed_power_W: float) -> float:
        fraction = max(0.0, float(absorbed_power_W)) / self.reference_absorbed_power_W
        x = self.power_fractions
        y = self.incremental_current_A
        if fraction <= x[-1]:
            return float(np.interp(fraction, x, y))
        slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1.0e-30)
        return float(y[-1] + slope * (fraction - x[-1]))


_TABLE_CACHE: dict[tuple, KineticCurrentResponseTable] = {}
_CACHE_LOCK = Lock()


def _cache_key(config: SimulationConfig) -> tuple:
    kinetic = config.kinetic
    return (
        kinetic.model_dump_json(),
        float(config.initial.ne_m3),
        float(config.initial.Te_eV),
        float(config.geometry.R0_m),
        float(config.geometry.minor_radius_m),
        float(config.geometry.resolved_volume_m3),
    )


def build_kinetic_current_response(config: SimulationConfig) -> KineticCurrentResponseTable:
    """Return a cached zero-baseline-subtracted F-P current response table."""

    key = _cache_key(config)
    with _CACHE_LOCK:
        cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    fractions = np.asarray(config.kinetic.feedback_power_fractions, dtype=float)
    currents = []
    for fraction in fractions:
        result = run_kinetic(
            config,
            absorbed_power_W=(
                float(fraction) * float(config.kinetic.feedback_reference_absorbed_power_W)
            ),
        )
        currents.append(result.final_moments.current_parallel_A)
    values = np.asarray(currents, dtype=float)
    increments = values - values[0]
    table = KineticCurrentResponseTable(
        fractions,
        increments,
        float(config.kinetic.feedback_reference_absorbed_power_W),
        float(config.kinetic.feedback_reference_density_m3 or config.initial.ne_m3),
        config.kinetic.rf_diffusion_model,
    )
    with _CACHE_LOCK:
        return _TABLE_CACHE.setdefault(key, table)


class KineticMomentCurrentDrive:
    """Replace selected empirical current channels with an F-P moment table."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.baseline = CurrentDriveModel(config.geometry, config.rf)
        self.table = build_kinetic_current_response(config)

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
        density_ratio = max(0.0, state.ne_m3) / max(self.table.reference_density_m3, 1.0)
        density_ratio = min(density_ratio, self.config.kinetic.feedback_density_ratio_cap)
        kinetic_current = self.table.current_at(rf_terms.power_absorbed_W) * density_ratio
        direct = kinetic_current if self.config.kinetic.feedback_replace_direct_rf else empirical.I_rf_A
        tail = 0.0 if self.config.kinetic.feedback_replace_hot_tail else empirical.I_orbit_A
        return CurrentComponents(
            direct,
            tail,
            empirical.I_vfc_A,
            I_kinetic_A=kinetic_current,
            current_drive_model="fp_moment_response_table",
        )
