"""Kinetic interfaces and compatibility imports.

The reduced solver is re-exported here so existing imports keep working. The
explicit sentinel remains available when a caller requires the full CQL3D
physics scope rather than the reduced backend.
"""

from __future__ import annotations

import numpy as np

from .fokker_planck import (
    KineticState2V,
)


class NotImplementedFokkerPlanck:
    """Full-scope sentinel; it does not silently return reduced physics."""

    def rhs(self, state: KineticState2V, E_parallel_V_m: float, B_T: float, dt_s: float) -> np.ndarray:
        raise NotImplementedError(
            "Full production use requires relativistic C_en+C_ei+C_ee+Q_RF, bounce averaging, "
            "and validated loss boundaries. Use ReducedFokkerPlanck2V only within its documented scope."
        )
