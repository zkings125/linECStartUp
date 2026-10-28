"""Explicit literature-anchor surrogate for EBW momentum and VFC ramps.

This module is deliberately separate from the default startup physics.  It is
used to test whether a prescribed EBW momentum source and a prescribed
vertical-field ramp are sufficient to reproduce published MAST current
anchors.  The fitted coefficients are not transferable device predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ..config import GeometryConfig, RFConfig
from ..state import GlobalState
from .current_drive import CurrentComponents, CurrentDriveModel
from .magnetic import MagneticConfigurationState
from .open_field_loss import LossTimes
from .rf import RFSourceTerms


@dataclass(frozen=True)
class CalibratedEBWSettings:
    """Auditable coefficients for the temporary MAST anchor closure."""

    current_per_absorbed_power_A_per_W: float
    ramp_gain_max: float = 1.0
    ramp_shape_exponent: float = 1.0
    ramp_start_field_T: float = 0.0
    ramp_end_field_T: float = 0.0
    vertical_field_waveform: tuple[tuple[float, float], ...] = ()
    ramp_enabled: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.current_per_absorbed_power_A_per_W):
            raise ValueError("EBW current efficiency must be finite")
        if self.current_per_absorbed_power_A_per_W < 0.0:
            raise ValueError("EBW current efficiency must be non-negative")
        if not math.isfinite(self.ramp_gain_max) or self.ramp_gain_max < 1.0:
            raise ValueError("ramp_gain_max must be finite and >= 1")
        if not math.isfinite(self.ramp_shape_exponent) or self.ramp_shape_exponent <= 0.0:
            raise ValueError("ramp_shape_exponent must be finite and positive")
        if self.vertical_field_waveform:
            times = [point[0] for point in self.vertical_field_waveform]
            if any(not math.isfinite(float(value)) for point in self.vertical_field_waveform for value in point):
                raise ValueError("vertical-field waveform must be finite")
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("vertical-field waveform times must be strictly increasing")

    def ramp_progress(self, vertical_field_T: float) -> float:
        """Return normalized cumulative ramp progress from the field magnitude."""

        if not self.ramp_enabled:
            return 0.0
        start = abs(self.ramp_start_field_T)
        end = abs(self.ramp_end_field_T)
        span = end - start
        if abs(span) <= 1.0e-15:
            return 0.0
        progress = (abs(float(vertical_field_T)) - start) / span
        return max(0.0, min(1.0, progress))

    def ramp_gain(self, vertical_field_T: float) -> float:
        progress = self.ramp_progress(vertical_field_T)
        return 1.0 + (self.ramp_gain_max - 1.0) * progress**self.ramp_shape_exponent

    def ramp_rate_T_per_s(self, t_s: float) -> float:
        points: Sequence[tuple[float, float]] = self.vertical_field_waveform
        if not self.ramp_enabled or len(points) < 2:
            return 0.0
        t = float(t_s)
        if t < points[0][0] or t > points[-1][0]:
            return 0.0
        for (left_t, left_b), (right_t, right_b) in zip(points, points[1:]):
            if left_t <= t <= right_t:
                return (right_b - left_b) / (right_t - left_t)
        return 0.0


@dataclass(frozen=True)
class EBWMomentumDiagnostics:
    absorbed_power_W: float
    base_current_A: float
    enhanced_current_A: float
    ramp_gain: float
    ramp_progress: float
    vertical_field_T: float
    vertical_field_rate_T_per_s: float


class CalibratedEBWMomentumDrive:
    """Add a fitted absorbed-power-normalized EBW momentum current.

    The ordinary hot-tail and vertical-charge terms remain provided by
    :class:`CurrentDriveModel`.  Only the EBW momentum term is calibrated.
    """

    model_name = "literature_anchor_ebw_momentum_vfc_ramp_surrogate"

    def __init__(
        self,
        geometry: GeometryConfig,
        rf: RFConfig,
        settings: CalibratedEBWSettings,
    ) -> None:
        self.settings = settings
        self.base = CurrentDriveModel(geometry, rf)
        self._last = EBWMomentumDiagnostics(0.0, 0.0, 0.0, 1.0, 0.0, geometry.B_vertical_T, 0.0)

    @property
    def last_diagnostics(self) -> EBWMomentumDiagnostics:
        return self._last

    def evaluate(
        self,
        state: GlobalState,
        rf_terms: RFSourceTerms,
        magnetic_state: MagneticConfigurationState | None = None,
        loss_times: LossTimes | None = None,
        vertical_field_T: float | None = None,
        t_s: float | None = None,
    ) -> CurrentComponents:
        base = self.base.evaluate(
            state,
            rf_terms,
            magnetic_state=magnetic_state,
            loss_times=loss_times,
            vertical_field_T=vertical_field_T,
            t_s=t_s,
        )
        field = float(vertical_field_T if vertical_field_T is not None else 0.0)
        time = float(t_s if t_s is not None else 0.0)
        eta = self.settings.current_per_absorbed_power_A_per_W
        base_ebw = self.base.rf.current_drive_sign * eta * rf_terms.power_absorbed_W
        gain = self.settings.ramp_gain(field)
        enhanced = base_ebw * gain
        self._last = EBWMomentumDiagnostics(
            absorbed_power_W=rf_terms.power_absorbed_W,
            base_current_A=base_ebw,
            enhanced_current_A=enhanced,
            ramp_gain=gain,
            ramp_progress=self.settings.ramp_progress(field),
            vertical_field_T=field,
            vertical_field_rate_T_per_s=self.settings.ramp_rate_T_per_s(time),
        )
        return CurrentComponents(
            I_rf_A=base.I_rf_A + enhanced,
            I_orbit_A=base.I_orbit_A,
            I_vfc_A=base.I_vfc_A,
            I_kinetic_A=enhanced,
            current_drive_model=self.model_name,
        )
